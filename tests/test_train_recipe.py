import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

import numpy as np
import pytest
import torch
from lejepa.loss import lejepa_loss
from lejepa.model import LeJEPAEncoder
from lejepa.sigreg import SlicedEppsPulley
from lejepa.train import build_scheduler, build_transform, resolve_config, train
from PIL import Image
from torchvision import transforms


class _ZeroSigreg:
    def __call__(self, x, A=None):
        return x.sum() * 0.0


def _has(transform_list, cls):
    return any(isinstance(t, cls) for t in transform_list)


def test_train_transform_recipe():
    tf = build_transform(32, train=True)
    steps = list(tf.transforms)
    assert _has(steps, transforms.Normalize)
    assert _has(steps, transforms.RandomResizedCrop)
    assert _has(steps, transforms.RandomHorizontalFlip)

    random_apply = [t for t in steps if isinstance(t, transforms.RandomApply)]
    assert any(
        any(isinstance(inner, transforms.ColorJitter) for inner in t.transforms)
        for t in random_apply
    )
    assert any(
        any(isinstance(inner, transforms.GaussianBlur) for inner in t.transforms)
        for t in random_apply
    )
    random_solarize = getattr(transforms, "RandomSolarize", None)
    if random_solarize is not None:
        solarize = [t for t in steps if isinstance(t, random_solarize)]
    else:
        solarize = [
            t
            for t in random_apply
            if any(type(inner).__name__ == "Solarize" for inner in t.transforms)
        ]
    assert len(solarize) == 1
    assert solarize[0].p == 0.2
    blur = next(
        t for t in random_apply if any(isinstance(i, transforms.GaussianBlur) for i in t.transforms)
    )
    assert blur.p == 0.5


def test_eval_transform_is_deterministic_and_normalized():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (40, 40, 3), dtype=np.uint8))
    tf = build_transform(32, train=False)
    a = tf(img)
    b = tf(img)
    assert torch.equal(a, b)
    assert a.shape == (3, 32, 32)
    # ImageNet normalization pushes pixels outside [0, 1].
    assert a.min() < 0.0 or a.max() > 1.0


def test_loss_uses_global_views_for_mu():
    torch.manual_seed(0)
    V, N, K = 4, 5, 3
    z = torch.randn(V, N, K)

    _, lp2, _ = lejepa_loss(z, _ZeroSigreg(), 0.0, V_g=2)
    mu2 = z[:2].mean(dim=0)
    expected2 = ((mu2.unsqueeze(0) - z) ** 2).mean()
    assert torch.allclose(lp2, expected2)

    _, lp1, _ = lejepa_loss(z, _ZeroSigreg(), 0.0, V_g=1)
    assert not torch.allclose(lp1, lp2)

    with pytest.raises(ValueError):
        lejepa_loss(z, _ZeroSigreg(), V_g=0)
    with pytest.raises(ValueError):
        lejepa_loss(z, _ZeroSigreg(), V_g=V + 1)


def test_loss_sigreg_modes():
    calls = []

    class Recorder:
        def __call__(self, x, A=None):
            calls.append(tuple(x.shape))
            return x.sum() * 0.0

    z = torch.randn(4, 6, 8)
    lejepa_loss(z, Recorder(), 0.02, sigreg_mode="per_view")
    assert calls == [(4, 6, 8)]

    calls.clear()
    lejepa_loss(z, Recorder(), 0.02, sigreg_mode="joint")
    assert calls == [(24, 8)]

    with pytest.raises(ValueError):
        lejepa_loss(z, Recorder(), sigreg_mode="nope")


def test_loss_shared_directions_are_one_call():
    calls = []

    class Recorder:
        def __call__(self, x, A=None):
            calls.append(tuple(x.shape))
            return x.sum() * 0.0

    z = torch.randn(3, 7, 16)
    lejepa_loss(z, Recorder(), 0.02)
    assert calls == [(3, 7, 16)]


def test_scheduler_is_per_step_and_moves_probe_group():
    p0 = torch.nn.Parameter(torch.zeros(1))
    p1 = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW(
        [
            {"params": [p0], "lr": 2e-3},
            {"params": [p1], "lr": 1e-3},
        ]
    )
    cfg = resolve_config(
        {"epochs": 2, "warmup_epochs": 1.0, "warmup_start_factor": 0.01, "eta_min": 1e-3}
    )
    sched = build_scheduler(opt, steps_per_epoch=4, cfg=cfg)

    lrs = []
    for _ in range(8):
        lrs.append((opt.param_groups[0]["lr"], opt.param_groups[1]["lr"]))
        opt.step()
        sched.step()

    assert lrs[0][0] < lrs[3][0]  # linear warmup
    assert lrs[4][0] > lrs[-1][0]  # cosine decay after warmup
    assert lrs[0][1] < lrs[3][1]  # probe group follows the same schedule
    assert abs(lrs[-1][0] - cfg["eta_min"]) < 5e-4


def test_model_applies_drop_path_and_dynamic_size():
    enc = LeJEPAEncoder(
        img_size=32,
        patch_size=4,
        embed_dim=32,
        depth=2,
        num_heads=2,
        drop_path_rate=0.1,
        dynamic_img_size=True,
    )
    assert enc.vit.embed_dim == 32
    assert tuple(enc.vit.patch_embed.patch_size) == (4, 4)
    assert any(type(m).__name__ == "DropPath" for m in enc.vit.modules())
    assert enc(torch.randn(2, 3, 32, 32)).shape == (2, 512)


def _make_imagefolder(root, n_per_class=4, size=32, seed=0):
    rng = np.random.default_rng(seed)
    for c in range(2):
        d = root / f"class{c}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_class):
            arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i}.png")


def test_train_smoke_writes_logs_and_full_checkpoint(tmp_path):
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    _make_imagefolder(train_root, seed=0)
    _make_imagefolder(val_root, seed=1)
    out_dir = tmp_path / "run"

    cfg = {
        "dataset": "imagefolder",
        "data_dir": str(train_root),
        "val_dir": str(val_root),
        "img_size": 32,
        "patch_size": 4,
        "embed_dim": 32,
        "depth": 1,
        "num_heads": 2,
        "drop_path_rate": 0.0,
        "proj_hidden": 32,
        "proj_dim": 16,
        "num_classes": 2,
        "num_slices": 8,
        "V": 2,
        "V_g": 2,
        "batch_size": 4,
        "epochs": 1,
        "lr": 1e-3,
        "probe_lr": 1e-3,
        "precision": "fp32",
        "num_workers": 0,
        "log_every": 1,
        "diag_batches": 1,
        "save_every": 1,
        "seed": 0,
    }
    result = train(cfg, out_dir=str(out_dir))
    assert set(result) == {"enc", "proj", "probe"}

    run_meta = json.loads((out_dir / "run.json").read_text())
    assert run_meta["config"]["pred_reduction"] == "mean_over_k"
    assert run_meta["config"]["sigreg_mode"] == "per_view"
    assert run_meta["config"]["t_max"] == 3.0
    assert run_meta["config"]["n_knots"] == 17
    assert run_meta["config"]["V_g"] == 2

    records = [json.loads(line) for line in (out_dir / "log.jsonl").read_text().splitlines()]
    steps = [r for r in records if r["kind"] == "step"]
    epochs = [r for r in records if r["kind"] == "epoch"]
    assert steps and epochs
    assert {"loss", "lp", "ls", "lr", "step_time_ms", "imgs_per_s"} <= set(steps[0])
    epoch = epochs[-1]
    assert {"erank", "std_mean", "collapsed", "probe_acc", "probe_val_loss"} <= set(epoch)
    assert epoch["probe_acc"] is not None
    assert 0.0 <= epoch["probe_acc"] <= 1.0

    ckpt = torch.load(out_dir / "ckpt.pt", weights_only=False)
    assert {"enc", "proj", "probe", "opt", "sched", "epoch", "step", "config", "rng"} <= set(ckpt)


def test_train_rejects_missing_imagefolder(tmp_path):
    cfg = resolve_config(
        {
            "dataset": "imagefolder",
            "data_dir": str(tmp_path / "does_not_exist"),
            "epochs": 1,
            "num_workers": 0,
            "precision": "fp32",
        }
    )
    with pytest.raises(FileNotFoundError):
        train(cfg, out_dir=str(tmp_path / "run"))


def test_sigreg_known_distribution_sanity():
    torch.manual_seed(0)
    sig = SlicedEppsPulley(8, num_slices=64)
    normal = torch.randn(512, 8)
    shifted = torch.randn(512, 8) * 3.0 + 2.0
    assert sig(normal) < sig(shifted)
