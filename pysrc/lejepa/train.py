import argparse
import contextlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone

import torch
from common.seeding import seed_everything, worker_init_fn
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .diagnostics import embedding_diagnostics
from .loss import lejepa_loss
from .model import LeJEPAEncoder, Projector
from .sigreg import SlicedEppsPulley

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULTS = {
    "img_size": 128,
    "patch_size": 8,
    "embed_dim": 384,
    "depth": 12,
    "num_heads": 6,
    "drop_path_rate": 0.1,
    "dynamic_img_size": True,
    "proj_hidden": 2048,
    "proj_dim": 16,
    "num_classes": 10,
    "num_slices": 256,
    "t_max": 3.0,
    "n_knots": 17,
    "V": 4,
    "V_g": None,
    "lamb": 0.02,
    "pred_reduction": "mean_over_k",
    "sigreg_mode": "per_view",
    "batch_size": 256,
    "epochs": 10,
    "lr": 2e-3,
    "wd": 5e-2,
    "probe_lr": 1e-3,
    "probe_wd": 1e-7,
    "eta_min": 1e-3,
    "warmup_epochs": 1.0,
    "warmup_start_factor": 0.01,
    "precision": "bf16",
    "seed": 0,
    "num_workers": 2,
    "log_every": 50,
    "diag_batches": 4,
    "nan_guard": True,
    "save_every": 0,
    "dataset": "imagefolder",
    "data_root": "./data",
    "download": True,
}


def resolve_config(cfg):
    """Merge user config over defaults and fill V_g from V."""
    out = {**DEFAULTS, **cfg}
    if out["V_g"] is None:
        out["V_g"] = out["V"]
    return out


def _solarize(threshold, p):
    """Solarize with probability p.

    torchvision 0.29 renamed ``Solarize`` to ``RandomSolarize`` and folded the
    probability into the transform. Support both spellings.
    """
    random_solarize = getattr(transforms, "RandomSolarize", None)
    if random_solarize is not None:
        return random_solarize(threshold=threshold, p=p)
    return transforms.RandomApply([transforms.Solarize(threshold=threshold)], p=p)


def build_transform(img_size=128, train=True):
    """Augmentation recipe from protocol section 5.5 (small-scale M recipe)."""
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5
                ),
                _solarize(threshold=128, p=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, base, transform, V=4):
        self.base = base
        self.transform = transform
        self.V = V

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        views = [self.transform(img) for _ in range(self.V)]
        return torch.stack(views), label


def to_view_major(views):
    """(B, V, ...) -> (V*B, ...) so that .view(V, B, ...) is view-major.

    The DataLoader stacks views image-major (row r = b*V + v). Flattening
    directly and then calling .view(V, B, K) silently scrambles the view and
    batch axes. Transposing first makes the row order v*B + b.
    """
    if views.dim() < 2:
        raise ValueError(f"expected (B, V, ...), got {tuple(views.shape)}")
    V = views.size(1)
    B = views.size(0)
    return views.transpose(0, 1).reshape(V * B, *views.shape[2:])


def build_model(cfg):
    enc = LeJEPAEncoder(
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        drop_path_rate=cfg["drop_path_rate"],
        dynamic_img_size=cfg["dynamic_img_size"],
    )
    proj = Projector(
        in_dim=512,
        hidden=cfg["proj_hidden"],
        out_dim=cfg["proj_dim"],
    )
    sig = SlicedEppsPulley(
        in_dim=cfg["proj_dim"],
        num_slices=cfg["num_slices"],
        t_max=cfg["t_max"],
        n_knots=cfg["n_knots"],
    )
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, cfg["num_classes"]))
    return enc, proj, sig, probe


def build_datasets(cfg):
    """Return (train_base, val_base, train_transform, eval_transform).

    ``train_base`` is untransformed so MultiViewDataset can apply a fresh random
    augmentation per view. ``val_base`` already applies the deterministic test
    transform (Resize + CenterCrop).
    """
    img_size = cfg["img_size"]
    train_tf = build_transform(img_size, train=True)
    eval_tf = build_transform(img_size, train=False)

    name = cfg["dataset"]
    if name == "cifar10":
        root = cfg["data_root"]
        train_base = datasets.CIFAR10(
            root=root, train=True, download=cfg["download"], transform=None
        )
        val_base = datasets.CIFAR10(
            root=root, train=False, download=cfg["download"], transform=eval_tf
        )
        return train_base, val_base, train_tf, eval_tf

    if name != "imagefolder":
        raise ValueError(f"unknown dataset={name!r}")

    data_dir = cfg.get("data_dir")
    if not data_dir or not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"dataset='imagefolder' needs an existing data_dir, got {data_dir!r}"
        )
    train_base = datasets.ImageFolder(data_dir, transform=None)
    val_dir = cfg.get("val_dir")
    if val_dir is None:
        val_base = None
    else:
        if not os.path.isdir(val_dir):
            raise FileNotFoundError(f"val_dir does not exist: {val_dir!r}")
        val_base = datasets.ImageFolder(val_dir, transform=eval_tf)
    return train_base, val_base, train_tf, eval_tf


def build_optimizer(enc, proj, probe, cfg):
    """One AdamW with two groups so the same schedule acts on the probe."""
    return AdamW(
        [
            {
                "params": list(enc.parameters()) + list(proj.parameters()),
                "lr": cfg["lr"],
                "weight_decay": cfg["wd"],
            },
            {
                "params": list(probe.parameters()),
                "lr": cfg["probe_lr"],
                "weight_decay": cfg["probe_wd"],
            },
        ]
    )


def build_scheduler(opt, steps_per_epoch, cfg):
    """Per-step linear warmup then cosine annealing (protocol section 5.5)."""
    total_steps = max(1, cfg["epochs"] * steps_per_epoch)
    warmup_steps = int(round(cfg["warmup_epochs"] * steps_per_epoch))
    warmup_steps = max(1, min(warmup_steps, total_steps))
    warmup = LinearLR(
        opt,
        start_factor=cfg["warmup_start_factor"],
        total_iters=warmup_steps,
    )
    if warmup_steps >= total_steps:
        return warmup
    cosine = CosineAnnealingLR(
        opt,
        T_max=total_steps - warmup_steps,
        eta_min=cfg["eta_min"],
    )
    return SequentialLR(opt, schedulers=[warmup, cosine], milestones=[warmup_steps])


def autocast_context(device, precision):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast("cuda", dtype=torch.float16)
    raise ValueError(f"unknown precision={precision!r}")


def device_info(device):
    info = {
        "type": device.type,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "name": props.name,
                "capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(props.total_memory / 2**30, 2),
            }
        )
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                timeout=5,
            )
            info["driver"] = out.strip().splitlines()[0]
        except Exception:
            info["driver"] = None
    return info


def git_commit():
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


@torch.no_grad()
def evaluate_probe(enc, probe, loader, device, precision):
    """Probe accuracy on single center-crop views (protocol section 5.5)."""
    enc_train, probe_train = enc.training, probe.training
    enc.eval()
    probe.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    correct = 0
    total = 0
    loss_sum = 0.0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device, precision):
            emb = enc(images)
        logits = probe(emb.float())
        loss_sum += ce(logits, labels).item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.numel()
    enc.train(enc_train)
    probe.train(probe_train)
    if total == 0:
        return {"probe_acc": None, "probe_val_loss": None}
    return {"probe_acc": correct / total, "probe_val_loss": loss_sum / total}


def _rng_state():
    state = {
        "torch": torch.get_rng_state(),
        "numpy": None,
        "python": None,
    }
    import random

    import numpy as np

    state["numpy"] = np.random.get_state()
    state["python"] = random.getstate()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint(path, enc, proj, probe, opt, sched, epoch, step, cfg, metrics):
    torch.save(
        {
            "enc": enc.state_dict(),
            "proj": proj.state_dict(),
            "probe": probe.state_dict(),
            "opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": cfg,
            "git_commit": git_commit(),
            "metrics": metrics,
            "rng": _rng_state(),
        },
        path,
    )


def train(cfg, out_dir="runs"):
    cfg = resolve_config(cfg)
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V = cfg["V"]

    train_base, val_base, train_tf, _ = build_datasets(cfg)
    train_ds = MultiViewDataset(train_base, train_tf, V=V)
    generator = torch.Generator()
    generator.manual_seed(cfg["seed"])
    loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
        worker_init_fn=worker_init_fn if cfg["num_workers"] > 0 else None,
        generator=generator,
    )
    if len(loader) == 0:
        raise ValueError("empty training loader; check batch_size and dataset size")
    val_loader = (
        DataLoader(
            val_base,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"],
            pin_memory=device.type == "cuda",
        )
        if val_base is not None
        else None
    )

    enc, proj, sig, probe = build_model(cfg)
    enc.to(device)
    proj.to(device)
    sig.to(device)
    probe.to(device)

    opt = build_optimizer(enc, proj, probe, cfg)
    sched = build_scheduler(opt, len(loader), cfg)
    scaler = (
        torch.amp.GradScaler("cuda")
        if device.type == "cuda" and cfg["precision"] == "fp16"
        else None
    )
    ce = nn.CrossEntropyLoss()

    meta = {
        "git_commit": git_commit(),
        "config": cfg,
        "device": device_info(device),
        "seed": cfg["seed"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(out_dir, "run.json"), "w") as f:
        json.dump(meta, f, indent=2)

    log_path = os.path.join(out_dir, "log.jsonl")
    with open(log_path, "w"):
        pass  # truncate previous runs in this directory

    global_step = 0
    best_acc = None
    lamb = cfg["lamb"]
    K = cfg["proj_dim"]

    for epoch in range(cfg["epochs"]):
        enc.train()
        proj.train()
        probe.train()
        diag_chunks = []
        epoch_losses = []
        epoch_lp = []
        epoch_ls = []
        epoch_times = []

        for views, labels in loader:
            step_idx = global_step
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            views = views.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            B = views.size(0)
            v = to_view_major(views)  # (V*B, C, H, W)
            opt.zero_grad(set_to_none=True)
            with autocast_context(device, cfg["precision"]):
                emb = enc(v)
                z = proj(emb).view(V, B, -1)
            zf = z.float()
            loss, lp, ls = lejepa_loss(
                zf,
                sig,
                lamb,
                pred_reduction=cfg["pred_reduction"],
                V_g=cfg["V_g"],
                sigreg_mode=cfg["sigreg_mode"],
            )
            probe_logits = probe(emb.detach().float())
            probe_labels = labels.repeat(V)
            probe_loss = ce(probe_logits, probe_labels)

            if cfg["nan_guard"]:
                for name, value in (
                    ("loss", loss),
                    ("lp", lp),
                    ("ls", ls),
                    ("probe_loss", probe_loss),
                ):
                    if not torch.isfinite(value):
                        raise RuntimeError(f"non-finite {name} at step {step_idx}: {value.item()}")

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.scale(probe_loss).backward()
                scaler.unscale_(opt)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                probe_loss.backward()
                opt.step()

            # Record the LR actually used for this step, then advance it.
            step_lr = opt.param_groups[0]["lr"]
            step_probe_lr = opt.param_groups[1]["lr"]
            sched.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            global_step += 1
            epoch_losses.append(loss.item())
            epoch_lp.append(lp.item())
            epoch_ls.append(ls.item())
            epoch_times.append(dt)
            if len(diag_chunks) < cfg["diag_batches"]:
                diag_chunks.append(zf.detach().reshape(-1, K).cpu())

            if step_idx % cfg["log_every"] == 0:
                record = {
                    "kind": "step",
                    "step": step_idx,
                    "epoch": epoch,
                    "loss": loss.item(),
                    "lp": lp.item(),
                    "ls": ls.item(),
                    "probe_loss": probe_loss.item(),
                    "lr": step_lr,
                    "probe_lr": step_probe_lr,
                    "step_time_ms": dt * 1000.0,
                    "imgs_per_s": (B * V) / dt if dt > 0 else None,
                }
                with open(log_path, "a") as f:
                    f.write(json.dumps(record) + "\n")

        diag = embedding_diagnostics(torch.cat(diag_chunks))
        probe_metrics = (
            evaluate_probe(enc, probe, val_loader, device, cfg["precision"])
            if val_loader is not None
            else {"probe_acc": None, "probe_val_loss": None}
        )
        summary = {
            "kind": "epoch",
            "epoch": epoch,
            "loss": sum(epoch_losses) / len(epoch_losses),
            "lp": sum(epoch_lp) / len(epoch_lp),
            "ls": sum(epoch_ls) / len(epoch_ls),
            "step_time_ms_mean": 1000.0 * sum(epoch_times) / len(epoch_times),
            "imgs_per_s": (cfg["batch_size"] * V) / (sum(epoch_times) / len(epoch_times)),
            **diag,
            **probe_metrics,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(summary) + "\n")
        acc = probe_metrics["probe_acc"]
        if acc is not None and (best_acc is None or acc > best_acc):
            best_acc = acc
        print(
            f"epoch {epoch} loss {summary['loss']:.4f} "
            f"lp {summary['lp']:.4f} ls {summary['ls']:.4f} "
            f"erank {summary['erank']:.2f} acc {acc}"
        )

        if cfg["save_every"] and (epoch + 1) % cfg["save_every"] == 0:
            save_checkpoint(
                os.path.join(out_dir, "ckpt_last.pt"),
                enc,
                proj,
                probe,
                opt,
                sched,
                epoch,
                global_step,
                cfg,
                {"best_probe_acc": best_acc, **probe_metrics},
            )

    save_checkpoint(
        os.path.join(out_dir, "ckpt.pt"),
        enc,
        proj,
        probe,
        opt,
        sched,
        cfg["epochs"] - 1,
        global_step,
        cfg,
        {"best_probe_acc": best_acc},
    )
    return {"enc": enc, "proj": proj, "probe": probe}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train LeJEPA from a JSON config.")
    parser.add_argument("--config", required=True, help="path to a JSON config file")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    with open(args.config) as f:
        cfg = json.load(f)
    out_dir = args.out_dir or cfg.get(
        "out_dir", os.path.join("runs", os.path.splitext(os.path.basename(args.config))[0])
    )
    train(cfg, out_dir=out_dir)


if __name__ == "__main__":
    main()
