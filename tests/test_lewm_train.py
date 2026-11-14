"""Trainer smoke test and schedule/statistics unit checks (CPU, synthetic data)."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

from lewm import LewmConfig  # noqa: E402
from lewm.train import (  # noqa: E402
    TrainConfig,
    effective_rank,
    train,
    train_config_from_dict,
    warmup_cosine_lambda,
)

h5py = pytest.importorskip("h5py")


def test_warmup_cosine_schedule():
    total, warmup = 100, 10
    assert warmup_cosine_lambda(0, total, warmup) == pytest.approx(0.1)
    assert warmup_cosine_lambda(warmup - 1, total, warmup) == pytest.approx(1.0)
    assert warmup_cosine_lambda(warmup, total, warmup) == pytest.approx(1.0)
    assert warmup_cosine_lambda(total - 1, total, warmup) < 1e-3
    assert warmup_cosine_lambda(total, total, warmup) == pytest.approx(0.0)


def test_pusht_config_file():
    path = Path(__file__).parents[1] / "configs" / "lewm_pusht.json"
    cfg = train_config_from_dict(json.loads(path.read_text()))
    assert cfg.epochs == 10
    assert cfg.batch_size == 128
    assert cfg.frameskip == 5
    assert cfg.model.embed_dim == 192
    assert cfg.model.predictor_dropout == 0.1
    assert cfg.model.num_steps == 4


def test_effective_rank_detects_collapse():
    g = torch.Generator().manual_seed(0)
    isotropic = torch.randn(512, 32, generator=g)
    collapsed = torch.randn(512, 1, generator=g).expand(-1, 32)
    assert effective_rank(isotropic) > 20
    assert effective_rank(collapsed) < 1.01


def _write_episodes(path, lengths=(16, 16, 16, 16), action_dim=2, shape=(28, 28, 3)):
    total = sum(lengths)
    pixels = np.zeros((total, *shape), dtype=np.uint8)
    actions = np.zeros((total, action_dim), dtype=np.float32)
    episode_idx = np.zeros(total, dtype=np.int64)
    step_idx = np.zeros(total, dtype=np.int64)
    offset = 0
    for ep, length in enumerate(lengths):
        for step in range(length):
            row = offset + step
            pixels[row] = (row * 7) % 255
            actions[row] = [step / max(1, length - 1), -step / max(1, length - 1)]
            episode_idx[row] = ep
            step_idx[row] = step
        offset += length
    with h5py.File(path, "w") as handle:
        handle.create_dataset("pixels", data=pixels)
        handle.create_dataset("action", data=actions)
        handle.create_dataset("episode_idx", data=episode_idx)
        handle.create_dataset("step_idx", data=step_idx)


def test_train_smoke(tmp_path):
    data_path = tmp_path / "episodes.h5"
    _write_episodes(data_path)

    model_cfg = LewmConfig(
        image_size=28,
        patch_size=14,
        embed_dim=32,
        encoder_depth=1,
        encoder_heads=2,
        encoder_head_dim=16,
        encoder_mlp_dim=64,
        predictor_depth=1,
        predictor_heads=2,
        predictor_head_dim=16,
        predictor_mlp_dim=64,
        predictor_dropout=0.0,
        projector_hidden_dim=64,
        action_smoothed_dim=None,
        history_size=3,
        num_preds=1,
        action_dim=2,
        frameskip=2,
        sigreg_knots=5,
        sigreg_num_proj=8,
    )
    cfg = TrainConfig(
        data_path=str(data_path),
        out_dir=str(tmp_path / "run"),
        num_steps=4,
        frameskip=2,
        img_size=28,
        batch_size=2,
        epochs=1,
        lr=5e-5,
        precision="fp32",
        num_workers=0,
        val_fraction=0.2,
        seed=0,
        log_every=1,
        limit_steps=2,
        model=model_cfg,
    )
    summary = train(cfg)

    assert summary["steps"] == 2
    log_lines = (tmp_path / "run" / "train.jsonl").read_text().strip().splitlines()
    records = [json.loads(line) for line in log_lines]
    assert records[0]["type"] == "run"
    assert "config" in records[0]
    assert "device" in records[0]
    assert any(r["type"] == "train" for r in records)
    assert any(r["type"] == "eval" for r in records)
    assert records[-1]["type"] == "done"

    checkpoint = torch.load(
        tmp_path / "run" / "last.pt", map_location="cpu", weights_only=False
    )
    assert checkpoint["step"] == 2
    assert any(key.startswith("encoder.") for key in checkpoint["model"])
