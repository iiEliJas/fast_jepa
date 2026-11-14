"""Minimal LeWM trainer.

The reference training recipe (``~/projects/le-wm/train.py`` + configs):

- AdamW, lr ``5e-5``, weight decay ``1e-3``, gradient clip 1.0;
- ``LinearWarmupCosineAnnealingLR`` stepped **once per optimizer step** (the
  config label says ``epoch`` but ``stable_pretraining`` steps it per step):
  1% linear warmup from 0, then cosine to ``eta_min=0``;
- mixed precision bf16 (RTX 4060) / fp16 (T4) / fp32;
- loss ``pred_loss + 0.09 * sigreg_loss`` with no stop-gradient or EMA.

Everything is logged to JSONL with the git commit, config, device properties
and CUDA/driver versions, per the protocol.
"""

import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import EpisodeDataset, make_image_transform, random_split
from .model import LewmConfig, build_lewm, build_sigreg


@dataclass
class TrainConfig:
    data_path: str = ""
    out_dir: str = "runs/lewm"
    num_steps: int = 4
    frameskip: int = 5
    action_dim: int = 2
    img_size: int = 224
    batch_size: int = 128
    grad_accum_steps: int = 1
    epochs: int = 10
    lr: float = 5e-5
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    sigreg_weight: float = 0.09
    precision: str = "bf16"  # bf16 | fp16 | fp32
    num_workers: int = 0
    val_fraction: float = 0.1
    seed: int = 0
    log_every: int = 50
    eval_every_epochs: int = 1
    save_every_epochs: int = 1
    limit_steps: int = 0
    cache_data: bool = False
    model: LewmConfig = field(default_factory=LewmConfig)


def train_config_from_dict(values: dict) -> TrainConfig:
    """Build a config from a JSON dict, with an optional nested ``model``."""
    values = dict(values)
    model = values.pop("model", None)
    cfg = TrainConfig(**values)
    if model is not None:
        cfg.model = LewmConfig(**model)
    return cfg


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine_lambda(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup then cosine to zero, per optimizer step."""
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def effective_rank(embeddings: torch.Tensor) -> float:
    """Entropy effective rank of the embedding covariance spectrum."""
    x = embeddings.detach().float().reshape(-1, embeddings.size(-1))
    x = x - x.mean(dim=0, keepdim=True)
    if x.size(0) < 2:
        return 0.0
    singular = torch.linalg.svdvals(x)
    total = singular.sum()
    if total <= 0:
        return 0.0
    p = singular / total
    entropy = -(p * torch.log(p + 1e-12)).sum()
    return float(torch.exp(entropy))


def collate(batch):
    return {
        key: torch.stack([item[key] for item in batch])
        for key in batch[0].keys()
    }


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(__file__),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _device_profile(device):
    profile = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
    }
    if torch.cuda.is_available():
        profile.update(
            {
                "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(device),
                "gpu_capability": torch.cuda.get_device_capability(device),
                "gpu_memory_gb": round(
                    torch.cuda.get_device_properties(device).total_memory / 1e9, 2
                ),
            }
        )
    return profile


def _autocast_context(device, precision):
    if precision == "fp32" or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def _embedding_stats(embeddings):
    flat = embeddings.detach().float().reshape(-1, embeddings.size(-1))
    return {
        "emb_std_mean": float(flat.std(dim=0).mean()),
        "emb_abs_mean": float(flat.abs().mean()),
        "effective_rank": effective_rank(embeddings),
    }


def build_loaders(cfg: TrainConfig):
    transform = make_image_transform(cfg.img_size, resize=cfg.img_size != 224)
    dataset = EpisodeDataset(
        cfg.data_path,
        keys=("pixels", "action"),
        num_steps=cfg.num_steps,
        frameskip=cfg.frameskip,
        transform=transform,
        cache=cfg.cache_data,
    )
    cfg.action_dim = dataset.get_dim("action")
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(
        dataset, [1 - cfg.val_fraction, cfg.val_fraction], generator=generator
    )
    loader_kwargs = dict(
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=cfg.num_workers > 0,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return dataset, train_loader, val_loader


@torch.no_grad()
def evaluate(model, sigreg, loader, cfg, device, max_batches=20):
    model.eval()
    totals = {"pred_loss": 0.0, "sigreg_loss": 0.0, "loss": 0.0}
    count = 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = _move_batch(batch, device)
        with _autocast_context(device, cfg.precision):
            out = model.lewm_loss(batch, sigreg, cfg.sigreg_weight)
        for key in totals:
            totals[key] += float(out[key])
        count += 1
    model.train()
    return {f"val_{key}": value / max(1, count) for key, value in totals.items()}


def save_checkpoint(path, model, optimizer, cfg, step, epoch, sigreg):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sigreg": sigreg.state_dict(),
            "config": asdict(cfg),
            "step": step,
            "epoch": epoch,
        },
        path,
    )


def train(cfg: TrainConfig):
    seed_everything(cfg.seed)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, train_loader, val_loader = build_loaders(cfg)
    action_dim = dataset.get_dim("action")
    cfg.action_dim = action_dim
    model_cfg = replace(
        cfg.model,
        action_dim=action_dim,
        frameskip=cfg.frameskip,
        image_size=cfg.img_size,
        history_size=cfg.num_steps - 1,
        num_preds=1,
    )
    model = build_lewm(model_cfg).to(device)
    sigreg = build_sigreg(model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.grad_accum_steps))
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.limit_steps:
        total_steps = min(total_steps, cfg.limit_steps)
    warmup_steps = max(1, int(0.01 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: warmup_cosine_lambda(step, total_steps, warmup_steps),
    )
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=(cfg.precision == "fp16" and device.type == "cuda"),
    )

    log_path = out_dir / "train.jsonl"
    start = time.time()
    step = 0
    epoch = -1
    with log_path.open("a") as log:
        def optimizer_step(out, step_start):
            nonlocal step
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % cfg.log_every == 0 or step == 1:
                record = {
                    "type": "train",
                    "step": step,
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                    "pred_loss": out["pred_loss"].detach().item(),
                    "sigreg_loss": out["sigreg_loss"].detach().item(),
                    "loss": out["loss"].detach().item(),
                    "step_time_s": time.time() - step_start,
                    "peak_mem_gb": (
                        torch.cuda.max_memory_allocated() / 1e9
                        if torch.cuda.is_available()
                        else 0.0
                    ),
                    **_embedding_stats(out["emb"]),
                }
                log.write(json.dumps(record) + "\n")
                log.flush()

        log.write(
            json.dumps(
                {
                    "type": "run",
                    "git_commit": _git_commit(),
                    "config": asdict(cfg),
                    "model_config": asdict(model_cfg),
                    "device": _device_profile(device),
                }
            )
            + "\n"
        )
        log.flush()

        stop = False
        for epoch in range(cfg.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_start = time.time()
            pending = 0
            for batch_index, batch in enumerate(train_loader):
                if cfg.limit_steps and step >= cfg.limit_steps:
                    stop = True
                    break
                t0 = time.time()
                batch = _move_batch(batch, device)
                with _autocast_context(device, cfg.precision):
                    out = model.lewm_loss(batch, sigreg, cfg.sigreg_weight)
                    loss = out["loss"] / cfg.grad_accum_steps
                scaler.scale(loss).backward()
                pending += 1

                if pending == cfg.grad_accum_steps:
                    optimizer_step(out, t0)
                    pending = 0

            if pending and not stop:
                optimizer_step(out, t0)

            if cfg.eval_every_epochs and (epoch + 1) % cfg.eval_every_epochs == 0:
                metrics = evaluate(model, sigreg, val_loader, cfg, device)
                metrics.update(
                    {
                        "type": "eval",
                        "step": step,
                        "epoch": epoch,
                        "epoch_time_s": time.time() - epoch_start,
                    }
                )
                log.write(json.dumps(metrics) + "\n")
                log.flush()

            if cfg.save_every_epochs and (epoch + 1) % cfg.save_every_epochs == 0:
                save_checkpoint(
                    out_dir / "last.pt", model, optimizer, cfg, step, epoch, sigreg
                )

            if stop:
                break

    save_checkpoint(out_dir / "last.pt", model, optimizer, cfg, step, epoch, sigreg)
    total_time = time.time() - start
    summary = {
        "type": "done",
        "steps": step,
        "epochs": epoch + 1,
        "wall_time_s": total_time,
        "steps_per_s": step / max(total_time, 1e-9),
    }
    with log_path.open("a") as log:
        log.write(json.dumps(summary) + "\n")
    return summary


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Train LeWM on an HDF5 episode file")
    parser.add_argument("--config", help="JSON config (see configs/lewm_pusht.json)")
    parser.add_argument("--data-path")
    parser.add_argument("--out-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--frameskip", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--img-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--sigreg-weight", type=float)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--limit-steps", type=int)
    parser.add_argument(
        "--cache-data", action=argparse.BooleanOptionalAction, default=None
    )
    args = parser.parse_args(argv)

    values = {}
    if args.config:
        with open(args.config) as handle:
            values = json.load(handle)
    overrides = {
        key: value
        for key, value in vars(args).items()
        if key != "config" and value is not None
    }
    values.update(overrides)
    if not values.get("data_path"):
        parser.error("--data-path or a config with data_path is required")
    cfg = train_config_from_dict(values)

    summary = train(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
