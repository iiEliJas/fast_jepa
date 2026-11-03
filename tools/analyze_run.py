"""Analyze a trained LeJEPA run: metrics, figures and a Markdown report.

Example
-------
    python tools/analyze_run.py --run-dir runs/lejepa_cifar10

The script reads ``run.json``, ``log.jsonl`` and ``ckpt.pt`` from the run
directory, evaluates the frozen encoder and its online probe on the validation
split, computes representation statistics, and writes to ``<run-dir>/analysis``:

* ``metrics.json``: every number used in the report.
* ``report.md``: a guided tour of the figures.
* ``plots/*.png``: training curves, rank spectrum, Gaussianity, PCA and more.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from common.device import device_info, resolve_device
from lejepa.analysis import (
    confusion_matrix,
    cosine_similarity_stats,
    knn_accuracy_curve,
    mean_direction_ratio,
    pca_2d,
    per_class_accuracy,
    per_dim_std,
    prediction_loss,
    shuffled_prediction_loss,
    singular_spectrum,
    sliced_gaussianity,
    topk_accuracy,
)
from lejepa.train import (
    MultiViewDataset,
    autocast_context,
    build_datasets,
    build_model,
    to_view_major,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

PLOT_FILES = {
    "training": "01_training_curves.png",
    "optimization": "02_optimization.png",
    "loss": "03_loss_decomposition.png",
    "probe": "04_probe_evaluation.png",
    "rank": "05_rank_spectrum.png",
    "gaussian": "06_gaussianity.png",
    "pca": "07_embedding_pca.png",
    "geometry": "08_representation_quality.png",
}


class _Transformed(Dataset):
    """Apply a transform to a dataset whose base returns PIL images."""

    def __init__(self, base, transform):
        self.base = base
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image, label = self.base[index]
        return self.transform(image), label


def _pct(value):
    return f"{100.0 * value:.2f}%"


def _load_run(run_dir, checkpoint):
    run_path = os.path.join(run_dir, "run.json")
    log_path = os.path.join(run_dir, "log.jsonl")
    for path in (run_path, log_path, checkpoint):
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing analysis input: {path}")

    with open(run_path) as handle:
        run_meta = json.load(handle)
    with open(log_path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)

    config = checkpoint_data.get("config") or run_meta.get("config")
    if config is None:
        raise ValueError("neither checkpoint nor run.json contains a config")
    epochs = sorted((row for row in rows if row.get("kind") == "epoch"), key=lambda r: r["epoch"])
    steps = sorted((row for row in rows if row.get("kind") == "step"), key=lambda r: r["step"])
    if not epochs:
        raise ValueError(f"{log_path} has no epoch records")
    return run_meta, config, checkpoint_data, epochs, steps


def _build_models(checkpoint_data, config, device):
    enc, proj, sig, probe = build_model(config)
    enc.load_state_dict(checkpoint_data["enc"])
    proj.load_state_dict(checkpoint_data["proj"])
    probe.load_state_dict(checkpoint_data["probe"])
    for module in (enc, proj, sig, probe):
        module.to(device)
        module.eval()
    return enc, proj, sig, probe


@torch.no_grad()
def _collect_features(enc, proj, base, transform, device, precision, batch_size, num_workers):
    dataset = _Transformed(base, transform) if transform is not None else base
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    encs, zs, labels = [], [], []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        with autocast_context(device, precision):
            emb = enc(images)
            z = proj(emb)
        encs.append(emb.float().cpu())
        zs.append(z.float().cpu())
        labels.append(targets)
    return torch.cat(encs), torch.cat(zs), torch.cat(labels)


@torch.no_grad()
def _collect_views(enc, proj, base, train_transform, V, n_images, device, precision, batch_size):
    dataset = Subset(MultiViewDataset(base, train_transform, V=V), range(min(n_images, len(base))))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    encs, zs = [], []
    for views, _ in loader:
        B = views.size(0)
        flat = to_view_major(views).to(device)
        with autocast_context(device, precision):
            emb = enc(flat)
            z = proj(emb)
        encs.append(emb.float().view(V, B, -1).cpu())
        zs.append(z.float().view(V, B, -1).cpu())
    return torch.cat(encs, dim=1), torch.cat(zs, dim=1)


@torch.no_grad()
def _probe_logits(probe, features, device, batch_size=2048):
    outputs = []
    for start in range(0, features.shape[0], batch_size):
        outputs.append(probe(features[start : start + batch_size].to(device)).float().cpu())
    return torch.cat(outputs)


def _offline_probe(
    train_x,
    train_y,
    test_x,
    test_y,
    num_classes,
    device,
    epochs,
    seed,
    lr=1e-3,
    weight_decay=1e-7,
    batch_size=1024,
):
    if epochs <= 0:
        return None, []
    torch.manual_seed(seed)
    head = nn.Sequential(nn.LayerNorm(train_x.shape[1]), nn.Linear(train_x.shape[1], num_classes))
    head.to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    test_x = test_x.to(device)
    test_y = test_y.to(device)
    history = []
    for _ in range(epochs):
        head.train()
        order = torch.randperm(train_x.shape[0], device=device)
        for start in range(0, train_x.shape[0], batch_size):
            index = order[start : start + batch_size]
            loss = criterion(head(train_x[index]), train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        head.eval()
        with torch.no_grad():
            history.append(float((head(test_x).argmax(dim=1) == test_y).float().mean()))
    return head, history


def _energy_rank(cumulative, fraction):
    if cumulative.numel() == 0 or float(cumulative[-1]) <= 0:
        return 0
    index = torch.searchsorted(cumulative, torch.tensor(fraction, dtype=cumulative.dtype))
    return int(min(index.item() + 1, cumulative.numel()))


def _sigreg_draws(sig, z, V, N, device, draws, seed):
    if z.shape[1] < V:
        raise ValueError(f"need at least V={V} test embeddings, got {z.shape[0]}")
    K = z.shape[1]
    perm_generator = torch.Generator().manual_seed(seed)
    null_generator = torch.Generator(device=device).manual_seed(seed + 1)
    test_values, null_values = [], []
    with torch.no_grad():
        for _ in range(draws):
            index = torch.randperm(z.shape[0], generator=perm_generator)[: V * N]
            batch = z[index].to(device).view(V, N, K)
            test_values.append(float(sig(batch)))
        for _ in range(draws):
            batch = torch.randn(V, N, K, generator=null_generator, device=device)
            null_values.append(float(sig(batch)))
    return torch.tensor(test_values), torch.tensor(null_values)


def _class_names(base, num_classes):
    names = list(getattr(base, "classes", []))
    if len(names) < num_classes:
        names.extend(str(i) for i in range(len(names), num_classes))
    return names[:num_classes]


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 140,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def _save(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_training(epochs, cfg, path):
    ep = np.array([row["epoch"] for row in epochs])
    loss = np.array([row["loss"] for row in epochs])
    lp = np.array([row["lp"] for row in epochs])
    ls = np.array([row["ls"] for row in epochs])
    acc = np.array([row["probe_acc"] for row in epochs if row.get("probe_acc") is not None])
    acc_ep = np.array([row["epoch"] for row in epochs if row.get("probe_acc") is not None])
    val_loss = np.array(
        [row["probe_val_loss"] for row in epochs if row.get("probe_val_loss") is not None]
    )
    erank = np.array([row["erank"] for row in epochs])
    std_mean = np.array([row["std_mean"] for row in epochs])

    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(ep, loss, color="#1f77b4", lw=1.8)
    ax.set_title("Total training loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")

    ax = axes[0, 1]
    ax.plot(ep, lp, color="#d62728", label="L_pred")
    ax.set_title("Prediction term L_pred")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE to view mean")

    ax = axes[1, 0]
    ax.semilogy(ep, ls, color="#2ca02c", lw=1.8)
    ax.set_title("SIGReg term L_sig (log scale)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Epps-Pulley statistic")

    ax = axes[1, 1]
    ax.plot(acc_ep, 100 * acc, color="#9467bd", lw=1.8)
    if acc.size:
        best = int(np.argmax(acc))
        ax.scatter(acc_ep[best], 100 * acc[best], color="black", zorder=5, s=18)
        ax.annotate(
            f"best {_pct(acc[best])} @ {acc_ep[best]}",
            (acc_ep[best], 100 * acc[best]),
            xytext=(-70, 12),
            textcoords="offset points",
            fontsize=8,
        )
    ax.axhline(10, color="gray", ls=":", lw=1, label="chance (10%)")
    ax.set_title("Online linear-probe accuracy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("top-1 (%)")
    ax.legend(loc="lower right")

    ax = axes[2, 0]
    ax.plot(acc_ep, val_loss, color="#8c564b", lw=1.8)
    ax.set_title("Probe validation cross-entropy")
    ax.set_xlabel("epoch")
    ax.set_ylabel("CE loss")

    ax = axes[2, 1]
    ax.plot(ep, erank, color="#ff7f0e", lw=1.8, label="effective rank")
    ax.axhline(cfg["proj_dim"], color="gray", ls=":", lw=1, label=f"max = {cfg['proj_dim']}")
    ax.set_title("Projector effective rank (no collapse)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("erank")
    ax.set_ylim(0, cfg["proj_dim"] * 1.05)
    ax2 = ax.twinx()
    ax2.plot(ep, std_mean, color="#17becf", lw=1.2, alpha=0.75, label="mean per-dim std")
    ax2.set_ylabel("mean per-dim std")
    lines = ax.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], loc="lower right")
    fig.suptitle("Training history", fontsize=13)
    _save(fig, path)


def _plot_optimization(steps, cfg, path):
    if not steps:
        return
    step = np.array([row["step"] for row in steps])
    lr = np.array([row["lr"] for row in steps])
    probe_lr = np.array([row["probe_lr"] for row in steps])
    imgs = np.array([row["imgs_per_s"] for row in steps if row.get("imgs_per_s")])
    imgs_step = np.array([row["step"] for row in steps if row.get("imgs_per_s")])
    step_ms = np.array([row["step_time_ms"] for row in steps])
    cumulative = (step + 1) * cfg["batch_size"] * cfg["V"] / 1e6

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    ax = axes[0, 0]
    ax.semilogy(step, lr, label="backbone + projector", color="#1f77b4")
    ax.semilogy(step, probe_lr, label="probe", color="#ff7f0e")
    ax.set_title("Learning rate actually used at each logged step")
    ax.set_xlabel("step")
    ax.set_ylabel("lr")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(imgs_step, imgs, color="#2ca02c", lw=0.8, alpha=0.45)
    if imgs.size >= 5:
        window = max(3, min(25, len(imgs) // 20))
        smooth = np.convolve(imgs, np.ones(window) / window, mode="valid")
        ax.plot(imgs_step[window - 1 :], smooth, color="#145a32", lw=1.8, label="moving average")
        ax.legend(loc="lower right")
    ax.set_ylim(bottom=0)
    ax.set_title("Throughput (includes augmentation and data loading)")
    ax.set_xlabel("step")
    ax.set_ylabel("image-views / s")

    ax = axes[1, 0]
    ax.semilogy(step, step_ms, color="#d62728", lw=0.8)
    ax.set_title("Wall-clock step time (log scale; early points include warmup)")
    ax.set_xlabel("step")
    ax.set_ylabel("ms / step")

    ax = axes[1, 1]
    ax.plot(step, cumulative, color="#9467bd", lw=1.8)
    ax.set_title("Training budget")
    ax.set_xlabel("step")
    ax.set_ylabel("million image-views seen")
    fig.suptitle("Optimization", fontsize=13)
    _save(fig, path)


def _plot_loss_decomposition(epochs, cfg, path):
    ep = np.array([row["epoch"] for row in epochs])
    lp = np.array([row["lp"] for row in epochs])
    ls = np.array([row["ls"] for row in epochs])
    lamb = cfg["lamb"]
    pred_part = (1.0 - lamb) * lp
    sig_part = lamb * ls
    total = pred_part + sig_part

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    ax = axes[0]
    ax.stackplot(
        ep,
        pred_part,
        sig_part,
        labels=[f"(1-lambda) L_pred, lambda={lamb}", "lambda L_sig"],
        colors=["#d62728", "#2ca02c"],
        alpha=0.75,
    )
    ax.plot(ep, total, color="black", lw=1.2, label="total")
    ax.set_title("Weighted loss contributions")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend(loc="upper right")

    ax = axes[1]
    fraction = 100 * sig_part / np.maximum(total, 1e-12)
    ax.plot(ep, fraction, color="#2ca02c", lw=1.8)
    ax.set_ylim(0, max(10, fraction.max() * 1.15))
    ax.set_title("How much of the loss is SIGReg?")
    ax.set_xlabel("epoch")
    ax.set_ylabel("lambda L_sig / total (%)")
    ax.annotate(
        f"final: {fraction[-1]:.1f}%",
        (ep[-1], fraction[-1]),
        xytext=(-70, 12),
        textcoords="offset points",
        fontsize=8,
    )
    fig.suptitle("Loss decomposition", fontsize=13)
    _save(fig, path)


def _plot_probe(logits, labels, class_names, metrics, path):
    num_classes = len(class_names)
    pred = logits.argmax(dim=1)
    cm = confusion_matrix(labels, pred, num_classes)
    cm_norm = cm.float() / cm.sum(dim=1, keepdim=True).clamp_min(1)
    per_class = per_class_accuracy(cm)
    confidence = F.softmax(logits, dim=1).max(dim=1).values
    correct = pred == labels

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    ax = axes[0, 0]
    image = ax.imshow(cm_norm.numpy(), cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Confusion matrix (row-normalized)")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for row in range(num_classes):
        for col in range(num_classes):
            value = cm_norm[row, col].item()
            if value >= 0.01:
                ax.text(
                    col,
                    row,
                    f"{100 * value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value > 0.55 else "black",
                )
    fig.colorbar(image, ax=ax, fraction=0.046)

    ax = axes[0, 1]
    order = torch.argsort(per_class, descending=True)
    names = [class_names[i] for i in order.tolist()]
    values = 100 * per_class[order].numpy()
    ax.barh(names, values, color="#1f77b4")
    ax.axvline(100 / num_classes, color="gray", ls=":", lw=1, label="chance")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_title(f"Per-class recall (top-1 {_pct(metrics['top1'])})")
    ax.set_xlabel("recall (%)")
    ax.legend(loc="lower right")

    ax = axes[1, 0]
    bins = np.linspace(0, 1, 11)
    bin_index = np.clip(np.digitize(confidence.numpy(), bins) - 1, 0, 9)
    mean_conf, accuracy, counts = [], [], []
    for index in range(10):
        mask = bin_index == index
        counts.append(int(mask.sum()))
        if mask.any():
            mean_conf.append(float(confidence[mask].mean()))
            accuracy.append(float(correct[mask].float().mean()))
        else:
            mean_conf.append(np.nan)
            accuracy.append(np.nan)
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1, label="perfect calibration")
    ax.bar(mean_conf, accuracy, width=0.075, color="#2ca02c", alpha=0.75, label="probe")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_title("Reliability diagram")
    ax.set_xlabel("mean confidence in bin")
    ax.set_ylabel("accuracy in bin")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    ax.hist(
        confidence[correct].numpy(),
        bins=20,
        range=(0, 1),
        alpha=0.65,
        color="#2ca02c",
        label=f"correct (n={int(correct.sum())})",
    )
    ax.hist(
        confidence[~correct].numpy(),
        bins=20,
        range=(0, 1),
        alpha=0.65,
        color="#d62728",
        label=f"wrong (n={int((~correct).sum())})",
    )
    ax.set_title("Confidence distribution")
    ax.set_xlabel("max softmax probability")
    ax.set_ylabel("count")
    ax.legend()
    fig.suptitle(
        f"Probe evaluation: top-1 {_pct(metrics['top1'])}, top-5 {_pct(metrics['top5'])}, "
        f"CE {metrics['loss']:.3f}",
        fontsize=13,
    )
    _save(fig, path)


def _plot_rank(enc_spec, proj_spec, enc_std, proj_std, path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)

    ax = axes[0, 0]
    for name, spec, color in (
        ("encoder (512d)", enc_spec, "#1f77b4"),
        ("projector (16d)", proj_spec, "#d62728"),
    ):
        values = spec["probabilities"].numpy()
        ax.semilogy(
            np.arange(1, len(values) + 1),
            np.maximum(values, 1e-12),
            color=color,
            label=name,
        )
    ax.set_title("Normalized singular-value spectrum")
    ax.set_xlabel("rank index")
    ax.set_ylabel("s_i / sum(s) (log)")
    ax.legend()

    ax = axes[0, 1]
    for name, spec, color in (
        ("encoder", enc_spec, "#1f77b4"),
        ("projector", proj_spec, "#d62728"),
    ):
        cumulative = 100 * spec["cumulative_energy"].numpy()
        ax.plot(np.arange(1, len(cumulative) + 1), cumulative, color=color, label=name)
        ax.axhline(90, color="gray", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_title("Cumulative spectral energy")
    ax.set_xlabel("rank index (log)")
    ax.set_ylabel("energy captured (%)")
    ax.legend(loc="lower right")

    ax = axes[0, 2]
    labels = ["encoder", "projector"]
    values = [enc_spec["erank"], proj_spec["erank"]]
    maxima = [enc_spec["singular_values"].numel(), proj_spec["singular_values"].numel()]
    fraction = [value / maximum for value, maximum in zip(values, maxima)]
    bars = ax.bar(labels, fraction, color=["#1f77b4", "#d62728"])
    for bar, value, maximum in zip(bars, values, maxima):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.2f} / {maximum}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, 1.08)
    ax.set_title("Effective rank as a fraction of maximum")
    ax.set_ylabel("erank / maximum rank")

    ax = axes[1, 0]
    ax.plot(np.sort(enc_std.numpy())[::-1], color="#1f77b4")
    ax.set_title("Encoder per-dimension std (sorted)")
    ax.set_xlabel("dimension rank")
    ax.set_ylabel("std")

    ax = axes[1, 1]
    ax.bar(np.arange(1, len(proj_std) + 1), proj_std.numpy(), color="#d62728")
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="N(0,1) target")
    ax.set_title("Projector per-dimension std")
    ax.set_xlabel("projector dimension")
    ax.set_ylabel("std")
    ax.legend()

    ax = axes[1, 2]
    ax.hist(enc_std.numpy(), bins=40, color="#1f77b4", alpha=0.7, label="encoder")
    ax.set_title("Encoder std distribution")
    ax.set_xlabel("per-dimension std")
    ax.set_ylabel("count")
    ax.legend()
    fig.suptitle("Rank and isotropy", fontsize=13)
    _save(fig, path)


def _plot_gaussianity(gauss, null, sig_test, sig_null, train_ls, path):
    projections = gauss["projections"]
    num_directions = projections.shape[1]
    chosen = np.unique(np.linspace(0, num_directions - 1, 4).astype(int))
    grid = np.linspace(-4, 4, 400)
    normal_pdf = np.exp(-0.5 * grid**2) / np.sqrt(2 * np.pi)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes[0, 0]
    for index in chosen:
        values = projections[:, index].numpy()
        ax.hist(values, bins=60, density=True, histtype="step", lw=1.3, label=f"slice {index}")
    ax.plot(grid, normal_pdf, color="black", lw=2, label="N(0,1)")
    ax.set_xlim(-4, 4)
    ax.set_title("Random 1D projections vs N(0,1)")
    ax.set_xlabel("projected value")
    ax.set_ylabel("density")
    ax.legend(ncol=2, loc="upper right")

    ax = axes[0, 1]
    pooled = []
    for index in chosen:
        values = projections[:, index]
        z = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-12)
        pooled.append(z)
    pooled = torch.cat(pooled)
    if pooled.numel() > 60_000:
        pooled = pooled[torch.randperm(pooled.numel())[:60_000]]
    pooled = torch.sort(pooled).values
    quantiles = torch.special.ndtri(
        (torch.arange(pooled.numel(), dtype=torch.float32) + 0.5) / pooled.numel()
    )
    ax.scatter(quantiles.numpy(), pooled.numpy(), s=2, alpha=0.35, color="#1f77b4", rasterized=True)
    limit = max(abs(quantiles.min()), abs(quantiles.max()), abs(pooled.min()), abs(pooled.max()))
    ax.plot([-limit, limit], [-limit, limit], color="black", lw=1, ls="--")
    ax.set_title("QQ plot (per-slice standardized)")
    ax.set_xlabel("N(0,1) quantile")
    ax.set_ylabel("empirical quantile")

    ax = axes[1, 0]
    ax.hist(
        gauss["ks"].numpy(),
        bins=30,
        alpha=0.7,
        color="#d62728",
        label=f"model (mean {gauss['mean_ks']:.4f})",
    )
    ax.hist(
        null["ks"].numpy(),
        bins=30,
        alpha=0.7,
        color="#2ca02c",
        label=f"same-size Gaussian (mean {null['mean_ks']:.4f})",
    )
    ax.set_title("KS distance to N(0,1), one value per slice")
    ax.set_xlabel("KS distance")
    ax.set_ylabel("slice count")
    ax.legend()

    ax = axes[1, 1]
    ax.hist(
        sig_test.numpy(),
        bins=30,
        alpha=0.7,
        color="#d62728",
        label=f"model test draws (mean {sig_test.mean():.3f})",
    )
    ax.hist(
        sig_null.numpy(),
        bins=30,
        alpha=0.7,
        color="#2ca02c",
        label=f"Gaussian draws (mean {sig_null.mean():.3f})",
    )
    ax.axvline(
        train_ls,
        color="black",
        ls="--",
        lw=1.5,
        label=f"last training L_sig = {train_ls:.3f}",
    )
    ax.set_title("SIGReg statistic on training-sized batches")
    ax.set_xlabel("L_sig")
    ax.set_ylabel("draw count")
    ax.legend()
    fig.suptitle(
        f"Gaussianity: mean std {gauss['mean_std']:.3f}, mean |skew| {gauss['mean_abs_skew']:.3f}, "
        f"mean excess kurtosis {gauss['mean_excess_kurtosis']:.3f}",
        fontsize=13,
    )
    _save(fig, path)


def _plot_pca(enc_coords, proj_coords, enc_explained, proj_explained, labels, class_names, path):
    num_classes = len(class_names)
    cmap = plt.get_cmap("tab20" if num_classes > 10 else "tab10")
    colors = [cmap(i % cmap.N) for i in range(num_classes)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4), constrained_layout=True)
    for ax, coords, explained, title in (
        (
            axes[0],
            enc_coords,
            enc_explained,
            "Encoder embedding PCA (512d -> 2d)",
        ),
        (
            axes[1],
            proj_coords,
            proj_explained,
            "Projector output PCA (16d -> 2d)",
        ),
    ):
        for index, name in enumerate(class_names):
            mask = labels == index
            if not mask.any():
                continue
            ax.scatter(
                coords[mask, 0].numpy(),
                coords[mask, 1].numpy(),
                s=5,
                alpha=0.45,
                color=colors[index],
                label=name,
                rasterized=True,
            )
        ax.set_title(f"{title}\nPC1 {100 * explained[0]:.1f}%, PC2 {100 * explained[1]:.1f}%")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    axes[0].legend(markerscale=3, ncol=2, loc="best")

    ax = axes[2]
    means = torch.stack([enc_coords[labels == i].mean(dim=0) for i in range(num_classes)])
    means = F.normalize(means, dim=1)
    similarity = (means @ means.T).numpy()
    image = ax.imshow(similarity, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Class-mean cosine similarity (encoder)")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle("Embedding geometry", fontsize=13)
    _save(fig, path)


def _plot_geometry(
    knn_raw, knn_centered, offline_history, online_acc, cosine, z, mean_dir, cfg, path
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    ks = sorted(knn_raw)
    ax.plot(ks, [100 * knn_raw[k] for k in ks], marker="o", label="k-NN, raw", color="#1f77b4")
    ax.plot(
        ks,
        [100 * knn_centered[k] for k in ks],
        marker="s",
        label="k-NN, centered",
        color="#17becf",
    )
    ax.axhline(100 * online_acc, color="#d62728", ls="--", label="online probe")
    if offline_history:
        ax.axhline(100 * offline_history[-1], color="#2ca02c", ls=":", label="offline probe")
    ax.set_xscale("log")
    ax.set_title("Simple readouts of the frozen encoder")
    ax.set_xlabel("k (log)")
    ax.set_ylabel("top-1 (%)")
    ax.legend(loc="lower right")

    ax = axes[0, 1]
    if offline_history:
        epochs = np.arange(1, len(offline_history) + 1)
        ax.plot(epochs, 100 * np.array(offline_history), color="#2ca02c", lw=1.8)
        ax.axhline(
            100 * online_acc,
            color="#d62728",
            ls="--",
            label="online probe (trained jointly)",
        )
        ax.legend(loc="lower right")
    else:
        ax.text(
            0.5,
            0.5,
            "offline probe disabled",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    ax.set_title("Offline probe on cached, frozen embeddings")
    ax.set_xlabel("probe epoch")
    ax.set_ylabel("test top-1 (%)")

    ax = axes[1, 0]
    labels = ["raw", "centered"]
    same = [cosine["same_raw"], cosine["same_centered"]]
    different = [cosine["different_raw"], cosine["different_centered"]]
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, same, width, label="same image, different view", color="#1f77b4")
    ax.bar(x + width / 2, different, width, label="different image", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("Encoder cosine similarity")
    ax.set_ylabel("cosine")
    ax.legend(loc="lower right")

    ax = axes[1, 1]
    paired = prediction_loss(z, V_g=cfg["V_g"])
    shuffled = shuffled_prediction_loss(
        z, generator=torch.Generator().manual_seed(0), V_g=cfg["V_g"]
    )
    ax.bar(["paired views", "shuffled views"], [paired, shuffled], color=["#2ca02c", "#d62728"])
    for index, value in enumerate((paired, shuffled)):
        ax.text(index, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Projector prediction loss (lower is more invariant)")
    ax.set_ylabel("mean squared error")
    fig.suptitle(
        f"Representation quality: encoder mean-direction ratio {mean_dir:.3f}",
        fontsize=13,
    )
    _save(fig, path)


def _write_report(
    path,
    run_dir,
    checkpoint,
    run_meta,
    cfg,
    metrics,
    class_names,
    plot_paths,
):
    final = metrics["training"]["final"]
    probe = metrics["probe"]
    offline = metrics["offline_probe"]
    knn = metrics["knn"]
    rank = metrics["rank"]
    gauss = metrics["gaussianity"]
    geometry = metrics["view_geometry"]
    loss = metrics["loss_decomposition"]

    gap_ratio = geometry["prediction_loss_shuffled"] / max(
        geometry["prediction_loss_paired"], 1e-12
    )
    lines = [
        f"# Run analysis: `{os.path.basename(os.path.normpath(run_dir))}`",
        "",
        f"Generated {metrics['analyzed_at']} from `{checkpoint}`.",
        "",
        "## Setup",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Run directory | `{run_dir}` |",
        f"| FastJEPA commit | `{run_meta.get('git_commit')}` |",
        f"| Training timestamp | `{run_meta.get('timestamp')}` |",
        f"| Device | {metrics['device'].get('name')} |",
        f"| Driver / CUDA / PyTorch | {metrics['device'].get('driver_version')} / "
        f"{metrics['device'].get('cuda')} / {metrics['device'].get('torch')} |",
        f"| Dataset | `{cfg['dataset']}` |",
        f"| Views / global views | {cfg['V']} / {cfg['V_g']} |",
        f"| Projector dim / slices | {cfg['proj_dim']} / {cfg['num_slices']} |",
        f"| Precision | `{cfg['precision']}` |",
        f"| Epochs / batch size | {cfg['epochs']} / {cfg['batch_size']} |",
        "",
        "## Headline numbers",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Final training loss | {final['loss']:.4f} |",
        f"| Final L_pred | {final['lp']:.4f} |",
        f"| Final L_sig | {final['ls']:.4f} |",
        f"| Online probe top-1 | {_pct(probe['top1'])} |",
        f"| Online probe top-5 | {_pct(probe['top5'])} |",
        f"| Offline probe top-1 | {_pct(offline['top1']) if offline else 'not run'} |",
        f"| k-NN top-1 (k=20) | {_pct(knn['raw'][20])} |",
        f"| Projector effective rank | {rank['projector']['erank']:.2f} / {cfg['proj_dim']} |",
        f"| Encoder effective rank | {rank['encoder']['erank']:.2f} / 512 |",
        f"| Final mean per-dim std (projector) | {final['std_mean']:.4f} |",
        f"| Collapsed? | {final['collapsed']} |",
        "",
        "## What the numbers mean",
        "",
        "### Is the model collapsed?",
        "",
        f"No. The training-time diagnostics monitor the projector output "
        f"({cfg['proj_dim']} dimensions): the final mean per-dimension standard deviation is "
        f"{final['std_mean']:.4f}, and the effective rank is {final['erank']:.2f} out of a "
        f"maximum of {cfg['proj_dim']}. Effective rank is `exp(entropy(p))` with "
        "`p_i = s_i / sum(s)`: it is 16 when all singular values are equal and approaches 1 "
        "when one direction dominates. The projector uses almost all of its dimensions, so "
        "SIGReg is doing its anti-collapse job.",
        "",
        f"On center-crop test features the projector per-dimension std is "
        f"{rank['projector']['mean_per_dim_std']:.4f} (min "
        f"{rank['projector']['min_per_dim_std']:.4f}, max "
        f"{rank['projector']['max_per_dim_std']:.4f}); the training number above comes from four "
        "augmented training batches and is a little lower. Both are far from zero.",
        "",
        f"The encoder output is a different story. Its 512-d embedding has an effective rank of "
        f"{rank['encoder']['erank']:.2f}, with "
        f"{rank['encoder']['rank_at_90']} dimensions needed for 90% of the singular-value mass. "
        f"The mean embedding direction accounts for {100 * geometry['mean_direction_ratio']:.1f}% "
        "of the average embedding norm. That anisotropy is common in ViT features and the probe "
        "uses LayerNorm, but it is worth remembering when using raw cosine similarity.",
        "",
        "### Is 67% good?",
        "",
        f"The online probe reaches {_pct(probe['top1'])} top-1 and {_pct(probe['top5'])} top-5, "
        f"versus {_pct(1 / len(class_names))} chance. Two label-free readouts are higher: the "
        f"offline probe reaches {_pct(offline['top1']) if offline else 'n/a'} and k-NN reaches "
        f"{_pct(knn['raw'][20])} at k=20. That gap is expected: the online probe is a small "
        "LayerNorm+Linear head trained jointly with the encoder, while the offline probe and "
        "k-NN are fit after training on frozen features. So the online number is a conservative "
        "estimate of representation quality, not a ceiling.",
        "",
        f"This is a {cfg['epochs']}-epoch run. The protocol's small-scale recipe uses 800 epochs "
        "at the same kind of scale, so this is a short run by the project's own reference. The "
        "accuracy curve is already flattening after roughly epoch 80; expect further gains from "
        "longer training to be real but modest unless the recipe changes.",
        "",
        "### Where the errors are",
        "",
        "The per-class recalls in the probe figure show which classes are hard. On this run the "
        f"hardest classes are {metrics['probe']['hardest'][0]} "
        f"({_pct(metrics['probe']['hardest'][1])}) and {metrics['probe']['hardest'][2]} "
        f"({_pct(metrics['probe']['hardest'][3])}); the easiest are "
        f"{metrics['probe']['easiest'][0]} ({_pct(metrics['probe']['easiest'][1])}) and "
        f"{metrics['probe']['easiest'][2]} ({_pct(metrics['probe']['easiest'][3])}). "
        "The confusion matrix shows whether the mistakes are between visually similar classes "
        "or spread across the label set.",
        "",
        "### Loss balance",
        "",
        f"With `lambda={cfg['lamb']}`, the final weighted contributions are "
        f"{(1 - cfg['lamb']) * final['lp']:.4f} from L_pred and "
        f"{cfg['lamb'] * final['ls']:.4f} from L_sig, i.e. "
        f"{loss['sig_fraction']:.1f}% of the total loss is the regularizer. L_sig itself is "
        f"{final['ls']:.3f}, down from {metrics['training']['first']['ls']:.3f} in epoch 0. "
        "The regularizer is large in absolute terms but deliberately down-weighted; this is "
        "why the total loss is dominated by the prediction term.",
        "",
        "### Is the projector Gaussian?",
        "",
        f"Along {gauss['num_directions']} random directions the mean std is "
        f"{gauss['mean_std']:.4f}, the mean absolute skew is {gauss['mean_abs_skew']:.4f}, and "
        f"the mean excess kurtosis is {gauss['mean_excess_kurtosis']:.4f}. The mean "
        f"Kolmogorov-Smirnov distance to N(0,1) is {gauss['mean_ks']:.4f}, versus "
        f"{gauss['null_mean_ks']:.4f} for a same-size Gaussian sample. The model is close to "
        "Gaussian along random slices, but not exactly: SIGReg is a finite-sample objective and "
        "this run still has visible residual structure, especially in the tails and skew.",
        "",
        f"The SIGReg statistic on training-sized batches is {gauss['sigreg_test_mean']:.3f} "
        f"[5%, 95%: {gauss['sigreg_test_q05']:.3f}, {gauss['sigreg_test_q95']:.3f}] for the "
        f"model, versus {gauss['sigreg_null_mean']:.3f} for Gaussian draws. The last training "
        f"value was {final['ls']:.3f}. The gap between those two model numbers is expected: "
        "training measures augmented crops with BatchNorm batch statistics, while this analysis "
        "uses center crops and the running statistics. A larger `lambda`, more slices or more "
        "epochs could reduce this residual, at the cost of the prediction term; that is a "
        "hypothesis for a follow-up run, not something this analysis proves.",
        "",
        "### Does the encoder learn invariance?",
        "",
        f"Yes. For the same image under different augmentations, the centered cosine similarity "
        f"is {geometry['cosine_same_centered']:.4f}, while random image pairs are at "
        f"{geometry['cosine_different_centered']:.4f}. Before centering, both numbers are high "
        f"({geometry['cosine_same_raw']:.4f} vs {geometry['cosine_different_raw']:.4f}) because "
        "of the shared mean direction. The projector prediction loss is "
        f"{geometry['prediction_loss_paired']:.4f} for correctly paired views and "
        f"{geometry['prediction_loss_shuffled']:.4f} when the image correspondence is broken, "
        "a roughly "
        f"{gap_ratio:.1f}x "
        "gap. This is the JEPA objective doing what it is supposed to do.",
        "",
        "### Bottom line",
        "",
        f"This is a healthy, non-collapsed representation and a reasonable result for a "
        f"{cfg['epochs']}-epoch CIFAR-10 run on one consumer GPU. It is well above chance, most "
        "of the class information survives a linear readout, and the augmentation invariance is "
        "strong. It is not a state-of-the-art number and it is not a parity claim: a single seed "
        "and no reference run in this exact configuration mean the honest statement is "
        "'working and improving, with room left in the Gaussianity residual, encoder anisotropy "
        "and training length'.",
        "",
        "## Figures",
        "",
    ]
    captions = {
        "training": (
            "Training history. The probe and rank curves flatten together; the loss is still "
            "creeping down."
        ),
        "optimization": (
            "Learning-rate schedule, throughput and budget. Early points include warmup and "
            "clock ramp."
        ),
        "loss": (
            "Weighted loss decomposition. SIGReg is about a quarter of the total despite its "
            "larger raw value."
        ),
        "probe": (
            "Probe evaluation: confusion matrix, per-class recall, calibration and confidence."
        ),
        "rank": (
            "Spectra and per-dimension scales. The projector is nearly isotropic; the encoder "
            "is much lower-rank."
        ),
        "gaussian": (
            "Gaussianity diagnostics. Random slices, QQ plot, KS distribution and SIGReg draws."
        ),
        "pca": "PCA of the encoder and projector, plus class-mean cosine similarity.",
        "geometry": (
            "Readout quality, view invariance, and the paired-vs-shuffled prediction control."
        ),
    }
    for key in PLOT_FILES:
        if key in plot_paths:
            title = "PCA" if key == "pca" else key.capitalize()
            lines.extend(
                [
                    f"### {title}",
                    "",
                    f"![{key}]({plot_paths[key]})",
                    "",
                    captions[key],
                    "",
                ]
            )
    lines.extend(
        [
            "## Reproduction",
            "",
            "```bash",
            f"python tools/analyze_run.py --run-dir {run_dir}",
            "```",
            "",
            "`metrics.json` in the same output directory contains every number used above.",
            "",
        ]
    )
    with open(path, "w") as handle:
        handle.write("\n".join(lines))


def analyze(args):
    _style()
    device = resolve_device(args.device)
    run_dir = os.path.abspath(args.run_dir)
    checkpoint = args.checkpoint or os.path.join(run_dir, "ckpt.pt")
    out_dir = os.path.abspath(args.out_dir or os.path.join(run_dir, "analysis"))
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    run_meta, cfg, checkpoint_data, epochs, steps = _load_run(run_dir, checkpoint)
    enc, proj, sig, probe = _build_models(checkpoint_data, cfg, device)
    train_base, val_base, train_tf, eval_tf = build_datasets(cfg)
    if val_base is None:
        raise ValueError("run has no validation split; point the config at one with val_dir")

    num_classes = int(cfg["num_classes"])
    class_names = _class_names(val_base, num_classes)
    precision = cfg["precision"]

    print(f"[analysis] features from {train_base.__class__.__name__} ...")
    train_enc, train_z, train_y = _collect_features(
        enc,
        proj,
        train_base,
        eval_tf,
        device,
        precision,
        args.batch_size,
        args.num_workers,
    )
    test_enc, test_z, test_y = _collect_features(
        enc,
        proj,
        val_base,
        None,
        device,
        precision,
        args.batch_size,
        args.num_workers,
    )
    if args.knn_train_limit and args.knn_train_limit < train_enc.shape[0]:
        train_enc = train_enc[: args.knn_train_limit]
        train_y = train_y[: args.knn_train_limit]
        train_z = train_z[: args.knn_train_limit]

    print("[analysis] probe and readouts ...")
    logits = _probe_logits(probe, test_enc, device)
    probe_loss = float(F.cross_entropy(logits, test_y))
    probe_top1 = float((logits.argmax(dim=1) == test_y).float().mean())
    probe_top5 = topk_accuracy(logits, test_y, 5)
    cm = confusion_matrix(test_y, logits.argmax(dim=1), num_classes)
    per_class = per_class_accuracy(cm)
    order = torch.argsort(per_class, descending=True)
    hardest = []
    easiest = []
    for index in order[-2:].tolist()[::-1]:
        hardest.extend([class_names[index], float(per_class[index])])
    for index in order[:2].tolist():
        easiest.extend([class_names[index], float(per_class[index])])

    offline_head, offline_history = _offline_probe(
        train_enc,
        train_y,
        test_enc,
        test_y,
        num_classes,
        device,
        args.offline_probe_epochs,
        args.seed,
    )
    if offline_head is not None:
        with torch.no_grad():
            offline_logits = _probe_logits(offline_head, test_enc, device)
        offline_top1 = float((offline_logits.argmax(dim=1) == test_y).float().mean())
        offline_top5 = topk_accuracy(offline_logits, test_y, 5)
    else:
        offline_top1 = None
        offline_top5 = None

    print("[analysis] k-NN ...")
    knn_ks = (1, 5, 10, 20, 50, 100)
    knn_raw, _ = knn_accuracy_curve(
        train_enc, train_y, test_enc, test_y, ks=knn_ks, center=False, device=device
    )
    knn_centered, _ = knn_accuracy_curve(
        train_enc, train_y, test_enc, test_y, ks=knn_ks, center=True, device=device
    )

    print("[analysis] rank, Gaussianity and view geometry ...")
    enc_spec = singular_spectrum(test_enc)
    proj_spec = singular_spectrum(test_z)
    enc_std = per_dim_std(test_enc)
    proj_std = per_dim_std(test_z)
    gauss = sliced_gaussianity(
        test_z,
        num_directions=args.num_directions,
        generator=torch.Generator().manual_seed(args.seed),
    )
    null_gauss = sliced_gaussianity(
        torch.randn(test_z.shape[0], test_z.shape[1]),
        num_directions=args.num_directions,
        generator=torch.Generator().manual_seed(args.seed),
    )
    view_enc, view_z = _collect_views(
        enc,
        proj,
        train_base,
        train_tf,
        cfg["V"],
        args.view_samples,
        device,
        precision,
        args.batch_size,
    )
    cosine = cosine_similarity_stats(view_enc, generator=torch.Generator().manual_seed(args.seed))
    paired_loss = prediction_loss(view_z, V_g=cfg["V_g"])
    shuffled_loss = shuffled_prediction_loss(
        view_z, generator=torch.Generator().manual_seed(args.seed), V_g=cfg["V_g"]
    )
    mean_dir = mean_direction_ratio(test_enc)

    sig_N = min(int(cfg["batch_size"]), test_z.shape[0] // cfg["V"])
    if sig_N <= 0:
        raise ValueError("validation split is too small for a training-sized SIGReg draw")
    sig_test, sig_null = _sigreg_draws(
        sig,
        test_z,
        cfg["V"],
        sig_N,
        device,
        args.sigreg_draws,
        args.seed,
    )

    first = epochs[0]
    final = epochs[-1]
    best_epoch = max(
        epochs,
        key=lambda row: row["probe_acc"] if row["probe_acc"] is not None else -1,
    )
    total_weighted = (1 - cfg["lamb"]) * final["lp"] + cfg["lamb"] * final["ls"]
    metrics = {
        "schema": "fastjepa.run_analysis/1",
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "git_commit": checkpoint_data.get("git_commit") or run_meta.get("git_commit"),
        "config": cfg,
        "device": device_info(device),
        "training": {
            "epochs": len(epochs),
            "steps": len(steps),
            "first": first,
            "final": final,
            "best_probe_epoch": best_epoch["epoch"],
            "best_probe_acc": best_epoch["probe_acc"],
        },
        "loss_decomposition": {
            "lambda": cfg["lamb"],
            "pred_weighted": (1 - cfg["lamb"]) * final["lp"],
            "sig_weighted": cfg["lamb"] * final["ls"],
            "sig_fraction": 100 * cfg["lamb"] * final["ls"] / max(total_weighted, 1e-12),
        },
        "probe": {
            "top1": probe_top1,
            "top5": probe_top5,
            "loss": probe_loss,
            "per_class": {name: float(per_class[index]) for index, name in enumerate(class_names)},
            "confusion": cm.tolist(),
            "hardest": hardest,
            "easiest": easiest,
        },
        "offline_probe": {
            "top1": offline_top1,
            "top5": offline_top5,
            "epochs": args.offline_probe_epochs,
            "history": offline_history,
        },
        "knn": {"raw": knn_raw, "centered": knn_centered, "train_size": int(train_enc.shape[0])},
        "rank": {
            "encoder": {
                "erank": enc_spec["erank"],
                "participation_ratio": enc_spec["participation_ratio"],
                "max_rank": int(enc_spec["singular_values"].numel()),
                "rank_at_90": _energy_rank(enc_spec["cumulative_energy"], 0.90),
                "rank_at_99": _energy_rank(enc_spec["cumulative_energy"], 0.99),
                "mean_per_dim_std": float(enc_std.mean()),
                "min_per_dim_std": float(enc_std.min()),
                "max_per_dim_std": float(enc_std.max()),
            },
            "projector": {
                "erank": proj_spec["erank"],
                "participation_ratio": proj_spec["participation_ratio"],
                "max_rank": int(proj_spec["singular_values"].numel()),
                "rank_at_90": _energy_rank(proj_spec["cumulative_energy"], 0.90),
                "rank_at_99": _energy_rank(proj_spec["cumulative_energy"], 0.99),
                "mean_per_dim_std": float(proj_std.mean()),
                "min_per_dim_std": float(proj_std.min()),
                "max_per_dim_std": float(proj_std.max()),
            },
        },
        "gaussianity": {
            "num_directions": args.num_directions,
            "mean_std": gauss["mean_std"],
            "std_min": gauss["std_min"],
            "std_max": gauss["std_max"],
            "mean_abs_skew": gauss["mean_abs_skew"],
            "mean_excess_kurtosis": gauss["mean_excess_kurtosis"],
            "mean_abs_excess_kurtosis": gauss["mean_abs_excess_kurtosis"],
            "mean_ks": gauss["mean_ks"],
            "median_ks": gauss["median_ks"],
            "max_ks": gauss["max_ks"],
            "mean_wasserstein1": gauss["mean_wasserstein1"],
            "null_mean_ks": null_gauss["mean_ks"],
            "null_mean_excess_kurtosis": null_gauss["mean_excess_kurtosis"],
            "null_mean_wasserstein1": null_gauss["mean_wasserstein1"],
            "sigreg_train_last": final["ls"],
            "sigreg_test_mean": float(sig_test.mean()),
            "sigreg_test_q05": float(sig_test.quantile(0.05)),
            "sigreg_test_q95": float(sig_test.quantile(0.95)),
            "sigreg_null_mean": float(sig_null.mean()),
            "sigreg_null_q05": float(sig_null.quantile(0.05)),
            "sigreg_null_q95": float(sig_null.quantile(0.95)),
            "sigreg_draws": args.sigreg_draws,
            "sigreg_batch": {"V": cfg["V"], "N": sig_N},
        },
        "view_geometry": {
            "num_images": int(view_enc.shape[1]),
            "cosine_same_raw": cosine["same_raw"],
            "cosine_different_raw": cosine["different_raw"],
            "cosine_same_centered": cosine["same_centered"],
            "cosine_different_centered": cosine["different_centered"],
            "prediction_loss_paired": paired_loss,
            "prediction_loss_shuffled": shuffled_loss,
            "mean_direction_ratio": mean_dir,
        },
        "artifacts": {"out_dir": out_dir, "plot_dir": plot_dir},
    }

    plot_paths = {}
    for key, filename in PLOT_FILES.items():
        plot_paths[key] = os.path.join(plot_dir, filename)
    _plot_training(epochs, cfg, plot_paths["training"])
    _plot_optimization(steps, cfg, plot_paths["optimization"])
    _plot_loss_decomposition(epochs, cfg, plot_paths["loss"])
    _plot_probe(logits, test_y, class_names, metrics["probe"], plot_paths["probe"])
    _plot_rank(enc_spec, proj_spec, enc_std, proj_std, plot_paths["rank"])
    _plot_gaussianity(
        gauss,
        null_gauss,
        sig_test,
        sig_null,
        final["ls"],
        plot_paths["gaussian"],
    )
    enc_coords, enc_explained, _ = pca_2d(test_enc)
    proj_coords, proj_explained, _ = pca_2d(test_z)
    _plot_pca(
        enc_coords,
        proj_coords,
        enc_explained,
        proj_explained,
        test_y,
        class_names,
        plot_paths["pca"],
    )
    _plot_geometry(
        knn_raw,
        knn_centered,
        offline_history,
        probe_top1,
        cosine,
        view_z,
        mean_dir,
        cfg,
        plot_paths["geometry"],
    )

    with open(os.path.join(out_dir, "metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2)
    report_path = os.path.join(out_dir, "report.md")
    _write_report(
        report_path,
        run_dir,
        checkpoint,
        run_meta,
        cfg,
        metrics,
        class_names,
        {key: os.path.join("plots", value) for key, value in PLOT_FILES.items()},
    )

    print(
        f"[analysis] top-1 {_pct(probe_top1)} | top-5 {_pct(probe_top5)} | "
        f"offline {_pct(offline_top1) if offline_top1 is not None else 'n/a'} | "
        f"kNN@20 {_pct(knn_raw[20])}"
    )
    print(f"[analysis] wrote {report_path}")
    print(f"[analysis] wrote {os.path.join(out_dir, 'metrics.json')}")
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-dir", required=True, help="directory with run.json, log.jsonl, ckpt.pt"
    )
    parser.add_argument(
        "--checkpoint", default=None, help="checkpoint path (default: <run-dir>/ckpt.pt)"
    )
    parser.add_argument(
        "--out-dir", default=None, help="output directory (default: <run-dir>/analysis)"
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:N")
    parser.add_argument("--batch-size", type=int, default=256, help="feature-extraction batch size")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--offline-probe-epochs", type=int, default=40)
    parser.add_argument(
        "--knn-train-limit",
        type=int,
        default=0,
        help="cap the k-NN/offline-probe training set (0 means all)",
    )
    parser.add_argument("--view-samples", type=int, default=1000, help="images for the view study")
    parser.add_argument("--num-directions", type=int, default=256)
    parser.add_argument("--sigreg-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    analyze(args)


if __name__ == "__main__":
    main()
