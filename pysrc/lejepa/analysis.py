"""Numerical helpers for inspecting a trained LeJEPA run.

This module is deliberately free of plotting and model imports: it only needs
torch, so tests and tools can reuse the same definitions.  ``tools/analyze_run.py``
turns these numbers into figures and a Markdown report.

The functions answer the questions that matter after a run:

* How much of the representation is actually used? (:func:`singular_spectrum`)
* Is the projector output close to the Gaussian SIGReg targets?
  (:func:`sliced_gaussianity`, :func:`ks_distance_to_normal`)
* Does the encoder collapse? (:func:`per_dim_std`,
  :func:`mean_direction_ratio`)
* Are augmented views of the same image closer than random images?
  (:func:`cosine_similarity_stats`, :func:`prediction_loss`)
* How well do simple readouts do? (:func:`knn_accuracy_curve`,
  :func:`topk_accuracy`, :func:`confusion_matrix`)
"""

import torch
import torch.nn.functional as F


def _check_2d(name, x):
    if x.dim() != 2:
        raise ValueError(f"{name} must be 2D (N, D), got {tuple(x.shape)}")
    if x.numel() == 0:
        raise ValueError(f"{name} must not be empty")


def singular_spectrum(x):
    """Effective rank and spectrum of a centered embedding matrix.

    ``erank = exp(entropy(p))`` with ``p_i = s_i / sum(s)``.  All equal singular
    values give ``erank == D``; one dominant direction gives ``erank -> 1``.
    This is the same definition as ``lejepa.diagnostics.embedding_diagnostics``.
    """
    _check_2d("x", x)
    x = x.float()
    xc = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(xc)
    total = s.sum()
    if not torch.isfinite(total) or total <= 0:
        p = torch.zeros_like(s)
        erank = 0.0
        participation = 0.0
        cumulative = torch.zeros_like(s)
    else:
        p = s / total
        erank = float(torch.exp(torch.special.entr(p).sum()))
        participation = float(total.square() / s.square().sum())
        cumulative = torch.cumsum(s, dim=0) / total
    return {
        "singular_values": s,
        "probabilities": p,
        "cumulative_energy": cumulative,
        "erank": erank,
        "participation_ratio": participation,
        "total": float(total),
    }


def per_dim_std(x):
    """Population standard deviation of every feature dimension."""
    _check_2d("x", x)
    return x.float().std(dim=0, unbiased=False)


def mean_direction_ratio(x):
    """||mean(x)|| / mean(||x||).

    High values mean a common offset dominates the embedding norm.  The raw
    cosine similarity between two samples then looks high even when the
    centered representations are unrelated.
    """
    _check_2d("x", x)
    x = x.float()
    mean_norm = x.mean(dim=0).norm()
    row_norm = x.norm(dim=1).mean()
    if row_norm <= 0:
        return 0.0
    return float(mean_norm / row_norm)


def normal_quantiles(n, device=None, dtype=torch.float32):
    """Quantiles of N(0, 1) at the midpoints of ``n`` equal-probability bins."""
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    q = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n
    return torch.special.ndtri(q)


def ks_distance_to_normal(x):
    """Kolmogorov-Smirnov distance between ``x`` and N(0, 1).

    The data are *not* standardized first, so a shifted or rescaled
    distribution is penalized.  This complements the SIGReg value, which is a
    weighted characteristic-function statistic.
    """
    if x.dim() != 1:
        raise ValueError(f"x must be 1D, got {tuple(x.shape)}")
    if x.numel() == 0:
        raise ValueError("x must not be empty")
    x = x.float()
    xs = torch.sort(x).values
    n = xs.numel()
    ecdf = torch.arange(1, n + 1, device=x.device, dtype=x.dtype) / n
    cdf = torch.special.ndtr(xs)
    left = torch.cat([torch.zeros(1, device=x.device, dtype=x.dtype), ecdf[:-1]])
    return float(torch.maximum((ecdf - cdf).abs(), (left - cdf).abs()).max())


def wasserstein1_to_normal(x):
    """One-dimensional Wasserstein-1 distance to N(0, 1)."""
    if x.dim() != 1:
        raise ValueError(f"x must be 1D, got {tuple(x.shape)}")
    if x.numel() == 0:
        raise ValueError("x must not be empty")
    xs = torch.sort(x.float()).values
    q = normal_quantiles(xs.numel(), device=xs.device, dtype=xs.dtype)
    return float((xs - q).abs().mean())


def sliced_gaussianity(x, num_directions=256, generator=None):
    """Compare random one-dimensional projections of ``x`` to N(0, 1).

    SIGReg only constrains these marginals (Cramer-Wold), not the full joint
    distribution.  This function returns per-slice statistics and their means,
    which is what the training loss is built from.
    """
    _check_2d("x", x)
    x = x.float()
    D = x.shape[1]
    if generator is None:
        generator = torch.Generator(device=x.device).manual_seed(0)
    A = torch.randn(D, num_directions, generator=generator, device=x.device, dtype=x.dtype)
    A = A / A.norm(dim=0, keepdim=True).clamp_min(torch.finfo(x.dtype).tiny)
    proj = x @ A  # (N, S)

    mean = proj.mean(dim=0)
    std = proj.std(dim=0, unbiased=False)
    safe_std = torch.where(std > 0, std, torch.ones_like(std))
    z = (proj - mean) / safe_std
    z = torch.where(std > 0, z, torch.zeros_like(z))
    skew = (z**3).mean(dim=0)
    excess_kurtosis = (z**4).mean(dim=0) - 3.0

    ks = torch.tensor(
        [ks_distance_to_normal(proj[:, i]) for i in range(num_directions)],
        device=proj.device,
        dtype=proj.dtype,
    )
    w1 = torch.tensor(
        [wasserstein1_to_normal(proj[:, i]) for i in range(num_directions)],
        device=proj.device,
        dtype=proj.dtype,
    )
    return {
        "projections": proj,
        "directions": A,
        "mean": mean,
        "std": std,
        "skew": skew,
        "excess_kurtosis": excess_kurtosis,
        "ks": ks,
        "wasserstein1": w1,
        "mean_std": float(std.mean()),
        "std_min": float(std.min()),
        "std_max": float(std.max()),
        "mean_abs_skew": float(skew.abs().mean()),
        "mean_excess_kurtosis": float(excess_kurtosis.mean()),
        "mean_abs_excess_kurtosis": float(excess_kurtosis.abs().mean()),
        "mean_ks": float(ks.mean()),
        "median_ks": float(ks.median()),
        "max_ks": float(ks.max()),
        "mean_wasserstein1": float(w1.mean()),
    }


def pca_2d(x):
    """Project ``x`` onto its first two principal components.

    Returns coordinates, the full explained-variance ratio and the components.
    The covariance route is deterministic and cheap for the feature sizes used
    here (at most a few thousand dimensions).
    """
    _check_2d("x", x)
    x = x.float()
    xc = x - x.mean(dim=0, keepdim=True)
    denom = max(1, x.shape[0] - 1)
    cov = xc.T @ xc / denom
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0)
    evecs = evecs[:, order]
    coords = xc @ evecs[:, :2]
    total = evals.sum()
    explained = evals / total if total > 0 else torch.zeros_like(evals)
    return coords, explained, evecs


def confusion_matrix(y_true, y_pred, num_classes):
    """Rows are true labels, columns are predictions."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {tuple(y_true.shape)} vs {tuple(y_pred.shape)}")
    y_true = y_true.long().flatten()
    y_pred = y_pred.long().flatten()
    flat = y_true * num_classes + y_pred
    return torch.bincount(flat, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )


def per_class_accuracy(cm):
    """Per-class recall from a confusion matrix; classes with no samples get 0."""
    cm = cm.float()
    support = cm.sum(dim=1)
    correct = cm.diag()
    return torch.where(support > 0, correct / support.clamp_min(1), torch.zeros_like(correct))


def topk_accuracy(logits, y_true, k):
    """Top-k accuracy for ``logits`` (N, C) and integer labels ``y_true`` (N,)."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D, got {tuple(logits.shape)}")
    if y_true.dim() != 1:
        raise ValueError(f"y_true must be 1D, got {tuple(y_true.shape)}")
    k = min(k, logits.shape[1])
    top = logits.topk(k, dim=1).indices
    return float((top == y_true[:, None]).any(dim=1).float().mean())


def knn_accuracy_curve(
    train_x,
    train_y,
    test_x,
    test_y,
    ks=(1, 5, 20, 50),
    chunk_size=512,
    center=False,
    device=None,
):
    """Majority-vote k-NN accuracy for several k, using cosine similarity.

    ``center=True`` subtracts the training mean from both sets first.  The
    similarity matrix is computed in test chunks so the full
    ``len(test) x len(train)`` matrix is never materialized.
    """
    _check_2d("train_x", train_x)
    _check_2d("test_x", test_x)
    if train_x.shape[1] != test_x.shape[1]:
        raise ValueError("train_x and test_x must have the same feature dimension")
    if train_y.numel() != train_x.shape[0] or test_y.numel() != test_x.shape[0]:
        raise ValueError("label counts do not match feature rows")
    ks = sorted({int(k) for k in ks})
    if not ks or ks[0] < 1:
        raise ValueError(f"ks must be positive, got {ks}")
    max_k = min(ks[-1], train_x.shape[0])

    train_x = train_x.float()
    test_x = test_x.float()
    if center:
        mean = train_x.mean(dim=0, keepdim=True)
        train_x = train_x - mean
        test_x = test_x - mean
    train_n = F.normalize(train_x, dim=1)
    test_n = F.normalize(test_x, dim=1)
    if device is not None:
        train_n = train_n.to(device)
        test_n = test_n.to(device)
        train_y = train_y.to(device)

    correct = {k: 0 for k in ks}
    for start in range(0, test_n.shape[0], chunk_size):
        sim = test_n[start : start + chunk_size] @ train_n.T
        idx = sim.topk(max_k, dim=1).indices
        labels = train_y[idx]
        for k in ks:
            kk = min(k, max_k)
            pred = torch.mode(labels[:, :kk], dim=1).values
            correct[k] += int((pred == test_y[start : start + chunk_size].to(pred.device)).sum())
    n = test_y.numel()
    return {k: correct[k] / n for k in ks}, n


def prediction_loss(z, V_g=None):
    """LeJEPA prediction term for a ``(V, N, K)`` projector output."""
    if z.dim() != 3:
        raise ValueError(f"z must be 3D (V, N, K), got {tuple(z.shape)}")
    V = z.shape[0]
    if V_g is None:
        V_g = V
    if not 1 <= V_g <= V:
        raise ValueError(f"V_g must be in [1, {V}], got {V_g}")
    mu = z[:V_g].mean(dim=0)
    return float(((mu.unsqueeze(0) - z).square().mean()))


def shuffled_prediction_loss(z, generator=None, V_g=None):
    """Control: break the image correspondence between views, then score."""
    if z.dim() != 3:
        raise ValueError(f"z must be 3D (V, N, K), got {tuple(z.shape)}")
    if generator is None:
        generator = torch.Generator(device=z.device).manual_seed(0)
    N = z.shape[1]
    shuffled = torch.stack(
        [z[v][torch.randperm(N, generator=generator, device=z.device)] for v in range(z.shape[0])]
    )
    return prediction_loss(shuffled, V_g=V_g)


def cosine_similarity_stats(views, generator=None):
    """Raw and centered cosine similarity for same-image vs random-view pairs.

    ``views`` is ``(V, N, D)``; different views of the same row are augmented
    versions of the same image.  The centered variant removes the global mean
    direction, which otherwise inflates every cosine.
    """
    if views.dim() != 3:
        raise ValueError(f"views must be 3D (V, N, D), got {tuple(views.shape)}")
    V, N, D = views.shape
    if V < 2:
        raise ValueError("need at least two views")
    if generator is None:
        generator = torch.Generator(device=views.device).manual_seed(0)

    raw = F.normalize(views.float(), dim=-1)
    centered = F.normalize(views.float() - views.float().mean(dim=(0, 1), keepdim=True), dim=-1)

    same_raw, diff_raw, same_centered, diff_centered = [], [], [], []
    for a in range(V):
        for b in range(a + 1, V):
            perm = torch.randperm(N, generator=generator, device=views.device)
            same_raw.append((raw[a] * raw[b]).sum(dim=-1))
            diff_raw.append((raw[a] * raw[b][perm]).sum(dim=-1))
            same_centered.append((centered[a] * centered[b]).sum(dim=-1))
            diff_centered.append((centered[a] * centered[b][perm]).sum(dim=-1))

    return {
        "same_raw": float(torch.stack(same_raw).mean()),
        "different_raw": float(torch.stack(diff_raw).mean()),
        "same_centered": float(torch.stack(same_centered).mean()),
        "different_centered": float(torch.stack(diff_centered).mean()),
    }
