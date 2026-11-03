import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

import torch
from lejepa.analysis import (
    confusion_matrix,
    knn_accuracy_curve,
    ks_distance_to_normal,
    per_class_accuracy,
    prediction_loss,
    singular_spectrum,
    sliced_gaussianity,
    topk_accuracy,
)
from lejepa.diagnostics import embedding_diagnostics
from lejepa.loss import lejepa_loss
from lejepa.model import LeJEPAEncoder, Projector
from lejepa.sigreg import SlicedEppsPulley


def test_model_forward():
    enc = LeJEPAEncoder(img_size=32, patch_size=4, embed_dim=64, depth=2, num_heads=2)
    proj = Projector(512, 128, 16)
    x = torch.randn(4, 3, 32, 32)
    e = enc(x)
    z = proj(e)
    assert e.shape == (4, 512)
    assert z.shape == (4, 16)
    # proj_head always outputs 512, so assert the timm overrides directly.
    assert enc.vit.embed_dim == 64
    assert len(enc.vit.blocks) == 2
    assert tuple(enc.vit.patch_embed.patch_size) == (4, 4)


def test_loss_and_diagnostics():
    sig = SlicedEppsPulley(8, 32)
    z = torch.randn(4, 8, 8)
    loss, lp, ls = lejepa_loss(z, sig, 0.02)
    assert torch.isfinite(loss)
    d = embedding_diagnostics(z.view(-1, 8))
    assert "erank" in d


def test_singular_spectrum_full_rank_and_collapse():
    isotropic = torch.cat([torch.eye(8), -torch.eye(8)])
    stats = singular_spectrum(isotropic)
    assert abs(stats["erank"] - 8.0) < 1e-5
    assert abs(stats["participation_ratio"] - 8.0) < 1e-5
    assert torch.allclose(stats["cumulative_energy"][-1], torch.tensor(1.0))

    collapsed = singular_spectrum(torch.zeros(32, 16))
    assert collapsed["erank"] == 0.0
    assert collapsed["participation_ratio"] == 0.0
    assert torch.isfinite(collapsed["cumulative_energy"]).all()


def test_ks_distance_to_normal_is_small_for_normal_and_large_for_shift():
    torch.manual_seed(0)
    assert ks_distance_to_normal(torch.randn(10_000)) < 0.02
    assert ks_distance_to_normal(torch.randn(10_000) + 2.0) > 0.3


def test_sliced_gaussianity_recovers_standard_normal():
    torch.manual_seed(0)
    sample = torch.randn(20_000, 16)
    stats = sliced_gaussianity(sample, num_directions=64)
    assert abs(stats["mean_std"] - 1.0) < 0.05
    assert stats["mean_abs_skew"] < 0.06
    assert stats["mean_abs_excess_kurtosis"] < 0.10
    assert stats["mean_ks"] < 0.02


def test_confusion_matrix_per_class_and_topk():
    y_true = torch.tensor([0, 1, 2, 2])
    y_pred = torch.tensor([0, 2, 2, 2])
    cm = confusion_matrix(y_true, y_pred, 3)
    assert cm.tolist() == [[1, 0, 0], [0, 0, 1], [0, 0, 2]]
    assert per_class_accuracy(cm).tolist() == [1.0, 0.0, 1.0]

    logits = torch.tensor([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]])
    labels = torch.tensor([0, 1, 2])
    assert topk_accuracy(logits, labels, 1) == 1.0
    assert topk_accuracy(logits, labels, 2) == 1.0


def test_knn_accuracy_curve_separates_clusters():
    centers = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    train_x = torch.cat([c + 0.01 * torch.randn(30, 2) for c in centers])
    train_y = torch.cat([torch.full((30,), i) for i in range(3)])
    test_x = torch.cat([c + 0.01 * torch.randn(10, 2) for c in centers])
    test_y = torch.cat([torch.full((10,), i) for i in range(3)])
    acc, n = knn_accuracy_curve(train_x, train_y, test_x, test_y, ks=(1, 5))
    assert n == 30
    assert acc[1] > 0.99
    assert acc[5] > 0.99


def test_prediction_loss_is_zero_for_identical_views():
    z = torch.randn(4, 8, 16)
    same = z[0].unsqueeze(0).expand(4, -1, -1).clone()
    assert prediction_loss(same) == 0.0
    assert prediction_loss(z) > 0.0
