import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

import torch
from lejepa.diagnostics import embedding_diagnostics
from lejepa.loss import lejepa_loss
from lejepa.sigreg import SlicedEppsPulley
from lejepa.train import to_view_major


def test_view_major_layout_and_probe_labels():
    B, V = 6, 3
    views = torch.arange(B * V).reshape(B, V, 1, 1, 1).float()
    z = to_view_major(views).view(V, B, -1).squeeze(-1).long()
    expected = torch.arange(B * V).reshape(B, V).permute(1, 0)
    assert torch.equal(z, expected)

    labels = torch.arange(B)
    probe_labels = labels.repeat(V)
    for v in range(V):
        for b in range(B):
            assert probe_labels[v * B + b].item() == labels[b].item()


def test_lejepa_loss_uses_one_sigreg_call_with_all_views():
    calls = []

    class Recorder:
        def __call__(self, x, A=None):
            calls.append(tuple(x.shape))
            return x.sum() * 0.0

    z = torch.randn(4, 8, 16)
    lejepa_loss(z, Recorder(), 0.02)
    assert calls == [(4, 8, 16)]


def test_lejepa_loss_rejects_unknown_reduction():
    sig = SlicedEppsPulley(16, num_slices=8)
    z = torch.randn(2, 4, 16)
    try:
        lejepa_loss(z, sig, pred_reduction="typo")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown pred_reduction")


def test_diagnostics_collapse_is_finite():
    for z in (torch.zeros(64, 16), torch.ones(64, 16)):
        d = embedding_diagnostics(z)
        assert d["erank"] == 0.0
        assert d["collapsed"] is True
        assert d["std_mean"] == 0.0


def test_diagnostics_random_is_not_collapsed():
    torch.manual_seed(0)
    d = embedding_diagnostics(torch.randn(256, 16))
    assert d["erank"] > 1.0
    assert d["collapsed"] is False


def test_sigreg_fp64_without_manual_buffer_cast():
    sig = SlicedEppsPulley(4, num_slices=8)
    x = torch.randn(6, 4, dtype=torch.float64)
    out = sig(x)
    assert out.dtype == torch.float64
    assert torch.isfinite(out)


def test_sigreg_3d_matches_2d_with_shared_directions():
    torch.manual_seed(0)
    sig = SlicedEppsPulley(4, num_slices=8)
    z = torch.randn(3, 5, 4)
    A = torch.randn(4, 8)
    A = A / A.norm(dim=0, keepdim=True)
    v3 = sig(z, A=A)
    v2 = torch.stack([sig(z[i], A=A) for i in range(3)]).mean()
    assert torch.allclose(v3, v2, atol=1e-6)
