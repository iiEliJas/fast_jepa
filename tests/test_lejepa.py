import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))
import torch
from lejepa.loss import lejepa_loss
from lejepa.sigreg import SlicedEppsPulley


def test_sigreg_finite():
    m = SlicedEppsPulley(8, 32)
    x = torch.randn(16, 8)
    v = m(x)
    assert torch.isfinite(v)


def test_sigreg_matches_committed_reference():
    """Guard the quadrature and weights against accidental numeric drift."""
    ref_path = Path(__file__).resolve().parents[1] / "tests" / "data" / "sigreg_ref.pt"
    data = torch.load(ref_path, weights_only=False)
    m = SlicedEppsPulley(data["x"].shape[1], num_slices=data["A"].shape[1])
    value = m(data["x"], A=data["A"])
    assert torch.allclose(value, data["val"], atol=1e-6, rtol=1e-6)


def test_loss_finite():
    sig = SlicedEppsPulley(8, 32)
    z = torch.randn(4, 8, 8)
    loss, lp, ls = lejepa_loss(z, sig, 0.02)
    assert torch.isfinite(loss)
