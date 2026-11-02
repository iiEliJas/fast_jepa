import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))
import torch
from lejepa.loss import lejepa_loss
from lejepa.sigreg import SlicedEppsPulley


def test_sigreg_finite():
    m = SlicedEppsPulley(8, 32)
    x = torch.randn(16, 8)
    v = m(x)
    assert torch.isfinite(v)


def test_loss_finite():
    sig = SlicedEppsPulley(8, 32)
    z = torch.randn(4, 8, 8)
    loss, lp, ls = lejepa_loss(z, sig, 0.02)
    assert torch.isfinite(loss)
