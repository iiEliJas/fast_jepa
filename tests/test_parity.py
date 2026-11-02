import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))
sys.path.insert(0, os.path.expanduser("~/projects/lejepa"))

import torch

try:
    HAS_OFFICIAL = True
except Exception:
    HAS_OFFICIAL = False

import importlib.util

_spec = importlib.util.spec_from_file_location(
    "fj_sigreg", os.path.join(os.path.dirname(__file__), "..", "pysrc", "lejepa", "sigreg.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
SlicedEppsPulley = _mod.SlicedEppsPulley


def test_sigreg_parity():
    if not HAS_OFFICIAL:
        import pytest

        pytest.skip("official lejepa not available")
    try:
        from lejepa.univariate import EppsPulley
    except Exception:
        import pytest

        pytest.skip("official lejepa not available")

    torch.manual_seed(0)
    N, K, S = 64, 8, 16
    x = torch.randn(N, K)
    ep = EppsPulley(t_max=3.0, n_points=17)
    # generate same A as our test will use
    A = torch.randn(K, S)
    A = A / A.norm(dim=0, keepdim=True)
    # official path: call univariate on x@A
    with torch.no_grad():
        our = SlicedEppsPulley(K, num_slices=S)
    our_val = our(x, A=A)
    # official via direct
    off_val = ep(x @ A).mean()
    assert torch.allclose(our_val, off_val, atol=1e-5)
