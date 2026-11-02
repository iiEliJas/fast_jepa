"""Parity against the pinned official LeJEPA implementation.

The official package is imported in a *subprocess* because this repository also
has a top-level ``lejepa`` package. Importing both in one interpreter is
ambiguous: whichever is imported first wins ``sys.modules["lejepa"]``. Running
the official code in a clean process with ``PYTHONPATH`` pointing at the pinned
checkout avoids that and keeps the test honest.

Pinned checkout: github.com/rbalestr-lab/lejepa @
c293d291ca87cd4fddee9d3fffe4e914c7272052 (override with LEJEPA_OFFICIAL_ROOT).
"""

import os
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

from lejepa.sigreg import SlicedEppsPulley  # noqa: E402

OFFICIAL_ROOT = os.path.abspath(
    os.environ.get("LEJEPA_OFFICIAL_ROOT", os.path.expanduser("~/projects/lejepa"))
)
OFFICIAL_COMMIT = "c293d291ca87cd4fddee9d3fffe4e914c7272052"

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(OFFICIAL_ROOT, "lejepa")),
    reason=f"official lejepa not found at {OFFICIAL_ROOT}",
)

_OFFICIAL_SCRIPT = r"""
import sys
import torch

root, in_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, root)
from lejepa.univariate import EppsPulley

data = torch.load(in_path, weights_only=False)
ep = EppsPulley(t_max=data["t_max"], n_points=data["n_points"])
if data.get("fp64", False):
    ep.t = ep.t.double()
    ep.weights = ep.weights.double()
    ep.phi = ep.phi.double()

x = data["x"].to(data["dtype"])
A = data["A"].to(data["dtype"])
x = x.requires_grad_(data.get("grad", False))

if data.get("views", False):
    per_view = torch.stack([ep(x[v] @ A) for v in range(x.shape[0])])
    per_slice = per_view.mean(dim=0)
else:
    per_view = None
    per_slice = ep(x @ A)

result = {"per_slice": per_slice.detach().cpu()}
if per_view is not None:
    result["per_view"] = per_view.detach().cpu()
if data.get("grad", False):
    per_slice.mean().backward()
    result["grad"] = x.grad.detach().cpu()
torch.save(result, out_path)
"""


def _run_official(payload, tmp_path):
    in_path = tmp_path / "in.pt"
    out_path = tmp_path / "out.pt"
    script = tmp_path / "official_parity.py"
    torch.save(payload, in_path)
    script.write_text(_OFFICIAL_SCRIPT)

    env = dict(os.environ)
    env["PYTHONPATH"] = OFFICIAL_ROOT
    subprocess.run(
        [sys.executable, str(script), OFFICIAL_ROOT, str(in_path), str(out_path)],
        check=True,
        cwd=str(tmp_path),
        env=env,
        capture_output=True,
        text=True,
    )
    return torch.load(out_path, weights_only=False)


def _normalized(directions):
    return directions / directions.norm(dim=0, keepdim=True)


@pytest.mark.parametrize("t_max,n_points", [(3.0, 17), (5.0, 9)])
def test_forward_per_slice_matches_official(tmp_path, t_max, n_points):
    torch.manual_seed(0)
    N, K, S = 64, 8, 16
    x = torch.randn(N, K)
    A = _normalized(torch.randn(K, S))

    out = _run_official(
        {"x": x, "A": A, "t_max": t_max, "n_points": n_points, "dtype": torch.float32},
        tmp_path,
    )
    ours = SlicedEppsPulley(K, num_slices=S, t_max=t_max, n_knots=n_points)(
        x, A=A, return_per_slice=True
    )
    assert ours.shape == (S,)
    assert torch.allclose(ours, out["per_slice"], atol=1e-5, rtol=1e-5)


def test_3d_shared_directions_match_official(tmp_path):
    torch.manual_seed(0)
    V, N, K, S = 3, 16, 8, 8
    z = torch.randn(V, N, K)
    A = _normalized(torch.randn(K, S))

    out = _run_official(
        {
            "x": z,
            "A": A,
            "t_max": 3.0,
            "n_points": 17,
            "dtype": torch.float32,
            "views": True,
        },
        tmp_path,
    )
    sig = SlicedEppsPulley(K, num_slices=S)
    ours = sig(z, A=A, return_per_slice=True)
    assert torch.allclose(ours, out["per_slice"], atol=1e-5, rtol=1e-5)

    # Same A for every view: the 3D call equals the mean of the 2D calls.
    per_view = torch.stack([sig(z[v], A=A) for v in range(V)])
    assert torch.allclose(sig(z, A=A), per_view.mean(), atol=1e-6)


def test_gradient_matches_official_fp64(tmp_path):
    torch.manual_seed(0)
    N, K, S = 32, 4, 8
    x = torch.randn(N, K, dtype=torch.float64)
    A = _normalized(torch.randn(K, S, dtype=torch.float64))

    out = _run_official(
        {
            "x": x,
            "A": A,
            "t_max": 3.0,
            "n_points": 17,
            "dtype": torch.float64,
            "fp64": True,
            "grad": True,
        },
        tmp_path,
    )
    ours_x = x.detach().clone().requires_grad_(True)
    SlicedEppsPulley(K, num_slices=S)(ours_x, A=A).backward()

    assert torch.allclose(ours_x.grad, out["grad"], atol=1e-10, rtol=1e-8)
