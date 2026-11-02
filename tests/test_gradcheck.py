import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

import torch
from lejepa.sigreg import SlicedEppsPulley


def test_sigreg_gradcheck():
    torch.manual_seed(0)
    sig = SlicedEppsPulley(4, num_slices=8)
    A = torch.randn(4, 8, dtype=torch.float64)
    A = A / A.norm(dim=0, keepdim=True)

    def f(inp):
        return sig(inp, A=A)

    x = torch.randn(6, 4, dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(f, (x,), eps=1e-6, atol=1e-4)
