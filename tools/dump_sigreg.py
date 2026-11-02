import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

import torch
from lejepa.sigreg import SlicedEppsPulley


def main():
    torch.manual_seed(0)
    sig = SlicedEppsPulley(8, 16)
    x = torch.randn(16, 8)
    A = torch.randn(8, 16)
    A = A / A.norm(dim=0, keepdim=True)
    val = sig(x, A=A)
    out_dir = os.path.join(os.path.dirname(__file__), "..", "bench")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"x": x, "A": A, "val": val}, os.path.join(out_dir, "sigreg_ref.pt"))
    print("dumped", val.item())


if __name__ == "__main__":
    main()
