import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

import torch
from lejepa.sigreg import SlicedEppsPulley


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for N in [256, 512, 1024]:
        sig = SlicedEppsPulley(16, 256).to(device)
        x = torch.randn(N, 16, device=device, requires_grad=True)
        for _ in range(5):
            sig(x).backward()
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(20):
            sig(x).backward()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / 20 * 1000
        print(f"N={N} fwd+bwd_ms={dt:.3f}")


if __name__ == "__main__":
    main()
