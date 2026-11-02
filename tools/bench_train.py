import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

import torch
from lejepa.loss import lejepa_loss
from lejepa.model import LeJEPAEncoder, Projector
from lejepa.sigreg import SlicedEppsPulley


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = LeJEPAEncoder(img_size=32, patch_size=4, embed_dim=64, depth=2, num_heads=2).to(device)
    proj = Projector(512, 128, 16).to(device)
    sig = SlicedEppsPulley(16, 32).to(device)
    params = list(enc.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3)
    V, B = 2, 8
    x = torch.randn(B, 3, 32, 32, device=device)

    # warmup
    for _ in range(3):
        emb = enc(x.repeat(V, 1, 1, 1))
        z = proj(emb).view(V, B, -1)
        loss, _, _ = lejepa_loss(z, sig, 0.02)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    n = 10
    for _ in range(n):
        emb = enc(x.repeat(V, 1, 1, 1))
        z = proj(emb).view(V, B, -1)
        loss, _, _ = lejepa_loss(z, sig, 0.02)
        opt.zero_grad()
        loss.backward()
        opt.step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / n
    imgs = B * V
    print(f"throughput_imgs_per_s={imgs / dt:.1f} step_ms={dt * 1000:.2f}")
    if device == "cuda":
        print(f"peak_mem_mb={torch.cuda.max_memory_allocated() / 1e6:.1f}")


if __name__ == "__main__":
    main()
