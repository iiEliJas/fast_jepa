import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

import torch
from lejepa.loss import lejepa_loss
from lejepa.model import LeJEPAEncoder, Projector
from lejepa.sigreg import SlicedEppsPulley
from torch.optim import AdamW


def test_overfit_tiny():
    torch.manual_seed(0)
    enc = LeJEPAEncoder(img_size=32, patch_size=4, embed_dim=64, depth=2, num_heads=2)
    proj = Projector(512, 128, 16)
    sig = SlicedEppsPulley(16, 32)
    params = list(enc.parameters()) + list(proj.parameters())
    opt = AdamW(params, lr=1e-3)
    x = torch.randn(4, 3, 32, 32)
    losses = []
    for _ in range(5):
        emb = enc(x)
        z = proj(emb).view(1, 4, 16) if False else proj(emb).unsqueeze(0).repeat(2, 1, 1)
        # make two identical views
        z = proj(emb).unsqueeze(0).repeat(2, 1, 1)
        loss, _, _ = lejepa_loss(z, sig, 0.02)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] <= losses[0] + 1e-3
