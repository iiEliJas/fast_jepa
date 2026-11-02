import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

import torch
from lejepa.loss import lejepa_loss
from lejepa.model import LeJEPAEncoder, Projector
from lejepa.sigreg import SlicedEppsPulley
from lejepa.train import to_view_major
from torch.optim import AdamW


def test_overfit_tiny_multiview():
    torch.manual_seed(0)
    enc = LeJEPAEncoder(
        img_size=32,
        patch_size=4,
        embed_dim=32,
        depth=1,
        num_heads=2,
        drop_path_rate=0.0,
    )
    proj = Projector(512, 64, 16)
    sig = SlicedEppsPulley(16, 32)
    opt = AdamW(list(enc.parameters()) + list(proj.parameters()), lr=3e-3)

    B, V = 4, 4
    base = torch.randn(B, 3, 32, 32)
    losses, preds = [], []
    for _ in range(30):
        views = base.unsqueeze(1).repeat(1, V, 1, 1, 1)
        views = views + 0.05 * torch.randn_like(views)
        v = to_view_major(views)
        emb = enc(v)
        z = proj(emb).view(V, B, 16)
        loss, lp, _ = lejepa_loss(z, sig, 0.02)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        preds.append(lp.item())

    assert torch.isfinite(torch.tensor(losses)).all()
    assert losses[-1] < losses[0]
    assert preds[-1] < preds[0]
