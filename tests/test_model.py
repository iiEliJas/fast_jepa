import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

import torch
from lejepa.diagnostics import embedding_diagnostics
from lejepa.loss import lejepa_loss
from lejepa.model import LeJEPAEncoder, Projector
from lejepa.sigreg import SlicedEppsPulley


def test_model_forward():
    enc = LeJEPAEncoder(img_size=32, patch_size=4, embed_dim=64, depth=2, num_heads=2)
    proj = Projector(512, 128, 16)
    x = torch.randn(4, 3, 32, 32)
    e = enc(x)
    z = proj(e)
    assert e.shape == (4, 512)
    assert z.shape == (4, 16)
    # proj_head always outputs 512, so assert the timm overrides directly.
    assert enc.vit.embed_dim == 64
    assert len(enc.vit.blocks) == 2
    assert tuple(enc.vit.patch_embed.patch_size) == (4, 4)


def test_loss_and_diagnostics():
    sig = SlicedEppsPulley(8, 32)
    z = torch.randn(4, 8, 8)
    loss, lp, ls = lejepa_loss(z, sig, 0.02)
    assert torch.isfinite(loss)
    d = embedding_diagnostics(z.view(-1, 8))
    assert "erank" in d
