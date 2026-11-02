import json
import os

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .diagnostics import embedding_diagnostics
from .loss import lejepa_loss
from .model import LeJEPAEncoder, Projector
from .sigreg import SlicedEppsPulley


def build_transform(img_size=128):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )


class MultiViewDataset(torch.utils.data.Dataset):
    def __init__(self, base, transform, V=4):
        self.base = base
        self.transform = transform
        self.V = V

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        views = [self.transform(img) for _ in range(self.V)]
        return torch.stack(views), label


def to_view_major(views):
    """(B, V, ...) -> (V*B, ...) so that .view(V, B, ...) is view-major.

    The DataLoader stacks views image-major (row r = b*V + v). Flattening
    directly and then calling .view(V, B, K) silently scrambles the view and
    batch axes. Transposing first makes the row order v*B + b.
    """
    if views.dim() < 2:
        raise ValueError(f"expected (B, V, ...), got {tuple(views.shape)}")
    V = views.size(1)
    B = views.size(0)
    return views.transpose(0, 1).reshape(V * B, *views.shape[2:])


def build_model(cfg):
    enc = LeJEPAEncoder(
        img_size=cfg.get("img_size", 128),
        patch_size=cfg.get("patch_size", 8),
        embed_dim=cfg.get("embed_dim", 384),
        depth=cfg.get("depth", 12),
        num_heads=cfg.get("num_heads", 6),
    )
    proj = Projector(
        in_dim=512,
        hidden=cfg.get("proj_hidden", 2048),
        out_dim=cfg.get("proj_dim", 16),
    )
    sig = SlicedEppsPulley(
        in_dim=cfg.get("proj_dim", 16),
        num_slices=cfg.get("num_slices", 256),
    )
    probe = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, cfg.get("num_classes", 10)))
    return enc, proj, sig, probe


def train(cfg, out_dir="runs"):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V = cfg.get("V", 4)
    transform = build_transform(cfg.get("img_size", 128))
    base = datasets.ImageFolder(cfg["data_dir"]) if os.path.isdir(cfg.get("data_dir", "")) else None
    if base is None:
        # fallback to CIFAR via torchvision
        base = datasets.CIFAR10(root=cfg.get("data_root", "./data"), train=True, download=True)
        # wrap to PIL
    ds = MultiViewDataset(base, transform, V=V)
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 64), shuffle=True, num_workers=2)

    enc, proj, sig, probe = build_model(cfg)
    enc.to(device)
    proj.to(device)
    sig.to(device)
    probe.to(device)

    params = list(enc.parameters()) + list(proj.parameters())
    opt = AdamW(params, lr=cfg.get("lr", 2e-3), weight_decay=cfg.get("wd", 5e-2))
    probe_opt = AdamW(probe.parameters(), lr=cfg.get("probe_lr", 1e-3), weight_decay=1e-7)

    warmup = LinearLR(opt, start_factor=0.01, total_iters=1)
    cosine = CosineAnnealingLR(opt, T_max=cfg.get("epochs", 10), eta_min=1e-3)
    sched = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[1])

    lamb = cfg.get("lamb", 0.02)
    log_path = os.path.join(out_dir, "log.jsonl")
    for epoch in range(cfg.get("epochs", 10)):
        enc.train()
        proj.train()
        for views, labels in loader:
            views = views.to(device)  # (B, V, C, H, W)
            B = views.size(0)
            v = to_view_major(views)  # (V*B, C, H, W)
            emb = enc(v)  # (V*B, 512)
            z = proj(emb).view(V, B, -1)
            loss, lp, ls = lejepa_loss(z, sig, lamb)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # online probe on detached embeddings; view-major rows repeat labels
            with torch.no_grad():
                probe_in = emb.detach()
            probe_logits = probe(probe_in)
            probe_labels = labels.repeat(V).to(device)
            probe_loss = nn.CrossEntropyLoss()(probe_logits, probe_labels)
            probe_opt.zero_grad()
            probe_loss.backward()
            probe_opt.step()

        sched.step()
        diag = embedding_diagnostics(z.view(-1, z.size(-1)).detach().cpu())
        with open(log_path, "a") as f:
            f.write(
                json.dumps(
                    {"epoch": epoch, "loss": loss.item(), "lp": lp.item(), "ls": ls.item(), **diag}
                )
                + "\n"
            )
        print(f"epoch {epoch} loss {loss.item():.4f}")

    torch.save(
        {"enc": enc.state_dict(), "proj": proj.state_dict()}, os.path.join(out_dir, "ckpt.pt")
    )
    return enc, proj
