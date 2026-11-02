import timm
from torch import nn


class LeJEPAEncoder(nn.Module):
    def __init__(
        self,
        img_size=128,
        patch_size=8,
        embed_dim=384,
        depth=12,
        num_heads=6,
        drop_path_rate=0.1,
        dynamic_img_size=True,
    ):
        super().__init__()
        self.vit = timm.create_model(
            "vit_small_patch8_224",
            pretrained=False,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_classes=0,
            drop_path_rate=drop_path_rate,
            dynamic_img_size=dynamic_img_size,
        )
        self.proj_head = nn.Linear(embed_dim, 512)

    def forward(self, x):
        feat = self.vit(x)
        return self.proj_head(feat)


class Projector(nn.Module):
    def __init__(self, in_dim=512, hidden=2048, out_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)
