"""LeWorldModel modules.

Ported from the official LeWM implementation (MIT, commit ``8edfeb3``):
``module.py`` in ``~/projects/le-wm``. Parameter names and the numerics are
kept identical so the released 72 MB checkpoint loads with ``strict=True`` and
the parity tests can compare every tensor.

Two reference quirks are intentional and must not be "cleaned up":

1. The action-conditioned block applies a non-affine LayerNorm before the
   AdaLN modulation, and the attention/FFN submodules each start with their
   own affine LayerNorm. Activations are therefore normalized twice.
2. The projector order is ``Linear -> BatchNorm -> GELU -> Linear``, not the
   LeJEPA ``Linear -> BN -> ReLU`` recipe.
"""

import torch
import torch.nn.functional as F
from lejepa.sigreg import SlicedEppsPulley
from torch import nn


class SIGReg(SlicedEppsPulley):
    """LeWM's SIGReg: 1024 slices, 17 knots on ``[0, 3]``.

    The official class lives outside the JEPA ``state_dict``. This subclass is
    the same statistic as ``pysrc/lejepa/sigreg.py``; for LeWM the input is
    ``(T, B, D)`` and the statistic is averaged over time and slices.
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__(in_dim=None, num_slices=num_proj, t_max=3.0, n_knots=knots)

    @property
    def num_proj(self):
        return self.num_slices


def modulate(x, shift, scale):
    """AdaLN-zero modulation."""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Causal multi-head attention with an internal affine LayerNorm."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (
            t.reshape(*t.shape[:-1], self.heads, -1).transpose(-3, -2) for t in qkv
        )
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=drop, is_causal=causal
        )
        out = out.transpose(-3, -2)
        return self.to_out(out.reshape(*out.shape[:-2], -1))


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero action conditioning."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard pre-LN transformer block (used by the reference Transformer)."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [block_class(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

    def forward(self, x, c=None):
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)
        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)
        return self.output_proj(x)


class Embedder(nn.Module):
    """Action encoder: Conv1d over the feature axis, then a 4x MLP."""

    def __init__(self, input_dim=10, smoothed_dim=10, emb_dim=10, mlp_scale=4):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        # Reference casts to float32 for bf16 inputs; keep float64 for
        # finite-difference gradchecks.
        if x.dtype != torch.float64:
            x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        return self.embed(x)


class MLP(nn.Module):
    """``Linear -> norm -> activation -> Linear`` (LeWM projector)."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive action-conditioned embedding predictor."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        return self.transformer(x, c)
