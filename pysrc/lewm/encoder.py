"""Vision Transformer encoder used by LeWorldModel.

This is an explicit reimplementation of the HuggingFace ``ViTModel`` that the
reference LeWM code creates through
``stable_pretraining.backbone.utils.vit_hf("tiny", patch_size=14, ...)``.

Why explicit instead of importing ``transformers``:

- the official LeWM checkpoint was trained with the transformers 4.x ViT
  layout (``encoder.encoder.layer.N.attention.attention.query``); the exact
  module names below let ``load_state_dict(..., strict=True)`` work;
- no ``transformers`` dependency in the ground-truth package;
- the same module map is the specification for the future CUDA encoder.

Reference behaviour (read from transformers 4.x/5.x ``modeling_vit.py``):
CLS token + learned position embedding, pre-LayerNorm blocks, eager attention
with softmax in FP32, exact GELU MLP, final LayerNorm, CLS output. Defaults:
``layer_norm_eps=1e-12``, dropout 0, qkv bias, no pooling layer.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ViTModelOutput:
    """Minimal HuggingFace-style output container."""

    last_hidden_state: torch.Tensor


class ViTPatchEmbeddings(nn.Module):
    def __init__(self, image_size=224, patch_size=14, in_chans=3, embed_dim=192):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.image_size = (image_size, image_size)
        self.patch_size = (patch_size, patch_size)
        self.num_channels = in_chans
        self.projection = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, pixel_values):
        if pixel_values.shape[1] != self.num_channels:
            raise ValueError(
                f"expected {self.num_channels} image channels, "
                f"got {pixel_values.shape[1]}"
            )
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class ViTEmbeddings(nn.Module):
    def __init__(self, image_size=224, patch_size=14, in_chans=3, embed_dim=192):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.patch_embeddings = ViTPatchEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.0)
        self.patch_size = patch_size
        self.image_size = (image_size, image_size)

    def interpolate_pos_encoding(self, embeddings, height, width):
        """Bicubic interpolation, matching HuggingFace ``ViTEmbeddings``."""
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        new_height = height // self.patch_size
        new_width = width // self.patch_size
        sqrt_num_positions = int(math.sqrt(num_positions))
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        ).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        batch_size, _, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(
                embeddings, height, width
            )
        else:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"input size {height}x{width} does not match "
                    f"{self.image_size[0]}x{self.image_size[1]} and "
                    "interpolate_pos_encoding=False"
                )
            embeddings = embeddings + self.position_embeddings

        return self.dropout(embeddings)


class ViTSelfAttention(nn.Module):
    """Eager multi-head self-attention with FP32 softmax (reference path)."""

    def __init__(self, embed_dim=192, num_heads=3, head_dim=64, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = head_dim**-0.5
        inner_dim = num_heads * head_dim
        self.query = nn.Linear(embed_dim, inner_dim, bias=qkv_bias)
        self.key = nn.Linear(embed_dim, inner_dim, bias=qkv_bias)
        self.value = nn.Linear(embed_dim, inner_dim, bias=qkv_bias)

    def forward(self, hidden_states):
        batch, seq, _ = hidden_states.shape
        q = self.query(hidden_states).view(batch, seq, self.num_heads, self.head_dim)
        k = self.key(hidden_states).view(batch, seq, self.num_heads, self.head_dim)
        v = self.value(hidden_states).view(batch, seq, self.num_heads, self.head_dim)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        weights = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        # At least FP32 softmax (the reference always upcasts to FP32). Keep
        # FP64 intact so finite-difference gradchecks are meaningful.
        softmax_dtype = (
            torch.float32
            if weights.dtype in (torch.float16, torch.bfloat16)
            else weights.dtype
        )
        weights = F.softmax(weights, dim=-1, dtype=softmax_dtype).to(q.dtype)
        out = torch.matmul(weights, v).transpose(1, 2).reshape(batch, seq, -1)
        return out


class ViTSelfOutput(nn.Module):
    def __init__(self, embed_dim=192, inner_dim=192):
        super().__init__()
        self.dense = nn.Linear(inner_dim, embed_dim)

    def forward(self, hidden_states):
        return self.dense(hidden_states)


class ViTAttention(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3, head_dim=64):
        super().__init__()
        inner_dim = num_heads * head_dim
        self.attention = ViTSelfAttention(embed_dim, num_heads, head_dim)
        self.output = ViTSelfOutput(embed_dim, inner_dim)

    def forward(self, hidden_states):
        return self.output(self.attention(hidden_states))


class ViTIntermediate(nn.Module):
    def __init__(self, embed_dim=192, mlp_dim=768):
        super().__init__()
        self.dense = nn.Linear(embed_dim, mlp_dim)

    def forward(self, hidden_states):
        return F.gelu(self.dense(hidden_states))


class ViTOutput(nn.Module):
    def __init__(self, embed_dim=192, mlp_dim=768):
        super().__init__()
        self.dense = nn.Linear(mlp_dim, embed_dim)

    def forward(self, hidden_states):
        return self.dense(hidden_states)


class ViTLayer(nn.Module):
    def __init__(
        self,
        embed_dim=192,
        num_heads=3,
        head_dim=64,
        mlp_dim=768,
        layer_norm_eps=1e-12,
    ):
        super().__init__()
        self.attention = ViTAttention(embed_dim, num_heads, head_dim)
        self.layernorm_before = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.intermediate = ViTIntermediate(embed_dim, mlp_dim)
        self.output = ViTOutput(embed_dim, mlp_dim)
        self.dropout = nn.Dropout(0.0)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = self.dropout(hidden_states) + residual

        residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.output(hidden_states)
        hidden_states = self.dropout(hidden_states) + residual
        return hidden_states


class ViTEncoderLayers(nn.Module):
    """Container with the 4.x name ``layer`` so state-dict keys match."""

    def __init__(self, depth=12, **kwargs):
        super().__init__()
        self.layer = nn.ModuleList([ViTLayer(**kwargs) for _ in range(depth)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states):
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return hidden_states


class ViTEncoder(nn.Module):
    """ViT-tiny/patch14 encoder with a CLS-token output."""

    def __init__(
        self,
        image_size=224,
        patch_size=14,
        in_chans=3,
        embed_dim=192,
        depth=12,
        num_heads=3,
        head_dim=64,
        mlp_dim=768,
        layer_norm_eps=1e-12,
        initializer_range=0.02,
    ):
        super().__init__()
        self.embeddings = ViTEmbeddings(
            image_size=image_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.encoder = ViTEncoderLayers(
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            mlp_dim=mlp_dim,
            layer_norm_eps=layer_norm_eps,
        )
        self.layernorm = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.initializer_range = initializer_range
        self._init_weights()

    def _init_weights(self):
        """Match HuggingFace ViT initialization for from-scratch runs."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                module.weight.data.normal_(mean=0.0, std=self.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()
        nn.init.trunc_normal_(
            self.embeddings.cls_token, mean=0.0, std=self.initializer_range
        )
        nn.init.trunc_normal_(
            self.embeddings.position_embeddings,
            mean=0.0,
            std=self.initializer_range,
        )

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        embeddings = self.embeddings(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )
        hidden_states = self.encoder(embeddings)
        hidden_states = self.layernorm(hidden_states)
        return ViTModelOutput(last_hidden_state=hidden_states)
