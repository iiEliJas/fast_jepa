"""LeWorldModel: the joint-embedding predictive world model (LeWM).

Reference: ``jepa.py`` in the official LeWM repository (MIT), commit
``8edfeb336732b5f3ce7b8b210d0ba370a09e2cac``. The public API is kept close to
that file (``encode``/``predict``/``rollout``/``criterion``/``get_cost``) so the
same tests can drive both implementations.

Shapes used throughout:

- ``pixels``: ``(B, T, C, H, W)``, uint8/float in ``[0, 1]`` style normalization
- ``action``: ``(B, T, frameskip * action_dim)``, i.e. one row per frame
- ``emb``:    ``(B, T, embed_dim)``
- ``act_emb``:``(B, T, embed_dim)``

Training uses ``T = history_size + num_preds`` (4 for the released config) and
teacher-forced one-step prediction at every context position. Planning uses a
separate rollout path that starts from ``H`` context frames and predicts the
future autoregressively.
"""

from dataclasses import dataclass

import torch
from torch import nn

from .encoder import ViTEncoder
from .modules import MLP, ARPredictor, Embedder, SIGReg


@dataclass
class LewmConfig:
    """Hyperparameters of the released Push-T LeWM model."""

    # Encoder (HuggingFace ViT-tiny/patch14)
    image_size: int = 224
    patch_size: int = 14
    in_chans: int = 3
    embed_dim: int = 192
    encoder_depth: int = 12
    encoder_heads: int = 3
    encoder_head_dim: int = 64
    encoder_mlp_dim: int = 768
    layer_norm_eps: float = 1e-12

    # Predictor
    predictor_depth: int = 6
    predictor_heads: int = 16
    predictor_head_dim: int = 64
    predictor_mlp_dim: int = 2048
    predictor_dropout: float = 0.1
    predictor_emb_dropout: float = 0.0

    # Projectors / action encoder
    projector_hidden_dim: int = 2048
    action_smoothed_dim: int | None = None  # None -> frameskip * action_dim

    # Data layout
    history_size: int = 3
    num_preds: int = 1
    action_dim: int = 2
    frameskip: int = 5

    # SIGReg / loss
    sigreg_knots: int = 17
    sigreg_num_proj: int = 1024
    sigreg_weight: float = 0.09

    @property
    def num_steps(self) -> int:
        return self.history_size + self.num_preds

    @property
    def action_block_dim(self) -> int:
        return self.frameskip * self.action_dim


class JEPA(nn.Module):
    """LeWM world model.

    Module layout and parameter names are identical to the official model, so
    checkpoints trained with the reference code load with ``strict=True``.
    """

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
        num_preds=1,
    ):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.num_preds = int(num_preds)

    # ------------------------------------------------------------------
    # Core forward pieces
    # ------------------------------------------------------------------
    def encode(self, info):
        """Encode observations and actions.

        ``info`` must contain ``pixels`` of shape ``(B, T, C, H, W)``; if it
        also contains ``action``, the action embedding is added as
        ``info["act_emb"]``. Returns a new dict; the input is not mutated.
        """
        info = dict(info)
        pixels = info["pixels"]
        if pixels.ndim != 5:
            raise ValueError(f"expected pixels of shape (B, T, C, H, W), got {tuple(pixels.shape)}")
        # The reference casts to float32 to support bf16 data. Keep float64
        # untouched so finite-difference gradchecks exercise the full model.
        if pixels.dtype != torch.float64:
            pixels = pixels.float()
        b = pixels.size(0)
        pixels = pixels.reshape(b * pixels.size(1), *pixels.shape[2:])
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # CLS token
        emb = self.projector(pixels_emb)
        info["emb"] = emb.reshape(b, -1, emb.size(-1))

        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])

        return info

    def predict(self, emb, act_emb):
        """Predict the next embedding at every position.

        ``emb``/``act_emb``: ``(B, T, D)`` -> ``(B, T, D)``. Position ``t``
        predicts the embedding after the action at position ``t``.
        """
        preds = self.predictor(emb, act_emb)
        b = emb.size(0)
        preds = self.pred_proj(preds.reshape(b * preds.size(1), preds.size(-1)))
        return preds.reshape(b, -1, preds.size(-1))

    def forward(self, info):
        """Convenience alias used by the trainer: encode + predict."""
        info = self.encode(info)
        if "act_emb" not in info:
            return info
        ctx = info["emb"][:, : self.predictor.num_frames]
        act = info["act_emb"][:, : self.predictor.num_frames]
        info["pred_emb"] = self.predict(ctx, act)
        return info

    # ------------------------------------------------------------------
    # Training objective
    # ------------------------------------------------------------------
    def lewm_loss(self, info, sigreg=None, weight=None):
        """LeWM training loss.

        ``pred_loss`` is the teacher-forced next-embedding MSE at every context
        position; ``sigreg_loss`` is SIGReg over the batch at each time step,
        averaged over time. No stop-gradient and no EMA.

        ``sigreg`` must be an instance of :class:`lewm.modules.SIGReg` (or any
        callable with the same ``(T, B, D)`` contract). It is intentionally not
        a submodule, so it stays out of the model ``state_dict`` exactly like
        the reference implementation.
        """
        out = self.encode(info)
        emb = out["emb"]
        act_emb = out["act_emb"]
        ctx_len = self.predictor.num_frames
        n_preds = self.num_preds
        expected_steps = ctx_len + n_preds
        if emb.size(1) != expected_steps:
            raise ValueError(
                f"expected {expected_steps} frames "
                f"(history {ctx_len} + num_preds {n_preds}), got {emb.size(1)}"
            )

        ctx_emb = emb[:, :ctx_len]
        ctx_act = act_emb[:, :ctx_len]
        tgt_emb = emb[:, n_preds:]
        pred_emb = self.predict(ctx_emb, ctx_act)

        pred_loss = (pred_emb - tgt_emb).pow(2).mean()
        out["pred_emb"] = pred_emb

        if sigreg is None:
            out["pred_loss"] = pred_loss
            out["loss"] = pred_loss
            return out

        if weight is None:
            weight = 0.09
        sigreg_loss = sigreg(emb.transpose(0, 1))
        out["pred_loss"] = pred_loss
        out["sigreg_loss"] = sigreg_loss
        out["loss"] = pred_loss + weight * sigreg_loss
        return out

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def rollout(self, info, action_sequence, history_size=None):
        """Roll out candidate action sequences from context frames.

        ``info["pixels"]``: ``(B, S, H, C, H_img, W_img)``.
        ``action_sequence``: ``(B, S, T, action_dim)`` blocks. As in the
        reference, its first ``H`` rows fill the action history and the
        remaining ``T - H`` rows are fed one per rollout step. In the released
        evaluation ``H = 1``, so every row is a future action.

        Returns a copy of ``info`` with ``predicted_emb`` of shape
        ``(B, S, T + 1, D)``. The last entry is the terminal state used by the
        cost.
        """
        if history_size is None:
            history_size = getattr(self.predictor, "num_frames", 3)

        if "pixels" not in info:
            raise KeyError("pixels not in info_dict")
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)

        info = dict(info)
        info["action"] = act_0
        n_steps = T - H

        # Encode the context; one frame per batch item is expected here.
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        emb = emb.reshape(B * S, *_init["emb"].shape[1:]).clone()

        act = act_0.reshape(B * S, *act_0.shape[2:])
        act_future = act_future.reshape(B * S, *act_future.shape[2:])

        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            pred_emb = self.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
            emb = torch.cat([emb, pred_emb], dim=1)

            next_act = act_future[:, t : t + 1, :]
            act = torch.cat([act, next_act], dim=1)

        # Predict the terminal state after the last future action.
        act_emb = self.action_encoder(act)
        pred_emb = self.predict(emb[:, -HS:], act_emb[:, -HS:])[:, -1:]
        emb = torch.cat([emb, pred_emb], dim=1)

        info["predicted_emb"] = emb.reshape(B, S, *emb.shape[1:])
        return info

    def criterion(self, info):
        """Terminal cost per candidate.

        Squared L2 distance summed over the embedding dimension between the
        last predicted embedding and the last goal embedding. Accepts a goal
        embedding of shape ``(B, D)``, ``(B, T, D)`` or ``(B, S, T, D)``.
        """
        pred_emb = info["predicted_emb"]
        goal_emb = info["goal_emb"]
        pred_last = pred_emb[..., -1, :]
        if goal_emb.ndim == 2:
            goal_last = goal_emb  # (B, D)
        elif goal_emb.ndim == 3:
            goal_last = goal_emb[:, -1]  # (B, T, D) -> (B, D)
        else:
            goal_last = goal_emb[..., -1, :]  # (B, S, T, D) -> (B, S, D)
        if goal_last.ndim == 2 and pred_last.ndim == 3:
            goal_last = goal_last.unsqueeze(1)  # broadcast over candidates
        return (pred_last - goal_last.detach()).pow(2).sum(dim=-1)

    def get_cost(self, info_dict, action_candidates):
        """Cost of candidate action sequences, shape ``(B, S)``.

        Mirrors the reference ``JEPA.get_cost``: the goal image is encoded once
        per call and the candidate latents are compared at the terminal step.
        """
        if "goal" not in info_dict:
            raise KeyError("goal not in info_dict")

        device = next(self.parameters()).device
        info_dict = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in info_dict.items()}

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]
        for key in list(goal.keys()):
            if key.startswith("goal_"):
                goal[key[len("goal_") :]] = goal.pop(key)
        goal.pop("action", None)
        goal.pop("action_history", None)
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)
        return self.criterion(info_dict)


def build_lewm(cfg: LewmConfig | None = None, **overrides) -> JEPA:
    """Build the reference LeWM architecture.

    ``overrides`` are applied to a copy of ``cfg`` (or to the default config).
    """
    cfg = cfg or LewmConfig()
    if overrides:
        from dataclasses import replace

        cfg = replace(cfg, **overrides)

    encoder = ViTEncoder(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        in_chans=cfg.in_chans,
        embed_dim=cfg.embed_dim,
        depth=cfg.encoder_depth,
        num_heads=cfg.encoder_heads,
        head_dim=cfg.encoder_head_dim,
        mlp_dim=cfg.encoder_mlp_dim,
        layer_norm_eps=cfg.layer_norm_eps,
    )
    predictor = ARPredictor(
        num_frames=cfg.history_size,
        depth=cfg.predictor_depth,
        heads=cfg.predictor_heads,
        mlp_dim=cfg.predictor_mlp_dim,
        input_dim=cfg.embed_dim,
        hidden_dim=cfg.embed_dim,
        output_dim=cfg.embed_dim,
        dim_head=cfg.predictor_head_dim,
        dropout=cfg.predictor_dropout,
        emb_dropout=cfg.predictor_emb_dropout,
    )
    smoothed = cfg.action_smoothed_dim or cfg.action_block_dim
    action_encoder = Embedder(
        input_dim=cfg.action_block_dim,
        smoothed_dim=smoothed,
        emb_dim=cfg.embed_dim,
    )
    projector = MLP(
        input_dim=cfg.embed_dim,
        hidden_dim=cfg.projector_hidden_dim,
        output_dim=cfg.embed_dim,
        norm_fn=nn.BatchNorm1d,
        act_fn=nn.GELU,
    )
    pred_proj = MLP(
        input_dim=cfg.embed_dim,
        hidden_dim=cfg.projector_hidden_dim,
        output_dim=cfg.embed_dim,
        norm_fn=nn.BatchNorm1d,
        act_fn=nn.GELU,
    )
    return JEPA(
        encoder,
        predictor,
        action_encoder,
        projector,
        pred_proj,
        num_preds=cfg.num_preds,
    )


def build_sigreg(cfg: LewmConfig | None = None) -> SIGReg:
    cfg = cfg or LewmConfig()
    return SIGReg(knots=cfg.sigreg_knots, num_proj=cfg.sigreg_num_proj)
