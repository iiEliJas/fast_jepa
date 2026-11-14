"""Full-model parity for LeWM: checkpoint load, forward, loss, rollout, grads.

Reference: pinned ``~/projects/le-wm`` (MIT) at ``8edfeb3``. The official
encoder is the HuggingFace ``ViTModel``; its module names changed in
transformers 5, so the checkpoint keys are translated in this test. Our own
encoder keeps the 4.x names that the checkpoint was trained with.

The released 72 MB Push-T checkpoint is used when it is in the HuggingFace
cache (or ``LEWM_PUSH_T_WEIGHTS``). Everything else runs with random weights.
"""

import glob
import importlib.util
import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

from lewm import LewmConfig, build_lewm, build_sigreg  # noqa: E402
from lewm.modules import SIGReg  # noqa: E402

OFFICIAL_ROOT = os.path.abspath(
    os.environ.get("LEWM_OFFICIAL_ROOT", os.path.expanduser("~/projects/le-wm"))
)
CHECKPOINT = os.environ.get("LEWM_PUSH_T_WEIGHTS")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(OFFICIAL_ROOT),
    reason=f"official le-wm not found at {OFFICIAL_ROOT}",
)

CFG = LewmConfig()
SMALL_IMAGE = 28  # 2x2 patches with patch_size=14


def _load_by_path(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(OFFICIAL_ROOT, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def official():
    module = _load_by_path("official_lewm_module", "module.py")
    jepa = _load_by_path("official_lewm_jepa", "jepa.py")
    try:
        from transformers import ViTConfig, ViTModel
    except ImportError:  # pragma: no cover - dev dependency
        pytest.skip("transformers is required for the official encoder")
    return module, jepa, ViTConfig, ViTModel


def _checkpoint_path():
    if CHECKPOINT:
        return CHECKPOINT if os.path.isfile(CHECKPOINT) else None
    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--quentinll--lewm-pusht/snapshots/*/weights.pt"
    )
    paths = sorted(glob.glob(pattern))
    return paths[-1] if paths else None


def _to_transformers_5(key):
    key = key.replace(".attention.attention.query.", ".attention.q_proj.")
    key = key.replace(".attention.attention.key.", ".attention.k_proj.")
    key = key.replace(".attention.attention.value.", ".attention.v_proj.")
    key = key.replace(".attention.output.dense.", ".attention.o_proj.")
    key = key.replace(".intermediate.dense.", ".mlp.fc1.")
    key = key.replace(".output.dense.", ".mlp.fc2.")
    return key.replace("encoder.layer.", "layers.")


class _OfficialEncoder(nn.Module):
    def __init__(self, ViTConfig, ViTModel):
        super().__init__()
        cfg = ViTConfig(
            hidden_size=CFG.embed_dim,
            num_hidden_layers=CFG.encoder_depth,
            num_attention_heads=CFG.encoder_heads,
            intermediate_size=CFG.encoder_mlp_dim,
            image_size=CFG.image_size,
            patch_size=CFG.patch_size,
            num_channels=CFG.in_chans,
            qkv_bias=True,
            layer_norm_eps=CFG.layer_norm_eps,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        self.vit = ViTModel(cfg, add_pooling_layer=False)

    def forward(self, pixel_values, interpolate_pos_encoding=False):
        return self.vit(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)


def _load_into_official(ref, state_dict):
    """Load a LeWM state dict into the reference model via 5.x key names."""
    translated = {}
    for key, value in state_dict.items():
        if key.startswith("encoder."):
            key = "encoder.vit." + _to_transformers_5(key[len("encoder.") :])
        translated[key] = value
    ref.load_state_dict(translated, strict=True)


def build_official(official, state_dict=None):
    module, jepa, ViTConfig, ViTModel = official
    encoder = _OfficialEncoder(ViTConfig, ViTModel)
    model = jepa.JEPA(
        encoder=encoder,
        predictor=module.ARPredictor(
            num_frames=CFG.history_size,
            depth=CFG.predictor_depth,
            heads=CFG.predictor_heads,
            mlp_dim=CFG.predictor_mlp_dim,
            input_dim=CFG.embed_dim,
            hidden_dim=CFG.embed_dim,
            output_dim=CFG.embed_dim,
            dim_head=CFG.predictor_head_dim,
            dropout=CFG.predictor_dropout,
            emb_dropout=CFG.predictor_emb_dropout,
        ),
        action_encoder=module.Embedder(input_dim=CFG.action_block_dim, emb_dim=CFG.embed_dim),
        projector=module.MLP(
            input_dim=CFG.embed_dim,
            hidden_dim=CFG.projector_hidden_dim,
            output_dim=CFG.embed_dim,
            norm_fn=nn.BatchNorm1d,
        ),
        pred_proj=module.MLP(
            input_dim=CFG.embed_dim,
            hidden_dim=CFG.projector_hidden_dim,
            output_dim=CFG.embed_dim,
            norm_fn=nn.BatchNorm1d,
        ),
    )
    if state_dict is not None:
        _load_into_official(model, state_dict)
    return model


def _batch(batch=2, image=SMALL_IMAGE, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "pixels": torch.rand(batch, CFG.num_steps, 3, image, image, generator=g),
        "action": torch.randn(batch, CFG.num_steps, CFG.action_block_dim, generator=g),
    }


def _build_pair(official):
    """Ours and the reference with identical weights."""
    torch.manual_seed(0)
    ours = build_lewm().eval()
    ref = build_official(official).eval()
    _load_into_official(ref, ours.state_dict())
    return ours, ref


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def test_official_checkpoint_loads_strict():
    path = _checkpoint_path()
    if path is None:
        pytest.skip("official PushT checkpoint not cached")

    model = build_lewm()
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    assert len(model.state_dict()) == 303
    assert sum(p.numel() for p in model.parameters()) == 18_034_478


def test_checkpoint_forward_parity(official):
    """The released Push-T weights must reproduce the reference forward pass."""
    path = _checkpoint_path()
    if path is None:
        pytest.skip("official PushT checkpoint not cached")

    state_dict = torch.load(path, map_location="cpu")
    ours = build_lewm().eval()
    ours.load_state_dict(state_dict, strict=True)
    ref = build_official(official).eval()
    _load_into_official(ref, state_dict)

    batch = _batch()
    with torch.no_grad():
        ours_out = ours.encode(batch)
        ref_out = ref.encode(dict(batch))
        torch.testing.assert_close(ours_out["emb"], ref_out["emb"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            ours.predict(ours_out["emb"][:, :3], ours_out["act_emb"][:, :3]),
            ref.predict(ref_out["emb"][:, :3], ref_out["act_emb"][:, :3]),
            atol=1e-5,
            rtol=1e-5,
        )

        sigreg = build_sigreg()
        torch.manual_seed(3)
        ours_loss = ours.lewm_loss(batch, sigreg, CFG.sigreg_weight)

        emb = ref_out["emb"]
        pred = ref.predict(emb[:, :3], ref_out["act_emb"][:, :3])
        pred_loss = (pred - emb[:, 1:]).pow(2).mean()
        torch.manual_seed(3)
        sigreg_loss = official[0].SIGReg()(emb.transpose(0, 1))
        torch.testing.assert_close(ours_loss["pred_loss"], pred_loss, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(ours_loss["sigreg_loss"], sigreg_loss, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Forward / loss parity
# ---------------------------------------------------------------------------
def test_encode_predict_parity(official):
    ours, ref = _build_pair(official)

    batch = _batch()
    with torch.no_grad():
        ours_out = ours.encode(batch)
        ref_out = ref.encode(dict(batch))
        torch.testing.assert_close(ours_out["emb"], ref_out["emb"], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ours_out["act_emb"], ref_out["act_emb"], atol=0, rtol=0)

        ctx = CFG.history_size
        torch.testing.assert_close(
            ours.predict(ours_out["emb"][:, :ctx], ours_out["act_emb"][:, :ctx]),
            ref.predict(ref_out["emb"][:, :ctx], ref_out["act_emb"][:, :ctx]),
            atol=1e-5,
            rtol=1e-5,
        )


def test_loss_parity(official):
    module = official[0]
    ours, ref = _build_pair(official)

    batch = _batch()
    sigreg = build_sigreg()
    seed = 123

    torch.manual_seed(seed)
    out = ours.lewm_loss(batch, sigreg, CFG.sigreg_weight)

    ref_out = ref.encode(dict(batch))
    emb = ref_out["emb"]
    ctx = CFG.history_size
    pred = ref.predict(emb[:, :ctx], ref_out["act_emb"][:, :ctx])
    pred_loss = (pred - emb[:, CFG.num_preds :]).pow(2).mean()
    torch.manual_seed(seed)
    sigreg_loss = module.SIGReg(knots=17, num_proj=1024)(emb.transpose(0, 1))
    loss = pred_loss + CFG.sigreg_weight * sigreg_loss

    torch.testing.assert_close(out["pred_loss"], pred_loss, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(out["sigreg_loss"], sigreg_loss, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out["loss"], loss, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Rollout and cost parity
# ---------------------------------------------------------------------------
def test_rollout_and_cost_parity(official):
    ours, ref = _build_pair(official)

    B, S, H, horizon = 2, 3, 1, 5
    g = torch.Generator().manual_seed(4)
    pixels = torch.rand(B, S, H, 3, SMALL_IMAGE, SMALL_IMAGE, generator=g)
    goal = torch.rand(B, S, H, 3, SMALL_IMAGE, SMALL_IMAGE, generator=g)
    actions = torch.randn(B, S, horizon, CFG.action_block_dim, generator=g)
    info = {"pixels": pixels, "goal": goal, "action": torch.zeros(B, S, H, CFG.action_block_dim)}

    with torch.no_grad():
        ours_cost = ours.get_cost(dict(info), actions)
        our_roll = ours.rollout(dict(info), actions)

        # Reference path: same get_cost steps, but keep the goal embedding
        # around so the terminal cost can be computed without the pinned
        # criterion's shape assumption.
        goal_info = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        goal_info["pixels"] = goal_info["goal"]
        goal_info.pop("action", None)
        goal_emb = ref.encode(goal_info)["emb"]

        ref_info = dict(info)
        ref_info["goal_emb"] = goal_emb
        ref_roll = ref.rollout(ref_info, actions)
        ref_cost = (
            (ref_roll["predicted_emb"][..., -1, :] - goal_emb[:, -1].unsqueeze(1)).pow(2).sum(-1)
        )

    torch.testing.assert_close(
        our_roll["predicted_emb"],
        ref_roll["predicted_emb"],
        atol=1e-5,
        rtol=1e-5,
    )
    torch.testing.assert_close(ours_cost, ref_cost, atol=1e-4, rtol=1e-5)
    assert ours_cost.shape == (B, S)


# ---------------------------------------------------------------------------
# Finite-difference gradcheck in FP64
# ---------------------------------------------------------------------------
def test_gradcheck_fp64():
    cfg = LewmConfig(
        image_size=SMALL_IMAGE,
        patch_size=14,
        embed_dim=32,
        encoder_depth=1,
        encoder_heads=2,
        encoder_head_dim=16,
        encoder_mlp_dim=64,
        predictor_depth=1,
        predictor_heads=2,
        predictor_head_dim=16,
        predictor_mlp_dim=64,
        predictor_dropout=0.0,
        projector_hidden_dim=64,
        action_smoothed_dim=None,
        history_size=3,
        num_preds=1,
        action_dim=2,
        frameskip=2,
        sigreg_knots=5,
        sigreg_num_proj=4,
    )
    model = build_lewm(cfg).double().eval()

    # AdaLN-zero starts with zero modulation, which makes the whole action
    # path contribute zero gradient. Perturb it so the action encoder is
    # actually exercised by the gradcheck.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("adaLN_modulation.1.weight"):
                param.normal_(0.0, 0.02)

    class FixedSIGReg(nn.Module):
        def __init__(self):
            super().__init__()
            self.sigreg = SIGReg(knots=cfg.sigreg_knots, num_proj=cfg.sigreg_num_proj)
            g = torch.Generator().manual_seed(0)
            A = torch.randn(cfg.embed_dim, cfg.sigreg_num_proj, generator=g, dtype=torch.float64)
            self.register_buffer("A", A / A.norm(dim=0, keepdim=True))

        def forward(self, x):
            return self.sigreg(x, A=self.A)

    sigreg = FixedSIGReg()
    g = torch.Generator().manual_seed(1)
    batch = {
        "pixels": torch.rand(
            1, cfg.num_steps, 3, SMALL_IMAGE, SMALL_IMAGE, generator=g, dtype=torch.float64
        ),
        "action": torch.randn(
            1, cfg.num_steps, cfg.action_block_dim, generator=g, dtype=torch.float64
        ),
    }

    model.zero_grad(set_to_none=True)
    out = model.lewm_loss(batch, sigreg, 0.09)
    out["loss"].backward()

    named = dict(model.named_parameters())
    probes = [
        ("encoder.embeddings.patch_embeddings.projection.weight", (0, 0, 0, 0)),
        ("encoder.encoder.layer.0.attention.attention.query.weight", (0, 0)),
        ("encoder.layernorm.weight", (0,)),
        ("predictor.pos_embedding", (0, 0, 0)),
        ("predictor.transformer.layers.0.attn.to_qkv.weight", (0, 0)),
        ("predictor.transformer.layers.0.adaLN_modulation.1.weight", (0, 0)),
        ("projector.net.3.weight", (0, 0)),
        ("action_encoder.embed.0.weight", (0, 0)),
    ]

    eps = 1e-5
    for name, index in probes:
        param = named[name]
        analytic = param.grad[index].item()
        with torch.no_grad():
            param[index] += eps
            plus = model.lewm_loss(batch, sigreg, 0.09)["loss"].item()
            param[index] -= 2 * eps
            minus = model.lewm_loss(batch, sigreg, 0.09)["loss"].item()
            param[index] += eps
        numeric = (plus - minus) / (2 * eps)
        torch.testing.assert_close(
            torch.tensor(numeric),
            torch.tensor(analytic),
            atol=1e-7,
            rtol=1e-4,
            msg=f"gradcheck failed for {name}{index}",
        )
