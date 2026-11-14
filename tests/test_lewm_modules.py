"""Op-level parity: our LeWM modules vs the pinned official implementation.

Official checkout: ``~/projects/le-wm`` at
``8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`` (override with
``LEWM_OFFICIAL_ROOT``). The official ``module.py`` is loaded by path so it
cannot shadow this package's modules.
"""

import importlib.util
import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

from lewm.modules import MLP, ARPredictor, Embedder, SIGReg  # noqa: E402

OFFICIAL_ROOT = os.path.abspath(
    os.environ.get("LEWM_OFFICIAL_ROOT", os.path.expanduser("~/projects/le-wm"))
)

pytestmark = pytest.mark.skipif(
    not os.path.isdir(OFFICIAL_ROOT),
    reason=f"official le-wm not found at {OFFICIAL_ROOT}",
)


@pytest.fixture(scope="module")
def official_module():
    path = os.path.join(OFFICIAL_ROOT, "module.py")
    if not os.path.isfile(path):
        pytest.skip(f"official module.py not found at {path}")
    spec = importlib.util.spec_from_file_location("official_lewm_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _copy(official, ours):
    """Load the official weights into ours, requiring an exact key match."""
    ours.load_state_dict(official.state_dict(), strict=True)
    return ours


def test_predictor_parity(official_module):
    kwargs = dict(
        num_frames=3,
        depth=6,
        heads=16,
        mlp_dim=2048,
        input_dim=192,
        hidden_dim=192,
        output_dim=192,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.0,
    )
    torch.manual_seed(0)
    official = official_module.ARPredictor(**kwargs)
    ours = _copy(official, ARPredictor(**kwargs))
    official.eval()
    ours.eval()

    x = torch.randn(2, 3, 192)
    c = torch.randn(2, 3, 192)
    with torch.no_grad():
        expected = official(x, c)
        actual = ours(x, c)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_embedder_parity(official_module):
    official = official_module.Embedder(input_dim=10, emb_dim=192)
    ours = _copy(official, Embedder(input_dim=10, emb_dim=192))
    official.eval()
    ours.eval()

    x = torch.randn(2, 4, 10)
    with torch.no_grad():
        torch.testing.assert_close(ours(x), official(x), atol=0, rtol=0)


def test_mlp_parity(official_module):
    for norm_fn in (nn.BatchNorm1d, nn.LayerNorm):
        kwargs = dict(
            input_dim=192,
            output_dim=192,
            hidden_dim=2048,
            norm_fn=norm_fn,
        )
        official = official_module.MLP(**kwargs)
        ours = _copy(official, MLP(**kwargs))
        official.eval()
        ours.eval()

        x = torch.randn(7, 192)
        with torch.no_grad():
            torch.testing.assert_close(ours(x), official(x), atol=0, rtol=0)


def test_sigreg_parity(official_module):
    official = official_module.SIGReg(knots=17, num_proj=64)
    ours = SIGReg(knots=17, num_proj=64)

    # same buffers
    for key, value in official.state_dict().items():
        torch.testing.assert_close(ours.state_dict()[key], value, atol=0, rtol=0)

    x = torch.randn(4, 32, 192)

    # Directions are sampled internally; seeding both calls makes them equal.
    torch.manual_seed(7)
    expected = official(x)
    torch.manual_seed(7)
    actual = ours(x)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    torch.manual_seed(7)
    official(x1).backward()
    torch.manual_seed(7)
    ours(x2).backward()
    torch.testing.assert_close(x2.grad, x1.grad, atol=1e-6, rtol=1e-5)


def test_encoder_matches_huggingface():
    transformers = pytest.importorskip("transformers")
    from lewm.encoder import ViTEncoder

    cfg = transformers.ViTConfig(
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
        image_size=224,
        patch_size=14,
        num_channels=3,
        qkv_bias=True,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    reference = transformers.ViTModel(cfg, add_pooling_layer=False)

    def to_transformers_5(key):
        key = key.replace(".attention.attention.query.", ".attention.q_proj.")
        key = key.replace(".attention.attention.key.", ".attention.k_proj.")
        key = key.replace(".attention.attention.value.", ".attention.v_proj.")
        key = key.replace(".attention.output.dense.", ".attention.o_proj.")
        key = key.replace(".intermediate.dense.", ".mlp.fc1.")
        key = key.replace(".output.dense.", ".mlp.fc2.")
        return key.replace("encoder.layer.", "layers.")

    torch.manual_seed(0)
    ours = ViTEncoder().eval()
    reference.load_state_dict(
        {to_transformers_5(k): v for k, v in ours.state_dict().items()},
        strict=True,
    )
    reference.eval()

    for size in (224, 28):
        x = torch.rand(2, 3, size, size)
        with torch.no_grad():
            actual = ours(x, interpolate_pos_encoding=True).last_hidden_state
            expected = reference(
                x, interpolate_pos_encoding=True
            ).last_hidden_state
        # The reference uses SDPA; ours uses explicit eager attention, which is
        # what transformers 4.x did when the checkpoint was trained.
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
