"""CEM planner parity and behaviour.

The reference update rule is transcribed from
``stable_worldmodel/planning/solver/cem.py`` (MIT); the transcription in this
file is deliberately literal so the test does not depend on that package.
"""

import os
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

from lewm import CEMConfig, CEMSolver, LewmConfig, build_lewm  # noqa: E402


class QuadraticCost(nn.Module):
    """Toy cost whose optimum is a known action sequence."""

    def __init__(self, target):
        super().__init__()
        self.register_buffer("target", target)
        self.calls = []

    def get_cost(self, info_dict, candidates):
        self.calls.append(candidates.detach().clone())
        return (candidates - self.target).pow(2).sum(dim=(2, 3))


def _expand(info, num_samples):
    out = {}
    for key, value in info.items():
        out[key] = value.unsqueeze(1).expand(value.size(0), num_samples, *value.shape[1:])
    return out


def reference_cem(
    cost,
    info,
    *,
    horizon,
    action_dim,
    num_samples,
    topk,
    n_steps,
    seed,
    init_action=None,
):
    """Literal transcription of the reference CEM solve loop."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total_envs = len(next(iter(info.values())))
    if init_action is None:
        mean = torch.zeros(total_envs, horizon, action_dim)
    else:
        assert init_action.size(1) <= horizon
        remaining = horizon - init_action.size(1)
        tail = torch.zeros(total_envs, remaining, action_dim)
        mean = torch.cat([init_action, tail], dim=1)
    var = torch.ones(total_envs, horizon, action_dim)

    for _ in range(n_steps):
        candidates = torch.randn(total_envs, num_samples, horizon, action_dim, generator=generator)
        candidates = candidates * var.unsqueeze(1) + mean.unsqueeze(1)
        candidates[:, 0] = mean

        costs = cost.get_cost(_expand(info, num_samples), candidates)
        _, topk_inds = torch.topk(costs, k=topk, dim=1, largest=False)
        batch_indices = torch.arange(total_envs).unsqueeze(1).expand(-1, topk)
        elites = candidates[batch_indices, topk_inds]
        mean = elites.mean(dim=1)
        var = elites.std(dim=1, correction=0)

    return mean, var


def test_cem_matches_reference_update_rule():
    B, H, D = 2, 3, 4
    g = torch.Generator().manual_seed(0)
    target = torch.randn(B, 1, H, D, generator=g)
    init_action = torch.randn(B, 1, D, generator=g)
    info = {"pixels": torch.zeros(B, 1, 1)}

    kwargs = dict(
        horizon=H,
        action_dim=D,
        num_samples=6,
        topk=2,
        n_steps=4,
        seed=11,
    )
    cost = QuadraticCost(target)
    solver = CEMSolver(cost, CEMConfig(**kwargs, batch_size=B))
    out = solver.solve(dict(info), init_action=init_action)

    expected_mean, expected_var = reference_cem(
        QuadraticCost(target),
        info,
        init_action=init_action,
        **{k: v for k, v in kwargs.items() if k != "action_dim"},
        action_dim=D,
    )

    torch.testing.assert_close(out["actions"], expected_mean, atol=1e-6, rtol=0)
    torch.testing.assert_close(out["var"][0], expected_var, atol=1e-6, rtol=0)
    assert out["actions"].shape == (B, H, D)

    # Candidate 0 is forced to the current mean at every iteration. The first
    # mean is the warm start zero-padded to the full horizon.
    padded_init = torch.cat([init_action, torch.zeros(B, H - init_action.size(1), D)], dim=1)
    first = cost.calls[0]
    torch.testing.assert_close(first[:, 0], padded_init, atol=0, rtol=0)


def test_cem_is_seed_deterministic_and_improves_cost():
    B, H, D = 1, 2, 3
    g = torch.Generator().manual_seed(1)
    target = torch.randn(B, 1, H, D, generator=g)
    info = {"pixels": torch.zeros(B, 1, 1)}
    cfg = CEMConfig(horizon=H, action_dim=D, num_samples=32, topk=8, n_steps=6, seed=5)

    first = CEMSolver(QuadraticCost(target), cfg).solve(dict(info))
    second = CEMSolver(QuadraticCost(target), cfg).solve(dict(info))
    torch.testing.assert_close(first["actions"], second["actions"], atol=0, rtol=0)

    cost = QuadraticCost(target)
    CEMSolver(cost, cfg).solve(dict(info))
    initial = cost.calls[0][:, 0]
    final = cost.calls[-1][:, 0]
    initial_err = (initial - target[:, 0]).pow(2).sum()
    final_err = (final - target[:, 0]).pow(2).sum()
    assert final_err < initial_err


@pytest.mark.parametrize("seed", [0])
def test_cem_with_real_model_smoke(seed):
    cfg = LewmConfig(
        image_size=28,
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
    model = build_lewm(cfg).eval()
    B = 1
    g = torch.Generator().manual_seed(seed)
    info = {
        "pixels": torch.rand(B, 1, 3, 28, 28, generator=g),
        "goal": torch.rand(B, 1, 3, 28, 28, generator=g),
        "action": torch.zeros(B, 1, cfg.action_block_dim),
    }
    solver = CEMSolver(
        model,
        CEMConfig(
            horizon=2,
            action_dim=cfg.action_block_dim,
            num_samples=6,
            topk=2,
            n_steps=2,
            seed=seed,
        ),
    )
    out = solver.solve(info)
    assert out["actions"].shape == (B, 2, cfg.action_block_dim)
    assert torch.isfinite(out["actions"]).all()
    assert len(out["costs"]) == B
