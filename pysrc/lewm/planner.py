"""Cross-Entropy Method planner for LeWM.

Ported from ``stable_worldmodel/planning/solver/cem.py`` (MIT) so that the
Python planner is a correct oracle for the future CUDA implementation.

Differences from the reference solver are structural only: this version takes
``horizon``/``action_dim`` directly instead of calling ``configure()`` with a
gymnasium action space, and it does not print timing by default.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class CEMConfig:
    horizon: int = 5
    action_dim: int = 10  # frameskip * env action dim
    num_samples: int = 300
    topk: int = 30
    n_steps: int = 30
    var_scale: float = 1.0
    batch_size: int = 1
    device: str = "cpu"
    seed: int = 1234
    verbose: bool = False


class CEMSolver:
    """Cross-Entropy Method planner.

    The solver expects ``info_dict`` tensors of shape ``(B, ...)`` (one context
    frame, ``history_len = 1``) and a cost object exposing
    ``get_cost(info_dict, action_candidates) -> (B, S)``, e.g. ``lewm.JEPA``.

    Candidates have shape ``(B, S, horizon, action_dim)``. After ``n_steps``
    iterations the returned ``actions`` are the final distribution mean,
    shape ``(B, horizon, action_dim)``.
    """

    def __init__(self, cost, config: CEMConfig | None = None, **overrides):
        cfg = config or CEMConfig()
        if overrides:
            from dataclasses import replace

            cfg = replace(cfg, **overrides)
        self.cost = cost
        self.cfg = cfg
        self.torch_gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
        try:
            self._dtype = next(cost.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32

    # -- properties kept for API parity with the reference solver ----------
    @property
    def horizon(self) -> int:
        return self.cfg.horizon

    @property
    def action_dim(self) -> int:
        return self.cfg.action_dim

    @property
    def num_samples(self) -> int:
        return self.cfg.num_samples

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def init_action_distrib(self, n_envs, actions=None):
        """Initial mean and std of the action distribution."""
        var = self.cfg.var_scale * torch.ones(
            [n_envs, self.horizon, self.action_dim], dtype=self.dtype
        )
        if actions is None:
            mean = torch.zeros([n_envs, 0, self.action_dim], dtype=self.dtype)
        else:
            mean = actions

        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            tail = torch.zeros(
                [n_envs, remaining, self.action_dim],
                dtype=self.dtype,
                device=mean.device,
            )
            mean = torch.cat([mean, tail], dim=1).to(mean.device)
        return mean, var

    def _prepare_init_action(self, info_dict, init_action, n_envs, device):
        """Extend or zero-pad a warm-start plan to the full horizon."""
        if init_action is not None:
            assert init_action.shape[0] == n_envs, (
                f"init_action batch {init_action.shape[0]} != n_envs {n_envs}"
            )
            assert init_action.shape[2] == self.action_dim, (
                f"init_action action_dim {init_action.shape[2]} != {self.action_dim}"
            )
            init_action = init_action.to(device)

        n_prev = init_action.shape[1] if init_action is not None else 0
        remaining = self.horizon - n_prev
        if remaining <= 0:
            return init_action

        tail = torch.zeros(n_envs, remaining, self.action_dim, device=device)
        if init_action is not None:
            return torch.cat([init_action, tail], dim=1)
        return tail

    @torch.inference_mode()
    def solve(self, info_dict, init_action=None):
        start_time = time.time()
        cfg = self.cfg
        device = cfg.device
        total_envs = len(next(iter(info_dict.values())))

        init_action = self._prepare_init_action(info_dict, init_action, total_envs, device)
        mean, var = self.init_action_distrib(total_envs, init_action)
        mean = mean.to(device)
        var = var.to(device)

        outputs = {"costs": [], "mean": [], "var": []}

        for start_idx in range(0, total_envs, cfg.batch_size):
            end_idx = min(start_idx + cfg.batch_size, total_envs)
            current_bs = end_idx - start_idx

            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            expanded = {}
            for key, value in info_dict.items():
                v_batch = value[start_idx:end_idx]
                if torch.is_tensor(value):
                    target_dtype = self.dtype if v_batch.is_floating_point() else None
                    v_batch = (
                        v_batch.to(device=device, dtype=target_dtype)
                        .unsqueeze(1)
                        .expand(current_bs, cfg.num_samples, *v_batch.shape[1:])
                    )
                elif isinstance(value, np.ndarray):
                    v_batch = np.repeat(v_batch[:, None, ...], cfg.num_samples, axis=1)
                expanded[key] = v_batch

            final_batch_cost = None

            for _ in range(cfg.n_steps):
                candidates = torch.randn(
                    current_bs,
                    cfg.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=device,
                    dtype=self.dtype,
                )
                candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)
                candidates[:, 0] = batch_mean

                costs = self.cost.get_cost(expanded, candidates)
                assert isinstance(costs, torch.Tensor), f"expected a tensor cost, got {type(costs)}"
                assert costs.ndim == 2 and costs.shape == (
                    current_bs,
                    cfg.num_samples,
                ), f"expected cost shape {(current_bs, cfg.num_samples)}, got {tuple(costs.shape)}"

                topk_vals, topk_inds = torch.topk(costs, k=cfg.topk, dim=1, largest=False)
                batch_indices = (
                    torch.arange(current_bs, device=device).unsqueeze(1).expand(-1, cfg.topk)
                )
                topk_candidates = candidates[batch_indices, topk_inds]

                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1, correction=0)
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()

            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var
            outputs["costs"].extend(final_batch_cost)

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]
        outputs["solve_time"] = time.time() - start_time

        if cfg.verbose:
            print(f"CEM solve time: {outputs['solve_time']:.4f} seconds")
        return outputs
