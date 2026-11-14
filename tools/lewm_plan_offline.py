#!/usr/bin/env python3
"""Offline CEM planning check for LeWM on an HDF5 episode file.

This does not need gymnasium: it takes one episode from a dataset, uses the
frame at ``--start`` as the planning context and the frame ``--goal-offset``
later as the goal, runs the same CEM solver used at evaluation, and reports
the terminal latent cost and the planned action blocks.

Example:

    .venv/bin/python tools/lewm_plan_offline.py \
        --data-path /data/pusht_expert_train.h5 \
        --checkpoint runs/lewm/last.pt \
        --episode 0 --start 10 --goal-offset 25 --device cuda
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pysrc"))

import torch  # noqa: E402
from lewm import CEMConfig, CEMSolver, LewmConfig, build_lewm  # noqa: E402
from lewm.data import EpisodeDataset, make_image_transform  # noqa: E402


def load_checkpoint(path, overrides):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model" in payload:
        saved = payload.get("model_config") or payload.get("config", {}).get("model", {})
        cfg = LewmConfig(**{**saved, **overrides})
        state_dict = payload["model"]
    else:
        cfg = LewmConfig(**overrides)
        state_dict = payload
    model = build_lewm(cfg)
    model.load_state_dict(state_dict, strict=True)
    return model, cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--n-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args(argv)

    dataset = EpisodeDataset(
        args.data_path,
        keys=("pixels", "action"),
        num_steps=4,
        frameskip=args.frameskip,
        transform=make_image_transform(args.img_size, resize=args.img_size != 224),
    )
    action_dim = dataset.get_dim("action")
    context = dataset.get_frame(args.episode, args.start)
    goal = dataset.get_frame(args.episode, args.start + args.goal_offset)

    overrides = dict(
        action_dim=action_dim,
        frameskip=args.frameskip,
        image_size=args.img_size,
        action_smoothed_dim=None,
    )
    model, cfg = load_checkpoint(args.checkpoint, overrides)
    model = model.to(args.device).eval()

    info = {
        "pixels": context.unsqueeze(0).unsqueeze(0).to(args.device),
        "goal": goal.unsqueeze(0).unsqueeze(0).to(args.device),
        "action": torch.zeros(
            1, 1, args.action_block * action_dim, device=args.device
        ),
    }
    solver = CEMSolver(
        model,
        CEMConfig(
            horizon=args.horizon,
            action_dim=args.action_block * action_dim,
            num_samples=args.num_samples,
            topk=args.topk,
            n_steps=args.n_steps,
            seed=args.seed,
            device=args.device,
            verbose=True,
        ),
    )
    output = solver.solve(info)

    # Report the terminal latent cost of the final mean too. get_cost expects
    # the solver's expanded layout: (B, S, history, ...).
    actions = output["actions"].to(args.device)
    with torch.no_grad():
        cost_info = {
            key: value.unsqueeze(1)
            for key, value in info.items()
        }
        cost = model.get_cost(cost_info, actions.unsqueeze(1))

    result = {
        "episode": args.episode,
        "start": args.start,
        "goal_offset": args.goal_offset,
        "config": {
            "horizon": args.horizon,
            "action_block": args.action_block,
            "action_dim": action_dim,
            "num_samples": args.num_samples,
            "topk": args.topk,
            "n_steps": args.n_steps,
            "seed": args.seed,
            "device": args.device,
        },
        "final_topk_cost": output["costs"][0],
        "final_mean_cost": float(cost[0, 0]),
        "actions": output["actions"][0].tolist(),
        "solve_time_s": output["solve_time"],
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(text + "\n")
    return result


if __name__ == "__main__":
    main()
