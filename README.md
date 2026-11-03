# FastJEPA

FastJEPA is a from-scratch C/CUDA project around the Joint Embedding Predictive
Architecture (JEPA) family. It has three tracks:

1. **Track A (foundation):** a C/CUDA implementation of LeJEPA training (ViT
   encoder + SIGReg regularizer) that is measurably faster than the official
   PyTorch reference under a documented benchmark.
2. **Track B (research):** a study of whether adding a JEPA-style latent
   prediction objective (with SIGReg) to a byte latent transformer improves
   data efficiency or representation quality.
3. **Track C (research):** a minimal JEPA world model trained from pixels, with
   a fast CUDA planner (CEM/MPPI) operating in latent space on a single
   consumer GPU.

The project targets small models on small datasets on a single consumer GPU
(NVIDIA RTX 4060, 8 GB). Python/PyTorch code lives in `pysrc/` as the ground
truth and benchmark baseline; the training binary in `csrc/` does not depend on
PyTorch.

## What works today

The Python ground truth is complete enough to train LeJEPA and to compare
against the pinned official implementation:

- `pysrc/lejepa/sigreg.py`: sliced Epps-Pulley SIGReg with FP32 statistics for
  FP16/BF16 inputs, FP64 support for gradchecks, shared directions across
  views, and per-slice output for parity tests.
- `pysrc/lejepa/model.py`: timm ViT-S encoder (`vit_small_patch8_224`) with
  `drop_path_rate` and `dynamic_img_size`, plus the `Linear-BN-ReLU` projector.
- `pysrc/lejepa/loss.py`: LeJEPA objective with `V_g`, `pred_reduction` and
  `per_view` / `joint` SIGReg modes.
- `pysrc/lejepa/train.py`: full M-style augmentation recipe, train/val loaders,
  online probe, per-step warmup + cosine schedule, FP32/BF16/FP16, seeding,
  NaN guard, JSONL logging and full checkpoints.
- `configs/lejepa_parity.json` and `configs/lejepa_cifar10.json`: runnable
  recipes.
- `tests/`: op tests against PyTorch, FP64 gradchecks, official parity in an
  isolated subprocess, overfit and end-to-end training smoke tests.
- `bench/`: SIGReg forward/backward and synthetic training-step benchmarks with
  a shared CUDA-event harness and JSONL results.

## Quick start

```bash
# Install the Python ground truth (torch/torchvision/timm must be available).
pip install -r requirements.txt
pip install -e . --no-deps

# Tests and lint.
pytest tests/ -q
ruff check .
ruff format --check .

# Train (CIFAR-10 config; data is downloaded into data/ on first use).
python -m lejepa.train --config configs/lejepa_cifar10.json

# Benchmarks (writes bench/results/*.jsonl).
python bench/run_benchmarks.py
```

The C/CUDA side is built with CMake (`cmake -S . -B build -DUSE_CUDA=ON`), but
`csrc/` is still a skeleton.

## Status and honest limitations

- **C/CUDA kernels are not implemented yet.** `csrc/` contains the build
  skeleton only. No speedup over the official implementation is claimed until
  a kernel exists and a script in `bench/` reproduces the number.
- **Current benchmarks compare two Python implementations**: this repository's
  SIGReg and the pinned official SIGReg, under the same shapes, dtype and
  timing protocol. Results and reproduction commands are in
  `docs/experiments/001_sigreg_benchmark.md`.
- **Track B and Track C are not started.** Their sections of
  `fast_JEPA_protocol.md` are the plan, not a description of working code.
- The official LeJEPA checkout used for parity and benchmarks is licensed
  CC BY-NC 4.0. It is an external reference only; see `docs/reuse.md`.

## Layout

```
csrc/       C and CUDA code (skeleton)
pysrc/      PyTorch ground truth and benchmark baselines
tests/      pytest suite and reference dumps
tools/      reference-dump and diagnostic scripts
bench/      benchmark scripts and raw JSONL results
configs/    JSON training configs
docs/       protocol, conventions, reuse audit, experiment write-ups
```

The single source of truth for scope, success criteria and numerical rules is
`fast_JEPA_protocol.md`. Agent instructions are in `AGENTS.md`.

## License

Apache 2.0 (see `LICENSE`).
