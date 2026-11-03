# FastJEPA

FastJEPA is a from-scratch C/CUDA project around the Joint Embedding Predictive
Architecture (JEPA) family. What i am trying to implement:

- **Foundation**: a fast C/CUDA implementation of LeJEPA training (ViT encoder +
  SIGReg regularizer) that is measurably faster than the official PyTorch
  reference under a documented benchmark.
- **Research**: a study of whether adding a JEPA-style latent prediction objective
  (with SIGReg) to a byte latent transformer improves data efficiency or
  representation quality. 
  [Note: this will be done in its own repo, but the code will be shared with FastJEPA.]
- **Research**: a minimal JEPA world model trained from pixels, with a fast CUDA
  planner (CEM/MPPI) operating in latent space on a single consumer GPU.

The project targets small models on small datasets on a single consumer GPU
(NVIDIA RTX 4060, 8 GB). Python/PyTorch code lives in `pysrc/` as the ground
truth and benchmark baseline. The training binary in `csrc/` does not depend on
PyTorch and is intended to be a from scratch C/CUDA implementation of LeJEPA training.

## Status 

While the C/CUDA side is not implemented yet, i have currently implemented a full
python ground truth to train and learn the LeJEPA architecture and resulting model.
And i have done some comparisons with the official implementation to ensure parity. 
The current status of the project is:

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
- `tools/analyze_run.py`: read a finished run, evaluate the frozen encoder,
  and write a Markdown report plus plots for loss curves, effective rank,
  Gaussianity, PCA, probe errors and view invariance.
- `pysrc/lejepa/analysis.py`: the pure-torch numbers behind that report
  (spectrum, effective rank, sliced Gaussianity, k-NN, confusion matrix).
- `tests/`: op tests against PyTorch, FP64 gradchecks, official parity in an
  isolated subprocess, overfit and end-to-end training smoke tests.
- `bench/`: SIGReg forward/backward and synthetic training-step benchmarks with
  a shared CUDA-event harness and JSONL results.

## Quick start

Make sure to checkout the well written and documented ´CLI_REFERENCE.md´ for a complete
list of commands and options.

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

# Analyze a finished run (writes runs/<name>/analysis/report.md and plots/).
pip install -e ".[analysis]" --no-deps   # adds matplotlib
python tools/analyze_run.py --run-dir runs/lejepa_cifar10

# Benchmarks (writes bench/results/*.jsonl).
python bench/run_benchmarks.py
```

The C/CUDA side is built with CMake (`cmake -S . -B build -DUSE_CUDA=ON`), but
`csrc/` is still a skeleton.

## Experiments

A complete and structured list of experiments and results is maintained in `docs/EXPERIMENT_LOG.md`.
So if you are interested -> check it out ;D

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

## License

Apache 2.0 (see `LICENSE`).
