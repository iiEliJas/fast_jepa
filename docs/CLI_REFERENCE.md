# CLI Reference

Python ground truth entry points. After `pip install -e . --no-deps` the
`pysrc` packages are importable from any directory; `PYTHONPATH=pysrc` still
works if you prefer not to install.

## Install

```bash
pip install -r requirements.txt
pip install -e . --no-deps

# Optional LeWM pieces
pip install -e ".[lewm-data]"    # h5py, for the HDF5 episode loader
pip install -e ".[lewm-parity]"  # einops + transformers, for official parity
pip install -e ".[lewm-env]"     # gymnasium/gym-pusht/stable-worldmodel, later
```

## Tests

```bash
pytest tests/ -q
pytest tests/test_lewm_*.py -q   # LeWM only
```

The parity tests run the official LeJEPA checkout in a subprocess. They skip
when it is missing. Point them at another checkout with:

```bash
LEJEPA_OFFICIAL_ROOT=~/projects/lejepa pytest tests/test_parity.py -q
```

## Training

```bash
python -m lejepa.train --config configs/lejepa_cifar10.json
python -m lejepa.train --config configs/lejepa_parity.json --out-dir runs/parity
fastjepa-train --config configs/lejepa_cifar10.json
```

Outputs go to `runs/<config-name>/` unless `--out-dir` is given:

- `run.json`: git commit, resolved config, device/driver/CUDA, timestamp, seed.
- `log.jsonl`: per-step and per-epoch records. Epoch records include
  `epoch_seconds`, `elapsed_seconds`, `eta_seconds` and `eta_at`.
- `ckpt.pt` and `ckpt_epoch_*.pt`: model, optimizer, scheduler, RNG state.

The terminal prints one line per epoch with the epoch duration, total elapsed
time and the estimated finish time:

```
epoch 3 loss 0.1931 lp 0.1458 ls 2.3721 erank 15.42 acc 0.6406 | epoch 43s elapsed 2m 54s eta 10m 06s (15:42:11)
```

`configs/lejepa_parity.json` expects Imagenette under
`data/imagenette2-160/{train,val}`. Missing `data_dir` is an error, not a silent
fallback.

## Run analysis

`tools/analyze_run.py` reads `run.json`, `log.jsonl` and `ckpt.pt`, evaluates
the frozen encoder and its probe on the validation split, and writes a guided
report. Install the optional plotting dependency first:

```bash
pip install -e ".[analysis]" --no-deps   # matplotlib
python tools/analyze_run.py --run-dir runs/lejepa_cifar10
```

Outputs go to `runs/<name>/analysis/`:

- `report.md`: headline metrics, interpretation and figure guide.
- `metrics.json`: every number used in the report, plus config and device.
- `plots/*.png`: training curves, optimization, loss decomposition, probe
  confusion/calibration, rank spectrum, Gaussianity, PCA and representation
  quality.

Useful options: `--checkpoint` for `ckpt_last.pt`, `--out-dir` to keep the
report elsewhere, `--device cpu`, `--knn-train-limit` for a smaller k-NN set,
and `--offline-probe-epochs 0` to skip the offline probe. The numerical helpers
live in `pysrc/lejepa/analysis.py` and are covered by the test suite.

## LeWM

Training on an HDF5 episode file (see `configs/lewm_pusht.json`):

```bash
python -m lewm.train --config configs/lewm_pusht.json \
    --data-path /data/pusht_expert_train.h5 --epochs 10
fastjepa-lewm-train --config configs/lewm_pusht.json --data-path ...
```

Outputs go to `--out-dir` (`runs/lewm_pusht` by default in the config):

- `train.jsonl`: first record has the git commit, config, model config,
  device/driver/CUDA; then per-step losses, learning rate, embedding std,
  effective rank, step time and peak memory; then eval records.
- `last.pt`: model, optimizer, SIGReg buffers, config, step and epoch.
  Load with `torch.load(..., weights_only=False)`.

Useful options: `--limit-steps` for smoke runs, `--precision fp32|bf16|fp16`,
`--grad-accum-steps` for a smaller batch, `--cache-data` when RAM allows,
`--num-workers`.

Offline CEM planning on one dataset episode (no gymnasium needed):

```bash
python tools/lewm_plan_offline.py --data-path /data/pusht_expert_train.h5 \
    --checkpoint runs/lewm_pusht/last.pt --episode 0 --start 10 \
    --goal-offset 25 --device cuda
```

It prints the final top-k cost, the terminal cost of the mean plan, the planned
action blocks and the solve time; `--out` writes the same JSON to a file.

LeWM tests:

```bash
pytest tests/test_lewm_modules.py tests/test_lewm_model.py \
    tests/test_lewm_planner.py tests/test_lewm_data.py \
    tests/test_lewm_train.py -q
```

The module/model/planner parity tests need the official checkout at
`~/projects/le-wm` (override with `LEWM_OFFICIAL_ROOT`) and, for the encoder
comparison, `transformers`. The checkpoint test uses the cached
`quentinll/lewm-pusht` weights or `LEWM_PUSH_T_WEIGHTS`.

## Benchmarks

```bash
python bench/run_benchmarks.py            # full grid, both implementations
python bench/run_benchmarks.py --quick    # small grid
python bench/run_benchmarks.py --impl fastjepa  # skip the official baseline
python bench/bench_sigreg.py --help
python bench/bench_train.py --help
```

Raw results are written to `bench/results/*.jsonl`. The official baseline is
read from `~/projects/lejepa` by default; override with `--official-root` or
`LEJEPA_OFFICIAL_ROOT`.

## Reference dumps

```bash
python tools/dump_sigreg.py   # writes tests/data/sigreg_ref.pt
```

## Ruff

```bash
ruff check .
ruff format --check .
```
