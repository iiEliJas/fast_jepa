# CLI Reference

Python ground truth entry points. After `pip install -e . --no-deps` the
`pysrc` packages are importable from any directory; `PYTHONPATH=pysrc` still
works if you prefer not to install.

## Install

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

## Tests

```bash
pytest tests/ -q
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
- `log.jsonl`: per-step and per-epoch records.
- `ckpt.pt` and `ckpt_epoch_*.pt`: model, optimizer, scheduler, RNG state.

`configs/lejepa_parity.json` expects Imagenette under
`data/imagenette2-160/{train,val}`. Missing `data_dir` is an error, not a silent
fallback.

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
