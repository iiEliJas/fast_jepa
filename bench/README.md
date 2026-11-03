# Benchmarks

Python-vs-Python baselines for SIGReg and a synthetic training step. The C/CUDA
kernels do not exist yet, so nothing here is a Track A speedup claim.

## Run

```bash
python bench/run_benchmarks.py              # full N grid, both implementations
python bench/run_benchmarks.py --quick      # small grid / fewer iterations
python bench/run_benchmarks.py --impl fastjepa
```

The official baseline is read from `~/projects/lejepa` by default. Override
with `--official-root` or `LEJEPA_OFFICIAL_ROOT`. If the checkout is missing,
the runner falls back to `--impl fastjepa` and says so.

Raw results land in `bench/results/*.jsonl`; the write-up is
`docs/experiments/001_sigreg_benchmark.md`.

## Timing method

`bench/harness.py` follows protocol section 5.7:

- warmup iterations are discarded;
- CUDA work uses one `torch.cuda.Event` pair per iteration and an explicit
  synchronize; CPU falls back to `time.perf_counter`;
- median, p25, p75, IQR, mean, standard deviation, min and max are reported;
- a clock-ramp pass runs before the first measured configuration;
- every record carries git commit, device name/capability/memory, driver,
  CUDA/cuDNN versions, clocks and the full config.

## Record schema

Common fields: `schema`, `kind`, `bench`, `timestamp`, `git_commit`, `device`,
`config`, `timer`, `iters`, `warmup`, `median_ms`, `p25_ms`, `p75_ms`,
`iqr_ms`, `mean_ms`, `std_ms`, `min_ms`, `max_ms`.

SIGReg records add `impl`, `mode` (`forward` / `backward`), `views`, `shape`,
`N`, `K`, `S`, `dtype` and `module_dtype`.

Training records add `sigreg`, `phase` (`forward`, `forward_backward`,
`full_step`), `images_per_s` and `peak_mem_mb` (full step only).

## Files

- `bench_sigreg.py`: SIGReg forward/backward over the N grid.
- `bench_train.py`: synthetic training step with either SIGReg.
- `run_benchmarks.py`: one-command runner and markdown summary.
- `harness.py`: timing, statistics, device metadata, JSONL I/O.
- `results/`: raw JSONL from the published run.
