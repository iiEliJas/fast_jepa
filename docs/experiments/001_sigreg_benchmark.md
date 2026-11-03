# Experiment 001: SIGReg and training-step benchmarks (Python)

Date: 2026-11-02

This experiment compares this repository's SIGReg implementation with the
pinned official LeJEPA implementation under identical shapes and timing
settings. It is a Python-vs-Python baseline. No C/CUDA kernel exists yet, so
this experiment makes no Track A speedup claim.

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060, 8 GB, capability 8.9, 24 SMs |
| Driver | 616.92 |
| CUDA / cuDNN | 13.2 / 9.26.0 |
| PyTorch | 2.15.0+cu132 |
| SM clock during run | 2490-2610 MHz (max 3105 MHz) |
| Memory clock during run | 8251 MHz |
| FastJEPA HEAD | `4b62fbfb4afd796430a043aa8fdb0556e866e447` |
| Official LeJEPA | `c293d291ca87cd4fddee9d3fffe4e914c7272052` |
| Our SIGReg | `pysrc/lejepa/sigreg.py` (`SlicedEppsPulley`) |
| Official SIGReg | `lejepa.multivariate.SlicingUnivariateTest` + `lejepa.univariate.EppsPulley` |

At measurement time the benchmark scripts and this document were uncommitted
working-tree changes on top of that commit; each JSONL record contains the
commit of the tree it actually ran in.

Method (protocol section 5.7): FP32, `K=16`, `S=256`, `t_max=3.0`,
`n_knots=17`; 20 warmup iterations discarded; 50 timed iterations; each
iteration timed with a `torch.cuda.Event` pair and an explicit synchronize;
median and interquartile range reported. A 50-iteration clock-ramp pass runs
before the first measured configuration. Every raw record carries the git
commit, device properties, driver, CUDA version and config.

The official checkout is imported from `~/projects/lejepa`; our modules are
loaded by file path so the two packages do not collide over `sys.modules`.

## SIGReg forward and forward+backward (2D, `N x K`)

Median [p25, p75] in milliseconds.

| N | ours fwd | official fwd | ours fwd+bwd | official fwd+bwd |
|---:|---:|---:|---:|---:|
| 256 | 0.466 [0.448, 0.500] | 0.502 [0.479, 0.563] | 1.300 [1.222, 1.426] | 1.200 [1.149, 1.351] |
| 512 | 0.463 [0.439, 0.485] | 0.526 [0.486, 0.561] | 1.371 [1.324, 1.542] | 1.442 [1.403, 1.524] |
| 1024 | 0.580 [0.568, 0.614] | 0.757 [0.747, 0.789] | 2.372 [2.285, 2.614] | 2.330 [2.293, 2.474] |
| 2048 | 1.231 [1.216, 1.264] | 1.319 [1.311, 1.374] | 4.362 [4.285, 4.752] | 4.462 [4.376, 4.836] |
| 4096 | 2.374 [2.338, 2.477] | 2.363 [2.322, 2.512] | 8.817 [8.463, 9.020] | 8.772 [8.498, 9.052] |
| 8192 | 4.437 [4.370, 4.908] | 4.461 [4.372, 4.915] | 17.122 [16.923, 17.222] | 17.163 [17.068, 17.306] |

Forward: ours is about 7% faster at N=256, 12% faster at N=512 and 23% faster
at N=1024; the two are within noise at N=2048, N=4096 and N=8192. Backward: the
differences are inside or near the interquartile range at every N. The official
implementation is slightly faster at N=256 and N=1024, ours at N=512, 2048,
4096 and 8192. Do not claim a backward win from this run.

## SIGReg forward with four views (`4 x N x K`)

Median [p25, p75] in milliseconds.

| N | ours | official |
|---:|---:|---:|
| 256 | 0.573 [0.560, 0.605] | 0.754 [0.745, 0.774] |
| 512 | 1.175 [1.168, 1.185] | 1.344 [1.324, 1.401] |
| 1024 | 2.291 [2.277, 2.390] | 2.370 [2.344, 2.636] |
| 2048 | 4.469 [4.369, 4.808] | 4.494 [4.388, 5.015] |

The shared-direction path avoids re-sampling directions per view, which is
where the small-N gain comes from: 24% at N=256, 13% at N=512, 3% at N=1024,
and within noise at N=2048 where both are effectively bandwidth-bound.

## Synthetic training step

V=2 views, B=8 images, 32x32 input, 2-layer ViT (`embed_dim=64`, patch 4,
2 heads), projector `512 -> 128 -> 16`, FP32, AdamW. Synthetic tensors, so
data loading and augmentation are excluded. 5 warmup steps, 30 timed steps,
same event timing as above.

| SIGReg | forward (ms) | forward+backward (ms) | full step (ms) | images/s | peak MB |
|---|---:|---:|---:|---:|---:|
| ours | 2.592 [2.546, 2.717] | 6.876 [6.414, 7.671] | 7.376 [7.203, 8.023] | 2169 | 22.0 |
| official | 2.700 [2.642, 2.893] | 7.194 [6.568, 8.054] | 7.617 [7.355, 8.832] | 2101 | 22.0 |

The full-step difference is about 3% in this configuration, with overlapping
interquartile ranges. The model is tiny and launch-bound, so this is not a
representative ViT-S 128px number. Treat it as a harness smoke test, not a
headline result.

## Reproduction

```bash
# Both implementations, full N grid, writes bench/results/*.jsonl.
python bench/run_benchmarks.py

# Individual pieces.
python bench/bench_sigreg.py --impl both --mode both --views 1
python bench/bench_sigreg.py --impl both --mode forward --views 4
python bench/bench_train.py --sigreg both

# Quick smoke test.
python bench/run_benchmarks.py --quick
```

Raw results from this run:

- `bench/results/sigreg.jsonl`
- `bench/results/sigreg_views.jsonl`
- `bench/results/train.jsonl`

The official baseline is read from `~/projects/lejepa` by default; override
with `--official-root` or `LEJEPA_OFFICIAL_ROOT`.

## Limitations

- One consumer GPU, no clock locking. Clocks are sampled during the run and
  reported, but absolute numbers drift by a few percent between runs.
- The official implementation is timed through its public
  `SlicingUnivariateTest` interface, which samples directions internally each
  call. Ours samples directions each call as well, so the comparison is fair,
  but neither is a pure kernel benchmark.
- No `torch.compile` baseline yet (protocol section 5.7 lists it as a
  baseline).
- No C/CUDA kernel exists, so the C side of the benchmark protocol is not
  exercised.
- The official LeJEPA checkout is CC BY-NC 4.0; it is used here only as an
  external reference. See `docs/reuse.md`.
