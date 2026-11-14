# Conventions (from fBLT)

Status: A0 audit refreshed 2026-11-07 against the real owner checkout.

| Item | Value |
|---|---|
| Source repository | `~/projects/fast_BLT` (not `~/projects/fBLT`; that path does not exist) |
| Commit | `72deb5cd749bae0fba2625210f0841c7eaae77c8` |
| Describe | `v1.0.1-34-g72deb5c` |
| License | Apache-2.0 (`LICENSE`) |
| Audited on | 2026-11-07 |
| Supersedes | the `5da89ec4...` audit |

This document records the conventions the FastJEPA C/CUDA tree follows. The
protocol is the source of truth for correctness and numerics; where only style
is involved, fBLT wins. The reuse decisions live in `docs/reuse.md`.

## 1. Build

- CMake, minimum version 3.18. Host code is C99. CUDA code is C++17.
- The CUDA switch is `USE_CUDA` (already in the root `CMakeLists.txt`), not a
  new `FJ_USE_CUDA` spelling. CUDA builds define `FJ_WITH_CUDA`; C99
  translation units never include CUDA headers.
- Default CUDA architectures are `75;89` (T4 and RTX 4060). Override with
  `-DCMAKE_CUDA_ARCHITECTURES=...`. A single-arch build is valid for local
  iteration; release builds cover both.
- `CUDAToolkit_ROOT` / `CMAKE_CUDA_COMPILER` may be supplied explicitly. The
  root `CMakeLists.txt` already prefers `/usr/local/cuda` when present.
- Include directories are `csrc` and `tests`, globally, so headers are included
  as `core/tensor.h`, `ops/sigreg.h`, etc.
- Core sources are listed explicitly. Test sources may use
  `file(GLOB_RECURSE ... CONFIGURE_DEPENDS)` like fBLT.
- Link `m`; with CUDA also `CUDA::cudart`, `CUDA::cublas`, `CUDA::cublasLt`.
- Compiler flags: `-O2 -Wall -Wextra`. No `-ffast-math`. No implicit TF32.
- Embed `FJ_GIT_COMMIT` from `git rev-parse HEAD` (fallback `unknown`) and log
  it in every run.
- Targets:
  - `fj_core` — static C/CUDA library.
  - `fastjepa` — CLI (`info`, `train`, `eval`, `convert`, `dump`).
  - `fj_test` — C test suite.
  - `bench_harness` + `bench_*` — benchmarks.
  - `test` — custom target that runs `fj_test` (fBLT convention). A `ctest`
    registration may be added as a convenience but is not the primary entry.
  - `cuda-smoke`, `cuda-sanitize` when `USE_CUDA=ON`.
- Typical commands:
  ```bash
  cmake -S . -B build -DUSE_CUDA=OFF
  cmake --build build -j
  cmake --build build --target test

  cmake -S . -B build-cuda -DUSE_CUDA=ON
  cmake --build build-cuda -j
  cmake --build build-cuda --target test
  cmake --build build-cuda --target cuda-smoke
  ```
- Python stays outside the core build: `.venv`, `pip install -e . --no-deps`,
  `pytest tests/`.

## 2. Layout and naming

```
csrc/
  core/      dtype, tensor, allocator, backend/error, device_info, rng,
             timer, json, jsonl, config
  ops/       matmul, cast, elementwise/bias/residual, layernorm, gelu,
             softmax, cross_entropy, attention, sigreg, batchnorm,
             optim/adamw, reductions, row_stats
  models/    vit, projector, lejepa model, probe, builder, checkpoint,
             param registry
  data/      packed dataset reader, sampler, augmentation, replay
  train/     training loop, scheduler, metrics, diagnostics, probe loop
  planner/   LeWM only
  main.c     CLI
tests/
  test_helpers.h/.c, test_main.c, test_suite.h
  unit/      per-op C tests
  parity/    golden dumps vs pysrc
  data/      generated golden blobs (small, reproducible)
bench/
  harness.h/.c, bench_sigreg.c, bench_vit.c, bench_train.c, results/
tools/
  pack_dataset.py, dump_reference.py, convert_checkpoint.py,
  compare_c_py.py, analyze_run.py
```

The protocol's `common/` and `kernels/` names are replaced by fBLT's `core/`
and `ops/`, as explicitly allowed by protocol 8.1. The empty placeholder
directories are removed when real sources land.

- Public C symbols use `fj_`; public macros/types use `FJ_`.
- Headers use `#ifndef FJ_<MODULE>_<NAME>_H` and `extern "C"` guards.
- Files are `.h` + `.c` or `.cu`; CPU fallbacks are `*_cpu.c`.
- Keep fBLT names out of the API. No `blt_*` symbol is exported from
  `fj_core`; adaptation is recorded once in `docs/reuse.md`.
- One module per file. Static helpers stay file-local. Public headers live next
  to the implementation.
- No per-file provenance headers. The pinned fBLT commit lives in
  `docs/reuse.md`.

## 3. Error checking and debug mode

- `FJ_FATAL(...)`, `FJ_WARN(...)`, `FJ_REQUIRE(cond, ...)` mirror fBLT's
  `BLT_*` macros and terminate via `exit(EXIT_FAILURE)`.
- `FJ_CHECK_CUDA(call)` checks a CUDA API return value.
  `FJ_CHECK_LAST_ERROR(what)` checks `cudaGetLastError()` after a launch.
- No silent unchecked launch. Every `.cu` file ends a launch with a check.
- `FJ_DEBUG=1`:
  - synchronize after every launch (`cudaDeviceSynchronize`),
  - scan op outputs for NaN/Inf,
  - check arena bounds and tensor contiguity at op entry.
  Release mode checks launch errors only.
- `FJ_DETERMINISTIC=1`:
  - `CUBLAS_PEDANTIC_MATH`, fixed GEMM algorithm selection,
  - fixed-order reductions (no atomics),
  - CPU path is always deterministic.
  Tests use it; benchmarks report it separately.
- fBLT has diagnostic scans in `train_diag.c` but no global debug-sync mode.
  FastJEPA adds the global mode; reuse the diagnostic scan style.

## 4. Tensors, arenas and ownership

- `fj_tensor`: `data`, `shape[4]`, `strides[4]`, `ndim`, `numel`, `dtype`,
  `backend`, `is_view`. Maximum rank is 4.
- Row-major strides by default. Most fBLT ops assume contiguous storage; every
  FastJEPA op therefore checks `is_contiguous` or uses explicit strides.
- `fj_arena`: bump allocator over a host or device buffer, 64-byte aligned,
  no individual free, reset per step. A named/tagged workspace layer tracks
  high-water marks per stage.
- `fj_tensor_create` zero-initializes; `fj_tensor_create_uninit` does not.
- Host bookkeeping for device arenas uses `container_alloc` so containers never
  land in device memory.
- Transfers use `fj_tensor_upload` / `fj_tensor_download`; raw `memcpy` is only
  valid for host tensors.
- A device scratch arena may be sized with `FJ_SCRATCH_MB`; it is reset between
  steps and must not hold live state.
- `csrc/` must not include PyTorch, NumPy, timm or Python headers.

## 5. Dtypes and precision

| Dtype | fBLT at audit | FastJEPA |
|---|---|---|
| FP32 | reference path | reference path |
| BF16 | mixed-precision matmul operands; FP32 masters | supported on sm_80+ (RTX 4060) |
| FP16 | absent | added for T4 (`sm_75`) |
| UINT8 | byte data | image data, labels |
| INT32 | present | class labels, indices |
| FP8 E4M3 | cuBLASLt path | not planned for Track A |

Rules:

- FP32 is the reference. TF32 is off by default; a TF32 experiment gets a
  separate config flag and a parity check.
- SIGReg statistics, softmax, LayerNorm, BatchNorm running stats,
  cross-entropy and all reductions run in FP32 regardless of compute dtype.
- Mixed precision keeps FP32 master weights and optimizer state, plus a
  compute-dtype shadow. GEMMs accumulate in FP32; weight gradients accumulate
  in FP32.
- FP16 uses dynamic loss scaling (init 65536, growth 2 every 2000 good steps,
  backoff 0.5 on non-finite, skip on overflow). BF16 needs no scaler.
- fBLT BF16 casts only forward matmul operands and the backward input-gradient
  GEMM; weight-gradient GEMMs and the optimizer stay FP32. FastJEPA starts from
  that path, then measures whether a wider AMP path is needed for PyTorch
  autocast parity.
- Audit finding: the fBLT device table lists the T4 at 65 TFLOP/s in the BF16
  column. Turing (`sm_75`) has no BF16 tensor cores; that number is FP16. In
  FastJEPA the table is corrected and BF16 availability is queried, not
  inferred from the table.
- SIGReg projection dtype is explicit: default FP32; `sigreg_proj_dtype=compute`
  reproduces PyTorch autocast. Log the choice.

## 6. Ops: what fBLT provides and what changes

| Op | fBLT file | fBLT status at audit | FastJEPA action |
|---|---|---|---|
| GEMM | `ops/matmul.h`, `ops/linalg.cu`, `ops/linalg_cpu.c` | 2-D row-major; FP32 `cublasSgemm`, BF16 `cublasGemmEx`, FP8 cuBLASLt; no bias/activation epilogue; no batched/strided GEMM; no FP16 | adapt: add FP16 + bias epilogue; keep the existing cuBLAS path as the first implementation and evaluate cuBLASLt only with a benchmark (D1); keep 2-D `[M,K]@[K,N]` |
| Attention | `ops/attn_core.h/.cu`, `models/attention.c` | single-head raw-buffer core; naive fwd, optional flash fwd via `BLT_FLASH_ATTN=1`; additive mask or causal; fwd stores weights/LSE; backward implemented; FP32 only | adapt: full bidirectional (`mask=NULL`, `is_causal=0`), batch/head wrapper, ViT head dim 64, optional FP16/BF16 |
| Elementwise | `ops/elementwise.h/.cu/_cpu.c` | add/mul/scale/scaled_copy, FP32 only | adapt: dtype support, bias/residual add, dropout mask |
| GELU | `ops/gelu.h`, `elementwise_cpu.c` | tanh approximation only, fwd/bwd | rewrite exact `erf` GELU; keep tanh as config option |
| LayerNorm | `ops/layernorm.h`, `reductions_cpu.c`, `reductions.cu` | 2-D FP32 forward only; biased variance; eps >= 1e-12; **backward calls `BLT_FATAL`** | adapt fwd; write backward; affine weight+bias; support flattened rows |
| RMSNorm | `ops/rmsnorm.h`, reductions | fwd/bwd/add-bwd | keep for Track B and LeWM; not used by ViT |
| Softmax | `ops/softmax.h`, reductions | last-dim FP32 fwd/bwd | reuse/adapt |
| Cross-entropy | `ops/cross_entropy.h`, reductions | 2-D logits, UINT8 targets, mean over rows; fwd/bwd | adapt: INT32 labels for the probe; keep FP32 stats |
| Optimizer | `ops/optim.h/.cu/_cpu.c`, `train_optim.h/.c` | per-tensor SGD/AdamW; decoupled weight decay; selective norm decay default; grad clip | adapt: per-parameter-group lr/wd, decay-all parity mode, FP16 scaler |
| Cast | `ops/cast.h/.cu/_cpu.c` | FP32<->BF16, FP8; CPU bit-exact BF16 | adapt: add FP16 |
| Row stats | `ops/row_stats.h/.cu/_cpu.c` | entropy and argmax | reuse for diagnostics |
| Reductions | `ops/reductions.*` | softmax/CE/LN/RMSNorm kernels | adapt as above |
| SIGReg | none | — | rewrite (A2) |
| BatchNorm1d | none | — | rewrite (A3 projector) |
| Patch embedding | none | — | rewrite (im2col + GEMM) |
| CLS/pos embed | none | — | rewrite |
| DropPath | none | — | rewrite |
| Image augmentation | none | — | rewrite |
| Probe | none | — | rewrite |
| CUDA event timer | none | host `CLOCK_MONOTONIC` only | add to bench harness |
| Philox RNG | none | host splitmix64 helpers only | add counter-based RNG |
| Global debug/NaN mode | none | diagnostic scans only | add |

Important fBLT details confirmed at audit:

- `blt_matmul` is row-major `C = A @ B` mapped to cuBLAS with swapped operands;
  `beta = 0` overwrites the output. Accumulation must happen in a wrapper.
- `blt_matmul_backward` also overwrites `grad_a` / `grad_b`; it does not
  accumulate.
- `blt_attention_head_core` is a raw-buffer single-head primitive. Passing
  `mask = NULL` and `is_causal = false` gives full bidirectional attention.
  Backward accumulates into `grad_q/k/v`.
- `BLT_FLASH_ATTN=1` selects the tiled forward kernel; it still writes
  post-softmax weights for backward. It is not a memory-saving flash backward.
- LayerNorm forward uses biased variance and `eps` inside `sqrt(var + eps)`;
  backward is not implemented.
- GELU is the tanh approximation; the ViT needs exact `erf` GELU.
- Cross-entropy targets are `UINT8`; the probe needs `INT32` class labels.
- fBLT's default optimizer skips weight decay on norm weights; FastJEPA's
  parity mode must decay every encoder+projector parameter like pysrc.

## 7. Models, checkpoints and configs

- Parameters are registered in a stable, named flat order matching the
  PyTorch `state_dict`. `param_visitor` provides weight/grad pairs plus
  `is_norm`, `decay`, group and layer metadata; reuse that pattern for the
  optimizer, diagnostics, BF16 refresh and checkpointing.
- Checkpoint container (fBLT v2): magic `FBLT`, version, metadata JSON,
  tensor count, then per tensor `u16 name_len`, name, `u32 ndim`,
  `u32 dims[]`, `f32 data[]`. Load validates every name and shape.
- FastJEPA container (`FJEP`, little-endian):
  ```
  magic "FJEP", u32 version
  u32 meta_len, char meta_json[meta_len]   // config, step, git, RNG counters
  u32 num_tensors
  per tensor: u16 name_len, char name[], u32 ndim, u32 dims[],
              u32 dtype, u64 nbytes, bytes data (row-major)
  ```
  `tools/convert_checkpoint.py` maps PyTorch `state_dict` names to `FJEP` and
  back. Optimizer state is a separate `FJEP`-style file or a metadata section.
- Configs are JSON files under `configs/`, parsed by `core/json`, with CLI
  overrides. fBLT's C trainer is CLI-first and its JSON parser is used by
  ablation sweeps; FastJEPA standardizes on JSON config + explicit overrides.
  The resolved config is written to the run directory.
- Run directory layout matches pysrc:
  - `run.json` — git commit, resolved config, device/driver/CUDA, timestamp,
    seed, BF16/FP16 capability.
  - `log.jsonl` — step and epoch records.
  - `ckpt.pt` / C `ckpt.fjep` equivalents, optimizer and scheduler state.
- JSONL training records follow pysrc:
  - step: `kind="step"`, `step`, `epoch`, `loss`, `lp`, `ls`, `probe_loss`,
    `lr`, `probe_lr`, `step_time_ms`, `imgs_per_s`.
  - epoch: `kind="epoch"`, `epoch`, mean `loss`/`lp`/`ls`,
    `step_time_ms_mean`, `imgs_per_s`, embedding diagnostics (`erank`, std,
    effective rank, etc.), `probe_acc`, `probe_val_loss`.
- fBLT's metrics stream uses `{"event":"train",...}`. FastJEPA uses the pysrc
  `kind` schema so C and Python logs can be compared directly.
- Device metadata fields follow `pysrc/common/device.py`: `backend`, `name`,
  `capability`, `total_memory_gb`, `multiprocessor_count`, `driver_version`,
  clocks, `cuda`, `cudnn` where available. C adds `cuda_runtime_version`,
  `cuda_driver_version` and BF16 support.

## 8. Tests

- `tests/test_main.c` holds arrays of `{name, fn}` cases grouped core / model /
  parity, prints `[PASS]`/`[FAIL]`, and returns nonzero on failure.
- `tests/test_suite.h` declares every case; `tests/test_helpers.h` provides
  `TEST_ASSERT`, `TEST_ASSERT_CLOSE`, `TEST_ASSERT_VOID`,
  `TEST_ASSERT_CLOSE_VOID`.
- fBLT golden format: `u32 ndim`, `u32 dims[ndim]`, then little-endian FP32
  values. The loader only supports FP32.
- FastJEPA replaces it with a manifest + raw blobs:
  `manifest.json` entries `{name, shape, dtype, file, atol, rtol}`; dtypes
  FP32/FP64/FP16/BF16/U8/I32; little-endian; row-major. This is required for
  FP64 gradchecks and label tensors.
- Parity data generation is Python-side (`tools/dump_reference.py`,
  `tests/py/...`) against pysrc, then consumed by C tests. Never compare
  through two independent RNGs; dump the inputs and directions.
- Tolerance conventions:
  - exact for BF16 bit rounding and integer/label tensors;
  - `1e-6` rel for FP64 finite differences;
  - `1e-4` rel typical FP32 fwd; `1e-3` rel typical FP32 bwd;
  - documented per entry in the manifest.
- Commands:
  ```bash
  cmake --build build --target test
  pytest tests/ -q
  compute-sanitizer --tool memcheck ./build-cuda/fj_test
  ```
- fBLT runs ASan+UBSan in CI and `compute-sanitizer` locally. FastJEPA adopts
  both for CPU and CUDA changes.

## 9. Benchmarks

- fBLT `bench/harness.h/.c` provides:
  - host `CLOCK_MONOTONIC` timer (nanoseconds),
  - `bench_stats`: n, mean, population stddev, min, max, nearest-rank
    p50/p90/p99,
  - `bench_collect_samples`: discard one cold pass, then warmup, then timed
    iterations,
  - JSONL: `timestamp`, `phase`, `name`, `tag`, `latency{...}`,
    `metrics{...}`.
- fBLT's C harness has **no CUDA event timer**; CUDA benchmarks synchronize
  internally and time host wall-clock. Protocol 5.7 requires device-side
  timing. FastJEPA extends the harness with `cudaEvent` timing and an explicit
  `cudaEventSynchronize`.
- FastJEPA JSONL aligns with `bench/harness.py`:
  `schema`, `kind="bench"`, `bench`, ISO `timestamp`, `git_commit`, `device`,
  `config`, `timer` (`cudaEvent` or `perf_counter`), `iters`, `warmup`,
  `median_ms`, `p25_ms`, `p75_ms`, `iqr_ms`, `mean_ms`, `std_ms`, `min_ms`,
  `max_ms`, plus domain metrics.
- Rules: discard warmup; at least 20 timed iterations; report median and IQR;
  run 3 times and report the median for performance claims; record the exact
  build command, commit and hardware.
- Every performance claim needs a script in `bench/` and raw JSONL. No claim
  from a one-off run.

## 10. Style

- `.clang-format`: LLVM base, 4-space indent, 120-column limit, attached
  braces, pointer aligned right, short functions/ifs/loops allowed on one line.
- Comments explain the math (protocol section references) and the memory
  layout of every non-trivial kernel. State units in names when ambiguous.
- No dead code, no commented-out blocks, no silent deviations.
- Small focused commits; each commit states the protocol section it implements.
- Do not commit on the owner's behalf.

## 11. Environment and verification

- The agent sandbox used for this audit has no `cmake`, `nvcc` or
  `nvidia-smi`. It does have PyTorch `2.15.0+cu132`, CUDA `13.2` runtime and
  timm `1.0.30` in `.venv`, which were used for architecture checks only.
- All CUDA build/test/bench commands must run on the RTX 4060 machine
  (sm_89) or Kaggle T4 (sm_75), and the outputs must be pasted into the
  experiment log.
- Record the exact toolchain in `run.json`: `nvcc --version`, `cmake
  --version`, `gcc --version`, driver version, CUDA runtime/driver version,
  device properties, BF16/FP16 support, git commit.
- Do not infer BF16 support from the architecture name alone; query
  `cudaDeviceGetAttribute(cudaDevAttrBf16Conversion...)` or the compute
  capability and log the result.

## 12. A0 resolutions

- BLT path: `~/projects/fast_BLT`; `docs/reuse.md` updated.
- BLT commit pinned: `72deb5cd749bae0fba2625210f0841c7eaae77c8`.
- Directory names: `csrc/core` and `csrc/ops` confirmed.
- Reuse mechanism (D7): direct in-tree adaptation. The audited fBLT subset is
  copied into `csrc/` under `fj_*` names. There is no `third_party/fblt/`
  directory, no per-file provenance header and no license copy; the owner holds
  the fBLT copyright. No fBLT snapshot is built and no `blt_*` symbol enters
  `fj_core`.
- Config format: JSON; pysrc log schema; FJEP checkpoint container.
- Bench harness: fBLT statistics/JSONL pattern, extended with CUDA events and
  the `bench/harness.py` schema.

Still open (section 9 of the plan): D1 GEMM backend details, D2 first parity
target, D5 multi-GPU, checkpoint interop, online probe, and mixed-precision
residual-stream semantics.
