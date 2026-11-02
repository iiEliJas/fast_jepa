# Conventions (from BLT, `~/projects/fast_BLT`, commit to be pinned)

## Build
- CMake, C99 host, CUDA C++17 when `USE_CUDA=ON`. Options `USE_CUDA`. Default arch 89; override `-DCMAKE_CUDA_ARCHITECTURES`.
- Global include dirs `csrc tests`. C sources listed explicitly in CMakeLists (no globbing).
- Targets: `test`, CPU/CUDA. Python via `pip install -e .` + pyproject.

## Naming
- Prefix `blt_` for public C functions/types; this project uses `fj_` consistently. Keep BLT names only when reusing verbatim under `third_party/`.
- Headers: `#ifndef BLT_<MODULE>_<NAME>_H` / `#define ...`, `extern "C"` guards.
- Files: `.h` + `.c`/`.cu`; CPU fallbacks suffixed `_cpu.c` (e.g. `elementwise_cpu.c`).

## Error checking
- `BLT_FATAL(...)` for CUDA launch/malloc/copy failures with `cudaGetErrorString`.
- Post-launch `cudaGetLastError()` check pattern in `.cu` files.
- No silent unchecked launches.

## Tensors / memory
- `blt_tensor` handles with arena allocator (`core/allocator.h`). Shapes/strides in `core/tensor.h`.
- CPU reference implementations alongside CUDA (`*_cpu.c`).

## Tests
- C tests in `tests/`, helpers in `tests/test_helpers.h`: `TEST_ASSERT`, `TEST_ASSERT_CLOSE`, golden binary tensor loading via `g_test_data_dir`.
- Run `cmake --build build --target test`.

## Benchmarks
- `bench/harness.h`: timer, stats (mean/stddev/min/max/p50/p90/p99), `bench_collect_samples`, JSONL output.
- Warmup discarded; >=20 iterations recommended.

## Style
- `.clang-format`, `.editorconfig` present; `-Wall -Wextra`.

## Config / logging
- JSON configs; JSONL results. Python package wraps C binary.

## Reuse note
- BLT is single-GPU, FP32 only, no BF16. Our project needs FP16/BF16 and SIGReg; those are new.
