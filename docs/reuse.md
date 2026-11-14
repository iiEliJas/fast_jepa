# Reuse audit (WP A0)

Status: refreshed 2026-11-07 against the real fBLT checkout. Every row in
protocol 8.3 has a concrete decision. Component-level tests listed here are the
ones that must pass in this project before a reused component is accepted.

## 1. Pinned references

| Project | Checkout | Commit | License | Use |
|---|---|---|---|---|
| Fast BLT (owner) | `~/projects/fast_BLT` | `72deb5cd749bae0fba2625210f0841c7eaae77c8` (`v1.0.1-34-g72deb5c`) | Apache-2.0 | source of build conventions, core/ops patterns, harnesses, selected implementations |
| Official LeJEPA | `~/projects/lejepa` | `c293d291ca87cd4fddee9d3fffe4e914c7272052` | CC BY-NC 4.0 (LICENSE file; `pyproject.toml` says MIT, LICENSE wins) | external parity and benchmark reference only |
| LeJEPA paper | `~/projects/paper/LeJEPA_PAPER.md` | — | paper | math and recipe reference |
| Official LeWM | `~/projects/le-wm` | `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` | MIT | external parity reference; closely ported into `pysrc/lewm/` with attribution |
| Python reference env | project `.venv` | — | — | PyTorch `2.15.0+cu132`, CUDA `13.2`, timm `1.0.30` |

Path correction: this file previously named `~/projects/fBLT` as the owner
checkout. That path does not exist on the owner machine. `AGENTS.md` already
names `~/projects/fast_BLT`, which is the real checkout. This document uses
`~/projects/fast_BLT` everywhere.

The official LeJEPA repository is non-commercial. It is executed as an
external baseline in a subprocess and is not copied into `csrc/` or `pysrc/`.
Do not vendor it without re-checking the license.

## 2. Reuse mechanism (D7)

**Decision: direct in-tree adaptation, no vendored snapshot.**

The audited fBLT subset is copied into `csrc/` and adapted under `fj_*` names.
The pinned commit in section 1 is the only provenance record. There is no
`third_party/fblt/` directory, no per-file provenance header and no bundled
license copy.

Rationale: the owner wrote fBLT and holds its copyright. Apache-2.0 imposes
notice obligations on licensees redistributing the work, not on the copyright
holder reusing their own code, and fBLT has no NOTICE file or third-party
source to carry forward. `fast_jepa`'s root `LICENSE` covers the combined
project. A duplicate license and a comment on every file would be bookkeeping
with no legal effect.

Why adaptation rather than a submodule/shared library:

- fBLT is a complete application with its own `blt_tensor`, model stack,
  patcher, inference paths and top-level CMake. Building it as a library would
  force its types and targets into FastJEPA and duplicate state.
- The owner specifically wants to copy and tweak functions/kernels, not link a
  second framework.
- A full git submodule/subtree adds fetch and drift cost for code we will never
  use (byte patcher, local encoder/decoder, KV cache, diffusion decoder,
  speculation, inference).

Rules:

1. Copy only the audited functions/files listed below.
2. Rename `blt_`/`BLT_` to `fj_`/`FJ_`; do not export upstream names.
3. Record non-trivial semantic changes in the work-package notes.
4. Re-test in FastJEPA shapes/dtypes; fBLT's tests do not count.
5. If this project later gains outside copyright holders or vendors a
   third-party library, revisit this decision and add the required notices
   before merging.

If the owner later prefers a submodule/shared library, switch before A1 lands;
otherwise this is the mechanism for Track A.

## 3. Decisions

Legend: **reuse** = copy nearly verbatim + rename; **adapt** = copy and change
semantics/API/dtype/shape; **rewrite** = write new (no useful fBLT source);
**reference** = read for pattern only; **external** = run unchanged outside the
tree; **N/A** = not needed for Track A.

### 3.1 Protocol 8.3 rows

| Component | fBLT source @ `72deb5cd` | Decision | Required changes | Tests that validate it here |
|---|---|---|---|---|
| Build system, error macros, device queries, debug build | `CMakeLists.txt`, `csrc/core/backend.h`, `csrc/core/backend.c`, `csrc/core/device_info.h/.c`, `csrc/ops/smoke.cu` | adapt | `fj_core`/`fastjepa`/`fj_test` targets; `USE_CUDA` + `FJ_WITH_CUDA`; arch `75;89`; `FJ_GIT_COMMIT`; cublasLt link; real `fj_device_info` with driver/runtime/BF16 query; global `FJ_DEBUG` sync + NaN scan | CPU+CUDA configure/build; `fastjepa info`; debug NaN test; `cuda-smoke` |
| Timers, benchmark harness | `bench/harness.h`, `bench/harness.c`, `csrc/core/platform.h`, `bench/cuda_bench.c` | adapt | keep stats/JSONL; add cudaEvent timer + explicit sync; align schema with `bench/harness.py` (median/IQR, git/device/config, milliseconds); 3-run rule | harness self-test; SIGReg bench vs PyTorch `torch.cuda.Event`; schema round-trip |
| Test harness, golden dumps, tolerances, gradcheck | `tests/test_helpers.h/.c`, `tests/test_main.c`, `tests/test_suite.h`, `tests/py/golden.py` | adapt | keep `test_main` grouping and assertion macros; replace f32-only blob with manifest + dtype-aware loader (FP64/FP16/BF16/U8/I32); per-entry atol/rtol; FP64 finite differences | load/compare each dtype; corrupt-shape rejection; C-vs-pysrc op parity |
| GEMM wrappers | `csrc/ops/matmul.h`, `csrc/ops/linalg.cu`, `csrc/ops/linalg_cpu.c` | adapt | add FP16; keep FP32/BF16/FP8; add bias epilogue; keep the existing cuBLAS Sgemm/GemmEx path first and evaluate cuBLASLt only if measured (D1; fBLT's cuBLASLt path is FP8-only); keep row-major `[M,K]@[K,N]`; wrapper that accumulates instead of overwriting; no batched/strided GEMM in first release | ViT/projector shapes vs PyTorch; bias; fwd/bwd gradcheck; FP16/BF16 tolerance |
| Attention fwd/bwd | `csrc/ops/attn_core.h`, `csrc/ops/attn_core.cu`, `csrc/ops/attn_core_cpu.c`, `csrc/models/attention.c` | adapt | full bidirectional (`mask=NULL`, `is_causal=0`); batch/head wrapper; ViT head dim 64, T=65/257, H=3/6; FP32 first then optional FP16/BF16; keep naive and optional flash; no dropout | fwd/bwd vs PyTorch SDPA with fused attention off; flash vs naive; non-contiguous QKV views; head-dim 64 |
| FFN fwd/bwd | `csrc/ops/swiglu.h`, `csrc/ops/elementwise.h/.cu/_cpu.c` | adapt | fBLT has SwiGLU, not GELU-MLP; reuse elementwise infrastructure, rewrite exact `erf` GELU + bias + residual + dropout; tanh GELU only as config option | exact GELU vs `nn.GELU(approximate='none')`; MLP fwd/bwd; bias/residual |
| Normalization | `csrc/ops/layernorm.h`, `csrc/ops/reductions_cpu.c`, `csrc/ops/reductions.cu`, `csrc/ops/rmsnorm.h` | adapt | LayerNorm fwd exists (biased variance, eps in sqrt); **backward is `BLT_FATAL`** — write it; affine weight+bias; support flattened `[rows,d]`; keep RMSNorm for Track B and LeWM, not ViT | LN fwd/bwd vs PyTorch, eps 1e-6, gradcheck; RMSNorm only if used later |
| Optimizer, schedules, clipping, loss scaling | `csrc/ops/optim.h/.cu/_cpu.c`, `csrc/train_optim.h/.c`, `csrc/train_schedule.h/.c`, `csrc/train_diag.c` | adapt | keep per-tensor AdamW math; add parameter groups (encoder+projector, probe); decay-all parity mode; FP16 dynamic scaler; grad clip default off; per-step warmup+cosine; keep grad/update/component norm diagnostics | one step vs PyTorch; group lr/wd; scheduler lr grid exact; resume state; FP16 overflow skip |
| Checkpointing, JSONL logging, config parsing | `csrc/models/checkpoint.h/.c`, `csrc/models/param_visitor.h/.c`, `csrc/core/json.h/.c`, `csrc/train_blt_d.c` | adapt | FJEP container with dtype + metadata; PyTorch `state_dict` converter; stable param order; optimizer/scheduler/RNG state; JSON config + CLI overrides; pysrc `kind` log schema and `run.json` | save/load round-trip bitwise; PyTorch ↔ FJEP conversion; resume parity; log schema validation |
| Multi-GPU NCCL | none at audit | N/A | fBLT is single-GPU/single-process; defer until after A7 | none for Track A |
| Byte-stream loader, CE, entropy patching, local enc/dec | `csrc/models/local_*`, `patcher.*`, `hash_ngram.*`, `entropy*.*`, `ops/cross_entropy.h` | N/A (Track B) | Track B builds on fBLT later; do not port into Track A | Track B plan |

### 3.2 Additional fBLT components worth reusing

| Component | fBLT source @ `72deb5cd` | Decision | Required changes | Tests |
|---|---|---|---|---|
| Tensor + arena allocator | `csrc/core/tensor.h/.c`, `csrc/core/allocator.h/.c` | reuse/adapt | add FP16; named/tagged workspaces + high-water mark; contiguity check; 3-D/4-D views; same 64-byte arena semantics | tensor view/stride tests; arena OOM/alignment; H2D/D2H |
| Dtype helpers | `csrc/core/dtype.h` | adapt | add FP16; name/size/parse helpers | size/name round-trip |
| Backend dispatch + determinism | `csrc/core/backend.h/.c` | reuse/adapt | `fj_` names; `BLT_DISPATCH` → `FJ_DISPATCH`; keep `g_*_deterministic` pattern | CPU/CUDA dispatch; deterministic repeat |
| JSON parser | `csrc/core/json.h/.c` | reuse | arena-owned parse, fatal on malformed, typed accessors; our configs fit | existing JSON unit tests + parse both `configs/*.json` |
| Platform timing | `csrc/core/platform.h` | reuse | monotonic clock; keep Windows/POSIX shim (not required for Linux) | timer monotonicity |
| Device peak table / MFU/MBU | `csrc/core/device_info.h/.c`, `csrc/core/flops.h/.c` | reuse | 4060/T4 already present; log unknown-device warning; add real device props separately | table lookup, longest-match, MFU/MBU formulas |
| Cast FP32↔BF16/FP8 | `csrc/ops/cast.h`, `csrc/ops/cast_cpu.c`, `csrc/ops/cast_cuda.cu` | adapt | add FP16; keep CPU bit-exact BF16 as ground truth | BF16 bit-exact round-trip; FP16 max error; CUDA parity |
| Elementwise add/mul/scale | `csrc/ops/elementwise.h/.cu/_cpu.c` | adapt | dtype support; bias/residual/dropout fused forms | fwd/bwd vs PyTorch |
| Softmax | `csrc/ops/softmax.h`, reductions | reuse/adapt | FP32 stats; dtype support optional | last-dim fwd/bwd vs PyTorch |
| Cross-entropy | `csrc/ops/cross_entropy.h`, reductions | adapt | add INT32 labels (probe); keep UINT8 for byte path; mean reduction; fwd/bwd | 10-class and 256-class vs PyTorch |
| Row stats | `csrc/ops/row_stats.h/.cu/_cpu.c` | reuse | entropy/argmax for diagnostics; no probe reuse | entropy/argmax vs NumPy/PyTorch |
| Param visitor | `csrc/models/param_visitor.h/.c` | adapt | stable ordering, groups, `is_norm`/`decay`, weight+grad pairs; use for optimizer/diagnostics/checkpoint/refresh | order matches checkpoint; visitor counts == tensor count |
| BF16 mixed precision | `csrc/models/model.c`, `cast.*`, `linalg.cu`, `docs/experiments/E010_bf16_mixed_precision.md` | adapt | keep FP32 masters/shadow; start with fBLT's operand-only BF16; extend only if parity requires; add FP16 scaler | BF16 fwd/bwd vs FP32; refresh; loss parity |
| Flash attention | `csrc/ops/attn_core.cu` (`BLT_FLASH_ATTN`) | reference/adapt | fwd-only tiled + LSE; still stores weights; benchmark for ViT T=257 before adopting | flash vs naive fwd/bwd; speed + memory bench |
| Diagnostics | `csrc/train_diag.h/.c` | adapt | activation/parameter NaN/Inf scans, component norms, grad/update norms; emit pysrc-style JSONL | diagnostic unit tests + a deliberately broken tensor |
| Bench CUDA patterns | `bench/cuda_bench.c` | reference | random fill, warmup, sync-before-timing, TFLOP/s/MFU | self-test |

### 3.3 New in FastJEPA (no fBLT equivalent)

| Component | Needed for | Decision | Reference | Tests |
|---|---|---|---|---|
| SIGReg fwd/bwd | A2 | rewrite | protocol 5.2–5.4, `pysrc/lejepa/sigreg.py`, official `EppsPulley` | same-A PyTorch parity, FP64 gradcheck, per-slice, benchmark |
| BatchNorm1d fwd/bwd | projector | rewrite | PyTorch `BatchNorm1d`, full `V*B` batch | train/eval fwd/bwd vs PyTorch, running stats |
| Patch embedding (im2col + GEMM) | ViT | rewrite | timm Conv2d patch embed with bias | vs PyTorch conv, bias, fwd/bwd |
| CLS token + learned pos embed | ViT | rewrite | timm state dict | load PyTorch weights, fwd parity |
| Exact GELU (erf) | ViT MLP | rewrite | PyTorch `nn.GELU(approximate='none')` | fwd/bwd |
| DropPath per-sample mask | ViT training | rewrite | timm `DropPath`, per-block `linspace(0,0.1,depth)` | deterministic replay, fwd/bwd |
| Image augmentation | A5 | rewrite | torchvision recipe, protocol 5.5 | replay dump, statistical tests |
| Packed dataset reader/sampler | A5 | rewrite | CIFAR-10/ImageFolder | decode/order parity |
| Online probe | A6 | rewrite | pysrc `probe` + CE | CE fwd/bwd, accuracy parity |
| FJEP ↔ PyTorch converter | A1/A7 | rewrite | pysrc `state_dict` | round-trip + load into both |
| Counter RNG (Philox) | A1/A5 | rewrite | counter-based design in plan 2.5 | reproducibility across launch shapes, distribution |
| CUDA event timer | A1/bench | rewrite | `bench/harness.py` | vs torch events |
| Global debug sync/NaN scan | A1 | rewrite | fBLT diagnostic style | broken-launch/NaN test |
| AdaLN-zero blocks | LeWM | rewrite | LeWM | later |
| CEM/MPPI planner | LeWM | rewrite | LeWM | later |
| HDF5 trajectory loader | LeWM | rewrite | LeWM | later |

### 3.4 Do not reuse

These fBLT modules are specific to the byte-latent transformer or its inference
paths and have no place in Track A:

- `csrc/models/local_encoder.*`, `local_decoder.*`, `local_common.*`
- `csrc/models/patcher.*`, `entropy*.*`, `hash_ngram.*`, `byte_embedding.*`
- `csrc/models/global_transformer.*` (decoder-only, RMSNorm, RoPE, SwiGLU)
- `csrc/models/block_diffusion.*`, `cross_attention.*`, `transformer.*`,
  `transformer_stack.*`
- `csrc/infer/*` (KV cache, speculation, block generation, stats)
- `csrc/core/decode_select.*`, `generate_greedy.*`
- `csrc/ops/mask_builder.*` (causal/block-causal masks), `patch_pool.*`,
  `rope.*`, `gather_scatter.*` (unless a later data path needs it)
- `csrc/train_sweep.c`, `csrc/tools/*`, `csrc/e2e_blt_dv.c`, `csrc/infer.c`

If Track B starts, it reuses the byte path directly from fBLT as its own
codebase, per the protocol.

## 4. Test obligations for every reused component

Protocol 9.7 applies: a component passing fBLT tests is not accepted here until
it passes FastJEPA shapes and dtypes.

1. **Op-level parity.** Random inputs, same tensors and same SIGReg directions
   as pysrc. Compare with the tolerance in the manifest.
2. **Finite-difference gradcheck** in FP64 on tiny shapes for every hand-written
   backward (SIGReg, LN, GELU, attention, BN, conv/patch embed, full tiny model).
3. **Shape matrix.** ViT: `rows = V*B` from 512 (CIFAR) to 8224 per microbatch
   (Imagenette), `K` 192/384/512/768/1536/2048, head dim 64, T 65/257.
4. **Dtype matrix.** FP32 reference; BF16 on 4060; FP16 on T4. Stats and
   reductions stay FP32.
5. **Determinism.** `FJ_DETERMINISTIC=1` repeated runs match; replay mode for
   first-N-step parity.
6. **Benchmark.** Every performance-related reuse gets a `bench/` entry and a
   raw JSONL result before a claim.
7. **Sanitizers.** ASan/UBSan on CPU; `compute-sanitizer` for new CUDA kernels.

## 5. Known gaps at the pinned commit

- No FP16 anywhere: dtype, cast, GEMM, optimizer, tests.
- No BatchNorm1d.
- LayerNorm backward is unimplemented (CPU and CUDA).
- GELU is tanh-only; ViT needs exact erf.
- Cross-entropy targets are UINT8 only; probe needs INT32 labels.
- No general RNG module; only host splitmix64/`fill_uniform` helpers.
- No CUDA event timer in the C bench harness; only host monotonic time.
- No global debug sync/NaN mode; only diagnostic scans.
- No image pipeline, augmentation, patch embedding, CLS/pos embed, DropPath.
- No online probe.
- No NCCL/multi-GPU.
- The fBLT device table lists the T4 at 65 TFLOP/s in the BF16 column. Turing
  (`sm_75`) has no BF16 tensor cores; that figure is FP16. FastJEPA treats the
  cell as FP16 and gates BF16 by compute capability / device attribute.
- fBLT's optimizer defaults to selective norm decay; pysrc decays every
  encoder+projector parameter. Parity mode must override this.
- fBLT's checkpoint stores FP32 tensors and fBLT model names; FastJEPA needs a
  dtype field, PyTorch name mapping and optimizer state.

## 6. A0 acceptance

- Every protocol 8.3 row has a decision above.
- Pinned commit is recorded; no fBLT files are vendored.
- The path discrepancy is resolved (`~/projects/fast_BLT`).
- D7 mechanism and directory names are recorded in `docs/conventions.md`.
- No code was added. A1 is the first implementation work package.

## 7. LeWM reuse

Status: added 2026-11-14. Reference checkout `~/projects/le-wm` @
`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`, MIT (`LICENSE` checked).

**Decision: direct in-tree adaptation, attribution in file docstrings.**

`jepa.py`, `module.py` and the CEM update rule are ported into `pysrc/lewm/`
and kept numerically and key-compatible with the reference, because the
released Push-T checkpoint must load with `strict=True`. The official checkout
is not vendored and is used only by the parity tests; `stable-worldmodel`,
`stable-pretraining`, Lightning and Hydra stay out of the runtime dependency
set.

| Component | Reference source | Decision | Changes | Tests |
|---|---|---|---|---|
| Encoder | HF `ViTModel` created by `stable_pretraining` `vit_hf` | rewrite (explicit) | explicit ViT-tiny/patch14 with transformers **4.x** parameter names; CLS output; bicubic position interpolation for non-224 inputs; SDPA replaced by explicit eager attention; FP16/BF16 softmax stays FP32, FP64 stays FP64 | `test_lewm_modules.py::test_encoder_matches_huggingface`, checkpoint strict load |
| SIGReg | `module.SIGReg` | reuse | subclass of `lejepa.sigreg.SlicedEppsPulley`; same buffers and statistic; shared direction matrix supported for tests | `test_sigreg_parity` (value and input gradient are bit-identical) |
| Predictor, AdaLN block, attention, FFN, embedder, MLP | `module.py` | reuse/adapt | same parameter names and double-LayerNorm quirk; `num_frames` exposed as an attribute; FP64 preserved for gradcheck | `test_predictor_parity`, `test_embedder_parity`, `test_mlp_parity` |
| JEPA forward, loss, rollout, cost | `jepa.py`, `train.py` | adapt | non-mutating `encode`/`rollout`; robust goal-embedding broadcast in `criterion`; same terminal-sum cost | `test_lewm_model.py` (forward, loss, rollout, cost, checkpoint) |
| CEM solver | `stable_worldmodel/planning/solver/cem.py` | adapt | explicit `horizon`/`action_dim` instead of gym `configure()`; no callbacks, no logging in the solver; identical sampling, elite update and candidate-0 mean forcing | `test_lewm_planner.py` (literal reference transcription) |
| HDF5 clip dataset | `stable_worldmodel/data/dataset.py` | rewrite | contiguous-episode fast path; exact `frameskip` windowing and action reshape; worker-safe HDF5 reopen | `test_lewm_data.py` |
| Trainer | `train.py` + `lewm.yaml` | rewrite | per-optimizer-step warmup+cosine, JSONL diagnostics, checkpoint with optimizer and SIGReg state | `test_lewm_train.py` |

Deliberate deviations from the pinned code:

1. **Explicit ViT instead of `transformers`.** The checkpoint uses 4.x module
   names; transformers 5 renamed them. The explicit encoder avoids a runtime
   dependency and is the specification for the future CUDA encoder. Numerics
   agree with transformers 5.x to `2e-6` (SDPA vs eager attention).
2. **FP64 is not downcast.** The reference casts pixels and action inputs to
   float32. We keep float64 so the protocol's finite-difference gradcheck
   covers the full model. FP32/BF16 behaviour is unchanged.
3. **`criterion` accepts batched and unbatched goal embeddings.** The pinned
   implementation assumes `(B, S, T, D)` and errors on the `(B, T, D)` shape
   produced by its own `get_cost` with the current world wrapper. Our version
   keeps the same terminal-sum-of-squares cost and broadcasts `(B, D)` /
   `(B, T, D)` correctly.
4. **Planner API.** `CEMSolver` takes `horizon` and `action_dim` directly and
   has no gymnasium/callback dependency. The numerical update is unchanged.
5. **Dataset indexing rejects non-contiguous episodes** instead of silently
   reading the wrong rows.

Validation: `.venv/bin/python -m pytest tests/test_lewm_*.py -q` (23 tests)
and the full suite (58 tests) pass on CPU. CUDA execution, dataset training
and planner success rates are still to be measured on the owner's 4060; see
`docs/experiments/003_lewm_parity.md`.
