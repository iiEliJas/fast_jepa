# Reuse audit (WP A0)

BLT repo: `~/projects/fast_BLT`. Pin exact commit hash before reuse
(`git -C ~/projects/fast_BLT rev-parse HEAD`).

## Decisions

| Component | Needed for | Fit | Decision | Notes |
|---|---|---|---|---|
| Build system (CMake, C99/CUDA17, arch flags) | all | direct | reuse as-is | follow CMake structure; add sm_75+sm_89 |
| Error-check macros / cuda shim | all | direct | reuse/adapt | BLT uses BLT_FATAL + cudaGetLastError; adopt same pattern |
| Device queries / backend | all | direct | reuse | `core/backend.c`, `core/platform.h` |
| Tensor + arena allocator | all | direct | reuse | `core/tensor.h`, `core/allocator.h` |
| JSON config parsing | all | direct | reuse | `core/json.h` |
| Timers / benchmark harness | 5.7 | direct | reuse | `bench/harness.h` (clock_gettime based; add cudaEvent timing) |
| Test harness (golden binary tensors, tolerances) | 9 | direct | reuse | `tests/test_helpers.h` |
| GEMM wrappers (cuBLAS/cuBLASLt) | ViT/projector | audit | adapt | check decision D1; BLT matmul interface in `ops/matmul.h` |
| Attention fwd/bwd | ViT, predictor, Track B | adapt | adapt | BLT causal/block-causal masks; ViT needs full bidirectional; head dim 64 |
| FFN | ViT/predictor | adapt | adapt | BLT SwiGLU; add GELU variant + backward |
| Normalization | ViT/LeWM | adapt | adapt | BLT RMSNorm; add LayerNorm (affine+non-affine) |
| Optimizer AdamW, schedules, grad clip, loss scaling | A6, Track C | direct/adapt | adapt | check per-group lr, per-step vs per-epoch |
| Checkpointing, JSONL logging, config parsing | all | direct/adapt | adapt | config format vs `configs/` |
| Multi-GPU NCCL | D5 | audit | audit | check if present (BLT is single-GPU) |
| Byte-stream data loader, CE, entropy patching, local enc/dec | Track B | direct | reuse (it's the codebase) | Track B builds on BLT |
| SIGReg fwd/bwd | A2 | new | rewrite | no BLT equivalent |
| BatchNorm1d fwd/bwd | projector | new | rewrite | no BLT equivalent |
| Patch embedding, CLS, pos emb (+interp) | ViT | new | rewrite | no BLT equivalent |
| AdaLN-zero blocks | LeWM predictor | new | rewrite | no BLT equivalent |
| GPU image augmentation | A5 | new | rewrite | no BLT equivalent |
| Image/trajectory data pipelines (HDF5) | A5/C | new | rewrite | no BLT equivalent |
| CEM/MPPI planner kernels | C | new | rewrite | no BLT equivalent |

## Reuse rules
1. Audit first; record commit hash + decision + tests.
2. No silent copies; choose submodule/subtree/shared lib/recorded copy (D7).
3. Re-validate in project shapes/dtypes (protocol 9.7).
4. Reuse discipline too.
5. Benefits measured.

## Known gaps to verify
- BLT has no BF16 path (FP32 only) -> implement per protocol 4.
- BLT is single process -> NCCL data-parallel is new if needed.
- BLT kernels tuned for long causal sequences; re-benchmark for 256-token ViT attention and 4-token predictor sequences.
