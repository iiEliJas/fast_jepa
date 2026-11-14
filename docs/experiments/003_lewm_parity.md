# Experiment 003: LeWM implementation parity (CPU)

Date: 2026-11-14

First implementation of LeWorldModel: an independent PyTorch port of
the official LeWM model, planner, data windowing and training recipe, checked
against the pinned reference on the laptop CPU with small tensors. This is a
correctness experiment, not a speed claim and not a training result. No dataset
was downloaded and no GPU was available.

## Setup

| Item | Value |
|---|---|
| Machine | laptop, Intel Core i5-8300H, 8 threads, ~3.8 GB RAM |
| GPU | none in this environment (`torch.cuda.is_available() == False`) |
| Python | 3.14.4 |
| PyTorch | 2.15.0+cu132 (CPU execution) |
| transformers | 5.23.0 (parity tests only) |
| einops | 0.8.2 (official checkout only) |
| h5py | 3.16.0 (synthetic dataset tests) |
| FastJEPA HEAD | `efa38630f9d49315eaea4eca2d4b83538f25ef50` plus the uncommitted working tree below |
| Official LeWM | `~/projects/le-wm` @ `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` (MIT) |
| Official checkpoint | `quentinll/lewm-pusht` `weights.pt` (72 MB, 303 keys) |

Working tree at measurement: `pysrc/lewm/{encoder,modules,model,planner,data,train}.py`,
`tests/test_lewm_{modules,model,planner,data,train}.py`,
`docs/reuse.md`, `pyproject.toml`, and one backward-compatible change in
`pysrc/lejepa/sigreg.py` (`in_dim` optional).

## Method

The official `module.py` and `jepa.py` are loaded by file path (not through
`sys.modules`) and given the same weights as our model. The official encoder
is HuggingFace `ViTModel`; transformers 5 renamed the ViT modules, so the test
translates the checkpoint keys (`encoder.encoder.layer` -> `encoder.vit.layers`,
`attention.attention.query` -> `attention.q_proj`, and so on) and loads with
`strict=True`. Our encoder keeps the 4.x names used by the released
checkpoint.

Comparisons:

1. **Module op parity**: `ARPredictor`, `Embedder`, `MLP`, `SIGReg` and the
   explicit `ViTEncoder` against the reference classes.
2. **Checkpoint parity**: load the released Push-T weights into our model with
   `strict=True`, into the reference model through the key translation, then
   compare `encode`, `predict`, both loss terms, `rollout` and `get_cost`.
3. **Gradient parity**: FP64 central differences on a tiny model
   (`embed_dim=32`, depth 1, 28x28 images) for encoder, predictor,
   projector and action-encoder parameters. AdaLN-zero is perturbed first,
   because at initialization its zero modulation makes all action-path
   gradients exactly zero.
4. **Planner parity**: the official CEM solve loop is transcribed literally in
   the test (sampling with a seeded `torch.Generator`, candidate 0 forced to
   the mean, top-k elite mean/std update). Our solver must produce the same
   mean and std.
5. **Data windowing**: synthetic HDF5 file with known row indices, verifying
   frames at `start + k*frameskip`, the action reshape
   `(num_steps, frameskip * action_dim)` and split semantics.
6. **Trainer**: one CPU run with 2 optimizer steps on synthetic 28x28 data,
   JSONL records and checkpoint round-trip.

All runs use fixed seeds. Commands:

```
.venv/bin/python -m pytest tests/test_lewm_modules.py -q
.venv/bin/python -m pytest tests/test_lewm_model.py -q
.venv/bin/python -m pytest tests/test_lewm_planner.py -q
.venv/bin/python -m pytest tests/test_lewm_data.py -q
.venv/bin/python -m pytest tests/test_lewm_train.py -q
.venv/bin/python -m pytest -q
```

## Results

```
tests/test_lewm_modules.py  5 passed
tests/test_lewm_model.py    6 passed
tests/test_lewm_planner.py  3 passed
tests/test_lewm_data.py     5 passed
tests/test_lewm_train.py    4 passed
total LeWM                23 passed in 11.69 s
full repository suite     58 passed in 40.23 s
```

Measured differences (max absolute, FP32, eval mode, 28x28 inputs):

| Quantity | Difference / tolerance |
|---|---|
| Encoder CLS embedding | `2.6e-6` (HF-5 SDPA reference vs our eager attention) |
| Action embedding | `0` (bit-identical) |
| Predictor output | `1.8e-6` |
| `pred_loss` (with checkpoint) | `< 1e-6` |
| `sigreg_loss` (same seed, same directions) | `< 1e-5` |
| Rollout `predicted_emb` | `< 1e-5` |
| `get_cost` | `< 1e-4` |
| CEM actions vs reference transcription | `< 1e-6` |
| FP64 gradcheck, 8 probes | `rtol=1e-4` |

Model size: 303 state-dict entries, 18,034,478 parameters, exact match with
the released checkpoint. SIGReg buffers and its value and input gradient are
bit-identical to `module.SIGReg` when the direction matrix is the same.

One reference quirk surfaced: the pinned `JEPA.criterion` expects `goal_emb`
of shape `(B, S, T, D)` and calls `expand_as`, but the pinned `get_cost`
produces `(B, T, D)`, so the two do not compose for the Push-T planning shapes.
Our `criterion` computes the same terminal sum-of-squares cost and broadcasts
`(B, D)`, `(B, T, D)` and `(B, S, T, D)`.

## Limitations

- CPU only; no CUDA kernel, no bf16/fp16 run and no memory measurement.
- Small 28x28 inputs for most comparisons. At 224x224 the encoder difference
  is `2.0e-6`, same order.
- Random or released weights, synthetic data; no real Push-T episode, no
  training run and no planning success rate yet.
- The official CEM is compared against a transcription, not against
  `stable_worldmodel` itself (not installed in this environment). The
  transcription was checked line by line against
  `stable_worldmodel/planning/solver/cem.py`.
- Training-step parity with dropout on and identical data order needs the main
  PC and the reference environment.

## Next steps on the main PC

1. `pytest` on the 4060; measure step time and peak memory for batch 128 at
   224x224 in bf16.
2. Self-collected Push-T subset: overfit test, then a short training run.
3. Load the official checkpoint into `pysrc/lewm`, run the CEM planner in the
   Push-T environment, and record success rate plus planning latency.
4. Only then start 10-epoch training on the official 13.1 GB dataset.

## Reproduction

```
git -C ~/projects/le-wm rev-parse HEAD   # 8edfeb336732b5f3ce7b8b210d0ba370a09e2cac
.venv/bin/python -m pytest tests/test_lewm_modules.py tests/test_lewm_model.py \
    tests/test_lewm_planner.py tests/test_lewm_data.py tests/test_lewm_train.py -q
```
