# Experiment Log

- 001 (2026-11-02): SIGReg forward/backward and synthetic training-step
  comparison against official LeJEPA on an RTX 4060; CUDA-event harness,
  JSONL raw results. -> `docs/experiments/001_sigreg_benchmark.md`
- 002 (2026-11-03): First full CIFAR-10 LeJEPA run (ViT-S, 100 epochs, bf16,
  RTX 4060); 67.21% online probe, 70.07% offline probe, 70.81% k-NN, no
  collapse, with rank/Gaussianity/invariance analysis. ->
  `docs/experiments/002_cifar10_lejepa_analysis.md`
- 003 (2026-11-14): LeWM PyTorch port parity on CPU; released Push-T
  checkpoint loads strict (303 keys, 18,034,478 params), encoder/predictor/loss
  differences <= 2.6e-6, FP64 gradcheck green, CEM matches the reference
  update rule, 23 LeWM tests and 58 total tests pass. No GPU, data or training
  yet. -> `docs/experiments/003_lewm_parity.md`
