# FastJEPA

FastJEPA is a from-scratch C/CUDA project around the Joint Embedding Predictive
Architecture (JEPA) family. It has three tracks:

1. **Track A (foundation):** a C/CUDA implementation of LeJEPA training (ViT
   encoder + SIGReg regularizer) that is measurably faster than the official
   PyTorch reference under a documented benchmark.
2. **Track B (research):** a study of whether adding a JEPA-style latent
   prediction objective (with SIGReg) to a byte latent transformer improves
   data efficiency or representation quality.
3. **Track C (research):** a minimal JEPA world model trained from pixels, with
   a fast CUDA planner (CEM/MPPI) operating in latent space on a single
   consumer GPU.

The project targets small models on small datasets on a single consumer GPU
(NVIDIA RTX 4060, 8 GB). Python/PyTorch code lives in `pysrc/` as the ground
truth and benchmark baseline; the training binary in `csrc/` does not depend on
PyTorch.

## Status

Early setup stage. Repository layout and reuse audit are in place. Core kernels,
models, training loop, and benchmarks are not yet implemented. See `docs/` for
conventions and the reuse audit.

## License

Apache 2.0 (see `LICENSE`).
