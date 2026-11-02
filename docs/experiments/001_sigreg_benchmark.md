# Experiment 001: SIGReg and training benchmarks (Python)

Date: 2026-11-02

## Setup
- Device: as reported by torch.cuda (this run: CPU fallback? check local)
- Our impl: `pysrc/lejepa/sigreg.py` (SlicedEppsPulley)
- Official: `lejepa` EppsPulley + SlicingUnivariateTest (`~/projects/lejepa`)
- Protocol: 20 timed runs after 5 warmups, cuda sync where applicable.

## SIGReg forward latency (ms)

| N | ours | official |
|---|------|----------|
| 256 | 6.285 | 7.136 |
| 512 | 11.690 | 13.502 |
| 1024 | 30.099 | 24.729 |

## SIGReg forward+backward latency (ms)

| N | ours | official |
|---|------|----------|
| 256 | 18.563 | 24.843 |
| 512 | 37.610 | 45.998 |
| 1024 | 83.244 | 87.859 |

## Training step (V=2, B=8, 32x32, 2-layer ViT)

| metric | value |
|--------|-------|
| throughput | 367.3 imgs/s |
| step time | 43.57 ms |

## Reproduce
```bash
PYTHONPATH=pysrc .venv/bin/python tools/bench_sigreg.py
PYTHONPATH=pysrc .venv/bin/python tools/bench_sigreg_backward.py
PYTHONPATH=pysrc .venv/bin/python tools/bench_train.py
PYTHONPATH=~/projects/lejepa .venv/bin/python /tmp/opencode/bench_official.py
PYTHONPATH=~/projects/lejepa .venv/bin/python /tmp/opencode/bench_official_bwd.py
```
