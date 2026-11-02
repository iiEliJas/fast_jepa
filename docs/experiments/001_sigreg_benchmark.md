# Experiment 001: SIGReg latency comparison (Python)

Date: 2026-11-02

## Setup
- Device: as reported by torch.cuda
- Our impl: `pysrc/lejepa/sigreg.py` (SlicedEppsPulley)
- Official: `lejepa` EppsPulley + SlicingUnivariateTest (pinned repo `~/projects/lejepa`)

## Results (20 runs after warmup)

| N | ours (ms) | official (ms) |
|---|-----------|---------------|
| 256 | 6.285 | 7.136 |
| 512 | 11.690 | 13.502 |
| 1024 | 30.099 | 24.729 |

## Reproduce
```bash
PYTHONPATH=pysrc .venv/bin/python tools/bench_sigreg.py
PYTHONPATH=~/projects/lejepa .venv/bin/python /tmp/opencode/bench_official.py
```
