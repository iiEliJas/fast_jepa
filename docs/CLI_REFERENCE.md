# CLI Reference

Python ground truth entry points (run from repo root with `PYTHONPATH=pysrc`).

## Tests

```bash
PYTHONPATH=pysrc .venv/bin/python -m pytest tests/ -q
```

With official LeJEPA available for parity:

```bash
PYTHONPATH=~/projects/lejepa:pysrc .venv/bin/python -m pytest tests/ -q
```

## Training (skeleton)

```bash
PYTHONPATH=pysrc .venv/bin/python -c "from lejepa.train import train; train({'data_dir':'./data','epochs':1})"
```

## Ruff

```bash
PYTHONPATH=pysrc .venv/bin/ruff check .
PYTHONPATH=pysrc .venv/bin/ruff format --check .
```
