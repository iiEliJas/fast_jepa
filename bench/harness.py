"""Shared benchmark harness for the Python baselines.

Everything in ``bench/`` writes one JSONL record per measurement so that our
implementation and the pinned official implementation can be compared in the
same format. Timing follows protocol section 5.7: warmup is discarded,
CUDA work is timed with ``torch.cuda.Event`` and an explicit synchronize, and
every record carries the git commit, device properties, driver, CUDA version
and the exact configuration.

The module depends on torch and this repository's ``common.device`` metadata
helper.  It must stay importable in a process that has the official ``lejepa``
package on ``sys.path`` as well as in one that has this repository's ``lejepa``
package.
"""

import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYSRC = os.path.join(REPO_ROOT, "pysrc")
DEFAULT_OFFICIAL_ROOT = os.environ.get(
    "LEJEPA_OFFICIAL_ROOT", os.path.expanduser("~/projects/lejepa")
)
SCHEMA_VERSION = 1

sys.path.insert(0, PYSRC)
from common.device import device_info as _device_info  # noqa: E402
from common.device import resolve_device  # noqa: E402


def git_commit(path=REPO_ROOT):
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def has_official(root=DEFAULT_OFFICIAL_ROOT):
    root = os.path.abspath(os.path.expanduser(root))
    return os.path.isdir(os.path.join(root, "lejepa"))


def device_info(device=None):
    """Device/software metadata plus the git commit for benchmark records."""
    info = _device_info(device)
    info["git_commit"] = git_commit()
    return info


def synchronize(device=None):
    if resolve_device(device).type == "cuda":
        torch.cuda.synchronize()


def _summarize(samples, timer):
    ordered = sorted(samples)
    quartiles = statistics.quantiles(ordered, n=4) if len(ordered) > 1 else [ordered[0]] * 3
    p25, _, p75 = quartiles
    return {
        "timer": timer,
        "iters": len(ordered),
        "median_ms": statistics.median(ordered),
        "p25_ms": p25,
        "p75_ms": p75,
        "iqr_ms": p75 - p25,
        "mean_ms": statistics.fmean(ordered),
        "std_ms": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def time_call(fn, warmup=20, iters=50, device="auto"):
    """Time ``fn`` and return median/spread in milliseconds.

    CUDA uses one event pair per iteration and synchronizes before reading the
    elapsed time, so each sample is a real device-side duration.  CPU falls
    back to ``time.perf_counter``.  Warmup calls are discarded.
    """
    device = resolve_device(device)
    for _ in range(warmup):
        fn()
    synchronize(device)

    samples = []
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(iters):
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end))
        timer = "cudaEvent"
    else:
        for _ in range(iters):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e3)
        timer = "perf_counter"

    result = _summarize(samples, timer)
    result["warmup"] = warmup
    return result


def base_record(bench, device=None, config=None, **fields):
    """Build a record with the fields every benchmark result must carry."""
    device = resolve_device(device)
    record = {
        "schema": SCHEMA_VERSION,
        "kind": "bench",
        "bench": bench,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "device": device_info(device),
        "config": config or {},
    }
    record.update(fields)
    return record


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return path


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_module_from_path(name, path):
    """Load a module by file path under a unique name.

    The repository package is called ``lejepa``, the same name as the official
    package.  Loading our modules by path lets a benchmark import both without
    fighting over ``sys.modules["lejepa"]``.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_ours_sigreg():
    return load_module_from_path("fj_sigreg_bench", os.path.join(PYSRC, "lejepa", "sigreg.py"))


def load_official_sigreg(official_root=DEFAULT_OFFICIAL_ROOT):
    """Import the pinned official SIGReg from a checkout.

    ``sys.path`` is prepended so the official package wins even when this
    repository is installed in editable mode.  The import is verified instead
    of assumed.
    """
    root = os.path.abspath(os.path.expanduser(official_root))
    if not os.path.isdir(os.path.join(root, "lejepa")):
        raise FileNotFoundError(f"official lejepa checkout not found at {root}")
    if "lejepa" in sys.modules:
        loaded = os.path.abspath(getattr(sys.modules["lejepa"], "__file__", ""))
        if not loaded.startswith(root + os.sep):
            raise RuntimeError(
                f"lejepa is already imported from {loaded}; run official benchmarks "
                "in a fresh process or before importing the local package"
            )
    sys.path.insert(0, root)
    from lejepa.multivariate import SlicingUnivariateTest
    from lejepa.univariate import EppsPulley

    loaded = os.path.abspath(sys.modules["lejepa"].__file__)
    if not loaded.startswith(root + os.sep):
        raise RuntimeError(f"imported lejepa from {loaded}, expected {root}")
    return SlicingUnivariateTest, EppsPulley


def format_table(records, columns):
    """Render records as a markdown table; ``columns`` is (key, header) pairs."""
    header = "| " + " | ".join(title for _, title in columns) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, rule]
    for record in records:
        cells = []
        for key, _ in columns:
            value = record.get(key, "")
            if value is None:
                value = ""
            elif isinstance(value, float):
                value = f"{value:.4f}"
            cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
