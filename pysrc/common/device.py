"""Device and software-stack metadata shared by run logs and benchmarks.

Every record must say what hardware and driver produced it (protocol section
9).  Keeping one implementation avoids ``driver`` vs ``driver_version`` drift
between ``run.json`` and ``bench/results/*.jsonl``.
"""

import os
import platform
import shutil
import subprocess

import torch


def _nvidia_smi():
    if not shutil.which("nvidia-smi"):
        return {}
    query = "driver_version,clocks.sm,clocks.max.sm,clocks.mem"
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    fields = [field.strip() for field in out.stdout.strip().splitlines()[0].split(",")]
    if len(fields) != 4:
        return {}
    driver, clock_sm, clock_max, clock_mem = fields
    return {
        "driver_version": driver,
        "clock_sm_mhz": _as_int(clock_sm),
        "clock_max_sm_mhz": _as_int(clock_max),
        "clock_mem_mhz": _as_int(clock_mem),
    }


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def resolve_device(device="auto"):
    if isinstance(device, torch.device):
        return device
    if device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def device_info(device="auto"):
    """Return a JSON-serialisable description of the device and software stack."""
    device = resolve_device(device)
    info = {
        "backend": device.type,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(
            {
                "name": props.name,
                "capability": f"{props.major}.{props.minor}",
                "total_memory_gb": round(props.total_memory / 2**30, 2),
                "multiprocessor_count": props.multi_processor_count,
            }
        )
        info.update(_nvidia_smi())
    else:
        info["name"] = platform.processor() or "cpu"
        info["cpu_count"] = os.cpu_count()
    return info
