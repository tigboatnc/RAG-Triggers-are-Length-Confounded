"""System + memory reporter (v3, CUDA-only).

Used at the start of every phase script. Returns a dict and prints a
human-readable block. Captures OS, arch, CPU, RAM, GPU + VRAM, and the
versions of torch / transformers / bitsandbytes / accelerate so every run
is reproducible.

Mac/Metal probes were stripped in v3; if you need them, see _AGENT_v2.md
or restore from git history of sysinfo.py.

Usage (standalone):
    python scripts/sysinfo.py

Usage (from a phase script):
    from sysinfo import report
    info = report()  # prints + returns dict
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any


def _safe_run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=5, check=False)
        if out.returncode == 0:
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    return None


def _gpu_info() -> dict[str, Any]:
    info: dict[str, Any] = {"backend": "cpu", "details": None}
    if not shutil.which("nvidia-smi"):
        return info
    out = _safe_run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return info
    info["backend"] = "cuda"
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        name, total, used, free, driver = parts
        gpus.append(
            {
                "name": name,
                "memory_total_mb": int(total),
                "memory_used_mb": int(used),
                "memory_free_mb": int(free),
                "driver": driver,
            }
        )
    info["details"] = gpus
    return info


def _torch_cuda_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False, "reason": "torch not installed"}
    out: dict[str, Any] = {
        "torch_version": torch.__version__,
        "available": bool(torch.cuda.is_available()),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        out["device_name"] = torch.cuda.get_device_name(0)
        out["cuda_version"] = torch.version.cuda
        out["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    return out


def _ram_info() -> dict[str, float]:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {
            "total_gb": round(vm.total / 1024**3, 2),
            "available_gb": round(vm.available / 1024**3, 2),
            "used_gb": round(vm.used / 1024**3, 2),
            "percent": vm.percent,
        }
    except ImportError:
        return {"error": "psutil not installed"}


def _process_rss_gb() -> float | None:
    try:
        import psutil

        return round(psutil.Process(os.getpid()).memory_info().rss / 1024**3, 3)
    except ImportError:
        return None


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version(name)
        except PackageNotFoundError:
            return None
    except ImportError:
        return None


def report(print_block: bool = True) -> dict[str, Any]:
    info: dict[str, Any] = {
        "os": platform.system(),
        "release": platform.release(),
        "arch": platform.machine(),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "ram": _ram_info(),
        "gpu": _gpu_info(),
        "torch_cuda": _torch_cuda_info(),
        "process_rss_gb": _process_rss_gb(),
        "packages": {
            "torch": _pkg_version("torch"),
            "transformers": _pkg_version("transformers"),
            "accelerate": _pkg_version("accelerate"),
            "bitsandbytes": _pkg_version("bitsandbytes"),
            "safetensors": _pkg_version("safetensors"),
            "huggingface_hub": _pkg_version("huggingface_hub"),
            "sentence_transformers": _pkg_version("sentence_transformers"),
            "lancedb": _pkg_version("lancedb"),
            "datasets": _pkg_version("datasets"),
            "psutil": _pkg_version("psutil"),
            "numpy": _pkg_version("numpy"),
        },
    }
    if print_block:
        print("=" * 60)
        print("SYSINFO")
        print("=" * 60)
        print(json.dumps(info, indent=2))
        print("=" * 60)
    return info


def rss_delta_gb(baseline_gb: float | None) -> float | None:
    """Convenience: compute current RSS minus a previously-captured baseline."""
    cur = _process_rss_gb()
    if cur is None or baseline_gb is None:
        return None
    return round(cur - baseline_gb, 3)


def vram_used_mb() -> int | None:
    """Quick poll of GPU 0 used VRAM via nvidia-smi. None if unavailable."""
    if not shutil.which("nvidia-smi"):
        return None
    out = _safe_run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    if not out:
        return None
    try:
        return int(out.splitlines()[0].strip())
    except (ValueError, IndexError):
        return None


if __name__ == "__main__":
    report()
