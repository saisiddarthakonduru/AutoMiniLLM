"""
System utilities: CPU/GPU detection, memory usage, size formatting, timing.
"""
import gc
import os
import platform
import time
from typing import Dict, Optional

import psutil
import torch


def detect_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' based on availability."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_label() -> str:
    dev = detect_device()
    if dev == "cuda":
        name = torch.cuda.get_device_name(0)
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
        return f"CUDA ({name}) – {vram_gb} GB VRAM"
    if dev == "mps":
        return "Apple Silicon GPU (MPS)"
    return f"CPU ({psutil.cpu_count(logical=True)} logical cores)"


def get_system_info() -> Dict:
    svmem = psutil.virtual_memory()
    dev = detect_device()
    info = {
        "device": dev,
        "device_label": device_label(),
        "python": platform_or("python_version", "3.x"),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": psutil.cpu_count(logical=True),
        "cpu_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(svmem.total / (1024 ** 3), 1),
        "available_ram_gb": round(svmem.available / (1024 ** 3), 1),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_ram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1)
        if torch.cuda.is_available()
        else None,
    }
    return info


def platform_or(attr: str = "python_version", default: str = "3.x") -> str:
    try:
        import platform

        return str(getattr(platform, attr)())
    except Exception:
        return default


def get_ram_usage_mb() -> float:
    """Current resident memory of this process in MB."""
    proc = psutil.Process(os.getpid())
    return round(proc.memory_info().rss / (1024 * 1024), 2)


def measure_model_ram_mb(model: torch.nn.Module) -> float:
    """Best-effort measurement of RAM a model's tensors occupy (weights + buffers)."""
    total_bytes = 0
    seen = set()
    for p in list(model.parameters()) + list(model.buffers()):
        if id(p) in seen:
            continue
        seen.add(id(p))
        total_bytes += p.numel() * p.element_size()
    return round(total_bytes / (1024 * 1024), 4)


def get_file_size_mb(path: str) -> float:
    if not path or not os.path.exists(path):
        return 0.0
    return round(os.path.getsize(path) / (1024 * 1024), 3)


def run_gc(device: str = None):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()


def timeit(fn, *args, warmup_ms=200, **kwargs):
    """Run `fn` for at least warmup_ms measuring per-call latency in ms."""
    # warmup
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) < warmup_ms / 1000.0:
        fn(*args, **kwargs)

    latencies = []
    t_end = time.perf_counter() + 0.6
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    if not latencies:
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    return {
        "mean_ms": round(sum(latencies) / len(latencies), 3),
        "median_ms": round(latencies[len(latencies) // 2], 3),
        "p95_ms": round(latencies[int(len(latencies) * 0.95) - 1], 3),
        "runs": len(latencies),
        "min_ms": round(latencies[0], 3),
        "max_ms": round(latencies[-1], 3),
    }