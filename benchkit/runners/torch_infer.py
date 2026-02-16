from __future__ import annotations

import time
from dataclasses import asdict

import torch

from benchkit.config import RunConfig
from benchkit.models.registry import get_model


def _resolve_device(requested: str) -> torch.device:
    req = requested.lower().strip()
    if req == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if req not in ("cpu", "cuda"):
        # allow things like "cuda:0"
        try:
            d = torch.device(req)
            if d.type == "cuda" and not torch.cuda.is_available():
                return torch.device("cpu")
            return d
        except Exception:
            return torch.device("cpu")
    return torch.device(req)


def run_inference(cfg: RunConfig) -> dict:
    device = _resolve_device(cfg.device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = get_model(cfg.model, cfg.input_shape).to(device)
    model.eval()

    c, h, w = cfg.input_shape
    x = torch.randn(cfg.batch_size, c, h, w, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(cfg.warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

    # Timed
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(cfg.steps):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_time = t1 - t0
    samples = cfg.batch_size * cfg.steps
    throughput = samples / total_time if total_time > 0 else 0.0
    latency_ms = (total_time / cfg.steps) * 1000.0 if cfg.steps > 0 else 0.0

    gpu = {}
    if device.type == "cuda":
        gpu = {
            "cuda_max_memory_allocated_bytes": float(torch.cuda.max_memory_allocated()),
            "cuda_max_memory_reserved_bytes": float(torch.cuda.max_memory_reserved()),
        }

    return {
        "config": asdict(cfg),
        "device_used": str(device),
        "total_time_s": total_time,
        "throughput_samples_per_s": throughput,
        "latency_ms_per_step": latency_ms,
        **gpu,
    }

