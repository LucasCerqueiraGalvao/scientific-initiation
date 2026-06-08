from __future__ import annotations

import random
from time import perf_counter

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_callable(
    fn,
    device: torch.device,
    warmup_iterations: int,
    measure_iterations: int,
) -> tuple[dict[str, float | int], torch.Tensor]:
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            _ = fn()

        synchronize_if_needed(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        timings_ms: list[float] = []
        last_output = None
        for _ in range(measure_iterations):
            synchronize_if_needed(device)
            start = perf_counter()
            last_output = fn()
            synchronize_if_needed(device)
            elapsed_ms = (perf_counter() - start) * 1000.0
            timings_ms.append(elapsed_ms)

    peak_memory_bytes = 0
    if device.type == "cuda":
        peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))

    metrics = {
        "mean_latency_ms": float(np.mean(timings_ms)),
        "p50_latency_ms": float(np.percentile(timings_ms, 50)),
        "p95_latency_ms": float(np.percentile(timings_ms, 95)),
        "peak_memory_bytes": peak_memory_bytes,
    }
    return metrics, last_output
