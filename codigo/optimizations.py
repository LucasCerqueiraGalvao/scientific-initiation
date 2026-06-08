from __future__ import annotations

import torch


def prune_by_magnitude(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0.0, 1.0).")

    if tensor.numel() == 0 or sparsity == 0.0:
        return tensor.clone()

    flattened = tensor.abs().reshape(-1)
    keep_count = max(int(round((1.0 - sparsity) * flattened.numel())), 1)
    threshold = torch.topk(flattened, k=keep_count, largest=True).values.min()
    mask = tensor.abs() >= threshold
    return tensor * mask


def fake_quantize_int8(tensor: torch.Tensor) -> torch.Tensor:
    max_abs = tensor.abs().max()
    if max_abs.item() == 0.0:
        return tensor.clone()

    scale = max_abs / 127.0
    quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    return quantized.to(torch.float32) * scale
