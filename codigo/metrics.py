from __future__ import annotations

import numpy as np
import torch


def compute_quality_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    reference_np = reference.detach().cpu().reshape(-1).numpy().astype(np.float64)
    candidate_np = candidate.detach().cpu().reshape(-1).numpy().astype(np.float64)

    residual = reference_np - candidate_np
    mse = float(np.mean(np.square(residual)))
    mae = float(np.mean(np.abs(residual)))

    reference_mean = np.mean(reference_np)
    ss_res = float(np.sum(np.square(residual)))
    ss_tot = float(np.sum(np.square(reference_np - reference_mean)))
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot)

    reference_norm = float(np.linalg.norm(reference_np))
    candidate_norm = float(np.linalg.norm(candidate_np))
    if reference_norm == 0.0 and candidate_norm == 0.0:
        cosine_similarity = 1.0
    elif reference_norm == 0.0 or candidate_norm == 0.0:
        cosine_similarity = 0.0
    else:
        cosine_similarity = float(np.dot(reference_np, candidate_np) / (reference_norm * candidate_norm))

    has_invalid_values = bool(torch.isnan(candidate).any().item() or torch.isinf(candidate).any().item())

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "cosine_similarity": cosine_similarity,
        "has_invalid_values": has_invalid_values,
    }


def estimate_dense_projection_flops(batch_size: int, sequence_length: int, dimension: int) -> int:
    return 2 * batch_size * sequence_length * dimension * dimension


def estimate_self_attention_flops(batch_size: int, sequence_length: int, dimension: int) -> int:
    projection_flops = 4 * 2 * batch_size * sequence_length * dimension * dimension
    attention_scores_flops = 2 * batch_size * sequence_length * sequence_length * dimension
    attention_values_flops = 2 * batch_size * sequence_length * sequence_length * dimension
    softmax_approx_flops = 5 * batch_size * sequence_length * sequence_length
    return projection_flops + attention_scores_flops + attention_values_flops + softmax_approx_flops


def estimate_tensor_memory_bytes(*tensors: torch.Tensor) -> int:
    return int(sum(tensor.numel() * tensor.element_size() for tensor in tensors))
