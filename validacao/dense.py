from __future__ import annotations

import numpy as np
import torch


def dense_projection_numpy(
    inputs: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Implementacao direta de `Y = XW^T + b` em NumPy."""
    output = np.matmul(inputs, np.swapaxes(weight, -1, -2))
    if bias is not None:
        output = output + bias
    return output


def dense_projection_torch(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Implementacao direta de `Y = XW^T + b` em PyTorch."""
    output = torch.matmul(inputs, weight.transpose(-1, -2))
    if bias is not None:
        output = output + bias
    return output
