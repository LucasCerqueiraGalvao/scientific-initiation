from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SparsityObservation:
    total_elements: int
    zero_elements: int
    sparsity_ratio: float
    uses_sparse_storage: bool
    uses_sparse_kernel: bool

    @property
    def validity_level(self) -> str:
        if self.sparsity_ratio <= 0.0:
            return "nao_aderente"
        if self.uses_sparse_storage and self.uses_sparse_kernel:
            return "hardware"
        return "conceitual_algoritmico"


def observe_sparsity(
    tensor: torch.Tensor,
    *,
    uses_sparse_storage: bool = False,
    uses_sparse_kernel: bool = False,
) -> SparsityObservation:
    total = int(tensor.numel())
    zeros = int((tensor == 0).sum().item())
    ratio = 0.0 if total == 0 else zeros / total
    return SparsityObservation(
        total_elements=total,
        zero_elements=zeros,
        sparsity_ratio=float(ratio),
        uses_sparse_storage=uses_sparse_storage,
        uses_sparse_kernel=uses_sparse_kernel,
    )
