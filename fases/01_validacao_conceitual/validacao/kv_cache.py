from __future__ import annotations

from dataclasses import dataclass

import torch

from validacao.attention import scaled_dot_product_attention_torch
from validacao.dense import dense_projection_torch


@dataclass(frozen=True)
class CachedAttentionResult:
    output: torch.Tensor
    cached_key_value_tokens: int
    naive_key_value_tokens: int
    final_cache_length: int

    @property
    def reused_key_value_tokens(self) -> int:
        return self.naive_key_value_tokens - self.cached_key_value_tokens


def causal_attention_mask(seq_len: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    future_positions = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    mask = mask.masked_fill(future_positions, torch.finfo(dtype).min)
    return mask


def causal_self_attention_full(
    sequence: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
) -> torch.Tensor:
    """Atencao causal recalculando Q, K e V para toda a sequencia."""
    query = dense_projection_torch(sequence, q_weight)
    key = dense_projection_torch(sequence, k_weight)
    value = dense_projection_torch(sequence, v_weight)
    mask = causal_attention_mask(sequence.shape[1], device=sequence.device, dtype=sequence.dtype)
    return scaled_dot_product_attention_torch(query, key, value, additive_attention_mask=mask)


def causal_self_attention_with_cache(
    sequence: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
) -> CachedAttentionResult:
    """Atencao causal token a token, armazenando K/V ja calculados."""
    key_cache: torch.Tensor | None = None
    value_cache: torch.Tensor | None = None
    outputs: list[torch.Tensor] = []

    for position in range(sequence.shape[1]):
        token = sequence[:, position : position + 1, :]
        query = dense_projection_torch(token, q_weight)
        key = dense_projection_torch(token, k_weight)
        value = dense_projection_torch(token, v_weight)

        key_cache = key if key_cache is None else torch.cat([key_cache, key], dim=1)
        value_cache = value if value_cache is None else torch.cat([value_cache, value], dim=1)
        outputs.append(scaled_dot_product_attention_torch(query, key_cache, value_cache))

    seq_len = sequence.shape[1]
    return CachedAttentionResult(
        output=torch.cat(outputs, dim=1),
        cached_key_value_tokens=seq_len,
        naive_key_value_tokens=(seq_len * (seq_len + 1)) // 2,
        final_cache_length=seq_len,
    )
