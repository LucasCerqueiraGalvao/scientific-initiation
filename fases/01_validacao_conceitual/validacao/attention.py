from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as torch_functional


def softmax_numpy(values: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def scaled_dot_product_attention_numpy(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    additive_attention_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Implementacao direta de softmax(QK^T / sqrt(d_k))V em NumPy."""
    d_k = query.shape[-1]
    scores = np.matmul(query, np.swapaxes(key, -1, -2)) / math.sqrt(d_k)
    if additive_attention_mask is not None:
        scores = scores + additive_attention_mask
    weights = softmax_numpy(scores, axis=-1)
    return np.matmul(weights, value)


def scaled_dot_product_attention_torch(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    additive_attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Implementacao direta de softmax(QK^T / sqrt(d_k))V em PyTorch."""
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)
    if additive_attention_mask is not None:
        scores = scores + additive_attention_mask
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value)


def split_heads(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Converte `[batch, seq, d_model]` para `[batch, heads, seq, d_head]`."""
    if tensor.ndim != 3:
        raise ValueError("tensor deve ter shape [batch, seq, d_model]")

    batch_size, seq_len, d_model = tensor.shape
    if d_model % num_heads != 0:
        raise ValueError("d_model deve ser divisivel por num_heads")

    head_dim = d_model // num_heads
    return tensor.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)


def combine_heads(tensor: torch.Tensor) -> torch.Tensor:
    """Converte `[batch, heads, seq, d_head]` para `[batch, seq, d_model]`."""
    if tensor.ndim != 4:
        raise ValueError("tensor deve ter shape [batch, heads, seq, d_head]")

    batch_size, num_heads, seq_len, head_dim = tensor.shape
    return tensor.transpose(1, 2).contiguous().reshape(batch_size, seq_len, num_heads * head_dim)


def split_heads_numpy(tensor: np.ndarray, num_heads: int) -> np.ndarray:
    """Converte ``[batch, seq, d_model]`` para ``[batch, heads, seq, d_head]``."""
    array = np.asarray(tensor)
    if array.ndim != 3:
        raise ValueError("tensor deve ter shape [batch, seq, d_model]")
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ValueError("num_heads deve ser inteiro positivo")
    batch_size, seq_len, d_model = array.shape
    if d_model % num_heads != 0:
        raise ValueError("d_model deve ser divisivel por num_heads")
    head_dim = d_model // num_heads
    return np.ascontiguousarray(array.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3))


def combine_heads_numpy(tensor: np.ndarray) -> np.ndarray:
    """Converte ``[batch, heads, seq, d_head]`` para ``[batch, seq, d_model]``."""
    array = np.asarray(tensor)
    if array.ndim != 4:
        raise ValueError("tensor deve ter shape [batch, heads, seq, d_head]")
    batch_size, num_heads, seq_len, head_dim = array.shape
    return np.ascontiguousarray(array.transpose(0, 2, 1, 3)).reshape(
        batch_size,
        seq_len,
        num_heads * head_dim,
    )


def multi_head_attention_from_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_heads: int,
    additive_attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Atencao multi-head transparente a partir de Q, K e V ja projetados."""
    query_heads = split_heads(query, num_heads)
    key_heads = split_heads(key, num_heads)
    value_heads = split_heads(value, num_heads)
    attention_heads = scaled_dot_product_attention_torch(
        query_heads,
        key_heads,
        value_heads,
        additive_attention_mask=additive_attention_mask,
    )
    return combine_heads(attention_heads)


def multi_head_self_attention_numpy(
    inputs: np.ndarray,
    w_q: np.ndarray,
    w_k: np.ndarray,
    w_v: np.ndarray,
    w_o: np.ndarray,
    *,
    num_heads: int,
    additive_attention_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Self-attention multi-head completa com projecoes, em NumPy."""
    from validacao.dense import dense_projection_numpy

    query = split_heads_numpy(dense_projection_numpy(inputs, w_q), num_heads)
    key = split_heads_numpy(dense_projection_numpy(inputs, w_k), num_heads)
    value = split_heads_numpy(dense_projection_numpy(inputs, w_v), num_heads)
    context = scaled_dot_product_attention_numpy(
        query,
        key,
        value,
        additive_attention_mask=additive_attention_mask,
    )
    return dense_projection_numpy(combine_heads_numpy(context), w_o)


def multi_head_self_attention_torch(
    inputs: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
    *,
    num_heads: int,
    additive_attention_mask: torch.Tensor | None = None,
    use_sdpa: bool = True,
) -> torch.Tensor:
    """Self-attention multi-head completa com o mesmo contrato do benchmark."""
    from validacao.dense import dense_projection_torch

    query = split_heads(dense_projection_torch(inputs, w_q), num_heads)
    key = split_heads(dense_projection_torch(inputs, w_k), num_heads)
    value = split_heads(dense_projection_torch(inputs, w_v), num_heads)
    if use_sdpa:
        context = torch_functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=additive_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
    else:
        context = scaled_dot_product_attention_torch(
            query,
            key,
            value,
            additive_attention_mask=additive_attention_mask,
        )
    return dense_projection_torch(combine_heads(context), w_o)
