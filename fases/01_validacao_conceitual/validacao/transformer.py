from __future__ import annotations

import math

import torch

from validacao.attention import combine_heads, scaled_dot_product_attention_torch, split_heads
from validacao.classificacao import TransformerChecklist


def sinusoidal_position_encoding(
    seq_len: int,
    d_model: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=dtype) * (-math.log(10000.0) / d_model)
    )
    encoding = torch.zeros(seq_len, d_model, device=device, dtype=dtype)
    encoding[:, 0::2] = torch.sin(position * div_term)
    if d_model > 1:
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0)


class SimplifiedTransformerBlock(torch.nn.Module):
    """Bloco pequeno para validacao: MHA + residual/norm + FFN."""

    def __init__(
        self,
        *,
        d_model: int,
        num_heads: int,
        ffn_hidden_dim: int,
        use_positional_encoding: bool = True,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model deve ser divisivel por num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.use_positional_encoding = use_positional_encoding

        self.q_proj = torch.nn.Linear(d_model, d_model)
        self.k_proj = torch.nn.Linear(d_model, d_model)
        self.v_proj = torch.nn.Linear(d_model, d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, ffn_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(ffn_hidden_dim, d_model),
        )
        self.norm2 = torch.nn.LayerNorm(d_model)

    def add_positional_information(self, sequence: torch.Tensor) -> torch.Tensor:
        if not self.use_positional_encoding:
            return sequence
        return sequence + sinusoidal_position_encoding(
            sequence.shape[1],
            sequence.shape[2],
            device=sequence.device,
            dtype=sequence.dtype,
        )

    def project_qkv(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            split_heads(self.q_proj(hidden), self.num_heads),
            split_heads(self.k_proj(hidden), self.num_heads),
            split_heads(self.v_proj(hidden), self.num_heads),
        )

    def attention_core(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return scaled_dot_product_attention_torch(query, key, value)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError("sequence deve ter shape [batch, seq, d_model]")

        hidden = self.add_positional_information(sequence)
        query, key, value = self.project_qkv(hidden)
        attention = self.attention_core(query, key, value)
        attention = combine_heads(attention)

        hidden = self.norm1(hidden + self.out_proj(attention))
        return self.norm2(hidden + self.ffn(hidden))


def transformer_block_checklist(block: SimplifiedTransformerBlock) -> TransformerChecklist:
    return TransformerChecklist(
        receives_sequence=True,
        has_qkv_projection=True,
        has_scaled_attention=True,
        has_cross_position_dependency=True,
        has_positional_information_or_justification=block.use_positional_encoding,
        has_residual_norm_ffn_or_justification=True,
        is_complete_architecture=False,
    )
