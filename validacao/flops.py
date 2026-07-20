from __future__ import annotations

from dataclasses import dataclass


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} deve ser inteiro positivo")


def dense_projection_flops(
    *,
    batch_size: int,
    seq_len: int,
    in_features: int,
    out_features: int,
    bias: bool = True,
) -> int:
    """FLOPs para `Y = XW^T + b`, contando multiplicacao e soma separadas."""
    for name, value in {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "in_features": in_features,
        "out_features": out_features,
    }.items():
        _require_positive_int(name, value)

    matmul_flops = 2 * batch_size * seq_len * in_features * out_features
    bias_flops = batch_size * seq_len * out_features if bias else 0
    return int(matmul_flops + bias_flops)


def qkv_projection_flops(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    bias: bool = True,
) -> int:
    return 3 * dense_projection_flops(
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=d_model,
        out_features=d_model,
        bias=bias,
    )


def attention_matmul_flops(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
) -> int:
    """FLOPs das duas multiplicacoes centrais: `QK^T` e `AV`.

    Softmax, escalonamento e mascaras ficam fora desta conta para manter uma
    formula pequena e explicitamente rastreavel.
    """
    for name, value in {
        "batch_size": batch_size,
        "seq_len": seq_len,
        "d_model": d_model,
        "num_heads": num_heads,
    }.items():
        _require_positive_int(name, value)

    if d_model % num_heads != 0:
        raise ValueError("d_model deve ser divisivel por num_heads")

    head_dim = d_model // num_heads
    score_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    weighted_value_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
    return int(score_flops + weighted_value_flops)


def ffn_linear_flops(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    hidden_dim: int,
    bias: bool = True,
) -> int:
    """FLOPs das duas camadas lineares do FFN; ativacao nao incluida."""
    first = dense_projection_flops(
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=d_model,
        out_features=hidden_dim,
        bias=bias,
    )
    second = dense_projection_flops(
        batch_size=batch_size,
        seq_len=seq_len,
        in_features=hidden_dim,
        out_features=d_model,
        bias=bias,
    )
    return int(first + second)


@dataclass(frozen=True)
class TransformerBlockFlops:
    qkv_projection: int
    attention_matmuls: int
    output_projection: int
    ffn_linear: int

    @property
    def total(self) -> int:
        return self.qkv_projection + self.attention_matmuls + self.output_projection + self.ffn_linear


def transformer_block_flops(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    ffn_hidden_dim: int,
    bias: bool = True,
) -> TransformerBlockFlops:
    return TransformerBlockFlops(
        qkv_projection=qkv_projection_flops(
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=d_model,
            bias=bias,
        ),
        attention_matmuls=attention_matmul_flops(
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=d_model,
            num_heads=num_heads,
        ),
        output_projection=dense_projection_flops(
            batch_size=batch_size,
            seq_len=seq_len,
            in_features=d_model,
            out_features=d_model,
            bias=bias,
        ),
        ffn_linear=ffn_linear_flops(
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=d_model,
            hidden_dim=ffn_hidden_dim,
            bias=bias,
        ),
    )
