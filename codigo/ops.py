import math

import torch


def dense_projection(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    output = x @ weight.transpose(-1, -2)
    if bias is not None:
        output = output + bias
    return output


def single_head_self_attention(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_k: torch.Tensor,
    w_v: torch.Tensor,
    w_o: torch.Tensor,
) -> torch.Tensor:
    d_model = x.shape[-1]
    q = x @ w_q.transpose(-1, -2)
    k = x @ w_k.transpose(-1, -2)
    v = x @ w_v.transpose(-1, -2)

    scores = q @ k.transpose(-1, -2)
    scores = scores / math.sqrt(d_model)
    attention_weights = torch.softmax(scores, dim=-1)
    context = attention_weights @ v

    return context @ w_o.transpose(-1, -2)
