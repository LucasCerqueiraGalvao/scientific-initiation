from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizationObservation:
    dtype: str
    storage_bytes: int
    is_low_precision_storage: bool
    is_float32_simulation: bool


def tensor_storage_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def symmetric_int8_quantize(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantizacao simetrica simples para validar conceito e erro numerico.

    Retorna ``(quantized_int8, scale, dequantized_float32)``. A terceira saida
    e deliberadamente uma dequantizacao para analise numerica, nao evidencia de
    ganho de hardware.
    """
    max_abs = tensor.abs().max()
    if max_abs.item() == 0.0:
        scale = torch.tensor(1.0, dtype=torch.float32, device=tensor.device)
        quantized = torch.zeros_like(tensor, dtype=torch.int8)
        return quantized, scale, quantized.to(torch.float32)

    scale = max_abs / 127.0
    quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
    dequantized = quantized.to(torch.float32) * scale
    return quantized, scale, dequantized


def symmetric_int8_quantize_per_row(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantiza cada linha independentemente e devolve simulacao em float32."""
    if tensor.ndim < 2:
        raise ValueError("quantizacao por linha exige tensor com pelo menos duas dimensoes")
    source = tensor.to(torch.float32)
    max_abs = source.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / 127.0)
    quantized = torch.clamp(torch.round(source / scale), -127, 127).to(torch.int8)
    dequantized = quantized.to(torch.float32) * scale
    return quantized, scale, dequantized


def observe_quantization_storage(tensor: torch.Tensor, original_dtype: torch.dtype = torch.float32) -> QuantizationObservation:
    dtype_name = str(tensor.dtype).replace("torch.", "")
    storage_bytes = tensor_storage_bytes(tensor)
    is_low_precision_storage = tensor.element_size() < torch.empty((), dtype=original_dtype).element_size()
    is_float32_simulation = tensor.dtype == torch.float32
    return QuantizationObservation(
        dtype=dtype_name,
        storage_bytes=storage_bytes,
        is_low_precision_storage=bool(is_low_precision_storage),
        is_float32_simulation=bool(is_float32_simulation),
    )
