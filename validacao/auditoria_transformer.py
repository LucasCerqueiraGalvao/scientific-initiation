from __future__ import annotations

from dataclasses import dataclass

import torch

from validacao.classificacao import TransformerChecklist, classify_transformer_implementation


@dataclass(frozen=True)
class TransformerAuditReport:
    architecture_label: str
    receives_sequence: bool
    preserves_sequence_shape: bool
    has_qkv_projection: bool
    has_scaled_attention: bool
    scaled_attention_matches_reference: bool
    has_cross_position_dependency: bool
    has_positional_information_or_justification: bool
    has_residual_norm_ffn_or_justification: bool
    deterministic_in_eval: bool
    can_be_called_transformer: bool
    can_be_called_simplified_block: bool
    failed_criteria: tuple[str, ...]

    @property
    def is_transformer_family(self) -> bool:
        return self.can_be_called_transformer or self.can_be_called_simplified_block


def _has_linear(module: torch.nn.Module, name: str) -> bool:
    return isinstance(getattr(module, name, None), torch.nn.Linear)


def _has_layer_norm(module: torch.nn.Module, name: str) -> bool:
    return isinstance(getattr(module, name, None), torch.nn.LayerNorm)


def _has_ffn(module: torch.nn.Module) -> bool:
    ffn = getattr(module, "ffn", None)
    if not isinstance(ffn, torch.nn.Sequential):
        return False
    return sum(isinstance(child, torch.nn.Linear) for child in ffn) >= 2


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _safe_forward(module: torch.nn.Module, sequence: torch.Tensor) -> torch.Tensor | None:
    try:
        with torch.inference_mode():
            return module(sequence)
    except Exception:
        return None


def _scaled_attention_matches_reference(
    module: torch.nn.Module,
    sequence: torch.Tensor,
    *,
    atol: float,
) -> bool:
    if not all(hasattr(module, name) for name in ("add_positional_information", "project_qkv", "attention_core")):
        return False

    with torch.inference_mode():
        hidden = module.add_positional_information(sequence)
        query, key, value = module.project_qkv(hidden)
        actual = module.attention_core(query, key, value)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
        )
    return bool(torch.allclose(actual, expected, atol=atol))


def _has_cross_position_dependency(
    module: torch.nn.Module,
    sequence: torch.Tensor,
    *,
    atol: float,
) -> bool:
    if sequence.shape[1] < 2:
        return False

    modified = sequence.clone()
    modified[:, 1, :] = modified[:, 1, :] + 5.0
    baseline = _safe_forward(module, sequence)
    changed = _safe_forward(module, modified)
    if baseline is None or changed is None:
        return False
    return not torch.allclose(baseline[:, 0, :], changed[:, 0, :], atol=atol)


def audit_transformer_module(
    module: torch.nn.Module,
    *,
    batch_size: int = 2,
    seq_len: int = 3,
    d_model: int | None = None,
    atol: float = 1e-6,
    seed: int = 2026,
) -> TransformerAuditReport:
    module.eval()
    device = _module_device(module)
    if d_model is None:
        d_model = int(getattr(module, "d_model", 4))

    torch.manual_seed(seed)
    sequence = torch.randn(batch_size, seq_len, d_model, device=device)
    first_output = _safe_forward(module, sequence)
    second_output = _safe_forward(module, sequence)

    receives_sequence = sequence.ndim == 3 and first_output is not None
    preserves_sequence_shape = bool(first_output is not None and tuple(first_output.shape) == tuple(sequence.shape))
    has_qkv_projection = all(_has_linear(module, name) for name in ("q_proj", "k_proj", "v_proj"))
    has_scaled_attention = _scaled_attention_matches_reference(module, sequence, atol=atol)
    has_cross_position_dependency = _has_cross_position_dependency(module, sequence, atol=atol)
    has_positional_information_or_justification = bool(
        getattr(module, "use_positional_encoding", False)
        or getattr(module, "positional_information_justification", "")
    )
    has_residual_norm_ffn_or_justification = bool(
        _has_layer_norm(module, "norm1")
        and _has_layer_norm(module, "norm2")
        and _has_linear(module, "out_proj")
        and _has_ffn(module)
    )
    deterministic_in_eval = bool(
        first_output is not None
        and second_output is not None
        and torch.allclose(first_output, second_output, atol=atol)
    )

    checklist = TransformerChecklist(
        receives_sequence=receives_sequence and preserves_sequence_shape,
        has_qkv_projection=has_qkv_projection,
        has_scaled_attention=has_scaled_attention,
        has_cross_position_dependency=has_cross_position_dependency,
        has_positional_information_or_justification=has_positional_information_or_justification,
        has_residual_norm_ffn_or_justification=has_residual_norm_ffn_or_justification,
        is_complete_architecture=bool(getattr(module, "is_complete_transformer_architecture", False)),
    )
    architecture_label = classify_transformer_implementation(checklist)

    criteria = {
        "receives_sequence": receives_sequence,
        "preserves_sequence_shape": preserves_sequence_shape,
        "has_qkv_projection": has_qkv_projection,
        "has_scaled_attention": has_scaled_attention,
        "has_cross_position_dependency": has_cross_position_dependency,
        "has_positional_information_or_justification": has_positional_information_or_justification,
        "has_residual_norm_ffn_or_justification": has_residual_norm_ffn_or_justification,
        "deterministic_in_eval": deterministic_in_eval,
    }
    failed_criteria = tuple(name for name, passed in criteria.items() if not passed)

    return TransformerAuditReport(
        architecture_label=architecture_label,
        receives_sequence=receives_sequence,
        preserves_sequence_shape=preserves_sequence_shape,
        has_qkv_projection=has_qkv_projection,
        has_scaled_attention=has_scaled_attention,
        scaled_attention_matches_reference=has_scaled_attention,
        has_cross_position_dependency=has_cross_position_dependency,
        has_positional_information_or_justification=has_positional_information_or_justification,
        has_residual_norm_ffn_or_justification=has_residual_norm_ffn_or_justification,
        deterministic_in_eval=deterministic_in_eval,
        can_be_called_transformer=architecture_label == "transformer",
        can_be_called_simplified_block=architecture_label == "bloco_transformer_simplificado",
        failed_criteria=failed_criteria,
    )
