from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerChecklist:
    """Checklist minimo para classificar uma implementacao relacionada a Transformer."""

    receives_sequence: bool
    has_qkv_projection: bool
    has_scaled_attention: bool
    has_cross_position_dependency: bool
    has_positional_information_or_justification: bool
    has_residual_norm_ffn_or_justification: bool = False
    is_complete_architecture: bool = False

    def positional_requirement_met(self) -> bool:
        return self.has_positional_information_or_justification

    def attention_core_met(self) -> bool:
        return (
            self.receives_sequence
            and self.has_qkv_projection
            and self.has_scaled_attention
            and self.has_cross_position_dependency
        )


def classify_transformer_implementation(checklist: TransformerChecklist) -> str:
    if (
        checklist.attention_core_met()
        and checklist.positional_requirement_met()
        and checklist.has_residual_norm_ffn_or_justification
        and checklist.is_complete_architecture
    ):
        return "transformer"

    if checklist.attention_core_met() and checklist.positional_requirement_met():
        return "bloco_transformer_simplificado"

    if checklist.has_scaled_attention or checklist.has_qkv_projection:
        return "operacao_inspirada_em_transformer"

    return "nao_aderente"
