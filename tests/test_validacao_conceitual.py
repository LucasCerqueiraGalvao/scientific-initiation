from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
import torch
import torch.nn.utils.prune as prune
from sklearn.metrics import mean_absolute_error as sklearn_mae
from sklearn.metrics import mean_squared_error as sklearn_mse
from sklearn.metrics import r2_score as sklearn_r2
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from validacao.auditoria_transformer import audit_transformer_module
from validacao.attention import (
    multi_head_attention_from_qkv,
    scaled_dot_product_attention_numpy,
    scaled_dot_product_attention_torch,
    split_heads,
)
from validacao.classificacao import (
    TransformerChecklist,
    classify_transformer_implementation,
)
from validacao.dense import dense_projection_numpy, dense_projection_torch
from validacao.flops import (
    attention_matmul_flops,
    dense_projection_flops,
    transformer_block_flops,
)
from validacao.kv_cache import (
    causal_self_attention_full,
    causal_self_attention_with_cache,
)
from validacao.metrics import (
    cosine_similarity,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from validacao.protocolo import (
    CSV_COLUMNS,
    empty_benchmark_record,
    validate_benchmark_record,
)
from validacao.quantization import (
    observe_quantization_storage,
    symmetric_int8_quantize,
)
from validacao.sparsity import observe_sparsity
from validacao.transformer import (
    SimplifiedTransformerBlock,
    transformer_block_checklist,
)


def test_dense_projection_matches_manual_definition() -> None:
    inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    weight = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    bias = torch.tensor([0.5, -0.5])

    actual = dense_projection_torch(inputs, weight, bias)
    expected = torch.tensor([[[1.5, 3.5], [3.5, 7.5]]])
    numpy_expected = dense_projection_numpy(inputs.numpy(), weight.numpy(), bias.numpy())

    assert torch.allclose(actual, expected)
    assert torch.allclose(actual, torch.from_numpy(numpy_expected))


def test_scaled_dot_product_attention_matches_independent_numpy_reference() -> None:
    query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    value = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])

    actual = scaled_dot_product_attention_torch(query, key, value)
    expected = scaled_dot_product_attention_numpy(
        query.numpy(),
        key.numpy(),
        value.numpy(),
    )
    expected_from_formula = torch.tensor(
        [[[1.6604769, 2.6604769], [2.3395231, 3.3395231]]]
    )

    assert torch.allclose(actual, torch.from_numpy(expected), atol=1e-6)
    assert torch.allclose(actual, expected_from_formula, atol=1e-6)


def test_scaled_dot_product_attention_matches_pytorch_reference() -> None:
    torch.manual_seed(42)
    query = torch.randn(1, 1, 3, 4)
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 4)

    actual = scaled_dot_product_attention_torch(query, key, value)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_scaled_dot_product_attention_additive_mask_matches_pytorch_reference() -> None:
    torch.manual_seed(7)
    query = torch.randn(1, 1, 3, 4)
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 4)
    additive_mask = torch.tensor(
        [
            [0.0, 0.0, torch.finfo(torch.float32).min],
            [0.0, 0.0, 0.0],
            [torch.finfo(torch.float32).min, 0.0, 0.0],
        ]
    )

    actual = scaled_dot_product_attention_torch(
        query,
        key,
        value,
        additive_attention_mask=additive_mask,
    )
    expected = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=additive_mask,
        dropout_p=0.0,
    )

    assert torch.allclose(actual, expected, atol=1e-6)


def test_multi_head_attention_preserves_shape_and_matches_split_reference() -> None:
    torch.manual_seed(11)
    sequence = torch.randn(2, 3, 4)
    num_heads = 2

    actual = multi_head_attention_from_qkv(
        sequence,
        sequence,
        sequence,
        num_heads=num_heads,
    )
    expected_heads = torch.nn.functional.scaled_dot_product_attention(
        split_heads(sequence, num_heads),
        split_heads(sequence, num_heads),
        split_heads(sequence, num_heads),
        dropout_p=0.0,
    )
    expected = expected_heads.transpose(1, 2).contiguous().reshape(2, 3, 4)

    assert actual.shape == sequence.shape
    assert torch.allclose(actual, expected, atol=1e-6)


def test_attention_output_depends_on_other_sequence_positions() -> None:
    query = torch.zeros(1, 2, 2)
    key = torch.zeros(1, 2, 2)
    value_a = torch.tensor([[[1.0, 1.0], [3.0, 3.0]]])
    value_b = torch.tensor([[[1.0, 1.0], [9.0, 9.0]]])

    output_a = scaled_dot_product_attention_torch(query, key, value_a)
    output_b = scaled_dot_product_attention_torch(query, key, value_b)

    assert not torch.allclose(output_a[:, 0, :], output_b[:, 0, :])


def test_transformer_checklist_classifies_expected_levels() -> None:
    attention_only = TransformerChecklist(
        receives_sequence=False,
        has_qkv_projection=False,
        has_scaled_attention=True,
        has_cross_position_dependency=False,
        has_positional_information_or_justification=False,
    )
    simplified_block = TransformerChecklist(
        receives_sequence=True,
        has_qkv_projection=True,
        has_scaled_attention=True,
        has_cross_position_dependency=True,
        has_positional_information_or_justification=True,
    )
    full_transformer = TransformerChecklist(
        receives_sequence=True,
        has_qkv_projection=True,
        has_scaled_attention=True,
        has_cross_position_dependency=True,
        has_positional_information_or_justification=True,
        has_residual_norm_ffn_or_justification=True,
        is_complete_architecture=True,
    )

    assert classify_transformer_implementation(attention_only) == "operacao_inspirada_em_transformer"
    assert classify_transformer_implementation(simplified_block) == "bloco_transformer_simplificado"
    assert classify_transformer_implementation(full_transformer) == "transformer"


def test_simplified_transformer_block_has_expected_components_and_shape() -> None:
    torch.manual_seed(123)
    block = SimplifiedTransformerBlock(d_model=4, num_heads=2, ffn_hidden_dim=8).eval()
    sequence = torch.randn(2, 3, 4)

    output = block(sequence)
    checklist = transformer_block_checklist(block)

    assert output.shape == sequence.shape
    assert classify_transformer_implementation(checklist) == "bloco_transformer_simplificado"
    assert hasattr(block, "q_proj")
    assert hasattr(block, "k_proj")
    assert hasattr(block, "v_proj")
    assert hasattr(block, "norm1")
    assert hasattr(block, "ffn")


def test_transformer_audit_validates_current_nn_as_simplified_transformer_block() -> None:
    torch.manual_seed(123)
    block = SimplifiedTransformerBlock(d_model=4, num_heads=2, ffn_hidden_dim=8).eval()

    report = audit_transformer_module(block, d_model=4, seq_len=3)

    assert report.architecture_label == "bloco_transformer_simplificado"
    assert report.is_transformer_family
    assert not report.can_be_called_transformer
    assert report.can_be_called_simplified_block
    assert report.receives_sequence
    assert report.preserves_sequence_shape
    assert report.has_qkv_projection
    assert report.has_scaled_attention
    assert report.scaled_attention_matches_reference
    assert report.has_cross_position_dependency
    assert report.has_positional_information_or_justification
    assert report.has_residual_norm_ffn_or_justification
    assert report.deterministic_in_eval
    assert report.failed_criteria == ()


def test_transformer_audit_rejects_plain_neural_network_as_transformer() -> None:
    plain_nn = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 4)).eval()

    report = audit_transformer_module(plain_nn, d_model=4, seq_len=3)

    assert report.architecture_label == "nao_aderente"
    assert not report.is_transformer_family
    assert not report.has_qkv_projection
    assert not report.has_scaled_attention
    assert "has_qkv_projection" in report.failed_criteria
    assert "has_scaled_attention" in report.failed_criteria


def test_kv_cache_matches_full_causal_attention_and_reuses_previous_keys_values() -> None:
    torch.manual_seed(5)
    sequence = torch.randn(1, 4, 3)
    identity = torch.eye(3)

    full_output = causal_self_attention_full(sequence, identity, identity, identity)
    cached = causal_self_attention_with_cache(sequence, identity, identity, identity)

    assert torch.allclose(cached.output, full_output, atol=1e-6)
    assert cached.final_cache_length == 4
    assert cached.cached_key_value_tokens == 4
    assert cached.naive_key_value_tokens == 10
    assert cached.reused_key_value_tokens == 6


def test_theoretical_flops_formulas_match_small_known_cases() -> None:
    assert dense_projection_flops(
        batch_size=2,
        seq_len=3,
        in_features=4,
        out_features=5,
    ) == 270
    assert attention_matmul_flops(
        batch_size=2,
        seq_len=3,
        d_model=4,
        num_heads=2,
    ) == 288

    block_flops = transformer_block_flops(
        batch_size=2,
        seq_len=3,
        d_model=4,
        num_heads=2,
        ffn_hidden_dim=8,
    )

    assert block_flops.qkv_projection == 648
    assert block_flops.output_projection == 216
    assert block_flops.ffn_linear == 840
    assert block_flops.total == 1992


def test_benchmark_csv_protocol_requires_fixed_columns_and_validity_level() -> None:
    record = empty_benchmark_record(
        torch_version="2.11.0+cu128",
        cuda_version="12.8",
        gpu_name="NVIDIA GeForce RTX 4070 Ti SUPER",
        device="cuda",
        model_kind="simplified_transformer_block",
        scenario="baseline",
        batch_size=2,
        seq_len=3,
        d_model=4,
        num_heads=2,
        dtype="float32",
        pruning_sparsity=0.0,
        quantization_method="none",
        validity_level="algoritmico",
    )

    validation = validate_benchmark_record(record)
    invalid = validate_benchmark_record({"validity_level": "promessa_sem_evidencia"})
    with_extra_column = dict(record)
    with_extra_column["coluna_solta"] = "nao pode entrar no CSV final"
    extra_validation = validate_benchmark_record(with_extra_column)

    assert list(record.keys()) == CSV_COLUMNS
    assert validation.is_valid
    assert not invalid.is_valid
    assert invalid.invalid_validity_level == "promessa_sem_evidencia"
    assert not extra_validation.is_valid
    assert extra_validation.extra_columns == ("coluna_solta",)


def test_manual_metrics_match_known_values_and_sklearn() -> None:
    reference = np.array([1.0, 2.0, 3.0])
    candidate = np.array([1.0, 2.0, 4.0])

    assert mean_squared_error(reference, candidate) == pytest.approx(1.0 / 3.0)
    assert mean_absolute_error(reference, candidate) == pytest.approx(1.0 / 3.0)
    assert r2_score(reference, candidate) == pytest.approx(0.5)

    assert mean_squared_error(reference, candidate) == pytest.approx(sklearn_mse(reference, candidate))
    assert mean_absolute_error(reference, candidate) == pytest.approx(sklearn_mae(reference, candidate))
    assert r2_score(reference, candidate) == pytest.approx(sklearn_r2(reference, candidate))


def test_cosine_similarity_matches_definition_and_sklearn() -> None:
    reference = np.array([1.0, 1.0, 0.0])
    candidate = np.array([1.0, 0.0, 0.0])

    expected = 1.0 / math.sqrt(2.0)

    assert cosine_similarity(reference, candidate) == pytest.approx(expected)
    assert cosine_similarity(reference, candidate) == pytest.approx(
        sklearn_cosine(reference.reshape(1, -1), candidate.reshape(1, -1))[0, 0]
    )


def test_pytorch_pruning_creates_expected_sparsity_but_not_hardware_claim() -> None:
    layer = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        layer.weight.copy_(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                ]
            )
        )

    prune.l1_unstructured(layer, name="weight", amount=0.5)

    observation = observe_sparsity(layer.weight, uses_sparse_storage=False, uses_sparse_kernel=False)

    assert observation.zero_elements == 4
    assert observation.sparsity_ratio == pytest.approx(0.5)
    assert observation.validity_level == "conceitual_algoritmico"
    assert hasattr(layer, "weight_mask")


def test_symmetric_int8_quantization_reduces_storage_and_tracks_numeric_error() -> None:
    tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)

    quantized, scale, dequantized = symmetric_int8_quantize(tensor)
    q_observation = observe_quantization_storage(quantized)
    deq_observation = observe_quantization_storage(dequantized)

    assert quantized.dtype == torch.int8
    assert q_observation.storage_bytes == 5
    assert tensor.numel() * tensor.element_size() == 20
    assert scale.item() > 0
    assert q_observation.is_low_precision_storage
    assert not q_observation.is_float32_simulation
    assert deq_observation.is_float32_simulation
    assert torch.max(torch.abs(tensor - dequantized)).item() <= scale.item()


def test_torchao_int8_weight_only_v2_stores_internal_int8_weight_and_runs_linear_without_warnings() -> None:
    pytest.importorskip("torchao")
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    model = torch.nn.Linear(4, 3).eval()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        quantize_(model, Int8WeightOnlyConfig(version=2))

    output = model(torch.randn(2, 4))

    assert output.shape == (2, 3)
    assert caught == []
    assert type(model.weight).__name__ == "Int8Tensor"
    assert model.weight.qdata.dtype == torch.int8
    assert model.weight.qdata.element_size() < model.weight.dequantize().element_size()
