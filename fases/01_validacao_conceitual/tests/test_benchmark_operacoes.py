from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from validacao import benchmark_operacoes
from validacao.benchmark_operacoes import (
    ExperimentConfigurationError,
    ExperimentEnvironmentError,
    build_operation_inputs,
    config_from_mapping,
    execute_experiment_suite,
    load_experiment_config,
    prepare_scenario,
    resolve_experiment_device,
    run_independent_benchmark,
)
from validacao.protocolo import CSV_COLUMNS


PHASE_ROOT = Path(__file__).resolve().parents[1]


def smoke_config():
    return load_experiment_config(PHASE_ROOT / "experimentos" / "smoke_cpu.json")


def test_primary_config_matches_formal_protocol() -> None:
    config = load_experiment_config(PHASE_ROOT / "experimentos" / "benchmark_principal_gpu.json")

    assert config.purpose == "primary_benchmark"
    assert config.device == "cuda"
    assert config.required_gpu_name == "NVIDIA GeForce RTX 4070 Ti SUPER"
    assert config.seed == 42
    assert config.batch_sizes == (1,)
    assert config.sequence_lengths == (64, 128, 256)
    assert config.dimensions == (128, 256, 512)
    assert config.operations == ("dense_projection", "self_attention")
    assert config.scenarios == ("baseline", "pruning_magnitude", "quantization_int8")
    assert config.pruning_sparsity == 0.5
    assert config.warmup_iterations == 20
    assert config.measure_iterations == 50
    assert config.independent_runs == 2
    assert config.schema_version == 2
    assert config.input_distribution == "standard_normal"
    assert config.weight_initialization == "xavier_normal"
    assert config.bias_initialization == "zeros"


def test_cpu_diagnostic_uses_one_formal_grid_point_and_full_measurement_protocol() -> None:
    config = load_experiment_config(PHASE_ROOT / "experimentos" / "diagnostico_cpu_l64_d128.json")

    assert config.purpose == "cpu_diagnostic"
    assert config.device == "cpu"
    assert config.seed == 42
    assert config.sequence_lengths == (64,)
    assert config.dimensions == (128,)
    assert config.warmup_iterations == 20
    assert config.measure_iterations == 50
    assert config.independent_runs == 2


def test_config_rejects_auto_device_and_primary_cpu() -> None:
    data = smoke_config().to_dict()
    data["device"] = "auto"
    with pytest.raises(ExperimentConfigurationError, match="auto nao e reproduzivel"):
        config_from_mapping(data)

    data = smoke_config().to_dict()
    data["purpose"] = "primary_benchmark"
    with pytest.raises(ExperimentConfigurationError, match="exige device=cuda"):
        config_from_mapping(data)


def test_pruning_is_exact_and_quantization_does_not_claim_low_precision_kernel_storage() -> None:
    tensors = build_operation_inputs(
        "self_attention",
        batch_size=1,
        seq_len=4,
        dimension=4,
        seed=42,
        device=torch.device("cpu"),
    )
    pruned = prepare_scenario("self_attention", "pruning_magnitude", tensors, pruning_sparsity=0.5)
    quantized = prepare_scenario("self_attention", "quantization_int8", tensors, pruning_sparsity=0.5)

    assert pruned.observed_sparsity == pytest.approx(0.5)
    assert not pruned.uses_sparse_kernel
    assert quantized.compact_representation_bytes < quantized.runtime_tensor_bytes
    assert all(tensor.dtype == torch.float32 for tensor in quantized.tensors.values())
    assert not quantized.uses_low_precision_storage_in_kernel


def test_xavier_initialization_scales_weights_and_zeroes_bias() -> None:
    dimension = 128
    tensors = build_operation_inputs(
        "dense_projection",
        batch_size=1,
        seq_len=4,
        dimension=dimension,
        seed=42,
        device=torch.device("cpu"),
        weight_initialization="xavier_normal",
        bias_initialization="zeros",
    )

    expected_std = 1.0 / dimension**0.5
    assert float(tensors["weight"].std()) == pytest.approx(expected_std, rel=0.05)
    assert torch.count_nonzero(tensors["bias"]).item() == 0


def test_independent_runs_reuse_inputs_and_outputs_but_rotate_scenario_order() -> None:
    config = smoke_config()
    first = run_independent_benchmark(config, run_index=1, device=torch.device("cpu"))
    second = run_independent_benchmark(config, run_index=2, device=torch.device("cpu"))

    assert first.metadata["scenario_order"] == ["baseline", "pruning_magnitude", "quantization_int8"]
    assert second.metadata["scenario_order"] == ["pruning_magnitude", "quantization_int8", "baseline"]

    def hashes(run, field: str) -> dict[tuple[object, ...], str]:
        return {
            (
                item["operation"],
                item["batch_size"],
                item["seq_len"],
                item["dimension"],
                item["scenario"],
            ): str(item[field])
            for item in run.metadata["fingerprints"]
        }

    assert hashes(first, "input_sha256") == hashes(second, "input_sha256")
    assert hashes(first, "output_sha256") == hashes(second, "output_sha256")


def test_suite_persists_partial_runs_manifest_logs_and_fixed_schema(tmp_path: Path) -> None:
    suite = execute_experiment_suite(smoke_config(), tmp_path / "evidence")

    assert len(suite.run_csv_paths) == 2
    assert suite.log_path.exists()
    assert (suite.output_dir / "config.snapshot.json").exists()
    assert (suite.output_dir / "environment.json").exists()
    manifest = json.loads(suite.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert len(manifest["runs"]) == 2

    for csv_path in suite.run_csv_paths:
        with csv_path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        assert reader.fieldnames == CSV_COLUMNS
        assert len(rows) == 6
        assert csv_path.with_suffix(".metadata.json").exists()

    with pytest.raises(FileExistsError, match="nao esta vazio"):
        execute_experiment_suite(smoke_config(), suite.output_dir)


def test_environment_is_captured_before_output_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "evidence"
    captured = False

    def capture_before_write(_device: torch.device) -> dict[str, object]:
        nonlocal captured
        captured = True
        assert not output_dir.exists()
        return {"git": {"worktree_clean_at_start": True}}

    monkeypatch.setattr(benchmark_operacoes, "environment_metadata", capture_before_write)
    execute_experiment_suite(smoke_config(), output_dir)

    assert captured
    environment = json.loads((output_dir / "environment.json").read_text(encoding="utf-8"))
    assert environment["git"]["worktree_clean_at_start"] is True


def test_primary_environment_gate_fails_clearly_without_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_experiment_config(PHASE_ROOT / "experimentos" / "benchmark_principal_gpu.json")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ExperimentEnvironmentError, match="nao encontrou dispositivo CUDA"):
        resolve_experiment_device(config)
