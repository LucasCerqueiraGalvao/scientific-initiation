from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from validacao.analise_benchmark_operacoes_v3 import (
    OperationAnalysisErrorV3,
    bootstrap_confidence_interval,
    load_experiment_v3,
    write_analysis_v3,
)
from validacao.attention import (
    multi_head_self_attention_numpy,
    multi_head_self_attention_torch,
)
from validacao.benchmark_operacoes import load_experiment_config
from validacao.benchmark_operacoes_v3 import (
    BenchmarkProfileV3,
    build_operation_inputs_v3,
    config_v3_from_mapping,
    exact_magnitude_prune,
    execute_experiment_suite_v3,
    load_experiment_config_v3,
    magnitude_prune_2to4,
)
from validacao.protocolo import (
    CSV_COLUMNS_V3,
    empty_benchmark_record_v3,
    validate_benchmark_record_v3,
)
from validacao.quantization import symmetric_int8_quantize_per_row


PHASE_ROOT = Path(__file__).resolve().parents[1]


def _smoke_config():
    return load_experiment_config_v3(PHASE_ROOT / "experimentos" / "smoke_v3_cpu.json")


def test_v3_config_has_balanced_contract_and_legacy_configs_still_load() -> None:
    config = load_experiment_config_v3(PHASE_ROOT / "experimentos" / "robustez_sintetica_v3.json")
    legacy = load_experiment_config(PHASE_ROOT / "experimentos" / "benchmark_principal_gpu.json")

    assert config.schema_version == 3
    assert config.data_seeds == (11, 23, 42, 101, 2026)
    assert len(config.profiles) == 13
    assert config.measurement.repetitions == 3
    assert len(config.scenarios) == 6
    expected_records = (
        len(config.operations)
        * len(config.profiles)
        * len(config.data_seeds)
        * len(config.scenarios)
        * config.measurement.repetitions
    )
    assert expected_records == 2340
    assert expected_records * config.measurement.measure_iterations == 117000
    assert legacy.schema_version == 2


def test_v3_config_rejects_non_divisible_heads_and_missing_baseline() -> None:
    data = _smoke_config().to_dict()
    data["profiles"][0]["num_heads"] = 3
    with pytest.raises(ValueError, match="divisivel"):
        config_v3_from_mapping(data)

    data = _smoke_config().to_dict()
    data["scenarios"] = [scenario for scenario in data["scenarios"] if scenario["kind"] != "baseline"]
    with pytest.raises(ValueError, match="baseline"):
        config_v3_from_mapping(data)


def test_multi_head_operation_matches_numpy_manual_and_sdpa() -> None:
    generator = np.random.default_rng(2026)
    inputs = generator.standard_normal((2, 3, 8), dtype=np.float32)
    weights = [
        (generator.standard_normal((8, 8), dtype=np.float32) / np.float32(np.sqrt(8))).astype(np.float32)
        for _ in range(4)
    ]

    expected = multi_head_self_attention_numpy(inputs, *weights, num_heads=4)
    actual_sdpa = multi_head_self_attention_torch(
        torch.from_numpy(inputs),
        *(torch.from_numpy(weight) for weight in weights),
        num_heads=4,
        use_sdpa=True,
    ).numpy()
    actual_manual = multi_head_self_attention_torch(
        torch.from_numpy(inputs),
        *(torch.from_numpy(weight) for weight in weights),
        num_heads=4,
        use_sdpa=False,
    ).numpy()

    assert np.allclose(actual_sdpa, expected, atol=1e-6, rtol=1e-5)
    assert np.allclose(actual_manual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("sparsity", [0.1, 0.25, 0.5, 0.75])
def test_pruning_levels_and_per_row_quantization_are_exact(sparsity: float) -> None:
    tensor = torch.arange(1, 41, dtype=torch.float32).reshape(5, 8)
    assert torch.count_nonzero(exact_magnitude_prune(tensor, sparsity) == 0).item() == round(40 * sparsity)

    sparse = magnitude_prune_2to4(tensor)
    assert torch.count_nonzero(sparse == 0).item() == 20
    assert (torch.count_nonzero(sparse.reshape(5, 2, 4), dim=-1) == 2).all()

    quantized, scale, dequantized = symmetric_int8_quantize_per_row(tensor)
    assert quantized.dtype == torch.int8
    assert scale.shape == (5, 1)
    assert dequantized.shape == tensor.shape
    assert torch.all((tensor - dequantized).abs().amax(dim=-1, keepdim=True) <= scale)


def test_v3_protocol_requires_status_and_complete_lineage() -> None:
    unsupported = empty_benchmark_record_v3(status="unsupported", skip_reason="kernel ausente")
    invalid = empty_benchmark_record_v3(status="complete")

    assert list(unsupported) == CSV_COLUMNS_V3
    assert validate_benchmark_record_v3(unsupported).is_valid
    assert not validate_benchmark_record_v3(invalid).is_valid


def test_v3_smoke_writes_seeds_repetitions_timings_and_paired_analysis(tmp_path: Path) -> None:
    suite = execute_experiment_suite_v3(_smoke_config(), tmp_path / "evidence")
    loaded = load_experiment_v3(suite.output_dir)
    outputs = write_analysis_v3(suite.output_dir)

    assert len(loaded.results) == 48
    assert len(loaded.comparisons) == 40
    assert len(loaded.timings) == 144
    assert loaded.results["data_seed"].nunique() == 2
    assert loaded.results["repeat_index"].nunique() == 2
    assert set(loaded.results["status"]) == {"complete"}
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.paths())

    manifest = json.loads(suite.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert len(manifest["runs"]) == 2
    with suite.run_paths[0].open(encoding="utf-8", newline="") as handle:
        assert csv.DictReader(handle).fieldnames == CSV_COLUMNS_V3


def test_v3_resume_reuses_verified_runs_and_rejects_changed_config(tmp_path: Path) -> None:
    config = _smoke_config()
    suite = execute_experiment_suite_v3(config, tmp_path / "evidence")
    manifest = json.loads(suite.manifest_path.read_text(encoding="utf-8"))
    interrupted_run = manifest["runs"].pop()
    manifest["status"] = "running"
    suite.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (suite.output_dir / interrupted_run["csv"]).unlink()
    (suite.output_dir / interrupted_run["timings"]).unlink()

    resumed = execute_experiment_suite_v3(config, suite.output_dir, resume=True)

    assert len(resumed.run_paths) == 2
    changed = config.to_dict()
    changed["data_seeds"] = [999]
    with pytest.raises(RuntimeError, match="diverge"):
        execute_experiment_suite_v3(
            config_v3_from_mapping(changed),
            suite.output_dir,
            resume=True,
        )


def test_v3_analysis_rejects_tampered_run(tmp_path: Path) -> None:
    suite = execute_experiment_suite_v3(_smoke_config(), tmp_path / "evidence")
    suite.run_paths[0].write_text(suite.run_paths[0].read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(OperationAnalysisErrorV3, match="checksum"):
        load_experiment_v3(suite.output_dir)


def test_bootstrap_is_deterministic_and_contains_constant_value() -> None:
    first = bootstrap_confidence_interval([1.0, 2.0, 3.0], resamples=1000, seed=2026)
    second = bootstrap_confidence_interval([1.0, 2.0, 3.0], resamples=1000, seed=2026)
    constant = bootstrap_confidence_interval([2.0] * 5, resamples=1000, seed=2026)
    median = bootstrap_confidence_interval(
        [1.0, 1.0, 1.0, 100.0],
        resamples=1000,
        seed=2026,
        statistic="median",
    )

    assert first == second
    assert constant == (2.0, 2.0)
    assert median[0] <= 1.0 <= median[1]

    with pytest.raises(ValueError, match="estatistica bootstrap"):
        bootstrap_confidence_interval([1.0, 2.0], statistic="mode")


def test_profile_type_documents_all_macro_parameters() -> None:
    profile = BenchmarkProfileV3("example", 4, 128, 512, 8, "standard_normal")
    assert (profile.batch_size, profile.seq_len, profile.d_model, profile.num_heads) == (4, 128, 512, 8)


def test_data_seed_is_deterministic_and_changes_tensor_hash() -> None:
    profile = BenchmarkProfileV3("hash", 1, 8, 16, 4, "standard_normal")
    first = build_operation_inputs_v3("dense_projection", profile, 11)
    repeated = build_operation_inputs_v3("dense_projection", profile, 11)
    changed = build_operation_inputs_v3("dense_projection", profile, 23)

    assert torch.equal(first["x"], repeated["x"])
    assert torch.equal(first["weight"], repeated["weight"])
    assert not torch.equal(first["x"], changed["x"])
    assert not torch.equal(first["weight"], changed["weight"])
