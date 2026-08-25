from __future__ import annotations

import pandas as pd
import pytest
import torch

from validacao.analise_resultados import (
    build_conclusions,
    compare_against_baseline,
    load_and_validate_results,
    write_article_outputs,
)
from validacao.benchmark_controlado import (
    ControlledBenchmarkConfig,
    HardwareEvidence,
    ScenarioValidationError,
    apply_scenario,
    promoted_validity_level,
    run_controlled_benchmark,
    validate_scenarios_registered,
    write_benchmark_outputs,
)
from validacao.pesquisa import RESEARCH_QUESTIONS, validate_research_contract
from validacao.protocolo import CSV_COLUMNS, validate_benchmark_record


def tiny_cpu_config(*, scenarios: tuple[str, ...] = ("baseline", "pruning_magnitude", "quantization_int8")) -> ControlledBenchmarkConfig:
    return ControlledBenchmarkConfig(
        batch_sizes=(1,),
        seq_lens=(3,),
        d_models=(4,),
        num_heads=(2,),
        scenarios=scenarios,
        seed=123,
        warmup=0,
        repetitions=2,
        device="cpu",
    )


def test_research_questions_have_metrics_evidence_and_hypotheses() -> None:
    issues = validate_research_contract()
    metric_names = {metric for question in RESEARCH_QUESTIONS for metric in question.metrics}

    assert issues == ()
    assert {"mse", "mae", "r2", "cosine_similarity", "validity_level"}.issubset(metric_names)


def test_scenario_gate_rejects_unregistered_scenario() -> None:
    with pytest.raises(ScenarioValidationError, match="cenario sem matriz"):
        validate_scenarios_registered(("baseline", "cenario_sem_evidencia"))


def test_validity_levels_do_not_promote_pruning_or_quantization_without_hardware_evidence() -> None:
    assert promoted_validity_level("pruning_magnitude", HardwareEvidence()) == "conceitual_algoritmico"
    assert (
        promoted_validity_level(
            "quantization_int8",
            HardwareEvidence(uses_low_precision_storage=True, uses_low_precision_kernel=False),
        )
        == "numerico"
    )


def test_manual_quantization_keeps_runtime_weights_float32_and_has_no_hardware_evidence() -> None:
    model = torch.nn.Linear(4, 4).eval()

    evidence = apply_scenario(model, "quantization_int8", pruning_sparsity=0.5)

    assert model.weight.dtype == torch.float32
    assert model._scientific_validation_quantized_dtype == "torch.int8"
    assert not evidence.uses_low_precision_storage
    assert not evidence.uses_low_precision_kernel


def test_pilot_defaults_match_first_formal_scenarios() -> None:
    assert ControlledBenchmarkConfig().scenarios == (
        "baseline",
        "pruning_magnitude",
        "quantization_int8",
    )
    assert (
        promoted_validity_level(
            "torchao_int8",
            HardwareEvidence(uses_low_precision_storage=True, uses_low_precision_kernel=True),
        )
        == "hardware"
    )


def test_controlled_benchmark_records_environment_seed_metrics_and_schema() -> None:
    run = run_controlled_benchmark(tiny_cpu_config())

    assert len(run.records) == 3
    assert run.metadata["seed"] == 123
    assert run.metadata["device"] == "cpu"
    assert not run.metadata["used_cuda_synchronization"]

    scenarios = {record["scenario"] for record in run.records}
    assert scenarios == {"baseline", "pruning_magnitude", "quantization_int8"}

    for record in run.records:
        assert list(record.keys()) == CSV_COLUMNS
        assert validate_benchmark_record(record).is_valid
        assert record["torch_version"]
        assert record["latency_ms_mean"] > 0.0
        assert record["throughput_tokens_s"] > 0.0
        assert record["theoretical_flops"] == 996

    baseline = next(record for record in run.records if record["scenario"] == "baseline")
    pruning = next(record for record in run.records if record["scenario"] == "pruning_magnitude")
    quantized = next(record for record in run.records if record["scenario"] == "quantization_int8")

    assert baseline["mse"] == pytest.approx(0.0)
    assert baseline["r2"] == pytest.approx(1.0)
    assert baseline["cosine_similarity"] == pytest.approx(1.0)
    assert pruning["validity_level"] == "conceitual_algoritmico"
    assert quantized["validity_level"] == "numerico"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA indisponivel")
def test_controlled_benchmark_uses_cuda_synchronization_when_device_is_cuda() -> None:
    run = run_controlled_benchmark(tiny_cpu_config(scenarios=("baseline",)).__class__(
        batch_sizes=(1,),
        seq_lens=(2,),
        d_models=(4,),
        num_heads=(2,),
        scenarios=("baseline",),
        seed=123,
        warmup=0,
        repetitions=1,
        device="cuda",
    ))

    assert run.metadata["device"].startswith("cuda")
    assert run.metadata["used_cuda_synchronization"]


def test_benchmark_outputs_and_article_analysis_are_schema_valid(tmp_path) -> None:
    run = run_controlled_benchmark(tiny_cpu_config())
    csv_path = tmp_path / "benchmark.csv"
    output_dir = tmp_path / "analise"

    write_benchmark_outputs(run, csv_path)
    dataframe = load_and_validate_results(csv_path)
    comparisons = compare_against_baseline(dataframe)
    conclusions = build_conclusions(comparisons)
    generated = write_article_outputs(csv_path, output_dir)

    assert (tmp_path / "benchmark.metadata.json").exists()
    assert isinstance(dataframe, pd.DataFrame)
    assert len(comparisons) == 2
    assert conclusions
    assert all(conclusion.evidence_source.startswith("CSV:") for conclusion in conclusions)
    assert all(conclusion.matrix_entry for conclusion in conclusions)
    assert all(path.exists() for path in generated)
