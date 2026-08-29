from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from validacao.analise_benchmark_operacoes import (
    OperationAnalysisError,
    load_experiment,
    write_operation_analysis,
)
from validacao.benchmark_operacoes import execute_experiment_suite, load_experiment_config


PHASE_ROOT = Path(__file__).resolve().parents[1]


def _create_smoke(tmp_path: Path) -> Path:
    config = load_experiment_config(PHASE_ROOT / "experimentos" / "smoke_cpu.json")
    return execute_experiment_suite(config, tmp_path / "evidence").output_dir


def test_analysis_validates_reproducibility_and_generates_complete_outputs(tmp_path: Path) -> None:
    evidence_dir = _create_smoke(tmp_path)
    loaded = load_experiment(evidence_dir)
    outputs = write_operation_analysis(evidence_dir)

    assert loaded.reproducibility["passed"] is True
    assert len(loaded.results) == 12
    assert len(loaded.comparisons) == 8
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.paths())

    summary = pd.read_csv(outputs.summary)
    comparison_summary = pd.read_csv(outputs.comparison_summary)
    report = outputs.report.read_text(encoding="utf-8")
    assert outputs.report.suffix == ".tex"
    assert r"\chapter{Relatório preliminar" in report
    assert set(summary["scenario"]) == {"baseline", "pruning_magnitude", "quantization_int8"}
    assert (summary["independent_runs"] == 2).all()
    assert set(comparison_summary["scenario"]) == {"pruning_magnitude", "quantization_int8"}
    assert "não sustentam conclusão de hardware" in report
    assert r"Pergunta $\rightarrow$ H1" in report

    with pytest.raises(FileExistsError, match="nao esta vazio"):
        write_operation_analysis(evidence_dir)


def test_analysis_rejects_tampered_csv_by_manifest_checksum(tmp_path: Path) -> None:
    evidence_dir = _create_smoke(tmp_path)
    manifest = json.loads((evidence_dir / "manifest.json").read_text(encoding="utf-8"))
    csv_path = evidence_dir / manifest["runs"][0]["csv"]
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(OperationAnalysisError, match="checksum do CSV divergiu"):
        load_experiment(evidence_dir)
