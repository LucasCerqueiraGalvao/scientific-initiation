from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from validacao import comparacao_frameworks
from validacao.attention import scaled_dot_product_attention_numpy
from validacao.comparacao_frameworks import (
    CANDIDATE_NAMES,
    COMPARISON_COLUMNS,
    ComparisonDependencyError,
    FrameworkComparisonConfig,
    FrameworkComparisonRun,
    build_attention_comparison_cases,
    keras_to_pytorch_layout,
    pytorch_to_keras_layout,
    run_framework_comparison,
    write_comparison_outputs,
)


def test_layout_adapter_round_trip_preserves_values() -> None:
    original = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)

    keras_layout = pytorch_to_keras_layout(original)
    restored = keras_to_pytorch_layout(keras_layout)

    assert keras_layout.shape == (2, 4, 3, 5)
    assert restored.shape == original.shape
    assert np.array_equal(restored, original)


def test_layout_adapter_rejects_non_attention_tensor() -> None:
    with pytest.raises(ValueError, match="quatro dimensoes"):
        pytorch_to_keras_layout(np.zeros((2, 3, 4), dtype=np.float32))


def test_tiny_case_matches_known_manual_value() -> None:
    tiny = build_attention_comparison_cases()[0]
    actual = scaled_dot_product_attention_numpy(tiny.query, tiny.key, tiny.value)
    expected = np.array(
        [[[[1.6604769, 2.6604769], [2.3395231, 3.3395231]]]],
        dtype=np.float32,
    )

    assert actual.shape == (1, 1, 2, 2)
    assert np.allclose(actual, expected, atol=1e-6)


def test_missing_tensorflow_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_import(name: str):
        if name == "tensorflow":
            raise ModuleNotFoundError("tensorflow")
        raise AssertionError(f"import inesperado: {name}")

    monkeypatch.setattr(comparacao_frameworks.importlib, "import_module", missing_import)

    with pytest.raises(ComparisonDependencyError, match="requirements-comparacao.txt"):
        comparacao_frameworks._load_keras_tensorflow()


def test_non_tensorflow_keras_backend_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tensorflow = SimpleNamespace(__version__="test")
    fake_keras = SimpleNamespace(
        __version__="test",
        backend=SimpleNamespace(backend=lambda: "jax"),
    )

    def fake_import(name: str):
        return {"tensorflow": fake_tensorflow, "keras": fake_keras}[name]

    monkeypatch.setattr(comparacao_frameworks.importlib, "import_module", fake_import)

    with pytest.raises(ComparisonDependencyError, match="backend Keras deve ser tensorflow"):
        comparacao_frameworks._load_keras_tensorflow()


@pytest.fixture(scope="module")
def comparison_run() -> FrameworkComparisonRun:
    pytest.importorskip("tensorflow", reason="instale requirements-comparacao.txt")
    return run_framework_comparison(FrameworkComparisonConfig(seed=2026))


def test_all_frameworks_match_in_every_case(comparison_run: FrameworkComparisonRun) -> None:
    assert len(comparison_run.records) == 9
    assert comparison_run.passed
    assert {record["candidate"] for record in comparison_run.records} == set(CANDIDATE_NAMES)
    assert {record["case_id"] for record in comparison_run.records} == {
        "tiny_manual_unmasked",
        "seeded_multihead_unmasked",
        "seeded_multihead_additive_mask",
    }

    for record in comparison_run.records:
        assert list(record) == COMPARISON_COLUMNS
        assert record["passed"] is True
        assert np.isfinite(float(record["max_abs_error"]))
        assert np.isfinite(float(record["mse"]))
        assert float(record["cosine_similarity"]) == pytest.approx(1.0, abs=1e-6)


def test_additive_mask_case_covers_every_candidate(comparison_run: FrameworkComparisonRun) -> None:
    masked_records = [
        record
        for record in comparison_run.records
        if record["case_id"] == "seeded_multihead_additive_mask"
    ]

    assert len(masked_records) == 3
    assert {record["candidate"] for record in masked_records} == set(CANDIDATE_NAMES)
    assert all(record["mask_kind"] == "additive_bias" for record in masked_records)
    assert all(record["passed"] for record in masked_records)


def test_comparison_outputs_are_complete_and_auditable(
    tmp_path: Path,
    comparison_run: FrameworkComparisonRun,
) -> None:
    csv_path, latex_path, metadata_path = write_comparison_outputs(comparison_run, tmp_path)

    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        assert reader.fieldnames == COMPARISON_COLUMNS

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    latex = latex_path.read_text(encoding="utf-8")

    assert len(rows) == 9
    assert all(row["passed"] == "True" for row in rows)
    assert metadata["comparison_kind"] == "numerical_correctness"
    assert metadata["performance_claim"] is False
    assert metadata["keras_backend"] == "tensorflow"
    assert metadata["tensorflow_version"]
    assert metadata["keras_version"]
    assert latex_path.suffix == ".tex"
    assert r"\chapter{Comparação numérica" in latex
    assert "não compara desempenho" in latex
    assert "Todos os casos ficaram dentro" in latex


def test_cli_returns_dependency_error_without_tensorflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def dependency_error(_config: FrameworkComparisonConfig) -> FrameworkComparisonRun:
        raise ComparisonDependencyError("instale requirements-comparacao.txt")

    monkeypatch.setattr(comparacao_frameworks, "run_framework_comparison", dependency_error)

    exit_code = comparacao_frameworks.main(["--output-dir", str(tmp_path)])

    assert exit_code == 2
    assert "requirements-comparacao.txt" in capsys.readouterr().err
