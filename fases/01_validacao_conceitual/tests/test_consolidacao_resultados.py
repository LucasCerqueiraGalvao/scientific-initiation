from __future__ import annotations

import csv
import json
from pathlib import Path

from validacao.consolidacao_resultados import _dedupe_latest, consolidate_results


def test_dedupe_keeps_highest_priority_row() -> None:
    rows = [
        {"model_id": "m", "scenario": "s", "canonical_source_priority": 1, "value": "old"},
        {"model_id": "m", "scenario": "s", "canonical_source_priority": 2, "value": "new"},
        {"model_id": "m", "scenario": "other", "canonical_source_priority": 1, "value": "kept"},
    ]

    selected, duplicates = _dedupe_latest(rows, ("model_id", "scenario"))

    assert {(row["scenario"], row["value"]) for row in selected} == {
        ("s", "new"),
        ("other", "kept"),
    }
    assert duplicates[0]["value"] == "old"


def test_consolidation_writes_manifest_and_empty_outputs_for_missing_sources(tmp_path: Path) -> None:
    evidence = tmp_path / "benchmarks"
    evidence.mkdir()
    output = tmp_path / "canonical"

    manifest_path = consolidate_results(evidence, output)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert {item["path"] for item in manifest["artifacts"]} >= {
        "canonical_model_quality.csv",
        "canonical_operation_summary.csv",
        "source_selection.json",
    }
    with (output / "canonical_model_quality.csv").open(encoding="utf-8", newline="") as handle:
        assert list(csv.reader(handle)) == [[]]
