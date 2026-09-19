from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence


CANONICAL_OPERATION_SOURCES = (
    "robustez_sintetica_v3_2026-09-13",
    "hardware_nativo_v3_2026-09-14",
    "hardware_stress_complementar_v3_2026-09-17",
    "hardware_crossover_v3_2026-09-19",
)

CANONICAL_MODEL_SOURCES = (
    "modelos_opt_v1_2026-09-15",
    "modelos_opt_2_7b_2026-09-15",
    "modelos_opt_6_7b_2026-09-16",
    "modelos_opt_complementar_lacunas_2026-09-17",
    "modelos_opt_6_7b_complementar_lacunas_2026-09-17",
    "modelos_opt_sensibilidade_350m_2026-09-19",
    "modelos_opt_sensibilidade_1_3b_2026-09-19",
    "modelos_opt_hibridos_1_3b_quality_2026-09-19",
    "modelos_opt_hibridos_1_3b_finalistas_2026-09-19",
    "modelos_opt_hibridos_6_7b_quality_2026-09-19",
)

EXCLUDED_SOURCES = (
    "smoke_cpu_2026-08-25",
    "smoke_hardware_v3_2026-09-14",
    "smoke_hardware_stress_complementar_2026-09-17",
    "modelos_opt_smoke_2026-09-15",
    "modelos_opt_complementar_smoke_2026-09-17",
    "diagnostico_cpu_l64_d128_2026-08-25",
    "diagnostico_cpu_l64_d128_v2_2026-08-25",
    "benchmark_gpu_4070ti_super_2026-08-29",
)


@dataclass(frozen=True)
class SourceRecord:
    name: str
    priority: int
    purpose: str
    path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], columns: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = seen
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_manifest(source: SourceRecord) -> dict[str, object]:
    manifest = source.path / "manifest.json"
    config = source.path / "config.snapshot.json"
    environment = source.path / "environment.json"
    result: dict[str, object] = {
        "name": source.name,
        "purpose": source.purpose,
        "priority": source.priority,
        "path": str(source.path),
        "manifest": manifest.name if manifest.exists() else "",
        "manifest_sha256": _sha256(manifest) if manifest.exists() else "",
        "config": config.name if config.exists() else "",
        "config_sha256": _sha256(config) if config.exists() else "",
        "environment": environment.name if environment.exists() else "",
        "environment_sha256": _sha256(environment) if environment.exists() else "",
    }
    if manifest.exists():
        payload = _read_json(manifest)
        if isinstance(payload, Mapping):
            result["status"] = payload.get("status", "")
            result["schema_version"] = payload.get("schema_version", "")
    return result


def _tag_rows(rows: Iterable[Mapping[str, object]], source: SourceRecord, table: str) -> list[dict[str, object]]:
    return [
        {
            **dict(row),
            "canonical_source": source.name,
            "canonical_source_priority": source.priority,
            "canonical_table": table,
        }
        for row in rows
    ]


def _to_float(value: object) -> float | None:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _method_family(scenario: object) -> str:
    name = str(scenario).lower()
    if name.startswith("baseline"):
        return "baseline"
    if "int8_fake" in name:
        return "int8_fake"
    if "int8_dynamic" in name:
        return "int8_dynamic"
    if "int8_weight_only" in name:
        return "int8_weight_only"
    if "2to4" in name or "2:4" in name:
        return "pruning_2to4"
    if "pruning" in name:
        return "pruning_unstructured"
    return "other"


def _evidence_level(scenario: object) -> str:
    family = _method_family(scenario)
    if family == "baseline":
        return "reference"
    if family in {"int8_fake", "pruning_unstructured"}:
        return "A_numeric_change"
    if family == "int8_weight_only":
        return "B_physical_compression"
    if family in {"int8_dynamic", "pruning_2to4"}:
        return "C_physical_acceleration_candidate"
    return "unclassified"


def _measurement_label(operation: object) -> str:
    if str(operation) == "model_ttft":
        return "prefill_to_first_logit"
    return str(operation)


def _enrich_quality_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        increase = _to_float(item.get("perplexity_increase_percent"))
        item["method_family"] = _method_family(item.get("scenario", ""))
        item["evidence_level"] = _evidence_level(item.get("scenario", ""))
        for threshold in (2, 5, 10):
            key = f"delta_ppl_le_{threshold}pct"
            item[key] = "" if increase is None else increase <= threshold
        enriched.append(item)
    return enriched


def _enrich_performance_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for row in rows:
        item = dict(row)
        item["method_family"] = _method_family(item.get("scenario", ""))
        item["evidence_level"] = _evidence_level(item.get("scenario", ""))
        item["measurement_label"] = _measurement_label(item.get("operation", ""))
        if "kernel_confirmed" in item:
            kernel_confirmed = str(item.get("kernel_confirmed", "")).lower() == "true"
            item["physical_kernel_status"] = "confirmed" if kernel_confirmed else "not_confirmed"
        elif "uses_sparse_kernel" in item or "uses_int8_kernel" in item:
            sparse = str(item.get("uses_sparse_kernel", "")).lower() == "true"
            int8 = str(item.get("uses_int8_kernel", "")).lower() == "true"
            item["physical_kernel_status"] = "confirmed" if sparse or int8 else "not_confirmed"
        else:
            item["physical_kernel_status"] = ""
        enriched.append(item)
    return enriched


def _dedupe_latest(
    rows: Sequence[Mapping[str, object]],
    key_fields: Sequence[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected: dict[tuple[str, ...], dict[str, object]] = {}
    duplicates: list[dict[str, object]] = []
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        current = selected.get(key)
        if current is None:
            selected[key] = dict(row)
            continue
        current_priority = int(current.get("canonical_source_priority", -1))
        row_priority = int(row.get("canonical_source_priority", -1))
        if row_priority >= current_priority:
            duplicates.append({**current, "canonical_duplicate_resolution": "replaced_by_higher_or_equal_priority"})
            selected[key] = dict(row)
        else:
            duplicates.append({**dict(row), "canonical_duplicate_resolution": "discarded_lower_priority"})
    return list(selected.values()), duplicates


def _artifact_record(path: Path, rows: int) -> dict[str, object]:
    return {
        "path": path.name,
        "records": rows,
        "sha256": _sha256(path),
    }


def consolidate_results(
    evidence_root: str | Path,
    output_dir: str | Path,
) -> Path:
    evidence = Path(evidence_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    sources: list[SourceRecord] = []
    for priority, name in enumerate(CANONICAL_OPERATION_SOURCES, start=10):
        purpose = "synthetic_stress" if "stress" in name else "synthetic_hardware" if "hardware" in name else "synthetic_numerical"
        sources.append(SourceRecord(name, priority, purpose, evidence / name))
    for priority, name in enumerate(CANONICAL_MODEL_SOURCES, start=100):
        sources.append(SourceRecord(name, priority, "opt_pretrained", evidence / name))

    accepted_sources = [_source_manifest(source) for source in sources]
    excluded_sources = [
        {
            "name": name,
            "path": str(evidence / name),
            "reason": "smoke, diagnostico historico ou run substituido por evidencia posterior",
            "exists": (evidence / name).exists(),
        }
        for name in EXCLUDED_SOURCES
    ]

    operation_summary: list[dict[str, object]] = []
    operation_complete: list[dict[str, object]] = []
    model_quality: list[dict[str, object]] = []
    model_performance_summary: list[dict[str, object]] = []
    model_performance_raw: list[dict[str, object]] = []

    for source in sources:
        if source.purpose.startswith("synthetic"):
            operation_summary.extend(
                _tag_rows(_read_csv(source.path / "analise_v3" / "resumo_estatistico.csv"), source, "operation_summary")
            )
            operation_complete.extend(
                _tag_rows(_read_csv(source.path / "analise_v3" / "resultados_completos.csv"), source, "operation_complete")
            )
        else:
            model_quality.extend(
                _tag_rows(_read_csv(source.path / "analise_modelos" / "resumo_qualidade.csv"), source, "model_quality")
            )
            model_performance_summary.extend(
                _tag_rows(_read_csv(source.path / "analise_modelos" / "resumo_desempenho.csv"), source, "model_performance_summary")
            )
            model_performance_raw.extend(
                _tag_rows(_read_csv(source.path / "performance.csv"), source, "model_performance_raw")
            )

    canonical_quality, duplicate_quality = _dedupe_latest(
        model_quality,
        ("model_id", "scenario", "comparison_group", "optimization_scope"),
    )
    canonical_perf_summary, duplicate_perf_summary = _dedupe_latest(
        model_performance_summary,
        ("model_id", "operation", "scenario"),
    )
    canonical_perf_raw, duplicate_perf_raw = _dedupe_latest(
        model_performance_raw,
        ("case_id",),
    )

    canonical_quality = _enrich_quality_rows(canonical_quality)
    canonical_perf_summary = _enrich_performance_rows(canonical_perf_summary)
    canonical_perf_raw = _enrich_performance_rows(canonical_perf_raw)

    artifacts: list[dict[str, object]] = []
    outputs = {
        "canonical_operation_summary.csv": operation_summary,
        "canonical_operation_complete.csv": operation_complete,
        "canonical_model_quality.csv": canonical_quality,
        "canonical_model_performance_summary.csv": canonical_perf_summary,
        "canonical_model_performance_raw.csv": canonical_perf_raw,
        "discarded_model_quality_duplicates.csv": duplicate_quality,
        "discarded_model_performance_summary_duplicates.csv": duplicate_perf_summary,
        "discarded_model_performance_raw_duplicates.csv": duplicate_perf_raw,
    }
    for name, rows in outputs.items():
        path = output / name
        _write_csv(path, rows)
        artifacts.append(_artifact_record(path, len(rows)))

    source_report_path = output / "source_selection.json"
    source_report = {
        "accepted_sources": accepted_sources,
        "excluded_sources": excluded_sources,
        "canonical_rules": {
            "smoke_policy": "smokes e diagnosticos historicos sao preservados fora das conclusoes finais",
            "stress_policy": "hardware_stress_complementar e mantido separado por canonical_source",
            "dedupe_policy": "chaves repetidas usam maior canonical_source_priority; empates preservam a ultima fonte lida",
            "quality_acceptable_policy": "campo original preservado; colunas delta_ppl_le_2pct/5pct/10pct sao exploratorias",
            "evidence_levels": {
                "A_numeric_change": "altera numericamente pesos/saidas sem garantir caminho fisico de armazenamento ou kernel",
                "B_physical_compression": "reduz armazenamento fisico de pesos, sem implicar aceleracao",
                "C_physical_acceleration_candidate": "usa caminho potencialmente acelerado; exige physical_kernel_status=confirmed para conclusao forte",
            },
            "ttft_policy": "operation=model_ttft foi preservado para compatibilidade; measurement_label=prefill_to_first_logit descreve melhor a metrica atual",
        },
    }
    source_report_path.write_text(json.dumps(source_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    artifacts.append(_artifact_record(source_report_path, len(accepted_sources) + len(excluded_sources)))

    manifest_path = output / "canonical_manifest.json"
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "evidence_root": str(evidence),
        "artifacts": artifacts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolida evidencias canonicas da pesquisa.")
    parser.add_argument(
        "--evidence-root",
        default="fases/01_validacao_conceitual/evidencias/benchmarks",
    )
    parser.add_argument(
        "--output",
        default="output/canonical",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = consolidate_results(args.evidence_root, args.output)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
