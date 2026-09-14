from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validacao.analise_benchmark_operacoes_v3 import (
    bootstrap_confidence_interval,
    pair_against_baselines,
)
from validacao.benchmark_modelos_opt import (
    PROMPT_COLUMNS,
    QUALITY_COLUMNS,
    TIMING_COLUMNS,
    WINDOW_COLUMNS,
    ModelExperimentConfig,
    model_config_from_mapping,
)
from validacao.protocolo import CSV_COLUMNS_V3


class ModelAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedModelExperiment:
    root: Path
    config: ModelExperimentConfig
    manifest: dict[str, object]
    quality: pd.DataFrame
    windows: pd.DataFrame
    prompts: pd.DataFrame
    performance: pd.DataFrame
    timings: pd.DataFrame
    comparisons: pd.DataFrame


@dataclass(frozen=True)
class ModelAnalysisOutputs:
    quality_summary: Path
    performance_comparisons: Path
    performance_summary: Path
    cases: Path
    quality_plot: Path
    speedup_plot: Path
    memory_plot: Path
    storage_plot: Path
    pareto_plot: Path
    report: Path
    manifest: Path

    def paths(self) -> tuple[Path, ...]:
        return tuple(value for value in self.__dict__.values())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelAnalysisError(f"JSON invalido: {path}") from exc
    if not isinstance(value, dict):
        raise ModelAnalysisError(f"JSON deve ser objeto: {path}")
    return value


def _artifact_map(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    artifacts = manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ModelAnalysisError("lista de artefatos invalida")
    result: dict[str, Mapping[str, object]] = {}
    for item in artifacts:
        if not isinstance(item, Mapping) or "path" not in item or "sha256" not in item:
            raise ModelAnalysisError("entrada de artefato invalida")
        result[str(item["path"])] = item
    return result


def _load_csv(root: Path, artifacts: Mapping[str, Mapping[str, object]], name: str, columns: list[str]) -> pd.DataFrame:
    item = artifacts.get(name)
    if item is None:
        raise ModelAnalysisError(f"artefato ausente: {name}")
    path = root / name
    if not path.is_file() or _sha256(path) != item["sha256"]:
        raise ModelAnalysisError(f"checksum divergente: {name}")
    frame = pd.read_csv(path, keep_default_na=False)
    if list(frame.columns) != columns:
        raise ModelAnalysisError(f"schema divergente: {name}")
    if len(frame) != int(item.get("records", -1)):
        raise ModelAnalysisError(f"contagem divergente: {name}")
    return frame


def _numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    copy = frame.copy()
    for column in columns:
        if column in copy:
            copy[column] = pd.to_numeric(copy[column], errors="coerce")
    return copy


def load_model_experiment(evidence_dir: str | Path) -> LoadedModelExperiment:
    root = Path(evidence_dir)
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        raise ModelAnalysisError("manifesto OPT nao esta completo")
    config = model_config_from_mapping(_read_json(root / "config.snapshot.json"))
    artifacts = _artifact_map(manifest)
    for name, item in artifacts.items():
        path = root / name
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise ModelAnalysisError(f"checksum divergente: {name}")
    quality = _load_csv(root, artifacts, "quality.csv", QUALITY_COLUMNS)
    windows = _load_csv(root, artifacts, "quality_windows.csv", WINDOW_COLUMNS)
    prompts = _load_csv(root, artifacts, "prompt_metrics.csv", PROMPT_COLUMNS)
    performance = _load_csv(root, artifacts, "performance.csv", CSV_COLUMNS_V3)
    timings = _load_csv(root, artifacts, "timings.csv", TIMING_COLUMNS)

    quality = _numeric(
        quality,
        [
            "parameter_count", "parameter_storage_bytes", "target_linear_count",
            "pruning_sparsity", "observed_sparsity", "mean_loss", "perplexity",
            "perplexity_increase_ratio", "logits_mse", "logits_cosine",
            "logits_kl_divergence", "top1_agreement", "generated_token_agreement",
            "exact_generation_rate",
        ],
    )
    performance = _numeric(
        performance,
        [
            "data_seed", "repeat_index", "batch_size", "seq_len", "d_model", "num_heads",
            "latency_ms_mean", "latency_ms_p50", "latency_ms_p95", "throughput_tokens_s",
            "peak_memory_allocated_bytes", "peak_memory_reserved_bytes",
            "parameter_storage_bytes", "mse", "cosine_similarity", "kl_divergence",
            "top1_agreement", "loss", "perplexity",
        ],
    )
    complete_quality = quality[quality["status"] == "complete"]
    if complete_quality.empty:
        raise ModelAnalysisError("nenhum resultado de qualidade completo")
    if complete_quality[["perplexity", "mean_loss"]].isna().any().any():
        raise ModelAnalysisError("resultado de qualidade completo possui numero ausente")
    if not performance.empty:
        if performance["case_id"].duplicated().any():
            raise ModelAnalysisError("case_id de desempenho duplicado")
        comparisons = pair_against_baselines(performance)
        expected_timings = len(performance[performance["status"] == "complete"]) * config.performance.measure_iterations
        if len(timings) != expected_timings:
            raise ModelAnalysisError(
                f"quantidade de timings divergente: esperada={expected_timings}, observada={len(timings)}"
            )
    else:
        comparisons = pd.DataFrame()
        if not timings.empty:
            raise ModelAnalysisError("timings existem sem registros de desempenho")
    return LoadedModelExperiment(
        root=root,
        config=config,
        manifest=manifest,
        quality=quality,
        windows=windows,
        prompts=prompts,
        performance=performance,
        timings=timings,
        comparisons=comparisons,
    )


def summarize_quality(loaded: LoadedModelExperiment) -> pd.DataFrame:
    scenario_map = {scenario.identifier: scenario for scenario in loaded.config.scenarios}
    rows: list[dict[str, object]] = []
    for row in loaded.quality.itertuples(index=False):
        scenario = scenario_map[row.scenario]
        rows.append(
            {
                "model_id": row.model_id,
                "scenario": row.scenario,
                "kind": scenario.kind,
                "comparison_group": row.comparison_group,
                "optimization_scope": row.optimization_scope,
                "status": row.status,
                "skip_reason": row.skip_reason,
                "pruning_sparsity": row.pruning_sparsity,
                "observed_sparsity": row.observed_sparsity,
                "perplexity": row.perplexity,
                "perplexity_increase_percent": (
                    row.perplexity_increase_ratio * 100
                    if row.status == "complete" and not math.isnan(row.perplexity_increase_ratio)
                    else math.nan
                ),
                "logits_mse": row.logits_mse,
                "logits_kl_divergence": row.logits_kl_divergence,
                "top1_agreement": row.top1_agreement,
                "generated_token_agreement": row.generated_token_agreement,
                "storage_megabytes": row.parameter_storage_bytes / (1024**2)
                if row.status == "complete"
                else math.nan,
                "quality_acceptable": row.quality_acceptable,
            }
        )
    return pd.DataFrame(rows)


def summarize_performance(loaded: LoadedModelExperiment) -> pd.DataFrame:
    if loaded.comparisons.empty:
        return pd.DataFrame(
            columns=[
                "model_id", "operation", "scenario", "paired_cases", "speedup_mean",
                "speedup_median", "speedup_ci95_low", "speedup_ci95_high",
                "memory_ratio_mean", "memory_ratio_median", "memory_ratio_ci95_low",
                "memory_ratio_ci95_high", "storage_ratio_mean", "kernel_confirmed", "conclusion",
            ]
        )
    rows: list[dict[str, object]] = []
    groups = ["model_id", "operation", "scenario"]
    for keys, group in loaded.comparisons.groupby(groups, sort=True):
        speedup = group["speedup_vs_baseline"].to_numpy(dtype=np.float64)
        low, high = bootstrap_confidence_interval(
            speedup,
            resamples=10000,
            seed=2026,
            statistic="median",
        )
        memory_ratio = group["peak_memory_ratio_vs_baseline"].to_numpy(dtype=np.float64)
        memory_low, memory_high = bootstrap_confidence_interval(
            memory_ratio,
            resamples=10000,
            seed=2026,
            statistic="median",
        )
        finite_memory = memory_ratio[np.isfinite(memory_ratio)]
        kernel_confirmed = bool(
            group["uses_sparse_kernel"].astype(str).str.lower().eq("true").all()
            or group["uses_int8_kernel"].astype(str).str.lower().eq("true").all()
        )
        median = float(np.median(speedup))
        conclusion = (
            "sustentada"
            if median >= 1.05 and low > 1.0 and kernel_confirmed
            else "contradita"
            if high < 1.0
            else "mista"
        )
        rows.append(
            {
                **dict(zip(groups, keys)),
                "paired_cases": len(group),
                "speedup_mean": float(np.mean(speedup)),
                "speedup_median": median,
                "speedup_ci95_low": low,
                "speedup_ci95_high": high,
                "memory_ratio_mean": (
                    float(np.mean(finite_memory)) if finite_memory.size else math.nan
                ),
                "memory_ratio_median": (
                    float(np.median(finite_memory)) if finite_memory.size else math.nan
                ),
                "memory_ratio_ci95_low": memory_low,
                "memory_ratio_ci95_high": memory_high,
                "storage_ratio_mean": float(group["storage_ratio_vs_baseline"].mean()),
                "kernel_confirmed": kernel_confirmed,
                "conclusion": conclusion,
            }
        )
    return pd.DataFrame(rows)


def _empty_plot(path: Path, title: str, message: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_quality(summary: pd.DataFrame, path: Path) -> None:
    data = summary[(summary["status"] == "complete") & (summary["kind"] == "pruning_magnitude")]
    if data.empty:
        _empty_plot(path, "Qualidade por sparsity", "Sem casos completos de pruning denso")
        return
    fig, axis = plt.subplots(figsize=(9, 5))
    for model_id, group in data.groupby("model_id"):
        values = group.groupby("pruning_sparsity")["perplexity_increase_percent"].median()
        axis.plot(values.index, values.values, marker="o", label=model_id)
    axis.axhline(5.0, color="#9d3d38", linewidth=1, linestyle="--")
    axis.set_xlabel("Sparsity")
    axis.set_ylabel("Aumento de perplexidade (%)")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_performance(summary: pd.DataFrame, path: Path, metric: str, ylabel: str) -> None:
    if summary.empty:
        _empty_plot(path, ylabel, "Sem casos completos de desempenho")
        return
    labels = [f"{row.model_id.split('/')[-1]}\n{row.operation}\n{row.scenario}" for row in summary.itertuples()]
    values = summary[metric].to_numpy(dtype=np.float64)
    fig, axis = plt.subplots(figsize=(max(9, len(labels) * 0.55), 5))
    axis.bar(labels, values, color="#3a6b5c")
    axis.axhline(1.0, color="#9d3d38", linewidth=1)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_memory(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        _empty_plot(path, "Memoria de pico", "Sem casos completos de desempenho")
        return
    labels = [f"{row.model_id.split('/')[-1]}\n{row.operation}\n{row.scenario}" for row in summary.itertuples()]
    values = summary["memory_ratio_median"].to_numpy(dtype=np.float64)
    ci_low = summary["memory_ratio_ci95_low"].to_numpy(dtype=np.float64)
    ci_high = summary["memory_ratio_ci95_high"].to_numpy(dtype=np.float64)
    positions = np.arange(len(values))
    fig, axis = plt.subplots(figsize=(max(9, len(labels) * 0.55), 5))
    axis.bar(positions, values, color="#4d6781")
    axis.vlines(positions, ci_low, ci_high, color="#202020", linewidth=1)
    axis.hlines(ci_low, positions - 0.08, positions + 0.08, color="#202020", linewidth=1)
    axis.hlines(ci_high, positions - 0.08, positions + 0.08, color="#202020", linewidth=1)
    axis.set_xticks(positions, labels)
    axis.axhline(1.0, color="#9d3d38", linewidth=1)
    axis.set_ylabel("Memoria de pico relativa ao baseline")
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_pareto(quality: pd.DataFrame, performance: pd.DataFrame, path: Path) -> None:
    if performance.empty:
        _empty_plot(path, "Fronteira qualidade e desempenho", "Sem casos completos de desempenho")
        return
    perf = performance.groupby(["model_id", "scenario"], as_index=False)["speedup_vs_baseline"].median()
    data = quality[quality["status"] == "complete"].merge(perf, on=["model_id", "scenario"])
    if data.empty:
        _empty_plot(path, "Fronteira qualidade e desempenho", "Sem intersecao entre qualidade e desempenho")
        return
    fig, axis = plt.subplots(figsize=(8, 5))
    for row in data.itertuples():
        axis.scatter(row.speedup_vs_baseline, row.perplexity_increase_percent, s=45)
        axis.annotate(f"{row.model_id.split('/')[-1]}/{row.scenario}", (row.speedup_vs_baseline, row.perplexity_increase_percent), fontsize=7)
    axis.axvline(1.0, color="#777777", linewidth=1)
    axis.set_xlabel("Speedup mediano")
    axis.set_ylabel("Aumento de perplexidade (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _report(loaded: LoadedModelExperiment, quality: pd.DataFrame, performance: pd.DataFrame) -> str:
    complete = int((quality["status"] == "complete").sum())
    unsupported = int((quality["status"] == "unsupported").sum())
    quality_failed = int((quality["status"] == "failed").sum())
    performance_complete = int((loaded.performance["status"] == "complete").sum())
    performance_unsupported = int((loaded.performance["status"] == "unsupported").sum())
    performance_failed = int((loaded.performance["status"] == "failed").sum())
    accepted = int(quality["quality_acceptable"].astype(str).str.lower().eq("true").sum())
    sustained = int((performance["conclusion"] == "sustentada").sum()) if not performance.empty else 0
    return "\n".join(
        [
            r"\chapter{Analise dos modelos OPT pre-treinados}",
            r"\section{Escopo}",
            rf"Foram configurados {len(loaded.config.models)} modelos OPT e {len(loaded.config.scenarios)} variantes por modelo, sem treinamento ou fine-tuning.",
            r"A avaliacao usa loss, perplexidade, divergencia dos logits e concordancia de geracao; desempenho e separado em prefill, TTFT e decode com KV cache.",
            r"\section{Cobertura}",
            rf"Casos de qualidade completos: {complete}; incompativeis preservados: {unsupported}; falhos: {quality_failed}; dentro do limite operacional: {accepted}.",
            rf"Casos de desempenho completos: {performance_complete}; incompativeis preservados: {performance_unsupported}; falhos: {performance_failed}.",
            r"\section{Desempenho fisico}",
            rf"Comparacoes com ganho sustentado pelos criterios pre-registrados: {sustained}.",
            r"Ganho fisico so e aceito com speedup mediano de pelo menos 1,05, intervalo de 95\% acima de 1 e kernel confirmado.",
            r"\section{Limites}",
            r"Os limites de 2\% para quantizacao e 5\% para pruning sao criterios deste experimento, nao regras universais.",
        ]
    ) + "\n"


def write_model_analysis(evidence_dir: str | Path, output_dir: str | Path | None = None) -> ModelAnalysisOutputs:
    loaded = load_model_experiment(evidence_dir)
    output = Path(output_dir) if output_dir else loaded.root / "analise_modelos"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"diretorio de analise nao esta vazio: {output}")
    output.mkdir(parents=True, exist_ok=True)

    quality = summarize_quality(loaded)
    performance = summarize_performance(loaded)
    quality_cases = loaded.quality[["model_id", "scenario", "status", "skip_reason"]].copy()
    quality_cases.insert(0, "evidence_type", "quality")
    quality_cases["operation"] = ""
    quality_cases["profile_id"] = ""
    quality_cases["repeat_index"] = ""
    performance_cases = loaded.performance[
        ["model_id", "scenario", "operation", "profile_id", "repeat_index", "status", "skip_reason"]
    ].copy()
    performance_cases.insert(0, "evidence_type", "performance")
    cases = pd.concat(
        [
            quality_cases[
                [
                    "evidence_type", "model_id", "scenario", "operation", "profile_id",
                    "repeat_index", "status", "skip_reason",
                ]
            ],
            performance_cases,
        ],
        ignore_index=True,
    )
    paths = ModelAnalysisOutputs(
        quality_summary=output / "resumo_qualidade.csv",
        performance_comparisons=output / "comparacoes_desempenho.csv",
        performance_summary=output / "resumo_desempenho.csv",
        cases=output / "casos_suportados_ignorados_falhos.csv",
        quality_plot=output / "qualidade_por_sparsity_e_modelo.png",
        speedup_plot=output / "speedup_prefill_decode.png",
        memory_plot=output / "memoria_pico_ci95.png",
        storage_plot=output / "armazenamento_relativo.png",
        pareto_plot=output / "pareto_qualidade_latencia.png",
        report=output / "relatorio_modelos_opt.tex",
        manifest=output / "analysis_manifest.json",
    )
    quality.to_csv(paths.quality_summary, index=False, lineterminator="\n")
    loaded.comparisons.to_csv(paths.performance_comparisons, index=False, lineterminator="\n")
    performance.to_csv(paths.performance_summary, index=False, lineterminator="\n")
    cases.to_csv(paths.cases, index=False, lineterminator="\n")
    _plot_quality(quality, paths.quality_plot)
    _plot_performance(performance, paths.speedup_plot, "speedup_median", "Speedup mediano")
    _plot_memory(performance, paths.memory_plot)
    _plot_performance(performance, paths.storage_plot, "storage_ratio_mean", "Armazenamento relativo")
    _plot_pareto(quality, loaded.comparisons, paths.pareto_plot)
    paths.report.write_text(_report(loaded, quality, performance), encoding="utf-8")
    artifacts = [path for path in paths.paths() if path != paths.manifest]
    manifest = {
        "schema_version": 1,
        "source_manifest_sha256": _sha256(loaded.root / "manifest.json"),
        "artifacts": [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in artifacts
        ],
    }
    paths.manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analisa qualidade e desempenho dos modelos OPT.")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--output-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = write_model_analysis(args.evidence_dir, args.output_dir)
    except (ModelAnalysisError, FileExistsError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"falha na analise OPT: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(outputs.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
