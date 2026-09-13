from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validacao.benchmark_operacoes_v3 import OperationExperimentConfigV3, config_v3_from_mapping
from validacao.protocolo import CSV_COLUMNS_V3


class OperationAnalysisErrorV3(ValueError):
    pass


@dataclass(frozen=True)
class LoadedExperimentV3:
    evidence_dir: Path
    config: OperationExperimentConfigV3
    manifest: dict[str, object]
    environment: dict[str, object]
    results: pd.DataFrame
    timings: pd.DataFrame
    comparisons: pd.DataFrame


@dataclass(frozen=True)
class AnalysisOutputsV3:
    enriched_results: Path
    comparisons: Path
    summary: Path
    hypotheses: Path
    unsupported: Path
    timings: Path
    report: Path
    speedup_plot: Path
    pruning_plot: Path
    seed_plot: Path
    heatmap_plot: Path
    pareto_plot: Path
    manifest: Path

    def paths(self) -> tuple[Path, ...]:
        return tuple(self.__dict__.values())


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperationAnalysisErrorV3(f"JSON invalido ou ausente: {path}") from exc
    if not isinstance(value, dict):
        raise OperationAnalysisErrorV3(f"JSON deve conter objeto: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise OperationAnalysisErrorV3(f"artefato ausente: {path}") from exc
    return digest.hexdigest()


def _verify_artifact(root: Path, name: object, expected_hash: object) -> Path:
    path = root / str(name)
    if _sha256(path) != str(expected_hash):
        raise OperationAnalysisErrorV3(f"checksum divergiu: {path.name}")
    return path


def _numeric(dataframe: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = dataframe.copy()
    for column in columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def pair_against_baselines(results: pd.DataFrame) -> pd.DataFrame:
    complete = results[results["status"] == "complete"].copy()
    baselines = complete.set_index("case_id", drop=False)
    rows: list[dict[str, object]] = []
    for candidate in complete.itertuples(index=False):
        if candidate.case_id == candidate.baseline_case_id:
            continue
        if candidate.baseline_case_id not in baselines.index:
            raise OperationAnalysisErrorV3(
                f"baseline ausente para caso {candidate.case_id}: {candidate.baseline_case_id}"
            )
        baseline = baselines.loc[candidate.baseline_case_id]
        if isinstance(baseline, pd.DataFrame):
            raise OperationAnalysisErrorV3(f"case_id duplicado: {candidate.baseline_case_id}")
        for field in (
            "data_seed",
            "repeat_index",
            "profile_id",
            "model_id",
            "model_revision",
            "operation",
            "comparison_group",
        ):
            if str(getattr(candidate, field)) != str(baseline[field]):
                raise OperationAnalysisErrorV3(f"pareamento divergente no campo {field}")
        rows.append(
            {
                "case_id": candidate.case_id,
                "baseline_case_id": candidate.baseline_case_id,
                "stage": candidate.stage,
                "model_id": candidate.model_id,
                "model_revision": candidate.model_revision,
                "operation": candidate.operation,
                "scenario": candidate.scenario,
                "comparison_group": candidate.comparison_group,
                "profile_id": candidate.profile_id,
                "data_seed": candidate.data_seed,
                "repeat_index": candidate.repeat_index,
                "input_distribution": candidate.input_distribution,
                "batch_size": candidate.batch_size,
                "seq_len": candidate.seq_len,
                "d_model": candidate.d_model,
                "num_heads": candidate.num_heads,
                "pruning_sparsity": candidate.pruning_sparsity,
                "speedup_vs_baseline": float(baseline["latency_ms_mean"]) / candidate.latency_ms_mean,
                "throughput_ratio_vs_baseline": candidate.throughput_tokens_s / float(baseline["throughput_tokens_s"]),
                "peak_memory_ratio_vs_baseline": (
                    candidate.peak_memory_allocated_bytes / float(baseline["peak_memory_allocated_bytes"])
                    if float(baseline["peak_memory_allocated_bytes"]) > 0
                    else math.nan
                ),
                "storage_ratio_vs_baseline": (
                    candidate.parameter_storage_bytes / float(baseline["parameter_storage_bytes"])
                    if float(baseline["parameter_storage_bytes"]) > 0
                    else math.nan
                ),
                "mse": candidate.mse,
                "mae": candidate.mae,
                "r2": candidate.r2,
                "cosine_similarity": candidate.cosine_similarity,
                "uses_sparse_kernel": candidate.uses_sparse_kernel,
                "uses_int8_kernel": candidate.uses_int8_kernel,
            }
        )
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(
        columns=[
            "case_id", "baseline_case_id", "stage", "model_id", "model_revision",
            "operation", "scenario", "comparison_group", "profile_id", "data_seed",
            "repeat_index", "input_distribution", "batch_size", "seq_len", "d_model",
            "num_heads", "pruning_sparsity", "speedup_vs_baseline",
            "throughput_ratio_vs_baseline", "peak_memory_ratio_vs_baseline",
            "storage_ratio_vs_baseline", "mse", "mae", "r2", "cosine_similarity",
            "uses_sparse_kernel", "uses_int8_kernel",
        ]
    )


def load_experiment_v3(evidence_dir: str | Path) -> LoadedExperimentV3:
    root = Path(evidence_dir)
    manifest = _read_json(root / "manifest.json")
    if manifest.get("schema_version") != 3 or manifest.get("status") != "complete":
        raise OperationAnalysisErrorV3("manifesto v3 nao esta completo")
    config_data = _read_json(root / str(manifest["config"]))
    config = config_v3_from_mapping(config_data)
    environment = _read_json(root / str(manifest["environment"]))
    frames: list[pd.DataFrame] = []
    for item in manifest.get("runs", []):
        if not isinstance(item, dict):
            raise OperationAnalysisErrorV3("entrada de run invalida")
        csv_path = _verify_artifact(root, item["csv"], item["csv_sha256"])
        _verify_artifact(root, item["timings"], item["timings_sha256"])
        frame = pd.read_csv(csv_path, keep_default_na=False)
        if list(frame.columns) != CSV_COLUMNS_V3:
            raise OperationAnalysisErrorV3(f"schema CSV v3 divergente: {csv_path.name}")
        frames.append(frame)
    if not frames:
        raise OperationAnalysisErrorV3("nenhum run no manifesto")
    results = pd.concat(frames, ignore_index=True)
    numeric_columns = [
        "data_seed", "repeat_index", "batch_size", "seq_len", "d_model", "num_heads",
        "pruning_sparsity", "observed_sparsity", "latency_ms_mean", "latency_ms_p50",
        "latency_ms_p95", "throughput_tokens_s", "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes", "parameter_storage_bytes", "theoretical_flops",
        "mse", "mae", "r2", "cosine_similarity",
    ]
    results = _numeric(results, numeric_columns)
    complete = results[results["status"] == "complete"]
    required_numeric = [
        "latency_ms_mean", "latency_ms_p50", "latency_ms_p95", "throughput_tokens_s",
        "parameter_storage_bytes", "mse", "mae", "r2", "cosine_similarity",
    ]
    if complete[required_numeric].isna().any().any():
        raise OperationAnalysisErrorV3("registro completo contem numero ausente")
    if (complete[["latency_ms_mean", "throughput_tokens_s", "parameter_storage_bytes"]] <= 0).any().any():
        raise OperationAnalysisErrorV3("latencia, throughput e armazenamento devem ser positivos")
    if results["case_id"].duplicated().any():
        raise OperationAnalysisErrorV3("case_id duplicado")
    timings_path = _verify_artifact(root, manifest["timings"], manifest["timings_sha256"])
    timings = pd.read_csv(timings_path)
    expected_samples = len(complete) * config.measurement.measure_iterations
    if len(timings) != expected_samples:
        raise OperationAnalysisErrorV3(
            f"quantidade de timings divergente: esperada={expected_samples}, observada={len(timings)}"
        )
    return LoadedExperimentV3(
        evidence_dir=root,
        config=config,
        manifest=manifest,
        environment=environment,
        results=results,
        timings=timings,
        comparisons=pair_against_baselines(results),
    )


def bootstrap_confidence_interval(
    values: Sequence[float] | np.ndarray,
    *,
    resamples: int = 10000,
    seed: int = 2026,
    statistic: str = "mean",
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return math.nan, math.nan
    if array.size == 1:
        return float(array[0]), float(array[0])
    if statistic not in {"mean", "median"}:
        raise ValueError(f"estatistica bootstrap nao suportada: {statistic}")
    generator = np.random.default_rng(seed)
    sample_indices = generator.integers(0, array.size, size=(resamples, array.size))
    samples = array[sample_indices]
    estimates = samples.mean(axis=1) if statistic == "mean" else np.median(samples, axis=1)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def summarize_comparisons_v3(
    comparisons: pd.DataFrame,
    *,
    resamples: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = ["stage", "operation", "scenario", "comparison_group"]
    for group_key, group in comparisons.groupby(groups, sort=True, dropna=False):
        speedup = group["speedup_vs_baseline"].to_numpy(dtype=np.float64)
        ci_low, ci_high = bootstrap_confidence_interval(
            speedup,
            resamples=resamples,
            seed=seed,
            statistic="median",
        )
        hardware_confirmed = bool(
            group["uses_sparse_kernel"].astype(str).str.lower().eq("true").all()
            or group["uses_int8_kernel"].astype(str).str.lower().eq("true").all()
        )
        median_speedup = float(np.median(speedup))
        if median_speedup >= 1.05 and ci_low > 1.0 and hardware_confirmed:
            conclusion = "sustentada"
        elif ci_high < 1.0:
            conclusion = "contradita"
        else:
            conclusion = "mista"
        rows.append(
            {
                **dict(zip(groups, group_key)),
                "paired_cases": len(group),
                "data_seeds": group["data_seed"].nunique(),
                "repetitions": group["repeat_index"].nunique(),
                "speedup_mean": float(np.mean(speedup)),
                "speedup_median": median_speedup,
                "speedup_std": float(np.std(speedup, ddof=1)) if len(speedup) > 1 else 0.0,
                "speedup_ci95_low": ci_low,
                "speedup_ci95_high": ci_high,
                "storage_ratio_mean": float(group["storage_ratio_vs_baseline"].mean()),
                "mse_mean": float(group["mse"].mean()),
                "mae_mean": float(group["mae"].mean()),
                "r2_mean": float(group["r2"].mean()),
                "cosine_similarity_mean": float(group["cosine_similarity"].mean()),
                "hardware_kernel_confirmed": hardware_confirmed,
                "performance_conclusion": conclusion,
            }
        )
    return pd.DataFrame(rows)


def summarize_hypotheses(comparisons: pd.DataFrame, *, resamples: int, seed: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["operation", "profile_id", "data_seed", "repeat_index"]
    fake = comparisons[comparisons["scenario"] == "int8_fake_per_row"]
    for pruning_name in ("pruning_10", "pruning_25", "pruning_50", "pruning_75"):
        pruning = comparisons[comparisons["scenario"] == pruning_name]
        merged = pruning.merge(fake, on=keys, suffixes=("_pruning", "_int8"))
        if merged.empty:
            continue
        difference = merged["mse_pruning"] - merged["mse_int8"]
        ci_low, ci_high = bootstrap_confidence_interval(difference, resamples=resamples, seed=seed)
        rate = float((difference > 0).mean())
        conclusion = "sustentada" if rate >= 0.8 and ci_low > 0 else "contradita" if ci_high < 0 else "mista"
        rows.append(
            {
                "hypothesis": "H1_H5",
                "comparison": f"int8_vs_{pruning_name}",
                "paired_cases": len(merged),
                "int8_lower_mse_rate": rate,
                "mse_difference_mean": float(difference.mean()),
                "mse_difference_ci95_low": ci_low,
                "mse_difference_ci95_high": ci_high,
                "conclusion": conclusion,
            }
        )
    pruning_all = comparisons[comparisons["scenario"].str.startswith("pruning_")].copy()
    if not pruning_all.empty:
        median_by_level = pruning_all.groupby("pruning_sparsity")["mse"].median().sort_index()
        monotonic = bool(median_by_level.is_monotonic_increasing)
        rows.append(
            {
                "hypothesis": "H6",
                "comparison": "mse_vs_sparsity",
                "paired_cases": len(pruning_all),
                "int8_lower_mse_rate": math.nan,
                "mse_difference_mean": math.nan,
                "mse_difference_ci95_low": math.nan,
                "mse_difference_ci95_high": math.nan,
                "conclusion": "sustentada" if monotonic else "mista",
            }
        )
    return pd.DataFrame(rows)


def _save_speedup(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        _save_empty_plot(path, "Speedup", "Sem candidatos completos")
        return
    fig, axis = plt.subplots(figsize=(10, 5))
    labels = [f"{row.operation}\n{row.scenario}" for row in summary.itertuples()]
    values = summary["speedup_median"].to_numpy()
    ci_low = summary["speedup_ci95_low"].to_numpy()
    ci_high = summary["speedup_ci95_high"].to_numpy()
    positions = np.arange(len(values))
    axis.bar(positions, values, color="#3a6b5c")
    axis.vlines(positions, ci_low, ci_high, color="#202020", linewidth=1)
    axis.hlines(ci_low, positions - 0.08, positions + 0.08, color="#202020", linewidth=1)
    axis.hlines(ci_high, positions - 0.08, positions + 0.08, color="#202020", linewidth=1)
    axis.set_xticks(positions, labels)
    axis.axhline(1.0, color="#9d3d38", linewidth=1)
    axis.axhline(1.05, color="#c58b2a", linewidth=1, linestyle="--")
    axis.set_ylabel("Speedup vs baseline")
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_pruning(comparisons: pd.DataFrame, path: Path) -> None:
    data = comparisons[comparisons["scenario"].str.startswith("pruning_")]
    if data.empty:
        _save_empty_plot(path, "Qualidade por sparsity", "Sem casos completos de pruning")
        return
    fig, axis = plt.subplots(figsize=(8, 5))
    for operation, group in data.groupby("operation"):
        values = group.groupby("pruning_sparsity")["mse"].median()
        axis.plot(values.index, values.values, marker="o", label=operation)
    axis.set_xlabel("Sparsity")
    axis.set_ylabel("MSE mediano")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_seed_boxplot(comparisons: pd.DataFrame, path: Path) -> None:
    if comparisons.empty:
        _save_empty_plot(path, "Dispersao entre seeds", "Sem candidatos completos")
        return
    scenarios = sorted(comparisons["scenario"].unique())
    data = [comparisons.loc[comparisons["scenario"] == scenario, "mse"].to_numpy() for scenario in scenarios]
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.boxplot(data, tick_labels=scenarios, showfliers=False)
    axis.set_ylabel("MSE entre seeds e perfis")
    axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_heatmap(comparisons: pd.DataFrame, path: Path) -> None:
    data = comparisons[comparisons["scenario"] == "int8_fake_per_row"]
    if data.empty:
        _save_empty_plot(path, "Sequencia e dimensao", "Sem casos INT8 fake completos")
        return
    pivot = data.pivot_table(index="seq_len", columns="d_model", values="mse", aggfunc="median")
    fig, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(pivot.columns)), labels=pivot.columns)
    axis.set_yticks(range(len(pivot.index)), labels=pivot.index)
    axis.set_xlabel("d_model")
    axis.set_ylabel("seq_len")
    fig.colorbar(image, ax=axis, label="MSE mediano")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_pareto(comparisons: pd.DataFrame, path: Path) -> None:
    if comparisons.empty:
        _save_empty_plot(path, "Fronteira de Pareto", "Sem candidatos completos")
        return
    grouped = comparisons.groupby(["operation", "scenario"], as_index=False).agg(
        speedup=("speedup_vs_baseline", "median"),
        cosine=("cosine_similarity", "mean"),
    )
    fig, axis = plt.subplots(figsize=(8, 5))
    for row in grouped.itertuples():
        axis.scatter(row.speedup, row.cosine, s=45)
        axis.annotate(f"{row.operation}/{row.scenario}", (row.speedup, row.cosine), fontsize=7)
    axis.axvline(1.0, color="#777777", linewidth=1)
    axis.set_xlabel("Speedup mediano")
    axis.set_ylabel("Similaridade de cosseno media")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_empty_plot(path: Path, title: str, message: str) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _build_report(
    loaded: LoadedExperimentV3,
    summary: pd.DataFrame,
    hypotheses: pd.DataFrame,
) -> str:
    supported = int((summary["performance_conclusion"] == "sustentada").sum()) if not summary.empty else 0
    mixed = int((summary["performance_conclusion"] == "mista").sum()) if not summary.empty else 0
    contradicted = int((summary["performance_conclusion"] == "contradita").sum()) if not summary.empty else 0
    hypothesis_lines = "\n".join(
        rf"\item {row.hypothesis}: {row.comparison} -- \texttt{{{row.conclusion}}}."
        for row in hypotheses.itertuples()
    ) or r"\item Nenhuma hipotese comparativa aplicavel a esta configuracao."
    return "\n".join(
        [
            r"\chapter{Analise dos benchmarks de operacoes v3}",
            r"\section{Contrato experimental}",
            rf"Foram analisados {len(loaded.results)} registros, com {loaded.config.measurement.repetitions} repeticoes e {len(loaded.config.data_seeds)} seeds de dados.",
            r"As comparacoes sao pareadas por entrada, seed, perfil, operacao, repeticao e grupo de baseline.",
            r"\section{Desempenho}",
            rf"Classificacoes sustentadas: {supported}; mistas: {mixed}; contraditas: {contradicted}.",
            r"Uma alegacao de hardware exige speedup mediano de pelo menos 1,05, intervalo de 95\% acima de 1 e kernel confirmado.",
            r"\section{Hipoteses}",
            r"\begin{itemize}",
            hypothesis_lines,
            r"\end{itemize}",
            r"\section{Limites}",
            r"Resultados numericos sem kernel confirmado nao sustentam ganho fisico. Os intervalos bootstrap e todos os casos individuais permanecem nos CSVs.",
        ]
    ) + "\n"


def write_analysis_v3(evidence_dir: str | Path, output_dir: str | Path | None = None) -> AnalysisOutputsV3:
    loaded = load_experiment_v3(evidence_dir)
    output = Path(output_dir) if output_dir is not None else loaded.evidence_dir / "analise_v3"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"diretorio de analise nao esta vazio: {output}")
    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "resultados_completos.csv"
    comparisons_path = output / "comparacoes_pareadas.csv"
    summary_path = output / "resumo_estatistico.csv"
    hypotheses_path = output / "hipoteses.csv"
    unsupported_path = output / "casos_nao_executados.csv"
    timings_path = output / "timings.csv"
    report_path = output / "relatorio_v3.tex"
    speedup_path = output / "speedup_ci95.png"
    pruning_path = output / "qualidade_por_sparsity.png"
    seed_path = output / "dispersao_entre_seeds.png"
    heatmap_path = output / "heatmap_sequencia_dimensao.png"
    pareto_path = output / "pareto_qualidade_desempenho.png"
    manifest_path = output / "analysis_manifest.json"

    summary = summarize_comparisons_v3(
        loaded.comparisons,
        resamples=loaded.config.execution.bootstrap_resamples,
        seed=loaded.config.execution.bootstrap_seed,
    )
    hypotheses = summarize_hypotheses(
        loaded.comparisons,
        resamples=loaded.config.execution.bootstrap_resamples,
        seed=loaded.config.execution.bootstrap_seed,
    )
    loaded.results.to_csv(results_path, index=False, lineterminator="\n")
    loaded.comparisons.to_csv(comparisons_path, index=False, lineterminator="\n")
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    hypotheses.to_csv(hypotheses_path, index=False, lineterminator="\n")
    loaded.results[loaded.results["status"] != "complete"].to_csv(
        unsupported_path, index=False, lineterminator="\n"
    )
    loaded.timings.to_csv(timings_path, index=False, lineterminator="\n")
    report_path.write_text(_build_report(loaded, summary, hypotheses), encoding="utf-8")
    _save_speedup(summary, speedup_path)
    _save_pruning(loaded.comparisons, pruning_path)
    _save_seed_boxplot(loaded.comparisons, seed_path)
    _save_heatmap(loaded.comparisons, heatmap_path)
    _save_pareto(loaded.comparisons, pareto_path)

    artifacts = [
        results_path, comparisons_path, summary_path, hypotheses_path, unsupported_path,
        timings_path, report_path, speedup_path, pruning_path, seed_path, heatmap_path, pareto_path,
    ]
    manifest = {
        "schema_version": 1,
        "source_manifest_sha256": _sha256(loaded.evidence_dir / "manifest.json"),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "artifacts": [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in artifacts
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return AnalysisOutputsV3(
        enriched_results=results_path,
        comparisons=comparisons_path,
        summary=summary_path,
        hypotheses=hypotheses_path,
        unsupported=unsupported_path,
        timings=timings_path,
        report=report_path,
        speedup_plot=speedup_path,
        pruning_plot=pruning_path,
        seed_plot=seed_path,
        heatmap_plot=heatmap_path,
        pareto_plot=pareto_path,
        manifest=manifest_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analisa benchmarks de operacoes v3.")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--output-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        outputs = write_analysis_v3(args.evidence_dir, args.output_dir)
    except (OperationAnalysisErrorV3, FileExistsError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"falha na analise v3: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(outputs.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
