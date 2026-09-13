from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from validacao.analise_resultados import CONFIG_COLUMNS, load_and_validate_results
from validacao.benchmark_operacoes import OperationExperimentConfig, config_from_mapping


GROUP_COLUMNS = [*CONFIG_COLUMNS, "scenario", "validity_level", "quantization_method"]
COMPARISON_GROUP_COLUMNS = [*CONFIG_COLUMNS, "scenario", "validity_level"]


class OperationAnalysisError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedExperiment:
    evidence_dir: Path
    config: OperationExperimentConfig
    manifest: dict[str, object]
    environment: dict[str, object]
    results: pd.DataFrame
    comparisons: pd.DataFrame
    reproducibility: dict[str, object]


@dataclass(frozen=True)
class OperationAnalysisOutputs:
    enriched_results: Path
    comparisons: Path
    summary: Path
    comparison_summary: Path
    reproducibility: Path
    report: Path
    latency_plot: Path
    quality_plot: Path

    def paths(self) -> tuple[Path, ...]:
        return (
            self.enriched_results,
            self.comparisons,
            self.summary,
            self.comparison_summary,
            self.reproducibility,
            self.report,
            self.latency_plot,
            self.quality_plot,
        )


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OperationAnalysisError(f"JSON invalido ou ausente: {path}") from exc
    if not isinstance(value, dict):
        raise OperationAnalysisError(f"JSON deve conter objeto: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise OperationAnalysisError(f"artefato ausente: {path}") from exc
    return digest.hexdigest()


def _validate_numeric_results(dataframe: pd.DataFrame) -> None:
    numeric_columns = [
        "batch_size",
        "seq_len",
        "d_model",
        "num_heads",
        "latency_ms_mean",
        "latency_ms_p50",
        "latency_ms_p95",
        "throughput_tokens_s",
        "max_memory_bytes",
        "theoretical_flops",
        "mse",
        "mae",
        "r2",
        "cosine_similarity",
    ]
    for column in numeric_columns:
        values = pd.to_numeric(dataframe[column], errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise OperationAnalysisError(f"campo numerico invalido: {column}")
    if (dataframe[["latency_ms_mean", "throughput_tokens_s", "max_memory_bytes", "theoretical_flops"]] <= 0).any().any():
        raise OperationAnalysisError("custos medidos devem ser positivos")


def _fingerprint_key(item: dict[str, object]) -> tuple[object, ...]:
    return (
        item["operation"],
        item["batch_size"],
        item["seq_len"],
        item["dimension"],
        item["scenario"],
    )


def _verify_fingerprints(metadata_items: list[dict[str, object]]) -> dict[str, object]:
    if not metadata_items:
        raise OperationAnalysisError("nenhum metadado de execucao")

    input_maps: list[dict[tuple[object, ...], str]] = []
    output_maps: list[dict[tuple[object, ...], str]] = []
    for metadata in metadata_items:
        fingerprints = metadata.get("fingerprints")
        if not isinstance(fingerprints, list) or not fingerprints:
            raise OperationAnalysisError("metadado sem fingerprints")
        try:
            input_maps.append(
                {_fingerprint_key(item): str(item["input_sha256"]) for item in fingerprints}
            )
            output_maps.append(
                {_fingerprint_key(item): str(item["output_sha256"]) for item in fingerprints}
            )
        except (KeyError, TypeError) as exc:
            raise OperationAnalysisError("fingerprint incompleto") from exc

    inputs_match = all(item == input_maps[0] for item in input_maps[1:])
    outputs_match = all(item == output_maps[0] for item in output_maps[1:])
    result = {
        "independent_runs": len(metadata_items),
        "fingerprinted_cases_per_run": len(input_maps[0]),
        "input_hashes_match_across_runs": inputs_match,
        "output_hashes_match_across_runs": outputs_match,
        "timings_expected_to_differ": True,
        "passed": inputs_match and outputs_match,
    }
    if not result["passed"]:
        raise OperationAnalysisError("entradas ou saidas deterministicas divergiram entre execucoes")
    return result


def compare_each_run_against_baseline(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = ["independent_run", *CONFIG_COLUMNS]
    for keys, group in dataframe.groupby(group_columns, dropna=False, sort=True):
        baseline = group[group["scenario"] == "baseline"]
        if len(baseline) != 1:
            raise OperationAnalysisError("cada configuracao/execucao deve ter exatamente um baseline")
        baseline_row = baseline.iloc[0]
        key_values = dict(zip(group_columns, keys, strict=True))
        for _, row in group.iterrows():
            if row["scenario"] == "baseline":
                continue
            rows.append(
                {
                    **key_values,
                    "scenario": row["scenario"],
                    "validity_level": row["validity_level"],
                    "latency_ratio_vs_baseline": row["latency_ms_mean"] / baseline_row["latency_ms_mean"],
                    "throughput_ratio_vs_baseline": row["throughput_tokens_s"] / baseline_row["throughput_tokens_s"],
                    "memory_ratio_vs_baseline": row["max_memory_bytes"] / baseline_row["max_memory_bytes"],
                    "mse": row["mse"],
                    "mae": row["mae"],
                    "r2": row["r2"],
                    "cosine_similarity": row["cosine_similarity"],
                }
            )
    return pd.DataFrame(rows)


def load_experiment(evidence_dir: str | Path) -> LoadedExperiment:
    root = Path(evidence_dir)
    manifest = _read_json(root / "manifest.json")
    if manifest.get("status") != "complete":
        raise OperationAnalysisError("experimento nao esta completo")

    config_data = _read_json(root / str(manifest.get("config", "config.snapshot.json")))
    config = config_from_mapping(config_data)
    environment = _read_json(root / str(manifest.get("environment", "environment.json")))
    run_entries = manifest.get("runs")
    if not isinstance(run_entries, list) or len(run_entries) != config.independent_runs:
        raise OperationAnalysisError("manifesto nao contem todas as execucoes independentes")

    frames: list[pd.DataFrame] = []
    metadata_items: list[dict[str, object]] = []
    for entry in run_entries:
        if not isinstance(entry, dict):
            raise OperationAnalysisError("entrada de execucao invalida no manifesto")
        run_index = int(entry["independent_run"])
        csv_path = root / str(entry["csv"])
        metadata_path = root / str(entry["metadata"])
        if _sha256(csv_path) != entry["csv_sha256"]:
            raise OperationAnalysisError(f"checksum do CSV divergiu: {csv_path.name}")
        if _sha256(metadata_path) != entry["metadata_sha256"]:
            raise OperationAnalysisError(f"checksum dos metadados divergiu: {metadata_path.name}")

        frame = load_and_validate_results(csv_path)
        _validate_numeric_results(frame)
        frame.insert(0, "independent_run", run_index)
        frames.append(frame)
        metadata_items.append(_read_json(metadata_path))

    results = pd.concat(frames, ignore_index=True)
    duplicate_columns = ["independent_run", *CONFIG_COLUMNS, "scenario"]
    if results.duplicated(duplicate_columns).any():
        raise OperationAnalysisError("ha registro duplicado para configuracao/cenario/execucao")

    expected_per_run = (
        len(config.operations)
        * len(config.batch_sizes)
        * len(config.sequence_lengths)
        * len(config.dimensions)
        * len(config.scenarios)
    )
    observed = results.groupby("independent_run").size().to_dict()
    if any(int(count) != expected_per_run for count in observed.values()):
        raise OperationAnalysisError("quantidade de registros difere da grade configurada")

    reproducibility = _verify_fingerprints(metadata_items)
    comparisons = compare_each_run_against_baseline(results)
    return LoadedExperiment(
        evidence_dir=root,
        config=config,
        manifest=manifest,
        environment=environment,
        results=results,
        comparisons=comparisons,
        reproducibility=reproducibility,
    )


def summarize_runs(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby(GROUP_COLUMNS, as_index=False, dropna=False).agg(
        independent_runs=("independent_run", "nunique"),
        latency_ms_mean=("latency_ms_mean", "mean"),
        latency_ms_std=("latency_ms_mean", "std"),
        latency_ms_p50_mean=("latency_ms_p50", "mean"),
        latency_ms_p95_mean=("latency_ms_p95", "mean"),
        throughput_tokens_s_mean=("throughput_tokens_s", "mean"),
        throughput_tokens_s_std=("throughput_tokens_s", "std"),
        max_memory_bytes_mean=("max_memory_bytes", "mean"),
        max_memory_bytes_std=("max_memory_bytes", "std"),
        theoretical_flops=("theoretical_flops", "first"),
        mse_mean=("mse", "mean"),
        mse_std=("mse", "std"),
        mae_mean=("mae", "mean"),
        r2_mean=("r2", "mean"),
        cosine_similarity_mean=("cosine_similarity", "mean"),
    )
    return summary.fillna(0.0)


def summarize_comparisons(comparisons: pd.DataFrame) -> pd.DataFrame:
    summary = comparisons.groupby(COMPARISON_GROUP_COLUMNS, as_index=False, dropna=False).agg(
        independent_runs=("independent_run", "nunique"),
        latency_ratio_mean=("latency_ratio_vs_baseline", "mean"),
        latency_ratio_std=("latency_ratio_vs_baseline", "std"),
        throughput_ratio_mean=("throughput_ratio_vs_baseline", "mean"),
        throughput_ratio_std=("throughput_ratio_vs_baseline", "std"),
        memory_ratio_mean=("memory_ratio_vs_baseline", "mean"),
        memory_ratio_std=("memory_ratio_vs_baseline", "std"),
        mse_mean=("mse", "mean"),
        mae_mean=("mae", "mean"),
        r2_mean=("r2", "mean"),
        cosine_similarity_mean=("cosine_similarity", "mean"),
    )
    return summary.fillna(0.0)


def _format(value: object) -> str:
    number = float(value)
    return f"{number:.6g}" if math.isfinite(number) else str(number)


def _latex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "_": r"\_\allowbreak{}",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _build_report(
    loaded: LoadedExperiment,
    summary: pd.DataFrame,
    comparison_summary: pd.DataFrame,
) -> str:
    config = loaded.config
    gpu = loaded.environment.get("gpu", {})
    device_name = gpu.get("name", "") if isinstance(gpu, dict) else ""
    if config.purpose == "primary_benchmark":
        hardware_warning = (
            "Esta coleta foi configurada como benchmark principal em CUDA; as conclusões ainda devem respeitar o nível de validade de cada cenário."
        )
    elif config.purpose == "cpu_diagnostic":
        hardware_warning = (
            "Esta coleta é um diagnóstico em CPU de um ponto da grade formal. Os tempos não substituem a coleta na GPU-alvo e não sustentam conclusão de hardware."
        )
    else:
        hardware_warning = (
            "Esta coleta e um smoke test em CPU. Os tempos servem apenas para validar a pipeline e não sustentam conclusão de hardware."
        )
    lines = [
        rf"\chapter{{Relatório preliminar - {_latex_escape(config.experiment_id)}}}",
        "",
        r"\begin{icquote}",
        hardware_warning,
        r"\end{icquote}",
        "",
        r"\section{Pergunta, hipótese e desenho}",
        "",
        r"\begin{itemize}",
        r"\item Pergunta principal: quantização INT8 simulada preserva melhor a saída que pruning por magnitude de 50\% nas mesmas operações?",
        r"\item Hipótese: a quantização terá MSE/MAE menores e $R^2$/cosseno maiores que o pruning.",
        r"\item Amostra: tensores sintéticos de distribuição normal, gerados deterministicamente; não há dataset ou pré-processamento externo.",
        rf"\item Seed: \texttt{{{config.seed}}}; lotes: \texttt{{{_latex_escape(list(config.batch_sizes))}}}; sequências: \texttt{{{_latex_escape(list(config.sequence_lengths))}}}; dimensões: \texttt{{{_latex_escape(list(config.dimensions))}}}.",
        rf"\item Dados/pesos: entrada \texttt{{{_latex_escape(config.input_distribution)}}}, pesos \texttt{{{_latex_escape(config.weight_initialization)}}} e bias \texttt{{{_latex_escape(config.bias_initialization)}}}.",
        rf"\item Medição: \texttt{{{config.warmup_iterations}}} warm-ups, \texttt{{{config.measure_iterations}}} medições e \texttt{{{config.independent_runs}}} execuções independentes.",
        rf"\item Dispositivo observado: \texttt{{{_latex_escape(loaded.environment.get('device', ''))}}} {_latex_escape(device_name)}.",
        r"\item Inclusão: todos os casos válidos da grade e os três cenários registrados; exclusão: qualquer artefato com checksum, shape, dtype ou valor numérico inválido.",
        r"\end{itemize}",
        "",
        r"\section{Reprodutibilidade}",
        "",
        r"\begin{itemize}",
        rf"\item Hashes de entrada iguais entre execuções: \texttt{{{_latex_escape(loaded.reproducibility['input_hashes_match_across_runs'])}}}.",
        rf"\item Hashes de saída iguais entre execuções: \texttt{{{_latex_escape(loaded.reproducibility['output_hashes_match_across_runs'])}}}.",
        r"\item As latências não precisam ser idênticas: elas medem ruído e estado do sistema; entradas e saídas determinísticas precisam coincidir.",
        r"\end{itemize}",
        "",
        r"\section{Resultados por operação e cenário}",
        "",
        r"\begin{landscape}",
        r"\small",
        r"\begin{longtable}{@{}p{0.20\linewidth}p{0.20\linewidth}rrrrr@{}}",
        r"\toprule",
        r"\textbf{Operação} & \textbf{Cenário} & \textbf{Latência (ms)} & \textbf{DP} & \textbf{MSE} & \textbf{$R^2$} & \textbf{Cosseno} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Operação} & \textbf{Cenário} & \textbf{Latência (ms)} & \textbf{DP} & \textbf{MSE} & \textbf{$R^2$} & \textbf{Cosseno} \\",
        r"\midrule",
        r"\endhead",
    ]
    for _, row in summary.sort_values(["model_kind", "scenario"]).iterrows():
        lines.append(
            "{operation} & {scenario} & {latency} & {std} & {mse} & {r2} & {cosine} \\\\".format(
                operation=_latex_escape(row["model_kind"]),
                scenario=_latex_escape(row["scenario"]),
                latency=_format(row["latency_ms_mean"]),
                std=_format(row["latency_ms_std"]),
                mse=_format(row["mse_mean"]),
                r2=_format(row["r2_mean"]),
                cosine=_format(row["cosine_similarity_mean"]),
            )
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            r"\end{landscape}",
            "",
            r"\section{Interpretação preliminar}",
            "",
            r"\begin{itemize}",
        ]
    )
    for operation in config.operations:
        operation_rows = comparison_summary[comparison_summary["model_kind"] == operation]
        pruning = operation_rows[operation_rows["scenario"] == "pruning_magnitude"]
        quantized = operation_rows[operation_rows["scenario"] == "quantization_int8"]
        if not pruning.empty and not quantized.empty:
            pruning_mse = float(pruning["mse_mean"].mean())
            quantized_mse = float(quantized["mse_mean"].mean())
            relation = "compatível com" if quantized_mse < pruning_mse else "contrária a"
            lines.append(
                rf"\item \texttt{{{_latex_escape(operation)}}}: MSE da quantização \texttt{{{_format(quantized_mse)}}} e do pruning \texttt{{{_format(pruning_mse)}}}; evidência {relation} H1 nesta amostra."
            )
    lines.extend(
        [
            r"\item Pruning usa matriz densa e não usa kernel esparso; quantização é dequantizada para \texttt{float32} e não usa kernel INT8.",
            r"\item Razões de latência e memória estão nos CSVs de análise, mas só podem sustentar alegação de hardware se o experimento for principal e o caminho executado tiver validade \texttt{hardware}.",
            r"\item Resultados de smoke ou diagnóstico CPU não são estimativas finais de desempenho no hardware-alvo.",
            r"\end{itemize}",
            "",
            r"\section{Cadeia de evidência}",
            "",
            r"Pergunta $\rightarrow$ H1 $\rightarrow$ configuração versionada $\rightarrow$ CSVs por execução $\rightarrow$ checksums/hashes $\rightarrow$ resumo com média e DP $\rightarrow$ interpretação limitada.",
            "",
            r"\section{Reprodução}",
            "",
            r"\begin{Verbatim}[fontsize=\small]",
            "$env:PYTHONPATH = \"fases\\01_validacao_conceitual\"",
            ".\\.venv\\Scripts\\python.exe -m validacao.benchmark_operacoes --config <config.json> --output-dir <diretorio-novo>",
            ".\\.venv\\Scripts\\python.exe -m validacao.analise_benchmark_operacoes --evidence-dir <diretorio-novo>",
            r"\end{Verbatim}",
            "",
        ]
    )
    return "\n".join(lines)


def _plot_latency(comparison_summary: pd.DataFrame, path: Path) -> None:
    plot = comparison_summary.groupby(["model_kind", "scenario"], as_index=False).agg(
        ratio=("latency_ratio_mean", "mean"),
        std=("latency_ratio_std", "mean"),
    )
    labels = [f"{row.model_kind}\n{row.scenario}" for row in plot.itertuples()]
    colors = ["#d97706" if "pruning" in label else "#2563eb" for label in labels]
    plt.figure(figsize=(9, 4.8))
    plt.bar(labels, plot["ratio"], yerr=plot["std"], capsize=4, color=colors)
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--", label="baseline")
    plt.ylabel("Razão de latência vs. baseline")
    plt.title("Latência relativa por operação e cenário")
    plt.xticks(rotation=15, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_quality(comparison_summary: pd.DataFrame, path: Path) -> None:
    plot = comparison_summary.groupby(["model_kind", "scenario"], as_index=False).agg(
        cosine=("cosine_similarity_mean", "mean"),
    )
    labels = [f"{row.model_kind}\n{row.scenario}" for row in plot.itertuples()]
    colors = ["#d97706" if "pruning" in label else "#2563eb" for label in labels]
    lower = max(-1.0, float(plot["cosine"].min()) - 0.05)
    plt.figure(figsize=(9, 4.8))
    plt.bar(labels, plot["cosine"], color=colors)
    plt.ylim(lower, 1.01)
    plt.ylabel("Similaridade de cosseno")
    plt.title("Fidelidade da saída em relação ao baseline")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_operation_analysis(
    evidence_dir: str | Path,
    output_dir: str | Path | None = None,
) -> OperationAnalysisOutputs:
    loaded = load_experiment(evidence_dir)
    output_path = Path(output_dir) if output_dir is not None else loaded.evidence_dir / "analise"
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"diretorio de analise nao esta vazio: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    summary = summarize_runs(loaded.results)
    comparison_summary = summarize_comparisons(loaded.comparisons)
    enriched_path = output_path / "resultados_com_execucao.csv"
    comparisons_path = output_path / "comparacao_baseline_por_execucao.csv"
    summary_path = output_path / "resumo_execucoes.csv"
    comparison_summary_path = output_path / "resumo_comparacao_baseline.csv"
    reproducibility_path = output_path / "reprodutibilidade.json"
    report_path = output_path / "relatorio_preliminar.tex"
    latency_plot_path = output_path / "latencia_relativa.png"
    quality_plot_path = output_path / "qualidade_saida.png"

    loaded.results.to_csv(enriched_path, index=False, lineterminator="\n")
    loaded.comparisons.to_csv(comparisons_path, index=False, lineterminator="\n")
    summary.to_csv(summary_path, index=False, lineterminator="\n")
    comparison_summary.to_csv(comparison_summary_path, index=False, lineterminator="\n")
    reproducibility_path.write_text(
        json.dumps(loaded.reproducibility, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        _build_report(loaded, summary, comparison_summary),
        encoding="utf-8",
    )
    _plot_latency(comparison_summary, latency_plot_path)
    _plot_quality(comparison_summary, quality_plot_path)
    return OperationAnalysisOutputs(
        enriched_results=enriched_path,
        comparisons=comparisons_path,
        summary=summary_path,
        comparison_summary=comparison_summary_path,
        reproducibility=reproducibility_path,
        report=report_path,
        latency_plot=latency_plot_path,
        quality_plot=quality_plot_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Valida e analisa evidencias das operacoes isoladas.")
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--output-dir")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest_path = Path(args.evidence_dir) / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest, dict) and manifest.get("schema_version") == 3:
            from validacao.analise_benchmark_operacoes_v3 import write_analysis_v3

            outputs_v3 = write_analysis_v3(args.evidence_dir, args.output_dir)
            for path in outputs_v3.paths():
                print(path)
            return 0
        outputs = write_operation_analysis(args.evidence_dir, args.output_dir)
    except (OperationAnalysisError, FileExistsError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2
    for path in outputs.paths():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
