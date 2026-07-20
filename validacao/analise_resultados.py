from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from validacao.benchmark_controlado import SCENARIO_REGISTRY
from validacao.protocolo import CSV_COLUMNS, validate_benchmark_record


CONFIG_COLUMNS = ["model_kind", "batch_size", "seq_len", "d_model", "num_heads", "dtype"]


@dataclass(frozen=True)
class AnalysisConclusion:
    research_question_id: str
    scenario: str
    conclusion: str
    validity_level: str
    evidence_source: str
    matrix_entry: str


def load_and_validate_results(csv_path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(csv_path)
    if list(dataframe.columns) != CSV_COLUMNS:
        raise ValueError("CSV nao segue o schema obrigatorio")

    for record in dataframe.to_dict(orient="records"):
        validation = validate_benchmark_record(record)
        if not validation.is_valid:
            raise ValueError(f"registro invalido: {validation}")

    return dataframe


def summarize_by_scenario(dataframe: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
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
    grouped = dataframe.groupby(["scenario", "validity_level", "quantization_method"], as_index=False)
    return grouped[numeric_columns].mean()


def compare_against_baseline(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for _, group in dataframe.groupby(CONFIG_COLUMNS, dropna=False):
        baseline = group[group["scenario"] == "baseline"]
        if baseline.empty:
            raise ValueError("cada configuracao precisa ter cenario baseline")
        baseline_row = baseline.iloc[0]

        for _, row in group.iterrows():
            if row["scenario"] == "baseline":
                continue
            rows.append(
                {
                    **{column: row[column] for column in CONFIG_COLUMNS},
                    "scenario": row["scenario"],
                    "validity_level": row["validity_level"],
                    "matrix_entry": SCENARIO_REGISTRY[row["scenario"]].matrix_entry,
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


def build_conclusions(comparisons: pd.DataFrame) -> tuple[AnalysisConclusion, ...]:
    conclusions: list[AnalysisConclusion] = []
    for _, row in comparisons.iterrows():
        scenario = str(row["scenario"])
        latency_ratio = float(row["latency_ratio_vs_baseline"])
        memory_ratio = float(row["memory_ratio_vs_baseline"])
        cosine = float(row["cosine_similarity"])
        validity_level = str(row["validity_level"])

        if latency_ratio < 1.0 and cosine >= 0.99:
            performance_text = "reduziu latencia com alta similaridade de saida"
        elif memory_ratio < 1.0 and cosine >= 0.99:
            performance_text = "reduziu memoria registrada com alta similaridade de saida"
        else:
            performance_text = "nao demonstrou ganho fisico suficiente no protocolo atual"

        conclusions.append(
            AnalysisConclusion(
                research_question_id="RQ3",
                scenario=scenario,
                conclusion=f"{scenario}: {performance_text}; validade observada={validity_level}.",
                validity_level=validity_level,
                evidence_source=f"CSV:{scenario}",
                matrix_entry=str(row["matrix_entry"]),
            )
        )
    return tuple(conclusions)


def write_article_outputs(csv_path: str | Path, output_dir: str | Path) -> tuple[Path, ...]:
    dataframe = load_and_validate_results(csv_path)
    summary = summarize_by_scenario(dataframe)
    comparisons = compare_against_baseline(dataframe)
    conclusions = build_conclusions(comparisons)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    summary_path = output_path / "tabela_resumo_cenarios.csv"
    comparison_path = output_path / "comparacao_baseline.csv"
    conclusions_path = output_path / "conclusoes.md"
    latency_plot_path = output_path / "latencia_por_cenario.png"
    quality_plot_path = output_path / "qualidade_por_cenario.png"

    summary.to_csv(summary_path, index=False)
    comparisons.to_csv(comparison_path, index=False)
    conclusions_path.write_text(
        "\n".join(
            f"- {item.conclusion} Evidencia: {item.evidence_source}; matriz: {item.matrix_entry}."
            for item in conclusions
        ),
        encoding="utf-8",
    )

    plt.figure(figsize=(8, 4))
    sns.barplot(data=dataframe, x="scenario", y="latency_ms_mean")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(latency_plot_path)
    plt.close()

    plt.figure(figsize=(8, 4))
    sns.barplot(data=dataframe, x="scenario", y="cosine_similarity")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(quality_plot_path)
    plt.close()

    return summary_path, comparison_path, conclusions_path, latency_plot_path, quality_plot_path
