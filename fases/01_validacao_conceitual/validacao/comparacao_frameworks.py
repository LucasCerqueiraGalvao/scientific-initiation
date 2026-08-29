from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Sequence

import numpy as np
import torch

from validacao.attention import (
    scaled_dot_product_attention_numpy,
    scaled_dot_product_attention_torch,
)


COMPARISON_COLUMNS = [
    "case_id",
    "reference",
    "candidate",
    "dtype",
    "batch_size",
    "num_heads",
    "seq_len",
    "head_dim",
    "mask_kind",
    "max_abs_error",
    "mean_abs_error",
    "mse",
    "cosine_similarity",
    "atol",
    "rtol",
    "passed",
]

REFERENCE_NAME = "numpy_manual"
CANDIDATE_NAMES = (
    "pytorch_manual",
    "pytorch_sdpa",
    "keras_tensorflow_sdpa",
)


class ComparisonDependencyError(RuntimeError):
    """Dependencia opcional ausente ou backend Keras incorreto."""


@dataclass(frozen=True)
class AttentionComparisonCase:
    case_id: str
    query: np.ndarray
    key: np.ndarray
    value: np.ndarray
    additive_mask: np.ndarray | None
    mask_kind: str


@dataclass(frozen=True)
class FrameworkComparisonConfig:
    seed: int = 2026
    atol: float = 1e-6
    rtol: float = 1e-5


@dataclass(frozen=True)
class FrameworkComparisonRun:
    records: tuple[dict[str, object], ...]
    metadata: dict[str, object]

    @property
    def passed(self) -> bool:
        return bool(self.records) and all(bool(record["passed"]) for record in self.records)


def pytorch_to_keras_layout(tensor: np.ndarray) -> np.ndarray:
    """Converte `[B, H, T, D]` para `[B, T, H, D]`."""
    array = np.asarray(tensor)
    if array.ndim != 4:
        raise ValueError("tensor deve ter quatro dimensoes [B, H, T, D]")
    return np.ascontiguousarray(np.transpose(array, (0, 2, 1, 3)))


def keras_to_pytorch_layout(tensor: np.ndarray) -> np.ndarray:
    """Converte `[B, T, H, D]` para `[B, H, T, D]`."""
    array = np.asarray(tensor)
    if array.ndim != 4:
        raise ValueError("tensor deve ter quatro dimensoes [B, T, H, D]")
    return np.ascontiguousarray(np.transpose(array, (0, 2, 1, 3)))


def _load_keras_tensorflow() -> tuple[ModuleType, ModuleType]:
    try:
        tensorflow = importlib.import_module("tensorflow")
        keras = importlib.import_module("keras")
    except ModuleNotFoundError as exc:
        raise ComparisonDependencyError(
            "TensorFlow/Keras nao esta instalado. Instale "
            "fases/01_validacao_conceitual/requirements-comparacao.txt."
        ) from exc

    backend = str(keras.backend.backend())
    if backend != "tensorflow":
        raise ComparisonDependencyError(
            f"backend Keras deve ser tensorflow, mas o backend ativo e {backend!r}"
        )
    return keras, tensorflow


def _validate_attention_inputs(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    additive_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    arrays = tuple(np.asarray(item) for item in (query, key, value))
    if any(item.ndim != 4 for item in arrays):
        raise ValueError("Q, K e V devem ter shape [B, H, T, D]")
    if any(item.dtype != np.float32 for item in arrays):
        raise ValueError("a comparacao cruzada aceita apenas float32")

    query_arr, key_arr, value_arr = arrays
    if query_arr.shape[0:2] != key_arr.shape[0:2] or key_arr.shape[0:2] != value_arr.shape[0:2]:
        raise ValueError("Q, K e V devem ter o mesmo batch e numero de heads")
    if query_arr.shape[-1] != key_arr.shape[-1]:
        raise ValueError("Q e K devem ter a mesma dimensao por head")
    if key_arr.shape[-2] != value_arr.shape[-2]:
        raise ValueError("K e V devem ter o mesmo comprimento de sequencia")

    mask_arr = None if additive_mask is None else np.asarray(additive_mask)
    if mask_arr is not None and mask_arr.dtype != np.float32:
        raise ValueError("a mascara aditiva deve usar float32")

    return (
        np.ascontiguousarray(query_arr),
        np.ascontiguousarray(key_arr),
        np.ascontiguousarray(value_arr),
        None if mask_arr is None else np.ascontiguousarray(mask_arr),
    )


def scaled_dot_product_attention_keras_tensorflow(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    additive_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Executa a SDPA publica do Keras com backend TensorFlow em CPU."""
    query_arr, key_arr, value_arr, mask_arr = _validate_attention_inputs(
        query,
        key,
        value,
        additive_mask,
    )
    keras, tensorflow = _load_keras_tensorflow()

    query_keras = pytorch_to_keras_layout(query_arr)
    key_keras = pytorch_to_keras_layout(key_arr)
    value_keras = pytorch_to_keras_layout(value_arr)

    with tensorflow.device("/CPU:0"):
        output = keras.ops.nn.dot_product_attention(
            keras.ops.convert_to_tensor(query_keras, dtype="float32"),
            keras.ops.convert_to_tensor(key_keras, dtype="float32"),
            keras.ops.convert_to_tensor(value_keras, dtype="float32"),
            bias=None if mask_arr is None else keras.ops.convert_to_tensor(mask_arr, dtype="float32"),
            scale=None,
            is_causal=False,
            flash_attention=False,
        )
    output_numpy = np.asarray(keras.ops.convert_to_numpy(output), dtype=np.float32)
    return keras_to_pytorch_layout(output_numpy)


def build_attention_comparison_cases(seed: int = 2026) -> tuple[AttentionComparisonCase, ...]:
    tiny = AttentionComparisonCase(
        case_id="tiny_manual_unmasked",
        query=np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32),
        key=np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32),
        value=np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32),
        additive_mask=None,
        mask_kind="none",
    )

    random_generator = np.random.default_rng(seed)
    random_multihead = AttentionComparisonCase(
        case_id="seeded_multihead_unmasked",
        query=random_generator.standard_normal((2, 2, 3, 4), dtype=np.float32),
        key=random_generator.standard_normal((2, 2, 3, 4), dtype=np.float32),
        value=random_generator.standard_normal((2, 2, 3, 4), dtype=np.float32),
        additive_mask=None,
        mask_kind="none",
    )

    masked_generator = np.random.default_rng(seed + 1)
    additive_mask = np.array(
        [
            [
                [
                    [0.0, 0.0, -1.0e9],
                    [0.0, 0.0, 0.0],
                    [-1.0e9, 0.0, 0.0],
                ]
            ]
        ],
        dtype=np.float32,
    )
    masked_multihead = AttentionComparisonCase(
        case_id="seeded_multihead_additive_mask",
        query=masked_generator.standard_normal((1, 2, 3, 4), dtype=np.float32),
        key=masked_generator.standard_normal((1, 2, 3, 4), dtype=np.float32),
        value=masked_generator.standard_normal((1, 2, 3, 4), dtype=np.float32),
        additive_mask=additive_mask,
        mask_kind="additive_bias",
    )
    return tiny, random_multihead, masked_multihead


def _torch_manual_output(case: AttentionComparisonCase) -> np.ndarray:
    with torch.inference_mode():
        output = scaled_dot_product_attention_torch(
            torch.from_numpy(case.query),
            torch.from_numpy(case.key),
            torch.from_numpy(case.value),
            None if case.additive_mask is None else torch.from_numpy(case.additive_mask),
        )
    return output.detach().cpu().numpy()


def _torch_sdpa_output(case: AttentionComparisonCase) -> np.ndarray:
    with torch.inference_mode():
        output = torch.nn.functional.scaled_dot_product_attention(
            torch.from_numpy(case.query),
            torch.from_numpy(case.key),
            torch.from_numpy(case.value),
            attn_mask=None if case.additive_mask is None else torch.from_numpy(case.additive_mask),
            dropout_p=0.0,
            is_causal=False,
        )
    return output.detach().cpu().numpy()


def _cosine_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference_flat = reference.astype(np.float64, copy=False).reshape(-1)
    candidate_flat = candidate.astype(np.float64, copy=False).reshape(-1)
    denominator = float(np.linalg.norm(reference_flat) * np.linalg.norm(candidate_flat))
    if denominator == 0.0:
        return 1.0 if np.array_equal(reference_flat, candidate_flat) else 0.0
    return float(np.dot(reference_flat, candidate_flat) / denominator)


def _comparison_record(
    case: AttentionComparisonCase,
    candidate_name: str,
    reference: np.ndarray,
    candidate: np.ndarray,
    config: FrameworkComparisonConfig,
) -> dict[str, object]:
    same_shape = candidate.shape == reference.shape
    same_dtype = candidate.dtype == reference.dtype == np.float32
    finite = bool(np.isfinite(reference).all() and np.isfinite(candidate).all())

    if same_shape:
        difference = candidate.astype(np.float64) - reference.astype(np.float64)
        absolute_difference = np.abs(difference)
        max_abs_error = float(np.max(absolute_difference))
        mean_abs_error = float(np.mean(absolute_difference))
        mse = float(np.mean(np.square(difference)))
        cosine = _cosine_similarity(reference, candidate)
        close = bool(np.allclose(candidate, reference, atol=config.atol, rtol=config.rtol))
    else:
        max_abs_error = math.inf
        mean_abs_error = math.inf
        mse = math.inf
        cosine = math.nan
        close = False

    batch_size, num_heads, seq_len, head_dim = case.query.shape
    return {
        "case_id": case.case_id,
        "reference": REFERENCE_NAME,
        "candidate": candidate_name,
        "dtype": "float32",
        "batch_size": batch_size,
        "num_heads": num_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "mask_kind": case.mask_kind,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "mse": mse,
        "cosine_similarity": cosine,
        "atol": config.atol,
        "rtol": config.rtol,
        "passed": bool(same_shape and same_dtype and finite and close),
    }


def validate_comparison_record(record: dict[str, object]) -> None:
    if list(record) != COMPARISON_COLUMNS:
        raise ValueError("registro nao segue o schema da comparacao entre frameworks")
    if record["reference"] != REFERENCE_NAME:
        raise ValueError("referencia da comparacao deve ser numpy_manual")
    if record["candidate"] not in CANDIDATE_NAMES:
        raise ValueError(f"candidato desconhecido: {record['candidate']}")


def run_framework_comparison(
    config: FrameworkComparisonConfig = FrameworkComparisonConfig(),
) -> FrameworkComparisonRun:
    keras, tensorflow = _load_keras_tensorflow()
    records: list[dict[str, object]] = []
    cases = build_attention_comparison_cases(config.seed)

    for case in cases:
        reference = scaled_dot_product_attention_numpy(
            case.query,
            case.key,
            case.value,
            case.additive_mask,
        ).astype(np.float32, copy=False)
        candidates = {
            "pytorch_manual": _torch_manual_output(case),
            "pytorch_sdpa": _torch_sdpa_output(case),
            "keras_tensorflow_sdpa": scaled_dot_product_attention_keras_tensorflow(
                case.query,
                case.key,
                case.value,
                case.additive_mask,
            ),
        }
        for candidate_name, candidate in candidates.items():
            record = _comparison_record(case, candidate_name, reference, candidate, config)
            validate_comparison_record(record)
            records.append(record)

    metadata: dict[str, object] = {
        "schema_version": 1,
        "comparison_kind": "numerical_correctness",
        "performance_claim": False,
        "reference": REFERENCE_NAME,
        "candidates": list(CANDIDATE_NAMES),
        "seed": config.seed,
        "atol": config.atol,
        "rtol": config.rtol,
        "dtype": "float32",
        "device": "cpu",
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "tensorflow_version": tensorflow.__version__,
        "keras_version": keras.__version__,
        "keras_backend": str(keras.backend.backend()),
        "cases": [case.case_id for case in cases],
    }
    return FrameworkComparisonRun(records=tuple(records), metadata=metadata)


def _format_metric(value: object) -> str:
    number = float(value)
    if not math.isfinite(number):
        return str(number)
    return f"{number:.8e}"


def _latex_escape(value: object) -> str:
    text = str(value)
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
    return "".join(replacements.get(char, char) for char in text)


def _build_latex(run: FrameworkComparisonRun) -> str:
    lines = [
        r"\chapter{Comparação numérica da scaled dot-product attention}",
        "",
        r"\begin{icquote}",
        "Esta evidência valida correção numérica. Ela não compara desempenho e não sustenta alegações de hardware.",
        r"\end{icquote}",
        "",
        r"\begin{itemize}",
        rf"\item Referência: \texttt{{{_latex_escape(run.metadata['reference'])}}}.",
        rf"\item Seed: \texttt{{{_latex_escape(run.metadata['seed'])}}}.",
        rf"\item Dtype e dispositivo: \texttt{{{_latex_escape(run.metadata['dtype'])}}} em \texttt{{{_latex_escape(run.metadata['device'])}}}.",
        rf"\item Tolerâncias: \texttt{{atol={_latex_escape(run.metadata['atol'])}}} e \texttt{{rtol={_latex_escape(run.metadata['rtol'])}}}.",
        rf"\item Backend Keras: \texttt{{{_latex_escape(run.metadata['keras_backend'])}}}.",
        r"\end{itemize}",
        "",
        r"\begin{landscape}",
        r"\scriptsize",
        r"\begin{longtable}{@{}p{0.18\linewidth}p{0.16\linewidth}p{0.12\linewidth}rrrrc@{}}",
        r"\toprule",
        r"\textbf{Caso} & \textbf{Candidato} & \textbf{Máscara} & \textbf{Erro máximo} & \textbf{Erro médio} & \textbf{MSE} & \textbf{Cosseno} & \textbf{Passou} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"\textbf{Caso} & \textbf{Candidato} & \textbf{Máscara} & \textbf{Erro máximo} & \textbf{Erro médio} & \textbf{MSE} & \textbf{Cosseno} & \textbf{Passou} \\",
        r"\midrule",
        r"\endhead",
    ]
    for record in run.records:
        lines.append(
            "{case_id} & {candidate} & {mask_kind} & {max_error} & {mean_error} & {mse} & {cosine} & {passed} \\\\".format(
                case_id=_latex_escape(record["case_id"]),
                candidate=_latex_escape(record["candidate"]),
                mask_kind=_latex_escape(record["mask_kind"]),
                max_error=_format_metric(record["max_abs_error"]),
                mean_error=_format_metric(record["mean_abs_error"]),
                mse=_format_metric(record["mse"]),
                cosine=_format_metric(record["cosine_similarity"]),
                passed="sim" if record["passed"] else "não",
            )
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{longtable}",
            r"\end{landscape}",
            "",
            r"\section{Conclusão}",
            "",
            (
                "Todos os casos ficaram dentro das tolerâncias definidas."
                if run.passed
                else "Ao menos um caso ficou fora das tolerâncias definidas; a validação não foi aprovada."
            ),
            "",
            r"As versões completas do ambiente estão em \texttt{comparacao\_atencao.metadata.json} e os valores auditáveis em \texttt{comparacao\_atencao.csv}.",
            "",
        ]
    )
    return "\n".join(lines)


def write_comparison_outputs(
    run: FrameworkComparisonRun,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "comparacao_atencao.csv"
    latex_path = output_path / "comparacao_atencao.tex"
    metadata_path = output_path / "comparacao_atencao.metadata.json"

    for record in run.records:
        validate_comparison_record(record)

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=COMPARISON_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(run.records)

    latex_path.write_text(_build_latex(run), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(run.metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return csv_path, latex_path, metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compara a SDPA local com NumPy, PyTorch e Keras/TensorFlow.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = FrameworkComparisonConfig(seed=args.seed, atol=args.atol, rtol=args.rtol)
    try:
        run = run_framework_comparison(config)
    except ComparisonDependencyError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2

    generated = write_comparison_outputs(run, args.output_dir)
    for path in generated:
        print(path)
    print(f"comparacoes aprovadas: {sum(bool(item['passed']) for item in run.records)}/{len(run.records)}")
    return 0 if run.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
