from __future__ import annotations

from dataclasses import dataclass


CSV_COLUMNS = [
    "torch_version",
    "cuda_version",
    "gpu_name",
    "device",
    "model_kind",
    "scenario",
    "batch_size",
    "seq_len",
    "d_model",
    "num_heads",
    "dtype",
    "pruning_sparsity",
    "quantization_method",
    "validity_level",
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

VALIDITY_LEVELS = {
    "conceitual",
    "numerico",
    "algoritmico",
    "hardware",
    "conceitual_algoritmico",
    "numerico_algoritmico",
    "hardware_pendente",
}


@dataclass(frozen=True)
class BenchmarkRecordValidation:
    missing_columns: tuple[str, ...]
    extra_columns: tuple[str, ...]
    invalid_validity_level: str | None

    @property
    def is_valid(self) -> bool:
        return not self.missing_columns and not self.extra_columns and self.invalid_validity_level is None


def empty_benchmark_record(**overrides) -> dict[str, object]:
    record: dict[str, object] = {column: "" for column in CSV_COLUMNS}
    for key, value in overrides.items():
        if key not in CSV_COLUMNS:
            raise KeyError(f"coluna desconhecida: {key}")
        record[key] = value
    return record


def validate_benchmark_record(record: dict[str, object]) -> BenchmarkRecordValidation:
    columns = set(record)
    expected = set(CSV_COLUMNS)
    missing = tuple(column for column in CSV_COLUMNS if column not in columns)
    extra = tuple(sorted(columns - expected))

    validity_level = str(record.get("validity_level", ""))
    invalid_validity_level = None
    if "validity_level" in record and validity_level not in VALIDITY_LEVELS:
        invalid_validity_level = validity_level

    return BenchmarkRecordValidation(
        missing_columns=missing,
        extra_columns=extra,
        invalid_validity_level=invalid_validity_level,
    )
