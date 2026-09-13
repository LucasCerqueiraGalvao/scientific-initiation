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


CSV_COLUMNS_V3 = [
    "experiment_id",
    "stage",
    "case_id",
    "baseline_case_id",
    "data_seed",
    "repeat_index",
    "profile_id",
    "model_id",
    "model_revision",
    "operation",
    "optimization_scope",
    "scenario",
    "comparison_group",
    "status",
    "skip_reason",
    "batch_size",
    "seq_len",
    "d_model",
    "num_heads",
    "dtype",
    "backend",
    "compile_mode",
    "input_distribution",
    "pruning_method",
    "pruning_sparsity",
    "quantization_method",
    "weight_storage_dtype",
    "activation_dtype",
    "observed_sparsity",
    "uses_sparse_kernel",
    "uses_int8_kernel",
    "latency_ms_mean",
    "latency_ms_p50",
    "latency_ms_p95",
    "throughput_tokens_s",
    "peak_memory_allocated_bytes",
    "peak_memory_reserved_bytes",
    "parameter_storage_bytes",
    "theoretical_flops",
    "mse",
    "mae",
    "r2",
    "cosine_similarity",
    "kl_divergence",
    "top1_agreement",
    "loss",
    "perplexity",
    "input_sha256",
    "weight_sha256",
    "output_sha256",
    "profiler_trace",
]

V3_STATUSES = {"complete", "unsupported", "failed"}

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


@dataclass(frozen=True)
class BenchmarkRecordValidationV3:
    missing_columns: tuple[str, ...]
    extra_columns: tuple[str, ...]
    invalid_status: str | None
    invalid_complete_fields: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return not (
            self.missing_columns
            or self.extra_columns
            or self.invalid_status
            or self.invalid_complete_fields
        )


def empty_benchmark_record_v3(**overrides) -> dict[str, object]:
    record: dict[str, object] = {column: "" for column in CSV_COLUMNS_V3}
    for key, value in overrides.items():
        if key not in CSV_COLUMNS_V3:
            raise KeyError(f"coluna v3 desconhecida: {key}")
        record[key] = value
    return record


def validate_benchmark_record_v3(record: dict[str, object]) -> BenchmarkRecordValidationV3:
    columns = set(record)
    expected = set(CSV_COLUMNS_V3)
    missing = tuple(column for column in CSV_COLUMNS_V3 if column not in columns)
    extra = tuple(sorted(columns - expected))
    status = str(record.get("status", ""))
    invalid_status = None if status in V3_STATUSES else status
    required_when_complete = (
        "experiment_id",
        "stage",
        "case_id",
        "data_seed",
        "repeat_index",
        "profile_id",
        "operation",
        "scenario",
        "comparison_group",
        "batch_size",
        "seq_len",
        "d_model",
        "num_heads",
        "dtype",
        "latency_ms_mean",
        "latency_ms_p50",
        "latency_ms_p95",
        "throughput_tokens_s",
        "input_sha256",
        "weight_sha256",
        "output_sha256",
    )
    invalid_complete_fields: tuple[str, ...] = ()
    if status == "complete":
        invalid_complete_fields = tuple(
            field for field in required_when_complete if record.get(field, "") == ""
        )
    elif status in {"unsupported", "failed"} and not record.get("skip_reason", ""):
        invalid_complete_fields = ("skip_reason",)
    return BenchmarkRecordValidationV3(
        missing_columns=missing,
        extra_columns=extra,
        invalid_status=invalid_status,
        invalid_complete_fields=invalid_complete_fields,
    )
