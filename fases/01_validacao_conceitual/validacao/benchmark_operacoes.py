from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch

from validacao.attention import scaled_dot_product_attention_torch
from validacao.dense import dense_projection_torch
from validacao.flops import attention_matmul_flops, dense_projection_flops, qkv_projection_flops
from validacao.metrics import cosine_similarity, mean_absolute_error, mean_squared_error, r2_score
from validacao.protocolo import CSV_COLUMNS, empty_benchmark_record, validate_benchmark_record
from validacao.quantization import symmetric_int8_quantize, tensor_storage_bytes


CONFIG_KEYS_V1 = {
    "schema_version",
    "experiment_id",
    "purpose",
    "device",
    "required_gpu_name",
    "seed",
    "batch_sizes",
    "sequence_lengths",
    "dimensions",
    "operations",
    "scenarios",
    "pruning_sparsity",
    "warmup_iterations",
    "measure_iterations",
    "independent_runs",
    "scenario_order",
    "dtype",
}
CONFIG_KEYS_V2 = CONFIG_KEYS_V1 | {
    "input_distribution",
    "weight_initialization",
    "bias_initialization",
}
OPERATIONS = ("dense_projection", "self_attention")
SCENARIOS = ("baseline", "pruning_magnitude", "quantization_int8")
PURPOSES = ("smoke_test", "cpu_diagnostic", "primary_benchmark")


class ExperimentConfigurationError(ValueError):
    pass


class ExperimentEnvironmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class OperationExperimentConfig:
    schema_version: int
    experiment_id: str
    purpose: str
    device: str
    required_gpu_name: str
    seed: int
    batch_sizes: tuple[int, ...]
    sequence_lengths: tuple[int, ...]
    dimensions: tuple[int, ...]
    operations: tuple[str, ...]
    scenarios: tuple[str, ...]
    pruning_sparsity: float
    warmup_iterations: int
    measure_iterations: int
    independent_runs: int
    scenario_order: str
    dtype: str
    input_distribution: str
    weight_initialization: str
    bias_initialization: str

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        for key in ("batch_sizes", "sequence_lengths", "dimensions", "operations", "scenarios"):
            result[key] = list(result[key])
        if self.schema_version == 1:
            for key in ("input_distribution", "weight_initialization", "bias_initialization"):
                result.pop(key)
        return result


@dataclass(frozen=True)
class PreparedScenario:
    tensors: dict[str, torch.Tensor]
    validity_level: str
    quantization_method: str
    runtime_tensor_bytes: int
    compact_representation_bytes: int
    observed_sparsity: float
    uses_low_precision_storage_in_kernel: bool
    uses_sparse_kernel: bool


@dataclass(frozen=True)
class OperationBenchmarkRun:
    records: tuple[dict[str, object], ...]
    metadata: dict[str, object]


@dataclass(frozen=True)
class ExperimentSuite:
    output_dir: Path
    run_csv_paths: tuple[Path, ...]
    manifest_path: Path
    log_path: Path


def _as_unique_tuple(name: str, values: object, allowed: tuple[str, ...] | None = None) -> tuple:
    if not isinstance(values, list) or not values:
        raise ExperimentConfigurationError(f"{name} deve ser uma lista nao vazia")
    result = tuple(values)
    if len(set(result)) != len(result):
        raise ExperimentConfigurationError(f"{name} nao pode conter duplicatas")
    if allowed is not None:
        unknown = sorted(set(result) - set(allowed))
        if unknown:
            raise ExperimentConfigurationError(f"{name} contem valores desconhecidos: {unknown}")
    return result


def _require_positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ExperimentConfigurationError(f"{name} deve ser inteiro positivo")
    return value


def config_from_mapping(data: Mapping[str, object]) -> OperationExperimentConfig:
    if "schema_version" not in data:
        raise ExperimentConfigurationError("chave obrigatoria ausente: schema_version")
    schema_version = _require_positive_int("schema_version", data["schema_version"])
    if schema_version not in {1, 2}:
        raise ExperimentConfigurationError("schema_version suportado: 1 ou 2")
    expected_keys = CONFIG_KEYS_V1 if schema_version == 1 else CONFIG_KEYS_V2
    missing = sorted(expected_keys - set(data))
    extra = sorted(set(data) - expected_keys)
    if missing or extra:
        raise ExperimentConfigurationError(f"chaves invalidas; ausentes={missing}, extras={extra}")

    experiment_id = str(data["experiment_id"])
    if not re.fullmatch(r"[A-Za-z0-9._-]+", experiment_id):
        raise ExperimentConfigurationError("experiment_id deve ser seguro para nome de arquivo")

    purpose = str(data["purpose"])
    if purpose not in PURPOSES:
        raise ExperimentConfigurationError(f"purpose deve ser um de {PURPOSES}")

    device = str(data["device"])
    if device not in {"cpu", "cuda"}:
        raise ExperimentConfigurationError("device deve ser cpu ou cuda; auto nao e reproduzivel")
    if purpose == "primary_benchmark" and device != "cuda":
        raise ExperimentConfigurationError("benchmark principal exige device=cuda")

    dtype = str(data["dtype"])
    if dtype != "float32":
        raise ExperimentConfigurationError("o protocolo inicial aceita somente float32")

    input_distribution = str(data.get("input_distribution", "standard_normal"))
    weight_initialization = str(data.get("weight_initialization", "standard_normal"))
    bias_initialization = str(data.get("bias_initialization", "standard_normal"))
    if input_distribution != "standard_normal":
        raise ExperimentConfigurationError("input_distribution suportada: standard_normal")
    if weight_initialization not in {"standard_normal", "xavier_normal"}:
        raise ExperimentConfigurationError("weight_initialization deve ser standard_normal ou xavier_normal")
    if bias_initialization not in {"standard_normal", "zeros"}:
        raise ExperimentConfigurationError("bias_initialization deve ser standard_normal ou zeros")

    batch_sizes = _as_unique_tuple("batch_sizes", data["batch_sizes"])
    sequence_lengths = _as_unique_tuple("sequence_lengths", data["sequence_lengths"])
    dimensions = _as_unique_tuple("dimensions", data["dimensions"])
    for name, values in {
        "batch_sizes": batch_sizes,
        "sequence_lengths": sequence_lengths,
        "dimensions": dimensions,
    }.items():
        for value in values:
            _require_positive_int(name, value)

    operations = _as_unique_tuple("operations", data["operations"], OPERATIONS)
    scenarios = _as_unique_tuple("scenarios", data["scenarios"], SCENARIOS)
    if "baseline" not in scenarios:
        raise ExperimentConfigurationError("scenarios deve conter baseline")

    pruning_sparsity = float(data["pruning_sparsity"])
    if not 0.0 <= pruning_sparsity < 1.0:
        raise ExperimentConfigurationError("pruning_sparsity deve estar em [0, 1)")

    warmup_iterations = data["warmup_iterations"]
    if not isinstance(warmup_iterations, int) or isinstance(warmup_iterations, bool) or warmup_iterations < 0:
        raise ExperimentConfigurationError("warmup_iterations deve ser inteiro nao negativo")

    scenario_order = str(data["scenario_order"])
    if scenario_order != "rotate_by_run":
        raise ExperimentConfigurationError("scenario_order suportado: rotate_by_run")

    required_gpu_name = str(data["required_gpu_name"])
    if purpose == "primary_benchmark" and not required_gpu_name:
        raise ExperimentConfigurationError("benchmark principal exige required_gpu_name")

    return OperationExperimentConfig(
        schema_version=schema_version,
        experiment_id=experiment_id,
        purpose=purpose,
        device=device,
        required_gpu_name=required_gpu_name,
        seed=int(data["seed"]),
        batch_sizes=tuple(int(value) for value in batch_sizes),
        sequence_lengths=tuple(int(value) for value in sequence_lengths),
        dimensions=tuple(int(value) for value in dimensions),
        operations=tuple(str(value) for value in operations),
        scenarios=tuple(str(value) for value in scenarios),
        pruning_sparsity=pruning_sparsity,
        warmup_iterations=int(warmup_iterations),
        measure_iterations=_require_positive_int("measure_iterations", data["measure_iterations"]),
        independent_runs=_require_positive_int("independent_runs", data["independent_runs"]),
        scenario_order=scenario_order,
        dtype=dtype,
        input_distribution=input_distribution,
        weight_initialization=weight_initialization,
        bias_initialization=bias_initialization,
    )


def load_experiment_config(path: str | Path) -> OperationExperimentConfig:
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentConfigurationError(f"nao foi possivel ler a configuracao: {config_path}") from exc
    if not isinstance(data, dict):
        raise ExperimentConfigurationError("a configuracao deve ser um objeto JSON")
    return config_from_mapping(data)


def resolve_experiment_device(config: OperationExperimentConfig) -> torch.device:
    if config.device == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise ExperimentEnvironmentError(
                "benchmark requer CUDA, mas o PyTorch nao encontrou dispositivo CUDA"
            )
        device = torch.device("cuda:0")
        actual_name = torch.cuda.get_device_name(device)
        if config.required_gpu_name and config.required_gpu_name.casefold() not in actual_name.casefold():
            raise ExperimentEnvironmentError(
                f"GPU esperada {config.required_gpu_name!r}, mas foi encontrada {actual_name!r}"
            )
        return device
    return torch.device("cpu")


def set_deterministic_execution(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _git_metadata() -> dict[str, object]:
    def command(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return completed.stdout.strip()

    try:
        return {
            "commit": command("rev-parse", "HEAD"),
            "branch": command("branch", "--show-current"),
            "worktree_clean_at_start": command("status", "--porcelain") == "",
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"commit": "", "branch": "", "worktree_clean_at_start": False}


def environment_metadata(device: torch.device) -> dict[str, object]:
    package_names = ("numpy", "pandas", "scikit-learn", "matplotlib", "seaborn", "torchao")
    versions: dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = ""

    gpu: dict[str, object] = {
        "name": "",
        "total_memory_bytes": 0,
        "compute_capability": "",
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": f"{properties.major}.{properties.minor}",
        }

    return {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda or "",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "device": str(device),
        "gpu": gpu,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "package_versions": versions,
        "git": _git_metadata(),
    }


def _case_seed(config: OperationExperimentConfig, operation: str, batch_size: int, seq_len: int, dimension: int) -> int:
    operation_offset = OPERATIONS.index(operation) * 10_000_000
    return config.seed + operation_offset + batch_size * 1_000_000 + seq_len * 1_000 + dimension


def _randn(
    shape: tuple[int, ...],
    seed: int,
    device: torch.device,
    *,
    scale: float = 1.0,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return (torch.randn(shape, generator=generator, dtype=torch.float32) * scale).to(device)


def build_operation_inputs(
    operation: str,
    *,
    batch_size: int,
    seq_len: int,
    dimension: int,
    seed: int,
    device: torch.device,
    input_distribution: str = "standard_normal",
    weight_initialization: str = "standard_normal",
    bias_initialization: str = "standard_normal",
) -> dict[str, torch.Tensor]:
    if input_distribution != "standard_normal":
        raise ValueError(f"distribuicao de entrada desconhecida: {input_distribution}")
    if weight_initialization == "standard_normal":
        weight_scale = 1.0
    elif weight_initialization == "xavier_normal":
        # Para matrizes quadradas: sqrt(2 / (fan_in + fan_out)) = 1/sqrt(D).
        weight_scale = 1.0 / math.sqrt(dimension)
    else:
        raise ValueError(f"inicializacao de peso desconhecida: {weight_initialization}")

    if bias_initialization == "zeros":
        bias = torch.zeros(dimension, dtype=torch.float32, device=device)
    elif bias_initialization == "standard_normal":
        bias = _randn((dimension,), seed + 2, device)
    else:
        raise ValueError(f"inicializacao de bias desconhecida: {bias_initialization}")

    if operation == "dense_projection":
        return {
            "x": _randn((batch_size, seq_len, dimension), seed, device),
            "weight": _randn((dimension, dimension), seed + 1, device, scale=weight_scale),
            "bias": bias,
        }
    if operation == "self_attention":
        return {
            "x": _randn((batch_size, seq_len, dimension), seed, device),
            "w_q": _randn((dimension, dimension), seed + 1, device, scale=weight_scale),
            "w_k": _randn((dimension, dimension), seed + 2, device, scale=weight_scale),
            "w_v": _randn((dimension, dimension), seed + 3, device, scale=weight_scale),
            "w_o": _randn((dimension, dimension), seed + 4, device, scale=weight_scale),
        }
    raise ValueError(f"operacao desconhecida: {operation}")


def _weight_names(operation: str) -> tuple[str, ...]:
    if operation == "dense_projection":
        return ("weight",)
    if operation == "self_attention":
        return ("w_q", "w_k", "w_v", "w_o")
    raise ValueError(f"operacao desconhecida: {operation}")


def _exact_magnitude_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    total = tensor.numel()
    prune_count = int(round(total * sparsity))
    if prune_count == 0:
        return tensor.clone()
    if prune_count >= total:
        raise ValueError("pruning nao pode remover todos os elementos")
    indices = torch.topk(tensor.abs().reshape(-1), k=prune_count, largest=False).indices
    result = tensor.clone().reshape(-1)
    result[indices] = 0.0
    return result.reshape_as(tensor)


def _storage_bytes(tensors: Iterable[torch.Tensor]) -> int:
    return int(sum(tensor_storage_bytes(tensor) for tensor in tensors))


def prepare_scenario(
    operation: str,
    scenario: str,
    base_tensors: Mapping[str, torch.Tensor],
    *,
    pruning_sparsity: float,
) -> PreparedScenario:
    if scenario not in SCENARIOS:
        raise ValueError(f"cenario desconhecido: {scenario}")
    tensors = {name: tensor.clone() for name, tensor in base_tensors.items()}
    weights = _weight_names(operation)

    if scenario == "baseline":
        runtime_bytes = _storage_bytes(tensors.values())
        return PreparedScenario(
            tensors=tensors,
            validity_level="algoritmico",
            quantization_method="none",
            runtime_tensor_bytes=runtime_bytes,
            compact_representation_bytes=runtime_bytes,
            observed_sparsity=0.0,
            uses_low_precision_storage_in_kernel=False,
            uses_sparse_kernel=False,
        )

    if scenario == "pruning_magnitude":
        for name in weights:
            tensors[name] = _exact_magnitude_prune(tensors[name], pruning_sparsity)
        zeros = sum(int((tensors[name] == 0).sum().item()) for name in weights)
        total = sum(tensors[name].numel() for name in weights)
        runtime_bytes = _storage_bytes(tensors.values())
        return PreparedScenario(
            tensors=tensors,
            validity_level="conceitual_algoritmico",
            quantization_method="none",
            runtime_tensor_bytes=runtime_bytes,
            compact_representation_bytes=runtime_bytes,
            observed_sparsity=zeros / total,
            uses_low_precision_storage_in_kernel=False,
            uses_sparse_kernel=False,
        )

    compact_weight_bytes = 0
    for name in weights:
        quantized, scale, dequantized = symmetric_int8_quantize(tensors[name])
        compact_weight_bytes += tensor_storage_bytes(quantized) + tensor_storage_bytes(scale)
        tensors[name] = dequantized.to(tensors[name].device, dtype=torch.float32)
    non_weight_bytes = _storage_bytes(
        tensor for name, tensor in tensors.items() if name not in weights
    )
    return PreparedScenario(
        tensors=tensors,
        validity_level="numerico",
        quantization_method="manual_symmetric_int8_weight_only_dequantized",
        runtime_tensor_bytes=_storage_bytes(tensors.values()),
        compact_representation_bytes=non_weight_bytes + compact_weight_bytes,
        observed_sparsity=0.0,
        uses_low_precision_storage_in_kernel=False,
        uses_sparse_kernel=False,
    )


def run_operation(operation: str, tensors: Mapping[str, torch.Tensor]) -> torch.Tensor:
    if operation == "dense_projection":
        return dense_projection_torch(tensors["x"], tensors["weight"], tensors["bias"])
    if operation == "self_attention":
        query = dense_projection_torch(tensors["x"], tensors["w_q"])
        key = dense_projection_torch(tensors["x"], tensors["w_k"])
        value = dense_projection_torch(tensors["x"], tensors["w_v"])
        context = scaled_dot_product_attention_torch(query, key, value)
        return dense_projection_torch(context, tensors["w_o"])
    raise ValueError(f"operacao desconhecida: {operation}")


def operation_flops(operation: str, *, batch_size: int, seq_len: int, dimension: int) -> int:
    if operation == "dense_projection":
        return dense_projection_flops(
            batch_size=batch_size,
            seq_len=seq_len,
            in_features=dimension,
            out_features=dimension,
            bias=True,
        )
    if operation == "self_attention":
        return (
            qkv_projection_flops(
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=dimension,
                bias=False,
            )
            + attention_matmul_flops(
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=dimension,
                num_heads=1,
            )
            + dense_projection_flops(
                batch_size=batch_size,
                seq_len=seq_len,
                in_features=dimension,
                out_features=dimension,
                bias=False,
            )
        )
    raise ValueError(f"operacao desconhecida: {operation}")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_operation(
    operation: str,
    tensors: Mapping[str, torch.Tensor],
    *,
    device: torch.device,
    warmup_iterations: int,
    measure_iterations: int,
    cpu_memory_estimate: int,
) -> tuple[torch.Tensor, tuple[float, ...], int]:
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            run_operation(operation, tensors)
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        timings: list[float] = []
        output: torch.Tensor | None = None
        for _ in range(measure_iterations):
            _synchronize(device)
            start = time.perf_counter_ns()
            output = run_operation(operation, tensors)
            _synchronize(device)
            timings.append((time.perf_counter_ns() - start) / 1_000_000.0)

        if output is None:
            raise RuntimeError("nenhuma medicao executada")
        memory_bytes = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else int(cpu_memory_estimate)
        )
    return output.detach(), tuple(timings), memory_bytes


def _tensor_hash(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _tensor_mapping_hash(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_hash(tensors[name]).encode("ascii"))
    return digest.hexdigest()


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile, method="linear"))


def _strict_validate_record(record: dict[str, object]) -> None:
    structural = validate_benchmark_record(record)
    if not structural.is_valid:
        raise ValueError(f"registro invalido: {structural}")
    finite_fields = (
        "latency_ms_mean",
        "latency_ms_p50",
        "latency_ms_p95",
        "throughput_tokens_s",
        "mse",
        "mae",
        "r2",
        "cosine_similarity",
    )
    for field in finite_fields:
        if not math.isfinite(float(record[field])):
            raise ValueError(f"campo nao finito: {field}")
    for field in ("batch_size", "seq_len", "d_model", "num_heads", "max_memory_bytes", "theoretical_flops"):
        if int(record[field]) <= 0:
            raise ValueError(f"campo deve ser positivo: {field}")
    if float(record["latency_ms_mean"]) <= 0.0 or float(record["throughput_tokens_s"]) <= 0.0:
        raise ValueError("latencia e throughput devem ser positivos")


def _scenario_order(config: OperationExperimentConfig, run_index: int) -> tuple[str, ...]:
    scenarios = config.scenarios
    offset = (run_index - 1) % len(scenarios)
    return scenarios[offset:] + scenarios[:offset]


def _iter_cases(config: OperationExperimentConfig) -> Iterable[tuple[str, int, int, int]]:
    for operation in config.operations:
        for batch_size in config.batch_sizes:
            for seq_len in config.sequence_lengths:
                for dimension in config.dimensions:
                    yield operation, batch_size, seq_len, dimension


def run_independent_benchmark(
    config: OperationExperimentConfig,
    *,
    run_index: int,
    device: torch.device,
) -> OperationBenchmarkRun:
    if not 1 <= run_index <= config.independent_runs:
        raise ValueError("run_index fora da configuracao")
    set_deterministic_execution(config.seed)
    order = _scenario_order(config, run_index)
    records: list[dict[str, object]] = []
    fingerprints: list[dict[str, object]] = []

    for operation, batch_size, seq_len, dimension in _iter_cases(config):
        seed = _case_seed(config, operation, batch_size, seq_len, dimension)
        base_tensors = build_operation_inputs(
            operation,
            batch_size=batch_size,
            seq_len=seq_len,
            dimension=dimension,
            seed=seed,
            device=device,
            input_distribution=config.input_distribution,
            weight_initialization=config.weight_initialization,
            bias_initialization=config.bias_initialization,
        )
        with torch.inference_mode():
            reference_output = run_operation(operation, base_tensors).detach()
        if not bool(torch.isfinite(reference_output).all()):
            raise ValueError("baseline produziu NaN ou infinito")

        input_hash = _tensor_mapping_hash(base_tensors)
        for scenario in order:
            prepared = prepare_scenario(
                operation,
                scenario,
                base_tensors,
                pruning_sparsity=config.pruning_sparsity,
            )
            output, timings, memory_bytes = measure_operation(
                operation,
                prepared.tensors,
                device=device,
                warmup_iterations=config.warmup_iterations,
                measure_iterations=config.measure_iterations,
                cpu_memory_estimate=prepared.runtime_tensor_bytes,
            )
            if output.shape != reference_output.shape or output.dtype != reference_output.dtype:
                raise ValueError("shape ou dtype da saida candidata diverge do baseline")
            if not bool(torch.isfinite(output).all()):
                raise ValueError("cenario produziu NaN ou infinito")

            reference = reference_output.float().cpu().numpy()
            candidate = output.float().cpu().numpy()
            mean_latency = float(np.mean(np.asarray(timings, dtype=np.float64)))
            tokens = batch_size * seq_len
            record = empty_benchmark_record(
                torch_version=torch.__version__,
                cuda_version=torch.version.cuda or "",
                gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "",
                device=str(device),
                model_kind=operation,
                scenario=scenario,
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=dimension,
                num_heads=1,
                dtype=config.dtype,
                pruning_sparsity=config.pruning_sparsity if scenario == "pruning_magnitude" else 0.0,
                quantization_method=prepared.quantization_method,
                validity_level=prepared.validity_level,
                latency_ms_mean=mean_latency,
                latency_ms_p50=_percentile(timings, 50.0),
                latency_ms_p95=_percentile(timings, 95.0),
                throughput_tokens_s=float(tokens / (mean_latency / 1000.0)),
                max_memory_bytes=memory_bytes,
                theoretical_flops=operation_flops(
                    operation,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    dimension=dimension,
                ),
                mse=mean_squared_error(reference, candidate),
                mae=mean_absolute_error(reference, candidate),
                r2=r2_score(reference, candidate),
                cosine_similarity=cosine_similarity(reference, candidate),
            )
            _strict_validate_record(record)
            records.append(record)
            fingerprints.append(
                {
                    "operation": operation,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "dimension": dimension,
                    "scenario": scenario,
                    "case_seed": seed,
                    "input_sha256": input_hash,
                    "output_sha256": _tensor_hash(output),
                    "runtime_tensor_bytes": prepared.runtime_tensor_bytes,
                    "compact_representation_bytes": prepared.compact_representation_bytes,
                    "observed_sparsity": prepared.observed_sparsity,
                    "uses_low_precision_storage_in_kernel": prepared.uses_low_precision_storage_in_kernel,
                    "uses_sparse_kernel": prepared.uses_sparse_kernel,
                }
            )

    metadata = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "purpose": config.purpose,
        "independent_run": run_index,
        "seed": config.seed,
        "same_inputs_across_independent_runs": True,
        "input_distribution": config.input_distribution,
        "weight_initialization": config.weight_initialization,
        "bias_initialization": config.bias_initialization,
        "scenario_order": list(order),
        "warmup_iterations": config.warmup_iterations,
        "measure_iterations": config.measure_iterations,
        "percentile_method": "numpy.percentile_linear",
        "flops_scope": "dense_matmuls_and_bias; self_attention_excludes_softmax_and_scaling",
        "memory_measurement_kind": (
            "torch_cuda_max_memory_allocated" if device.type == "cuda" else "runtime_tensor_storage_estimate"
        ),
        "performance_claim_allowed": config.purpose == "primary_benchmark" and device.type == "cuda",
        "fingerprints": fingerprints,
    }
    return OperationBenchmarkRun(records=tuple(records), metadata=metadata)


def _write_run(run: OperationBenchmarkRun, csv_path: Path) -> None:
    for record in run.records:
        _strict_validate_record(record)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(run.records)
    csv_path.with_suffix(".metadata.json").write_text(
        json.dumps(run.metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute_experiment_suite(
    config: OperationExperimentConfig,
    output_dir: str | Path,
) -> ExperimentSuite:
    device = resolve_experiment_device(config)
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(f"diretorio de saida nao esta vazio: {output_path}")
    # Capture antes de criar artefatos no worktree; caso contrario, o proprio
    # snapshot faria um repositorio inicialmente limpo parecer sujo.
    captured_environment = environment_metadata(device)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = output_path / "config.snapshot.json"
    environment_path = output_path / "environment.json"
    manifest_path = output_path / "manifest.json"
    log_path = output_path / "execution.log"
    config_path.write_text(
        json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    environment_path.write_text(
        json.dumps(captured_environment, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    log_lines = [
        f"{datetime.now().astimezone().isoformat(timespec='seconds')} START {config.experiment_id}",
        f"purpose={config.purpose} device={device} seed={config.seed}",
    ]
    run_paths: list[Path] = []
    manifest: dict[str, object] = {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "status": "running",
        "config": config_path.name,
        "environment": environment_path.name,
        "log": log_path.name,
        "runs": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    try:
        for run_index in range(1, config.independent_runs + 1):
            log_lines.append(f"RUN {run_index}/{config.independent_runs} START")
            run = run_independent_benchmark(config, run_index=run_index, device=device)
            csv_path = output_path / f"run_{run_index:02d}.csv"
            _write_run(run, csv_path)
            metadata_path = csv_path.with_suffix(".metadata.json")
            run_paths.append(csv_path)
            manifest["runs"].append(
                {
                    "independent_run": run_index,
                    "csv": csv_path.name,
                    "csv_sha256": _file_sha256(csv_path),
                    "metadata": metadata_path.name,
                    "metadata_sha256": _file_sha256(metadata_path),
                    "records": len(run.records),
                }
            )
            log_lines.append(f"RUN {run_index}/{config.independent_runs} OK records={len(run.records)}")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        log_lines.append(f"{manifest['completed_at']} COMPLETE")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        log_lines.append(f"FAILED {type(exc).__name__}: {exc}")
        raise
    finally:
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return ExperimentSuite(
        output_dir=output_path,
        run_csv_paths=tuple(run_paths),
        manifest_path=manifest_path,
        log_path=log_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa benchmarks reproduziveis das operacoes isoladas.")
    parser.add_argument("--config", required=True, help="Configuracao JSON versionada.")
    parser.add_argument("--output-dir", required=True, help="Diretorio novo para evidencias.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_experiment_config(args.config)
        suite = execute_experiment_suite(config, args.output_dir)
    except (ExperimentConfigurationError, ExperimentEnvironmentError, FileExistsError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"falha experimental: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"experimento concluido: {config.experiment_id}")
    for path in suite.run_csv_paths:
        print(path)
    print(suite.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
