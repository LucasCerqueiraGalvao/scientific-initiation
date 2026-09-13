from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from validacao.attention import combine_heads, split_heads
from validacao.benchmark_operacoes import (
    ExperimentConfigurationError,
    ExperimentEnvironmentError,
    environment_metadata,
    resolve_experiment_device,
    set_deterministic_execution,
)
from validacao.flops import attention_matmul_flops, dense_projection_flops, qkv_projection_flops
from validacao.metrics import cosine_similarity, mean_absolute_error, mean_squared_error, r2_score
from validacao.protocolo import (
    CSV_COLUMNS_V3,
    empty_benchmark_record_v3,
    validate_benchmark_record_v3,
)
from validacao.quantization import symmetric_int8_quantize_per_row


OPERATIONS_V3 = ("dense_projection", "multi_head_self_attention")
SCENARIO_KINDS = (
    "baseline",
    "pruning_magnitude",
    "quantization_int8_fake_per_row",
    "quantization_int8_weight_only",
    "quantization_int8_dynamic",
    "pruning_2to4",
)
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
INPUT_DISTRIBUTIONS = (
    "standard_normal",
    "uniform_unit_variance",
    "outlier_normal",
)
TIMING_COLUMNS = [
    "case_id",
    "experiment_id",
    "repeat_index",
    "data_seed",
    "profile_id",
    "operation",
    "scenario",
    "iteration",
    "latency_ms",
]


@dataclass(frozen=True)
class BenchmarkProfileV3:
    identifier: str
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    input_distribution: str


@dataclass(frozen=True)
class ScenarioV3:
    identifier: str
    kind: str
    comparison_group: str
    dtype: str
    pruning_sparsity: float = 0.0


@dataclass(frozen=True)
class MeasurementV3:
    warmup_iterations: int
    measure_iterations: int
    repetitions: int


@dataclass(frozen=True)
class ExecutionV3:
    scenario_order: str
    compile: bool
    compile_mode: str
    save_raw_timings: bool
    bootstrap_resamples: int
    bootstrap_seed: int
    capabilities_file: str


@dataclass(frozen=True)
class HardwareRequirementsV3:
    compute_capability: str
    require_triton: bool
    require_cusparselt: bool
    temperature_abort_c: int


@dataclass(frozen=True)
class OperationExperimentConfigV3:
    schema_version: int
    experiment_id: str
    purpose: str
    stage: str
    device: str
    required_gpu_name: str
    data_seeds: tuple[int, ...]
    input_distributions: tuple[str, ...]
    profiles: tuple[BenchmarkProfileV3, ...]
    operations: tuple[str, ...]
    scenarios: tuple[ScenarioV3, ...]
    weight_initialization: str
    bias_initialization: str
    measurement: MeasurementV3
    execution: ExecutionV3
    hardware_requirements: HardwareRequirementsV3

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["data_seeds"] = list(self.data_seeds)
        result["input_distributions"] = list(self.input_distributions)
        result["operations"] = list(self.operations)
        result["profiles"] = [asdict(profile) for profile in self.profiles]
        result["scenarios"] = [asdict(scenario) for scenario in self.scenarios]
        return result


@dataclass(frozen=True)
class OperationSuiteV3:
    output_dir: Path
    run_paths: tuple[Path, ...]
    timings_path: Path | None
    manifest_path: Path


class DenseProjectionModule(nn.Module):
    def __init__(self, tensors: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        dimension = int(tensors["weight"].shape[0])
        self.projection = nn.Linear(dimension, dimension, bias=True)
        with torch.no_grad():
            self.projection.weight.copy_(tensors["weight"])
            self.projection.bias.copy_(tensors["bias"])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection(inputs)


class MultiHeadSelfAttentionModule(nn.Module):
    def __init__(self, tensors: Mapping[str, torch.Tensor], num_heads: int) -> None:
        super().__init__()
        dimension = int(tensors["w_q"].shape[0])
        self.num_heads = num_heads
        self.q_proj = nn.Linear(dimension, dimension, bias=False)
        self.k_proj = nn.Linear(dimension, dimension, bias=False)
        self.v_proj = nn.Linear(dimension, dimension, bias=False)
        self.out_proj = nn.Linear(dimension, dimension, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(tensors["w_q"])
            self.k_proj.weight.copy_(tensors["w_k"])
            self.v_proj.weight.copy_(tensors["w_v"])
            self.out_proj.weight.copy_(tensors["w_o"])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query = split_heads(self.q_proj(inputs), self.num_heads)
        key = split_heads(self.k_proj(inputs), self.num_heads)
        value = split_heads(self.v_proj(inputs), self.num_heads)
        context = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=False,
        )
        return self.out_proj(combine_heads(context))


def _positive_int(name: str, value: object, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "nao negativo" if allow_zero else "positivo"
        raise ExperimentConfigurationError(f"{name} deve ser inteiro {qualifier}")
    return value


def _unique_strings(name: str, value: object, allowed: Iterable[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ExperimentConfigurationError(f"{name} deve ser lista nao vazia")
    result = tuple(str(item) for item in value)
    if len(set(result)) != len(result):
        raise ExperimentConfigurationError(f"{name} nao pode conter duplicatas")
    unknown = sorted(set(result) - set(allowed))
    if unknown:
        raise ExperimentConfigurationError(f"{name} contem valores desconhecidos: {unknown}")
    return result


def config_v3_from_mapping(data: Mapping[str, object]) -> OperationExperimentConfigV3:
    expected = {
        "schema_version",
        "experiment_id",
        "purpose",
        "stage",
        "device",
        "required_gpu_name",
        "data_seeds",
        "input_distributions",
        "profiles",
        "operations",
        "scenarios",
        "weight_initialization",
        "bias_initialization",
        "measurement",
        "execution",
        "hardware_requirements",
    }
    missing = sorted(expected - set(data))
    extra = sorted(set(data) - expected)
    if missing or extra:
        raise ExperimentConfigurationError(f"chaves v3 invalidas; ausentes={missing}, extras={extra}")
    if data["schema_version"] != 3:
        raise ExperimentConfigurationError("schema_version v3 deve ser 3")

    stage = str(data["stage"])
    if stage not in {"synthetic_numerical", "synthetic_hardware"}:
        raise ExperimentConfigurationError("stage v3 desconhecido")
    device = str(data["device"])
    if device not in {"cpu", "cuda"}:
        raise ExperimentConfigurationError("device deve ser cpu ou cuda")
    if stage == "synthetic_hardware" and device != "cuda":
        raise ExperimentConfigurationError("synthetic_hardware exige CUDA")

    raw_seeds = data["data_seeds"]
    if not isinstance(raw_seeds, list) or not raw_seeds:
        raise ExperimentConfigurationError("data_seeds deve ser lista nao vazia")
    data_seeds = tuple(int(seed) for seed in raw_seeds)
    if len(set(data_seeds)) != len(data_seeds):
        raise ExperimentConfigurationError("data_seeds nao pode conter duplicatas")

    distributions = _unique_strings(
        "input_distributions",
        data["input_distributions"],
        INPUT_DISTRIBUTIONS,
    )
    raw_profiles = data["profiles"]
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ExperimentConfigurationError("profiles deve ser lista nao vazia")
    profiles: list[BenchmarkProfileV3] = []
    profile_ids: set[str] = set()
    for raw in raw_profiles:
        if not isinstance(raw, Mapping):
            raise ExperimentConfigurationError("cada profile deve ser objeto")
        if set(raw) != {"identifier", "batch_size", "seq_len", "d_model", "num_heads", "input_distribution"}:
            raise ExperimentConfigurationError("profile v3 possui chaves invalidas")
        identifier = str(raw["identifier"])
        if not identifier or identifier in profile_ids:
            raise ExperimentConfigurationError("identifier de profile vazio ou duplicado")
        profile_ids.add(identifier)
        profile = BenchmarkProfileV3(
            identifier=identifier,
            batch_size=_positive_int("batch_size", raw["batch_size"]),
            seq_len=_positive_int("seq_len", raw["seq_len"]),
            d_model=_positive_int("d_model", raw["d_model"]),
            num_heads=_positive_int("num_heads", raw["num_heads"]),
            input_distribution=str(raw["input_distribution"]),
        )
        if profile.input_distribution not in distributions:
            raise ExperimentConfigurationError("distribuicao do profile nao declarada")
        if profile.d_model % profile.num_heads != 0:
            raise ExperimentConfigurationError("d_model deve ser divisivel por num_heads")
        profiles.append(profile)

    raw_scenarios = data["scenarios"]
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ExperimentConfigurationError("scenarios deve ser lista nao vazia")
    scenarios: list[ScenarioV3] = []
    scenario_ids: set[str] = set()
    for raw in raw_scenarios:
        if not isinstance(raw, Mapping):
            raise ExperimentConfigurationError("cada scenario deve ser objeto")
        allowed_keys = {"identifier", "kind", "comparison_group", "dtype", "pruning_sparsity"}
        if set(raw) - allowed_keys or not {"identifier", "kind", "comparison_group", "dtype"} <= set(raw):
            raise ExperimentConfigurationError("scenario v3 possui chaves invalidas")
        scenario = ScenarioV3(
            identifier=str(raw["identifier"]),
            kind=str(raw["kind"]),
            comparison_group=str(raw["comparison_group"]),
            dtype=str(raw["dtype"]),
            pruning_sparsity=float(raw.get("pruning_sparsity", 0.0)),
        )
        if not scenario.identifier or scenario.identifier in scenario_ids:
            raise ExperimentConfigurationError("identifier de scenario vazio ou duplicado")
        scenario_ids.add(scenario.identifier)
        if scenario.kind not in SCENARIO_KINDS:
            raise ExperimentConfigurationError(f"kind desconhecido: {scenario.kind}")
        if scenario.dtype not in DTYPES:
            raise ExperimentConfigurationError(f"dtype desconhecido: {scenario.dtype}")
        if not 0.0 <= scenario.pruning_sparsity < 1.0:
            raise ExperimentConfigurationError("pruning_sparsity deve estar em [0, 1)")
        if scenario.kind == "pruning_2to4" and scenario.pruning_sparsity != 0.5:
            raise ExperimentConfigurationError("pruning 2:4 exige sparsity 0.5")
        scenarios.append(scenario)

    groups = {scenario.comparison_group for scenario in scenarios}
    for group in groups:
        baselines = [scenario for scenario in scenarios if scenario.comparison_group == group and scenario.kind == "baseline"]
        if len(baselines) != 1:
            raise ExperimentConfigurationError(f"comparison_group {group!r} exige exatamente um baseline")
        group_dtypes = {scenario.dtype for scenario in scenarios if scenario.comparison_group == group}
        if len(group_dtypes) != 1:
            raise ExperimentConfigurationError(f"comparison_group {group!r} mistura dtypes")

    measurement_raw = data["measurement"]
    execution_raw = data["execution"]
    hardware_raw = data["hardware_requirements"]
    if not isinstance(measurement_raw, Mapping) or set(measurement_raw) != {
        "warmup_iterations", "measure_iterations", "repetitions"
    }:
        raise ExperimentConfigurationError("measurement v3 invalido")
    if not isinstance(execution_raw, Mapping) or set(execution_raw) != {
        "scenario_order", "compile", "compile_mode", "save_raw_timings",
        "bootstrap_resamples", "bootstrap_seed", "capabilities_file",
    }:
        raise ExperimentConfigurationError("execution v3 invalido")
    if str(execution_raw["scenario_order"]) != "deterministic_shuffle":
        raise ExperimentConfigurationError("scenario_order suportado: deterministic_shuffle")
    if not isinstance(hardware_raw, Mapping) or set(hardware_raw) != {
        "compute_capability", "require_triton", "require_cusparselt", "temperature_abort_c"
    }:
        raise ExperimentConfigurationError("hardware_requirements v3 invalido")

    return OperationExperimentConfigV3(
        schema_version=3,
        experiment_id=str(data["experiment_id"]),
        purpose=str(data["purpose"]),
        stage=stage,
        device=device,
        required_gpu_name=str(data["required_gpu_name"]),
        data_seeds=data_seeds,
        input_distributions=distributions,
        profiles=tuple(profiles),
        operations=_unique_strings("operations", data["operations"], OPERATIONS_V3),
        scenarios=tuple(scenarios),
        weight_initialization=str(data["weight_initialization"]),
        bias_initialization=str(data["bias_initialization"]),
        measurement=MeasurementV3(
            warmup_iterations=_positive_int(
                "warmup_iterations", measurement_raw["warmup_iterations"], allow_zero=True
            ),
            measure_iterations=_positive_int("measure_iterations", measurement_raw["measure_iterations"]),
            repetitions=_positive_int("repetitions", measurement_raw["repetitions"]),
        ),
        execution=ExecutionV3(
            scenario_order="deterministic_shuffle",
            compile=bool(execution_raw["compile"]),
            compile_mode=str(execution_raw["compile_mode"]),
            save_raw_timings=bool(execution_raw["save_raw_timings"]),
            bootstrap_resamples=_positive_int("bootstrap_resamples", execution_raw["bootstrap_resamples"]),
            bootstrap_seed=int(execution_raw["bootstrap_seed"]),
            capabilities_file=str(execution_raw["capabilities_file"]),
        ),
        hardware_requirements=HardwareRequirementsV3(
            compute_capability=str(hardware_raw["compute_capability"]),
            require_triton=bool(hardware_raw["require_triton"]),
            require_cusparselt=bool(hardware_raw["require_cusparselt"]),
            temperature_abort_c=_positive_int("temperature_abort_c", hardware_raw["temperature_abort_c"]),
        ),
    )


def load_experiment_config_v3(path: str | Path) -> OperationExperimentConfigV3:
    config_path = Path(path)
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentConfigurationError(f"nao foi possivel ler config v3: {config_path}") from exc
    if not isinstance(data, dict):
        raise ExperimentConfigurationError("config v3 deve ser objeto JSON")
    return config_v3_from_mapping(data)


def _derived_seed(*parts: object) -> int:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & 0x7FFF_FFFF


def _random_tensor(shape: tuple[int, ...], seed: int, distribution: str) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    if distribution == "standard_normal":
        return torch.randn(shape, generator=generator, dtype=torch.float32)
    if distribution == "uniform_unit_variance":
        return (torch.rand(shape, generator=generator, dtype=torch.float32) * 2.0 - 1.0) * math.sqrt(3.0)
    if distribution == "outlier_normal":
        result = torch.randn(shape, generator=generator, dtype=torch.float32)
        flat = result.reshape(-1)
        flat[::100] *= 10.0
        return result
    raise ValueError(f"distribuicao desconhecida: {distribution}")


def build_operation_inputs_v3(
    operation: str,
    profile: BenchmarkProfileV3,
    data_seed: int,
) -> dict[str, torch.Tensor]:
    case_seed = _derived_seed(operation, profile.identifier, data_seed)
    dimension = profile.d_model
    inputs = _random_tensor(
        (profile.batch_size, profile.seq_len, dimension),
        case_seed,
        profile.input_distribution,
    )
    weight_names = ("weight",) if operation == "dense_projection" else ("w_q", "w_k", "w_v", "w_o")
    tensors: dict[str, torch.Tensor] = {"x": inputs}
    for index, name in enumerate(weight_names, start=1):
        tensors[name] = _random_tensor(
            (dimension, dimension),
            case_seed + index,
            "standard_normal",
        ) / math.sqrt(dimension)
    if operation == "dense_projection":
        tensors["bias"] = torch.zeros(dimension, dtype=torch.float32)
    return tensors


def _weight_names(operation: str) -> tuple[str, ...]:
    if operation == "dense_projection":
        return ("weight",)
    if operation == "multi_head_self_attention":
        return ("w_q", "w_k", "w_v", "w_o")
    raise ValueError(f"operacao desconhecida: {operation}")


def exact_magnitude_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    prune_count = int(round(tensor.numel() * sparsity))
    if prune_count == 0:
        return tensor.clone()
    if prune_count >= tensor.numel():
        raise ValueError("pruning nao pode remover todos os elementos")
    indices = torch.topk(tensor.abs().reshape(-1), k=prune_count, largest=False).indices
    result = tensor.clone().reshape(-1)
    result[indices] = 0
    return result.reshape_as(tensor)


def magnitude_prune_2to4(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 2 or tensor.shape[-1] % 4 != 0:
        raise ValueError("pruning 2:4 exige matriz 2D com colunas divisiveis por 4")
    grouped = tensor.reshape(tensor.shape[0], -1, 4)
    keep = grouped.abs().topk(2, dim=-1, largest=True).indices
    mask = torch.zeros_like(grouped, dtype=torch.bool).scatter_(-1, keep, True)
    return (grouped * mask).reshape_as(tensor)


def _prepare_base_tensors(
    operation: str,
    base_tensors: Mapping[str, torch.Tensor],
    scenario: ScenarioV3,
) -> dict[str, torch.Tensor]:
    tensors = {name: tensor.clone() for name, tensor in base_tensors.items()}
    if scenario.kind == "pruning_magnitude":
        for name in _weight_names(operation):
            tensors[name] = exact_magnitude_prune(tensors[name], scenario.pruning_sparsity)
    elif scenario.kind == "quantization_int8_fake_per_row":
        for name in _weight_names(operation):
            _, _, tensors[name] = symmetric_int8_quantize_per_row(tensors[name])
    elif scenario.kind == "pruning_2to4":
        for name in _weight_names(operation):
            tensors[name] = magnitude_prune_2to4(tensors[name])
    return tensors


def _module_from_tensors(
    operation: str,
    tensors: Mapping[str, torch.Tensor],
    profile: BenchmarkProfileV3,
    scenario: ScenarioV3,
    device: torch.device,
) -> tuple[nn.Module, torch.Tensor]:
    prepared = _prepare_base_tensors(operation, tensors, scenario)
    module: nn.Module
    if operation == "dense_projection":
        module = DenseProjectionModule(prepared)
    else:
        module = MultiHeadSelfAttentionModule(prepared, profile.num_heads)
    dtype = DTYPES[scenario.dtype]
    module = module.eval().to(device=device, dtype=dtype)
    inputs = prepared["x"].to(device=device, dtype=dtype)

    if scenario.kind in {"quantization_int8_weight_only", "quantization_int8_dynamic"}:
        try:
            from torchao.quantization import (
                Int8DynamicActivationInt8WeightConfig,
                Int8WeightOnlyConfig,
                quantize_,
            )
            from torchao.quantization.granularity import PerRow
        except ImportError as exc:
            raise RuntimeError("torchao indisponivel") from exc
        quantization_config = (
            Int8WeightOnlyConfig(version=2, granularity=PerRow())
            if scenario.kind == "quantization_int8_weight_only"
            else Int8DynamicActivationInt8WeightConfig(version=2, granularity=PerRow())
        )
        quantize_(module, quantization_config, device=device)
    elif scenario.kind == "pruning_2to4":
        try:
            from torch.sparse import to_sparse_semi_structured
        except ImportError as exc:
            raise RuntimeError("API semi-structured indisponivel") from exc
        for child in module.modules():
            if isinstance(child, nn.Linear):
                child.weight = nn.Parameter(
                    to_sparse_semi_structured(child.weight.detach()),
                    requires_grad=False,
                )
    return module, inputs


def _tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().to_dense() if tensor.layout != torch.strided else tensor.detach()
    array = value.float().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _mapping_hash(tensors: Mapping[str, torch.Tensor], names: Iterable[str] | None = None) -> str:
    digest = hashlib.sha256()
    selected = sorted(tensors if names is None else names)
    for name in selected:
        digest.update(name.encode("utf-8"))
        digest.update(_tensor_hash(tensors[name]).encode("ascii"))
    return digest.hexdigest()


def _module_storage_bytes(module: nn.Module) -> int:
    total = 0
    seen: set[int] = set()
    for parameter in module.parameters():
        candidates = []
        if hasattr(parameter, "qdata"):
            candidates.append(parameter.qdata)
            if hasattr(parameter, "scale"):
                candidates.append(parameter.scale)
        else:
            candidates.append(parameter)
        for tensor in candidates:
            try:
                pointer = int(tensor.untyped_storage().data_ptr())
                size = int(tensor.untyped_storage().nbytes())
            except (RuntimeError, AttributeError):
                pointer = id(tensor)
                size = int(tensor.numel() * tensor.element_size())
            if pointer not in seen:
                seen.add(pointer)
                total += size
    return total


def _module_weight_hash(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        digest.update(name.encode("utf-8"))
        weight = child.weight.detach()
        tensors = [weight]
        if hasattr(weight, "qdata"):
            tensors = [weight.qdata]
            if hasattr(weight, "scale"):
                tensors.append(weight.scale)
        for tensor in tensors:
            try:
                dense = tensor.to_dense() if tensor.layout != torch.strided else tensor
                value = dense.cpu().contiguous()
                digest.update(str(value.dtype).encode("ascii"))
                digest.update(json.dumps(list(value.shape)).encode("ascii"))
                digest.update(value.view(torch.uint8).numpy().tobytes())
            except (RuntimeError, TypeError):
                digest.update(repr(tensor).encode("utf-8"))
    return digest.hexdigest()


def _observed_sparsity(module: nn.Module) -> float:
    zeros = 0
    total = 0
    for child in module.modules():
        if isinstance(child, nn.Linear):
            weight = child.weight.detach()
            try:
                dense = weight.to_dense() if weight.layout != torch.strided else weight
                zeros += int(torch.count_nonzero(dense == 0).item())
                total += dense.numel()
            except RuntimeError:
                return 0.5
    return zeros / total if total else 0.0


def operation_flops_v3(operation: str, profile: BenchmarkProfileV3) -> int:
    if operation == "dense_projection":
        return dense_projection_flops(
            batch_size=profile.batch_size,
            seq_len=profile.seq_len,
            in_features=profile.d_model,
            out_features=profile.d_model,
            bias=True,
        )
    return (
        qkv_projection_flops(
            batch_size=profile.batch_size,
            seq_len=profile.seq_len,
            d_model=profile.d_model,
            bias=False,
        )
        + attention_matmul_flops(
            batch_size=profile.batch_size,
            seq_len=profile.seq_len,
            d_model=profile.d_model,
            num_heads=profile.num_heads,
        )
        + dense_projection_flops(
            batch_size=profile.batch_size,
            seq_len=profile.seq_len,
            in_features=profile.d_model,
            out_features=profile.d_model,
            bias=False,
        )
    )


def _measure(
    callable_module: Callable[[torch.Tensor], torch.Tensor],
    inputs: torch.Tensor,
    device: torch.device,
    measurement: MeasurementV3,
) -> tuple[torch.Tensor, tuple[float, ...], int, int]:
    with torch.inference_mode():
        for _ in range(measurement.warmup_iterations):
            callable_module(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        timings: list[float] = []
        output: torch.Tensor | None = None
        for _ in range(measurement.measure_iterations):
            if device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                output = callable_module(inputs)
                end.record()
                end.synchronize()
                timings.append(float(start.elapsed_time(end)))
            else:
                started = time.perf_counter_ns()
                output = callable_module(inputs)
                timings.append((time.perf_counter_ns() - started) / 1_000_000.0)
        if output is None:
            raise RuntimeError("nenhuma medicao executada")
        allocated = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        reserved = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
    return output.detach(), tuple(timings), allocated, reserved


def _case_id(
    config: OperationExperimentConfigV3,
    operation: str,
    profile: BenchmarkProfileV3,
    data_seed: int,
    repeat_index: int,
    scenario_id: str,
) -> str:
    value = f"{config.experiment_id}|{operation}|{profile.identifier}|{data_seed}|{repeat_index}|{scenario_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _scenario_order(
    config: OperationExperimentConfigV3,
    operation: str,
    profile: BenchmarkProfileV3,
    data_seed: int,
    repeat_index: int,
) -> tuple[ScenarioV3, ...]:
    scenarios = list(config.scenarios)
    random.Random(_derived_seed(config.experiment_id, operation, profile.identifier, data_seed, repeat_index)).shuffle(scenarios)
    return tuple(scenarios)


def _capabilities(config: OperationExperimentConfigV3, root: Path) -> dict[str, object]:
    if not config.execution.capabilities_file:
        return {}
    path = Path(config.execution.capabilities_file)
    if not path.is_absolute():
        path = root / path
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentEnvironmentError(f"arquivo de capacidades invalido: {path}") from exc
    if not isinstance(value, dict) or not value.get("passed", False):
        raise ExperimentEnvironmentError("portao de capacidades de hardware nao foi aprovado")
    return value


def _hardware_checks(capabilities: Mapping[str, object]) -> Mapping[str, object]:
    checks = capabilities.get("checks", {})
    return checks if isinstance(checks, Mapping) else {}


def _gpu_temperature_c() -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(completed.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def _enforce_temperature_limit(config: OperationExperimentConfigV3, device: torch.device) -> None:
    if device.type != "cuda":
        return
    temperature = _gpu_temperature_c()
    if temperature is not None and temperature >= config.hardware_requirements.temperature_abort_c:
        raise ExperimentEnvironmentError(
            f"temperatura da GPU atingiu {temperature} C; limite de seguranca "
            f"e {config.hardware_requirements.temperature_abort_c} C"
        )


def _baseline_map(config: OperationExperimentConfigV3) -> dict[str, ScenarioV3]:
    return {
        group: next(
            scenario
            for scenario in config.scenarios
            if scenario.comparison_group == group and scenario.kind == "baseline"
        )
        for group in {scenario.comparison_group for scenario in config.scenarios}
    }


def _run_repeat(
    config: OperationExperimentConfigV3,
    repeat_index: int,
    device: torch.device,
    capabilities: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    timing_records: list[dict[str, object]] = []
    baselines = _baseline_map(config)
    checks = _hardware_checks(capabilities)
    for operation in config.operations:
        for profile in config.profiles:
            for data_seed in config.data_seeds:
                _enforce_temperature_limit(config, device)
                set_deterministic_execution(data_seed)
                base_tensors = build_operation_inputs_v3(operation, profile, data_seed)
                input_hash = _tensor_hash(base_tensors["x"])
                source_weight_hash = _mapping_hash(base_tensors, _weight_names(operation))
                reference_outputs: dict[str, torch.Tensor] = {}
                for group, baseline in baselines.items():
                    module, inputs = _module_from_tensors(operation, base_tensors, profile, baseline, device)
                    with torch.inference_mode():
                        reference_outputs[group] = module(inputs).detach()
                    del module, inputs
                for scenario in _scenario_order(config, operation, profile, data_seed, repeat_index):
                    case_id = _case_id(config, operation, profile, data_seed, repeat_index, scenario.identifier)
                    baseline = baselines[scenario.comparison_group]
                    baseline_case_id = _case_id(
                        config,
                        operation,
                        profile,
                        data_seed,
                        repeat_index,
                        baseline.identifier,
                    )
                    try:
                        module, inputs = _module_from_tensors(operation, base_tensors, profile, scenario, device)
                        callable_module: Callable[[torch.Tensor], torch.Tensor] = module
                        if config.execution.compile:
                            callable_module = torch.compile(module, mode=config.execution.compile_mode)
                            with torch.inference_mode():
                                callable_module(inputs)
                            if device.type == "cuda":
                                torch.cuda.synchronize(device)
                        output, timings, allocated, reserved = _measure(
                            callable_module,
                            inputs,
                            device,
                            config.measurement,
                        )
                        reference = reference_outputs[scenario.comparison_group]
                        candidate_np = output.float().cpu().numpy()
                        reference_np = reference.float().cpu().numpy()
                        if output.shape != reference.shape or not bool(torch.isfinite(output).all()):
                            raise RuntimeError("saida candidata invalida")
                        uses_sparse = bool(
                            scenario.kind == "pruning_2to4" and checks.get("sparse_2to4", False)
                        )
                        uses_int8 = bool(
                            scenario.kind == "quantization_int8_dynamic"
                            and checks.get("int8_dynamic_kernel", False)
                        )
                        quantization_method = {
                            "quantization_int8_fake_per_row": "manual_symmetric_int8_per_row_dequantized",
                            "quantization_int8_weight_only": "torchao_int8_weight_only_per_row_v2",
                            "quantization_int8_dynamic": "torchao_int8_dynamic_activation_weight_per_row_v2",
                        }.get(scenario.kind, "none")
                        record = empty_benchmark_record_v3(
                            experiment_id=config.experiment_id,
                            stage=config.stage,
                            case_id=case_id,
                            baseline_case_id=baseline_case_id,
                            data_seed=data_seed,
                            repeat_index=repeat_index,
                            profile_id=profile.identifier,
                            operation=operation,
                            optimization_scope="operation_weights",
                            scenario=scenario.identifier,
                            comparison_group=scenario.comparison_group,
                            status="complete",
                            batch_size=profile.batch_size,
                            seq_len=profile.seq_len,
                            d_model=profile.d_model,
                            num_heads=profile.num_heads,
                            dtype=scenario.dtype,
                            backend="pytorch_cuda" if device.type == "cuda" else "pytorch_cpu",
                            compile_mode=config.execution.compile_mode if config.execution.compile else "eager",
                            input_distribution=profile.input_distribution,
                            pruning_method=(
                                "magnitude_per_layer"
                                if scenario.kind == "pruning_magnitude"
                                else "magnitude_2to4"
                                if scenario.kind == "pruning_2to4"
                                else "none"
                            ),
                            pruning_sparsity=scenario.pruning_sparsity,
                            quantization_method=quantization_method,
                            weight_storage_dtype=(
                                "int8"
                                if scenario.kind in {"quantization_int8_weight_only", "quantization_int8_dynamic"}
                                else scenario.dtype
                            ),
                            activation_dtype=scenario.dtype,
                            observed_sparsity=_observed_sparsity(module),
                            uses_sparse_kernel=uses_sparse,
                            uses_int8_kernel=uses_int8,
                            latency_ms_mean=float(np.mean(timings)),
                            latency_ms_p50=float(np.percentile(timings, 50)),
                            latency_ms_p95=float(np.percentile(timings, 95)),
                            throughput_tokens_s=profile.batch_size * profile.seq_len / (float(np.mean(timings)) / 1000.0),
                            peak_memory_allocated_bytes=allocated,
                            peak_memory_reserved_bytes=reserved,
                            parameter_storage_bytes=_module_storage_bytes(module),
                            theoretical_flops=operation_flops_v3(operation, profile),
                            mse=mean_squared_error(reference_np, candidate_np),
                            mae=mean_absolute_error(reference_np, candidate_np),
                            r2=r2_score(reference_np, candidate_np),
                            cosine_similarity=cosine_similarity(reference_np, candidate_np),
                            input_sha256=input_hash,
                            weight_sha256=_module_weight_hash(module),
                            output_sha256=_tensor_hash(output),
                            profiler_trace=str(capabilities.get("trace_file", "")),
                        )
                        validation = validate_benchmark_record_v3(record)
                        if not validation.is_valid:
                            raise RuntimeError(f"registro v3 invalido: {validation}")
                        records.append(record)
                        if config.execution.save_raw_timings:
                            for iteration, latency in enumerate(timings, start=1):
                                timing_records.append(
                                    {
                                        "case_id": case_id,
                                        "experiment_id": config.experiment_id,
                                        "repeat_index": repeat_index,
                                        "data_seed": data_seed,
                                        "profile_id": profile.identifier,
                                        "operation": operation,
                                        "scenario": scenario.identifier,
                                        "iteration": iteration,
                                        "latency_ms": latency,
                                    }
                                )
                        del module, callable_module, inputs, output
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    except Exception as exc:
                        record = empty_benchmark_record_v3(
                            experiment_id=config.experiment_id,
                            stage=config.stage,
                            case_id=case_id,
                            baseline_case_id=baseline_case_id,
                            data_seed=data_seed,
                            repeat_index=repeat_index,
                            profile_id=profile.identifier,
                            operation=operation,
                            optimization_scope="operation_weights",
                            scenario=scenario.identifier,
                            comparison_group=scenario.comparison_group,
                            status="unsupported" if config.stage == "synthetic_hardware" else "failed",
                            skip_reason=f"{type(exc).__name__}: {exc}",
                            batch_size=profile.batch_size,
                            seq_len=profile.seq_len,
                            d_model=profile.d_model,
                            num_heads=profile.num_heads,
                            dtype=scenario.dtype,
                            backend="pytorch_cuda" if device.type == "cuda" else "pytorch_cpu",
                            compile_mode=config.execution.compile_mode if config.execution.compile else "eager",
                            input_distribution=profile.input_distribution,
                            pruning_sparsity=scenario.pruning_sparsity,
                            input_sha256=input_hash,
                            weight_sha256=source_weight_hash,
                        )
                        validation = validate_benchmark_record_v3(record)
                        if not validation.is_valid:
                            raise RuntimeError(f"registro de falha v3 invalido: {validation}") from exc
                        records.append(record)
    return records, timing_records


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hash(config: OperationExperimentConfigV3) -> str:
    canonical = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _environment_hash(metadata: Mapping[str, object]) -> str:
    stable = {key: value for key, value in metadata.items() if key not in {"captured_at", "git"}}
    canonical = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def execute_experiment_suite_v3(
    config: OperationExperimentConfigV3,
    output_dir: str | Path,
    *,
    resume: bool = False,
) -> OperationSuiteV3:
    device = resolve_experiment_device(config)  # type: ignore[arg-type]
    if device.type == "cuda" and config.hardware_requirements.compute_capability:
        properties = torch.cuda.get_device_properties(device)
        actual = f"{properties.major}.{properties.minor}"
        if actual != config.hardware_requirements.compute_capability:
            raise ExperimentEnvironmentError(
                f"compute capability esperada {config.hardware_requirements.compute_capability}, encontrada {actual}"
            )
    output_path = Path(output_dir)
    manifest_path = output_path / "manifest.json"
    config_hash = _config_hash(config)
    code_hash = _code_hash()
    current_environment = environment_metadata(device)
    environment_hash = _environment_hash(current_environment)
    completed_repeats: set[int] = set()
    manifest: dict[str, object]
    if output_path.exists() and any(output_path.iterdir()):
        if not resume:
            raise FileExistsError(f"diretorio de saida nao esta vazio: {output_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentEnvironmentError("manifesto ausente para retomada") from exc
        if manifest.get("config_sha256") != config_hash:
            raise ExperimentEnvironmentError("configuracao diverge da execucao retomada")
        if manifest.get("code_sha256") != code_hash:
            raise ExperimentEnvironmentError("codigo diverge da execucao retomada")
        if manifest.get("environment_sha256") != environment_hash:
            raise ExperimentEnvironmentError("ambiente diverge da execucao retomada")
        for item in manifest.get("runs", []):
            if not isinstance(item, dict):
                continue
            csv_path = output_path / str(item.get("csv", ""))
            timing_path = output_path / str(item.get("timings", ""))
            if (
                csv_path.exists()
                and _sha256(csv_path) == item.get("csv_sha256")
                and (not timing_path.name or not timing_path.exists() or _sha256(timing_path) == item.get("timings_sha256"))
            ):
                completed_repeats.add(int(item["repeat_index"]))
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "config.snapshot.json").write_text(
            json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_path / "environment.json").write_text(
            json.dumps(current_environment, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 3,
            "experiment_id": config.experiment_id,
            "status": "running",
            "config": "config.snapshot.json",
            "config_sha256": config_hash,
            "code_sha256": code_hash,
            "environment": "environment.json",
            "environment_sha256": environment_hash,
            "runs": [],
            "resumed_at": [],
        }
    if resume:
        cast_resumed = manifest.setdefault("resumed_at", [])
        if isinstance(cast_resumed, list):
            cast_resumed.append(datetime.now().astimezone().isoformat(timespec="seconds"))
    manifest["status"] = "running"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    capabilities = _capabilities(config, Path.cwd())
    try:
        for repeat_index in range(1, config.measurement.repetitions + 1):
            if repeat_index in completed_repeats:
                continue
            records, timings = _run_repeat(config, repeat_index, device, capabilities)
            run_path = output_path / f"run_{repeat_index:02d}.csv"
            timing_path = output_path / f"run_{repeat_index:02d}.timings.csv"
            _write_csv(run_path, records, CSV_COLUMNS_V3)
            _write_csv(timing_path, timings, TIMING_COLUMNS)
            runs = manifest.setdefault("runs", [])
            if not isinstance(runs, list):
                raise RuntimeError("runs invalido no manifesto")
            runs.append(
                {
                    "repeat_index": repeat_index,
                    "csv": run_path.name,
                    "csv_sha256": _sha256(run_path),
                    "timings": timing_path.name,
                    "timings_sha256": _sha256(timing_path),
                    "records": len(records),
                    "timing_samples": len(timings),
                }
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        all_timings: list[dict[str, object]] = []
        for item in sorted(manifest["runs"], key=lambda value: value["repeat_index"]):  # type: ignore[index]
            with (output_path / item["timings"]).open(encoding="utf-8", newline="") as handle:
                all_timings.extend(csv.DictReader(handle))
        timings_path = output_path / "timings.csv"
        _write_csv(timings_path, all_timings, TIMING_COLUMNS)
        manifest["timings"] = timings_path.name
        manifest["timings_sha256"] = _sha256(timings_path)
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    run_paths = tuple(output_path / item["csv"] for item in manifest["runs"])  # type: ignore[index]
    return OperationSuiteV3(
        output_dir=output_path,
        run_paths=run_paths,
        timings_path=output_path / "timings.csv",
        manifest_path=manifest_path,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Executa benchmarks de operacoes no schema v3.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_experiment_config_v3(args.config)
        suite = execute_experiment_suite_v3(config, args.output_dir, resume=args.resume)
    except (ExperimentConfigurationError, ExperimentEnvironmentError, FileExistsError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"falha experimental v3: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"experimento v3 concluido: {config.experiment_id}")
    print(suite.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
