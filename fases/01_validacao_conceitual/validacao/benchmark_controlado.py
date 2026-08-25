from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.utils.prune as prune

from validacao.flops import transformer_block_flops
from validacao.metrics import cosine_similarity, mean_absolute_error, mean_squared_error, r2_score
from validacao.protocolo import CSV_COLUMNS, empty_benchmark_record, validate_benchmark_record
from validacao.quantization import symmetric_int8_quantize, tensor_storage_bytes
from validacao.transformer import SimplifiedTransformerBlock


@dataclass(frozen=True)
class HardwareEvidence:
    uses_low_precision_storage: bool = False
    uses_low_precision_kernel: bool = False
    uses_sparse_storage: bool = False
    uses_sparse_kernel: bool = False


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    matrix_entry: str
    initial_validity_level: str
    quantization_method: str
    pruning_sparsity: float
    hardware_claim_kind: str


@dataclass(frozen=True)
class ControlledBenchmarkConfig:
    batch_sizes: tuple[int, ...] = (1,)
    seq_lens: tuple[int, ...] = (4, 8)
    d_models: tuple[int, ...] = (8,)
    num_heads: tuple[int, ...] = (2,)
    scenarios: tuple[str, ...] = ("baseline", "pruning_magnitude", "quantization_int8")
    seed: int = 2026
    warmup: int = 5
    repetitions: int = 20
    device: str = "auto"
    pruning_sparsity: float = 0.5
    ffn_multiplier: int = 2


@dataclass(frozen=True)
class BenchmarkRun:
    records: tuple[dict[str, object], ...]
    metadata: dict[str, object]


SCENARIO_REGISTRY = {
    "baseline": ScenarioDefinition(
        name="baseline",
        matrix_entry="validacao.transformer.SimplifiedTransformerBlock",
        initial_validity_level="algoritmico",
        quantization_method="none",
        pruning_sparsity=0.0,
        hardware_claim_kind="none",
    ),
    "pruning_magnitude": ScenarioDefinition(
        name="pruning_magnitude",
        matrix_entry="torch.nn.utils.prune",
        initial_validity_level="conceitual_algoritmico",
        quantization_method="none",
        pruning_sparsity=0.5,
        hardware_claim_kind="pruning",
    ),
    "quantization_int8": ScenarioDefinition(
        name="quantization_int8",
        matrix_entry="validacao.quantization.symmetric_int8_quantize",
        initial_validity_level="numerico",
        quantization_method="manual_symmetric_int8_dequantized",
        pruning_sparsity=0.0,
        hardware_claim_kind="quantization",
    ),
    "torchao_int8": ScenarioDefinition(
        name="torchao_int8",
        matrix_entry="torchao.quantization.Int8WeightOnlyConfig(version=2)",
        initial_validity_level="numerico_algoritmico",
        quantization_method="torchao_int8_weight_only",
        pruning_sparsity=0.0,
        hardware_claim_kind="quantization",
    ),
    "pruning_plus_quantization": ScenarioDefinition(
        name="pruning_plus_quantization",
        matrix_entry="torch.nn.utils.prune + validacao.quantization.symmetric_int8_quantize",
        initial_validity_level="numerico_algoritmico",
        quantization_method="manual_symmetric_int8_dequantized",
        pruning_sparsity=0.5,
        hardware_claim_kind="combined",
    ),
}


class ScenarioValidationError(ValueError):
    pass


def set_reproducible_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitado, mas nao esta disponivel")
    return device


def validate_scenarios_registered(scenarios: Iterable[str]) -> None:
    unknown = sorted(set(scenarios) - set(SCENARIO_REGISTRY))
    if unknown:
        raise ScenarioValidationError(f"cenario sem matriz de validacao: {', '.join(unknown)}")


def promoted_validity_level(scenario: str, evidence: HardwareEvidence) -> str:
    definition = SCENARIO_REGISTRY[scenario]
    kind = definition.hardware_claim_kind
    if kind == "none":
        return definition.initial_validity_level
    if kind == "pruning" and evidence.uses_sparse_storage and evidence.uses_sparse_kernel:
        return "hardware"
    if kind == "quantization" and evidence.uses_low_precision_storage and evidence.uses_low_precision_kernel:
        return "hardware"
    if (
        kind == "combined"
        and evidence.uses_sparse_storage
        and evidence.uses_sparse_kernel
        and evidence.uses_low_precision_storage
        and evidence.uses_low_precision_kernel
    ):
        return "hardware"
    return definition.initial_validity_level


def environment_metadata(device: torch.device, seed: int) -> dict[str, object]:
    gpu_name = ""
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "",
        "gpu_name": gpu_name,
        "device": str(device),
        "seed": seed,
        "memory_measurement_kind": (
            "torch_cuda_max_memory_allocated" if device.type == "cuda" else "model_storage_estimate"
        ),
    }


def _iter_grid(config: ControlledBenchmarkConfig) -> Iterable[tuple[int, int, int, int]]:
    for batch_size in config.batch_sizes:
        for seq_len in config.seq_lens:
            for d_model in config.d_models:
                for num_heads in config.num_heads:
                    if d_model % num_heads != 0:
                        raise ValueError("d_model deve ser divisivel por num_heads")
                    yield batch_size, seq_len, d_model, num_heads


def _linear_modules(model: torch.nn.Module) -> Iterable[torch.nn.Linear]:
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            yield module


def _apply_pruning(model: torch.nn.Module, amount: float) -> HardwareEvidence:
    for module in _linear_modules(model):
        prune.l1_unstructured(module, name="weight", amount=amount)
    return HardwareEvidence(uses_sparse_storage=False, uses_sparse_kernel=False)


def _apply_manual_quantization(model: torch.nn.Module) -> HardwareEvidence:
    with torch.no_grad():
        for module in _linear_modules(model):
            target_weight = module.weight_orig if hasattr(module, "weight_orig") else module.weight
            quantized, _, dequantized = symmetric_int8_quantize(target_weight.detach())
            target_weight.copy_(dequantized.to(target_weight.device, dtype=target_weight.dtype))
            module._scientific_validation_quantized_dtype = str(quantized.dtype)
    # O int8 e usado para produzir a dequantizacao, mas os pesos que permanecem
    # no modulo e que chegam ao kernel continuam float32. O caminho valida erro
    # numerico; nao demonstra armazenamento ou execucao de baixa precisao.
    return HardwareEvidence(uses_low_precision_storage=False, uses_low_precision_kernel=False)


def _apply_torchao_quantization(model: torch.nn.Module) -> HardwareEvidence:
    try:
        from torchao.quantization import Int8WeightOnlyConfig, quantize_
    except ImportError as exc:
        raise ScenarioValidationError("torchao nao esta instalado") from exc

    quantize_(model, Int8WeightOnlyConfig(version=2))
    return HardwareEvidence(uses_low_precision_storage=True, uses_low_precision_kernel=False)


def apply_scenario(
    model: torch.nn.Module,
    scenario: str,
    *,
    pruning_sparsity: float,
) -> HardwareEvidence:
    validate_scenarios_registered((scenario,))
    if scenario == "baseline":
        return HardwareEvidence()
    if scenario == "pruning_magnitude":
        return _apply_pruning(model, pruning_sparsity)
    if scenario == "quantization_int8":
        return _apply_manual_quantization(model)
    if scenario == "torchao_int8":
        return _apply_torchao_quantization(model)
    if scenario == "pruning_plus_quantization":
        _apply_pruning(model, pruning_sparsity)
        _apply_manual_quantization(model)
        return HardwareEvidence(uses_low_precision_storage=True, uses_sparse_storage=False, uses_sparse_kernel=False)
    raise ScenarioValidationError(f"cenario desconhecido: {scenario}")


def _tensor_storage_estimate(tensor: torch.Tensor) -> int:
    qdata = getattr(tensor, "qdata", None)
    scale = getattr(tensor, "scale", None)
    if isinstance(qdata, torch.Tensor):
        total = tensor_storage_bytes(qdata)
        if isinstance(scale, torch.Tensor):
            total += tensor_storage_bytes(scale)
        return total

    tensor_impl = getattr(tensor, "tensor_impl", None)
    internal_data = getattr(tensor_impl, "data", None)
    if isinstance(internal_data, torch.Tensor):
        return tensor_storage_bytes(internal_data)
    return tensor_storage_bytes(tensor)


def estimate_model_storage_bytes(model: torch.nn.Module) -> int:
    total = 0
    for parameter in model.parameters():
        total += _tensor_storage_estimate(parameter)
    for buffer in model.buffers():
        total += _tensor_storage_estimate(buffer)
    return int(total)


def _synchronize_if_needed(device: torch.device) -> bool:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return True
    return False


def _measure_latency(
    fn: Callable[[], torch.Tensor],
    *,
    device: torch.device,
    warmup: int,
    repetitions: int,
) -> tuple[tuple[float, ...], int, bool]:
    if repetitions <= 0:
        raise ValueError("repetitions deve ser positivo")
    if warmup < 0:
        raise ValueError("warmup nao pode ser negativo")

    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        used_cuda_sync = _synchronize_if_needed(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        latencies: list[float] = []
        for _ in range(repetitions):
            _synchronize_if_needed(device)
            start = time.perf_counter()
            result = fn()
            _synchronize_if_needed(device)
            latencies.append((time.perf_counter() - start) * 1000.0)

        if device.type == "cuda":
            max_memory_bytes = int(torch.cuda.max_memory_allocated(device))
        else:
            max_memory_bytes = int(result.numel() * result.element_size())

    return tuple(latencies), max_memory_bytes, used_cuda_sync


def _percentile(values: tuple[float, ...], percentile: float) -> float:
    if not values:
        raise ValueError("valores vazios")
    sorted_values = sorted(values)
    index = round((len(sorted_values) - 1) * percentile)
    return float(sorted_values[index])


def run_controlled_benchmark(config: ControlledBenchmarkConfig = ControlledBenchmarkConfig()) -> BenchmarkRun:
    validate_scenarios_registered(config.scenarios)
    set_reproducible_seed(config.seed)
    device = resolve_device(config.device)
    env = environment_metadata(device, config.seed)

    records: list[dict[str, object]] = []
    used_cuda_synchronization = False

    for batch_size, seq_len, d_model, num_heads in _iter_grid(config):
        ffn_hidden_dim = d_model * config.ffn_multiplier
        set_reproducible_seed(config.seed)
        baseline_model = SimplifiedTransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            ffn_hidden_dim=ffn_hidden_dim,
        ).eval().to(device)
        inputs = torch.randn(batch_size, seq_len, d_model, device=device)
        with torch.inference_mode():
            baseline_output = baseline_model(inputs).detach()

        for scenario in config.scenarios:
            scenario_definition = SCENARIO_REGISTRY[scenario]
            model = copy.deepcopy(baseline_model).eval().to(device)
            evidence = apply_scenario(model, scenario, pruning_sparsity=config.pruning_sparsity)
            validity_level = promoted_validity_level(scenario, evidence)

            with torch.inference_mode():
                candidate_output = model(inputs).detach()

            latencies, measured_memory_bytes, sync_used = _measure_latency(
                lambda model=model, inputs=inputs: model(inputs),
                device=device,
                warmup=config.warmup,
                repetitions=config.repetitions,
            )
            used_cuda_synchronization = used_cuda_synchronization or sync_used

            reference = baseline_output.float().cpu().numpy()
            candidate = candidate_output.float().cpu().numpy()
            latency_mean = float(statistics.fmean(latencies))
            tokens = batch_size * seq_len
            model_storage = estimate_model_storage_bytes(model)
            max_memory_bytes = max(measured_memory_bytes, model_storage)
            flops = transformer_block_flops(
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=d_model,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
            ).total

            record = empty_benchmark_record(
                torch_version=env["torch_version"],
                cuda_version=env["cuda_version"],
                gpu_name=env["gpu_name"],
                device=env["device"],
                model_kind="simplified_transformer_block",
                scenario=scenario,
                batch_size=batch_size,
                seq_len=seq_len,
                d_model=d_model,
                num_heads=num_heads,
                dtype=str(inputs.dtype).replace("torch.", ""),
                pruning_sparsity=config.pruning_sparsity if "pruning" in scenario else scenario_definition.pruning_sparsity,
                quantization_method=scenario_definition.quantization_method,
                validity_level=validity_level,
                latency_ms_mean=latency_mean,
                latency_ms_p50=_percentile(latencies, 0.50),
                latency_ms_p95=_percentile(latencies, 0.95),
                throughput_tokens_s=float(tokens / (latency_mean / 1000.0)),
                max_memory_bytes=max_memory_bytes,
                theoretical_flops=flops,
                mse=mean_squared_error(reference, candidate),
                mae=mean_absolute_error(reference, candidate),
                r2=r2_score(reference, candidate),
                cosine_similarity=cosine_similarity(reference, candidate),
            )
            validation = validate_benchmark_record(record)
            if not validation.is_valid:
                raise ValueError(f"registro de benchmark invalido: {validation}")
            records.append(record)

    metadata = {
        **env,
        "warmup": config.warmup,
        "repetitions": config.repetitions,
        "scenarios": list(config.scenarios),
        "used_cuda_synchronization": used_cuda_synchronization,
    }
    return BenchmarkRun(records=tuple(records), metadata=metadata)


def write_benchmark_outputs(run: BenchmarkRun, csv_path: str | Path) -> None:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for record in run.records:
        validation = validate_benchmark_record(record)
        if not validation.is_valid:
            raise ValueError(f"registro de benchmark invalido: {validation}")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(run.records)

    metadata_path = path.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(run.metadata, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Executa benchmark controlado da fase de validacao.")
    parser.add_argument("--output", default="resultados/benchmark_controlado.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--scenarios",
        default="baseline,pruning_magnitude,quantization_int8",
        help="Lista separada por virgulas.",
    )
    args = parser.parse_args(argv)

    config = ControlledBenchmarkConfig(
        device=args.device,
        repetitions=args.repetitions,
        warmup=args.warmup,
        seed=args.seed,
        scenarios=tuple(part.strip() for part in args.scenarios.split(",") if part.strip()),
    )
    run = run_controlled_benchmark(config)
    write_benchmark_outputs(run, args.output)
    print(f"Registros salvos: {len(run.records)} em {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
