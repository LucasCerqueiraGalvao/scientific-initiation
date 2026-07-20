from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
from benchmark import measure_callable, resolve_device, set_seed
from metrics import (
    compute_quality_metrics,
    estimate_dense_projection_flops,
    estimate_self_attention_flops,
    estimate_tensor_memory_bytes,
)
from ops import dense_projection, single_head_self_attention
from optimizations import fake_quantize_int8, prune_by_magnitude


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa os benchmarks do estudo com Transformers.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experimento_padrao.json",
        help="Caminho para o arquivo JSON de configuração.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def randn(shape: tuple[int, ...], seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(*shape, generator=generator, dtype=torch.float32).to(device)


def build_inputs(operation: str, batch_size: int, sequence_length: int, dimension: int, seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    if operation == "dense_projection":
        return {
            "x": randn((batch_size, sequence_length, dimension), seed, device),
            "weight": randn((dimension, dimension), seed + 1, device),
            "bias": randn((dimension,), seed + 2, device),
        }

    if operation == "self_attention":
        return {
            "x": randn((batch_size, sequence_length, dimension), seed, device),
            "w_q": randn((dimension, dimension), seed + 1, device),
            "w_k": randn((dimension, dimension), seed + 2, device),
            "w_v": randn((dimension, dimension), seed + 3, device),
            "w_o": randn((dimension, dimension), seed + 4, device),
        }

    raise ValueError(f"Unsupported operation: {operation}")


def apply_scenario(operation: str, scenario: str, tensors: dict[str, torch.Tensor], pruning_sparsity: float) -> dict[str, torch.Tensor]:
    processed = {name: tensor.clone() for name, tensor in tensors.items()}

    if scenario == "baseline":
        return processed

    if scenario == "pruning_50":
        if operation == "dense_projection":
            processed["weight"] = prune_by_magnitude(processed["weight"], pruning_sparsity)
            return processed

        for name in ("w_q", "w_k", "w_v", "w_o"):
            processed[name] = prune_by_magnitude(processed[name], pruning_sparsity)
        return processed

    if scenario == "quantization_int8":
        for name, tensor in processed.items():
            processed[name] = fake_quantize_int8(tensor)
        return processed

    raise ValueError(f"Unsupported scenario: {scenario}")


def run_operation(operation: str, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    if operation == "dense_projection":
        return dense_projection(tensors["x"], tensors["weight"], tensors["bias"])

    if operation == "self_attention":
        return single_head_self_attention(
            tensors["x"],
            tensors["w_q"],
            tensors["w_k"],
            tensors["w_v"],
            tensors["w_o"],
        )

    raise ValueError(f"Unsupported operation: {operation}")


def estimate_flops(operation: str, batch_size: int, sequence_length: int, dimension: int) -> int:
    if operation == "dense_projection":
        return estimate_dense_projection_flops(batch_size, sequence_length, dimension)
    if operation == "self_attention":
        return estimate_self_attention_flops(batch_size, sequence_length, dimension)
    raise ValueError(f"Unsupported operation: {operation}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    set_seed(config["seed"])
    device = resolve_device(config["device"])

    rows: list[dict] = []
    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for repetition in range(config["repetitions"]):
        for operation in config["operations"]:
            for sequence_length in config["sequence_lengths"]:
                for dimension in config["dimensions"]:
                    base_seed = config["seed"] + (repetition * 10_000) + (sequence_length * 100) + dimension
                    baseline_tensors = build_inputs(
                        operation=operation,
                        batch_size=config["batch_size"],
                        sequence_length=sequence_length,
                        dimension=dimension,
                        seed=base_seed,
                        device=device,
                    )

                    with torch.inference_mode():
                        reference_output = run_operation(operation, baseline_tensors)

                    for scenario in config["scenarios"]:
                        scenario_tensors = apply_scenario(
                            operation=operation,
                            scenario=scenario,
                            tensors=baseline_tensors,
                            pruning_sparsity=config["pruning_sparsity"],
                        )

                        def execute() -> torch.Tensor:
                            return run_operation(operation, scenario_tensors)

                        timing_metrics, candidate_output = measure_callable(
                            fn=execute,
                            device=device,
                            warmup_iterations=config["warmup_iterations"],
                            measure_iterations=config["measure_iterations"],
                        )
                        quality_metrics = compute_quality_metrics(reference_output, candidate_output)

                        row = {
                            "experiment_id": experiment_id,
                            "device_requested": config["device"],
                            "device_used": device.type,
                            "repetition": repetition + 1,
                            "operation": operation,
                            "scenario": scenario,
                            "batch_size": config["batch_size"],
                            "sequence_length": sequence_length,
                            "dimension": dimension,
                            "warmup_iterations": config["warmup_iterations"],
                            "measure_iterations": config["measure_iterations"],
                            "theoretical_input_memory_bytes": estimate_tensor_memory_bytes(*scenario_tensors.values()),
                            "estimated_flops": estimate_flops(
                                operation=operation,
                                batch_size=config["batch_size"],
                                sequence_length=sequence_length,
                                dimension=dimension,
                            ),
                        }
                        row.update(timing_metrics)
                        row.update(quality_metrics)
                        rows.append(row)

    results_dir = ROOT / "resultados"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"benchmark_results_{experiment_id}.csv"

    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(output_path, index=False)

    print(f"Results written to: {output_path}")
    print(dataframe.head().to_string(index=False))


if __name__ == "__main__":
    main()
