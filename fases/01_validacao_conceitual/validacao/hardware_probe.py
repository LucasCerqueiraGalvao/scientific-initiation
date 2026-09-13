from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile


def _temperature_c() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def _profile_operators(callable_module, inputs: torch.Tensor) -> list[str]:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
        callable_module(inputs)
        torch.cuda.synchronize()
    return sorted({item.key for item in captured.key_averages()})


def detect_kernel_flags(operators: Sequence[str]) -> dict[str, bool]:
    text = " ".join(operators).casefold()
    return {
        "int8": "int_mm" in text or "int8" in text,
        "sparse_2to4": "cslt" in text or "cusparselt" in text or "sparse" in text,
    }


def _linear(dtype: torch.dtype) -> tuple[nn.Module, torch.Tensor]:
    model = nn.Linear(2048, 2048, bias=False, device="cuda", dtype=dtype).eval()
    inputs = torch.randn(16, 2048, device="cuda", dtype=dtype)
    return model, inputs


def run_hardware_probe(
    *,
    expected_gpu: str,
    expected_compute_capability: str,
    compile_mode: str = "reduce-overhead",
) -> tuple[dict[str, object], dict[str, list[str]]]:
    checks: dict[str, object] = {}
    operators: dict[str, list[str]] = {}
    checks["cuda_available"] = torch.cuda.is_available()
    if not checks["cuda_available"]:
        return {"passed": False, "checks": checks}, operators

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    properties = torch.cuda.get_device_properties(device)
    compute_capability = f"{properties.major}.{properties.minor}"
    checks["gpu_name"] = gpu_name
    checks["gpu_name_matches"] = expected_gpu.casefold() in gpu_name.casefold()
    checks["compute_capability"] = compute_capability
    checks["compute_capability_matches"] = compute_capability == expected_compute_capability
    checks["temperature_c"] = _temperature_c()

    try:
        import triton

        checks["triton"] = True
        checks["triton_version"] = getattr(triton, "__version__", "")
    except ImportError:
        checks["triton"] = False

    try:
        from torchao.quantization import (
            Int8DynamicActivationInt8WeightConfig,
            Int8WeightOnlyConfig,
            quantize_,
        )
        from torchao.quantization.granularity import PerRow

        weight_only, inputs = _linear(torch.bfloat16)
        reference = weight_only(inputs)
        quantize_(weight_only, Int8WeightOnlyConfig(version=2, granularity=PerRow()), device=device)
        compiled_weight_only = torch.compile(weight_only, mode=compile_mode)
        for _ in range(3):
            weight_output = compiled_weight_only(inputs)
        torch.cuda.synchronize()
        operators["int8_weight_only"] = _profile_operators(compiled_weight_only, inputs)
        checks["int8_weight_only"] = bool(torch.isfinite(weight_output).all())
        checks["int8_weight_storage"] = bool(
            all(
                getattr(child.weight, "qdata", torch.empty((), dtype=torch.float32)).dtype == torch.int8
                for child in weight_only.modules()
                if isinstance(child, nn.Linear)
            )
        )
        checks["int8_weight_only_max_abs_error"] = float((reference - weight_output).abs().max())

        dynamic, inputs = _linear(torch.bfloat16)
        quantize_(
            dynamic,
            Int8DynamicActivationInt8WeightConfig(version=2, granularity=PerRow()),
            device=device,
        )
        compiled_dynamic = torch.compile(dynamic, mode=compile_mode)
        for _ in range(3):
            dynamic_output = compiled_dynamic(inputs)
        torch.cuda.synchronize()
        operators["int8_dynamic"] = _profile_operators(compiled_dynamic, inputs)
        dynamic_flags = detect_kernel_flags(operators["int8_dynamic"])
        checks["int8_dynamic_forward"] = bool(torch.isfinite(dynamic_output).all())
        checks["int8_dynamic_kernel"] = dynamic_flags["int8"]
    except Exception as exc:
        checks["int8_error"] = f"{type(exc).__name__}: {exc}"
        checks.setdefault("int8_weight_only", False)
        checks.setdefault("int8_weight_storage", False)
        checks.setdefault("int8_dynamic_forward", False)
        checks.setdefault("int8_dynamic_kernel", False)

    try:
        from torch.sparse import to_sparse_semi_structured

        dense_weight = torch.randn(128, 128, device=device, dtype=torch.float16)
        grouped = dense_weight.reshape(128, 32, 4)
        keep = grouped.abs().topk(2, dim=-1).indices
        mask = torch.zeros_like(grouped, dtype=torch.bool).scatter_(-1, keep, True)
        pruned_weight = (grouped * mask).reshape_as(dense_weight)
        sparse_weight = to_sparse_semi_structured(pruned_weight)
        inputs = torch.randn(16, 128, device=device, dtype=torch.float16)

        def sparse_linear(value: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.linear(value, sparse_weight)

        compiled_sparse = torch.compile(sparse_linear, mode=compile_mode)
        for _ in range(3):
            sparse_output = compiled_sparse(inputs)
        torch.cuda.synchronize()
        operators["sparse_2to4"] = _profile_operators(compiled_sparse, inputs)
        sparse_flags = detect_kernel_flags(operators["sparse_2to4"])
        checks["sparse_2to4_forward"] = bool(torch.isfinite(sparse_output).all())
        checks["sparse_2to4"] = sparse_flags["sparse_2to4"]
    except Exception as exc:
        checks["sparse_2to4_error"] = f"{type(exc).__name__}: {exc}"
        checks["sparse_2to4_forward"] = False
        checks["sparse_2to4"] = False

    common_required = (
        "cuda_available",
        "gpu_name_matches",
        "compute_capability_matches",
        "triton",
    )
    accelerator_required = (
        "int8_weight_only",
        "int8_weight_storage",
        "int8_dynamic_forward",
        "int8_dynamic_kernel",
        "sparse_2to4_forward",
        "sparse_2to4",
    )
    return (
        {
            "schema_version": 1,
            "passed": all(bool(checks.get(name, False)) for name in common_required),
            "all_accelerators_passed": all(
                bool(checks.get(name, False)) for name in accelerator_required
            ),
            "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_cuda_build": torch.version.cuda or "",
            "required_checks": list(common_required),
            "accelerator_checks": list(accelerator_required),
            "checks": checks,
            "trace_file": "hardware_operators.json",
        },
        operators,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Valida kernels antes do benchmark fisico.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-gpu", default="NVIDIA GeForce RTX 4070 Ti SUPER")
    parser.add_argument("--expected-compute-capability", default="8.9")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result, operators = run_hardware_probe(
        expected_gpu=args.expected_gpu,
        expected_compute_capability=args.expected_compute_capability,
        compile_mode=args.compile_mode,
    )
    operators_path = output_path.with_name("hardware_operators.json")
    operators_path.write_text(json.dumps(operators, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    result["trace_file"] = str(operators_path)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_path)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
