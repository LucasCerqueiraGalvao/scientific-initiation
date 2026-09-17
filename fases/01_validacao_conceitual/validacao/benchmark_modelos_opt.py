from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from validacao.benchmark_operacoes import (
    ExperimentConfigurationError,
    ExperimentEnvironmentError,
    environment_metadata,
)
from validacao.benchmark_operacoes_v3 import (
    _module_storage_bytes,
    exact_magnitude_prune,
    magnitude_prune_2to4,
)
from validacao.hardware_probe import detect_kernel_flags
from validacao.protocolo import (
    CSV_COLUMNS_V3,
    empty_benchmark_record_v3,
    validate_benchmark_record_v3,
)
from validacao.quantization import symmetric_int8_quantize_per_row


MODEL_SCENARIO_KINDS = (
    "baseline",
    "pruning_magnitude",
    "quantization_int8_fake_per_row",
    "quantization_int8_weight_only",
    "quantization_int8_dynamic",
    "pruning_2to4",
)
COMMIT_HASH_LENGTH = 40
OPTIMIZATION_SCOPES = ("attention_only", "transformer_blocks")
QUALITY_COLUMNS = [
    "experiment_id", "model_id", "model_revision", "scenario", "comparison_group",
    "optimization_scope", "dtype", "status", "skip_reason", "parameter_count",
    "parameter_storage_bytes", "target_linear_count", "pruning_sparsity",
    "observed_sparsity", "mean_loss", "perplexity", "perplexity_increase_ratio",
    "logits_mse", "logits_cosine", "logits_kl_divergence", "top1_agreement",
    "generated_token_agreement", "exact_generation_rate", "quality_acceptable",
    "weight_sha256", "evaluation_subset_sha256",
]
WINDOW_COLUMNS = [
    "model_id", "scenario", "window_index", "start_token", "token_count", "loss", "perplexity"
]
PROMPT_COLUMNS = [
    "model_id", "scenario", "prompt_id", "input_tokens", "generated_tokens",
    "logits_mse", "logits_cosine", "logits_kl_divergence", "top1_agreement",
    "generated_token_agreement", "exact_generation",
]
TIMING_COLUMNS = [
    "case_id", "model_id", "scenario", "operation", "repeat_index", "iteration", "latency_ms"
]


@dataclass(frozen=True)
class ModelSpec:
    identifier: str
    revision: str


@dataclass(frozen=True)
class DatasetSpec:
    identifier: str
    configuration: str
    revision: str
    split: str


@dataclass(frozen=True)
class ModelScenario:
    identifier: str
    kind: str
    comparison_group: str
    dtype: str
    optimization_scope: str
    pruning_sparsity: float
    measure_performance: bool


@dataclass(frozen=True)
class QualitySpec:
    windows: int
    window_tokens: int
    prompts_file: str
    generation_tokens: int
    quantization_perplexity_limit: float
    pruning_perplexity_limit: float


@dataclass(frozen=True)
class PerformanceSpec:
    prefill_lengths: tuple[int, ...]
    decode_prompt_lengths: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    decode_tokens: int
    warmup_iterations: int
    measure_iterations: int
    repetitions: int
    use_kv_cache: bool


@dataclass(frozen=True)
class ModelExecutionSpec:
    compile: bool
    compile_mode: str
    local_files_only: bool
    capabilities_file: str


@dataclass(frozen=True)
class ModelExperimentConfig:
    schema_version: int
    experiment_id: str
    device: str
    required_gpu_name: str
    models: tuple[ModelSpec, ...]
    dataset: DatasetSpec
    scenarios: tuple[ModelScenario, ...]
    quality: QualitySpec
    performance: PerformanceSpec
    execution: ModelExecutionSpec

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["models"] = [asdict(model) for model in self.models]
        result["scenarios"] = [asdict(scenario) for scenario in self.scenarios]
        result["performance"]["prefill_lengths"] = list(self.performance.prefill_lengths)  # type: ignore[index]
        result["performance"]["decode_prompt_lengths"] = list(self.performance.decode_prompt_lengths)  # type: ignore[index]
        result["performance"]["batch_sizes"] = list(self.performance.batch_sizes)  # type: ignore[index]
        return result


@dataclass(frozen=True)
class EvaluationSubset:
    windows: tuple[tuple[int, tuple[int, ...]], ...]
    sha256: str
    performance_tokens: tuple[int, ...] = ()


@dataclass(frozen=True)
class BaselinePrompt:
    logits: torch.Tensor
    generated_ids: torch.Tensor


def _positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ExperimentConfigurationError(f"{name} deve ser inteiro positivo")
    return value


def _int_tuple(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ExperimentConfigurationError(f"{name} deve ser lista nao vazia")
    result = tuple(_positive_int(name, item) for item in value)
    if len(set(result)) != len(result):
        raise ExperimentConfigurationError(f"{name} nao pode conter duplicatas")
    return result


def _commit_hash(name: str, value: object) -> str:
    revision = str(value)
    if len(revision) != COMMIT_HASH_LENGTH or any(character not in "0123456789abcdef" for character in revision):
        raise ExperimentConfigurationError(f"{name} deve ser um commit SHA-1 completo")
    return revision


def model_config_from_mapping(data: Mapping[str, object]) -> ModelExperimentConfig:
    expected = {
        "schema_version", "experiment_id", "device", "required_gpu_name", "models",
        "dataset", "scenarios", "quality", "performance", "execution",
    }
    if set(data) != expected or data.get("schema_version") != 1:
        raise ExperimentConfigurationError("configuracao de modelos possui schema invalido")
    raw_models = data["models"]
    if not isinstance(raw_models, list) or not raw_models:
        raise ExperimentConfigurationError("models deve ser lista nao vazia")
    models: list[ModelSpec] = []
    for raw in raw_models:
        if not isinstance(raw, Mapping) or set(raw) != {"identifier", "revision"}:
            raise ExperimentConfigurationError("model spec invalida")
        models.append(ModelSpec(str(raw["identifier"]), _commit_hash("model revision", raw["revision"])))

    raw_dataset = data["dataset"]
    raw_quality = data["quality"]
    raw_performance = data["performance"]
    raw_execution = data["execution"]
    if not all(isinstance(value, Mapping) for value in (raw_dataset, raw_quality, raw_performance, raw_execution)):
        raise ExperimentConfigurationError("bloco de configuracao de modelos invalido")
    if set(raw_dataset) != {"identifier", "configuration", "revision", "split"}:
        raise ExperimentConfigurationError("dataset spec invalida")
    if set(raw_quality) != {
        "windows", "window_tokens", "prompts_file", "generation_tokens",
        "quantization_perplexity_limit", "pruning_perplexity_limit",
    }:
        raise ExperimentConfigurationError("quality spec invalida")
    if set(raw_performance) != {
        "prefill_lengths", "decode_prompt_lengths", "batch_sizes", "decode_tokens",
        "warmup_iterations", "measure_iterations", "repetitions", "use_kv_cache",
    }:
        raise ExperimentConfigurationError("performance spec invalida")
    if set(raw_execution) != {"compile", "compile_mode", "local_files_only", "capabilities_file"}:
        raise ExperimentConfigurationError("execution spec invalida")

    raw_scenarios = data["scenarios"]
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ExperimentConfigurationError("scenarios deve ser lista nao vazia")
    scenarios: list[ModelScenario] = []
    ids: set[str] = set()
    for raw in raw_scenarios:
        required = {
            "identifier", "kind", "comparison_group", "dtype", "optimization_scope",
            "pruning_sparsity", "measure_performance",
        }
        if not isinstance(raw, Mapping) or set(raw) != required:
            raise ExperimentConfigurationError("model scenario invalido")
        scenario = ModelScenario(
            identifier=str(raw["identifier"]),
            kind=str(raw["kind"]),
            comparison_group=str(raw["comparison_group"]),
            dtype=str(raw["dtype"]),
            optimization_scope=str(raw["optimization_scope"]),
            pruning_sparsity=float(raw["pruning_sparsity"]),
            measure_performance=bool(raw["measure_performance"]),
        )
        if scenario.identifier in ids or not scenario.identifier:
            raise ExperimentConfigurationError("scenario id vazio ou duplicado")
        ids.add(scenario.identifier)
        if scenario.kind not in MODEL_SCENARIO_KINDS:
            raise ExperimentConfigurationError(f"kind de modelo desconhecido: {scenario.kind}")
        if scenario.optimization_scope not in (*OPTIMIZATION_SCOPES, "none"):
            raise ExperimentConfigurationError("optimization_scope desconhecido")
        if scenario.dtype not in {"bfloat16", "float16"}:
            raise ExperimentConfigurationError("dtype de modelo deve ser bfloat16 ou float16")
        if not 0 <= scenario.pruning_sparsity < 1:
            raise ExperimentConfigurationError("pruning_sparsity invalida")
        scenarios.append(scenario)
    for group in {scenario.comparison_group for scenario in scenarios}:
        baselines = [scenario for scenario in scenarios if scenario.comparison_group == group and scenario.kind == "baseline"]
        if len(baselines) != 1:
            raise ExperimentConfigurationError(f"grupo {group} exige exatamente um baseline")

    device = str(data["device"])
    if device != "cuda":
        raise ExperimentConfigurationError("benchmark OPT final exige CUDA")
    return ModelExperimentConfig(
        schema_version=1,
        experiment_id=str(data["experiment_id"]),
        device=device,
        required_gpu_name=str(data["required_gpu_name"]),
        models=tuple(models),
        dataset=DatasetSpec(
            str(raw_dataset["identifier"]),
            str(raw_dataset["configuration"]),
            _commit_hash("dataset revision", raw_dataset["revision"]),
            str(raw_dataset["split"]),
        ),
        scenarios=tuple(scenarios),
        quality=QualitySpec(
            windows=_positive_int("windows", raw_quality["windows"]),
            window_tokens=_positive_int("window_tokens", raw_quality["window_tokens"]),
            prompts_file=str(raw_quality["prompts_file"]),
            generation_tokens=_positive_int("generation_tokens", raw_quality["generation_tokens"]),
            quantization_perplexity_limit=float(raw_quality["quantization_perplexity_limit"]),
            pruning_perplexity_limit=float(raw_quality["pruning_perplexity_limit"]),
        ),
        performance=PerformanceSpec(
            prefill_lengths=_int_tuple("prefill_lengths", raw_performance["prefill_lengths"]),
            decode_prompt_lengths=_int_tuple("decode_prompt_lengths", raw_performance["decode_prompt_lengths"]),
            batch_sizes=_int_tuple("batch_sizes", raw_performance["batch_sizes"]),
            decode_tokens=_positive_int("decode_tokens", raw_performance["decode_tokens"]),
            warmup_iterations=_positive_int("warmup_iterations", raw_performance["warmup_iterations"]),
            measure_iterations=_positive_int("measure_iterations", raw_performance["measure_iterations"]),
            repetitions=_positive_int("repetitions", raw_performance["repetitions"]),
            use_kv_cache=bool(raw_performance["use_kv_cache"]),
        ),
        execution=ModelExecutionSpec(
            compile=bool(raw_execution["compile"]),
            compile_mode=str(raw_execution["compile_mode"]),
            local_files_only=bool(raw_execution["local_files_only"]),
            capabilities_file=str(raw_execution["capabilities_file"]),
        ),
    )


def load_model_config(path: str | Path) -> ModelExperimentConfig:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentConfigurationError(f"nao foi possivel ler config OPT: {path}") from exc
    if not isinstance(value, dict):
        raise ExperimentConfigurationError("config OPT deve ser objeto")
    return model_config_from_mapping(value)


def is_target_linear(name: str, module: nn.Module, scope: str) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    normalized = name.casefold()
    if normalized.endswith("lm_head") or "embed" in normalized:
        return False
    if scope == "attention_only":
        return ".self_attn." in normalized and normalized.rsplit(".", 1)[-1] in {
            "q_proj", "k_proj", "v_proj", "out_proj"
        }
    if scope == "transformer_blocks":
        return ".layers." in normalized
    if scope == "none":
        return False
    raise ValueError(f"scope desconhecido: {scope}")


def target_linears(model: nn.Module, scope: str) -> tuple[tuple[str, nn.Linear], ...]:
    return tuple(
        (name, module)
        for name, module in model.named_modules()
        if is_target_linear(name, module, scope)
    )


def apply_model_scenario(model: nn.Module, scenario: ModelScenario) -> int:
    targets = target_linears(model, scenario.optimization_scope)
    if scenario.kind != "baseline" and not targets:
        raise RuntimeError(f"nenhuma camada encontrada para scope {scenario.optimization_scope}")
    if scenario.kind == "pruning_magnitude":
        with torch.no_grad():
            for _, module in targets:
                module.weight.copy_(exact_magnitude_prune(module.weight, scenario.pruning_sparsity))
    elif scenario.kind == "quantization_int8_fake_per_row":
        with torch.no_grad():
            for _, module in targets:
                _, _, dequantized = symmetric_int8_quantize_per_row(module.weight)
                module.weight.copy_(dequantized.to(module.weight.dtype))
    elif scenario.kind in {"quantization_int8_weight_only", "quantization_int8_dynamic"}:
        from torchao.quantization import (
            Int8DynamicActivationInt8WeightConfig,
            Int8WeightOnlyConfig,
            quantize_,
        )
        from torchao.quantization.granularity import PerRow

        quantization_config = (
            Int8WeightOnlyConfig(version=2, granularity=PerRow(), set_inductor_config=False)
            if scenario.kind == "quantization_int8_weight_only"
            else Int8DynamicActivationInt8WeightConfig(
                version=2,
                granularity=PerRow(),
                set_inductor_config=False,
            )
        )
        target_names = {name for name, _ in targets}
        quantize_(
            model,
            quantization_config,
            filter_fn=lambda module, name: name in target_names,
        )
    elif scenario.kind == "pruning_2to4":
        from torch.sparse import to_sparse_semi_structured

        with torch.no_grad():
            for _, module in targets:
                pruned = magnitude_prune_2to4(module.weight.detach())
                module.weight = nn.Parameter(to_sparse_semi_structured(pruned), requires_grad=False)
    return len(targets)


def model_storage_bytes(model: nn.Module) -> int:
    return _module_storage_bytes(model)


def observed_model_sparsity(model: nn.Module, scope: str, scenario_kind: str) -> float:
    if scenario_kind not in {"pruning_magnitude", "pruning_2to4"}:
        return 0.0
    if scenario_kind == "pruning_2to4":
        return 0.5
    zeros = 0
    count = 0
    for _, module in target_linears(model, scope):
        try:
            weight = module.weight.detach()
            dense = weight.to_dense() if weight.layout != torch.strided else weight
            zeros += int(torch.count_nonzero(dense == 0).item())
            count += dense.numel()
        except RuntimeError:
            return 0.5
    return zeros / count if count else 0.0


def _hash_values(values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def model_weight_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        if hasattr(parameter, "qdata"):
            value = parameter.qdata.detach().cpu().contiguous()
        else:
            try:
                value = parameter.detach().to_dense().cpu().contiguous()
            except RuntimeError:
                digest.update(repr(parameter).encode("utf-8"))
                continue
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_evaluation_subset(
    token_ids: Sequence[int],
    windows: int,
    window_tokens: int,
    performance_token_count: int = 0,
) -> EvaluationSubset:
    if len(token_ids) < window_tokens:
        raise ValueError("dataset tokenizado menor que uma janela")
    maximum_start = len(token_ids) - window_tokens
    starts = np.linspace(0, maximum_start, num=windows, dtype=np.int64)
    unique_starts = tuple(dict.fromkeys(int(value) for value in starts))
    if len(unique_starts) != windows:
        raise ValueError("dataset insuficiente para janelas distintas")
    selected = tuple(
        (start, tuple(int(value) for value in token_ids[start : start + window_tokens]))
        for start in unique_starts
    )
    if performance_token_count and len(token_ids) < performance_token_count:
        raise ValueError("dataset tokenizado menor que a entrada de desempenho")
    performance_tokens = tuple(int(value) for value in token_ids[:performance_token_count])
    return EvaluationSubset(
        selected,
        _hash_values(value for _, window in selected for value in window),
        performance_tokens,
    )


def load_evaluation_data(config: ModelExperimentConfig, tokenizer) -> EvaluationSubset:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ExperimentEnvironmentError("datasets nao instalado") from exc
    dataset = load_dataset(
        config.dataset.identifier,
        config.dataset.configuration,
        revision=config.dataset.revision,
        split=config.dataset.split,
    )
    text = "\n\n".join(str(item) for item in dataset["text"] if str(item).strip())
    token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    performance_token_count = max(
        *config.performance.prefill_lengths,
        *config.performance.decode_prompt_lengths,
    )
    return build_evaluation_subset(
        token_ids,
        config.quality.windows,
        config.quality.window_tokens,
        performance_token_count,
    )


def load_prompts(path: str | Path) -> tuple[dict[str, str], ...]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or len(value) != 20:
        raise ValueError("arquivo de prompts deve conter exatamente 20 itens")
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"id", "text"}:
            raise ValueError("prompt invalido")
        result.append({"id": str(item["id"]), "text": str(item["text"])})
    if len({item["id"] for item in result}) != len(result):
        raise ValueError("ids de prompt duplicados")
    return tuple(result)


def _cosine(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(reference.float(), candidate.float(), dim=-1).mean())


def _generated_agreement(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    size = min(reference.numel(), candidate.numel())
    if size == 0:
        return 1.0
    return float((reference.reshape(-1)[:size] == candidate.reshape(-1)[:size]).float().mean())


def evaluate_quality(
    model: nn.Module,
    tokenizer,
    subset: EvaluationSubset,
    prompts: Sequence[Mapping[str, str]],
    generation_tokens: int,
    *,
    model_id: str,
    scenario_id: str,
    baseline_prompts: Mapping[str, BaselinePrompt] | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, BaselinePrompt], dict[str, float]]:
    device = next(model.parameters()).device
    window_rows: list[dict[str, object]] = []
    losses: list[float] = []
    model.eval()
    with torch.inference_mode():
        for index, (start, window) in enumerate(subset.windows):
            input_ids = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
            output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
            loss = float(output.loss)
            losses.append(loss)
            window_rows.append(
                {
                    "model_id": model_id,
                    "scenario": scenario_id,
                    "window_index": index,
                    "start_token": start,
                    "token_count": len(window),
                    "loss": loss,
                    "perplexity": math.exp(min(loss, 80.0)),
                }
            )

    prompt_rows: list[dict[str, object]] = []
    produced_baselines: dict[str, BaselinePrompt] = {}
    for prompt in prompts:
        encoded = tokenizer(prompt["text"], return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
            logits = output.logits[:, -1, :].detach().float().cpu()
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=generation_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )[:, input_ids.shape[1] :].detach().cpu()
        if baseline_prompts is None:
            baseline = BaselinePrompt(logits=logits, generated_ids=generated)
            produced_baselines[prompt["id"]] = baseline
        else:
            baseline = baseline_prompts[prompt["id"]]
        baseline_log_probs = torch.nn.functional.log_softmax(baseline.logits, dim=-1)
        candidate_log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        kl = torch.nn.functional.kl_div(
            candidate_log_probs,
            baseline_log_probs.exp(),
            reduction="batchmean",
        )
        prompt_rows.append(
            {
                "model_id": model_id,
                "scenario": scenario_id,
                "prompt_id": prompt["id"],
                "input_tokens": int(input_ids.numel()),
                "generated_tokens": int(generated.numel()),
                "logits_mse": float(torch.mean((baseline.logits - logits) ** 2)),
                "logits_cosine": _cosine(baseline.logits, logits),
                "logits_kl_divergence": float(kl),
                "top1_agreement": float(
                    (baseline.logits.argmax(dim=-1) == logits.argmax(dim=-1)).float().mean()
                ),
                "generated_token_agreement": _generated_agreement(baseline.generated_ids, generated),
                "exact_generation": float(torch.equal(baseline.generated_ids, generated)),
            }
        )
    prompt_frame = {key: float(np.mean([float(row[key]) for row in prompt_rows])) for key in (
        "logits_mse", "logits_cosine", "logits_kl_divergence", "top1_agreement",
        "generated_token_agreement", "exact_generation",
    )}
    mean_loss = float(np.mean(losses))
    prompt_frame["mean_loss"] = mean_loss
    prompt_frame["perplexity"] = math.exp(min(mean_loss, 80.0))
    return window_rows, prompt_rows, produced_baselines, prompt_frame


def _measure_call(callable_fn: Callable[[], object], warmups: int, iterations: int) -> tuple[list[float], object]:
    for _ in range(warmups):
        callable_fn()
    torch.cuda.synchronize()
    timings: list[float] = []
    output: object = None
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = callable_fn()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    return timings, output


def profile_model_operators(
    model: nn.Module,
    token_source: Sequence[int],
    seq_len: int = 128,
    use_cache: bool = True,
) -> list[str]:
    effective_length = min(seq_len, len(token_source))
    source = tuple(token_source[:effective_length])
    if not source:
        raise ValueError("tokens insuficientes para o profiler do modelo")
    device = next(model.parameters()).device
    input_ids = torch.tensor(source, dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        for _ in range(2):
            model(input_ids=input_ids, use_cache=use_cache)
        torch.cuda.synchronize(device)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
            model(input_ids=input_ids, use_cache=use_cache)
            torch.cuda.synchronize(device)
    return sorted({item.key for item in captured.key_averages()})


def profile_model_decode_operators(
    model: nn.Module,
    token_source: Sequence[int],
    seq_len: int = 128,
    use_cache: bool = True,
) -> list[str]:
    effective_length = min(seq_len, len(token_source))
    source = tuple(token_source[:effective_length])
    if not source:
        raise ValueError("tokens insuficientes para o profiler de decode")
    device = next(model.parameters()).device
    input_ids = torch.tensor(source, dtype=torch.long, device=device).unsqueeze(0)
    with torch.inference_mode():
        for _ in range(2):
            token, cache = _prepare_decode(model, input_ids, use_cache)
            _decode_after_prefill(model, input_ids, token, cache, 1, use_cache)
        token, cache = _prepare_decode(model, input_ids, use_cache)
        torch.cuda.synchronize(device)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as captured:
            _decode_after_prefill(model, input_ids, token, cache, 1, use_cache)
            torch.cuda.synchronize(device)
    return sorted({item.key for item in captured.key_averages()})


def _prepare_decode(model, input_ids: torch.Tensor, use_cache: bool):
    with torch.inference_mode():
        output = model(input_ids=input_ids, use_cache=use_cache)
    token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    cache = output.past_key_values if use_cache else None
    return token, cache


def _decode_after_prefill(
    model,
    input_ids: torch.Tensor,
    token: torch.Tensor,
    cache,
    decode_steps: int,
    use_cache: bool,
) -> torch.Tensor:
    running_input = input_ids
    with torch.inference_mode():
        for _ in range(decode_steps):
            if not use_cache:
                running_input = torch.cat((running_input, token), dim=1)
            output = model(
                input_ids=token if use_cache else running_input,
                past_key_values=cache,
                use_cache=use_cache,
            )
            token = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cache = output.past_key_values if use_cache else None
    return token.detach().cpu()


def _measure_decode(
    model,
    input_ids: torch.Tensor,
    decode_tokens: int,
    use_cache: bool,
    warmups: int,
    iterations: int,
) -> tuple[list[float], torch.Tensor]:
    decode_steps = decode_tokens - 1
    if decode_steps < 1:
        raise ValueError("decode requer pelo menos dois tokens gerados")
    for _ in range(warmups):
        token, cache = _prepare_decode(model, input_ids, use_cache)
        _decode_after_prefill(model, input_ids, token, cache, decode_steps, use_cache)
    torch.cuda.synchronize()
    timings: list[float] = []
    generated_token = torch.empty(0, dtype=torch.long)
    for _ in range(iterations):
        token, cache = _prepare_decode(model, input_ids, use_cache)
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        generated_token = _decode_after_prefill(
            model,
            input_ids,
            token,
            cache,
            decode_steps,
            use_cache,
        )
        ended.record()
        ended.synchronize()
        timings.append(float(started.elapsed_time(ended)) / decode_steps)
    return timings, generated_token


def _case_id(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:20]


def failure_status(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}".casefold()
    unsupported_markers = (
        "not supported",
        "unsupported",
        "not implemented",
        "notimplementederror",
        "needs to be greater than",
    )
    return "unsupported" if any(marker in message for marker in unsupported_markers) else "failed"


def benchmark_model_performance(
    model,
    token_source: Sequence[int],
    config: ModelExperimentConfig,
    model_spec: ModelSpec,
    scenario: ModelScenario,
    quality_metrics: Mapping[str, float],
    storage_bytes: int,
    parameter_count: int,
    weight_hash: str,
    hidden_size: int,
    num_attention_heads: int,
    uses_sparse_kernel: Mapping[str, bool],
    uses_int8_kernel: Mapping[str, bool],
    profiler_trace: Mapping[str, str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    device = next(model.parameters()).device
    baseline_scenario = next(
        item for item in config.scenarios
        if item.comparison_group == scenario.comparison_group and item.kind == "baseline"
    )

    def make_record(
        *,
        operation: str,
        batch_size: int,
        seq_len: int,
        repeat_index: int,
        status: str,
        skip_reason: str = "",
        timings: Sequence[float] = (),
        throughput_units: int = 0,
        output_sha256: str = "",
    ) -> dict[str, object]:
        profile_prefix = "prefill" if operation == "model_prefill" else "decode"
        case_id = _case_id(
            config.experiment_id,
            model_spec.identifier,
            scenario.identifier,
            operation,
            batch_size,
            seq_len,
            repeat_index,
        )
        baseline_case = _case_id(
            config.experiment_id,
            model_spec.identifier,
            baseline_scenario.identifier,
            operation,
            batch_size,
            seq_len,
            repeat_index,
        )
        common = dict(
            experiment_id=config.experiment_id,
            stage="pretrained_hardware",
            case_id=case_id,
            baseline_case_id=baseline_case,
            data_seed=2026,
            repeat_index=repeat_index,
            profile_id=f"{profile_prefix}_b{batch_size}_l{seq_len}",
            model_id=model_spec.identifier,
            model_revision=model_spec.revision,
            operation=operation,
            optimization_scope=scenario.optimization_scope,
            scenario=scenario.identifier,
            comparison_group=scenario.comparison_group,
            status=status,
            skip_reason=skip_reason,
            batch_size=batch_size,
            seq_len=seq_len,
            d_model=hidden_size,
            num_heads=num_attention_heads,
            dtype=scenario.dtype,
            backend="transformers_pytorch_cuda",
            compile_mode=config.execution.compile_mode if config.execution.compile else "eager",
            input_distribution="tokens",
            pruning_method="magnitude_2to4" if scenario.kind == "pruning_2to4" else "none",
            pruning_sparsity=scenario.pruning_sparsity,
            quantization_method=scenario.kind if "quantization" in scenario.kind else "none",
            weight_storage_dtype="int8" if "int8" in scenario.kind else scenario.dtype,
            activation_dtype=scenario.dtype,
            observed_sparsity=scenario.pruning_sparsity,
            uses_sparse_kernel=uses_sparse_kernel.get(operation, False),
            uses_int8_kernel=uses_int8_kernel.get(operation, False),
        )
        if status != "complete":
            return empty_benchmark_record_v3(**common)
        mean_latency = float(np.mean(timings))
        source = list(token_source[:seq_len])
        return empty_benchmark_record_v3(
            **common,
            latency_ms_mean=mean_latency,
            latency_ms_p50=float(np.percentile(timings, 50)),
            latency_ms_p95=float(np.percentile(timings, 95)),
            throughput_tokens_s=throughput_units / (mean_latency / 1000.0),
            peak_memory_allocated_bytes=int(torch.cuda.max_memory_allocated(device)),
            peak_memory_reserved_bytes=int(torch.cuda.max_memory_reserved(device)),
            parameter_storage_bytes=storage_bytes,
            theoretical_flops=parameter_count,
            mse=quality_metrics["logits_mse"],
            mae=0.0,
            r2=0.0,
            cosine_similarity=quality_metrics["logits_cosine"],
            kl_divergence=quality_metrics["logits_kl_divergence"],
            top1_agreement=quality_metrics["top1_agreement"],
            loss=quality_metrics["mean_loss"],
            perplexity=quality_metrics["perplexity"],
            input_sha256=_hash_values(source),
            weight_sha256=weight_hash,
            output_sha256=output_sha256,
            profiler_trace=profiler_trace[operation],
        )

    def append_timings(record: Mapping[str, object], values: Sequence[float]) -> None:
        for iteration, latency in enumerate(values, start=1):
            timing_rows.append(
                {
                    "case_id": record["case_id"],
                    "model_id": model_spec.identifier,
                    "scenario": scenario.identifier,
                    "operation": record["operation"],
                    "repeat_index": record["repeat_index"],
                    "iteration": iteration,
                    "latency_ms": latency,
                }
            )

    def append_failure(
        operation: str,
        batch_size: int,
        seq_len: int,
        repeat_index: int,
        exc: Exception,
    ) -> None:
        records.append(
            make_record(
                operation=operation,
                batch_size=batch_size,
                seq_len=seq_len,
                repeat_index=repeat_index,
                status=failure_status(exc),
                skip_reason=f"{type(exc).__name__}: {exc}",
            )
        )

    decode_jobs: list[tuple[int, int, int]] = []
    for repeat_index in range(1, config.performance.repetitions + 1):
        for batch_size in config.performance.batch_sizes:
            for seq_len in config.performance.prefill_lengths:
                source = list(token_source[:seq_len])
                input_ids = torch.tensor(source, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)

                def prefill():
                    with torch.inference_mode():
                        return model(input_ids=input_ids, use_cache=config.performance.use_kv_cache)

                try:
                    torch.cuda.reset_peak_memory_stats(device)
                    timings, output = _measure_call(
                        prefill,
                        config.performance.warmup_iterations,
                        config.performance.measure_iterations,
                    )
                    output_hash = hashlib.sha256(
                        output.logits[:, -1, :].detach().float().cpu().contiguous().numpy().tobytes()
                    ).hexdigest()
                    del output
                    record = make_record(
                        operation="model_prefill",
                        batch_size=batch_size,
                        seq_len=seq_len,
                        repeat_index=repeat_index,
                        status="complete",
                        timings=timings,
                        throughput_units=batch_size * seq_len,
                        output_sha256=output_hash,
                    )
                    records.append(record)
                    append_timings(record, timings)
                except Exception as exc:
                    append_failure("model_prefill", batch_size, seq_len, repeat_index, exc)
            for prompt_len in config.performance.decode_prompt_lengths:
                source = list(token_source[:prompt_len])
                input_ids = torch.tensor(source, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
                def ttft():
                    with torch.inference_mode():
                        return model(input_ids=input_ids, use_cache=config.performance.use_kv_cache)

                try:
                    torch.cuda.reset_peak_memory_stats(device)
                    ttft_values, output = _measure_call(
                        ttft,
                        config.performance.warmup_iterations,
                        config.performance.measure_iterations,
                    )
                    output_hash = hashlib.sha256(
                        output.logits[:, -1, :].detach().float().cpu().contiguous().numpy().tobytes()
                    ).hexdigest()
                    del output
                    record = make_record(
                        operation="model_ttft",
                        batch_size=batch_size,
                        seq_len=prompt_len,
                        repeat_index=repeat_index,
                        status="complete",
                        timings=ttft_values,
                        throughput_units=batch_size,
                        output_sha256=output_hash,
                    )
                    records.append(record)
                    append_timings(record, ttft_values)
                except Exception as exc:
                    append_failure("model_ttft", batch_size, prompt_len, repeat_index, exc)
                decode_jobs.append((repeat_index, batch_size, prompt_len))

    for repeat_index, batch_size, prompt_len in decode_jobs:
        source = list(token_source[:prompt_len])
        input_ids = torch.tensor(source, dtype=torch.long, device=device).unsqueeze(0).repeat(batch_size, 1)
        try:
            torch.cuda.reset_peak_memory_stats(device)
            decode_values, generated_token = _measure_decode(
                model,
                input_ids,
                config.performance.decode_tokens,
                config.performance.use_kv_cache,
                config.performance.warmup_iterations,
                config.performance.measure_iterations,
            )
            record = make_record(
                operation="model_decode",
                batch_size=batch_size,
                seq_len=prompt_len,
                repeat_index=repeat_index,
                status="complete",
                timings=decode_values,
                throughput_units=batch_size,
                output_sha256=hashlib.sha256(
                    generated_token.contiguous().numpy().tobytes()
                ).hexdigest(),
            )
            records.append(record)
            append_timings(record, decode_values)
        except Exception as exc:
            append_failure("model_decode", batch_size, prompt_len, repeat_index, exc)
    for record in records:
        validation = validate_benchmark_record_v3(record)
        if not validation.is_valid:
            raise RuntimeError(f"registro de desempenho OPT invalido: {validation}")
    return records, timing_rows


def _load_model_and_tokenizer(model_spec: ModelSpec, dtype: torch.dtype, local_files_only: bool):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ExperimentEnvironmentError("transformers nao instalado") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec.identifier,
        revision=model_spec.revision,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_spec.identifier,
        revision=model_spec.revision,
        dtype=dtype,
        local_files_only=local_files_only,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda").eval()
    return model, tokenizer


def prefetch_models(config: ModelExperimentConfig) -> None:
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ExperimentEnvironmentError("dependencias Hugging Face ausentes") from exc
    for model_spec in config.models:
        AutoTokenizer.from_pretrained(model_spec.identifier, revision=model_spec.revision)
        AutoModelForCausalLM.from_pretrained(model_spec.identifier, revision=model_spec.revision, dtype=torch.bfloat16)
    load_dataset(
        config.dataset.identifier,
        config.dataset.configuration,
        revision=config.dataset.revision,
        split=config.dataset.split,
    )


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
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


def _baseline_cache_name(model_id: str, comparison_group: str) -> str:
    key = hashlib.sha256(f"{model_id}|{comparison_group}".encode("utf-8")).hexdigest()[:16]
    return f"baseline_prompts_{key}.pt"


def _save_baseline_cache(
    output: Path,
    *,
    model_id: str,
    scenario_id: str,
    comparison_group: str,
    perplexity: float,
    prompts: Mapping[str, BaselinePrompt],
) -> None:
    payload = {
        "schema_version": 1,
        "model_id": model_id,
        "scenario": scenario_id,
        "comparison_group": comparison_group,
        "perplexity": perplexity,
        "prompts": {
            prompt_id: {
                "logits": baseline.logits.cpu(),
                "generated_ids": baseline.generated_ids.cpu(),
            }
            for prompt_id, baseline in prompts.items()
        },
    }
    torch.save(payload, output / _baseline_cache_name(model_id, comparison_group))


def _load_baseline_cache(
    output: Path,
    *,
    model_id: str,
    comparison_group: str,
) -> tuple[dict[str, BaselinePrompt], float] | None:
    path = output / _baseline_cache_name(model_id, comparison_group)
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ExperimentEnvironmentError(f"cache de baseline invalido: {path.name}")
    if payload.get("model_id") != model_id or payload.get("comparison_group") != comparison_group:
        raise ExperimentEnvironmentError(f"cache de baseline pertence a outro grupo: {path.name}")
    raw_prompts = payload.get("prompts")
    if not isinstance(raw_prompts, Mapping):
        raise ExperimentEnvironmentError(f"cache de prompts invalido: {path.name}")
    prompts: dict[str, BaselinePrompt] = {}
    for prompt_id, raw in raw_prompts.items():
        if not isinstance(raw, Mapping):
            raise ExperimentEnvironmentError(f"prompt baseline invalido em {path.name}")
        logits = raw.get("logits")
        generated_ids = raw.get("generated_ids")
        if not isinstance(logits, torch.Tensor) or not isinstance(generated_ids, torch.Tensor):
            raise ExperimentEnvironmentError(f"tensores baseline invalidos em {path.name}")
        prompts[str(prompt_id)] = BaselinePrompt(logits=logits.cpu(), generated_ids=generated_ids.cpu())
    return prompts, float(payload["perplexity"])


def _model_config_hash(config: ModelExperimentConfig) -> str:
    canonical = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _model_code_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _model_environment_hash(metadata: Mapping[str, object]) -> str:
    stable = {key: value for key, value in metadata.items() if key not in {"captured_at", "git"}}
    canonical = json.dumps(stable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_model_checkpoint(
    output: Path,
    manifest: dict[str, object],
    subset_records: Mapping[str, object],
    quality_rows: list[dict[str, object]],
    window_rows: list[dict[str, object]],
    prompt_rows: list[dict[str, object]],
    performance_rows: list[dict[str, object]],
    timing_rows: list[dict[str, object]],
    operator_traces: Mapping[str, Sequence[str]],
) -> None:
    subset_path = output / "evaluation_subset.json"
    subset_path.write_text(
        json.dumps(subset_records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    artifacts = {
        "quality.csv": (QUALITY_COLUMNS, quality_rows),
        "quality_windows.csv": (WINDOW_COLUMNS, window_rows),
        "prompt_metrics.csv": (PROMPT_COLUMNS, prompt_rows),
        "performance.csv": (CSV_COLUMNS_V3, performance_rows),
        "timings.csv": (TIMING_COLUMNS, timing_rows),
    }
    entries: list[dict[str, object]] = []
    for name, (columns, rows) in artifacts.items():
        path = output / name
        _write_csv(path, columns, rows)
        entries.append({"path": name, "sha256": _sha256(path), "records": len(rows)})
    entries.append(
        {"path": subset_path.name, "sha256": _sha256(subset_path), "records": len(subset_records)}
    )
    operators_path = output / "model_operators.json"
    operators_path.write_text(
        json.dumps(operator_traces, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    entries.append(
        {
            "path": operators_path.name,
            "sha256": _sha256(operators_path),
            "records": len(operator_traces),
        }
    )
    for cache_path in sorted(output.glob("baseline_prompts_*.pt")):
        entries.append(
            {
                "path": cache_path.name,
                "sha256": _sha256(cache_path),
                "records": 1,
            }
        )
    manifest["artifacts"] = entries


def execute_model_experiment(
    config: ModelExperimentConfig,
    output_dir: str | Path,
    *,
    resume: bool = False,
    allow_code_hash_migration: bool = False,
    max_new_scenarios: int | None = None,
) -> Path:
    if max_new_scenarios is not None and max_new_scenarios <= 0:
        raise ExperimentConfigurationError("max_new_scenarios deve ser positivo")
    if not torch.cuda.is_available():
        raise ExperimentEnvironmentError("benchmark OPT exige CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if config.required_gpu_name.casefold() not in gpu_name.casefold():
        raise ExperimentEnvironmentError(f"GPU inesperada: {gpu_name}")
    capabilities: dict[str, object] = {}
    if config.execution.capabilities_file:
        capabilities = json.loads(Path(config.execution.capabilities_file).read_text(encoding="utf-8"))
        if not capabilities.get("passed", False):
            raise ExperimentEnvironmentError("portao de hardware nao aprovado")

    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    config_hash = _model_config_hash(config)
    code_hash = _model_code_hash()
    current_environment = environment_metadata(torch.device("cuda:0"))
    environment_hash = _model_environment_hash(current_environment)
    quality_rows: list[dict[str, object]] = []
    window_rows: list[dict[str, object]] = []
    prompt_rows: list[dict[str, object]] = []
    performance_rows: list[dict[str, object]] = []
    timing_rows: list[dict[str, object]] = []
    subset_records: dict[str, object] = {}
    operator_traces: dict[str, list[str]] = {}
    completed: set[tuple[str, str]] = set()

    if output.exists() and any(output.iterdir()):
        if not resume:
            raise FileExistsError(f"diretorio de saida nao esta vazio: {output}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentEnvironmentError("manifesto OPT ausente para retomada") from exc
        if manifest.get("config_sha256") != config_hash:
            raise ExperimentEnvironmentError("configuracao diverge da execucao OPT retomada")
        if manifest.get("code_sha256") != code_hash:
            if not allow_code_hash_migration:
                raise ExperimentEnvironmentError("codigo diverge da execucao OPT retomada")
            migrations = manifest.setdefault("code_sha256_migrations", [])
            if isinstance(migrations, list):
                migrations.append(
                    {
                        "from": manifest.get("code_sha256"),
                        "to": code_hash,
                        "reason": "retomada com cache persistente de baseline para reduzir recarregamentos",
                        "migrated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                    }
                )
            manifest["code_sha256"] = code_hash
        if manifest.get("environment_sha256") != environment_hash:
            raise ExperimentEnvironmentError("ambiente diverge da execucao OPT retomada")
        for item in manifest.get("artifacts", []):
            if not isinstance(item, Mapping):
                raise ExperimentEnvironmentError("artefato invalido no manifesto OPT")
            path = output / str(item.get("path", ""))
            if not path.exists() or _sha256(path) != item.get("sha256"):
                raise ExperimentEnvironmentError(f"artefato adulterado na retomada: {path.name}")
        quality_rows = _read_csv_rows(output / "quality.csv")
        window_rows = _read_csv_rows(output / "quality_windows.csv")
        prompt_rows = _read_csv_rows(output / "prompt_metrics.csv")
        performance_rows = _read_csv_rows(output / "performance.csv")
        timing_rows = _read_csv_rows(output / "timings.csv")
        try:
            stored_subsets = json.loads((output / "evaluation_subset.json").read_text(encoding="utf-8"))
            if isinstance(stored_subsets, dict):
                subset_records = stored_subsets
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentEnvironmentError("subconjunto de avaliacao invalido na retomada") from exc
        try:
            stored_operators = json.loads((output / "model_operators.json").read_text(encoding="utf-8"))
            if isinstance(stored_operators, dict):
                operator_traces = {
                    str(key): [str(operator) for operator in value]
                    for key, value in stored_operators.items()
                    if isinstance(value, list)
                }
        except (OSError, json.JSONDecodeError) as exc:
            raise ExperimentEnvironmentError("traces de operadores invalidos na retomada") from exc
        for item in manifest.get("models", []):
            if isinstance(item, Mapping) and item.get("status") in {"complete", "unsupported", "failed"}:
                completed.add((str(item.get("model_id")), str(item.get("scenario"))))
        manifest["status"] = "running"
        resumed_at = manifest.setdefault("resumed_at", [])
        if isinstance(resumed_at, list):
            resumed_at.append(datetime.now().astimezone().isoformat(timespec="seconds"))
    else:
        output.mkdir(parents=True, exist_ok=True)
        (output / "config.snapshot.json").write_text(
            json.dumps(config.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (output / "environment.json").write_text(
            json.dumps(current_environment, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "experiment_id": config.experiment_id,
            "status": "running",
            "config": "config.snapshot.json",
            "config_sha256": config_hash,
            "code_sha256": code_hash,
            "environment": "environment.json",
            "environment_sha256": environment_hash,
            "models": [],
            "resumed_at": [],
        }
    prompts = load_prompts(config.quality.prompts_file)
    subsets_by_model: dict[str, EvaluationSubset] = {}
    for model_id, raw_subset in subset_records.items():
        if not isinstance(raw_subset, Mapping):
            continue
        try:
            windows = tuple(
                (int(item["start"]), tuple(int(token) for token in item["tokens"]))
                for item in raw_subset["windows"]
            )
            performance_tokens = tuple(int(token) for token in raw_subset["performance_tokens"])
            subsets_by_model[model_id] = EvaluationSubset(
                windows,
                str(raw_subset["sha256"]),
                performance_tokens,
            )
        except (KeyError, TypeError, ValueError):
            raise ExperimentEnvironmentError(f"subconjunto armazenado invalido: {model_id}")
    _write_model_checkpoint(
        output,
        manifest,
        subset_records,
        quality_rows,
        window_rows,
        prompt_rows,
        performance_rows,
        timing_rows,
        operator_traces,
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    new_scenarios = 0
    try:
        for model_spec in config.models:
            baseline_prompts_by_group: dict[str, dict[str, BaselinePrompt]] = {}
            baseline_perplexity_by_group: dict[str, float] = {}
            baseline_token_source: tuple[int, ...] | None = None
            ordered = sorted(config.scenarios, key=lambda item: item.kind != "baseline")
            for scenario in ordered:
                scenario_key = (model_spec.identifier, scenario.identifier)
                already_completed = scenario_key in completed
                if already_completed and scenario.kind == "baseline":
                    cached_baseline = _load_baseline_cache(
                        output,
                        model_id=model_spec.identifier,
                        comparison_group=scenario.comparison_group,
                    )
                    if cached_baseline is not None:
                        baseline_prompts_by_group[scenario.comparison_group] = cached_baseline[0]
                        baseline_perplexity_by_group[scenario.comparison_group] = cached_baseline[1]
                        continue
                if already_completed and scenario.kind != "baseline":
                    continue
                dtype = torch.bfloat16 if scenario.dtype == "bfloat16" else torch.float16
                model = None
                benchmark_model = None
                performance: list[dict[str, object]] = []
                timings: list[dict[str, object]] = []
                try:
                    model, tokenizer = _load_model_and_tokenizer(
                        model_spec,
                        dtype,
                        config.execution.local_files_only,
                    )
                    subset = subsets_by_model.get(model_spec.identifier)
                    if subset is None:
                        subset = load_evaluation_data(config, tokenizer)
                        subsets_by_model[model_spec.identifier] = subset
                        subset_records[model_spec.identifier] = {
                            "sha256": subset.sha256,
                            "windows": [
                                {"start": start, "tokens": list(tokens)}
                                for start, tokens in subset.windows
                            ],
                            "performance_tokens": list(subset.performance_tokens),
                            "performance_tokens_sha256": _hash_values(subset.performance_tokens),
                        }
                    baseline_token_source = subset.performance_tokens
                    target_count = apply_model_scenario(model, scenario)
                    storage_bytes = model_storage_bytes(model)
                    weight_hash = model_weight_hash(model)
                    parameter_count = sum(parameter.numel() for parameter in model.parameters())
                    hidden_size = int(model.config.hidden_size)
                    num_attention_heads = int(model.config.num_attention_heads)
                    baselines = baseline_prompts_by_group.get(scenario.comparison_group)
                    windows, prompt_metrics, produced, metrics = evaluate_quality(
                        model,
                        tokenizer,
                        subset,
                        prompts,
                        config.quality.generation_tokens,
                        model_id=model_spec.identifier,
                        scenario_id=scenario.identifier,
                        baseline_prompts=baselines,
                    )
                    baseline_perplexity = (
                        metrics["perplexity"]
                        if scenario.kind == "baseline"
                        else baseline_perplexity_by_group[scenario.comparison_group]
                    )
                    perplexity_ratio = metrics["perplexity"] / baseline_perplexity
                    limit = (
                        config.quality.pruning_perplexity_limit
                        if "pruning" in scenario.kind
                        else config.quality.quantization_perplexity_limit
                        if "quantization" in scenario.kind
                        else 0.0
                    )
                    acceptable = scenario.kind == "baseline" or perplexity_ratio <= 1.0 + limit
                    sparsity = observed_model_sparsity(
                        model,
                        scenario.optimization_scope,
                        scenario.kind,
                    )
                    quality_record = {
                            "experiment_id": config.experiment_id,
                            "model_id": model_spec.identifier,
                            "model_revision": model_spec.revision,
                            "scenario": scenario.identifier,
                            "comparison_group": scenario.comparison_group,
                            "optimization_scope": scenario.optimization_scope,
                            "dtype": scenario.dtype,
                            "status": "complete",
                            "skip_reason": "",
                            "parameter_count": parameter_count,
                            "parameter_storage_bytes": storage_bytes,
                            "target_linear_count": target_count,
                            "pruning_sparsity": scenario.pruning_sparsity,
                            "observed_sparsity": sparsity,
                            "mean_loss": metrics["mean_loss"],
                            "perplexity": metrics["perplexity"],
                            "perplexity_increase_ratio": perplexity_ratio - 1.0,
                            "logits_mse": metrics["logits_mse"],
                            "logits_cosine": metrics["logits_cosine"],
                            "logits_kl_divergence": metrics["logits_kl_divergence"],
                            "top1_agreement": metrics["top1_agreement"],
                            "generated_token_agreement": metrics["generated_token_agreement"],
                            "exact_generation_rate": metrics["exact_generation"],
                            "quality_acceptable": acceptable,
                            "weight_sha256": weight_hash,
                            "evaluation_subset_sha256": subset.sha256,
                        }
                    if scenario.measure_performance and not already_completed:
                        benchmark_model = model
                        if config.execution.compile:
                            benchmark_model = torch.compile(model, mode=config.execution.compile_mode)
                        checks = capabilities.get("checks", {}) if isinstance(capabilities, dict) else {}
                        trace_key = f"{model_spec.identifier}|{scenario.identifier}"
                        trace_names = {
                            "model_prefill": f"{trace_key}|prefill",
                            "model_ttft": f"{trace_key}|prefill",
                            "model_decode": f"{trace_key}|decode",
                        }
                        flags_by_operation: dict[str, dict[str, bool]] = {}
                        prefill_name = trace_names["model_prefill"]
                        try:
                            operators = profile_model_operators(
                                benchmark_model,
                                baseline_token_source,
                                use_cache=config.performance.use_kv_cache,
                            )
                            operator_traces[prefill_name] = operators
                            flags_by_operation["model_prefill"] = detect_kernel_flags(operators)
                        except Exception as exc:
                            operator_traces[prefill_name] = [
                                f"PROFILE_{failure_status(exc).upper()}: {type(exc).__name__}: {exc}"
                            ]
                            flags_by_operation["model_prefill"] = {
                                "int8": False,
                                "sparse_2to4": False,
                            }

                        if config.execution.compile:
                            torch.compiler.reset()
                            benchmark_model = torch.compile(model, mode=config.execution.compile_mode)
                        decode_name = trace_names["model_decode"]
                        try:
                            operators = profile_model_decode_operators(
                                benchmark_model,
                                baseline_token_source,
                                use_cache=config.performance.use_kv_cache,
                            )
                            operator_traces[decode_name] = operators
                            flags_by_operation["model_decode"] = detect_kernel_flags(operators)
                        except Exception as exc:
                            operator_traces[decode_name] = [
                                f"PROFILE_{failure_status(exc).upper()}: {type(exc).__name__}: {exc}"
                            ]
                            flags_by_operation["model_decode"] = {
                                "int8": False,
                                "sparse_2to4": False,
                            }

                        if config.execution.compile:
                            torch.compiler.reset()
                            benchmark_model = torch.compile(model, mode=config.execution.compile_mode)
                        else:
                            benchmark_model = model
                        flags_by_operation["model_ttft"] = flags_by_operation["model_prefill"]
                        uses_sparse = {
                            operation: bool(
                                scenario.kind == "pruning_2to4"
                                and isinstance(checks, dict)
                                and checks.get("sparse_2to4")
                                and flags["sparse_2to4"]
                            )
                            for operation, flags in flags_by_operation.items()
                        }
                        uses_int8 = {
                            operation: bool(
                                scenario.kind == "quantization_int8_dynamic"
                                and isinstance(checks, dict)
                                and checks.get("int8_dynamic_kernel")
                                and flags["int8"]
                            )
                            for operation, flags in flags_by_operation.items()
                        }
                        performance, timings = benchmark_model_performance(
                            benchmark_model,
                            baseline_token_source,
                            config,
                            model_spec,
                            scenario,
                            metrics,
                            storage_bytes,
                            parameter_count,
                            weight_hash,
                            hidden_size,
                            num_attention_heads,
                            uses_sparse,
                            uses_int8,
                            {
                                operation: f"model_operators.json#{name}"
                                for operation, name in trace_names.items()
                            },
                        )
                    if scenario.kind == "baseline":
                        baseline_prompts_by_group[scenario.comparison_group] = produced
                        baseline_perplexity_by_group[scenario.comparison_group] = metrics["perplexity"]
                        _save_baseline_cache(
                            output,
                            model_id=model_spec.identifier,
                            scenario_id=scenario.identifier,
                            comparison_group=scenario.comparison_group,
                            perplexity=metrics["perplexity"],
                            prompts=produced,
                        )
                    if not already_completed:
                        quality_rows.append(quality_record)
                        window_rows.extend(windows)
                        prompt_rows.extend(prompt_metrics)
                        performance_rows.extend(performance)
                        timing_rows.extend(timings)
                        manifest["models"].append(
                            {"model_id": model_spec.identifier, "scenario": scenario.identifier, "status": "complete"}
                        )
                        completed.add(scenario_key)
                        new_scenarios += 1
                except Exception as exc:
                    if already_completed and scenario.kind == "baseline":
                        raise ExperimentEnvironmentError(
                            f"nao foi possivel reconstruir baseline {scenario.identifier} na retomada"
                        ) from exc
                    status = failure_status(exc)
                    quality_rows.append(
                        {
                            **{column: "" for column in QUALITY_COLUMNS},
                            "experiment_id": config.experiment_id,
                            "model_id": model_spec.identifier,
                            "model_revision": model_spec.revision,
                            "scenario": scenario.identifier,
                            "comparison_group": scenario.comparison_group,
                            "optimization_scope": scenario.optimization_scope,
                            "dtype": scenario.dtype,
                            "status": status,
                            "skip_reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    manifest["models"].append(
                        {
                            "model_id": model_spec.identifier,
                            "scenario": scenario.identifier,
                            "status": status,
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    completed.add(scenario_key)
                    new_scenarios += 1
                finally:
                    benchmark_model = None
                    if model is not None:
                        del model
                    if config.execution.compile:
                        torch.compiler.reset()
                    gc.collect()
                    torch.cuda.empty_cache()
                _write_model_checkpoint(
                    output,
                    manifest,
                    subset_records,
                    quality_rows,
                    window_rows,
                    prompt_rows,
                    performance_rows,
                    timing_rows,
                    operator_traces,
                )
                manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                if max_new_scenarios is not None and new_scenarios >= max_new_scenarios:
                    manifest["status"] = "running"
                    manifest["last_partial_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
                    manifest_path.write_text(
                        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    return manifest_path

        _write_model_checkpoint(
            output,
            manifest,
            subset_records,
            quality_rows,
            window_rows,
            prompt_rows,
            performance_rows,
            timing_rows,
            operator_traces,
        )
        manifest["status"] = "complete"
        manifest["completed_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Avalia modelos OPT pre-treinados sem treinamento.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--prefetch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-code-hash-migration", action="store_true")
    parser.add_argument("--max-new-scenarios", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_model_config(args.config)
        if args.prefetch:
            prefetch_models(config)
            print("modelos e dataset armazenados no cache")
            return 0
        if not args.output_dir:
            raise ExperimentConfigurationError("--output-dir e obrigatorio sem --prefetch")
        manifest = execute_model_experiment(
            config,
            args.output_dir,
            resume=args.resume,
            allow_code_hash_migration=args.allow_code_hash_migration,
            max_new_scenarios=args.max_new_scenarios,
        )
    except (ExperimentConfigurationError, ExperimentEnvironmentError, FileExistsError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"falha no benchmark OPT: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
