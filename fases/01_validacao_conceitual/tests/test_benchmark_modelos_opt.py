from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from validacao.benchmark_modelos_opt import (
    PROMPT_COLUMNS,
    QUALITY_COLUMNS,
    TIMING_COLUMNS,
    WINDOW_COLUMNS,
    ModelScenario,
    apply_model_scenario,
    build_evaluation_subset,
    evaluate_quality,
    failure_status,
    load_model_config,
    load_prompts,
    model_weight_hash,
    observed_model_sparsity,
    target_linears,
)
from validacao.analise_benchmark_modelos_opt import (
    ModelAnalysisError,
    load_model_experiment,
    write_model_analysis,
)
from validacao.hardware_probe import detect_kernel_flags
from validacao.protocolo import CSV_COLUMNS_V3, empty_benchmark_record_v3


PHASE_ROOT = Path(__file__).resolve().parents[1]


class TinySelfAttention(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dimension, dimension)
        self.k_proj = nn.Linear(dimension, dimension)
        self.v_proj = nn.Linear(dimension, dimension)
        self.out_proj = nn.Linear(dimension, dimension)


class TinyLayer(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.self_attn = TinySelfAttention(dimension)
        self.fc1 = nn.Linear(dimension, 2 * dimension)
        self.fc2 = nn.Linear(2 * dimension, dimension)


class TinyOPTStructure(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.decoder = nn.Module()
        self.model.decoder.embed_tokens = nn.Embedding(32, 8)
        self.model.decoder.layers = nn.ModuleList([TinyLayer(8)])
        self.lm_head = nn.Linear(8, 32, bias=False)


class TinyTokenizer:
    eos_token_id = 2

    def __call__(self, text: str, return_tensors: str | None = None, **_: object):
        values = [3 + (ord(character) % 20) for character in text][:8] or [3]
        tensor = torch.tensor([values], dtype=torch.long)
        if return_tensors == "pt":
            return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}
        return {"input_ids": values}


def test_opt_config_pins_models_dataset_and_protocol() -> None:
    config = load_model_config(PHASE_ROOT / "experimentos" / "modelos_opt_v1.json")

    assert [model.identifier for model in config.models] == [
        "facebook/opt-125m",
        "facebook/opt-350m",
        "facebook/opt-1.3b",
    ]
    assert all(len(model.revision) == 40 for model in config.models)
    assert config.dataset.revision == "f776294184f13b8ff2337b3841cf9269a6216d1e"
    assert config.quality.windows == 64
    assert config.performance.use_kv_cache

    altered = config.to_dict()
    altered["models"][0]["revision"] = "main"
    from validacao.benchmark_modelos_opt import model_config_from_mapping

    with pytest.raises(ValueError, match="SHA-1"):
        model_config_from_mapping(altered)


def test_opt_layer_selection_excludes_embeddings_and_lm_head() -> None:
    model = TinyOPTStructure()
    attention = target_linears(model, "attention_only")
    blocks = target_linears(model, "transformer_blocks")

    assert len(attention) == 4
    assert {name.rsplit(".", 1)[-1] for name, _ in attention} == {
        "q_proj", "k_proj", "v_proj", "out_proj"
    }
    assert len(blocks) == 6
    assert all("embed" not in name and not name.endswith("lm_head") for name, _ in blocks)


def test_observed_model_sparsity_is_reported_only_for_pruning() -> None:
    model = TinyOPTStructure()
    with torch.no_grad():
        model.model.decoder.layers[0].self_attn.q_proj.weight.zero_()

    assert observed_model_sparsity(model, "attention_only", "baseline") == 0.0
    assert observed_model_sparsity(
        model,
        "attention_only",
        "quantization_int8_dynamic",
    ) == 0.0
    magnitude_sparsity = observed_model_sparsity(
        model,
        "attention_only",
        "pruning_magnitude",
    )
    assert 0.0 < magnitude_sparsity < 1.0
    assert observed_model_sparsity(
        model,
        "attention_only",
        "pruning_2to4",
    ) == 0.5


def test_model_weight_hash_supports_bfloat16_bytes() -> None:
    model = TinyOPTStructure().to(dtype=torch.bfloat16)

    first = model_weight_hash(model)
    second = model_weight_hash(model)

    assert len(first) == 64
    assert first == second


def test_opt_pruning_changes_only_selected_layers_and_is_exact() -> None:
    model = TinyOPTStructure()
    embedding_before = model.model.decoder.embed_tokens.weight.detach().clone()
    head_before = model.lm_head.weight.detach().clone()
    scenario = ModelScenario(
        "attention_pruning_50",
        "pruning_magnitude",
        "bf16",
        "bfloat16",
        "attention_only",
        0.5,
        False,
    )

    count = apply_model_scenario(model, scenario)

    assert count == 4
    for _, module in target_linears(model, "attention_only"):
        assert torch.count_nonzero(module.weight == 0).item() == module.weight.numel() // 2
    assert torch.equal(model.model.decoder.embed_tokens.weight, embedding_before)
    assert torch.equal(model.lm_head.weight, head_before)


def test_evaluation_subset_is_uniform_reproducible_and_hash_sensitive() -> None:
    first = build_evaluation_subset(range(200), windows=4, window_tokens=16, performance_token_count=64)
    second = build_evaluation_subset(range(200), windows=4, window_tokens=16, performance_token_count=64)
    changed = build_evaluation_subset(range(1, 201), windows=4, window_tokens=16, performance_token_count=64)

    assert [start for start, _ in first.windows] == [0, 61, 122, 184]
    assert first == second
    assert first.sha256 != changed.sha256
    assert len(first.performance_tokens) == 64


def test_tiny_opt_quality_smoke_without_download() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.OPTConfig(
        vocab_size=32,
        hidden_size=16,
        word_embed_proj_dim=16,
        ffn_dim=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=64,
        dropout=0.0,
        attention_dropout=0.0,
        bos_token_id=2,
        eos_token_id=2,
        pad_token_id=1,
    )
    model = transformers.OPTForCausalLM(config).eval()
    candidate = copy.deepcopy(model).eval()
    token_ids = [3 + (index % 29) for index in range(32)]
    subset = build_evaluation_subset(token_ids, windows=2, window_tokens=8)
    prompts = load_prompts(PHASE_ROOT / "experimentos" / "prompts_opt_v1.json")

    _, _, baselines, baseline_metrics = evaluate_quality(
        model,
        TinyTokenizer(),
        subset,
        prompts,
        2,
        model_id="tiny-opt",
        scenario_id="baseline",
        baseline_prompts=None,
    )
    windows, prompt_rows, _, candidate_metrics = evaluate_quality(
        candidate,
        TinyTokenizer(),
        subset,
        prompts,
        2,
        model_id="tiny-opt",
        scenario_id="candidate",
        baseline_prompts=baselines,
    )

    assert len(windows) == 2
    assert len(prompt_rows) == 20
    assert candidate_metrics["perplexity"] == pytest.approx(baseline_metrics["perplexity"])
    assert candidate_metrics["logits_mse"] == pytest.approx(0.0)
    assert candidate_metrics["logits_kl_divergence"] == pytest.approx(0.0, abs=1e-7)
    assert candidate_metrics["top1_agreement"] == 1.0
    assert candidate_metrics["generated_token_agreement"] == 1.0


def test_kernel_detection_requires_expected_operator_names() -> None:
    assert detect_kernel_flags(["aten::_int_mm"])["int8"]
    assert detect_kernel_flags(
        ["void cutlass::Kernel2<cutlass_80_tensorop_i16832gemm_s8_128x64_128x3_tn_align16>()"]
    )["int8"]
    assert detect_kernel_flags(["triton_poi_fused_cslt_sparse_mm"])["sparse_2to4"]
    assert not detect_kernel_flags(["aten::mm"])["int8"]
    assert not detect_kernel_flags(["aten::linear"])["sparse_2to4"]


def test_model_failure_status_distinguishes_incompatibility_from_failure() -> None:
    assert failure_status(RuntimeError("operation is not supported")) == "unsupported"
    assert failure_status(RuntimeError("self.size(0) needs to be greater than 16")) == "unsupported"
    assert failure_status(AssertionError("unexpected compiler state")) == "failed"


def _write_rows(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_model_evidence(root: Path) -> Path:
    root.mkdir()
    config = load_model_config(PHASE_ROOT / "experimentos" / "modelos_opt_v1.json")
    snapshot = config.to_dict()
    (root / "config.snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    quality_rows = []
    for scenario, kind, ratio in (
        ("baseline_bf16", "baseline", 0.0),
        ("attention_pruning_10", "pruning", 0.01),
    ):
        row = {column: "" for column in QUALITY_COLUMNS}
        row.update(
            {
                "experiment_id": config.experiment_id,
                "model_id": config.models[0].identifier,
                "model_revision": config.models[0].revision,
                "scenario": scenario,
                "comparison_group": "bf16",
                "optimization_scope": "none" if kind == "baseline" else "attention_only",
                "dtype": "bfloat16",
                "status": "complete",
                "parameter_count": 10,
                "parameter_storage_bytes": 20,
                "target_linear_count": 0 if kind == "baseline" else 4,
                "pruning_sparsity": 0.0 if kind == "baseline" else 0.1,
                "observed_sparsity": 0.0 if kind == "baseline" else 0.1,
                "mean_loss": 2.0,
                "perplexity": 7.4,
                "perplexity_increase_ratio": ratio,
                "logits_mse": ratio,
                "logits_cosine": 1.0 - ratio,
                "logits_kl_divergence": ratio,
                "top1_agreement": 1.0,
                "generated_token_agreement": 1.0,
                "exact_generation_rate": 1.0,
                "quality_acceptable": True,
                "weight_sha256": "a" * 64,
                "evaluation_subset_sha256": "b" * 64,
            }
        )
        quality_rows.append(row)
    performance_rows = []
    for case_id, baseline_id, scenario, latency, int8 in (
        ("base", "base", "baseline_bf16", 2.0, False),
        ("candidate", "base", "attention_int8_weight_only", 1.5, False),
    ):
        performance_rows.append(
            empty_benchmark_record_v3(
                experiment_id=config.experiment_id,
                stage="pretrained_hardware",
                case_id=case_id,
                baseline_case_id=baseline_id,
                data_seed=2026,
                repeat_index=1,
                profile_id="prefill_b1_l64",
                model_id=config.models[0].identifier,
                model_revision=config.models[0].revision,
                operation="model_prefill",
                optimization_scope="none" if case_id == "base" else "attention_only",
                scenario=scenario,
                comparison_group="bf16",
                status="complete",
                batch_size=1,
                seq_len=64,
                d_model=768,
                num_heads=12,
                dtype="bfloat16",
                backend="test",
                compile_mode="test",
                input_distribution="tokens",
                pruning_method="none",
                pruning_sparsity=0.0,
                quantization_method="none" if case_id == "base" else "weight_only",
                weight_storage_dtype="bfloat16" if case_id == "base" else "int8",
                activation_dtype="bfloat16",
                observed_sparsity=0.0,
                uses_sparse_kernel=False,
                uses_int8_kernel=int8,
                latency_ms_mean=latency,
                latency_ms_p50=latency,
                latency_ms_p95=latency,
                throughput_tokens_s=64_000 / latency,
                peak_memory_allocated_bytes=100,
                peak_memory_reserved_bytes=120,
                parameter_storage_bytes=20 if case_id == "base" else 12,
                theoretical_flops=10,
                mse=0.0,
                mae=0.0,
                r2=1.0,
                cosine_similarity=1.0,
                kl_divergence=0.0,
                top1_agreement=1.0,
                loss=2.0,
                perplexity=7.4,
                input_sha256="c" * 64,
                weight_sha256="d" * 64,
                output_sha256="e" * 64,
            )
        )
    timings = [
        {
            "case_id": case_id,
            "model_id": config.models[0].identifier,
            "scenario": scenario,
            "operation": "model_prefill",
            "repeat_index": 1,
            "iteration": iteration,
            "latency_ms": latency,
        }
        for case_id, scenario, latency in (
            ("base", "baseline_bf16", 2.0),
            ("candidate", "attention_int8_weight_only", 1.5),
        )
        for iteration in range(1, 11)
    ]
    files = {
        "quality.csv": (QUALITY_COLUMNS, quality_rows),
        "quality_windows.csv": (WINDOW_COLUMNS, []),
        "prompt_metrics.csv": (PROMPT_COLUMNS, []),
        "performance.csv": (CSV_COLUMNS_V3, performance_rows),
        "timings.csv": (TIMING_COLUMNS, timings),
    }
    artifacts = []
    for name, (columns, rows) in files.items():
        path = root / name
        _write_rows(path, columns, rows)
        artifacts.append({"path": name, "sha256": _digest(path), "records": len(rows)})
    subset = root / "evaluation_subset.json"
    subset.write_text("{}\n", encoding="utf-8")
    artifacts.append({"path": subset.name, "sha256": _digest(subset), "records": 0})
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": config.experiment_id,
                "status": "complete",
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return root


def test_model_analysis_checks_integrity_pairs_baselines_and_writes_manifest(tmp_path: Path) -> None:
    evidence = _fake_model_evidence(tmp_path / "evidence")
    loaded = load_model_experiment(evidence)
    outputs = write_model_analysis(evidence)

    assert len(loaded.comparisons) == 1
    assert loaded.comparisons.iloc[0]["speedup_vs_baseline"] == pytest.approx(4 / 3)
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs.paths())
    with outputs.cases.open(encoding="utf-8", newline="") as handle:
        evidence_types = {row["evidence_type"] for row in csv.DictReader(handle)}
    assert evidence_types == {"quality", "performance"}

    with (evidence / "quality.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ModelAnalysisError, match="checksum"):
        load_model_experiment(evidence)


@pytest.mark.huggingface_download
def test_pinned_opt_revision_is_reachable_when_explicitly_enabled() -> None:
    if os.environ.get("RUN_HF_INTEGRATION") != "1":
        pytest.skip("defina RUN_HF_INTEGRATION=1 para teste real com download")
    transformers = pytest.importorskip("transformers")
    config = load_model_config(PHASE_ROOT / "experimentos" / "modelos_opt_v1.json")
    remote = transformers.AutoConfig.from_pretrained(
        config.models[0].identifier,
        revision=config.models[0].revision,
    )
    assert remote.model_type == "opt"
