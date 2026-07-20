from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchQuestion:
    identifier: str
    question: str
    metrics: tuple[str, ...]
    evidence_required: tuple[str, ...]


@dataclass(frozen=True)
class Hypothesis:
    identifier: str
    research_question_id: str
    statement: str
    confirmation_criterion: str
    rejection_criterion: str


RESEARCH_QUESTIONS = (
    ResearchQuestion(
        identifier="RQ1",
        question="Quantizacao int8 preserva melhor a saida do bloco Transformer simplificado do que pruning por magnitude?",
        metrics=("mse", "mae", "r2", "cosine_similarity"),
        evidence_required=("baseline_output", "candidate_output", "metricas_de_erro"),
    ),
    ResearchQuestion(
        identifier="RQ2",
        question="Quantizacao e pruning reduzem memoria teorica ou real no ambiente do estudo?",
        metrics=("max_memory_bytes",),
        evidence_required=("registro_de_ambiente", "medicao_ou_estimativa_de_memoria", "validity_level"),
    ),
    ResearchQuestion(
        identifier="RQ3",
        question="Alguma tecnica reduz latencia ou aumenta throughput sem degradacao excessiva da saida?",
        metrics=("latency_ms_mean", "latency_ms_p50", "latency_ms_p95", "throughput_tokens_s", "mse", "cosine_similarity"),
        evidence_required=("warmup", "repeticoes", "sincronizacao_quando_cuda", "baseline_comparavel"),
    ),
    ResearchQuestion(
        identifier="RQ4",
        question="Quando uma tecnica e apenas conceitual/numerica e quando pode ser chamada de hardware?",
        metrics=("validity_level",),
        evidence_required=("matriz_de_validacao", "dtype_ou_formato", "kernel_ou_caminho_de_execucao"),
    ),
)

HYPOTHESES = (
    Hypothesis(
        identifier="H1",
        research_question_id="RQ1",
        statement="Quantizacao int8 tera menor erro numerico do que pruning por magnitude no bloco controlado.",
        confirmation_criterion="MSE e MAE menores, R2 e cosseno maiores que pruning na mesma configuracao.",
        rejection_criterion="Pruning empata ou supera quantizacao nas metricas de fidelidade da saida.",
    ),
    Hypothesis(
        identifier="H2",
        research_question_id="RQ2",
        statement="Quantizacao int8 reduz memoria representacional; pruning por mascara densa nao prova reducao real de hardware.",
        confirmation_criterion="Quantizacao apresenta menor armazenamento observado/estimado e pruning sem sparse kernel fica sem nivel hardware.",
        rejection_criterion="Quantizacao nao reduz armazenamento ou pruning demonstra formato/kernel esparso exploravel.",
    ),
    Hypothesis(
        identifier="H3",
        research_question_id="RQ3",
        statement="Ganhos de latencia/throughput dependem do caminho real de execucao, nao apenas da tecnica declarada.",
        confirmation_criterion="Cenarios sem kernel/formato compativel nao sao promovidos a hardware mesmo que tenham erro aceitavel.",
        rejection_criterion="Ha evidencia de kernel/formato otimizado e medicao consistente de ganho fisico.",
    ),
    Hypothesis(
        identifier="H4",
        research_question_id="RQ4",
        statement="A matriz de validacao separa corretamente conclusoes conceituais, numericas, algoritmicas e de hardware.",
        confirmation_criterion="Toda conclusao aponta para registro CSV e linha conceitual da matriz.",
        rejection_criterion="Alguma conclusao exige evidencia que nao esta registrada.",
    ),
)


def validate_research_contract(
    questions: tuple[ResearchQuestion, ...] = RESEARCH_QUESTIONS,
    hypotheses: tuple[Hypothesis, ...] = HYPOTHESES,
) -> tuple[str, ...]:
    issues: list[str] = []
    question_ids = {question.identifier for question in questions}

    for question in questions:
        if not question.metrics:
            issues.append(f"{question.identifier}: pergunta sem metrica")
        if not question.evidence_required:
            issues.append(f"{question.identifier}: pergunta sem evidencia exigida")

    for hypothesis in hypotheses:
        if hypothesis.research_question_id not in question_ids:
            issues.append(f"{hypothesis.identifier}: RQ inexistente {hypothesis.research_question_id}")
        if not hypothesis.confirmation_criterion or not hypothesis.rejection_criterion:
            issues.append(f"{hypothesis.identifier}: criterio incompleto")

    return tuple(issues)
