# Estrutura final proposta do artigo/relatório

## 1. Introdução

Entra a pergunta central: quando compressão numérica e estrutural em
Transformers se converte em benefício físico real de memória e desempenho.
Incluir motivação, problema e contribuição.

## 2. Fundamentação

Explicar self-attention, projeções Q/K/V/O, quantização, pruning denso,
weight-only, INT8 dynamic e sparsity 2:4.

## 3. Trabalhos relacionados

Relacionar Transformers, pruning, quantização, LLM.int8, TorchAO, cuSPARSELt,
Wanda e SparseGPT como trabalhos que motivam os métodos e limites.

## 4. Metodologia

Descrever níveis A/B/C, validação conceitual, benchmarks sintéticos,
benchmarks físicos, modelos OPT, qualidade, desempenho, bootstrap e critérios
de aceitabilidade.

Tabela indicada: resumo do protocolo experimental.

## 5. Ambiente experimental

Detalhar GPU, VRAM, Docker/WSL2, CUDA, PyTorch, TorchAO, Transformers, Triton,
cuSPARSELt, cache Hugging Face, guards de GPU e limitações de clocks.

Tabela indicada: ambiente e versões.

## 6. Validação conceitual

Entram as 18 comparações, frameworks usados e erro máximo.

Tabela indicada: validações conceituais aprovadas.

## 7. Benchmarks sintéticos

Entram `dense_projection`, `multi_head_self_attention`, seeds, perfis,
distribuições, INT8 fake e pruning denso.

Gráficos indicados: qualidade por sparsity; speedup sintético apenas como
evidência de que não há conclusão física.

## 8. Benchmarks físicos

Entram hardware nativo, stress e crossover. Separar compressão, memória,
latência e kernel.

Gráficos indicados: microbenchmark/crossover; memória relativa; Pareto físico
quando metodologicamente adequado.

## 9. Avaliação em OPT

Entram OPT-125M, 350M, 1.3B, 2.7B e 6.7B. Mostrar qualidade, speedup,
armazenamento e VRAM por técnica.

Gráficos indicados: speedup por tamanho de OPT; Delta PPL versus speedup; VRAM
versus Delta PPL; comparação de qualidade no 6.7B.

## 10. Sensibilidade e híbridos

Entram sensibilidade por componente/layer em 350M e 1.3B, híbridos 1.3B e
híbridos 6.7B.

Gráficos indicados: sensibilidade por componente/layer; híbridos 1.3B; qualidade
dos híbridos 6.7B.

## 11. Discussão

Organizar por hipóteses H1-H12. Destacar resultados negativos como evidência.

Tabela indicada: matriz hipótese -> experimento -> métrica -> evidência ->
conclusão -> limitação.

## 12. Limitações

Entram uma única GPU, família OPT, dataset estreito, 20 prompts, greedy
decoding, clocks não fixados, backend weight-only específico, TTFT histórico e
espaço híbrido não exaustivo.

## 13. Conclusão

Retomar a tese central: compressão não implica aceleração, kernels importam,
qualidade muda a interpretação do speedup e políticas seletivas não se
transferiram diretamente para escala maior.

## 14. Trabalhos futuros

Apresentar naturalmente: kernels weight-only mais especializados, Wanda como
melhoria controlada para 2:4, SparseGPT como alternativa mais custosa,
quantização com tratamento de outliers, outras GPUs, outras famílias de LLM,
quantização sensível à camada e calibração/hybrid precision mais sofisticada.
