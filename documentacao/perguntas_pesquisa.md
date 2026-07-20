# Perguntas de pesquisa e criterios de validade

Este documento formaliza o que o estudo pretende responder e quais evidencias
sao necessarias para uma conclusao ser aceita no artigo.

## Perguntas de pesquisa

| ID | Pergunta | Metricas associadas | Evidencia minima |
| --- | --- | --- | --- |
| RQ1 | Quantizacao int8 preserva melhor a saida do bloco Transformer simplificado do que pruning por magnitude? | MSE, MAE, R2, similaridade de cosseno | Saida baseline, saida candidata e metricas de erro. |
| RQ2 | Quantizacao e pruning reduzem memoria teorica ou real no ambiente do estudo? | `max_memory_bytes` | Ambiente registrado, medicao ou estimativa de memoria e nivel de validade. |
| RQ3 | Alguma tecnica reduz latencia ou aumenta throughput sem degradacao excessiva da saida? | Latencia media, p50, p95, throughput, MSE e cosseno | Warmup, repeticoes, sincronizacao em CUDA e baseline comparavel. |
| RQ4 | Quando uma tecnica e apenas conceitual/numerica e quando pode ser chamada de hardware? | `validity_level` | Matriz de validacao, dtype/formato e kernel/caminho de execucao. |

## Hipoteses

| ID | Pergunta | Hipotese | Confirma se | Rejeita se |
| --- | --- | --- | --- | --- |
| H1 | RQ1 | Quantizacao int8 tera menor erro numerico do que pruning por magnitude no bloco controlado. | MSE e MAE menores, R2 e cosseno maiores que pruning na mesma configuracao. | Pruning empata ou supera quantizacao nas metricas de fidelidade. |
| H2 | RQ2 | Quantizacao int8 reduz memoria representacional; pruning por mascara densa nao prova reducao real de hardware. | Quantizacao apresenta menor armazenamento observado/estimado e pruning sem sparse kernel fica sem nivel hardware. | Quantizacao nao reduz armazenamento ou pruning demonstra formato/kernel esparso exploravel. |
| H3 | RQ3 | Ganhos de latencia/throughput dependem do caminho real de execucao, nao apenas da tecnica declarada. | Cenarios sem kernel/formato compativel nao sao promovidos a hardware mesmo que tenham erro aceitavel. | Ha evidencia de kernel/formato otimizado e medicao consistente de ganho fisico. |
| H4 | RQ4 | A matriz de validacao separa corretamente conclusoes conceituais, numericas, algoritmicas e de hardware. | Toda conclusao aponta para registro CSV e linha conceitual da matriz. | Alguma conclusao exige evidencia que nao esta registrada. |

## Regra de conclusao

Uma conclusao so pode entrar no texto final se ela apontar para:

1. uma pergunta de pesquisa;
2. uma hipotese ou criterio de verificacao;
3. uma linha da matriz de validacao;
4. um resultado CSV ou teste deterministico;
5. um nivel de validade coerente com a evidencia observada.

Essa regra e implementada no codigo por `validacao.pesquisa`,
`validacao.benchmark_controlado` e `validacao.analise_resultados`.
