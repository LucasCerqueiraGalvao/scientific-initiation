# Decisão metodológica: inicialização dos pesos do benchmark

## Problema observado

O diagnóstico CPU v1 usou entradas e pesos independentes com distribuição
`N(0,1)`. Para uma projeção com dimensão `D`, somar `D` produtos com pesos de
variância 1 faz a variância da saída crescer com a dimensão. Na self-attention,
esse efeito se repete nas projeções Q/K/V, nos scores e na projeção de saída.

O caso `B=1`, `L=64`, `D=128` mostrou o risco: a quantização ainda foi melhor
que o pruning nas métricas relativas, mas o MSE absoluto da self-attention foi
`732.65`. Esse valor não era comparável de forma limpa com dimensões menores,
porque misturava erro da técnica e aumento da escala do baseline.

## Alternativas consideradas

1. **Manter `N(0,1)` e interpretar cada dimensão separadamente.** Preservaria o
   primeiro gerador, mas enfraqueceria MSE/MAE como métricas da grade.
2. **Normalizar apenas MSE/MAE.** Acrescentaria novas métricas e mudaria o schema
   do benchmark, contrariando a decisão de preservar a interface.
3. **Escalar os pesos-base pela dimensão.** Mantém as métricas, controla a
   variância das projeções e continua usando exatamente os mesmos pesos entre
   cenários da mesma configuração.

A alternativa 3 foi escolhida. É uma decisão metodológica posterior à reunião,
não um pedido literal do professor.

## Regra operacional adotada

- entradas sintéticas: `N(0,1)`;
- matrizes quadradas: Xavier normal, com
  `desvio = sqrt(2 / (fan_in + fan_out)) = 1/sqrt(D)`;
- bias da projeção densa: zero;
- mesma seed `42`, dados e pesos-base para baseline, pruning e quantização;
- parâmetros registrados no schema de configuração v2 e nos metadados;
- evidências de schema v1 permanecem legíveis e versionadas.

## Confronto v1 versus v2

As duas coletas abaixo usam `B=1`, `L=64`, `D=128`, 20 warm-ups, 50 medições e
duas execuções. A única mudança metodológica relevante para a qualidade é a
inicialização.

| Operação/cenário | Métrica | v1 — pesos N(0,1) | v2 — Xavier | Leitura |
| --- | --- | ---: | ---: | --- |
| Projeção/quantização | MSE | 0.00995697 | 0.000077789 | Escala absoluta controlada. |
| Projeção/pruning | MSE | 9.30825 | 0.0727207 | Escala absoluta controlada. |
| Self-attention/quantização | MSE | 732.65 | 0.0000163925 | O caso deixa de ser dominado pela explosão de escala. |
| Self-attention/quantização | R² | 0.95428 | 0.999647 | Fidelidade relativa também melhora com scores não saturados. |
| Self-attention/quantização | Cosseno | 0.977141 | 0.999824 | Direção da saída quase preservada em v2. |
| Self-attention/pruning | MSE | 12144.35 | 0.0110127 | Pruning continua claramente mais destrutivo. |

Não se comparam as latências de v1 e v2 para concluir desempenho: foram coletas
CPU separadas, os caminhos não usam kernels esparsos/INT8 e a variabilidade
observada foi alta.

## Decisão para a coleta principal

O schema v2 com Xavier/bias zero é o contrato vigente. A v1 não foi apagada nem
reclassificada; ela documenta a ameaça à validade que motivou a correção. A
conclusão sobre H1 continua preliminar até a grade completa, mas agora MSE/MAE
têm uma escala experimental mais controlada entre dimensões.
