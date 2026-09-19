# Execução complementar de 19/09/2026

## Identidade da execução

- Commit usado: `04ee9796b8ead062a277859e2f186ebbc33f0297`.
- Branch: `main`, alinhada com `origin/main`.
- Worktree no preflight: limpa quanto a arquivos rastreados; temporários não rastreados preservados fora da evidência (`.tmp/`, `.playwright-mcp/`, `*.inspect.ndjson`).
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB, compute capability 8.9.
- Host: driver NVIDIA 581.80, CUDA reportado pelo driver 13.0.
- Contêiner: Linux/WSL2, Python 3.12.3, PyTorch 2.11.0+cu128, CUDA build 12.8, TorchAO 0.17.0, Transformers 5.17.0, datasets 5.0.1, accelerate 1.15.0, Triton 3.6.0, cuSPARSELt 0.7.1, cuBLAS 12.8.4.1, cuDNN 9.19.0.56.
- Observação de ambiente: TorchAO emitiu avisos sobre extensões opcionais CUTLASS/MXFP8 não carregadas, mas os caminhos usados pela bateria executaram e os kernels relevantes foram validados pelos campos de profiler.

## Runs

| Run | Diretório promovido | Duração aproximada | Status |
| --- | --- | ---: | --- |
| `crossover-20260919` | `hardware_crossover_v3_2026-09-19` | 11 min 16 s | completo |
| `sensibilidade-350m-20260919` | `modelos_opt_sensibilidade_350m_2026-09-19` | 14 min 04 s | completo |
| `sensibilidade-1-3b-20260919` | `modelos_opt_sensibilidade_1_3b_2026-09-19` | 15 min 20 s | completo |

Todos os runs foram copiados de `resultados/` para `fases/01_validacao_conceitual/evidencias/benchmarks/` sem sobrescrever evidências antigas.

## Validação do Crossover

- Registros completos: 1.200.
- Timings: 60.000.
- Comparações pareadas: 720.
- Casos não executados: 0.
- Status: 1.200 `complete`.
- NaN/Inf nas métricas principais: 0.
- Perfis: sequência L64/L128/L256/L512/L1024, batch B4/B8 e largura D1024/H16.
- Kernels: INT8 dynamic confirmou `uses_int8_kernel=True`; 2:4 confirmou `uses_sparse_kernel=True`; weight-only não confirmou kernel de computação, como esperado.

Resultado agregado:

| Operação | Técnica | Speedup mediano | IC 95% | Kernel | Conclusão |
| --- | --- | ---: | ---: | --- | --- |
| dense_projection | INT8 dynamic | 0,767x | 0,758-0,788 | confirmado | regressão |
| dense_projection | INT8 weight-only | 0,098x | 0,081-0,109 | não confirmado | regressão |
| dense_projection | 2:4 | 0,834x | 0,817-0,843 | confirmado | regressão |
| multi_head_self_attention | INT8 dynamic | 0,641x | 0,627-0,665 | confirmado | regressão |
| multi_head_self_attention | INT8 weight-only | 0,078x | 0,070-0,086 | não confirmado | regressão |
| multi_head_self_attention | 2:4 | 0,713x | 0,627-0,781 | confirmado | regressão |

Pontos locais:

- Nenhuma técnica atingiu ganho confirmado (`speedup >= 1,05x`, IC 95% acima de 1,0x e kernel confirmado).
- O melhor sinal local foi `multi_head_self_attention` com 2:4 em `cross_batch_b8`: mediana `1,021x`, IC `0,746-1,383`; classificado como ganho provável, não confirmado.
- INT8 dynamic permaneceu abaixo de 1,0x em todos os perfis medidos.
- INT8 weight-only permaneceu muito lento em todos os perfis e não demonstrou região de crossover.
- 2:4 teve kernel físico, mas não apresentou ganho sustentado; o batch maior sugere apenas uma região a investigar, não uma conclusão.

## Sensibilidade OPT-350M

Validação:

- Registros de qualidade: 9.
- Status: 9 `complete`.
- NaN/Inf: 0.
- Performance não foi medida por desenho experimental.

Mapa:

| Cenário | ΔPPL | KL | MSE | Top-1 | Token agreement | Oficial |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| attention all | -0,036% | 0,0176 | 0,4962 | 1,00 | 0,360 | aprovado |
| MLP all | 0,389% | 0,0080 | 0,1004 | 1,00 | 0,319 | aprovado |
| attention 0-7 | -0,350% | 0,0029 | 0,0356 | 1,00 | 0,591 | aprovado |
| attention 8-15 | 0,080% | 0,0040 | 0,0589 | 1,00 | 0,471 | aprovado |
| attention 16-23 | 0,550% | 0,0132 | 0,1931 | 1,00 | 0,445 | aprovado |
| MLP 0-7 | 0,298% | 0,0015 | 0,0239 | 1,00 | 0,626 | aprovado |
| MLP 8-15 | -0,243% | 0,0037 | 0,0635 | 1,00 | 0,541 | aprovado |
| MLP 16-23 | 0,884% | 0,0053 | 0,0315 | 1,00 | 0,442 | aprovado |

Interpretação: no OPT-350M, todas as regiões foram tolerantes pelo critério oficial. A região final é a mais sensível por ΔPPL, mas ainda dentro do limite de 2%.

## Sensibilidade OPT-1.3B

Validação:

- Registros de qualidade: 9.
- Status: 9 `complete`.
- NaN/Inf: 0.
- Performance não foi medida por desenho experimental.

Mapa:

| Cenário | ΔPPL | KL | MSE | Top-1 | Token agreement | Oficial |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| attention all | 4,113% | 0,0977 | 0,4398 | 0,80 | 0,106 | reprovado |
| MLP all | -1,098% | 0,0553 | 0,1539 | 0,95 | 0,186 | aprovado |
| attention 0-7 | -0,073% | 0,0219 | 0,0588 | 0,90 | 0,234 | aprovado |
| attention 8-15 | -0,201% | 0,0122 | 0,0304 | 0,90 | 0,486 | aprovado |
| attention 16-23 | 4,315% | 0,0492 | 0,3575 | 0,75 | 0,143 | reprovado |
| MLP 0-7 | -1,123% | 0,0432 | 0,1261 | 0,95 | 0,213 | aprovado |
| MLP 8-15 | -0,795% | 0,0111 | 0,0346 | 0,95 | 0,310 | aprovado |
| MLP 16-23 | 0,187% | 0,0078 | 0,0282 | 0,90 | 0,637 | aprovado |

Interpretação: no OPT-1.3B, a sensibilidade aparece claramente na atenção final. Quantizar toda a atenção falha, e a falha é explicada principalmente por `attention 16-23`. MLP é tolerante em todos os recortes testados.

## Comparação entre 350M e 1.3B

- O 350M é permissivo demais para descartar regiões sozinho: todos os cenários passaram.
- O 1.3B revela sensibilidade que não aparece no 350M, especialmente na atenção final.
- A tendência mais robusta é: MLP tolera INT8 dynamic melhor que atenção no modelo maior.
- Atenção inicial e intermediária permanecem aceitáveis no 1.3B.
- Atenção final deve ser preservada em BF16 nas próximas propostas híbridas.
- Token agreement cai bastante em quase todos os cenários quantizados, então deve continuar como métrica auxiliar, mesmo quando ΔPPL aprova.

## Hipóteses atualizadas

- A hipótese de que compressão física vira aceleração em operações isoladas ficou mais fraca: crossover físico não apareceu de forma sustentada.
- A hipótese de que weight-only melhora em workloads maiores ficou mais fraca neste backend: não houve região de compensação nos perfis medidos.
- A hipótese de que 2:4 pode ter uma região física promissora ficou parcialmente aberta pelo batch 8, mas sem confirmação estatística.
- A hipótese de híbridos ficou mais forte: há regiões claramente mais tolerantes, especialmente MLP e atenção não final no 1.3B.

## Descartes antes do 6.7B

Descartar para 6.7B nesta etapa:

- INT8 dynamic em atenção total.
- INT8 dynamic em atenção final isolada.
- Weight-only como candidato amplo de latência em operação isolada.
- 2:4 por magnitude como candidato de qualidade, salvo se for testado outro seletor de pesos.

Manter como candidatos:

- MLP INT8 dynamic com atenção BF16.
- MLP INT8 dynamic + atenção 0-15 INT8 dynamic + atenção final BF16.
- Atenção 0-7 INT8 dynamic + restante BF16.
- Atenção 8-15 INT8 dynamic + restante BF16.
- MLP 0-23 INT8 dynamic com atenção final preservada.

## Busca híbrida mínima proposta

Não executar ainda no 6.7B. Primeiro medir em OPT-1.3B, com qualidade antes de performance:

1. `hybrid_mlp_all_attention_bf16`.
2. `hybrid_attention_0_15_mlp_all_attention_16_23_bf16`.
3. `hybrid_attention_0_7_only`.
4. `hybrid_attention_8_15_only`.
5. `hybrid_mlp_0_15_attention_bf16`.
6. `hybrid_mlp_16_23_attention_bf16`.

Critério de avanço:

- Oficial: ΔPPL <= 2%.
- Exploratórios: ΔPPL <= 5% e <= 10% para análise, não para aprovação oficial.
- Rodar token agreement completo nas candidatas que passam qualidade.
- Rodar performance/VRAM somente em 2 a 4 finalistas.

Estimativa para 6.7B:

- 3 a 4 cenários de qualidade finalistas.
- 2 a 3 cenários com performance completa.
- Total recomendado: 5 a 7 runs reais no 6.7B, não uma grade ampla.

## Recomendações sobre 2:4 e weight-only

- Wanda/SparseGPT: vale considerar depois da busca híbrida INT8, porque 2:4 físico existe, mas magnitude simples degrada qualidade. Implementar no máximo um método calibrado e pequeno.
- Backend weight-only alternativo: não vale uma bateria ampla agora. O crossover sugere limitação forte do backend atual; testar outro backend só como microbenchmark isolado, se houver integração simples e profiler claro.
