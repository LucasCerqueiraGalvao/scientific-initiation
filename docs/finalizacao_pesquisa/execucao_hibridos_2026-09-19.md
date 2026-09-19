# Execução híbrida de 19/09/2026

## Identidade da execução

- Objetivo: testar se seletor fino por camada/componente permite configurações INT8 dynamic que preservem qualidade e melhorem desempenho.
- Commits de infraestrutura/configuração:
  - `1f4840082383f36bdc5ac92767884b701ef49710`: seletores híbridos e config inicial 1.3B.
  - `c9ba71b`: correção do runner guardado.
  - `6fc2ded`: config de finalistas 1.3B.
  - `467d401`: config de qualidade 6.7B.
- GPU: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB.
- Guarda de VRAM: 15.800 MiB.
- Ambiente: PyTorch 2.11.0+cu128, TorchAO 0.17.0, Transformers 5.17.0, datasets 5.0.1.

## Runs

| Run | Diretório promovido | Status |
| --- | --- | --- |
| `modelos-opt-hibridos-1-3b-quality-20260919` | `modelos_opt_hibridos_1_3b_quality_2026-09-19` | completo |
| `modelos-opt-hibridos-1-3b-finalistas-20260919` | `modelos_opt_hibridos_1_3b_finalistas_2026-09-19` | completo |
| `modelos-opt-hibridos-6-7b-quality-20260919` | `modelos_opt_hibridos_6_7b_quality_2026-09-19` | completo |

## Triagem OPT-1.3B

| Cenário | Lineares modificadas | Fração aprox. dos parâmetros elegíveis | ΔPPL | Oficial |
| --- | ---: | ---: | ---: | --- |
| `int8dyn_blocks_all_reference` | 144 | 100,0% | 2,340% | reprovado |
| `hybrid_mlp_all_attention_bf16` | 48 | 66,7% | -1,098% | aprovado |
| `hybrid_mlp_all_attention_0_7` | 80 | 77,8% | -0,815% | aprovado |
| `hybrid_mlp_all_attention_0_15` | 112 | 88,9% | -0,370% | aprovado |
| `hybrid_attention_0_15_only` | 64 | 22,2% | -0,102% | aprovado |
| `hybrid_mlp_0_15_attention_bf16` | 32 | 44,4% | -1,505% | aprovado |
| `control_attention_16_23_mlp_0_7` | 48 | 33,3% | 2,187% | reprovado |

Interpretação: preservar a atenção final foi suficiente para recuperar qualidade no OPT-1.3B. O controle que quantiza `attention 16-23` falhou, reforçando o mapa de sensibilidade anterior.

## Performance OPT-1.3B

| Cenário | ΔPPL | Prefill mediano | IC 95% | Prompt-forward mediano | IC 95% |
| --- | ---: | ---: | ---: | ---: | ---: |
| `int8dyn_blocks_all_reference` | 2,340% | 1,035x | 0,794-1,269 | 1,078x | 0,817-1,325 |
| `hybrid_mlp_all_attention_bf16` | -1,098% | 1,110x | 0,880-1,309 | 1,105x | 0,937-1,326 |
| `hybrid_mlp_all_attention_0_7` | -0,815% | 1,131x | 0,872-1,299 | 1,111x | 0,889-1,317 |
| `hybrid_mlp_all_attention_0_15` | -0,370% | 1,108x | 0,841-1,310 | 1,064x | 0,809-1,311 |
| `hybrid_attention_0_15_only` | -0,102% | 0,964x | 0,865-0,990 | 0,935x | 0,866-0,991 |
| `hybrid_mlp_0_15_attention_bf16` | -1,505% | 1,100x | 0,861-1,178 | 1,064x | 0,929-1,179 |

Nenhum híbrido atingiu ganho confirmado pelo critério rígido, porque os intervalos cruzam `1,0x`. Mesmo assim, MLP-only e híbridos com MLP tiveram medianas positivas, enquanto atenção isolada 0-15 regrediu.

## Validação OPT-6.7B

| Cenário | Lineares modificadas | Fração aprox. dos parâmetros elegíveis | ΔPPL | Oficial |
| --- | ---: | ---: | ---: | --- |
| `int8dyn_blocks_all_reference` | 192 | 100,0% | 165,052% | reprovado |
| `hybrid_mlp_all_attention_bf16` | 64 | 66,7% | 207,254% | reprovado |
| `hybrid_mlp_all_attention_0_7` | 96 | 75,0% | 160,236% | reprovado |
| `hybrid_mlp_all_attention_0_15` | 128 | 83,3% | 157,271% | reprovado |
| `hybrid_mlp_0_15_attention_bf16` | 32 | 33,3% | 197,073% | reprovado |

Interpretação: o padrão observado no OPT-1.3B não escalou para o OPT-6.7B. A técnica INT8 dynamic via TorchAO degradou a perplexidade de modo severo mesmo em recortes híbridos simples. Por isso, a etapa de performance 6.7B foi cancelada: nenhum candidato passou o critério oficial de qualidade.

## Conclusão

- Híbridos são metodologicamente úteis: no OPT-1.3B eles recuperam qualidade que o INT8 amplo perde.
- O ganho de performance em 1.3B é promissor, mas ainda não confirmado estatisticamente.
- O OPT-6.7B contradiz a extrapolação simples de que o mesmo híbrido tolerante em 1.3B será tolerante no modelo maior.
- A próxima investigação útil não é ampliar grade no 6.7B; é entender por que o INT8 dynamic do backend atual degrada tão fortemente o 6.7B e testar, se houver tempo, uma alternativa de quantização calibrada/controlada.
