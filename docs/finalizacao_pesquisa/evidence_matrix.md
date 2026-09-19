# Matriz de evidências

| Hipótese | Experimento | Métrica | Evidência | Conclusão | Limitação |
|---|---|---|---|---|---|
| H1 | Sintético v3, hardware v3, OPT | Speedup, MSE, PPL | Técnicas alteraram pesos/saídas e reduziram representação sem acelerar de forma automática. | Sustentada. | Uma GPU e backends específicos. |
| H2 | Todas as fases | Níveis A/B/C, kernel, storage, VRAM | INT8 fake, weight-only, INT8 dynamic e 2:4 ocuparam categorias diferentes. | Sustentada. | A classificação depende da instrumentação disponível. |
| H3 | Hardware v3, OPT físico | Profiler, speedup, IC 95% | Kernels físicos foram confirmados, mas o ganho variou por caminho e workload. | Sustentada. | Kernels não foram igualmente detalhados em todos os recortes. |
| H4 | `hardware_crossover_v3` | Speedup e IC 95% | INT8 dynamic, weight-only e 2:4 não atingiram mediana >=1.05x com IC acima de 1.0x. | Não sustentada. | Perfis mínimos, não grid exaustivo. |
| H5 | Crossover, OPT escala | Speedup por L/B/D e por modelo | O comportamento mudou com batch, sequência, largura e tamanho do modelo. | Parcialmente sustentada. | Influência observada não implica ganho útil. |
| H6 | Sintético v3, OPT pruning denso | MSE, PPL, VRAM, storage | Pruning por magnitude mostrou degradação controlável em baixa sparsity, mas não reduziu fisicamente VRAM nos caminhos densos. | Sustentada. | Não avalia pruning estruturado avançado nesse escopo. |
| H7 | Hardware 2:4, OPT 2:4 | Profiler, PPL, speedup | Sparse físico foi confirmado, mas magnitude 2:4 degradou qualidade e raramente sustentou ganho útil. | Parcialmente sustentada. | Não testa Wanda/SparseGPT. |
| H8 | OPT weight-only | VRAM, storage, PPL, speedup | Forte compressão e preservação de qualidade, com regressão de latência no backend medido. | Sustentada. | Não generalizar para todos os kernels weight-only. |
| H9 | Sensitivity 350M/1.3B | Delta PPL, KL, MSE, top-1 | Atenção, MLP e regiões apresentaram sensibilidades diferentes. | Sustentada. | Qualidade avaliada em WikiText-2 e 20 prompts. |
| H10 | Híbridos 1.3B e 6.7B | Delta PPL | Políticas aceitáveis no 1.3B falharam no 6.7B com Delta PPL entre aproximadamente +157% e +207%. | Não sustentada. | Espaço híbrido propositalmente pequeno. |
| H11 | Híbridos 1.3B finalistas, híbridos 6.7B | Delta PPL, speedup, IC 95% | Híbridos 1.3B tiveram medianas positivas, mas ICs cruzaram 1.0x; 6.7B falhou qualidade. | Não sustentada. | Não é prova de impossibilidade geral de híbridos. |
| H12 | Hardware stress complementar | Speedup, status, distribuição | Distribuição uniforme/outliers alterou comportamento observado, mas não produziu conclusão oposta. | Parcialmente sustentada. | Recorte de stress limitado a perfis específicos. |

## Uso da matriz

Esta matriz deve ser usada para rastrear cada afirmação do artigo até uma fonte
experimental. Ela também evita transformar resultado negativo em ausência de
resultado: regressões, incompatibilidades e falhas preservadas respondem onde
uma técnica deixa de sustentar a hipótese.
