# Hipóteses finais

| Hipótese | Classificação | Evidência principal | Leitura final |
|---|---|---|---|
| H1. Redução de precisão ou pesos não implica automaticamente redução de latência. | Sustentada | INT8 fake, pruning denso, weight-only e 2:4 frequentemente reduziram representação ou alteraram pesos sem acelerar. | Reduzir bits ou pesos é condição insuficiente para acelerar inferência. |
| H2. Compressão numérica, compressão física e aceleração física são fenômenos distintos. | Sustentada | Níveis A/B/C; INT8 fake preserva fidelidade, weight-only comprime, INT8 dynamic/2:4 exigem kernel. | A tese metodológica central foi confirmada. |
| H3. A aceleração depende da representação, kernel e backend. | Sustentada | Kernels INT8 dynamic e 2:4 foram confirmados, mas o ganho dependeu de modelo/workload. | O backend é parte do resultado, não detalhe operacional. |
| H4. Existe crossover favorável nas operações isoladas avaliadas. | Não sustentada | `hardware_crossover_v3` não encontrou speedup confirmado; melhores casos cruzaram 1.0x. | Nos perfis isolados avaliados, não houve ponto de virada estatisticamente confirmado. |
| H5. O tamanho/formato do workload influencia a eficácia das otimizações. | Parcialmente sustentada | Crossover variou por batch/sequência/largura; OPT-6.7B mostrou comportamento diferente dos menores. | Workload importa, mas não bastou para produzir crossover confirmado nas operações isoladas. |
| H6. Pruning não estruturado revela redundância, mas não produz ganho físico automaticamente. | Sustentada | Pruning denso degradou gradualmente a saída, mas não reduziu fisicamente VRAM nos caminhos densos. | Zeros lógicos não equivalem a aceleração. |
| H7. 2:4 executa sparsity física, mas a preservação de qualidade depende fortemente da estratégia de pruning. | Parcialmente sustentada | cuSPARSELt foi confirmado; magnitude pruning degradou qualidade, especialmente em blocos e modelos maiores. | O caminho físico existe, mas a seleção por magnitude não foi suficiente. |
| H8. Weight-only pode reduzir fortemente memória mantendo qualidade sem necessariamente acelerar. | Sustentada | OPT-6.7B weight-only preservou PPL e reduziu VRAM/armazenamento, mas teve forte regressão de latência. | Compressão física real não garantiu speedup no backend avaliado. |
| H9. Componentes/layers apresentam sensibilidades diferentes à quantização. | Sustentada | Sensibilidade em OPT-350M/1.3B mostrou diferenças entre atenção, MLP e regiões. | Há heterogeneidade útil para desenho seletivo. |
| H10. A política de sensibilidade encontrada em modelos menores é transferível para modelos maiores. | Não sustentada | Híbridos que preservaram qualidade no OPT-1.3B falharam no OPT-6.7B. | A sensibilidade local não se transferiu diretamente entre escalas. |
| H11. Uma configuração híbrida consegue combinar claramente o melhor de BF16 e INT8 dynamic. | Não sustentada | Híbridos 1.3B tiveram medianas positivas, mas ICs cruzaram 1.0x; híbridos 6.7B falharam qualidade. | Não foi demonstrado híbrido com qualidade e speedup confirmado no maior modelo. |
| H12. Distribuição/outliers influenciam comportamento dos caminhos quantizados. | Parcialmente sustentada | Stress complementar mediu uniforme e outliers; houve variação de comportamento, mas sem mudar a conclusão de speedup. | A distribuição afeta a medição, mas não reverteu os achados centrais. |

## Síntese

As hipóteses mais fortes são H1, H2, H3, H6, H8 e H9. H4, H10 e H11 foram
negadas pelos dados atuais. H5, H7 e H12 permanecem parcialmente sustentadas:
há evidência real, mas ela não fecha todas as causas possíveis.
