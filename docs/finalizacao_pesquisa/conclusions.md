# Conclusões atuais

## Síntese

A pesquisa já sustenta uma conclusão central: compressão numérica e compressão
física não viram aceleração automaticamente. O resultado depende de workload,
shape, kernel, escopo de aplicação e custo em qualidade. A evidência mais forte
para avanço agora está em seletor fino e híbridos, não em repetir grades enormes.

A execução complementar de 19/09/2026 reforçou essa leitura. O microbenchmark de
crossover não encontrou ganho sustentado em INT8 dynamic, weight-only ou 2:4 nas
operações isoladas. A sensibilidade em OPT-1.3B, porém, mostrou que MLP é bem
mais tolerante que atenção final, apontando para híbridos que preservem a atenção
final em BF16.

A execução híbrida posterior confirmou essa hipótese apenas no OPT-1.3B. Os
híbridos recuperaram qualidade e apresentaram medianas de speedup positivas em
prefill/prompt-forward, mas sem intervalo de confiança inteiramente acima de
`1,0x`. No OPT-6.7B, nenhum híbrido INT8 dynamic testado preservou qualidade; a
perplexidade aumentou de `157%` a `207%` nos recortes híbridos.

## Hipóteses

| Hipótese | Estado | Evidência atual |
| --- | --- | --- |
| H1: as implementações manuais representam as mesmas operações conceituais das bibliotecas. | Sustentada | 18/18 comparações passaram contra NumPy, PyTorch SDPA e TensorFlow/Keras. |
| H2: INT8 fake preserva melhor fidelidade que pruning denso em operações isoladas. | Sustentada | No sintético v3, INT8 fake teve MSE menor que pruning em todos os pares. |
| H3: pruning denso percentual acelera por si só. | Não sustentada | Pruning denso continua em caminho denso, sem redução consistente de VRAM/latência. |
| H4: kernels físicos sempre aceleram operações isoladas. | Não sustentada | Em `hardware_nativo_v3`, INT8 dynamic e 2:4 foram confirmados, mas ficaram mais lentos que baselines compilados no recorte. |
| H5: workload/shape afeta crossover de desempenho. | Parcial | Há indícios por sequência, batch e modelo, mas falta microbenchmark dedicado de crossover. |
| H6: INT8 em modelos pré-treinados preserva qualidade melhor que pruning agressivo. | Sustentada parcialmente | INT8 fake/weight-only ficaram estáveis; INT8 dynamic varia por modelo/escopo. |
| H7: pruning 2:4 por magnitude é suficiente para preservar qualidade. | Não sustentada | 2:4 por magnitude degradou qualidade nos OPTs medidos. |
| H8: modelos maiores tendem a expor mais oportunidade de speedup físico. | Parcial | OPT-6.7B mostrou speedup em alguns caminhos, mas o melhor ganho em blocos INT8 dynamic não preservou qualidade. |
| H9: armazenamento menor implica VRAM menor. | Parcial | Weight-only reduz armazenamento; VRAM depende do backend e de buffers intermediários. |
| H10: armazenamento menor implica latência menor. | Não sustentada | Weight-only ficou lento no backend medido. |
| H11: estratégias híbridas podem dominar extremos. | Parcial e limitada | No OPT-1.3B, híbridos preservaram qualidade e tiveram medianas de speedup positivas; no OPT-6.7B, todos os híbridos INT8 dynamic testados falharam qualidade. |

Após a rodada complementar, H11 ficou mais plausível: `attention 16-23` falhou
no OPT-1.3B, enquanto todos os recortes de MLP passaram no critério oficial.
Após a rodada híbrida, H11 deixou de ser uma hipótese aberta ampla: ela é
plausível para OPT-1.3B neste backend, mas não escalou para OPT-6.7B.

## Posição científica recomendada

- Separar sempre três perguntas: erro numérico, compressão física e aceleração
  física.
- Chamar INT8 fake e pruning denso de evidência nível A.
- Chamar INT8 weight-only de evidência nível B.
- Chamar INT8 dynamic e 2:4 de evidência nível C apenas quando o profiler
  confirmar o kernel.
- Tratar o OPT-6.7B como validação seletiva: ele é caro demais para exploração
  ampla e sensível demais para conclusões apressadas.
- Não executar performance 6.7B de híbridos INT8 dynamic simples enquanto a
  qualidade estiver reprovada.
- Priorizar, se houver continuação, uma investigação de quantização calibrada ou
  backend alternativo para explicar por que o INT8 dynamic degradou tanto o
  OPT-6.7B.
