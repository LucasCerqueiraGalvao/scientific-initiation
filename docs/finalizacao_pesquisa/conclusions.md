# Conclusões atuais

## Síntese

A pesquisa já sustenta uma conclusão central: compressão numérica e compressão
física não viram aceleração automaticamente. O resultado depende de workload,
shape, kernel, escopo de aplicação e custo em qualidade. A evidência mais forte
para avanço agora está em seletor fino e híbridos, não em repetir grades enormes.

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
| H11: estratégias híbridas podem dominar extremos. | Em aberto | Infraestrutura de seleção fina foi implementada; falta executar triagem e finalistas. |

## Posição científica recomendada

- Separar sempre três perguntas: erro numérico, compressão física e aceleração
  física.
- Chamar INT8 fake e pruning denso de evidência nível A.
- Chamar INT8 weight-only de evidência nível B.
- Chamar INT8 dynamic e 2:4 de evidência nível C apenas quando o profiler
  confirmar o kernel.
- Tratar o OPT-6.7B como validação seletiva: ele é caro demais para exploração
  ampla e sensível demais para conclusões apressadas.
- Priorizar próximos experimentos por sensibilidade de camada/componente e
  híbridos pequenos, porque eles atacam diretamente a tensão entre speedup e
  qualidade.
