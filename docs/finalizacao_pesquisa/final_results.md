# Resultados finais consolidados

## Narrativa científica

Este trabalho investigou em quais condições técnicas de compressão numérica e
estrutural em Transformers se convertem em benefícios físicos reais de memória
e desempenho.

A progressão experimental foi: equivalência conceitual; alteração numérica;
compressão física; kernels físicos; microbenchmarks; modelos OPT; qualidade;
escala; sensibilidade; tentativa híbrida; limites encontrados.

O resultado central é que reduzir bits ou pesos não implica automaticamente
acelerar inferência. Representação física e kernel são fundamentais; overheads
podem eliminar ganhos; microbenchmarks não predizem sempre o comportamento
end-to-end; speedup sem qualidade aceitável não basta; e estratégias seletivas
observadas em modelos menores não se transferem automaticamente para modelos
maiores.

## Principais resultados

- As 18 validações conceituais passaram; o maior erro absoluto foi
  `4.76837158e-7`.
- O benchmark sintético v3 teve 2.340 registros e 117.000 timings; INT8 fake
  teve MSE menor que pruning em todos os pares.
- O hardware físico confirmou caminhos INT8 dynamic e 2:4, mas as operações
  isoladas não tiveram speedup sustentado.
- O crossover físico não encontrou ponto de virada confirmado nas operações
  isoladas avaliadas.
- Weight-only reduziu VRAM/armazenamento e preservou qualidade, mas ficou muito
  lento no backend avaliado.
- INT8 dynamic acelerou alguns cenários grandes de OPT-6.7B, mas com
  degradação de qualidade fora dos limites.
- Híbridos no OPT-1.3B preservaram qualidade quando atenção sensível foi
  mantida, mas não tiveram speedup estatisticamente confirmado.
- Híbridos no OPT-6.7B falharam qualidade; o Delta PPL ficou aproximadamente
  entre +157% e +207%, e por isso o benchmark físico foi corretamente evitado.

## Tabela principal interpretativa

| Modelo/recorte | Técnica | Escopo | Tipo A/B/C | Kernel físico | Speedup | IC 95% | VRAM | Armazenamento | Delta PPL | Qualidade | Interpretação |
|---|---|---|---|---|---:|---|---:|---:|---:|---|---|
| Operação isolada | INT8 dynamic | dense projection | C | confirmado | 0.767x | 0.758-0.788 | reduzida | reduzido | n/a | n/a | Kernel físico existe, mas houve regressão. |
| Operação isolada | INT8 dynamic | multi-head attention | C | confirmado | 0.641x | 0.627-0.665 | reduzida | reduzido | n/a | n/a | Sem crossover confirmado. |
| Operação isolada | INT8 weight-only | multi-head attention | B | pesos empacotados | 0.078x | 0.070-0.086 | reduzida | reduzido | n/a | n/a | Compressão sem aceleração. |
| Operação isolada | 2:4 | multi-head attention | C | confirmado | 0.713x | 0.627-0.781 | reduzida | reduzido | n/a | n/a | Sparse físico não bastou para acelerar. |
| OPT-6.7B | INT8 dynamic | atenção | C | confirmado | 1.169x | 1.009-1.249 | reduzida | reduzido | +8.553% | não | Speedup confirmado, qualidade fora do limite. |
| OPT-6.7B | INT8 dynamic | blocos | C | confirmado | 1.756x | 1.171-2.137 | reduzida | reduzido | +165.052% | não | Aceleração forte, qualidade inviável. |
| OPT-6.7B | weight-only | atenção | B | compressão confirmada | 0.026x | n/d | 0.839x | 0.839x | -0.0035% | sim | Preserva qualidade, mas é muito lento. |
| OPT-6.7B | weight-only | blocos | B | compressão confirmada | 0.009x | n/d | 0.516x | 0.516x | -0.0150% | sim | Melhor compressão, pior latência. |
| OPT-6.7B | 2:4 | atenção | C | confirmado | 1.030x | 0.974-1.111 | reduzida | reduzido | +38.821% | não | Qualidade degrada e IC cruza 1.0x. |
| OPT-6.7B | 2:4 | blocos | C | confirmado | 1.073x | 1.017-1.228 | reduzida | reduzido | +994.987% | não | Speedup com colapso de qualidade. |
| OPT-1.3B | híbrido INT8 dynamic | MLP all, atenção BF16 | C | candidato | 1.110x | 0.881-1.309 | n/d | n/d | -1.098% | sim | Qualidade preservada, speedup não confirmado. |
| OPT-1.3B | híbrido INT8 dynamic | MLP all, atenção 0-7 BF16 | C | candidato | 1.131x | 0.872-1.299 | n/d | n/d | -0.815% | sim | Mediana positiva, IC cruza 1.0x. |
| OPT-1.3B | híbrido INT8 dynamic | MLP all, atenção 0-15 BF16 | C | candidato | 1.108x | 0.841-1.310 | n/d | n/d | -0.370% | sim | Seletividade ajudou qualidade. |
| OPT-6.7B | híbrido INT8 dynamic | MLP all, atenção BF16 | C | não medido após falha de qualidade | n/a | n/a | n/a | n/a | +207.254% | não | Candidato descartado antes do físico. |
| OPT-6.7B | híbrido INT8 dynamic | MLP all, atenção 0-15 BF16 | C | não medido após falha de qualidade | n/a | n/a | n/a | n/a | +157.271% | não | Política do 1.3B não transferiu. |

`n/d` indica informação não destacada no resumo final; o valor bruto permanece
nos CSVs canônicos quando medido. `n/a` indica combinação não aplicável ou não
executada por decisão metodológica.

## Resultados negativos preservados

- Ausência de crossover confirmado nas operações isoladas.
- Weight-only muito lento no backend avaliado.
- 2:4 sem ganho sustentado nas operações isoladas.
- Pruning denso sem redução física.
- INT8 dynamic rápido no OPT-6.7B, mas com qualidade muito degradada.
- Híbridos sem speedup confirmado no OPT-1.3B.
- Híbridos sem preservação de qualidade no OPT-6.7B.

Cada resultado negativo responde uma pergunta científica: ele separa onde a
técnica altera números, onde comprime representação e onde realmente acelera
sem violar qualidade.

## Contribuições

1. Framework experimental reproduzível para separar alteração numérica,
   compressão física e aceleração física.
2. Comparação multi-escala em OPT, de 125M a 6.7B, com qualidade, latência,
   VRAM e armazenamento.
3. Demonstração experimental da diferença entre zeros lógicos e sparsity
   fisicamente explorável.
4. Análise conjunta de microbenchmarks, kernels físicos e comportamento em
   modelos pré-treinados.
5. Evidência de sensibilidade heterogênea por componente/layer e de limitada
   transferibilidade entre escalas.
