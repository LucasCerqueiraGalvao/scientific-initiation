# Conclusões científicas finais

## Conclusão central

Nas evidências coletadas, reduzir bits ou pesos não implicou automaticamente
acelerar inferência em Transformers. A transformação precisa chegar à
representação física correta, acionar o kernel adequado e ainda compensar seus
overheads no workload real. Além disso, speedup sem qualidade aceitável não
caracteriza uma configuração útil.

## Híbridos

Nas configurações avaliadas, a seletividade por componente foi capaz de
preservar qualidade no OPT-1.3B, mas esse comportamento não se transferiu para
o OPT-6.7B. Portanto, a sensibilidade observada em modelos menores não se
mostrou diretamente transferível entre escalas.

Não foi demonstrada uma configuração híbrida que combinasse simultaneamente
preservação de qualidade e speedup confirmado no maior modelo avaliado.

Isso não significa que híbridos não funcionam em geral. Significa que, no
espaço experimental avaliado, a política derivada dos modelos menores não
fechou o compromisso qualidade/desempenho em escala maior.

## Weight-only

A investigação principal de weight-only pode ser considerada encerrada neste
backend:

- houve compressão física real;
- houve forte redução de VRAM/armazenamento;
- a qualidade foi muito bem preservada;
- nenhuma aceleração foi demonstrada;
- houve forte regressão de latência.

Essa conclusão vale para o backend avaliado, não para todos os kernels
weight-only possíveis.

## 2:4

O caminho 2:4 sparse físico foi confirmado, mas a seleção simples por magnitude
degradou fortemente a qualidade em modelos OPT. Isso deixa uma lacuna
específica: ainda não está separado se o problema principal é a sparsity 2:4 ou
o algoritmo simples de pruning.

Uma avaliação técnica indica que Wanda é a extensão mais simples e coerente se
for decidido fazer mais um experimento. Wanda é post-training, activation-aware,
dispensa fine-tuning pesado e pode preservar o mesmo backend sparse caso o
resultado final seja convertido para 2:4. SparseGPT também é relevante, mas é
mais custoso e mais complexo de integrar.

Classificação: útil mas opcional. A pesquisa já pode ser encerrada sem Wanda;
esse experimento acrescentaria evidência específica para H7, não para a tese
central inteira.

## Resposta sobre encerramento

Os dados atuais já são suficientes para encerrar a IC. A pesquisa responde à
pergunta principal com evidência rastreável: compressão numérica, compressão
física e aceleração física não são equivalentes, e o benefício final depende de
representação, kernel, workload, qualidade e escala.

Um experimento Wanda/SparseGPT 2:4 acrescentaria evidência suficiente para
justificar o custo? Classificação: opcional. Se houver tempo, Wanda é a melhor
escolha; se o objetivo for fechar relatório/artigo, não é necessário.

## Problemas que ainda impediriam encerramento

Não há bloqueio científico forte para encerrar. O que ainda precisa de cuidado
é editorial:

- garantir que smokes não sejam usados como evidência final;
- manter `TTFT` descrito como `prefill_to_first_logit`;
- não generalizar weight-only para todos os backends;
- não escrever que híbridos falharam em geral;
- não transformar speedup com Delta PPL alto em recomendação prática.
