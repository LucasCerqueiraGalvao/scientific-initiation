# Limitações finais

## Escopo experimental

- A pesquisa foi executada em uma única GPU principal: RTX 4070 Ti SUPER.
- A família de modelos avaliada foi OPT; os resultados não devem ser
  generalizados automaticamente para LLaMA, Mistral, Gemma ou outras famílias.
- O dataset principal de qualidade foi WikiText-2 com 64 janelas de 512 tokens,
  o que é adequado para triagem controlada, mas estreito para alegações gerais
  sobre linguagem.
- A avaliação de geração usou 20 prompts locais e decoding greedy de 64 tokens;
  isso não substitui benchmarks amplos de tarefas nem avaliação humana.
- Não foram usadas múltiplas seeds independentes de dataset/prompt.

## Medição

- Os clocks, power state e detalhes finos do driver não foram fixados.
- O protocolo controla parte do ruído com GPU ociosa, limite de temperatura,
  warm-up, sincronização CUDA, repetições e bootstrap, mas não elimina toda a
  variabilidade de ambiente.
- O campo histórico `model_ttft` mede `prefill_to_first_logit`, não TTFT
  completo de sistema com tokenização, fila, streaming e overhead de serviço.
- A confirmação de kernels é mais forte nos caminhos/protocolos em que o
  profiler registrou explicitamente os operadores esperados, principalmente
  prefill e operações isoladas.

## Técnicas

- INT8 fake mede alteração numérica com dequantização; não prova aceleração por
  aritmética inteira.
- Pruning denso percentual cria zeros lógicos, mas permanece em armazenamento e
  kernels densos quando não há representação física sparse.
- O INT8 weight-only medido representa o backend usado nesta infraestrutura. A
  forte regressão de latência observada não deve ser generalizada para todos os
  kernels ou frameworks weight-only.
- O 2:4 atual usa seleção simples por magnitude. A degradação de qualidade não
  invalida o formato 2:4 em si; indica que a estratégia de seleção de pesos
  avaliada foi insuficiente para preservar qualidade.
- O espaço híbrido foi propositalmente pequeno. Ele testa transferibilidade de
  uma política seletiva plausível, não uma busca exaustiva por configuração
  ótima.

## Interpretação

- Resultados locais em microbenchmarks não predizem necessariamente o
  comportamento end-to-end em modelos completos.
- Speedup sem qualidade aceitável não deve ser tratado como configuração útil.
- Qualidade aceitável sem speedup confirmado também não demonstra benefício de
  latência.
- A sensibilidade observada em modelos menores não se transferiu diretamente ao
  OPT-6.7B nas configurações avaliadas.
- Os resultados negativos são parte da evidência: eles delimitam onde compressão
  numérica, compressão física e aceleração física deixam de coincidir.
