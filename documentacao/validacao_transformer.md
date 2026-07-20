# Validacao arquitetural da NN usada no estudo

Este documento registra se a rede neural usada no estudo pode ser chamada de
Transformer.

Ele tambem registra a ponte metodologica entre:

```text
Vaswani et al. (2017) -> propriedade esperada -> script Python -> teste deterministico -> limite da conclusao
```

Essa ponte e importante porque o estudo nao deve depender do nome de uma classe
ou de uma afirmacao informal. Para dizer que um trecho e de Transformer,
precisamos mostrar que ele possui a propriedade descrita no artigo e que essa
propriedade foi testada.

## Resultado atual

A implementacao `validacao.transformer.SimplifiedTransformerBlock` esta
validada como:

```text
bloco_transformer_simplificado
```

Ela pertence a familia Transformer porque passa nos criterios essenciais de um
bloco:

- recebe entrada sequencial `[batch, seq_len, d_model]`;
- preserva shape sequencial na saida;
- projeta a entrada em Q, K e V;
- usa atencao escalonada compativel com `softmax(QK^T / sqrt(d_k))V`;
- usa multi-head attention;
- mistura informacao entre posicoes;
- usa informacao posicional sinusoidal;
- possui residual, normalizacao e feed-forward;
- e deterministica em modo de avaliacao.

Ela nao deve ser chamada de "Transformer completo" no artigo, porque o recorte
experimental usa um bloco controlado, nao uma arquitetura completa com pilha de
camadas, cabeca de tarefa, treinamento e protocolo de modelo final.

## Criterio de rastreabilidade

Cada componente aceito no estudo precisa passar por cinco perguntas:

1. O componente existe no Transformer de Vaswani et al.?
2. Qual propriedade observavel esse componente deve ter?
3. Onde essa propriedade aparece no nosso codigo?
4. Qual teste deterministico verifica essa propriedade?
5. O que esse teste ainda nao permite concluir?

Assim, a validacao nao diz apenas "o codigo roda". Ela diz "esta operacao
corresponde a este trecho definido pela literatura, dentro destes limites".

## Matriz Vaswani -> codigo -> teste

| Trecho no Transformer de Vaswani | Propriedade esperada | Onde esta no codigo | Teste deterministico | Comparacao usada | O que o teste prova | Limite metodologico |
| --- | --- | --- | --- | --- | --- | --- |
| Entrada sequencial | O modulo recebe tensores organizados como sequencia, com dimensao de lote, posicao e representacao. | `SimplifiedTransformerBlock.forward`, em `validacao/transformer.py`. | `test_simplified_transformer_block_has_expected_components_and_shape`. | Shape esperado `[batch, seq_len, d_model]`. | O bloco recebe e devolve uma representacao sequencial. | Nao prova tokenizacao, embedding real ou tarefa de linguagem. |
| Projecoes lineares densas | Entradas sao transformadas por matrizes aprendidas; no attention original essas projecoes geram Q, K, V e saida. | `q_proj`, `k_proj`, `v_proj`, `out_proj` em `validacao/transformer.py`; `dense_projection_torch` em `validacao/dense.py`. | `test_dense_projection_matches_manual_definition`; auditoria de Q/K/V. | Formula `Y = XW^T + b` calculada manualmente. | A operacao linear usada no bloco segue a definicao matematica esperada. | Nao prova que os pesos treinados sao bons; os pesos do bloco controlado sao fixos/iniciais. |
| Projecoes Q, K e V | A mesma sequencia gera consultas, chaves e valores para self-attention. | `project_qkv` em `validacao/transformer.py`. | `test_transformer_audit_validates_current_nn_as_simplified_transformer_block`. | Inspecao estrutural: existencia de `q_proj`, `k_proj`, `v_proj`. | O bloco tem as tres projecoes exigidas para attention de Transformer. | Nao prova arquitetura completa encoder-decoder. |
| Scaled dot-product attention | O nucleo de atencao deve calcular `softmax(QK^T / sqrt(d_k))V`. | `scaled_dot_product_attention_torch` e `scaled_dot_product_attention_numpy` em `validacao/attention.py`. | `test_scaled_dot_product_attention_matches_independent_numpy_reference`; `test_scaled_dot_product_attention_matches_pytorch_reference`. | Valor manual pequeno, NumPy independente e `torch.nn.functional.scaled_dot_product_attention`. | O algoritmo de atencao bate com a formula do artigo e com uma referencia pratica do PyTorch. | Nao prova por si so que existe um bloco completo; prova o nucleo de attention. |
| Mascara de atencao | A atencao pode receber restricoes aditivas, como ocorre em variantes mascaradas. | Parametro `additive_attention_mask` em `validacao/attention.py`. | `test_scaled_dot_product_attention_additive_mask_matches_pytorch_reference`. | `torch.nn.functional.scaled_dot_product_attention` com `attn_mask`. | A mascara usada no nosso nucleo e compativel com a referencia do PyTorch. | O bloco experimental padrao nao usa mascara causal por padrao. |
| Dependencia entre posicoes | A saida de uma posicao pode depender de outra posicao da sequencia. | Self-attention em `validacao/attention.py` e auditoria em `validacao/auditoria_transformer.py`. | `test_attention_output_depends_on_other_sequence_positions`; `test_transformer_audit_validates_current_nn_as_simplified_transformer_block`. | Alteracao controlada de um token e observacao de mudanca em outra posicao. | A operacao nao trata cada posicao de forma isolada; ha mistura de informacao sequencial. | Nao mede qualidade semantica dessa dependencia em uma tarefa real. |
| Multi-head attention | A representacao e dividida em cabecas, cada cabeca aplica attention, e depois as cabecas sao recombinadas. | `split_heads`, `combine_heads` e `multi_head_attention_from_qkv` em `validacao/attention.py`. | `test_multi_head_attention_preserves_shape_and_matches_split_reference`. | Aplicacao da SDPA do PyTorch em cada head e recombinacao manual. | A mecanica de separar/aplicar/recombinar heads esta correta. | Nao prova que o numero de heads escolhido e otimo. |
| Informacao posicional | Como attention pura nao conhece ordem por si so, o bloco precisa receber algum sinal de posicao. | `sinusoidal_position_encoding` e `add_positional_information` em `validacao/transformer.py`. | Auditoria: `report.has_positional_information_or_justification`. | Criterio arquitetural derivado de Vaswani et al. | O bloco adiciona informacao posicional antes das projecoes. | Nao prova que essa codificacao e a melhor para todas as tarefas. |
| Residual, normalizacao e FFN | O bloco Transformer usa subcamadas com conexoes residuais, normalizacao e rede feed-forward. | `norm1`, `norm2`, `ffn` e `out_proj` em `validacao/transformer.py`. | `test_simplified_transformer_block_has_expected_components_and_shape`; auditoria estrutural. | Checklist arquitetural do bloco. | O bloco tem os componentes essenciais para ser classificado como bloco Transformer simplificado. | Ainda falta pilha completa de camadas e cabeca de tarefa para chamar de Transformer completo. |
| Inferencia deterministica | Em avaliacao, a mesma entrada deve produzir a mesma saida, evitando ruido experimental. | `audit_transformer_module` em `validacao/auditoria_transformer.py`. | `test_transformer_audit_validates_current_nn_as_simplified_transformer_block`. | Duas execucoes em `eval()` com a mesma entrada. | O nucleo pode ser usado em benchmarks controlados. | Nao substitui avaliacao estatistica de desempenho em varias repeticoes. |
| Controle negativo | Uma NN comum nao deve ser aceita como Transformer so por ser rede neural. | Auditoria aplicada a `torch.nn.Sequential(Linear, ReLU, Linear)`. | `test_transformer_audit_rejects_plain_neural_network_as_transformer`. | Ausencia de Q/K/V e de scaled attention. | O validador distingue uma NN generica de um bloco da familia Transformer. | Nao classifica todos os tipos possiveis de arquitetura; classifica o recorte do estudo. |

## O que esta comprovado

Os testes comprovam que os scripts Python implementam deterministicamente os
trechos centrais escolhidos do Transformer de Vaswani et al. para o escopo do
projeto:

- projecao linear densa;
- geracao de Q, K e V;
- scaled dot-product attention;
- mascara aditiva de attention;
- dependencia entre posicoes da sequencia;
- multi-head attention;
- informacao posicional sinusoidal;
- subcamadas residual, normalizacao e FFN;
- comportamento deterministico em inferencia;
- rejeicao de uma NN comum como controle negativo.

Portanto, o objeto experimental atual e valido para estudar:

```text
operacoes centrais de Transformer
```

e tambem:

```text
bloco Transformer simplificado
```

## O que nao esta comprovado

Os testes nao comprovam que temos a arquitetura completa do artigo *Attention Is
All You Need*. Para isso, seria necessario incluir e validar elementos fora do
recorte atual, como:

- pilha completa de blocos;
- organizacao completa de encoder e/ou decoder;
- embeddings de entrada e saida;
- cabeca de tarefa;
- treinamento ou carregamento de pesos de um modelo real;
- avaliacao em tarefa ou dataset.

Por conta disso, a frase metodologicamente correta e:

```text
O estudo avalia trechos computacionalmente centrais de um bloco Transformer
simplificado, derivados da arquitetura de Vaswani et al. (2017), com foco em
inferencia controlada.
```

E a frase que deve ser evitada neste momento e:

```text
O estudo avalia uma Transformer completa.
```

## Como a verificacao e feita

A auditoria esta implementada em `validacao.auditoria_transformer`. Ela executa
testes comportamentais e estruturais:

1. passa uma sequencia pelo modulo;
2. verifica se a saida preserva `[batch, seq_len, d_model]`;
3. inspeciona Q/K/V, output projection, norm e FFN;
4. compara a atencao interna com `torch.nn.functional.scaled_dot_product_attention`;
5. altera um token e verifica se outra posicao pode mudar;
6. roda a mesma entrada duas vezes em `eval()` para testar determinismo;
7. aplica a classificacao `nao_aderente`, `operacao_inspirada_em_transformer`,
   `bloco_transformer_simplificado` ou `transformer`.

## Controle negativo

A mesma auditoria rejeita uma rede neural comum composta por camadas lineares e
ReLU. Esse controle e importante: ele mostra que o teste nao aceita qualquer
NN como Transformer.

## Regra para o texto final

No artigo, usar:

- "bloco Transformer simplificado" para o nucleo experimental atual;
- "operacao inspirada em Transformer" quando o teste isolar attention,
  projecao densa ou KV cache;
- "Transformer" apenas se houver arquitetura completa ou justificativa formal
  para todos os componentes ausentes.
