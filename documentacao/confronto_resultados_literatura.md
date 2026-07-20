# Confronto entre resultados observados e literatura

Este documento responde uma pergunta central da validacao: os numeros obtidos
nos testes batem com o comportamento esperado pelas definicoes aceitas na
literatura?

A resposta curta e: sim para os testes matematicos e de aderencia conceitual
ja criados; ainda nao para alegacoes de desempenho de hardware, porque essas
exigem benchmarks controlados e comparacao experimental equivalente.

## Criterio usado

Os artigos e documentacoes primarias nem sempre fornecem os mesmos tensores
pequenos que usamos nos testes. Por isso, a comparacao inicial nao e "mesmo
dataset, mesmo resultado final do artigo". A comparacao correta nesta fase e:

1. pegar a definicao publicada;
2. derivar um caso pequeno e deterministico;
3. calcular o valor esperado pela formula;
4. comparar a implementacao local com esse valor;
5. limitar a conclusao ao nivel que o teste realmente sustenta.

## Resultados numericos verificados

| Conceito | Fonte usada | Valor esperado pela definicao | Resultado observado | Bate? | Limite da conclusao |
| --- | --- | --- | --- | --- | --- |
| Projecao densa | Goodfellow et al.; Vaswani et al. (2017) para projecoes lineares em Q/K/V | Para `X=[[1,2],[3,4]]`, `W=[[1,0],[0,2]]`, `b=[0.5,-0.5]`, a saida esperada e `[[1.5,3.5],[3.5,7.5]]`. | O teste retornou exatamente `[[1.5, 3.5], [3.5, 7.5]]`. | Sim | Valida a operacao linear `Y=XW^T+b`, nao uma rede completa. |
| Scaled dot-product attention | Vaswani et al. (2017), formula `softmax(QK^T/sqrt(d_k))V` | Para `Q=K=[[1,0],[0,1]]`, `V=[[1,2],[3,4]]`, a saida esperada e `[[1.6604769, 2.6604769], [2.3395231, 3.3395231]]`. | O teste retornou `[[1.6604769, 2.6604769], [2.3395228, 3.3395233]]`. | Sim | Valida a operacao matematica de atencao escalonada, nao uma arquitetura Transformer completa. |
| Attention PyTorch | Documentacao PyTorch + Vaswani et al. (2017) | A API deve produzir a mesma operacao de scaled dot-product attention, com tolerancia numerica. | A saida de `torch.nn.functional.scaled_dot_product_attention` bateu com a implementacao manual em `1e-6`. | Sim | Valida a API como referencia algoritmica para attention pequena e deterministica. |
| Mascara em attention | Documentacao PyTorch SDPA | Mascara aditiva deve alterar os scores antes do `softmax` e produzir a mesma saida da API de referencia. | A implementacao manual com mascara bateu com PyTorch em `1e-6`. | Sim | Valida mascara aditiva no caso pequeno; outros tipos de mascara devem ser testados se usados. |
| Multi-head attention | Vaswani et al. (2017), divisao da representacao em multiplas heads | A saida apos split, attention por head e combine deve preservar shape `[batch, seq, d_model]`. | Para entrada `[2,3,4]` e `2` heads, a saida preservou `[2,3,4]` e bateu com PyTorch em `1e-6`. | Sim | Valida a mecanica multi-head, ainda sem treino ou avaliacao de tarefa. |
| Dependencia entre posicoes | Vaswani et al. (2017), ideia de self-attention sobre sequencias | Alterar o valor de uma posicao pode alterar a saida calculada em outra posicao. | O teste alterou um token e a saida de outra posicao mudou. | Sim | Valida dependencia sequencial minima; ainda nao prova qualidade de modelagem. |
| Bloco Transformer simplificado | Vaswani et al. (2017) + criterios do documento teorico | Deve conter entrada sequencial, Q/K/V, attention escalonada, posicao, residual, norm e FFN. | O bloco preservou shape `[2,3,4]` e foi classificado como `bloco_transformer_simplificado`. | Sim | Valida um bloco controlado; nao deve ser chamado de Transformer completo. |
| Auditoria Transformer da NN atual | Vaswani et al. (2017) + protocolo do projeto | Um bloco Transformer deve passar em Q/K/V, attention escalonada, dependencia posicional, informacao posicional, residual, norm e FFN; uma NN comum nao deve passar. | A NN atual passou como `bloco_transformer_simplificado`; uma NN linear comum foi rejeitada como `nao_aderente`. | Sim | Valida que o nucleo pertence a familia Transformer, mas nao autoriza chama-lo de Transformer completo. |
| KV cache | Literatura de inferencia autoregressiva e compressao de KV cache | Saida com cache deve bater com saida causal completa e reduzir recomputacao de K/V. | Para `4` tokens, cache projetou `4` K/V contra `10` no caminho ingenuo; saidas bateram em `1e-6`. | Sim | Valida equivalencia e reaproveitamento algoritmico; ganho de hardware ainda exige medicao. |
| FLOPs teoricos | Hennessy e Patterson; analise de complexidade de atencao em Vaswani et al. (2017) | Para o caso pequeno definido, projecao densa `270`, attention `288`, bloco `1992`. | Os testes retornaram `270`, `288` e `1992`. | Sim | Valida contagem teorica do recorte; FLOPs nao equivalem automaticamente a latencia. |
| MSE | Definicao estatistica e `sklearn.metrics.mean_squared_error` | Para `y=[1,2,3]`, `y_hat=[1,2,4]`, `MSE=(0^2+0^2+1^2)/3=0.3333333`. | `0.3333333333333333`. | Sim | Valida a metrica numerica. |
| MAE | Definicao estatistica e `sklearn.metrics.mean_absolute_error` | Para o mesmo exemplo, `MAE=(0+0+1)/3=0.3333333`. | `0.3333333333333333`. | Sim | Valida a metrica numerica. |
| R2 | Definicao de coeficiente de determinacao e `sklearn.metrics.r2_score` | Para o mesmo exemplo, `R2=1-SS_res/SS_tot=1-1/2=0.5`. | `0.5`. | Sim | Valida a metrica numerica. |
| Similaridade de cosseno | Manning, Raghavan e Schutze; scikit-learn pairwise metrics | Para `[1,1,0]` e `[1,0,0]`, `cos=1/sqrt(2)=0.70710678`. | `0.7071067811865475`. | Sim | Valida a metrica numerica. |
| Pruning L1 PyTorch | Tutorial/API PyTorch pruning | Em matriz `2x4`, `amount=0.5` deve zerar 4 de 8 pesos e criar mascara. | `4/8` zeros, esparsidade `0.5`, mascara `weight_mask` presente. | Sim, parcialmente | Valida pruning conceitual/algoritmico; nao valida ganho real de hardware porque o caminho continua denso. |
| Quantizacao simetrica int8 manual | Literatura de compressao/quantizacao e definicao operacional | Para tensor de 5 valores `float32`, esperado reduzir de `20` bytes para `5` bytes em armazenamento int8; escala `2/127=0.015748`. | `dtype=torch.int8`, `5` bytes contra `20` bytes; erro maximo `0.007874`, dentro do limite de arredondamento da escala. | Sim | Valida quantizacao numerica; nao valida kernel de inferencia. |
| torchao int8 weight-only v2 | Documentacao torchao `Int8WeightOnlyConfig` | Configuracao deve aplicar quantizacao simetrica int8 weight-only em camadas lineares. | Camada executa sem warnings, peso `Int8Tensor` possui `qdata` em `int8`. | Sim, parcialmente | Valida representacao e execucao basica; ainda falta provar kernel/caminho de hardware no recorte final. |

## O que ainda nao esta validado

Ainda nao podemos dizer que os resultados batem com artigos de desempenho como
compressao, ganho de latencia, throughput, uso de memoria em GPU ou speedup. Os
testes atuais validam definicao e comportamento local, nao reproduzem o setup
experimental de papers como os de compressao de Transformers ou KV cache.

Para validar desempenho contra literatura, o proximo protocolo precisa fixar:

1. modelo ou bloco usado;
2. tamanho de sequencia;
3. batch size;
4. dtype;
5. hardware;
6. kernel usado;
7. metrica medida;
8. baseline;
9. tolerancia de erro numerico;
10. repeticoes e intervalo de confianca.

## Conclusao metodologica

As ferramentas atuais sao adequadas para iniciar o trabalho porque passam na
validacao de definicao: projecao densa, attention, bloco simplificado, KV
cache, FLOPs, metricas, pruning conceitual e quantizacao numerica se comportam
como esperado pelas formulas e documentacoes primarias.

Mas ainda nao sao suficientes para afirmar ganho de hardware. Essa conclusao so
sera permitida quando pruning usar formato/kernel esparso exploravel, quando
quantizacao usar representacao menor durante a execucao real, e quando os
benchmarks forem comparaveis aos protocolos dos artigos escolhidos.

## Referencias

- Vaswani et al. (2017). *Attention Is All You Need*:
  <https://arxiv.org/abs/1706.03762>.
- PyTorch. `scaled_dot_product_attention`:
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>.
- PyTorch. `torch.nn.utils.prune.l1_unstructured`:
  <https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.prune.l1_unstructured.html>.
- PyTorch Tutorials. *Pruning Tutorial*:
  <https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html>.
- PyTorch. `torchao.quantization.Int8WeightOnlyConfig`:
  <https://docs.pytorch.org/ao/stable/api_reference/generated/torchao.quantization.Int8WeightOnlyConfig.html>.
- scikit-learn. `mean_squared_error`:
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html>.
- scikit-learn. `r2_score`:
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html>.
- scikit-learn. *Cosine similarity*:
  <https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity>.
