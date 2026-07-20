# Base teorica e validacao cientifica do modelo

## Objetivo do documento

Este documento organiza a primeira fase do trabalho: estabelecer uma base
teorica confiavel e um metodo de validacao antes de qualquer benchmark de
hardware. A regra de trabalho sera:

> Nenhum conceito tecnico deve ser usado no software ou no relatorio sem uma
> referencia cientifica confiavel, uma definicao operacional e um teste que
> verifique se a implementacao observada corresponde ao comportamento esperado.

Isso evita um erro comum em experimentos de software: medir desempenho de uma
implementacao que usa o nome de uma tecnica, mas nao realiza a tecnica de forma
cientificamente adequada. Por exemplo, uma funcao que quantiza e depois volta
imediatamente para `float32` pode ser util para estudar erro numerico, mas nao
deve ser apresentada como evidencia de ganho real de hardware.

## Principio metodologico

O trabalho deve separar quatro niveis de certeza:

1. Definicao: o termo usado esta definido corretamente pela literatura?
2. Matematica: a formula implementada bate com casos pequenos derivados da
   definicao?
3. Algoritmo: o software implementa o algoritmo, ou uma aproximacao aceitavel e
   explicitamente justificada?
4. Hardware: a implementacao realmente usa uma representacao,
   estrutura de dados, kernel ou caminho de execucao capaz de produzir ganho
   fisico mensuravel em memoria, latencia ou throughput?

Uma implementacao pode ser valida em um nivel e invalida em outro. Por exemplo,
zerar 50% dos pesos de uma matriz e continuar executando uma multiplicacao
densa pode ser um pruning conceitual/algoritmico, mas nao prova ganho real de
hardware.

## Matriz de rastreabilidade

Todo conceito, ferramenta ou algoritmo usado no projeto deve entrar em uma
matriz com os seguintes campos.

| Campo | Descricao |
| --- | --- |
| Conceito declarado | Nome usado no trabalho, como Transformer, quantizacao ou pruning. |
| Fonte cientifica | Artigo, livro-texto ou documentacao primaria que define o conceito. |
| Definicao aceita | Definicao curta usada pelo projeto. |
| Propriedade esperada no modelo | O que precisa aparecer na implementacao para o nome ser valido. |
| Teste de verificacao | Como verificar o comportamento esperado. |
| Resultado observado | Campo a preencher apos executar a verificacao. |
| Status | Aderente, parcialmente aderente ou nao aderente. |
| Nivel de validade | Conceitual, numerico, algoritmico ou hardware. |

Modelo de linha:

| Conceito declarado | Fonte cientifica | Definicao aceita | Propriedade esperada no modelo | Teste de verificacao | Resultado observado | Status | Nivel |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Transformer | Vaswani et al. (2017) | Arquitetura baseada em atencao, com tratamento de sequencias e informacao posicional. | Entrada sequencial, Q/K/V, atencao escalonada, dependencia entre posicoes e informacao posicional ou justificativa formal de ausencia. | Comparar a saida de um caso pequeno com implementacao de referencia e auditar componentes arquiteturais. | A preencher. | A preencher. | Conceitual/algoritmico. |

## Glossario operacional

### Rede neural

- Definicao cientifica: modelo computacional parametrizado composto por
  unidades ou camadas que aplicam transformacoes lineares e nao lineares para
  aproximar funcoes a partir de dados.
- Referencia confiavel: Goodfellow, Bengio e Courville, *Deep Learning*,
  especialmente a discussao de redes feedforward e modelos parametrizados:
  <https://www.deeplearningbook.org/>.
- Explicacao intuitiva: uma rede neural e uma funcao com muitos parametros
  ajustaveis. Ela transforma uma entrada em uma saida por etapas.
- Uso no trabalho: sera o conceito mais geral por tras dos modelos estudados.
  Um Transformer e um tipo especifico de arquitetura neural.
- Propriedades exigidas no codigo: parametros numericos, fluxo de entrada para
  saida, composicao de operacoes e possibilidade de inferencia com parametros
  fixos.
- Teste de aderencia: verificar se a implementacao possui parametros, executa
  uma passagem direta deterministica para uma entrada fixa e produz saida com
  dimensao esperada.

### Deep learning

- Definicao cientifica: subarea de aprendizado de maquina baseada em multiplas
  camadas de representacao, nas quais conceitos mais complexos sao construidos a
  partir de conceitos mais simples.
- Referencia confiavel: Goodfellow, Bengio e Courville, *Deep Learning*:
  <https://www.deeplearningbook.org/>.
- Explicacao intuitiva: deep learning e o uso de redes neurais com varias
  etapas de transformacao, permitindo aprender representacoes intermediarias.
- Uso no trabalho: delimita o contexto em que Transformers, compressao e
  inferencia sao discutidos.
- Propriedades exigidas no codigo: mais de uma transformacao parametrizada ou
  uma arquitetura explicitamente derivada de modelos profundos.
- Teste de aderencia: inspecionar a arquitetura. Se houver apenas uma
  multiplicacao linear isolada, o objeto deve ser descrito como uma operacao
  auxiliar, nao como modelo de deep learning completo.

### Transformer

- Definicao cientifica: arquitetura neural baseada em mecanismos de atencao,
  proposta por Vaswani et al. (2017), dispensando recorrencia e convolucoes na
  arquitetura original de sequencia para sequencia.
- Referencia confiavel: Vaswani et al., *Attention Is All You Need*, 2017:
  <https://arxiv.org/abs/1706.03762>.
- Explicacao intuitiva: um Transformer processa uma sequencia permitindo que
  cada posicao consulte outras posicoes por meio de atencao.
- Uso no trabalho: sera o objeto arquitetural de referencia. O projeto podera
  estudar uma operacao inspirada em Transformer, um bloco Transformer
  simplificado ou uma implementacao Transformer completa.
- Propriedades exigidas no codigo: entrada sequencial, projecoes Q/K/V,
  mecanismo de atencao, mistura de informacao entre posicoes, informacao
  posicional ou justificativa formal para remove-la, e transformacoes
  parametrizadas.
- Teste de aderencia: executar checklist arquitetural e comparar um caso
  pequeno com uma implementacao de referencia. Se faltar componente essencial,
  classificar corretamente como "operacao inspirada em Transformer" ou "bloco
  Transformer simplificado".

### Self-attention

- Definicao cientifica: mecanismo de atencao no qual consultas, chaves e valores
  sao derivados da mesma sequencia, permitindo que cada posicao combine
  informacao de outras posicoes da propria entrada.
- Referencia confiavel: Vaswani et al. (2017):
  <https://arxiv.org/abs/1706.03762>.
- Explicacao intuitiva: cada token "olha" para os outros tokens da mesma
  sequencia e decide quais importam mais para sua representacao.
- Uso no trabalho: sera uma das operacoes centrais a validar antes de estudar
  custo computacional.
- Propriedades exigidas no codigo: Q, K e V calculados da mesma entrada, matriz
  de atencao com dimensao compativel com o comprimento da sequencia e saida
  dependente de multiplas posicoes.
- Teste de aderencia: alterar um token nao trivial da sequencia e verificar se
  a saida de outras posicoes pode mudar. Comparar tambem a saida com uma
  implementacao direta da formula de atencao.

### Scaled dot-product attention

- Definicao cientifica: atencao definida por
  `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V`.
- Referencia confiavel: Vaswani et al. (2017):
  <https://arxiv.org/abs/1706.03762>.
- Explicacao intuitiva: calcula similaridades entre consultas e chaves,
  normaliza essas similaridades e usa os pesos resultantes para combinar os
  valores.
- Uso no trabalho: sera a forma padrao para validar a operacao de atencao.
- Propriedades exigidas no codigo: produto `QK^T`, divisao por `sqrt(d_k)`,
  `softmax` no eixo correto e multiplicacao pelos valores `V`.
- Teste de aderencia: criar tensores pequenos com valores conhecidos, calcular
  a saida manualmente ou por uma implementacao de referencia e exigir diferenca
  numerica dentro de uma tolerancia definida.

### Inferencia

- Definicao cientifica: fase em que um modelo ja definido ou treinado e usado
  para produzir saidas a partir de entradas, sem atualizar parametros.
- Referencia confiavel: Goodfellow, Bengio e Courville, *Deep Learning*:
  <https://www.deeplearningbook.org/>. Para uso pratico em software, a
  documentacao do PyTorch tambem distingue execucao de modelos em modo de
  avaliacao/inferencia.
- Explicacao intuitiva: inferencia e usar o modelo, nao treina-lo.
- Uso no trabalho: o foco sera medir custo e comportamento durante execucao de
  modelos ou operacoes com pesos fixos.
- Propriedades exigidas no codigo: ausencia de atualizacao de pesos, modo de
  avaliacao quando aplicavel, desativacao de gradientes quando o objetivo for
  apenas execucao e entradas controladas.
- Teste de aderencia: executar a mesma entrada duas vezes com os mesmos pesos e
  semente. Os pesos nao devem mudar e a saida deve ser reprodutivel, salvo
  operacoes explicitamente nao deterministicas.

### Projecao densa

- Definicao cientifica: transformacao linear parametrizada do tipo
  `Y = XW^T + b`, geralmente aplicada por multiplicacao matricial densa.
- Referencia confiavel: Goodfellow, Bengio e Courville para camadas lineares em
  redes neurais; Vaswani et al. (2017) para o uso de projecoes aprendidas em Q,
  K, V e saidas de atencao.
- Explicacao intuitiva: uma matriz de pesos transforma cada vetor de entrada em
  outro vetor.
- Uso no trabalho: sera um bloco basico para construir Q, K, V e camadas
  lineares simplificadas.
- Propriedades exigidas no codigo: matriz de pesos densa, dimensoes
  consistentes e multiplicacao matricial equivalente a transformacao linear.
- Teste de aderencia: comparar a saida da funcao com uma multiplicacao
  matricial direta em tensores pequenos; verificar dimensoes de entrada, pesos e
  saida.

### KV cache

- Definicao cientifica: armazenamento de chaves e valores ja calculados durante
  geracao autoregressiva, evitando recomputacao em passos futuros.
- Referencia confiavel: Google Research, *TurboQuant: Redefining AI efficiency
  with extreme compression*, 2026:
  <https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/>.
  O paper TurboQuant tambem trata quantizacao de vetores e KV cache:
  <https://arxiv.org/abs/2504.19874>.
- Explicacao intuitiva: o modelo guarda informacao intermediaria de tokens
  anteriores para nao recalcular tudo a cada novo token.
- Uso no trabalho: sera relevante quando o experimento sair de atencao em lote
  para cenarios autoregressivos ou compressao de memoria.
- Propriedades exigidas no codigo: armazenamento persistente de K e V entre
  passos de geracao, crescimento com o comprimento do contexto e reutilizacao em
  novos passos.
- Teste de aderencia: comparar saida com e sem cache para a mesma sequencia. As
  saidas devem ser numericamente equivalentes dentro de tolerancia, mas a versao
  com cache deve evitar recomputacao de K/V anteriores.

### Pruning

- Definicao cientifica: tecnica de compressao que remove parametros, conexoes,
  canais ou estruturas consideradas menos relevantes, produzindo esparsidade ou
  reducao estrutural.
- Referencia confiavel: Tang et al., *A Survey on Transformer Compression*,
  2024: <https://arxiv.org/abs/2402.05964>. Para implementacao em ferramenta, a
  documentacao oficial do PyTorch descreve `torch.nn.utils.prune`:
  <https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html>.
- Explicacao intuitiva: pruning corta partes do modelo que parecem pouco
  importantes.
- Uso no trabalho: sera uma tecnica candidata para comparacao com quantizacao.
- Propriedades exigidas no codigo: criterio de importancia, mascara ou remocao
  estrutural, taxa de poda mensuravel e impacto verificavel na saida.
- Teste de aderencia: medir fracao de pesos/estruturas removidas, comparar
  saida com baseline e verificar se a execucao usa representacao esparsa ou
  estrutura reduzida. Se apenas zerar pesos em matriz densa, classificar como
  pruning conceitual, nao como ganho de hardware.

### Sparsity

- Definicao cientifica: grau de esparsidade de uma representacao, normalmente
  associado a proporcao de elementos nulos ou conexoes ausentes em tensores,
  matrizes ou estruturas do modelo.
- Referencia confiavel: Tang et al. (2024) no contexto de compressao de
  Transformers; tutorial oficial de pruning do PyTorch para esparsificacao de
  redes neurais.
- Explicacao intuitiva: uma estrutura esparsa tem muitos zeros ou muitas
  conexoes removidas.
- Uso no trabalho: sera uma propriedade esperada de modelos podados.
- Propriedades exigidas no codigo: contagem objetiva de zeros, mascaras ou
  estruturas removidas; quando houver alegacao de hardware, uso de formato ou
  kernel que explore esparsidade.
- Teste de aderencia: calcular `zeros / total` e auditar se a multiplicacao
  realmente deixou de processar os elementos removidos. Se o custo computacional
  permanecer denso, a validade sera apenas conceitual ou numerica.

### Quantizacao

- Definicao cientifica: tecnica de representacao que mapeia valores de maior
  precisao para um conjunto menor de niveis discretos, reduzindo bits por valor
  e introduzindo erro controlado.
- Referencia confiavel: Tang et al. (2024) para quantizacao em compressao de
  Transformers; Zandieh et al., *TurboQuant*, 2025:
  <https://arxiv.org/abs/2504.19874>; documentacao `torchao` para fluxos e
  tipos de quantizacao em PyTorch:
  <https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html>.
- Explicacao intuitiva: quantizar e representar numeros com menos precisao para
  economizar memoria e potencialmente acelerar computacao.
- Uso no trabalho: sera uma tecnica candidata para reduzir custo de inferencia.
- Propriedades exigidas no codigo: escala ou codigo de quantizacao, valores em
  menor precisao ou tipo derivado, erro de dequantizacao mensuravel e, para
  hardware, kernel ou representacao que reduza custo real.
- Teste de aderencia: verificar dtype/bit-width, memoria ocupada, erro em
  relacao ao baseline e caminho de execucao. Se os tensores forem
  imediatamente convertidos para `float32`, classificar como simulacao numerica
  de quantizacao.

### Latencia

- Definicao cientifica: tempo decorrido para completar uma operacao, requisicao
  ou unidade de trabalho.
- Referencia confiavel: Hennessy e Patterson, *Computer Architecture: A
  Quantitative Approach*, texto classico de arquitetura de computadores.
- Explicacao intuitiva: latencia e quanto tempo uma execucao demora.
- Uso no trabalho: sera uma metrica de desempenho por operacao ou por passo de
  inferencia.
- Propriedades exigidas no codigo: medicao com relogio adequado, aquecimento,
  repeticoes, sincronizacao de GPU quando aplicavel e reporte de media e
  percentis.
- Teste de aderencia: executar benchmark com entradas fixas, descartar
  aquecimento, sincronizar antes/depois da medicao em GPU e reportar media, p50
  e p95.

### Throughput

- Definicao cientifica: quantidade de trabalho completada por unidade de tempo,
  tambem associada a bandwidth em sistemas computacionais.
- Referencia confiavel: Hennessy e Patterson, *Computer Architecture: A
  Quantitative Approach*.
- Explicacao intuitiva: throughput mede quanto o sistema processa por segundo.
- Uso no trabalho: podera ser medido como tokens/s, amostras/s ou operacoes/s.
- Propriedades exigidas no codigo: definicao explicita da unidade de trabalho e
  medicao em janela suficientemente estavel.
- Teste de aderencia: calcular `unidades_processadas / tempo_total` para lote,
  sequencia ou tokens definidos. Nao misturar throughput com latencia sem
  indicar a unidade.

### Memoria

- Definicao cientifica: recurso de armazenamento usado por parametros,
  ativacoes, tensores intermediarios, caches e estruturas auxiliares.
- Referencia confiavel: Hennessy e Patterson para memoria em sistemas
  computacionais; Tang et al. (2024) para reducao de memoria por compressao de
  Transformers.
- Explicacao intuitiva: memoria e o espaco necessario para guardar modelo,
  entradas e resultados intermediarios.
- Uso no trabalho: sera uma metrica central para comparar baseline, pruning e
  quantizacao.
- Propriedades exigidas no codigo: estimativa teorica de bytes e medicao de
  pico real quando possivel.
- Teste de aderencia: calcular `num_elementos * bytes_por_elemento` para
  tensores relevantes e comparar com medidores de alocacao do framework. Para
  KV cache, verificar crescimento com o comprimento do contexto.

### FLOPs

- Definicao cientifica: contagem de operacoes aritmeticas de ponto flutuante,
  usada como estimativa analitica de custo computacional.
- Referencia confiavel: Hennessy e Patterson para avaliacao quantitativa de
  desempenho; Vaswani et al. (2017) para analise de complexidade de atencao e
  arquiteturas sequenciais.
- Explicacao intuitiva: FLOPs estimam quantas contas numericas o algoritmo faz.
- Uso no trabalho: servira como custo teorico, separado da medicao real de
  tempo.
- Propriedades exigidas no codigo: formula documentada por operacao e dimensoes
  usadas no calculo.
- Teste de aderencia: derivar FLOPs esperados para multiplicacoes matriciais e
  atencao; validar com casos pequenos e manter claro que FLOPs nao equivalem
  automaticamente a latencia real.

### MSE

- Definicao cientifica: erro quadratico medio, calculado pela media dos
  quadrados das diferencas entre valores de referencia e valores estimados.
- Referencia confiavel: Hastie, Tibshirani e Friedman, *The Elements of
  Statistical Learning*: <https://hastie.su.domains/ElemStatLearn/>. Para
  definicao operacional em software, documentacao `mean_squared_error` do
  scikit-learn:
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html>.
- Explicacao intuitiva: mede erro medio dando peso maior a erros grandes.
- Uso no trabalho: comparara saidas otimizadas contra a saida baseline.
- Propriedades exigidas no codigo: comparacao ponto a ponto entre saida de
  referencia e saida candidata, com mesmo shape.
- Teste de aderencia: MSE deve ser zero para vetores identicos e positivo para
  diferencas. Testar com exemplos pequenos conhecidos.

### MAE

- Definicao cientifica: erro absoluto medio, calculado pela media dos valores
  absolutos das diferencas entre referencia e estimativa.
- Referencia confiavel: Hastie, Tibshirani e Friedman, *The Elements of
  Statistical Learning*; definicao operacional `mean_absolute_error` do
  scikit-learn:
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html>.
- Explicacao intuitiva: mede, em media, o tamanho do erro sem elevar ao
  quadrado.
- Uso no trabalho: complementara o MSE, por ser menos sensivel a erros extremos.
- Propriedades exigidas no codigo: comparacao ponto a ponto com mesmo shape.
- Teste de aderencia: MAE deve ser zero para vetores identicos e deve bater com
  calculo manual em exemplos pequenos.

### R2

- Definicao cientifica: coeficiente de determinacao, usado para avaliar quanto
  da variacao da referencia e explicada pela estimativa em relacao a um baseline
  medio.
- Referencia confiavel: Hastie, Tibshirani e Friedman; definicao operacional
  `r2_score` do scikit-learn:
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html>.
- Explicacao intuitiva: indica se a saida candidata acompanha bem a variacao da
  saida de referencia.
- Uso no trabalho: medira preservacao global do comportamento da saida em
  relacao ao baseline.
- Propriedades exigidas no codigo: referencia e candidato com mesma dimensao;
  tratamento explicito de casos constantes.
- Teste de aderencia: R2 deve ser 1 para previsoes perfeitas, 0 para predicao
  igual a media em casos nao constantes e pode ser negativo quando o candidato e
  pior que esse baseline.

### Similaridade de cosseno

- Definicao cientifica: similaridade entre vetores calculada como produto
  interno normalizado pelas normas dos vetores.
- Referencia confiavel: Manning, Raghavan e Schutze, *Introduction to
  Information Retrieval*, Cambridge University Press, 2008:
  <https://nlp.stanford.edu/IR-book/information-retrieval-book.html>. Definicao
  operacional `cosine_similarity` do scikit-learn:
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html>.
- Explicacao intuitiva: mede se dois vetores apontam em direcao parecida,
  independentemente da escala.
- Uso no trabalho: avaliara se a direcao da representacao foi preservada apos
  pruning ou quantizacao.
- Propriedades exigidas no codigo: achatamento ou agregacao definida dos
  vetores, tratamento de vetores nulos e formula normalizada.
- Teste de aderencia: vetores identicos nao nulos devem ter similaridade 1;
  vetores ortogonais devem ter similaridade 0; vetores opostos devem tender a
  -1.

## Criterios para chamar uma implementacao de Transformer

Uma implementacao so deve ser chamada de Transformer se atender aos criterios
minimos abaixo ou justificar formalmente qualquer variacao.

| Criterio | Verificacao minima |
| --- | --- |
| Entrada sequencial | Recebe sequencia de vetores ou embeddings com dimensoes explicitas. |
| Q/K/V | Projeta a entrada em consultas, chaves e valores. |
| Atencao escalonada | Implementa `softmax(QK^T / sqrt(d_k))V` ou documenta uma alternativa. |
| Dependencia entre posicoes | A saida de uma posicao pode depender de outras posicoes da sequencia. |
| Informacao posicional | Usa encoding/embedding posicional ou justifica por que o recorte nao precisa dele. |
| Componentes do bloco | Inclui ou justifica ausencia de residual, normalizacao e feed-forward quando o nome usado for bloco Transformer. |
| Referencia numerica | Tem saida comparada contra implementacao direta, biblioteca confiavel ou caso derivado manualmente. |

Classificacao correta:

- Operacao inspirada em Transformer: quando apenas uma operacao isolada for
  testada, como projecao densa ou scaled dot-product attention.
- Bloco Transformer simplificado: quando houver atencao, projecoes e parte dos
  componentes arquiteturais, mas com simplificacoes assumidas.
- Transformer: apenas quando a arquitetura tiver os elementos essenciais
  definidos pela literatura ou uma justificativa formal para cada diferenca.

## Validacao de ferramentas e algoritmos

Antes de usar uma ferramenta em benchmark, ela deve ser validada por tres
perguntas:

1. O que a ferramenta afirma implementar?
2. Qual referencia cientifica ou documentacao primaria sustenta essa afirmacao?
3. O comportamento observado confirma a afirmacao em nivel conceitual,
   numerico, algoritmico ou de hardware?

Matriz inicial de ferramentas candidatas:

| Ferramenta/algoritmo | Conceito declarado | Fonte de referencia | Verificacao necessaria | Classificacao esperada |
| --- | --- | --- | --- | --- |
| Implementacao manual de atencao | Scaled dot-product attention | Vaswani et al. (2017) | Comparar com calculo manual e/ou biblioteca de referencia. | Conceitual/algoritmica. |
| `torch.nn.MultiheadAttention` ou equivalente | Multi-head attention | Documentacao PyTorch + Vaswani et al. | Verificar shapes, mascaras, projecoes e saida contra referencia. | Algoritmica. |
| `torch.nn.utils.prune` | Pruning/esparsificacao | Tutorial oficial PyTorch + Tang et al. | Medir esparsidade, mascara e impacto na saida; verificar se ha ganho real de execucao. | Conceitual/algoritmica; hardware somente se houver kernel/formato adequado. |
| `torchao` | Quantizacao | Documentacao `torchao` + Tang et al. | Verificar dtype, bit-width, escalas, memoria e kernel usado. | Numerica/hardware se usar caminho eficiente. |
| TurboQuant | Quantizacao vetorial/KV cache | Zandieh et al. (2025), Google Research (2026) | Confirmar se ha implementacao disponivel e se o metodo usado corresponde ao paper. | Algoritmica/hardware apenas apos validacao. |

Regras de classificacao:

- Quantizacao valida para hardware exige representacao menor, kernel compativel
  ou reducao real de memoria. Caso contrario, e simulacao numerica.
- Pruning valido para hardware exige que a esparsidade seja explorada pela
  execucao. Caso contrario, e pruning conceitual.
- Uma biblioteca so sera aceita quando sua documentacao ou artigo explicar
  claramente o metodo usado e quando o comportamento observado bater com a
  definicao esperada.

## Procedimento inicial de trabalho

1. Revisar e completar este glossario com citacoes formatadas no padrao exigido
   pelo relatorio.
2. Escolher uma implementacao de referencia para atencao/Transformer.
3. Criar testes pequenos e deterministas para Q/K/V, atencao e metricas.
4. Validar ferramentas candidatas de pruning e quantizacao contra a matriz.
5. Somente depois disso, definir benchmarks de hardware.

O protocolo detalhado de passagem entre validacao e benchmark esta em
`fases/01_validacao_conceitual/docs/protocolo_experimental.md`.

As perguntas de pesquisa, hipoteses e criterios de conclusao estao em
`fases/01_validacao_conceitual/docs/perguntas_pesquisa.md`.

A validacao especifica da rede neural usada como bloco Transformer esta em
`fases/01_validacao_conceitual/docs/validacao_transformer.md`.

## Criterios de aceite desta fase

- Todo termo tecnico usado no trabalho possui referencia confiavel.
- Toda definicao possui uma propriedade verificavel no modelo ou software.
- Toda ferramenta futura pode ser avaliada pela matriz conceito-fonte-teste.
- O trabalho consegue dizer, com evidencia, se uma implementacao e valida para
  estudo conceitual, teste numerico, teste algoritmico ou teste real de
  hardware.
- Nenhuma conclusao sobre desempenho e apresentada antes da validacao
  conceitual e algoritmica.

## Referencias iniciais

- Goodfellow, I.; Bengio, Y.; Courville, A. *Deep Learning*. MIT Press, 2016.
  Disponivel em: <https://www.deeplearningbook.org/>.
- Vaswani, A. et al. *Attention Is All You Need*. arXiv:1706.03762, 2017.
  Disponivel em: <https://arxiv.org/abs/1706.03762>.
- Tang, Y. et al. *A Survey on Transformer Compression*. arXiv:2402.05964,
  2024. Disponivel em: <https://arxiv.org/abs/2402.05964>.
- Zandieh, A. et al. *TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate*. arXiv:2504.19874, 2025. Disponivel em:
  <https://arxiv.org/abs/2504.19874>.
- Google Research. *TurboQuant: Redefining AI efficiency with extreme
  compression*. 2026. Disponivel em:
  <https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/>.
- PyTorch. *Pruning Tutorial*. Disponivel em:
  <https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html>.
- PyTorch. *Quantization Overview - torchao*. Disponivel em:
  <https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html>.
- Hastie, T.; Tibshirani, R.; Friedman, J. *The Elements of Statistical
  Learning*. Springer, 2009. Disponivel em:
  <https://hastie.su.domains/ElemStatLearn/>.
- Manning, C. D.; Raghavan, P.; Schutze, H. *Introduction to Information
  Retrieval*. Cambridge University Press, 2008. Disponivel em:
  <https://nlp.stanford.edu/IR-book/information-retrieval-book.html>.
- Hennessy, J. L.; Patterson, D. A. *Computer Architecture: A Quantitative
  Approach*. Morgan Kaufmann. Referencia base para latencia, throughput,
  memoria e avaliacao quantitativa de desempenho.
- scikit-learn. *Model evaluation and metrics*. Disponivel em:
  <https://scikit-learn.org/stable/modules/model_evaluation.html>.
