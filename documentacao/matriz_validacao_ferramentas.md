# Matriz de validacao de ferramentas

Esta matriz registra se cada ferramenta candidata implementa, de forma
observavel, o conceito que afirma implementar. Ela deve ser atualizada sempre
que novos testes forem executados.

Os valores numericos derivados das definicoes e a comparacao com as fontes
primarias estao registrados em `documentacao/confronto_resultados_literatura.md`.

Status possiveis:

- Aderente: o comportamento observado bate com a definicao e com o teste.
- Parcialmente aderente: a ferramenta implementa parte do conceito, mas nao
  sustenta todas as conclusoes desejadas.
- Nao aderente: o comportamento observado nao sustenta o uso do conceito.
- Pendente: ainda nao foi possivel executar ou completar a verificacao.
- Fora do pipeline: ferramenta conhecida, mas excluida do fluxo atual por
  depreciacao, incompatibilidade ou falta de evidencia adequada.

Niveis de validade:

- Conceitual: serve para demonstrar a ideia.
- Numerico: permite medir erro ou fidelidade da saida.
- Algoritmico: implementa de fato o algoritmo esperado.
- Hardware: usa representacao, kernel ou estrutura capaz de sustentar alegacao
  de ganho fisico.
- Fora do pipeline: nao participa dos benchmarks ou testes automatizados.

Niveis de certeza usados no metodo:

- Definicao: termo sustentado por fonte cientifica ou primaria.
- Matematica: formula local bate com caso pequeno derivado.
- Algoritmo: ferramenta bate com implementacao transparente ou referencia.
- Hardware: execucao usa representacao, formato ou kernel compativel com ganho
  fisico mensuravel.

| Ferramenta/algoritmo | Conceito declarado | Fonte cientifica ou primaria | Teste aplicado | Resultado observado | Status | Nivel de validade |
| --- | --- | --- | --- | --- | --- | --- |
| Implementacao manual em `validacao.dense` | Projecao densa | Goodfellow et al. + Vaswani et al. (2017) | Comparar `Y=XW^T+b` com calculo manual e NumPy. | Passou em `pytest`: saida `[[1.5, 3.5], [3.5, 7.5]]` bate com a formula. | Aderente | Numerico/algoritmico |
| Implementacao manual em `validacao.attention` | Scaled dot-product attention | Vaswani et al. (2017), `Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V` | Comparar tensores pequenos contra calculo NumPy independente. | Passou em `pytest`: saida PyTorch manual igual ao calculo NumPy independente dentro de tolerancia `1e-6`. | Aderente | Conceitual/algoritmico |
| `torch.nn.functional.scaled_dot_product_attention` | Scaled dot-product attention | Documentacao PyTorch + Vaswani et al. (2017) | Comparar saida PyTorch com implementacao manual transparente. | Passou em `pytest`: saida da API PyTorch igual a implementacao manual transparente dentro de tolerancia `1e-6`. | Aderente | Algoritmico |
| `validacao.attention` com mascara e multi-head | Multi-head/self-attention | Vaswani et al. (2017) + documentacao PyTorch SDPA | Validar split/combine de heads, shape e mascara aditiva contra PyTorch. | Passou em `pytest`: saida multi-head preserva shape `[batch, seq, d_model]` e bate com referencia por heads em `1e-6`. | Aderente | Algoritmico |
| `validacao.transformer.SimplifiedTransformerBlock` | Bloco Transformer simplificado | Vaswani et al. (2017) + criterios do documento teorico | Verificar entrada sequencial, Q/K/V, atencao, posicao, residual, norm e FFN. | Passou em `pytest`: saida preserva shape e checklist classifica como `bloco_transformer_simplificado`. | Aderente | Conceitual/algoritmico |
| `validacao.auditoria_transformer` | Auditoria arquitetural de NN Transformer | Vaswani et al. (2017) + criterios do protocolo | Auditar componentes, comportamento sequencial, attention interna e controle negativo. | Passou em `pytest`: o bloco atual e classificado como `bloco_transformer_simplificado`; uma NN linear comum e rejeitada como `nao_aderente`. | Aderente | Conceitual/algoritmico |
| `validacao.kv_cache` | KV cache | Literatura/documentacao de inferencia autoregressiva e compressao de KV cache | Comparar atencao causal completa com versao incremental usando cache. | Passou em `pytest`: saida com cache bate com saida completa em `1e-6`; para 4 tokens, projeta 4 K/V contra 10 no caminho ingenuo. | Aderente | Algoritmico; hardware pendente |
| `validacao.flops` | FLOPs teoricos | Hennessy e Patterson + analise de complexidade em Vaswani et al. (2017) | Calcular casos pequenos para projecao densa, attention e bloco. | Passou em `pytest`: projecao densa `270`, attention `288`, bloco `1992` FLOPs no caso definido. | Aderente | Numerico |
| `validacao.protocolo` | Schema experimental CSV | Protocolo metodologico do projeto | Validar colunas fixas e nivel de validade permitido. | Passou em `pytest`: schema preserva 24 colunas e rejeita nivel sem evidencia. | Aderente | Conceitual |
| `validacao.pesquisa` | Perguntas de pesquisa e hipoteses | Metodo experimental do projeto | Verificar se toda pergunta tem metrica, evidencia e hipotese vinculada. | Passou em `pytest`: contrato de pesquisa nao apresenta inconsistencias. | Aderente | Conceitual |
| `validacao.benchmark_controlado` | Runner experimental controlado | Protocolo experimental do projeto + PyTorch | Rejeitar cenario sem matriz, registrar ambiente/seed, medir com repeticoes e validar CSV. | Passou em `pytest`: registros de benchmark sao validos, baseline tem erro zero e cenarios nao sao promovidos indevidamente a hardware. | Aderente | Algoritmico |
| `validacao.analise_resultados` | Analise final e evidencias do artigo | Protocolo experimental do projeto | Comparar cenarios contra baseline, gerar tabelas/graficos e conclusoes com fonte. | Passou em `pytest`: tabelas, graficos e conclusoes apontam para CSV e matriz. | Aderente | Algoritmico |
| Checklist em `validacao.classificacao` | Classificacao de operacao/bloco/Transformer | Vaswani et al. (2017) + criterios do documento teorico | Verificar casos de operacao inspirada, bloco simplificado e Transformer. | Passou em `pytest`: os casos minimos sao classificados corretamente. | Aderente | Conceitual |
| Metricas manuais em `validacao.metrics` | MSE, MAE, R2 e similaridade de cosseno | Hastie, Tibshirani e Friedman; Manning, Raghavan e Schutze; scikit-learn | Comparar exemplos pequenos contra resultados conhecidos e scikit-learn. | Passou em `pytest`: metricas manuais batem com valores conhecidos e com scikit-learn. | Aderente | Numerico |
| `torch.nn.utils.prune` | Pruning/esparsificacao | Tang et al. (2024) + tutorial oficial PyTorch pruning | Aplicar poda L1 em camada linear pequena e medir proporcao de zeros. | Passou em `pytest`: poda L1 produziu 50% de zeros e mascara `weight_mask`; nao usa armazenamento/kernel esparso no teste. | Parcialmente aderente | Conceitual/algoritmico; hardware pendente |
| Quantizacao simetrica `int8` em `validacao.quantization` | Quantizacao numerica | Tang et al. (2024), Zandieh et al. (2025) | Verificar dtype `int8`, reducao de bytes e erro de dequantizacao. | Passou em `pytest`: tensor quantizado usa `int8`, reduz bytes e erro de dequantizacao fica limitado pela escala. | Aderente | Numerico |
| `torch.ao.quantization.quantize_dynamic` | Quantizacao dinamica de camada linear | Documentacao PyTorch quantization | Avaliar manutencao da ferramenta no pipeline. | Removida da suite automatizada: API depreciada no ambiente atual e substituida por `torchao` v2. | Fora do pipeline | Fora do pipeline |
| `torchao` v2 | Quantizacao e otimizacao nativa PyTorch | Documentacao oficial torchao | Quantizar camada linear pequena com `Int8WeightOnlyConfig(version=2)` e verificar armazenamento interno. | Passou em `pytest`: camada roda sem warnings, peso e `Int8Tensor` e dados internos `qdata` usam `int8`. | Aderente | Numerico/algoritmico; hardware pendente |

## Regras para atualizar a matriz

1. Nenhuma ferramenta deve ser promovida para nivel hardware apenas por ter
   menor erro numerico.
2. Quantizacao so recebe nivel hardware se houver representacao menor e caminho
   de execucao compativel com essa representacao.
3. Pruning so recebe nivel hardware se a esparsidade for explorada por formato
   ou kernel adequado.
4. Se uma ferramenta passar apenas em testes pequenos, registrar validade
   conceitual/algoritmica e deixar a validade de hardware como pendente.
