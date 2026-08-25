# Auditoria do estado da IC — 25/08/2026

## Objetivo e fontes examinadas

Esta auditoria foi feita antes de ampliar a infraestrutura experimental. Foram
examinados o histórico Git completo, o código ativo, os testes, a documentação
metodológica, o protótipo legado, os materiais-fonte da apresentação, o plano
formal de trabalho e a reunião de 31/07/2026. Arquivos binários de apresentação
foram confrontados por seus fontes textuais e previews; o `.docx` do plano foi
extraído estruturalmente. A renderização visual do `.docx` não pôde ser feita
porque o ambiente não possui LibreOffice.

As fontes não têm o mesmo papel:

1. o plano formal define objetivo, operações, grade e protocolo submetidos;
2. a fala do professor delimita a sequência de validação e o foco prático;
3. a apresentação registra publicamente o desenho pretendido;
4. o código e os artefatos mostram o que existe de fato;
5. decisões deste diagnóstico são interpretações metodológicas e são marcadas
   como tais.

## Estado comprovado

| Área | Evidência observada | Avaliação |
| --- | --- | --- |
| Operações matemáticas | Projeção densa, atenção, máscaras, multi-head, FLOPs, métricas e KV cache possuem testes determinísticos. | Base conceitual sólida no recorte testado. |
| Arquitetura auxiliar | `SimplifiedTransformerBlock` contém Q/K/V, atenção, posição, residual, normalização e FFN. | É bloco simplificado; não é arquitetura Transformer completa. |
| Validação independente | NumPy manual, PyTorch manual, PyTorch SDPA e Keras/TensorFlow usam os mesmos tensores em três casos. | 9/9 comparações passaram; erro absoluto máximo `2.38418579e-07`. |
| Suíte | Comando canônico executado em 25/08/2026. | `32 passed, 1 skipped`; o `skip` exige CUDA. |
| Pipeline piloto | Runner de bloco, schema CSV e análise existem e possuem teste curto em CPU. | Útil como protótipo, mas não atende sozinho ao protocolo principal. |
| Evidências versionadas | A comparação entre frameworks possui CSV, Markdown e metadados. | Adequada para correção numérica, não para desempenho. |
| Hardware atual | Apenas Intel Iris Xe foi detectada; `nvidia-smi` ausente e PyTorch reporta zero GPUs CUDA. | A coleta principal na RTX 4070 Ti Super não pode ser feita nesta máquina. |

## Correspondência com a orientação

Atendidos:

- validação antes da otimização;
- comparação do mesmo núcleo com cálculo independente, PyTorch e
  TensorFlow/Keras;
- produção de uma tabela auditável;
- delimitação explícita de que a implementação não é uma Transformer completa;
- documentação da reunião e início da redação metodológica.

Parcialmente atendidos:

- o professor indicou self-attention como foco, mas o runner ativo mede um
  bloco inteiro;
- existe automação de tabelas e gráficos, mas ainda não há coleta canônica de
  desempenho;
- a escrita técnica é extensa, porém faltavam diário, checklist operacional e
  ligação explícita entre auditoria e decisões.

Ainda não atendidos:

- benchmark principal das operações no hardware-alvo;
- duas execuções independentes com consolidação de média e desvio-padrão;
- relatório de resultados de desempenho produzido durante a coleta.

## Inconsistências e riscos encontrados

### Escopo experimental

O runner atual mede `SimplifiedTransformerBlock`, enquanto o plano formal define
projeção densa e scaled dot-product attention e a reunião favorece a
self-attention isolada. Medir o bloco mistura FFN, normalizações e residuais com
o objeto de interesse.

**Decisão interpretativa:** o benchmark principal terá duas operações isoladas:
projeção densa, preservada por estar no plano formal, e self-attention de cabeça
única com projeções Q/K/V/saída, tratada como foco principal da orientação. O
bloco completo continuará apenas como evidência arquitetural e piloto.

### Parâmetros divergentes

O plano e os materiais de apresentação registram seed `42`, lote `1`,
`L={64,128,256}`, `D={128,256,512}`, 20 warm-ups, 50 medições e duas execuções.
O runner ativo usa por padrão seed `2026`, `L={4,8}`, `D=8`, 5 warm-ups, 20
medições e não identifica duas execuções independentes. Esses defaults são de
teste/piloto e não podem ser confundidos com o protocolo principal.

### Cenários e alegações de hardware

O plano principal contém baseline, pruning por magnitude de 50% e quantização
linear INT8. O runner adiciona `torchao_int8` por padrão, embora esse cenário não
esteja no primeiro desenho formal. Além disso:

- a poda atual mantém máscara e armazenamento denso, portanto não prova ganho
  físico;
- a quantização manual dequantiza os pesos para `float32` antes da operação,
  portanto mede fidelidade numérica, não kernel INT8;
- latência desses caminhos pode ser registrada como diagnóstico, mas não como
  evidência de aceleração por hardware;
- `max_memory_bytes` em CPU representa apenas uma estimativa incompleta e não
  deve ser chamado de pico real.

### Reprodutibilidade e rastreabilidade

- as dependências comuns usam limites mínimos, não versões fixas;
- o runner registra poucas informações de Python, sistema e bibliotecas;
- resultados padrão vão para `resultados/`, pasta ignorada pelo Git;
- não há configuração versionada para smoke e experimento principal;
- não há manifesto de execuções, hashes de entradas/saídas ou log persistente;
- o schema é verificado por nomes de colunas, mas ainda não rejeita todos os
  campos numéricos não finitos ou semanticamente inválidos;
- a análise atual escolhe um único baseline por configuração e não calcula
  incerteza entre execuções independentes.

### Viés de medição

Os cenários são medidos sempre na mesma ordem. Aquecimento, frequência,
temperatura e carga do sistema podem introduzir viés de ordem. A infraestrutura
principal deve registrar e variar deterministicamente essa ordem entre as duas
execuções.

### Cronograma

O plano declara vigência de maio a setembro de 2026, mas uma célula do
cronograma traz “2º semestre de 2027” para agosto/setembro. A reunião menciona
novembro de maneira incerta. Isso parece uma inconsistência documental, mas a
data institucional precisa ser confirmada externamente; o arquivo formal não foi
alterado.

## Conclusão da auditoria

A base matemática e a validação cruzada permitem sair da validação conceitual.
Ainda não permitem iniciar a coleta principal com o runner atual. Antes disso é
obrigatório alinhar o objeto medido, versionar configurações e evidências,
representar as duas execuções, fortalecer validações e provar a pipeline em um
smoke test. A coleta de GPU permanece condicionada à disponibilidade do
hardware-alvo.
