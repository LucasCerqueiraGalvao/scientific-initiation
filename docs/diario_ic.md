# Diário da iniciação científica

Este diário registra a evolução da pesquisa, as razões das decisões e as
evidências produzidas. Entradas anteriores à criação do arquivo são marcadas
como reconstruídas a partir de fontes versionadas.

## 31/07/2026 — Reunião de orientação (entrada reconstruída)

**Objetivo e contexto.** Revisar como validar o código já desenvolvido e definir
o caminho até os experimentos.

**Análises e decisões.** O orientador pediu comparação com TensorFlow/Keras e
PyTorch usando entradas e sementes equivalentes antes de qualquer otimização.
Ficou definido que resultados pequenos deveriam virar tabelas/gráficos e que a
self-attention seria o foco, evitando o custo de otimizar a arquitetura inteira.
Uma ou duas técnicas seriam escolhidas apenas após esse portão. A escrita do
relatório deveria começar imediatamente.

**Hipótese metodológica.** Se o mesmo núcleo produzir saídas equivalentes na
implementação manual e em bibliotecas consolidadas, diferenças posteriores
podem ser atribuídas com mais segurança às técnicas, e não a um erro básico da
atenção.

**Pendências deixadas.** Comparação independente, definição detalhada dos
benchmarks, tabelas de resultados, redação e confirmação das datas oficiais.

**Fonte.** Ata e transcrição em `docs/reunioes/2026-07-31/`.

## 24/08/2026 — Validação cruzada da atenção (entrada reconstruída)

**Objetivo.** Fechar o portão numérico solicitado na reunião.

**Implementação.** Foi criado `validacao.comparacao_frameworks` com três casos
determinísticos e os mesmos tensores NumPy compartilhados entre NumPy manual,
PyTorch manual, PyTorch SDPA e Keras/TensorFlow. O adaptador trata os layouts
`[B,H,T,D]` e `[B,T,H,D]`. TensorFlow é importado tardiamente e executado em CPU,
`float32`, sem dropout ou flash attention.

**Testes e resultados.** As nove comparações passaram com `atol=1e-6` e
`rtol=1e-5`; erro absoluto máximo `2.38418579e-07`. A suíte completa retornou
`32 passed, 1 skipped`; o teste ignorado exige CUDA. CSV, tabela Markdown e JSON
de metadados foram versionados em `evidencias/comparacao_frameworks/`.

**Interpretação.** A evidência sustenta equivalência numérica do núcleo nos casos
testados. Não sustenta velocidade, economia de memória ou superioridade entre
frameworks.

**Próximo passo.** Auditar e alinhar o runner de desempenho ao plano formal e ao
recorte da reunião antes de executar benchmarks principais.

## 25/08/2026 — Auditoria integral e portões pré-benchmark

**Objetivo.** Revisar criticamente repositório, histórico, código, documentação,
plano formal, apresentação, reunião e ambiente antes de implementar nova coleta.

**O que foi analisado.** Cinco commits preexistentes, código ativo e legado,
testes, requisitos, schemas, análise de resultados, materiais textuais da
apresentação, plano de trabalho e transcrição. O plano foi extraído
estruturalmente; a verificação visual do `.docx` ficou indisponível pela ausência
de LibreOffice.

**Problemas encontrados.** O runner ativo mede bloco inteiro; usa parâmetros de
piloto diferentes do plano; não representa duas execuções; grava por padrão em
pasta ignorada; registra ambiente insuficiente; usa ordem fixa de cenários; tem
semântica ambígua de memória; e inclui TorchAO por padrão fora do primeiro
desenho formal. A análise atual não estima variabilidade entre execuções.

**Decisões metodológicas.** O benchmark principal medirá projeção densa e
self-attention isoladas. A primeira preserva o plano formal; a segunda é o foco
explicitado pelo professor. O bloco simplificado continuará como validação
arquitetural/piloto. O schema CSV atual será preservado, e informações adicionais
irão para configuração, manifesto, log e metadados.

**Ambiente.** Python `3.11.9`, PyTorch `2.11.0+cu128` e build CUDA `12.8`, mas
`torch.cuda.is_available()` retornou falso, com zero dispositivos. O Windows só
expôs Intel Iris Xe; `nvidia-smi` não está instalado/disponível.

**Testes.** Suíte completa: `32 passed, 1 skipped`. Comparação canônica:
`9/9` aprovada. `pip check`: nenhuma dependência quebrada.

**Correção do runner piloto.** A evidência de armazenamento de baixa precisão
da quantização manual foi removida: o `int8` é intermediário e o peso executado
permanece `float32`. TorchAO também saiu dos cenários padrão, permanecendo como
extensão explícita. O metadado agora diferencia pico CUDA de estimativa de
armazenamento em CPU. Teste direcionado: `7 passed, 1 skipped` (CUDA).

**Resultados e interpretação.** A validação conceitual está madura, mas o
benchmark principal ainda não está liberado. A indisponibilidade da GPU é
externa; os demais portões controláveis serão tratados antes do smoke test.

**Arquivos criados/alterados nesta etapa.** `docs/diario_ic.md`,
`docs/auditoria_estado_2026-08-25.md`, `docs/checklist_pre_benchmark.md`, índice
de documentação e reconstrução classificada da reunião.

**Próximos passos.** Implementar configuração e runner das operações isoladas,
fortalecer validações, persistir evidências, executar smoke reprodutível e então
reavaliar a liberação da coleta principal.

### Continuação — infraestrutura das operações isoladas

**Implementação.** Foram adicionadas configurações JSON separadas para smoke em
CPU e experimento principal em CUDA. `validacao.benchmark_operacoes` carrega e
valida o contrato, gera dados sintéticos determinísticos, mede projeção densa e
self-attention de cabeça única e executa baseline, poda de 50% e quantização
INT8 simulada. Cada execução independente grava seu próprio CSV e metadados; um
manifesto com checksums e um log permitem preservar resultados parciais.

**Controles metodológicos.** O dispositivo não usa `auto`; o benchmark principal
falha se CUDA ou a GPU nominal estiverem ausentes. As duas execuções reutilizam
os mesmos dados/pesos e rotacionam deterministicamente a ordem dos cenários. Os
hashes confirmam entradas e saídas determinísticas. A poda continua densa e a
quantização executa pesos dequantizados em `float32`, fato registrado nos
metadados para impedir alegações de kernel otimizado.

**Teste.** `test_benchmark_operacoes.py`: `6 passed`. O teste verifica o contrato
formal, o portão de GPU, a poda exata, a semântica da quantização, a
reprodutibilidade entre execuções e todos os artefatos persistidos.

**Pendência imediata.** Consolidar execuções com média/desvio-padrão, integrar a
análise à pipeline e só então produzir a evidência canônica do smoke test.

### Continuação — análise entre execuções

**Implementação.** `validacao.analise_benchmark_operacoes` verifica os SHA-256
do manifesto, valida a completude da grade, confere hashes determinísticos e
associa cada candidato ao baseline da mesma execução. A análise preserva os
registros individuais, calcula média e desvio-padrão entre execuções e gera
comparações, relatório e gráficos.

**Proteção contra interpretação indevida.** O relatório identifica o propósito
do experimento. Para smoke em CPU, tempos e memória são rotulados como
diagnóstico da pipeline. A cadeia pergunta → hipótese → configuração → execução
→ resultado → interpretação fica explícita, e caminhos densos/dequantizados não
são promovidos a hardware.

**Testes.** Dois testes direcionados passaram: geração completa dos oito
artefatos de análise e rejeição de CSV adulterado por divergência de checksum.

**Próximo passo.** Fazer commit desta infraestrutura, executar o smoke canônico
em diretório versionável novo, inspecionar seus artefatos e repetir a análise.

### Continuação — smoke canônico e portão da GPU

**Objetivo.** Validar ponta a ponta leitura da configuração, geração dos dados,
execução, métricas, persistência, logs, integridade, análise e visualização antes
de qualquer grade maior.

**Configuração.** CPU, seed 42, `B=1`, `L=4`, `D=8`, projeção densa e
self-attention, três cenários, 2 warm-ups, 5 medições e 2 execuções
independentes. Evidência em
`evidencias/benchmarks/smoke_cpu_2026-08-25/`.

**Resultados da pipeline.** Manifesto `complete`; 6 registros por execução e 12
no total; CSVs e metadados passaram nos checksums; hashes de entrada e saída
coincidiram entre execuções; oito artefatos de análise foram gerados; os dois
gráficos foram inspecionados e estão legíveis.

**Qualidade da saída.** Na projeção densa, MSE de `7.28672e-05` para quantização
contra `0.352201` para pruning. Na self-attention, MSE de `0.00503124` contra
`25.8574`. O sinal é compatível com H1 nesta amostra sintética: a quantização
simulada preservou melhor a saída que a poda de 50%.

**Latência e limitação.** Os casos minúsculos apresentaram grande variabilidade
entre as duas execuções. A quantização simulada também executa dequantização em
`float32`, e a poda usa matriz densa. Portanto, razões de latência do smoke não
são evidência de aceleração nem de hardware.

**Portão negativo da GPU.** A tentativa controlada com a configuração principal
terminou com código 2 e mensagem de CUDA indisponível, antes de criar qualquer
diretório de saída. Isso confirma que o runner não fará coleta principal
acidental em CPU.

**Decisão.** O smoke está aprovado. A grade principal continua bloqueada apenas
pela ausência da RTX/CUDA neste ambiente. Como trabalho independente, será
executado um diagnóstico CPU em um ponto representativo da grade formal, sem
promovê-lo a resultado de hardware.

### Continuação — diagnóstico CPU v1 em L=64, D=128

**Objetivo e configuração.** Exercitar o primeiro ponto da grade formal com o
protocolo completo de 20 warm-ups, 50 medições e duas execuções, preservando
seed 42 e os mesmos cenários. A coleta terminou em aproximadamente dois segundos
de tempo registrado pelo log e gerou 12 registros analisáveis.

**Integridade e reprodutibilidade.** Manifesto completo, checksums válidos e
hashes de entradas/saídas idênticos entre execuções. Os gráficos foram
inspecionados e estão legíveis.

**Resultado de qualidade.** Na projeção densa, MSE `0.00995697` da quantização
contra `9.30825` do pruning. Na self-attention, MSE `732.65` contra `12144.35`;
R² da quantização `0.95428` e cosseno `0.97714`. H1 continua compatível, mas o
erro absoluto chamou atenção.

**Investigação do resultado inesperado.** O gerador v1 usa pesos independentes
`N(0,1)`. Em matrizes maiores, essa variância faz projeções e saída da atenção
crescerem com `D`. Assim, MSE/MAE aumentam também pela escala do baseline e não
podem ser comparados diretamente entre dimensões. Não se trata de adulterar um
resultado ruim: a coleta v1 foi preservada e identificou uma ameaça à validade.

**Decisão metodológica.** Antes do benchmark principal, tornar a distribuição
dos dados explícita na configuração e usar inicialização de pesos escalada pela
dimensão (Xavier para matrizes quadradas), mantendo entradas `N(0,1)` e bias
zero. O diagnóstico será repetido como v2; v1 continuará versionado como
evidência da correção.

**Latência.** As razões em CPU sugeriram valores abaixo do baseline em alguns
cenários, mas com desvio alto e sem kernel esparso/INT8. Nenhuma aceleração foi
concluída.

### Continuação — correção metodológica da inicialização

**Mudança.** O schema de configuração passou à versão 2 e agora registra
explicitamente distribuição das entradas, inicialização dos pesos e bias. As
entradas continuam `N(0,1)`; matrizes quadradas usam Xavier normal com desvio
`1/sqrt(D)`; o bias da projeção densa é zero. Evidências de schema v1 continuam
legíveis para preservar o histórico.

**Justificativa.** A mudança não foi escolhida para melhorar uma métrica
observada, mas para controlar a variância das projeções ao comparar diferentes
dimensões. A coleta v1 permanece intacta e será confrontada com v2.

**Testes.** A suíte passou com `45 passed, 1 skipped`. Um teste novo verifica a
escala empírica dos pesos e o bias zero; as evidências v1 foram reabertas e
mantiveram checksums e reprodutibilidade válidos.

**Próximo passo.** Commit da correção e repetição do diagnóstico em nova pasta
v2, sem sobrescrever o resultado anterior.
