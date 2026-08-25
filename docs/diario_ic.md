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
