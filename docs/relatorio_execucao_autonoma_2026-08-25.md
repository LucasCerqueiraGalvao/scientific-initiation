# Relatório da execução autônoma — 25/08/2026

## Resumo executivo

A validação numérica da atenção está concluída, a metodologia pré-benchmark foi
auditada e o pipeline de operações isoladas foi implementado, testado e usado em
três coletas CPU versionadas. Todos os portões controláveis pelo repositório
estão fechados. A coleta principal está corretamente bloqueada porque esta
máquina não possui CUDA nem a RTX 4070 Ti Super prevista no plano.

O diagnóstico maior revelou uma ameaça à validade — pesos `N(0,1)` faziam a
escala da saída crescer com `D`. O resultado original foi preservado, o gerador
foi corrigido para Xavier/bias zero e a coleta foi repetida. A hipótese H1
(quantização preserva melhor a saída que pruning de 50%) foi compatível nos
casos executados, mas ainda é preliminar e não equivale a ganho de hardware.

## O que foi feito

1. Auditoria do repositório, cinco commits preexistentes, código ativo/legado,
   documentação, plano formal, apresentação, reunião e ambiente.
2. Arquivamento da transcrição integral e reconstrução classificada da
   orientação, separando pedidos, decisões, dúvidas e interpretações.
3. Validação NumPy/PyTorch/TensorFlow-Keras do núcleo da atenção: 9/9 aprovada.
4. Criação do diário, checklist pré-benchmark e diagnóstico metodológico.
5. Correção da semântica de hardware do runner piloto.
6. Novo runner de projeção densa/self-attention com configuração versionada,
   lotes independentes, logs, hashes, manifesto e recusa de GPU incorreta.
7. Análise com checksums, baseline por execução, média, desvio-padrão, relatório
   e gráficos.
8. Smoke CPU ponta a ponta.
9. Diagnóstico CPU `L=64`, `D=128` v1, investigação da escala dos pesos,
   correção metodológica e repetição v2.
10. Suíte final observada: `45 passed, 1 skipped`; `pip check` sem problemas.

## Commits criados nesta execução

| Commit | Etapa lógica |
| --- | --- |
| `6e10da0` | Reunião, transcrição e direcionamentos do professor. |
| `2a42a5e` | Validação da atenção entre PyTorch e TensorFlow/Keras. |
| `49f081f` | Auditoria, diário e portões pré-benchmark. |
| `cedbeb2` | Correção das alegações de hardware do piloto. |
| `7113fa5` | Runner reproduzível das operações isoladas. |
| `bcea919` | Captura do ambiente antes de escrever artefatos. |
| `a0aebe5` | Análise entre execuções e validação de integridade. |
| `1474fb5` | Smoke CPU versionado. |
| `df061d8` | Configuração do diagnóstico CPU representativo. |
| `cc03231` | Diagnóstico v1 e registro do risco de inicialização. |
| `bf7b4bd` | Pesos Xavier e schema de configuração v2. |
| `4cd1ebf` | Diagnóstico v2 e confronto metodológico v1/v2. |

Este relatório constitui a etapa final de handoff e será versionado em commit
separado após a verificação final.

## Orientações recuperadas do professor

Pedidos explícitos:

- validar a implementação contra TensorFlow/Keras usando a mesma entrada e
  semente antes de otimizar;
- comparar equação/manual, PyTorch e TensorFlow em tabela ou gráfico;
- só depois escolher uma ou duas técnicas e comparar os resultados;
- concentrar o trabalho na self-attention e nas matrizes de maior custo, não em
  uma Transformer completa;
- escrever texto corrido e fechar avaliações com resultados/tabelas desde já.

Decisões ou sugestões condicionais:

- uma operação interna equivalente basta se a camada completa ocultar detalhes;
- inferência com modelo do Hugging Face é extensão dependente de tempo;
- novembro foi citado de forma incerta e não substitui a data institucional.

Interpretações posteriores, não atribuídas ao professor:

- manter projeção densa junto da self-attention por constar no plano formal;
- usar NumPy como referência matemática;
- escolher Keras Ops, tolerâncias, schema de artefatos e níveis de validade;
- usar Xavier/bias zero para controlar a escala entre dimensões.

## Pré-benchmark

| Portão | Status | Evidência |
| --- | --- | --- |
| Reunião reconstruída sem inventar requisitos | Concluído | `docs/reunioes/2026-07-31/`. |
| Atenção validada entre referências | Concluído | 9/9 em `evidencias/comparacao_frameworks/`. |
| Recorte e cenários fixados | Concluído | Projeção densa, self-attention, baseline/pruning/quantização. |
| Configuração formal codificada | Concluído | Seed 42, grade, 20/50, duas execuções. |
| Rastreabilidade e integridade | Concluído | Config, ambiente, log, manifesto, SHA-256 e fingerprints. |
| Análise entre execuções | Concluído | Baseline da mesma execução, média e DP. |
| Inicialização controlada | Concluído | Schema v2, Xavier e confronto v1/v2. |
| Smoke ponta a ponta | Concluído | 12 registros e reprodutibilidade aprovada. |
| GPU-alvo disponível | Bloqueado externamente | Intel Iris Xe; `nvidia-smi` ausente; CUDA count 0. |

## Benchmarks e validações executados

### Comparação da atenção entre frameworks

- Objetivo: provar correção do núcleo antes de otimizar.
- Configuração: três casos, CPU/float32, seed 2026, `atol=1e-6`, `rtol=1e-5`.
- Resultado: 9/9; erro absoluto máximo `2.38418579e-07`.
- Interpretação: equivalência numérica nos casos testados; nenhuma comparação de
  desempenho.

### Smoke CPU

- Objetivo: validar pipeline completa.
- Configuração: `B=1`, `L=4`, `D=8`, 2 warm-ups, 5 medições, 2 execuções.
- Resultado: 12 registros, checksums e hashes aprovados; gráficos legíveis.
- Interpretação: infraestrutura aprovada; tempos muito variáveis e sem valor de
  hardware. Quantização teve MSE menor que pruning nas duas operações.

### Diagnóstico CPU v1

- Objetivo: executar um ponto formal com 20/50 e duas execuções.
- Configuração: `B=1`, `L=64`, `D=128`, pesos `N(0,1)`.
- Resultado: H1 compatível, mas MSE da atenção quantizada `732.65`.
- Interpretação: resultado legítimo que revelou escala inadequada do baseline;
  não foi descartado nem “melhorado”.

### Diagnóstico CPU v2

- Objetivo: repetir o mesmo ponto após controlar a inicialização.
- Configuração: mesmos parâmetros, pesos Xavier e bias zero.
- Resultado: na projeção, MSE `7.7789e-05` (quantização) contra `0.0727207`
  (pruning); na atenção, `1.63925e-05` contra `0.0110127`. R²/cosseno da atenção
  quantizada: `0.999647`/`0.999824`.
- Interpretação: H1 compatível nesse ponto; escala absoluta controlada. Latência
  CPU abaixo do baseline em algumas barras não autoriza aceleração devido ao
  ruído e à ausência de kernels especializados.

## Problemas e riscos encontrados

- GPU-alvo indisponível nesta máquina.
- Runner anterior media bloco inteiro, divergindo do recorte de operações.
- Defaults anteriores divergiam da seed/grade/20/50/duas execuções do plano.
- Quantização manual executa pesos dequantizados `float32`; pruning permanece
  denso. Nenhum dos dois sustenta hoje alegação de kernel INT8/esparso.
- Memória em CPU é estimativa de armazenamento dos tensores, não pico real.
- Latências pequenas apresentaram alta variabilidade; duas execuções são o
  mínimo formal, não uma amostra estatística ampla.
- Requisitos comuns têm versões mínimas; cada coleta registra o snapshot exato.
- O plano declara maio-setembro de 2026, mas o cronograma contém uma célula com
  “2º semestre de 2027”; a reunião menciona novembro sem confirmação.
- O `.docx` formal foi extraído estruturalmente, mas não renderizado visualmente
  porque LibreOffice não está disponível.

## Pendências

### Críticas

- executar a grade principal na RTX 4070 Ti Super com CUDA;
- antes de alegar ganho físico, adotar e validar formato/kernel realmente INT8
  e/ou esparso; os cenários simulados atuais só sustentam validade numérica ou
  algorítmica.

### Importantes

- confirmar o calendário institucional e a inconsistência 2026/2027;
- incorporar tabelas e discussão v1/v2 ao texto acadêmico;
- após a coleta GPU, revisar ruído, outliers e suficiência de duas execuções;
- decidir, com o orientador, se a entrega formal manterá quantização/pruning
  simulados ou incluirá uma extensão de kernel real.

### Melhorias futuras

- modelo pré-treinado do Hugging Face;
- TorchAO/kernel INT8 e esparsidade estruturada em etapa própria;
- mais seeds/lotes/heads e intervalos de confiança;
- benchmark do bloco completo somente como extensão;
- comparação de desempenho entre frameworks, separada da validação numérica.

## Próximo passo recomendado

Mover o repositório para a máquina com RTX 4070 Ti Super, instalar o mesmo
ambiente, rodar primeiro a suíte e o smoke v2, confirmar o nome da GPU e então
executar `experimentos/benchmark_principal_gpu.json` em pasta nova. A coleta deve
ser analisada imediatamente e não pode ser chamada de ganho INT8/esparso sem uma
segunda etapa que comprove armazenamento e kernel compatíveis.
