# Reunião de 31 de julho de 2026

## Identificação

- **Tema:** validação da implementação, definição do recorte experimental e
  preparação dos benchmarks;
- **Participantes identificados na transcrição:** Lucas Galvão e Walter Silva
  Oliveira;
- **Horário registrado:** início às 16:01 (GMT-03:00);
- **Duração da transcrição:** 30 min e 7 s;
- **Registro integral:** [transcricao-completa.md](transcricao-completa.md);
- **Fonte no Google Docs:** [anotações e transcrição](https://docs.google.com/document/d/1WBgwj1gDAWh_7VGckHvPhRfSlcRzFIwL2JlnIDhRYbQ/edit?usp=drive_web&tab=t.v1ze8hoqd423);
- **Gravação indicada na ata:** [Google Drive](https://drive.google.com/file/d/1-FFyxhkpgkhlesSqULSIaujRSiDnT1UN/view?usp=drive_web).

A transcrição foi gerada automaticamente e pode conter erros de reconhecimento.
A cópia integral foi preservada sem correções; esta página concentra apenas a
leitura útil para o projeto.

## Leitura executiva

A reunião definiu uma sequência de trabalho: primeiro demonstrar que a
implementação local da atenção reproduz referências conhecidas usando as mesmas
entradas, pesos e sementes; depois aplicar uma ou duas técnicas de otimização e
compará-las por benchmarks controlados. O recorte recomendado foi o mecanismo de
autoatenção, sobretudo suas operações matriciais mais custosas, e não uma
Transformer completa. Tabelas, gráficos e texto acadêmico devem ser produzidos
junto com os experimentos, em vez de ficarem para o final.

## Decisões e direcionamentos

1. **Validar antes de otimizar.** Confrontar a implementação local com cálculo
   manual e bibliotecas consolidadas, mantendo entrada, pesos, semente e
   tolerância comparáveis.
2. **Isolar o objeto de estudo.** Priorizar a autoatenção e explicitar que o
   projeto não pretende implementar nem otimizar toda a arquitetura Transformer.
3. **Transformar testes em evidência.** Registrar resultados em tabelas e
   gráficos, acompanhados do passo a passo e da fundamentação teórica.
4. **Comparar poucas técnicas de forma controlada.** Depois do portão de
   validação, selecionar uma ou duas otimizações e medi-las contra o mesmo
   baseline.
5. **Começar a redação imediatamente.** Conectar conceitos, método, resultados e
   limitações enquanto os experimentos são executados.
6. **Manter modelos pré-treinados como extensão.** Uma inferência com modelo do
   Hugging Face pode enriquecer o trabalho, mas somente se houver tempo depois do
   núcleo experimental.
7. **Confirmar as datas oficiais.** A conversa mencionou novembro como referência
   provável para entrega/apresentação, mas sem confirmação institucional.

## Reconstrução classificada da orientação

Esta seção distingue o que aparece na fala do orientador das decisões e
interpretações feitas depois. Os intervalos remetem à transcrição integral.

### Pedidos explícitos do professor

- comparar a implementação local com uma biblioteca consolidada, citando
  TensorFlow ou Keras, com a mesma entrada e a mesma semente (`00:00:00–04:12`);
- validar passo a passo antes de aplicar técnicas de otimização (`00:02:53–05:35`);
- materializar a validação em tabela ou gráfico, comparando cálculo/equação,
  implementação manual, PyTorch e TensorFlow (`00:05:35–08:01`);
- depois da validação, escolher uma ou duas técnicas da metodologia e comparar
  os resultados (`00:08:01–09:18`);
- concentrar o estudo no mecanismo de self-attention e em suas operações de
  maior custo, em vez de otimizar uma Transformer inteira (`00:09:18–14:09`);
- começar a escrever texto corrido imediatamente e fechar as avaliações com
  resultados e tabelas (`00:23:13–24:34`).

### Decisões tomadas durante a reunião

- a ordem de trabalho ficou como validação da implementação, seleção das
  técnicas, aplicação controlada e comparação;
- o recorte principal ficou na self-attention, com a limitação de escopo e de
  tempo usada como justificativa;
- se uma camada completa de TensorFlow ocultasse o núcleo, bastaria comparar
  uma operação/interação equivalente da atenção (`00:14:09–15:21`);
- uma demonstração com modelo pré-treinado poderia ser feita somente se
  houvesse tempo (`00:15:21` em diante).

### Dúvidas levantadas na reunião

- como garantir que a implementação própria está correta;
- como definir formalmente o que pode ser chamado de Transformer;
- até onde é necessário validar a arquitetura, além das operações matemáticas;
- se TensorFlow expõe o núcleo interno de atenção de forma comparável;
- se haverá tempo para uma inferência com modelo do Hugging Face;
- quais são as datas institucionais efetivas de entrega e apresentação.

### Sugestões metodológicas, sem caráter de requisito adicional

- usar resultados pequenos e palpáveis para sustentar cada avanço;
- justificar o recorte pela relevância computacional do mecanismo escolhido e
  pelo tempo disponível;
- usar literatura para selecionar técnicas já conhecidas, sem exigir uma nova
  técnica de otimização;
- escrever durante o desenvolvimento para explicitar como os conceitos se
  encadeiam.

### Implementações e validações necessárias derivadas diretamente da reunião

- adaptador de layout e execução do mesmo núcleo de atenção em
  NumPy/manual, PyTorch e TensorFlow/Keras;
- casos determinísticos com entradas compartilhadas e tolerância explícita;
- artefato tabular com resultados e erros;
- benchmark controlado da self-attention com baseline comum e uma ou duas
  técnicas, depois da validação numérica;
- documentação em prosa de método, resultados e limites.

### Tarefas que podem ficar para depois dos primeiros benchmarks

- demonstração de inferência com modelo pré-treinado;
- ampliação para outros componentes ou para uma arquitetura Transformer
  completa;
- técnicas adicionais além do primeiro conjunto controlado.

### Interpretações posteriores — não são pedidos literais do professor

- usar NumPy como referência matemática independente;
- escolher `keras.ops.nn.dot_product_attention` como API TensorFlow/Keras;
- fixar `float32`, CPU, `atol=1e-6`, `rtol=1e-5` e seed `2026` na validação
  cruzada;
- manter CSV e JSON de metadados versionados;
- classificar evidências como conceituais, numéricas, algorítmicas ou de
  hardware;
- tratar a RTX 4070 Ti Super, seed `42`, grade, aquecimento e repetições como
  requisitos do plano formal de trabalho, não desta reunião.

## Situação no repositório em 24/08/2026

| Encaminhamento da reunião | Situação atual | Evidência ou lacuna |
| --- | --- | --- |
| Validar a matemática da atenção | Implementado no código e nos testes | Há comparação da atenção local com NumPy independente, valores controlados e `torch.nn.functional.scaled_dot_product_attention` em [test_validacao_conceitual.py](../../../fases/01_validacao_conceitual/tests/test_validacao_conceitual.py). |
| Comparar com bibliotecas consolidadas | Concluído para o núcleo da atenção | NumPy manual, PyTorch manual, PyTorch SDPA e Keras SDPA com backend TensorFlow foram comparados nos mesmos casos. As 9 comparações passaram; veja a [evidência versionada](../../../fases/01_validacao_conceitual/evidencias/comparacao_frameworks/comparacao_atencao.md). |
| Restringir o escopo à autoatenção | Parcialmente alinhado | A validação isola a atenção, mas o executor principal mede um `SimplifiedTransformerBlock`; é preciso decidir se o benchmark final será da atenção isolada ou do bloco simplificado. Ver [README da fase](../../../fases/01_validacao_conceitual/README.md). |
| Preparar benchmarks controlados | Infraestrutura pronta; execução canônica pendente | O executor, o esquema do CSV e os testes de contrato existem em [benchmark_controlado.py](../../../fases/01_validacao_conceitual/validacao/benchmark_controlado.py), mas não há resultados experimentais versionados. |
| Produzir tabelas, gráficos e conclusões | Automação pronta; artefatos pendentes | [analise_resultados.py](../../../fases/01_validacao_conceitual/validacao/analise_resultados.py) processa os CSVs, mas depende da coleta real. |
| Formalizar a metodologia | Em andamento avançado | A fase já possui protocolo, matriz de ferramentas, perguntas de pesquisa e documentação de validação em [docs da fase](../../../fases/01_validacao_conceitual/docs/). |
| Usar modelo pré-treinado | Não iniciado e não obrigatório | Deve continuar fora do caminho crítico até que validação, benchmarks e relatório estejam fechados. |

Essa tabela descreve o que está implementado e documentado; ela não afirma que a
bateria final de benchmarks já foi executada no ambiente-alvo.

## Próximos passos priorizados

### P0 — Fechar o contrato experimental

1. Escolher explicitamente o objeto medido no resultado principal: atenção
   isolada, como favorecido pela reunião, ou bloco Transformer simplificado.
2. Fixar baseline, técnicas de otimização, shapes, lotes, tamanhos de sequência,
   `dtype`, dispositivo, sementes, aquecimento, repetições e tolerâncias.
3. Preservar TensorFlow/Keras como portão de correção já concluído e reexecutar a
   comparação somente quando mudar a atenção, as versões dos frameworks ou as
   tolerâncias.
4. Confirmar com a orientação as datas institucionais de submissão, entrega e
   apresentação.

### P1 — Concluir a validação e coletar evidências

1. Reexecutar a suíte no ambiente fixado e guardar versão das dependências,
   hardware e resultado dos testes.
2. Fazer um piloto curto em CPU para validar o fluxo de ponta a ponta.
3. Executar a grade final no dispositivo-alvo, salvando CSV e metadados. Como
   `resultados/` está ignorado pelo Git, definir antes quais artefatos finais
   serão versionados ou publicados em outro local reprodutível.
4. Distinguir claramente simulação numérica de ganho efetivo de hardware; só usar
   esse último rótulo quando armazenamento, tipo numérico e kernel executado
   sustentarem a afirmação.

### P1 — Escrever o relatório junto com a análise

1. Gerar uma tabela de validação com entrada, saída esperada, saída local, saída
   da referência, erro máximo, tolerância e status.
2. Gerar tabelas e gráficos de latência, throughput, memória, erro e redução de
   operações para cada cenário contra o mesmo baseline.
3. Responder às perguntas de pesquisa com os resultados observados e registrar
   ameaças à validade, limitações e casos em que uma técnica não trouxe ganho.
4. Incorporar método, resultados e discussão ao texto acadêmico sem esperar o
   encerramento de toda a coleta.

### P2 — Extensão opcional

Depois de concluir o núcleo, avaliar uma demonstração de inferência com um modelo
pré-treinado do Hugging Face. A extensão não deve alterar o baseline nem atrasar
as entregas principais.

## Critério de conclusão desta etapa

A etapa pode ser considerada fechada quando houver:

- validação reproduzível da atenção contra as referências escolhidas;
- decisão documentada sobre o escopo e sobre TensorFlow/Keras;
- benchmarks comparáveis, com configuração e ambiente registrados;
- CSVs, tabelas e gráficos rastreáveis até a execução;
- conclusões compatíveis com o nível real de evidência, sem extrapolação para
  hardware ou Transformer completa;
- seção correspondente já integrada ao relatório do projeto.

## Trechos da reunião que sustentam os encaminhamentos

- **00:00–04:12:** necessidade de validar a implementação contra bibliotecas
  consolidadas antes de avançar;
- **05:35–08:01:** sugestão de tabelas e gráficos comparando cálculo manual,
  PyTorch e TensorFlow, além de iniciar o relatório;
- **08:01–14:09:** sequência validação → otimização → comparação e delimitação do
  escopo na autoatenção;
- **15:21–16:15:** modelo pré-treinado como possibilidade condicionada ao tempo e
  incerteza sobre o calendário de novembro;
- **23:13 em diante:** recomendação de redigir em prosa desde já e fechar cada
  avaliação com resultados e tabelas;
- **29:30–30:07:** compromisso de avançar nos benchmarks após a reunião.
