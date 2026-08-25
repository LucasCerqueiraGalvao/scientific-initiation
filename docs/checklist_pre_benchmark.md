# Checklist pré-benchmark

Atualizado em 25/08/2026. “Benchmark principal” significa a coleta destinada a
sustentar conclusões de desempenho no hardware-alvo. Smoke tests em CPU são
permitidos para validar a infraestrutura, desde que sejam rotulados como tal.

## Obrigatório antes do benchmark principal

- [x] **[OBRIGATÓRIO ANTES DO BENCHMARK]** Arquivar a reunião e separar pedidos
  explícitos, decisões, dúvidas e interpretações.
- [x] **[OBRIGATÓRIO ANTES DO BENCHMARK]** Validar a atenção manual contra
  referência matemática, PyTorch e TensorFlow/Keras com as mesmas entradas.
- [x] **[OBRIGATÓRIO ANTES DO BENCHMARK]** Registrar tabela, tolerâncias,
  versões e limites da validação cruzada.
- [x] **[OBRIGATÓRIO ANTES DO BENCHMARK]** Fixar o recorte: projeção densa como
  operação prevista no plano e self-attention como foco principal; bloco
  Transformer completo fora da coleta principal.
- [x] **[OBRIGATÓRIO ANTES DO BENCHMARK]** Fixar cenários principais: baseline,
  pruning por magnitude de 50% e quantização INT8 simulada.
- [x] **[OBRIGATÓRIO ANTES DO BENCHMARK]** Implementar runner das operações
  isoladas sem alterar o schema CSV existente.
- [x] **[OBRIGATÓRIO ANTES DO BENCHMARK]** Versionar configurações distintas de
  smoke e coleta principal.
- [x] **[OBRIGATÓRIO ANTES DO BENCHMARK]** Codificar e testar o contrato formal:
  seed 42,
  lote 1, grade `L={64,128,256}`, `D={128,256,512}`, 20 warm-ups, 50 medições e
  duas execuções independentes. A execução da grade continua condicionada à GPU.
- [x] **[OBRIGATÓRIO ANTES DO BENCHMARK]** Registrar Python, bibliotecas,
  sistema, dispositivo, parâmetros, ordem dos cenários e método de percentil.
- [x] **[OBRIGATÓRIO ANTES DO BENCHMARK]** Persistir CSV, metadados, manifesto,
  log e hashes de entradas/saídas em pasta versionável.
- [x] **[OBRIGATÓRIO ANTES DO BENCHMARK]** Rejeitar shapes, dtypes, métricas ou
  resultados inválidos, `NaN` e infinitos.
- [ ] **[OBRIGATÓRIO ANTES DO BENCHMARK]** Corrigir e documentar a inicialização
  dos pesos-base: o diagnóstico v1 mostrou que `N(0,1)` faz a escala da saída
  crescer com a dimensão e prejudica a comparação de MSE/MAE.
- [x] **[OBRIGATÓRIO ANTES DO BENCHMARK]** Fazer smoke test ponta a ponta,
  incluindo análise, gráficos e reexecução determinística.
- [ ] **[OBRIGATÓRIO ANTES DO BENCHMARK]** Confirmar que o dispositivo da coleta
  principal é CUDA e registrar a RTX 4070 Ti Super. **Bloqueio externo atual:**
  esta máquina só expõe Intel Iris Xe e zero dispositivos CUDA.

## Recomendado antes do benchmark principal

- [x] **[RECOMENDADO ANTES DO BENCHMARK]** Variar deterministicamente a ordem
  dos cenários entre execuções para reduzir viés de ordem.
- [x] **[RECOMENDADO ANTES DO BENCHMARK]** Registrar snapshot exato do ambiente
  instalado, sem transformar o arquivo de requisitos mínimos em um lock
  específico desta máquina.
- [x] **[RECOMENDADO ANTES DO BENCHMARK]** Fazer sanity checks de crescimento de
  FLOPs e shapes antes da grade completa.
- [x] **[RECOMENDADO ANTES DO BENCHMARK]** Consolidar as duas execuções com
  média, desvio-padrão e resultados individuais preservados.
- [ ] **[RECOMENDADO ANTES DO BENCHMARK]** Confirmar datas institucionais e
  corrigir, no canal apropriado, a inconsistência 2026/2027 do cronograma.

## Pode ser feito depois dos primeiros benchmarks

- [ ] **[PODE SER FEITO DEPOIS]** Inferência demonstrativa com modelo
  pré-treinado do Hugging Face.
- [ ] **[PODE SER FEITO DEPOIS]** Benchmark de bloco Transformer completo.
- [ ] **[PODE SER FEITO DEPOIS]** Cenários `torchao_int8`, combinação de poda e
  quantização e kernels esparsos.
- [ ] **[PODE SER FEITO DEPOIS]** Ampliação para mais seeds, lotes, números de
  cabeças ou tamanhos de sequência.
- [ ] **[PODE SER FEITO DEPOIS]** Comparação direta de desempenho entre
  TensorFlow e PyTorch; TensorFlow é apenas referência de correção nesta etapa.

## Regra de liberação

O benchmark principal só está liberado quando todos os itens obrigatórios
controláveis pelo repositório estiverem concluídos e o hardware CUDA-alvo estiver
disponível. Se o hardware não estiver disponível, a etapa deve encerrar com
smoke/diagnóstico de CPU claramente identificado, sem alegação de desempenho de
hardware.
