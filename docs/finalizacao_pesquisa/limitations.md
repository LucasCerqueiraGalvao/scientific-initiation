# Limitações atuais

## Metodológicas

- A métrica chamada historicamente de `model_ttft` não é TTFT completo de
  serviço; ela representa `prefill_to_first_logit`.
- `quality_acceptable` usa apenas Delta PPL. KL, MSE, top-1 agreement e token
  agreement são métricas complementares, não critérios oficiais de aprovação.
- O dataset de qualidade usa WikiText-2 com 64 janelas. Isso é adequado para
  triagem controlada, mas ainda pequeno para alegações gerais sobre linguagem.
- Os prompts locais são úteis para divergência de geração, mas não substituem
  avaliação humana ou benchmark amplo de tarefas.
- Smokes e diagnósticos são evidência de infraestrutura, não evidência para
  conclusão científica.
- Speedups devem ser interpretados por workload, modelo e operação. Um ganho em
  `prefill_to_first_logit` não prova ganho em decode autoregressivo.

## Técnicas

- INT8 fake mede alteração numérica e fidelidade, não aceleração física.
- Pruning denso percentual não reduz VRAM nem garante latência menor enquanto
  continuar executando por kernels densos.
- INT8 weight-only medido neste ambiente deve ser tratado como evidência do
  backend usado, não como refutação geral da técnica.
- O 2:4 por magnitude degradou qualidade, mas isso não invalida 2:4 com seleção
  de pesos mais informada, como Wanda ou SparseGPT.
- Conclusões causais sobre kernels só são fortes quando o profiler confirma o
  caminho físico.

## Ambiente

- Os manifestos antigos não registram com granularidade completa versões exatas
  de cuBLAS/cuBLASLt/cuDNN.
- Clocks, power state e interferência fina do driver WDDM não são totalmente
  controlados. O protocolo reduz ruído com guardas de GPU ociosa, warm-up,
  sincronização CUDA e repetições, mas não fixa todos os estados de energia.
- O decode com KV cache sob `torch.compile` e CUDA Graphs ficou preservado como
  falha técnica em parte das execuções, não como conclusão de desempenho.

## Escala

- OPT-6.7B deve ser usado para validar finalistas, não para busca combinatória.
- O ganho em modelos maiores apareceu em alguns caminhos físicos, mas nem sempre
  preservou qualidade. A hipótese de escala precisa ser formulada como
  condicional: escala pode favorecer speedup, desde que a técnica mantenha
  qualidade e use kernel adequado.
