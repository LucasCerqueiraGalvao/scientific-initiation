# Metodologia final consolidada

Este documento descreve o que foi efetivamente executado na pesquisa. Ele não
amplia retrospectivamente o protocolo; apenas organiza a metodologia usada nas
evidências canônicas.

## Fontes canônicas

A análise final usa os artefatos em `output/canonical/`, gerados por:

```powershell
$env:PYTHONPATH = "fases/01_validacao_conceitual"
.\.venv\Scripts\python.exe -m validacao.consolidacao_resultados `
  --evidence-root fases\01_validacao_conceitual\evidencias\benchmarks `
  --output output\canonical
```

Fontes aceitas:

- `robustez_sintetica_v3_2026-09-13`;
- `hardware_nativo_v3_2026-09-14`;
- `hardware_stress_complementar_v3_2026-09-17`;
- `modelos_opt_v1_2026-09-15`;
- `modelos_opt_2_7b_2026-09-15`;
- `modelos_opt_6_7b_2026-09-16`;
- `modelos_opt_complementar_lacunas_2026-09-17`;
- `modelos_opt_6_7b_complementar_lacunas_2026-09-17`;
- `hardware_crossover_v3_2026-09-19`;
- `modelos_opt_sensibilidade_350m_2026-09-19`;
- `modelos_opt_sensibilidade_1_3b_2026-09-19`;
- `modelos_opt_hibridos_1_3b_quality_2026-09-19`;
- `modelos_opt_hibridos_1_3b_finalistas_2026-09-19`;
- `modelos_opt_hibridos_6_7b_quality_2026-09-19`.

Fontes excluídas das conclusões:

- smokes;
- diagnósticos CPU antigos;
- benchmark GPU v1 quando há evidência v3 para a mesma pergunta;
- duplicatas de qualidade substituídas por runs complementares mais recentes.

Cada linha consolidada recebe `canonical_source` e
`canonical_source_priority`. Em duplicatas da mesma chave experimental, a fonte
de maior prioridade vence. As duplicatas descartadas permanecem registradas em
`discarded_*_duplicates.csv`.

## Níveis de evidência

- `A_numeric_change`: altera pesos ou saídas numericamente, mas não garante
  armazenamento compacto nem kernel físico. Inclui INT8 fake e pruning denso
  percentual.
- `B_physical_compression`: reduz armazenamento físico, sem promessa automática
  de latência. Inclui INT8 weight-only.
- `C_physical_acceleration_candidate`: usa caminho que pode acelerar execução,
  mas só sustenta conclusão física quando `physical_kernel_status=confirmed`.
  Inclui INT8 dynamic e pruning 2:4.

Essa separação é central: alteração numérica, compressão física e aceleração
física são fenômenos diferentes.

## Hardware e ambiente

Os benchmarks físicos foram executados em uma NVIDIA RTX 4070 Ti SUPER, com
16 GB de VRAM e compute capability 8.9. A execução principal usou Linux em
contêiner via Docker Desktop/WSL2, imagem PyTorch 2.11.0 com CUDA 12.8 e
cuDNN 9. As dependências científicas incluíram TorchAO, Triton, Transformers,
Datasets, Accelerate e cuSPARSELt.

Antes das baterias físicas, o wrapper verificava GPU disponível, baixa
ocupação, VRAM inicial, temperatura inicial, CUDA, Triton, forward INT8 finito,
conversão 2:4 e presença dos kernels esperados no profiler quando aplicável.
Os clocks e o power state da GPU não foram fixados; o controle de ruído veio
de warm-up, sincronização CUDA, repetição, GPU ociosa e intervalos bootstrap.

## Validação conceitual

A implementação manual de attention foi comparada com NumPy manual, PyTorch,
PyTorch SDPA e Keras/TensorFlow. Foram aprovadas 18 comparações, incluindo
scaled dot-product attention sem máscara, attention com máscara aditiva,
projeções Q/K/V/O e multi-head attention com `H={1,4,8}`.

As comparações usaram entradas e pesos compartilhados, tolerância
`atol=1e-6` e `rtol=1e-5`. O maior erro absoluto observado foi
`4.76837158e-7`, compatível com arredondamento de ponto flutuante.

## Benchmarks sintéticos

As operações sintéticas principais foram `dense_projection` e
`multi_head_self_attention`. O benchmark sintético v3 usou cinco seeds, 13
combinações de perfil/distribuição, seis cenários e três repetições. Cada
registro teve 20 warm-ups e 50 medições, totalizando 2.340 registros agregados
e 117.000 amostras individuais de latência.

Os cenários foram baseline FP32, pruning denso por magnitude em 10%, 25%, 50%
e 75%, e INT8 fake simétrico por linha com dequantização para FP32. Essa
bateria mede fidelidade numérica e robustez entre entradas; ela não prova
aceleração física.

## Benchmarks físicos de operações

As operações físicas usaram os mesmos objetos principais, mas com caminhos de
hardware: baseline BF16 compilado, INT8 weight-only, INT8 dynamic, baseline
FP16 compilado e pruning 2:4 FP16 com representação semiestruturada.

As medições usaram CUDA Events, sincronização CUDA antes/depois da região
medida, 20 warm-ups, 50 medições por registro, cinco seeds e três repetições.
Compilação, transformação e preparação ficaram fora da região cronometrada.

O microbenchmark de crossover (`hardware_crossover_v3_2026-09-19`) variou
sequência, batch e largura para investigar se havia ponto de virada favorável.
O stress complementar (`hardware_stress_complementar_v3_2026-09-17`) avaliou
distribuições uniforme e com outliers; ele não substitui o crossover por
tamanho de workload.

## Modelos OPT

Foram avaliados `facebook/opt-125m`, `facebook/opt-350m`,
`facebook/opt-1.3b`, `facebook/opt-2.7b` e `facebook/opt-6.7b`. Não houve
treinamento nem fine-tuning. Cada variante recarregou os pesos originais e
aplicou a técnica apenas para inferência.

Escopos usados:

- `attention_only`: `q_proj`, `k_proj`, `v_proj` e `out_proj` dos blocos
  Transformer;
- `transformer_blocks`: lineares internas dos blocos Transformer, incluindo
  atenção e MLP (`q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2`);
- seletores finos por camada/componente, usados nas fases de sensibilidade e
  híbridos.

Embeddings e `lm_head` foram excluídos dos escopos de otimização.

## Qualidade

A qualidade funcional usou WikiText-2 (`wikitext-2-raw-v1`) com 64 janelas não
sobrepostas de 512 tokens, distribuídas uniformemente no conjunto de validação.
O runner registra índices, tokens e hash do subconjunto.

Para cada variante foram coletados loss, perplexidade, divergência KL dos
logits contra baseline, MSE dos logits, top-1 agreement, geração greedy de 64
tokens em 20 prompts locais e token agreement da geração contra baseline.

O critério oficial de `quality_acceptable` usa apenas aumento percentual de
perplexidade: quantização aceitável quando `Delta PPL <= 2%`; pruning aceitável
quando `Delta PPL <= 5%`.

## Desempenho em OPT

O desempenho dos modelos mediu `model_prefill` e `model_ttft`. O nome histórico
`model_ttft` foi preservado para compatibilidade com CSVs, mas deve ser
descrito como `prefill_to_first_logit`, pois não inclui tokenização, fila de
serviço, streaming ou overhead de sistema de produção.

VRAM foi medida a partir dos contadores CUDA de memória alocada/reservada de
pico. Armazenamento foi calculado a partir dos parâmetros representados em cada
cenário. Speedup foi calculado contra o baseline pareado do mesmo modelo,
operação, perfil e repetição.

## Estatística

As comparações são pareadas por input, peso, seed, perfil, repetição e baseline
associado. A análise calcula média, mediana, desvio padrão e intervalo
bootstrap de 95% com 10.000 reamostragens e seed 2026.

Um speedup físico é considerado confirmado somente quando a mediana é
`>= 1.05x`, o intervalo de 95% fica inteiramente acima de `1.0x` e o kernel
físico compatível é confirmado quando a conclusão depende dele.

## Sensibilidade e híbridos

A fase de sensibilidade testou INT8 dynamic por componente e região em
OPT-350M e OPT-1.3B. As regiões principais foram início, meio e fim dos blocos;
os componentes principais foram atenção e MLP.

A fase híbrida testou configurações seletivas derivadas da sensibilidade. No
OPT-1.3B, algumas configurações preservaram qualidade quando a atenção sensível
foi mantida em BF16 e parte do MLP foi quantizada. No OPT-6.7B, as mesmas
políticas não preservaram qualidade. As candidatas de OPT-6.7B que falharam
qualidade não foram submetidas a benchmark físico.
