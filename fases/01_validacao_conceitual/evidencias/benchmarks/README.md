# Evidências de benchmarks

Esta pasta contém coletas versionadas. O rótulo da coleta determina o limite da
interpretação; smoke e diagnósticos em CPU não são resultados de hardware.

| Coleta | Propósito | Configuração | Status | Resultado principal |
| --- | --- | --- | --- | --- |
| [smoke_cpu_2026-08-25](smoke_cpu_2026-08-25/) | Validar pipeline ponta a ponta | B=1, L=4, D=8; 2 warm-ups, 5 medições, 2 execuções | Completo | 12 registros, checksums válidos e hashes determinísticos aprovados. |
| [diagnostico_cpu_l64_d128_2026-08-25](diagnostico_cpu_l64_d128_2026-08-25/) | Exercitar um ponto da grade formal em CPU | B=1, L=64, D=128; 20 warm-ups, 50 medições, 2 execuções | Completo, versão metodológica v1 | H1 compatível, mas pesos N(0,1) revelaram escala de saída inadequada para comparar MSE entre dimensões. |
| [diagnostico_cpu_l64_d128_v2_2026-08-25](diagnostico_cpu_l64_d128_v2_2026-08-25/) | Repetir o ponto após controlar a inicialização | B=1, L=64, D=128; Xavier, bias zero; protocolo completo | Completo, contrato vigente v2 | H1 compatível; hashes aprovados e escala absoluta controlada. |
| [benchmark_gpu_4070ti_super_2026-08-29](benchmark_gpu_4070ti_super_2026-08-29/) | Preservar a evidência GPU v1 | 2 operações, 3 comprimentos, 3 dimensões e 3 cenários | Completo, evidência histórica | 108 registros em duas execuções; pruning e INT8 fake sem representação física especializada. |
| [robustez_sintetica_v3_2026-09-13](robustez_sintetica_v3_2026-09-13/) | Testar robustez numérica entre entradas e arquiteturas | 13 perfis/distribuições, 5 seeds, 6 cenários e 3 repetições | Completo | 2.340 registros, 1.950 pares e 117 mil timings; INT8 fake teve menor MSE que pruning em todos os pares. |
| [hardware_probe_2026-09-14](hardware_probe_2026-09-14/) | Confirmar ambiente e kernels físicos | CUDA 12.8, SM 8.9, TorchAO e cuSPARSELt | Completo | Forward e traces aprovados para INT8 weight-only, INT8 dinâmico e sparse 2:4. |
| [smoke_hardware_v3_2026-09-14](smoke_hardware_v3_2026-09-14/) | Validar a bateria física ponta a ponta | Um perfil, 2 operações e 5 cenários | Completo | 10 registros, 30 timings e análise íntegra. |
| [hardware_nativo_v3_2026-09-14](hardware_nativo_v3_2026-09-14/) | Medir representações físicas nas operações isoladas | 11 perfis, 2 operações, 5 seeds, 5 cenários e 3 repetições | Completo | 1.650 registros e 82.500 timings; nenhum dos seis grupos otimizados obteve speedup sustentado. |
| [modelos_opt_smoke_2026-09-15](modelos_opt_smoke_2026-09-15/) | Validar runner OPT final | OPT-125M reduzido, qualidade, prefill, TTFT, decode e profiler | Completo | 5 variantes, 15 registros de desempenho e manifestos íntegros. |
| [modelos_opt_v1_2026-09-15](modelos_opt_v1_2026-09-15/) | Avaliar modelos pré-treinados | OPT-125M, OPT-350M e OPT-1.3B, 15 variantes por modelo | Completo com limitação de decode | 45 registros de qualidade, 1.296 registros de desempenho e 10.080 timings; prefill/TTFT completos, decode preservado como falha técnica. |
| [modelos_opt_2_7b_2026-09-15](modelos_opt_2_7b_2026-09-15/) | Extensão de escala | OPT-2.7B, 15 variantes | Completo | 15 registros de qualidade, 216 registros de desempenho e 1.440 timings; speedups mistos e qualidade preservada melhor por INT8. |
| [modelos_opt_6_7b_2026-09-16](modelos_opt_6_7b_2026-09-16/) | Extensão de escala guardada | OPT-6.7B, 15 variantes, guarda de VRAM em 15.800 MiB | Completo | 15 registros de qualidade, 216 registros de desempenho e 1.440 timings; speedup físico em alguns caminhos, mas sem qualidade aceitável nos blocos. |

Cada coleta deve preservar configuração, ambiente, manifesto, CSVs por execução,
timings e análise. Não sobrescreva uma pasta existente. Coletas v1/v2 usam
`analise/`; coletas v3 usam `analise_v3/`; modelos OPT usam `analise_modelos/`.
Markdown é usado somente neste README de navegação.

## Speedup e VRAM nos modelos OPT

As tabelas abaixo usam as medianas pareadas e os intervalos bootstrap de 95%
dos arquivos `analise_modelos/resumo_desempenho.csv`. A coluna `Padrão` é o
baseline associado de cada comparação. Valores acima de `1,0` em speedup são
mais rápidos que o baseline; valores abaixo de `1,0` em VRAM indicam menor uso
relativo de memória. A leitura científica precisa considerar também qualidade:
no OPT-6.7B, por exemplo, `blocks_int8_dynamic` acelerou, mas aumentou a
perplexidade em `165,05%`.

### Speedup mediano em prefill

| Modelo | Padrão | Attn INT8 dyn | Attn INT8 WO | Attn 2:4 | Blocos INT8 dyn | Blocos INT8 WO | Blocos 2:4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OPT-125M | 1.000 | 0.806 [0.677; 0.932] | 0.055 [0.051; 0.065] | 0.658 [0.621; 0.680] | 0.770 [0.610; 0.942] | 0.019 [0.018; 0.024] | 0.604 [0.553; 0.690] |
| OPT-350M | 1.000 | 0.798 [0.776; 0.903] | 0.035 [0.033; 0.044] | 0.665 [0.587; 0.752] | 0.816 [0.713; 1.008] | 0.012 [0.011; 0.016] | 0.711 [0.560; 0.758] |
| OPT-1.3B | 1.000 | 0.982 [0.878; 1.049] | 0.028 [0.027; 0.029] | 0.834 [0.780; 0.875] | 1.271 [0.958; 1.400] | 0.010 [0.009; 0.010] | 0.893 [0.807; 0.924] |
| OPT-2.7B | 1.000 | 1.024 [0.844; 1.134] | 0.030 [0.028; 0.039] | 0.873 [0.798; 0.907] | 1.305 [0.907; 1.588] | 0.010 [0.009; 0.014] | 1.061 [0.975; 1.078] |
| OPT-6.7B | 1.000 | 1.169 [1.009; 1.249] | 0.026 [0.025; 0.034] | 1.030 [0.974; 1.111] | 1.756 [1.171; 2.137] | 0.009 [0.009; 0.012] | 1.073 [1.017; 1.228] |

### Speedup mediano em TTFT

| Modelo | Padrão | Attn INT8 dyn | Attn INT8 WO | Attn 2:4 | Blocos INT8 dyn | Blocos INT8 WO | Blocos 2:4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OPT-125M | 1.000 | 0.799 [0.705; 0.902] | 0.053 [0.047; 0.074] | 0.688 [0.641; 0.743] | 0.757 [0.629; 0.929] | 0.019 [0.016; 0.026] | 0.720 [0.548; 0.751] |
| OPT-350M | 1.000 | 0.794 [0.692; 0.882] | 0.034 [0.031; 0.047] | 0.674 [0.627; 0.751] | 0.821 [0.663; 1.009] | 0.012 [0.011; 0.017] | 0.742 [0.627; 0.827] |
| OPT-1.3B | 1.000 | 0.983 [0.877; 1.073] | 0.027 [0.027; 0.034] | 0.818 [0.779; 0.887] | 1.256 [0.996; 1.512] | 0.009 [0.009; 0.012] | 0.915 [0.873; 0.932] |
| OPT-2.7B | 1.000 | 1.030 [0.848; 1.131] | 0.030 [0.028; 0.039] | 0.870 [0.797; 0.908] | 1.310 [0.915; 1.589] | 0.010 [0.009; 0.014] | 1.059 [0.971; 1.069] |
| OPT-6.7B | 1.000 | 1.166 [1.003; 1.247] | 0.026 [0.025; 0.034] | 1.035 [0.975; 1.140] | 1.744 [1.186; 2.140] | 0.009 [0.009; 0.012] | 1.072 [1.016; 1.246] |

### VRAM relativa mediana em prefill

| Modelo | Padrão | Attn INT8 dyn | Attn INT8 WO | Attn 2:4 | Blocos INT8 dyn | Blocos INT8 WO | Blocos 2:4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OPT-125M | 1.000 | 0.869 [0.869; 0.886] | 0.873 [0.873; 0.897] | 0.882 [0.882; 0.907] | 0.647 [0.646; 0.647] | 0.647 [0.647; 0.684] | 0.697 [0.697; 0.746] |
| OPT-350M | 1.000 | 0.849 [0.849; 0.855] | 0.849 [0.849; 0.855] | 0.999 [0.999; 0.999] | 0.546 [0.546; 0.546] | 0.546 [0.546; 0.577] | 0.679 [0.679; 0.698] |
| OPT-1.3B | 1.000 | 0.847 [0.847; 0.851] | 0.847 [0.847; 0.848] | 0.866 [0.866; 0.870] | 0.542 [0.542; 0.543] | 0.542 [0.542; 0.551] | 0.602 [0.602; 0.612] |
| OPT-2.7B | 1.000 | 0.842 [0.842; 0.849] | 0.842 [0.842; 0.849] | 0.862 [0.862; 0.868] | 0.526 [0.526; 0.546] | 0.526 [0.526; 0.545] | 0.585 [0.585; 0.602] |
| OPT-6.7B | 1.000 | 0.839 [0.839; 0.843] | 0.839 [0.839; 0.843] | 0.859 [0.859; 0.863] | 0.516 [0.516; 0.529] | 0.516 [0.516; 0.528] | 0.577 [0.577; 0.587] |
