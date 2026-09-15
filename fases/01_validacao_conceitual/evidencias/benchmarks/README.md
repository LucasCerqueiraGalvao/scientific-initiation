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

Cada coleta deve preservar configuração, ambiente, manifesto, CSVs por execução,
timings e análise. Não sobrescreva uma pasta existente. Coletas v1/v2 usam
`analise/`; coletas v3 usam `analise_v3/`; modelos OPT usam `analise_modelos/`.
Markdown é usado somente neste README de navegação.
