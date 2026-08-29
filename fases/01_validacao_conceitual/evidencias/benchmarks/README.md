# Evidências de benchmarks

Esta pasta contém coletas versionadas. O rótulo da coleta determina o limite da
interpretação; smoke e diagnósticos em CPU não são resultados de hardware.

| Coleta | Propósito | Configuração | Status | Resultado principal |
| --- | --- | --- | --- | --- |
| [smoke_cpu_2026-08-25](smoke_cpu_2026-08-25/) | Validar pipeline ponta a ponta | B=1, L=4, D=8; 2 warm-ups, 5 medições, 2 execuções | Completo | 12 registros, checksums válidos e hashes determinísticos aprovados. |
| [diagnostico_cpu_l64_d128_2026-08-25](diagnostico_cpu_l64_d128_2026-08-25/) | Exercitar um ponto da grade formal em CPU | B=1, L=64, D=128; 20 warm-ups, 50 medições, 2 execuções | Completo, versão metodológica v1 | H1 compatível, mas pesos N(0,1) revelaram escala de saída inadequada para comparar MSE entre dimensões. |
| [diagnostico_cpu_l64_d128_v2_2026-08-25](diagnostico_cpu_l64_d128_v2_2026-08-25/) | Repetir o ponto após controlar a inicialização | B=1, L=64, D=128; Xavier, bias zero; protocolo completo | Completo, contrato vigente v2 | H1 compatível; hashes aprovados e escala absoluta controlada. |

Cada coleta deve preservar configuração, ambiente, log, manifesto, CSVs por
execução, metadados e análise. Não sobrescreva uma pasta existente. O relatório
gerado pela análise é `analise/relatorio_preliminar.tex`; Markdown é usado
somente neste README de navegação.
