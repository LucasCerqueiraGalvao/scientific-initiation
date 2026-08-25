# Evidências de benchmarks

Esta pasta contém coletas versionadas. O rótulo da coleta determina o limite da
interpretação; smoke e diagnósticos em CPU não são resultados de hardware.

| Coleta | Propósito | Configuração | Status | Resultado principal |
| --- | --- | --- | --- | --- |
| [smoke_cpu_2026-08-25](smoke_cpu_2026-08-25/) | Validar pipeline ponta a ponta | B=1, L=4, D=8; 2 warm-ups, 5 medições, 2 execuções | Completo | 12 registros, checksums válidos e hashes determinísticos aprovados. |

Cada coleta deve preservar configuração, ambiente, log, manifesto, CSVs por
execução, metadados e análise. Não sobrescreva uma pasta existente.
