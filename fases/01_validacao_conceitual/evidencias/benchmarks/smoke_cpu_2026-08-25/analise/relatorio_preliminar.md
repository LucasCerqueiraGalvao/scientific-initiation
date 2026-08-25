# Relatório preliminar — smoke_cpu_v1

> Esta coleta e um smoke test em CPU. Os tempos servem apenas para validar a pipeline e não sustentam conclusão de hardware.

## Pergunta, hipótese e desenho

- Pergunta principal: quantização INT8 simulada preserva melhor a saída que pruning por magnitude de 50% nas mesmas operações?
- Hipótese: a quantização terá MSE/MAE menores e R²/cosseno maiores que o pruning.
- Amostra: tensores sintéticos de distribuição normal, gerados deterministicamente; não há dataset ou pré-processamento externo.
- Seed: `42`; lote(s): `[1]`; sequências: `[4]`; dimensões: `[8]`.
- Medição: `2` warm-ups, `5` medições e `2` execuções independentes.
- Dispositivo observado: `cpu`
- Inclusão: todos os casos válidos da grade e os três cenários registrados; exclusão: qualquer artefato com checksum, shape, dtype ou valor numérico inválido.

## Reprodutibilidade

- Hashes de entrada iguais entre execuções: `True`.
- Hashes de saída iguais entre execuções: `True`.
- As latências não precisam ser idênticas: elas medem ruído e estado do sistema; entradas e saídas determinísticas precisam coincidir.

## Resultados por operação e cenário

| Operação | Cenário | Latência média (ms) | DP entre execuções | MSE | R² | Cosseno |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dense_projection | baseline | 0.01942 | 0.0154715 | 0 | 1 | 1 |
| dense_projection | pruning_magnitude | 0.01916 | 0.0155563 | 0.352201 | 0.89254 | 0.947344 |
| dense_projection | quantization_int8 | 0.03504 | 0.0070145 | 7.28672e-05 | 0.999978 | 0.999989 |
| self_attention | baseline | 0.25577 | 0.0244518 | 0 | 1 | 1 |
| self_attention | pruning_magnitude | 0.31862 | 0.193012 | 25.8574 | 0.1343 | 0.570018 |
| self_attention | quantization_int8 | 0.51548 | 0.0405879 | 0.00503124 | 0.999832 | 0.999919 |

## Interpretação preliminar

- `dense_projection`: MSE da quantização `7.28672e-05` e do pruning `0.352201`; evidência compatível com H1 nesta amostra.
- `self_attention`: MSE da quantização `0.00503124` e do pruning `25.8574`; evidência compatível com H1 nesta amostra.
- Pruning usa matriz densa e não usa kernel esparso; quantização é dequantizada para `float32` e não usa kernel INT8.
- Razões de latência e memória estão nos CSVs de análise, mas só podem sustentar alegação de hardware se o experimento for principal e o caminho executado tiver validade `hardware`.
- Resultados deste smoke são diagnósticos da infraestrutura, não estimativas finais de desempenho.

## Cadeia de evidência

`pergunta → H1 → configuração versionada → CSVs por execução → checksums/hashes → resumo com média e DP → interpretação limitada`

## Reprodução

```powershell
$env:PYTHONPATH = "fases\01_validacao_conceitual"
.\.venv\Scripts\python.exe -m validacao.benchmark_operacoes --config <config.json> --output-dir <diretorio-novo>
.\.venv\Scripts\python.exe -m validacao.analise_benchmark_operacoes --evidence-dir <diretorio-novo>
```
