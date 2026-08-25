# Relatório preliminar — diagnostico_cpu_l64_d128_v2

> Esta coleta é um diagnóstico em CPU de um ponto da grade formal. Os tempos não substituem a coleta na GPU-alvo e não sustentam conclusão de hardware.

## Pergunta, hipótese e desenho

- Pergunta principal: quantização INT8 simulada preserva melhor a saída que pruning por magnitude de 50% nas mesmas operações?
- Hipótese: a quantização terá MSE/MAE menores e R²/cosseno maiores que o pruning.
- Amostra: tensores sintéticos de distribuição normal, gerados deterministicamente; não há dataset ou pré-processamento externo.
- Seed: `42`; lote(s): `[1]`; sequências: `[64]`; dimensões: `[128]`.
- Dados/pesos: entrada `standard_normal`, pesos `xavier_normal` e bias `zeros`.
- Medição: `20` warm-ups, `50` medições e `2` execuções independentes.
- Dispositivo observado: `cpu`
- Inclusão: todos os casos válidos da grade e os três cenários registrados; exclusão: qualquer artefato com checksum, shape, dtype ou valor numérico inválido.

## Reprodutibilidade

- Hashes de entrada iguais entre execuções: `True`.
- Hashes de saída iguais entre execuções: `True`.
- As latências não precisam ser idênticas: elas medem ruído e estado do sistema; entradas e saídas determinísticas precisam coincidir.

## Resultados por operação e cenário

| Operação | Cenário | Latência média (ms) | DP entre execuções | MSE | R² | Cosseno |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| dense_projection | baseline | 0.118959 | 0.0408976 | 0 | 1 | 1 |
| dense_projection | pruning_magnitude | 0.052224 | 0.00514774 | 0.0727207 | 0.926527 | 0.962572 |
| dense_projection | quantization_int8 | 0.063593 | 0.00422426 | 7.7789e-05 | 0.999921 | 0.999961 |
| self_attention | baseline | 0.706044 | 0.502764 | 0 | 1 | 1 |
| self_attention | pruning_magnitude | 0.363446 | 0.0394877 | 0.0110127 | 0.762797 | 0.873495 |
| self_attention | quantization_int8 | 0.412401 | 0.171529 | 1.63925e-05 | 0.999647 | 0.999824 |

## Interpretação preliminar

- `dense_projection`: MSE da quantização `7.7789e-05` e do pruning `0.0727207`; evidência compatível com H1 nesta amostra.
- `self_attention`: MSE da quantização `1.63925e-05` e do pruning `0.0110127`; evidência compatível com H1 nesta amostra.
- Pruning usa matriz densa e não usa kernel esparso; quantização é dequantizada para `float32` e não usa kernel INT8.
- Razões de latência e memória estão nos CSVs de análise, mas só podem sustentar alegação de hardware se o experimento for principal e o caminho executado tiver validade `hardware`.
- Resultados de smoke ou diagnóstico CPU não são estimativas finais de desempenho no hardware-alvo.

## Cadeia de evidência

`pergunta → H1 → configuração versionada → CSVs por execução → checksums/hashes → resumo com média e DP → interpretação limitada`

## Reprodução

```powershell
$env:PYTHONPATH = "fases\01_validacao_conceitual"
.\.venv\Scripts\python.exe -m validacao.benchmark_operacoes --config <config.json> --output-dir <diretorio-novo>
.\.venv\Scripts\python.exe -m validacao.analise_benchmark_operacoes --evidence-dir <diretorio-novo>
```
