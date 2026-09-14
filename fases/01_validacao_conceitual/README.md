# Fase 01 - Validação conceitual e experimental

Esta fase verifica se os conceitos, fórmulas, implementações e instrumentos de
medição estão corretos antes de produzir conclusões de desempenho.

## Resultado da fase

Estão validados:

- projeção linear densa e projeções Q, K e V;
- scaled dot-product attention, self-attention, multi-head e máscara aditiva;
- informação posicional, residual, normalização e FFN no bloco simplificado;
- inferência determinística e KV cache;
- representações conceituais de pruning, esparsidade e quantização;
- MSE, MAE, R², similaridade de cosseno e FLOPs analíticos;
- schema, configuração, integridade e reprodutibilidade da pipeline;
- equivalência do núcleo da atenção em NumPy, PyTorch e TensorFlow/Keras.

O objeto principal agora é a self-attention multi-head completa, com projeção
densa mantida como controle das operações Q/K/V/O. O benchmark GPU v1 está
concluído, assim como as baterias sintética v3 e física. O estudo OPT ainda não
autoriza conclusões finais antes da coleta completa em uma janela sem concorrência
da GPU.

## Estrutura

```text
fases/01_validacao_conceitual/
  docs/             textos acadêmicos em LaTeX
  experimentos/     configurações JSON
  validacao/        operações, runners e análises
  tests/            testes matemáticos e de contrato
  evidencias/       resultados auditáveis
```

Os geradores de relatório produzem fragmentos `.tex`; CSV, JSON, logs e figuras
continuam em seus formatos próprios.

## Ambiente

Na raiz do repositório:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

.\.venv\Scripts\python.exe -m pip install `
  torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128

.\.venv\Scripts\python.exe -m pip install `
  -r fases\01_validacao_conceitual\requirements-comparacao.txt
```

TensorFlow 2.21/Keras é executado em CPU para o portão de correção. PyTorch/CUDA
é o caminho do benchmark principal.

## Testes

```powershell
$env:PYTHONPATH = "fases\01_validacao_conceitual"
.\.venv\Scripts\python.exe -m pytest `
  fases\01_validacao_conceitual\tests -q -W error
```

Resultado atual no contêiner: `71 passed, 1 skipped`. O skip é a integração real com
Hugging Face, ativada explicitamente por `RUN_HF_INTEGRATION=1`; o smoke com um
OPT minúsculo criado por configuração roda sem download.

## Comparação numérica entre frameworks

```powershell
$env:PYTHONPATH = "fases\01_validacao_conceitual"
.\.venv\Scripts\python.exe -m validacao.comparacao_frameworks `
  --output-dir resultados\comparacao_frameworks `
  --seed 2026 --atol 1e-6 --rtol 1e-5
```

A evidência v1 aprovou 9/9 comparações. A validação v2 adiciona nove comparações
da multi-head completa, com `H={1,4,8}`, para totalizar 18. Consulte o
[relatório LaTeX](evidencias/comparacao_frameworks_v2_2026-09-13/comparacao_atencao.tex),
o [CSV](evidencias/comparacao_frameworks_v2_2026-09-13/comparacao_atencao.csv) e os
[metadados](evidencias/comparacao_frameworks_v2_2026-09-13/comparacao_atencao.metadata.json).
Essa etapa valida correção, não velocidade.

## Configurações de benchmark

| Arquivo | Uso | Limite da conclusão |
| --- | --- | --- |
| `experimentos/smoke_cpu.json` | Pipeline curta | Nenhuma conclusão de hardware. |
| `experimentos/diagnostico_cpu_l64_d128.json` | Um ponto formal em CPU | Diagnóstico metodológico. |
| `experimentos/benchmark_principal_gpu.json` | Grade CUDA na RTX 4070 Ti Super | Base para a análise principal, após auditoria do caminho. |
| `experimentos/smoke_v3_cpu.json` | Contrato v3 curto | Validação local sem conclusão de hardware. |
| `experimentos/robustez_sintetica_v3.json` | 13 perfis, 5 seeds e 6 cenários | Robustez numérica de dense e multi-head attention. |
| `experimentos/smoke_hardware_v3.json` | Portão físico curto | Confirma compilação, INT8 e 2:4 antes da bateria. |
| `experimentos/hardware_nativo_v3.json` | 1.650 casos físicos | Latência, memória e armazenamento com kernels compatíveis. |
| `experimentos/modelos_opt_smoke.json` | OPT-125M reduzido | Portão funcional de qualidade, prefill, TTFT, decode e profiler. |
| `experimentos/modelos_opt_v1.json` | OPT 125M, 350M e 1.3B | Qualidade e desempenho em pesos pré-treinados. |

O schema v3 preserva leitura dos schemas v1/v2 e acrescenta perfis, múltiplas
seeds, cenários parametrizados, grupos de baseline, timings brutos, memória,
armazenamento, hashes e confirmação de kernels.

## Resultado sintético v3

A execução completa está em
[`evidencias/benchmarks/robustez_sintetica_v3_2026-09-13/`](evidencias/benchmarks/robustez_sintetica_v3_2026-09-13/).
São 2.340 registros completos, 1.950 comparações pareadas e 117 mil amostras de
latência. O INT8 fake teve MSE menor que o pruning em 100% dos pares nos quatro
níveis, enquanto o MSE do pruning cresceu monotonicamente com a sparsity.

Os speedups foram classificados como mistos: os intervalos de 95% cruzaram
`1,0x`, o armazenamento continuou denso e nenhum kernel físico foi confirmado.
Isso é resultado esperado de uma bateria de fidelidade numérica, não evidência
de aceleração INT8 ou sparse.

## Resultado físico v3

O probe, o smoke e a coleta completa estão em
[`evidencias/benchmarks/hardware_probe_2026-09-14/`](evidencias/benchmarks/hardware_probe_2026-09-14/),
[`evidencias/benchmarks/smoke_hardware_v3_2026-09-14/`](evidencias/benchmarks/smoke_hardware_v3_2026-09-14/)
e [`evidencias/benchmarks/hardware_nativo_v3_2026-09-14/`](evidencias/benchmarks/hardware_nativo_v3_2026-09-14/).
A bateria contém 1.650 registros completos, 990 pares e 82.500 medições. O
profiler confirmou computação INT8 dinâmica e cuSPARSELt 2:4 nos cenários
correspondentes. Mesmo assim, os seis grupos de operação e técnica ficaram mais
lentos que o baseline compilado, com intervalos de 95% inteiramente abaixo de
`1,0x`; portanto, o ganho de latência foi contradito neste recorte.

## Executar as novas baterias

O Docker Desktop está funcional com backend WSL2, e seu disco de dados está em
`D:\DockerDesktopData`. O cache dos três modelos OPT e do WikiText está em
`D:\Caches\scientific-initiation\huggingface`. Para inspecionar o estado sem
iniciar uma medição:

```powershell
$cacheRoot = "D:\Caches\scientific-initiation\huggingface"
.\scripts\run_benchmarks_docker.ps1 -Action Status -CacheRoot $cacheRoot
```

A sequência abaixo executa somente as etapas pendentes e usa nomes estáveis para
retomada:

```powershell
.\scripts\run_benchmarks_docker.ps1 `
  -Action Remaining `
  -CacheRoot $cacheRoot `
  -RunId "final-20260913" `
  -Resume
```

Os runners escrevem checkpoints e manifestos. As análises geram intervalos
bootstrap, curvas por sparsity, heatmaps, boxplots por seed, speedup, memória,
armazenamento, Pareto e tabelas de casos suportados ou incompatíveis. Arquivos
com checksum divergente são recusados.

`Remaining` não repete a robustez sintética já concluída. Antes de cada etapa que
usa a GPU, o orquestrador exige jogo fechado, utilização média de cinco amostras
abaixo de 10%, até 2.048 MiB de VRAM ocupada e temperatura abaixo de 65 °C; o pico
também fica registrado no diagnóstico. Em WDDM, uma leitura residual pode ser
aceita somente com processos abaixo de 10%, potência até 35 W e clock até 300 MHz,
sem relaxar VRAM ou temperatura. Etapas completas e íntegras são ignoradas em
`-Resume`. Restam cerca de 4–8 horas: 15–40 minutos de suíte/smoke OPT, 3–6 horas
de OPT e 30–90 minutos de consolidação.

## Documentos técnicos

- [Base teórica](docs/base_teorica_validacao.tex)
- [Caderno narrativo detalhado](docs/explicacao_metodologia_validacao.tex)
- [Resumo operacional dos conceitos](docs/conceitos.tex)
- [Validação arquitetural](docs/validacao_transformer.tex)
- [Perguntas de pesquisa](docs/perguntas_pesquisa.tex)
- [Protocolo experimental](docs/protocolo_experimental.tex)
- [Matriz de ferramentas](docs/matriz_validacao_ferramentas.tex)
- [Confronto com a literatura](docs/confronto_resultados_literatura.tex)
- [Decisão de inicialização](docs/decisao_inicializacao_benchmark.tex)

## Próximo portão

1. executar o smoke OPT final quando o portão de GPU permitir;
2. executar OPT-125M, OPT-350M e OPT-1.3B;
3. revisar hashes, pareamentos, intervalos e casos incompatíveis;
4. atualizar RQ1–RQ8, compilar o PDF e versionar a evidência OPT.
