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
concluído, assim como as baterias sintética v3, física v3, OPT v1, OPT-2.7B,
OPT-6.7B e o complemento de 17/09/2026. A coleta OPT autoriza conclusões sobre
qualidade, prefill e TTFT em modelos pré-treinados; a etapa de decode ficou
preservada como falha técnica associada a `torch.compile`, CUDA Graphs e KV
cache.

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
| `experimentos/modelos_opt_v1.json` | OPT 125M, 350M e 1.3B | Qualidade, prefill e TTFT em pesos pré-treinados; decode registrado como limitação técnica. |
| `experimentos/modelos_opt_2_7b_incremental.json` | OPT-2.7B | Extensão de escala com qualidade, prefill e TTFT. |
| `experimentos/modelos_opt_6_7b_incremental_guarded.json` | OPT-6.7B | Extensão de escala com guarda de VRAM em 15.800 MiB e retomada por cenário. |
| `experimentos/hardware_stress_v3_complementar.json` | Perfis uniforme/outlier | Complemento físico dos perfis de stress para INT8/2:4. |
| `experimentos/modelos_opt_complementar_performance.json` | OPT 125M, 350M, 1.3B e 2.7B | Medição de desempenho/VRAM dos prunings percentuais e INT8 fake que antes tinham só qualidade. |
| `experimentos/modelos_opt_6_7b_complementar_guarded.json` | OPT-6.7B complementar | Mesma lacuna do complemento, mas com guarda de VRAM em 15.800 MiB e um cenário por container. |

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

## Resultado OPT v1

O smoke final e a coleta completa estão em
[`evidencias/benchmarks/modelos_opt_smoke_2026-09-15/`](evidencias/benchmarks/modelos_opt_smoke_2026-09-15/)
e [`evidencias/benchmarks/modelos_opt_v1_2026-09-15/`](evidencias/benchmarks/modelos_opt_v1_2026-09-15/).
Foram avaliados `facebook/opt-125m`, `facebook/opt-350m` e `facebook/opt-1.3b`
em revisões fixadas, sem treinamento ou fine-tuning. A execução completa gerou
45 registros de qualidade, 2.880 janelas de WikiText-2, 900 métricas de prompts,
1.296 registros de desempenho e 10.080 amostras de timing.

Na qualidade, 25 de 45 variantes ficaram dentro dos limites operacionais. As
quantizações INT8 fake e weight-only permaneceram estáveis nos três modelos. O
INT8 dinâmico ficou dentro do limite em OPT-125M e OPT-350M, mas excedeu o
limite em OPT-1.3B. O pruning por magnitude em atenção só permaneceu aceitável
em 10%; 25% já excedeu levemente o limite de 5%, e 50%/75% degradaram muito a
perplexidade. O pruning 2:4 degradou qualidade em todos os modelos, sobretudo
quando aplicado a todas as camadas lineares dos blocos.

No desempenho, prefill e TTFT foram medidos com registros completos. Nenhuma
comparação cumpriu simultaneamente speedup mediano de pelo menos `1,05x`,
intervalo de 95% inteiramente acima de `1,0x` e kernel confirmado. A única
tendência favorável apareceu em `blocks_int8_dynamic` no OPT-1.3B, mas o
intervalo ainda cruzou o nulo; portanto o ganho foi classificado como misto,
não sustentado. Todos os casos de `model_decode` foram preservados como
`failed` ou `unsupported`: o erro principal foi acesso a saída sobrescrita de
CUDA Graphs ao atualizar KV cache sob `torch.compile`.

## Extensão exploratória OPT-2.7B e OPT-6.7B

O `facebook/opt-2.7b` e o `facebook/opt-6.7b` são extensões adicionais para
observar se as tendências de qualidade, memória e latência mudam quando o número
de parâmetros cresce além da bateria OPT v1. Eles não substituem a bateria OPT
v1 já concluída e devem ser interpretados como análise incremental de escala.

O OPT-2.7B terminou 15 cenários, 216 registros de desempenho e 1.440 timings.
O OPT-6.7B também terminou 15 cenários, 216 registros de desempenho e 1.440
timings, usando guarda de VRAM em 15.800 MiB, retomada por cenário e execução
isolada por container. As evidências promovidas ficam em
[`evidencias/benchmarks/modelos_opt_2_7b_2026-09-15/`](evidencias/benchmarks/modelos_opt_2_7b_2026-09-15/)
e [`evidencias/benchmarks/modelos_opt_6_7b_2026-09-16/`](evidencias/benchmarks/modelos_opt_6_7b_2026-09-16/).

No OPT-6.7B, `blocks_int8_dynamic` teve speedup mediano de `1,756x` em prefill
e `1,744x` em TTFT, com intervalos de 95% acima de `1,0x`. Mesmo assim, a
perplexidade aumentou de `15,346` para `40,675`, ou `165,05%`; portanto, houve
speedup físico, mas não uma configuração final aceitável pelos limites de
qualidade.

## Complemento de lacunas

A execução complementar de 17/09/2026 foi feita para responder por que a planilha
tinha células vazias em pruning percentual/INT8 fake dos modelos OPT e nos perfis
físicos de stress. As evidências promovidas estão em
[`evidencias/benchmarks/hardware_stress_complementar_v3_2026-09-17/`](evidencias/benchmarks/hardware_stress_complementar_v3_2026-09-17/),
[`evidencias/benchmarks/modelos_opt_complementar_lacunas_2026-09-17/`](evidencias/benchmarks/modelos_opt_complementar_lacunas_2026-09-17/)
e [`evidencias/benchmarks/modelos_opt_6_7b_complementar_lacunas_2026-09-17/`](evidencias/benchmarks/modelos_opt_6_7b_complementar_lacunas_2026-09-17/).

O hardware stress acrescentou 300 registros e 15.000 timings. A bateria OPT de
lacunas acrescentou 40 registros de qualidade, 1.080 registros de desempenho e
7.200 timings. O resultado mais importante é negativo e útil: pruning denso
percentual e INT8 fake não reduziram VRAM nos modelos, porque continuam em
caminhos densos. No OPT-6.7B, `attention_pruning_10` preservou qualidade e ficou
em `0,9997x` de speedup mediano em prefill; `attention_pruning_25` preservou o
limite de qualidade, mas caiu para `0,924x`; 50% e 75% degradaram fortemente a
perplexidade.

Para reproduzir a extensão OPT-6.7B do zero, prepare o runtime se o Docker
Desktop voltar a falhar na inicialização por sockets temporários em `AppData` e
rode o wrapper guardado:

```powershell
.\scripts\prepare_docker_desktop_runtime.ps1 -StartDocker
.\scripts\run_opt_6_7b_complete_guarded.ps1
```

## Executar as novas baterias

O Docker Desktop está funcional com backend WSL2, e seu disco de dados está em
`D:\DockerDesktopData`. O cache dos três modelos OPT e do WikiText está em
`D:\Caches\scientific-initiation\huggingface`. Para inspecionar o estado sem
iniciar uma medição:

```powershell
$cacheRoot = "D:\Caches\scientific-initiation\huggingface"
.\scripts\run_benchmarks_docker.ps1 -Action Status -CacheRoot $cacheRoot
```

A sequência abaixo revalida as etapas finais e usa nomes estáveis para retomada:

```powershell
.\scripts\run_benchmarks_docker.ps1 `
  -Action Remaining `
  -CacheRoot $cacheRoot `
  -RunId "final-20260913" `
  -Resume
```

Para reproduzir somente a bateria complementar:

```powershell
.\scripts\run_benchmarks_docker.ps1 `
  -Action Complementary `
  -CacheRoot $cacheRoot `
  -RunId "complementar-20260917" `
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
`-Resume`. A execução final de 15/09/2026 completou suíte, smoke, hardware e
OPT; a execução complementar de 17/09/2026 completou hardware stress e lacunas
OPT até 6.7B. Os comandos agora servem para revalidar integridade ou retomar
apenas se algum manifesto for removido/incompleto.

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

1. decidir se `model_decode` será reexecutado com ajuste para CUDA Graphs/KV cache;
2. revisar a redação do paper curto e do relatório com o orientador;
3. preparar a apresentação final destacando que speedup físico e utilidade com
   qualidade preservada são conclusões diferentes.
