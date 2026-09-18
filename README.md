# Scientific Initiation

Iniciação científica sobre eficiência computacional de operações centrais de
Transformers em inferência, com comparação controlada entre baseline, pruning
por magnitude e quantização linear INT8.

O benchmark GPU v1 foi concluído na NVIDIA GeForce RTX 4070 Ti SUPER e sua
evidência foi preservada com hashes. A infraestrutura v3 ampliou o estudo para
cinco seeds, três distribuições, 13 perfis, multi-head attention, kernels reais
e modelos OPT pré-treinados. As baterias sintética v3, física v3, OPT v1,
OPT-2.7B e OPT-6.7B foram executadas; a etapa complementar de 17/09/2026
preencheu desempenho/VRAM dos prunings percentuais e INT8 fake nos OPT, além
dos perfis físicos de stress. A etapa OPT valida qualidade, prefill e TTFT,
preservando a falha técnica da medição de decode com KV cache e CUDA Graphs.

O Docker Desktop voltou a operar com backend WSL2. Seu disco de dados foi
realocado para `D:\DockerDesktopData`, preservando os contêineres existentes e
evitando que as imagens científicas ocupem o disco `C:`. A imagem do benchmark
foi construída com sucesso e suas dependências passaram em `pip check`; o
probe físico, o smoke e a bateria de 1.650 casos foram concluídos.

## Estado atual

| Frente | Estado | Evidência |
| --- | --- | --- |
| Scaled dot-product attention manual | Validada | Casos conhecidos, NumPy independente e PyTorch SDPA. |
| Bloco Transformer simplificado | Validado no recorte | Auditoria estrutural, dependência entre posições e controle negativo. |
| Comparação entre frameworks | Validada | 18 comparações cobrem SDPA e multi-head completa com projeções Q/K/V/O. |
| Pipeline de benchmark | Validada em CPU e GPU | Configuração, retomada, hashes, manifesto, análise pareada e gráficos. |
| Inicialização dos pesos | Corrigida | Contrato v2 com Xavier normal e bias zero. |
| Benchmark GPU v1 | Concluído | Duas execuções, 108 registros, hashes, análise e gráficos versionados. |
| Benchmark sintético v3 | Concluído | 2.340 registros completos, 1.950 pares e 117 mil timings na GPU. |
| Caminhos físicos | Concluído | 1.650 registros, 82.500 timings, TorchAO INT8 e pruning 2:4 auditados pelo profiler. |
| Modelos OPT v1 | Concluído com limitação de decode | OPT-125M, OPT-350M e OPT-1.3B: 45 registros de qualidade, 2.880 janelas, 900 métricas de prompts, 1.296 registros de desempenho e 10.080 timings. |
| Extensão OPT-2.7B e OPT-6.7B | Concluída | Cada modelo adicional gerou 15 registros de qualidade, 216 registros de desempenho e 1.440 timings. |
| Benchmarks complementares | Concluídos | Hardware stress: 300 registros e 15.000 timings. OPT lacunas: 40 registros de qualidade, 1.080 registros de desempenho e 7.200 timings. |
| Ganho real nas operações/modelos | Parcial e condicionado | Operações isoladas contradisseram speedup; no OPT-6.7B houve speedup físico em alguns caminhos, mas sem qualidade aceitável nos blocos. |

A posição científica correta é: a evidência v1 demonstra comportamento numérico
e a v3 sustenta a robustez numérica em múltiplas entradas. Em todos os casos
pareados, INT8 fake apresentou MSE menor que pruning, e o erro do pruning cresceu
com a sparsity. Na bateria física, INT8 dinâmico e 2:4 usaram os kernels esperados,
mas nenhum cenário foi mais rápido que seu baseline compilado. Nos modelos OPT,
INT8 manteve melhor qualidade que pruning e reduziu memória/armazenamento em
vários cenários. O complemento mostrou que pruning denso percentual e INT8 fake
nos OPT não reduzem VRAM, pois continuam executando em caminhos densos. No
OPT-6.7B, pruning de 10% em atenção preservou qualidade e ficou praticamente
neutro em latência (`0,9997x` em prefill), enquanto 25% ainda preservou o limite
de qualidade, mas caiu para `0,924x`; 50% e 75% degradaram a perplexidade. O
OPT-6.7B mostrou que a escala pode favorecer alguns caminhos físicos:
`blocks_int8_dynamic` atingiu speedup mediano de `1,756x` em prefill e `1,744x`
em TTFT. Contudo, a perplexidade subiu `165,05%`, portanto esse ganho não é uma
configuração final aceitável pelos critérios operacionais. Resultado físico
negativo também é evidência: representação menor não implica menor latência para
todo shape ou composição de operação.

## Documentação LaTeX

Toda a documentação acadêmica tem fonte canônica em LaTeX. Markdown é mantido
somente nos READMEs, porque eles funcionam como navegação operacional no GitHub.
CSV, JSON, logs e imagens são evidências; a transcrição original em Markdown e
o formulário institucional em Word são preservados como fontes brutas.

- [Relatório compilado](output/pdf/relatorio_ic_transformers.pdf)
- [Paper curto compilado](output/pdf/paper_ic_transformers.pdf)
- [Planilha editável dos benchmarks](output/spreadsheets/benchmarks_transformers_resumo.xlsx)
- [Fonte principal](docs/latex/relatorio_ic.tex)
- [Como compilar](docs/latex/README.md)
- [Índice dos documentos](docs/README.md)
- [Fase de validação](fases/01_validacao_conceitual/README.md)
- [Registro da reunião de 31/07/2026](docs/reunioes/2026-07-31/registro.tex)

O PDF reúne relatório executivo, plano de trabalho, fundamentação, protocolo,
validações, evidências preliminares, diário, reunião, material de estudo e a
transcrição integral.

## Estrutura do repositório

```text
docs/
  latex/                       fonte principal e identidade visual
  reunioes/                    registros em LaTeX e transcrições brutas
  apresentacoes/               slides e apoio em LaTeX
  plano_trabalho/              formulário institucional original

fases/01_validacao_conceitual/
  docs/                        fundamentação e método em LaTeX
  experimentos/                configurações JSON versionadas
  validacao/                   implementações e runners auditáveis
  tests/                       testes determinísticos e de contrato
  evidencias/                  CSV, JSON, logs, figuras e relatórios LaTeX

output/pdf/                    relatório acadêmico compilado
scripts/                       build documental e orquestração dos benchmarks
legado/                        protótipo inicial preservado
```

## Executar e reproduzir

Para testes locais:

```powershell
$env:PYTHONPATH = "fases\01_validacao_conceitual"
.\.venv\Scripts\python.exe -m pytest `
  fases\01_validacao_conceitual\tests -q -W error
```

O teste de integração com download é deliberadamente ignorado até
`RUN_HF_INTEGRATION=1` ser definido. Para revalidar os 18 casos conceituais:

```powershell
.\.venv\Scripts\python.exe -m validacao.comparacao_frameworks `
  --output-dir resultados\comparacao_frameworks_gpu_host `
  --seed 2026 --atol 1e-6 --rtol 1e-5
```

Aceite somente se as 18 linhas do CSV tiverem `passed=True`. Essa execução
não mede velocidade entre frameworks.

O ambiente físico é padronizado em Docker Linux. Nesta máquina, use sempre o
cache em `D:`. O comando `Status` não inicia medições e mostra Docker, imagem,
cache, GPU e espaço em disco:

```powershell
$cacheRoot = "D:\Caches\scientific-initiation\huggingface"
.\scripts\run_benchmarks_docker.ps1 -Action Status -CacheRoot $cacheRoot
```

Para revalidar a sequência final ou retomar uma execução interrompida, sem
repetir a bateria sintética v3:

```powershell
.\scripts\run_benchmarks_docker.ps1 `
  -Action Remaining `
  -CacheRoot $cacheRoot `
  -RunId "final-20260913" `
  -Resume
```

`Remaining` percorre `Build`, `Probe`, `Test`, `Smoke`, `ModelSmoke`, `Hardware`
e `Models`, mas ignora etapas já concluídas quando manifestos e checksums são
válidos.
As ações individuais continuam disponíveis; `All` mantém a sequência histórica,
incluindo `Prefetch` e `Synthetic`. Downloads ocorrem apenas no prefetch; testes
e coletas posteriores usam rede desativada. `-Resume` só retoma resultados com
o mesmo `RunId` quando configuração, código, ambiente e artefatos preservados
possuem hashes coincidentes.

Para reproduzir somente a bateria complementar que preencheu as lacunas da
planilha:

```powershell
.\scripts\run_benchmarks_docker.ps1 `
  -Action Complementary `
  -CacheRoot $cacheRoot `
  -RunId "complementar-20260917" `
  -Resume
```

`Complementary` roda status/probe, smoke complementar, hardware stress, OPT
complementar para 125M/350M/1.3B/2.7B e OPT-6.7B com guarda de VRAM em
15.800 MiB. Etapas completas são ignoradas se manifesto e checksums continuarem
válidos.

Antes de `Probe`, `Smoke`, `ModelSmoke`, `Hardware` e `Models`, o script amostra a GPU cinco
vezes e exige: processo do jogo fechado, uso médio abaixo de 10%, no máximo 2.048 MiB
de VRAM ocupada e temperatura abaixo de 65 °C. O pico de uso também é registrado
no diagnóstico. Como o WDDM pode reportar utilização residual incorreta, existe
um segundo critério conservador: soma dos processos abaixo de 10%, potência até
35 W e clock gráfico até 300 MHz, mantendo os mesmos limites de VRAM e temperatura.
O script nunca encerra processos para atender a esses limites.

Estado da sequência final na RTX 4070 Ti SUPER:

| Etapa | Tempo esperado |
| --- | ---: |
| Build, probe e bateria física | Concluídos |
| Suíte e smoke OPT final | Concluídos |
| OPT-125M, OPT-350M e OPT-1.3B | Concluídos |
| Complemento de hardware stress e OPT lacunas | Concluído |
| Análise, PDF e publicação | Atualizados nesta consolidação |
| **Pendência técnica** | investigar/reexecutar decode se necessário |

Extensão exploratória de escala: o OPT-2.7B e o OPT-6.7B foram executados
separadamente da bateria OPT v1. O OPT-6.7B usou guarda de VRAM em 15.800 MiB,
retomada por cenário e execução isolada por container para reduzir risco de
travamento.

Se o Docker Desktop falhar na inicialização por sockets temporários, rode antes:

```powershell
.\scripts\prepare_docker_desktop_runtime.ps1 -StartDocker
```

## Compilar o relatório

Com MiKTeX e XeLaTeX instalados:

```powershell
.\scripts\build_docs.ps1
```

O build faz três passagens e grava:

```text
output/pdf/relatorio_ic_transformers.pdf
```

Arquivos auxiliares ficam em `tmp/pdfs/latex/` e não são versionados.

## Próximos passos

1. decidir se a medição de decode será reexecutada com ajuste específico para
   CUDA Graphs/KV cache ou registrada definitivamente como limitação técnica;
2. revisar com o orientador a interpretação do pruning denso nos OPT e do
   OPT-6.7B, separando speedup físico de utilidade com qualidade preservada;
3. preparar apresentação/discussão dos resultados finais com o orientador.

## Cuidados de interpretação

- pruning não estruturado em uma matriz densa pode aumentar a esparsidade sem
  reduzir a latência;
- quantizar e dequantizar para `float32` valida erro numérico, não um kernel
  INT8;
- resultados CPU são diagnósticos da pipeline, não estimativas da RTX;
- FLOPs analíticos e tempo medido respondem perguntas diferentes;
- nenhuma conclusão deve ser promovida de algorítmica para hardware sem
  evidência do caminho físico executado.
