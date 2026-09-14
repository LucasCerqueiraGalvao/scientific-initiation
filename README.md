# Scientific Initiation

Iniciação científica sobre eficiência computacional de operações centrais de
Transformers em inferência, com comparação controlada entre baseline, pruning
por magnitude e quantização linear INT8.

O benchmark GPU v1 foi concluído na NVIDIA GeForce RTX 4070 Ti SUPER e sua
evidência foi preservada com hashes. A infraestrutura v3 amplia o estudo para
cinco seeds, três distribuições, 13 perfis, multi-head attention, kernels reais
e modelos OPT pré-treinados. A bateria sintética v3 também foi concluída; as
operações físicas foram coletadas com kernels confirmados, enquanto a bateria
OPT final permanece separada até uma janela sem concorrência da GPU.

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
| Modelos OPT | Smoke diagnóstico validado | Runner, cache offline e separação entre prefill, TTFT e decode validados; smoke limpo e coleta final pendentes. |
| Ganho real nas operações | Não observado | Os seis grupos físicos tiveram intervalo de speedup inteiramente abaixo de `1,0x`. |

A posição científica correta é: a evidência v1 demonstra comportamento numérico
e a v3 sustenta a robustez numérica em múltiplas entradas. Em todos os casos
pareados, INT8 fake apresentou MSE menor que pruning, e o erro do pruning cresceu
com a sparsity. Na bateria física, INT8 dinâmico e 2:4 usaram os kernels esperados,
mas nenhum cenário foi mais rápido que seu baseline compilado. Resultado físico
negativo também é evidência: representação menor não implica menor latência para
todo shape ou composição de operação.

## Documentação LaTeX

Toda a documentação acadêmica tem fonte canônica em LaTeX. Markdown é mantido
somente nos READMEs, porque eles funcionam como navegação operacional no GitHub.
CSV, JSON, logs e imagens são evidências; a transcrição original em Markdown e
o formulário institucional em Word são preservados como fontes brutas.

- [Relatório compilado](output/pdf/relatorio_ic_transformers.pdf)
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

Para executar somente o trabalho pendente, sem repetir a bateria sintética v3:

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

Antes de `Probe`, `Smoke`, `ModelSmoke`, `Hardware` e `Models`, o script amostra a GPU cinco
vezes e exige: processo do jogo fechado, uso médio abaixo de 10%, no máximo 2.048 MiB
de VRAM ocupada e temperatura abaixo de 65 °C. O pico de uso também é registrado
no diagnóstico. Como o WDDM pode reportar utilização residual incorreta, existe
um segundo critério conservador: soma dos processos abaixo de 10%, potência até
35 W e clock gráfico até 300 MHz, mantendo os mesmos limites de VRAM e temperatura.
O script nunca encerra processos para atender a esses limites.

Estimativa para a sequência restante na RTX 4070 Ti SUPER:

| Etapa | Tempo esperado |
| --- | ---: |
| Build, probe e bateria física | Concluídos |
| Suíte e smoke OPT final | 15–40 min |
| OPT-125M, OPT-350M e OPT-1.3B | 3–6 h |
| Análise, PDF e publicação | 30–90 min |
| **Total restante** | **4–8 h** |

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

1. executar o smoke OPT final quando o portão de GPU permitir;
2. avaliar os três OPT sem treinamento ou fine-tuning;
3. revisar os manifestos OPT, compilar o PDF e publicar a consolidação.

## Cuidados de interpretação

- pruning não estruturado em uma matriz densa pode aumentar a esparsidade sem
  reduzir a latência;
- quantizar e dequantizar para `float32` valida erro numérico, não um kernel
  INT8;
- resultados CPU são diagnósticos da pipeline, não estimativas da RTX;
- FLOPs analíticos e tempo medido respondem perguntas diferentes;
- nenhuma conclusão deve ser promovida de algorítmica para hardware sem
  evidência do caminho físico executado.
