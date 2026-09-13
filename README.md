# Scientific Initiation

Iniciação científica sobre eficiência computacional de operações centrais de
Transformers em inferência, com comparação controlada entre baseline, pruning
por magnitude e quantização linear INT8.

O benchmark GPU v1 foi concluído na NVIDIA GeForce RTX 4070 Ti SUPER e sua
evidência foi preservada com hashes. A infraestrutura v3 amplia o estudo para
cinco seeds, três distribuições, 13 perfis, multi-head attention, kernels reais
e modelos OPT pré-treinados. A bateria sintética v3 também foi concluída; as
coletas físicas e OPT permanecem separadas até a confirmação dos kernels.

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
| Caminhos físicos | Implementado | Docker, probe, TorchAO INT8 e pruning 2:4 com confirmação por profiler. |
| Modelos OPT | Preparado | Runner completo e 6,63 GB de modelos/dataset armazenados para uso offline. |
| Ganho real de pruning/INT8 | Não afirmado | Depende do caminho e do kernel efetivamente executados na GPU. |

A posição científica correta é: a evidência v1 demonstra comportamento numérico
e a v3 sustenta a robustez numérica em múltiplas entradas. Em todos os casos
pareados, INT8 fake apresentou MSE menor que pruning, e o erro do pruning cresceu
com a sparsity. Nenhuma delas prova aceleração física de INT8 ou pruning; essa
afirmação depende da bateria com profiler e kernels reais.

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
scripts/                       build da documentação
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

O ambiente físico é padronizado em Docker Linux. Com Docker Desktop usando WSL2
e integração NVIDIA ativos, cada ação pode ser executada isoladamente:

```powershell
.\scripts\run_benchmarks_docker.ps1 -Action Build
.\scripts\run_benchmarks_docker.ps1 -Action Probe
.\scripts\run_benchmarks_docker.ps1 -Action Prefetch
.\scripts\run_benchmarks_docker.ps1 -Action Smoke
.\scripts\run_benchmarks_docker.ps1 -Action Synthetic
.\scripts\run_benchmarks_docker.ps1 -Action Hardware
.\scripts\run_benchmarks_docker.ps1 -Action Models
```

O cache pode ficar em outro disco, sem alterar o experimento:

```powershell
.\scripts\run_benchmarks_docker.ps1 -Action Prefetch `
  -CacheRoot "D:\Caches\scientific-initiation\huggingface"
```

`-Action All` executa toda a sequência. Downloads ocorrem apenas no prefetch;
testes e coletas posteriores usam rede desativada. Configuração, código,
ambiente, CSVs e timings são hasheados, e `--resume` só aceita uma retomada
quando esses contratos coincidem.

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

1. restaurar o daemon Linux do Docker Desktop e passar o probe e o smoke;
2. coletar a bateria física sem alterar os JSONs;
3. avaliar os três OPT pré-treinados sem treinamento ou fine-tuning;
4. revisar casos suportados, incompatíveis e falhos antes de interpretar médias;
5. incorporar as evidências físicas e de modelos ao relatório.

## Cuidados de interpretação

- pruning não estruturado em uma matriz densa pode aumentar a esparsidade sem
  reduzir a latência;
- quantizar e dequantizar para `float32` valida erro numérico, não um kernel
  INT8;
- resultados CPU são diagnósticos da pipeline, não estimativas da RTX;
- FLOPs analíticos e tempo medido respondem perguntas diferentes;
- nenhuma conclusão deve ser promovida de algorítmica para hardware sem
  evidência do caminho físico executado.
