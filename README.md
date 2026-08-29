# Scientific Initiation

Iniciação científica sobre eficiência computacional de operações centrais de
Transformers em inferência, com comparação controlada entre baseline, pruning
por magnitude e quantização linear INT8.

O repositório está pronto para a coleta principal na máquina com uma NVIDIA
GeForce RTX 4070 Ti Super de 16 GB. A matemática da atenção, a arquitetura do
recorte, a equivalência NumPy/PyTorch/TensorFlow e a pipeline de benchmark já
foram validadas. O que falta é executar a grade CUDA e analisar a evidência do
hardware-alvo.

## Estado atual

| Frente | Estado | Evidência |
| --- | --- | --- |
| Scaled dot-product attention manual | Validada | Casos conhecidos, NumPy independente e PyTorch SDPA. |
| Bloco Transformer simplificado | Validado no recorte | Auditoria estrutural, dependência entre posições e controle negativo. |
| Comparação entre frameworks | Validada | 9/9 comparações NumPy, PyTorch manual, PyTorch SDPA e Keras/TensorFlow aprovadas. |
| Pipeline de benchmark | Validada em CPU | Configuração, duas execuções, hashes, manifesto, análise e gráficos. |
| Inicialização dos pesos | Corrigida | Contrato v2 com Xavier normal e bias zero. |
| Benchmark principal em CUDA | Pendente | Deve ser executado na RTX 4070 Ti Super. |
| Ganho real de pruning/INT8 | Não afirmado | Depende do caminho e do kernel efetivamente executados na GPU. |

A posição científica correta é: o projeto implementa e testa operações de um
**bloco Transformer simplificado**. Ele ainda não demonstra uma Transformer
completa nem superioridade de uma técnica de otimização em hardware.

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

## Rodar na máquina com a placa de vídeo

As instruções abaixo assumem Windows, PowerShell, Git, Python 3.11, driver
NVIDIA compatível e a RTX 4070 Ti Super disponível.

### 1. Clonar e criar o ambiente

```powershell
git clone https://github.com/LucasCerqueiraGalvao/scientific-initiation.git
cd scientific-initiation

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Instale primeiro a build CUDA fixada do PyTorch:

```powershell
.\.venv\Scripts\python.exe -m pip install `
  torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 `
  --index-url https://download.pytorch.org/whl/cu128
```

Depois instale as dependências comuns e a comparação TensorFlow/Keras:

```powershell
.\.venv\Scripts\python.exe -m pip install `
  -r fases\01_validacao_conceitual\requirements-comparacao.txt
```

TensorFlow/Keras é usado em CPU apenas para equivalência numérica. O benchmark
de desempenho continua no caminho PyTorch/CUDA.

### 2. Confirmar o ambiente antes da coleta

```powershell
nvidia-smi

.\.venv\Scripts\python.exe -c `
  "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0))"

.\.venv\Scripts\python.exe -m pip check
```

Não avance se `torch.cuda.is_available()` retornar `False`, se o nome da GPU não
for o esperado ou se `pip check` apontar conflito.

### 3. Executar a suíte completa

```powershell
$env:PYTHONPATH = "fases\01_validacao_conceitual"
.\.venv\Scripts\python.exe -m pytest `
  fases\01_validacao_conceitual\tests -q -W error
```

No ambiente local sem NVIDIA, o resultado registrado é `45 passed, 1 skipped`.
Na máquina CUDA, o teste protegido pelo portão da GPU deve executar em vez de
ser ignorado. TensorFlow/Keras não pode ser `skip` em nenhum dos ambientes.

### 4. Revalidar a atenção no clone novo

Use uma pasta nova em `resultados/`, que é ignorada pelo Git durante testes
locais:

```powershell
.\.venv\Scripts\python.exe -m validacao.comparacao_frameworks `
  --output-dir resultados\comparacao_frameworks_gpu_host `
  --seed 2026 --atol 1e-6 --rtol 1e-5
```

Aceite somente se as nove linhas do CSV tiverem `passed=True`. Essa execução
não mede velocidade entre frameworks.

### 5. Executar o benchmark principal

A configuração formal já está versionada em
`fases/01_validacao_conceitual/experimentos/benchmark_principal_gpu.json`. Ela
usa seed 42, lote 1, `L={64,128,256}`, `D={128,256,512}`, 20 warm-ups, 50
medições e duas execuções independentes.

O diretório de saída precisa ser novo e vazio:

```powershell
$env:PYTHONPATH = "fases\01_validacao_conceitual"
$evidenceDir = "resultados\benchmark_gpu_4070ti_super_YYYY-MM-DD"

.\.venv\Scripts\python.exe -m validacao.benchmark_operacoes `
  --config fases\01_validacao_conceitual\experimentos\benchmark_principal_gpu.json `
  --output-dir $evidenceDir

.\.venv\Scripts\python.exe -m validacao.analise_benchmark_operacoes `
  --evidence-dir $evidenceDir
```

Não reutilize uma pasta anterior e não edite CSVs manualmente: a análise valida
os checksums do manifesto e deve recusar evidência alterada.

### 6. Conferir a coleta

Antes de interpretar os números, confirme:

- duas execuções completas para todas as combinações e cenários;
- ausência de NaN e infinito;
- `manifest.json` e hashes aprovados;
- GPU, driver, CUDA, PyTorch, seed e configuração registrados;
- comparação de cada técnica contra o baseline da mesma execução e shape;
- relatório `relatorio_preliminar.tex`, CSVs consolidados e dois gráficos;
- distinção entre memória medida, memória teórica, latência e FLOPs analíticos.

### 7. Versionar a evidência aprovada

Depois de revisar a coleta, copie a pasta de `resultados/` para:

```text
fases/01_validacao_conceitual/evidencias/benchmarks/
```

Use um nome imutável com data e hardware. Em seguida, atualize o relatório
LaTeX, recompile o PDF, rode a suíte novamente e faça um commit que mantenha
configuração, ambiente, CSVs, manifesto, análise e texto juntos.

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

1. executar o clone e a validação CUDA na máquina alvo;
2. coletar a grade principal sem alterar a configuração versionada;
3. auditar se pruning usa caminho esparso e se INT8 usa kernel/representação
   realmente quantizados antes de falar em ganho de hardware;
4. analisar separadamente projeção densa e self-attention por shape;
5. incorporar tabelas, gráficos, dispersão entre execuções, limitações e ameaças
   à validade no relatório LaTeX;
6. publicar a evidência e o PDF atualizados no GitHub;
7. somente depois avaliar uma demonstração opcional com modelo pré-treinado.

## Cuidados de interpretação

- pruning não estruturado em uma matriz densa pode aumentar a esparsidade sem
  reduzir a latência;
- quantizar e dequantizar para `float32` valida erro numérico, não um kernel
  INT8;
- resultados CPU são diagnósticos da pipeline, não estimativas da RTX;
- FLOPs analíticos e tempo medido respondem perguntas diferentes;
- nenhuma conclusão deve ser promovida de algorítmica para hardware sem
  evidência do caminho físico executado.
