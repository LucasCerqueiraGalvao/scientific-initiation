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

O objeto arquitetural é um **bloco Transformer simplificado**. A coleta
principal em CUDA permanece pendente e nenhuma técnica é tratada como superior
antes dessa evidência.

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

Resultado registrado no ambiente sem NVIDIA: `45 passed, 1 skipped`. O único
skip é protegido por disponibilidade CUDA; a comparação TensorFlow/Keras roda
sem skip.

## Comparação numérica entre frameworks

```powershell
$env:PYTHONPATH = "fases\01_validacao_conceitual"
.\.venv\Scripts\python.exe -m validacao.comparacao_frameworks `
  --output-dir resultados\comparacao_frameworks `
  --seed 2026 --atol 1e-6 --rtol 1e-5
```

A evidência canônica aprovou 9/9 comparações; o maior erro absoluto foi
`2.38418579e-07`. Consulte o [relatório LaTeX](evidencias/comparacao_frameworks/comparacao_atencao.tex),
o [CSV](evidencias/comparacao_frameworks/comparacao_atencao.csv) e os
[metadados](evidencias/comparacao_frameworks/comparacao_atencao.metadata.json).
Essa etapa valida correção, não velocidade.

## Configurações de benchmark

| Arquivo | Uso | Limite da conclusão |
| --- | --- | --- |
| `experimentos/smoke_cpu.json` | Pipeline curta | Nenhuma conclusão de hardware. |
| `experimentos/diagnostico_cpu_l64_d128.json` | Um ponto formal em CPU | Diagnóstico metodológico. |
| `experimentos/benchmark_principal_gpu.json` | Grade CUDA na RTX 4070 Ti Super | Base para a análise principal, após auditoria do caminho. |

O schema v2 fixa entradas `N(0,1)`, pesos Xavier normal ajustados por dimensão e
bias zero. Evidências v1 são preservadas para documentar por que o contrato foi
corrigido.

## Executar a coleta principal

O diretório precisa ser novo e vazio:

```powershell
$env:PYTHONPATH = "fases\01_validacao_conceitual"
$evidenceDir = "resultados\benchmark_gpu_4070ti_super_YYYY-MM-DD"

.\.venv\Scripts\python.exe -m validacao.benchmark_operacoes `
  --config fases\01_validacao_conceitual\experimentos\benchmark_principal_gpu.json `
  --output-dir $evidenceDir

.\.venv\Scripts\python.exe -m validacao.analise_benchmark_operacoes `
  --evidence-dir $evidenceDir
```

A análise gera CSVs consolidados, `reprodutibilidade.json`,
`relatorio_preliminar.tex`, `latencia_relativa.png` e `qualidade_saida.png`.
Ela recusa artefatos cujo checksum não coincide com o manifesto.

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

1. confirmar driver, CUDA e nome da GPU no clone novo;
2. executar toda a suíte sem o skip de CUDA;
3. coletar a grade principal sem alterar o JSON versionado;
4. revisar hashes, dispersão e comparações por operação/shape;
5. verificar se pruning e INT8 acionam caminhos físicos compatíveis antes de
   classificar qualquer ganho como evidência de hardware;
6. versionar a coleta aprovada e atualizar o relatório LaTeX.
