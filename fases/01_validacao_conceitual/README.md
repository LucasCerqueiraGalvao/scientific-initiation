# Fase 01 - Validação Conceitual

Esta fase valida os conceitos, fórmulas, scripts Python e ferramentas que serão
usados nos benchmarks da IC.

O objetivo é verificar se estamos medindo a coisa certa antes de medir
desempenho.

## O Que Está Validado

- projeção linear densa;
- projeções `Q`, `K` e `V`;
- scaled dot-product attention;
- self-attention;
- multi-head attention;
- informação posicional;
- bloco Transformer simplificado;
- inferência determinística;
- KV cache;
- pruning;
- sparsity;
- quantização;
- latência;
- throughput;
- memória;
- schema de benchmark controlado.

O núcleo experimental atual é um **bloco Transformer simplificado**. Ele pertence
à família Transformer no recorte estudado, mas não deve ser chamado de
Transformer completa.

## Metodologia

Cada conceito segue a lógica:

```text
referência científica
-> definição operacional
-> propriedade esperada no código
-> lógica implementada em Python
-> teste determinístico
-> limite da conclusão
```

```mermaid
flowchart TD
    A["Literatura científica"] --> B["Definição operacional"]
    B --> C["Propriedade observável"]
    C --> D["Script Python"]
    D --> E["Teste determinístico"]
    E --> F["Matriz de validação"]
    F --> G["Benchmark controlado posterior"]
```

## Como Conceitos, Código E Testes Se Relacionam

Os conceitos documentados não têm uma relação 1 para 1 com arquivos `.py`.
Alguns conceitos pertencem ao mesmo domínio técnico e, por isso, ficam no mesmo
módulo. Por exemplo, scaled dot-product attention, self-attention e multi-head
attention ficam juntos porque todos fazem parte da lógica de atenção.

Por conta disso, a pasta `validacao/` tem menos arquivos do que a lista de
conceitos validados. Isso é esperado: os arquivos representam domínios técnicos,
não capítulos isolados.

A relação correta é:

```text
docs/conceitos.txt
  define os conceitos, propriedades esperadas e limites da conclusão

validacao/*.py
  implementa funções e classes pequenas, agrupadas por domínio técnico

tests/*.py
  importa essas funções/classes e compara com referências ou valores esperados
```

```mermaid
flowchart LR
    C["Conceitos documentados<br/>docs/conceitos.txt"] --> M["Módulos Python<br/>validacao/*.py"]
    M --> T["Testes automatizados<br/>tests/*.py"]
    T --> R["Evidência de aderência<br/>manual, NumPy, PyTorch, TensorFlow/Keras, sklearn, checklist ou schema"]
```

### Mapa Código-Teste

| Módulo Python | Conceitos cobertos | Como é validado |
| --- | --- | --- |
| `attention.py` | scaled dot-product attention, self-attention, multi-head attention, split/combine heads e máscara aditiva | Compara casos pequenos com valor esperado, NumPy independente e `torch.nn.functional.scaled_dot_product_attention`. |
| `comparacao_frameworks.py` | equivalência numérica da scaled dot-product attention entre frameworks | Usa os mesmos `Q`, `K`, `V` e máscara em NumPy manual, PyTorch manual, PyTorch SDPA e Keras SDPA com backend TensorFlow. |
| `transformer.py` | bloco Transformer simplificado, Q/K/V, informação posicional, residual, normalização e FFN | Instancia o bloco, verifica componentes, preservação de shape e classificação como bloco simplificado. |
| `auditoria_transformer.py` | auditoria arquitetural da NN, dependência entre posições, determinismo e controle negativo | Abre o bloco, extrai Q/K/V, compara attention interna com PyTorch e rejeita uma NN comum `Linear + ReLU + Linear`. |
| `dense.py` | projeção linear densa | Compara `Y = XW^T + b` com cálculo manual e NumPy. |
| `kv_cache.py` | KV cache e atenção causal com/sem cache | Compara saída com cache contra saída causal completa e mede reaproveitamento de K/V. |
| `sparsity.py` | sparsity e observação de zeros | Conta zeros em tensores pequenos e classifica o nível de validade. |
| `quantization.py` | quantização simétrica int8, armazenamento e erro numérico | Verifica dtype `int8`, bytes por elemento e erro após dequantização. |
| `flops.py` | custo teórico de projeções, attention e FFN | Compara fórmulas analíticas com casos pequenos conhecidos. |
| `metrics.py` | MSE, MAE, R2 e similaridade de cosseno usadas na análise posterior | Compara implementações manuais com valores conhecidos e `scikit-learn`. |
| `protocolo.py` | schema obrigatório dos resultados | Valida colunas e campos obrigatórios do CSV de benchmark. |
| `pesquisa.py` | perguntas de pesquisa e hipóteses | Verifica se perguntas, métricas e hipóteses estão registradas. |
| `benchmark_controlado.py` | baseline, pruning, quantização, latência, throughput, memória e nível de validade | Executa uma grade pequena, registra ambiente/seed e impede promoção indevida para `hardware`. |
| `analise_resultados.py` | análise posterior dos CSVs | Compara cenários com baseline, gera tabelas e conclusões classificadas por validade. |
| `classificacao.py` | rótulos de aderência à família Transformer | Classifica como `nao_aderente`, `operacao_inspirada_em_transformer`, `bloco_transformer_simplificado` ou `transformer`. |

Assim, a pergunta "quantos conceitos existem?" é respondida pelos documentos; a
pergunta "onde isso está implementado?" é respondida pelos módulos; e a pergunta
"como sei que funciona?" é respondida pelos testes.

## Estrutura Da Fase

```text
fases/01_validacao_conceitual/
  README.md
  requirements.txt
  requirements-comparacao.txt
  validacao/
  tests/
  docs/
  evidencias/
```

- `validacao/`: scripts Python transparentes e pequenos.
- `tests/`: testes conceituais, matemáticos e de protocolo.
- `docs/`: documentos metodológicos e matriz de validação.
- `evidencias/`: resultados pequenos e reproduzíveis que sustentam os portões de validação.

## Ambiente

Criar o ambiente na raiz do repositório:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Instalar PyTorch com CUDA 12.8:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Instalar dependências da fase:

```powershell
.\.venv\Scripts\python.exe -m pip install -r fases\01_validacao_conceitual\requirements.txt
```

Para incluir a validação cruzada com TensorFlow/Keras, instalar o requisito
adicional, que também referencia as dependências comuns:

```powershell
.\.venv\Scripts\python.exe -m pip install -r fases\01_validacao_conceitual\requirements-comparacao.txt
```

No Windows nativo, TensorFlow 2.21 é usado em CPU apenas para equivalência
numérica. Isso não altera o caminho PyTorch/CUDA dos benchmarks.

Se CUDA não estiver disponível, os testes conceituais podem rodar em CPU. Nesse
caso, conclusões de hardware ficam pendentes.

## Testes

Rodar da raiz do repositório:

```powershell
.\.venv\Scripts\python.exe -m pytest fases\01_validacao_conceitual\tests -q -W error
```

Resultado observado no ambiente de validação cruzada em 24/08/2026:

```text
32 passed, 1 skipped
```

O teste ignorado exige CUDA e permanece explicitamente marcado quando o
dispositivo não está disponível. Os testes TensorFlow/Keras não foram ignorados.

## Comparação Numérica Entre Frameworks

A comparação de correção roda em CPU e usa NumPy manual como referência. Ela não
mede velocidade e não sustenta conclusão de hardware.

```powershell
$env:PYTHONPATH = "fases\01_validacao_conceitual"
.\.venv\Scripts\python.exe -m validacao.comparacao_frameworks `
  --output-dir fases\01_validacao_conceitual\evidencias\comparacao_frameworks `
  --seed 2026 --atol 1e-6 --rtol 1e-5
```

A execução canônica aprovou as nove comparações. O maior erro absoluto observado
foi `2.38418579e-07`, abaixo da tolerância absoluta de `1e-6`. Consulte a
[tabela de comparação](evidencias/comparacao_frameworks/comparacao_atencao.md),
o [CSV auditável](evidencias/comparacao_frameworks/comparacao_atencao.csv) e os
[metadados do ambiente](evidencias/comparacao_frameworks/comparacao_atencao.metadata.json).

## Benchmark Piloto

O benchmark ainda é etapa experimental. Ele não deve ser usado sozinho para
afirmar ganho real de hardware.

Exemplo em CPU:

```powershell
$env:PYTHONPATH = "fases\01_validacao_conceitual"
.\.venv\Scripts\python.exe -m validacao.benchmark_controlado --device cpu --warmup 1 --repetitions 3 --output resultados\benchmark_controlado.csv
```

## Benchmark Das Operações Isoladas

O runner destinado ao protocolo principal é `validacao.benchmark_operacoes`.
Ele preserva o schema CSV existente, mas mede separadamente projeção densa e
self-attention de cabeça única. Configurações, execuções independentes, hashes,
ambiente, manifesto e log ficam fora do CSV para não quebrar a interface.

Configurações versionadas:

- `experimentos/smoke_cpu.json`: validação curta da pipeline, sem conclusão de
  hardware;
- `experimentos/benchmark_principal_gpu.json`: seed 42, lote 1,
  `L={64,128,256}`, `D={128,256,512}`, 20 warm-ups, 50 medições e duas execuções
  na RTX 4070 Ti Super.

Exemplo de smoke:

```powershell
$env:PYTHONPATH = "fases\01_validacao_conceitual"
.\.venv\Scripts\python.exe -m validacao.benchmark_operacoes `
  --config fases\01_validacao_conceitual\experimentos\smoke_cpu.json `
  --output-dir fases\01_validacao_conceitual\evidencias\benchmarks\smoke_cpu_<data>
```

O diretório precisa estar vazio. O runner não sobrescreve evidências. Em CPU,
`max_memory_bytes` é estimativa dos tensores de execução; em CUDA, é o pico
reportado por `torch.cuda.max_memory_allocated`. A contagem de FLOPs da
self-attention inclui projeções e multiplicações matriciais, mas exclui softmax
e escalonamento, conforme declarado nos metadados.

## O Que Ainda Não Está Afirmado

Esta fase ainda não afirma:

- que foi implementada uma Transformer completa;
- que pruning gera ganho real de hardware;
- que quantização acelera a execução em hardware;
- que alguma técnica é superior antes dos benchmarks controlados.

Para chamar um resultado de hardware, será necessário mostrar evidência de
representação, formato, kernel ou caminho de execução compatível.

## Documentos Principais

- [Base teórica e validação](docs/base_teorica_validacao.md)
- [Validação arquitetural da Transformer](docs/validacao_transformer.md)
- [Matriz de validação de ferramentas](docs/matriz_validacao_ferramentas.md)
- [Protocolo experimental](docs/protocolo_experimental.md)
- [Perguntas de pesquisa](docs/perguntas_pesquisa.md)
- [Confronto com literatura](docs/confronto_resultados_literatura.md)
- [Explicação narrativa da metodologia](docs/explicacao_metodologia_validacao.md)
- [Resumo operacional dos conceitos](docs/conceitos.txt)
- [Comparação NumPy, PyTorch e TensorFlow/Keras](evidencias/comparacao_frameworks/comparacao_atencao.md)
