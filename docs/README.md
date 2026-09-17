# Documentação

Este README é o índice de navegação no GitHub. A fonte acadêmica canônica é
LaTeX e o volume completo está disponível no
[PDF compilado](../output/pdf/relatorio_ic_transformers.pdf).

## Organização

```text
docs/
  latex/                 documento mestre, preâmbulo e capítulos executivos
  reunioes/              registros interpretados em LaTeX e fontes brutas
  apresentacoes/         slides, roteiro e guia de estudo
  plano_trabalho/        formulário institucional submetido
```

Textos de acompanhamento:

- [Diário da IC](diario_ic.tex)
- [Paper curto compilado](../output/pdf/paper_ic_transformers.pdf)
- [Auditoria de 25/08/2026](auditoria_estado_2026-08-25.tex)
- [Checklist pré-benchmark](checklist_pre_benchmark.tex)
- [Relatório da execução autônoma](relatorio_execucao_autonoma_2026-08-25.tex)
- [Registro da reunião de 31/07/2026](reunioes/2026-07-31/registro.tex)

Documentos técnicos da fase ativa:

- [Base teórica](../fases/01_validacao_conceitual/docs/base_teorica_validacao.tex)
- [Validação arquitetural](../fases/01_validacao_conceitual/docs/validacao_transformer.tex)
- [Protocolo experimental](../fases/01_validacao_conceitual/docs/protocolo_experimental.tex)
- [Matriz de ferramentas](../fases/01_validacao_conceitual/docs/matriz_validacao_ferramentas.tex)
- [Decisão sobre inicialização](../fases/01_validacao_conceitual/docs/decisao_inicializacao_benchmark.tex)

## Política de formatos

- `.tex`: documentação acadêmica editável;
- `.md`: somente READMEs e a transcrição original preservada;
- `.csv`, `.json`, `.log`, `.png`: evidência experimental;
- `.docx`: formulário institucional original;
- `.pptx`: apresentação original.

## Build

Na raiz do repositório, com MiKTeX/XeLaTeX:

```powershell
.\scripts\build_docs.ps1
```

O resultado é `output/pdf/relatorio_ic_transformers.pdf`.

Estado atual: o relatório completo e o paper curto já incorporam as extensões
OPT-2.7B e OPT-6.7B, incluindo a distinção entre speedup físico observado e
utilidade condicionada à preservação de qualidade.
