# Simpósio 2026

Materiais usados na apresentação do simpósio científico.

Esta pasta não faz parte da fase ativa de validação conceitual. Ela preserva os
slides, roteiros, guias, previews e scripts usados para preparar a apresentação.

## Estrutura

```text
simposio_2026/
  template - I simposio de CD.pptx
  gerador/
  preview_slides/
  roteiros/
  guias/
```

- `gerador/`: conteúdo estruturado, script PowerShell e arquivos gerados.
- `preview_slides/`: imagens exportadas dos slides para conferência visual.
- `roteiros/`: roteiro de fala em LaTeX.
- `guias/`: guia de estudo em LaTeX.

Fontes textuais:

- [Guia de estudo](guias/guia_estudo.tex)
- [Roteiro de 10 minutos](roteiros/roteiro_10_minutos.tex)

Os arquivos `.pptx`, previews e o gerador PowerShell são artefatos da
apresentação; a documentação textual canônica foi migrada para LaTeX.

Para uma apresentação atualizada ao orientador, use o relatório consolidado em
`../../../output/pdf/relatorio_ic_transformers.pdf` e o paper curto em
`../../../output/pdf/paper_ic_transformers.pdf`, pois estes já incluem OPT-2.7B
e OPT-6.7B.
