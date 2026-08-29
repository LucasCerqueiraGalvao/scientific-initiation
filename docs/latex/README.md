# Documentação LaTeX

Esta pasta contém a fonte canônica da documentação acadêmica da IC. O arquivo
principal é `relatorio_ic.tex`; os textos técnicos e registros são incluídos a
partir de fontes `.tex` mantidas próximas de seu contexto no repositório.

Para compilar no Windows com MiKTeX e XeLaTeX, execute na raiz:

```powershell
.\scripts\build_docs.ps1
```

O PDF final é gravado em `output/pdf/relatorio_ic_transformers.pdf`. Arquivos
auxiliares ficam em `tmp/pdfs/latex/` e não são versionados.

Política documental:

- `.tex`: texto acadêmico, método, atas interpretadas, diário e relatórios;
- `.md`: apenas READMEs de navegação no GitHub;
- `.csv`, `.json`, `.log` e `.png`: evidências auditáveis;
- `transcricao-completa.md`: cópia bruta preservada; a versão compilável está em
  `transcricao-completa.tex`;
- `.docx` e `.pptx`: originais institucionais ou materiais de apresentação
  preservados, nunca a fonte canônica do relatório.
