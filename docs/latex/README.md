# Documentação LaTeX

Esta pasta contém a fonte canônica da documentação acadêmica da IC. O arquivo
principal é `relatorio_ic.tex`; os textos técnicos e registros são incluídos a
partir de fontes `.tex` mantidas próximas de seu contexto no repositório.

Para compilar no Windows com MiKTeX e XeLaTeX, execute na raiz:

```powershell
.\scripts\build_docs.ps1
```

O PDF final é gravado em `output/pdf/relatorio_ic_transformers.pdf`. O paper
curto derivado fica em `output/pdf/paper_ic_transformers.pdf`. Arquivos
auxiliares ficam em `tmp/pdfs/latex/` e não são versionados.

Estado da documentação em 17/09/2026: os PDFs incluem validações conceituais
18/18, benchmark sintético v3, benchmark físico v3, OPT v1, OPT-2.7B e OPT-6.7B.
O relatório principal preserva a limitação técnica de decode com KV cache/CUDA
Graphs e registra que o speedup físico observado no OPT-6.7B não foi aceito como
configuração final por degradação de perplexidade.

Política documental:

- `.tex`: texto acadêmico, método, atas interpretadas, diário e relatórios;
- `.md`: apenas READMEs de navegação no GitHub;
- `.csv`, `.json`, `.log` e `.png`: evidências auditáveis;
- `transcricao-completa.md`: cópia bruta preservada; a versão compilável está em
  `transcricao-completa.tex`;
- `.docx` e `.pptx`: originais institucionais ou materiais de apresentação
  preservados, nunca a fonte canônica do relatório.
