# Paper da IC em Transformers

Este diretorio contem uma primeira versao de paper academico para a pesquisa de
eficiencia computacional em operacoes de Transformers. O desenho visual segue o
paper de referencia do projeto PCE, mas o conteudo e especifico desta iniciacao
cientifica.

## Arquivos

- `main.tex`: corpo principal do paper.
- `referencias.bib`: referencias bibliograficas usadas no texto.
- `figuras/`: brasao institucional e figuras copiadas das evidencias aprovadas.
- `main.pdf`: PDF compilado.

## Como compilar

No diretorio do projeto:

```powershell
.\docs\paper_ic_transformers\build_paper.ps1
```

O PDF tambem e copiado para a pasta padrao de saida:

```text
output\pdf\paper_ic_transformers.pdf
```

## Estado cientifico usado no paper

O artigo registra os resultados consolidados ate 15/09/2026:

- 18 comparacoes conceituais aprovadas entre implementacao manual, NumPy,
  PyTorch, PyTorch SDPA e Keras/TensorFlow.
- benchmark sintetico v3 com 2.340 registros e 117.000 amostras de latencia.
- benchmark fisico v3 com 1.650 registros e 82.500 amostras de latencia.
- kernels INT8 dinamico e 2:4 confirmados por profiler.
- avaliacao OPT v1 com 45 registros de qualidade, 1.296 registros de desempenho
  e 10.080 timings; decode registrado como limitacao tecnica.
