# Protocolo experimental antes dos benchmarks

Este protocolo define os portoes que uma tecnica precisa atravessar antes de
ser usada em comparacoes de desempenho. A intencao e evitar que o trabalho
misture tres coisas diferentes: definicao cientifica, validade numerica e ganho
real de hardware.

## Niveis de certeza

| Nivel | Pergunta respondida | Evidencia minima |
| --- | --- | --- |
| Definicao | O termo esta sendo usado corretamente? | Fonte cientifica ou primaria e definicao operacional. |
| Matematica | A formula local bate com a definicao? | Caso pequeno calculado manualmente e teste deterministico. |
| Algoritmo | A ferramenta implementa o metodo esperado? | Comparacao contra implementacao transparente ou API de referencia. |
| Hardware | A execucao pode sustentar ganho fisico? | Dtype/formato/kernel/medidor mostrando menor memoria, latencia ou maior throughput. |

Uma tecnica pode passar nos tres primeiros niveis e ainda continuar pendente em
hardware. Isso nao e falha; e uma conclusao experimental importante.

## Portoes antes de benchmark

1. O conceito precisa aparecer na matriz de validacao.
2. A definicao precisa apontar para fonte cientifica ou documentacao primaria.
3. Deve existir teste pequeno e deterministico para o comportamento essencial.
4. A classificacao deve dizer se o resultado e conceitual, numerico,
   algoritmico ou hardware.
5. O nucleo da attention deve bater com uma referencia matematica independente
   e com APIs publicas de frameworks consolidados.
6. Benchmark amplo so pode ser executado depois dos testes conceituais e da
   comparacao entre frameworks passarem.

## Validacao cruzada entre frameworks

Antes dos benchmarks, `validacao.comparacao_frameworks` executa os mesmos
tensores `Q`, `K` e `V` em quatro caminhos:

- formula transparente em NumPy, usada como referencia matematica;
- formula transparente em PyTorch;
- `torch.nn.functional.scaled_dot_product_attention`;
- `keras.ops.nn.dot_product_attention` com backend TensorFlow.

A comparacao usa `float32`, CPU, dropout desativado, flash attention desativada,
seed `2026`, `atol=1e-6` e `rtol=1e-5`. Os casos cobrem valor manual pequeno,
multi-head com entradas pseudoaleatorias compartilhadas e mascara aditiva. A
entrada e gerada uma unica vez pelo NumPy para evitar divergencia entre os
geradores aleatorios dos frameworks.

O CSV desta etapa tem schema proprio e nao altera o schema dos benchmarks de
desempenho. A evidencia canônica fica em
`evidencias/comparacao_frameworks/` e so e aprovada se shape, dtype, finitude e
tolerancias passarem em todas as comparacoes.

## Recorte experimental

O nucleo obrigatorio sera um bloco Transformer pequeno e controlado, com:

- entrada `[batch, seq_len, d_model]`;
- projecoes densas para Q, K e V;
- atencao multi-head baseada em `softmax(QK^T / sqrt(d_k))V`;
- informacao posicional sinusoidal;
- residual, normalizacao e feed-forward;
- saida comparada contra baseline para medir erro.

Esse nucleo deve ser chamado de "bloco Transformer simplificado". O termo
"Transformer" fica reservado para uma arquitetura completa ou para uma
implementacao que tenha justificativa formal para todos os componentes.

## Cenarios experimentais

| Cenario | Descricao | Validade inicial |
| --- | --- | --- |
| `baseline` | Bloco em `float32` ou dtype definido como referencia. | Algoritmica |
| `pruning_magnitude` | Poda por magnitude em pesos lineares. | Conceitual/algoritmica; hardware pendente |
| `quantization_int8` | Quantizacao int8 validada por dtype, bytes e erro. | Numerica/algoritmica; hardware pendente ate validar kernel |
| `pruning_plus_quantization` | Combinacao das duas tecnicas, apenas se ambas passarem nos testes isolados. | Pendente |

## Metricas

As medicoes de qualidade e custo devem ser separadas.

Metricas de qualidade:

- MSE;
- MAE;
- R2;
- similaridade de cosseno.

Metricas de custo:

- latencia media, p50 e p95;
- throughput em tokens/s;
- pico de memoria;
- FLOPs teoricos.

Em GPU, o benchmark deve usar warmup, sincronizacao antes/depois da medicao e
registro explicito do dispositivo.

## Schema CSV obrigatorio

Todo resultado experimental deve ser salvo com estas colunas, nesta ordem:

```text
torch_version,cuda_version,gpu_name,device,model_kind,scenario,batch_size,seq_len,d_model,num_heads,dtype,pruning_sparsity,quantization_method,validity_level,latency_ms_mean,latency_ms_p50,latency_ms_p95,throughput_tokens_s,max_memory_bytes,theoretical_flops,mse,mae,r2,cosine_similarity
```

O modulo `validacao.protocolo` valida esse schema antes de qualquer resultado
ser aceito como dado experimental do trabalho.

## Runner controlado

O benchmark validado deve ser executado pelo modulo
`validacao.benchmark_controlado`. Ele:

- rejeita cenarios que nao estao registrados no catalogo de validacao;
- evita APIs depreciadas no pipeline automatizado;
- fixa seed e registra ambiente em arquivo de metadados;
- usa o bloco Transformer simplificado como nucleo experimental;
- mede latencia com warmup, repeticoes e sincronizacao CUDA quando aplicavel;
- calcula erro contra baseline e FLOPs teoricos;
- grava CSV apenas se todas as colunas obrigatorias forem validas.

Este runner mede o bloco simplificado e deve ser tratado como piloto de
infraestrutura. Em CPU, `max_memory_bytes` e uma estimativa do armazenamento do
modelo; apenas em CUDA o campo usa `torch.cuda.max_memory_allocated`. A
quantizacao manual dequantiza pesos para `float32`, portanto nao alega
armazenamento nem kernel INT8. O benchmark principal das operacoes isoladas e
definido separadamente no checklist pre-benchmark.

## Runner principal das operacoes isoladas

`validacao.benchmark_operacoes` implementa o recorte reconciliado entre o plano
formal e a reuniao:

- `dense_projection`: operacao mantida porque aparece explicitamente no plano;
- `self_attention`: uma cabeca com projecoes Q/K/V/saida e SDPA, usada como foco
  principal sugerido pelo orientador;
- o bloco Transformer simplificado fica fora do resultado principal.

Os dados sao sinteticos pseudoaleatorios com distribuicao normal, pois o estudo
mede operacoes e fidelidade de saida, nao uma tarefa supervisionada ou um
dataset. Mesma seed, shapes, entradas e pesos-base sao usados em todos os
cenarios. Pruning e quantizacao afetam apenas as matrizes de pesos.

A configuracao `experimentos/benchmark_principal_gpu.json` codifica seed `42`,
lote `1`, `L={64,128,256}`, `D={128,256,512}`, 20 iteracoes de aquecimento, 50
medicoes e duas execucoes independentes. Cada execucao preserva os mesmos dados
e alterna deterministicamente a ordem dos cenarios. O p50 e p95 usam
`numpy.percentile` com interpolacao linear.

Cada lote independente grava CSV e JSON de metadados imediatamente. O diretorio
tambem recebe snapshot da configuracao, ambiente, log e manifesto com SHA-256.
O runner recusa diretorio nao vazio, `device=auto`, campos nao finitos e
benchmark principal sem CUDA/GPU nominal. Assim, uma interrupcao posterior nao
apaga o lote ja concluido.

`validacao.analise_benchmark_operacoes` valida os checksums do manifesto antes
de ler os dados. Cada candidato e comparado somente com o baseline da mesma
configuracao e execucao independente. Depois, as execucoes sao consolidadas com
media e desvio-padrao, preservando tambem os resultados individuais. Hashes de
entrada e saida precisam coincidir entre execucoes; latencias podem variar.

Comando padrao:

```powershell
.\.venv\Scripts\python.exe -m validacao.benchmark_controlado --output resultados\benchmark_controlado.csv
```

Para uma execucao curta de verificacao:

```powershell
.\.venv\Scripts\python.exe -m validacao.benchmark_controlado --output resultados\benchmark_teste.csv --warmup 1 --repetitions 3
```

## Analise dos resultados

A analise final deve usar `validacao.analise_resultados`. Ela valida o CSV,
compara cada cenario com o baseline, gera tabelas de resumo, gera graficos e
escreve conclusoes com origem explicita:

- evidencia no CSV;
- linha conceitual da matriz;
- nivel de validade observado.

Nenhuma conclusao deve ser escrita como ganho de hardware se o `validity_level`
do resultado nao for `hardware`.

## Validacao da NN como Transformer

A rede neural usada no nucleo experimental deve ser auditada por
`validacao.auditoria_transformer` antes dos benchmarks. O resultado aceito para
o recorte atual e `bloco_transformer_simplificado`.

Uma NN comum com camadas lineares nao pode passar nessa auditoria. Isso e um
controle negativo obrigatorio para garantir que o estudo nao esta chamando uma
rede neural generica de Transformer.

## Comparacao com literatura

Nesta fase, a comparacao com literatura deve ser feita assim:

- formulas e definicoes: comparacao direta com valores derivados;
- bibliotecas: comparacao contra implementacao transparente ou documentacao
  primaria;
- desempenho: comparacao apenas como tendencia ou ordem de grandeza, salvo se
  modelo, hardware, kernel, dtype, entrada e protocolo forem equivalentes.

## Criterio para conclusao de hardware

Uma tecnica so pode receber nivel "hardware" quando houver evidencia de que o
caminho executado usa a representacao otimizada:

- quantizacao: armazenamento menor e kernel/caminho compativel com o dtype;
- pruning: formato ou kernel que explore esparsidade;
- KV cache: reutilizacao de K/V entre passos e equivalencia numerica com a
  execucao sem cache;
- memoria: medicao real ou estimativa teorica claramente separada;
- latencia/throughput: medicao com repeticoes, warmup e sincronizacao.
