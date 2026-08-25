# Explicacao narrativa da metodologia de validacao

Este documento explica, em linguagem de apresentacao de projeto de pesquisa,
como estamos ligando os conceitos cientificos ao codigo. A ideia e construir
uma historia: primeiro entendemos o que cada conceito significa na literatura;
entao transformamos essa definicao em uma propriedade observavel; por conta
disso criamos uma implementacao pequena; e, por fim, testamos se o resultado
bate com o comportamento esperado.

O ponto central e simples: antes de medir desempenho, precisamos saber se aquilo
que estamos medindo e realmente o que dizemos que e. Se chamamos uma operacao
de Transformer, ela precisa ter propriedades de Transformer. Se chamamos uma
tecnica de quantizacao, ela precisa representar valores com menos precisao. Se
chamamos pruning de ganho de hardware, ele precisa usar algum formato ou kernel
que realmente explore esparsidade.

Entao, a metodologia do projeto segue esta cadeia:

```text
conceito cientifico
-> referencia confiavel
-> definicao operacional
-> implementacao pequena e transparente
-> teste deterministico
-> classificacao do nivel de validade
```

Os quatro niveis de validade usados no projeto sao:

| Nivel | O que responde | Exemplo |
| --- | --- | --- |
| Definicao | O termo esta correto de acordo com a literatura? | Transformer como arquitetura baseada em atencao. |
| Matematica | A formula local bate com um caso pequeno calculado? | `softmax(QK^T / sqrt(d_k))V`. |
| Algoritmo | A ferramenta implementa o comportamento esperado? | PyTorch SDPA bate com nossa attention manual. |
| Hardware | Ha evidencia de ganho fisico real? | Kernel, dtype ou formato que reduza memoria/latencia. |

Por conta disso, varias conclusoes do projeto ainda sao propositalmente
cautelosas. Por exemplo, uma quantizacao pode estar numericamente correta, mas
isso ainda nao quer dizer que ela gerou ganho real de hardware. Do mesmo modo,
uma poda pode zerar pesos corretamente, mas se a execucao continuar densa, a
conclusao e conceitual/algoritmica, nao de hardware.

## Capitulo 1 - Rede neural e deep learning

### Ideia inicial

Comecamos por rede neural porque Transformer nao aparece do nada. Um
Transformer e um tipo especifico de rede neural. Entao, antes de perguntar se
uma NN e Transformer, precisamos perguntar se ela pelo menos tem as
caracteristicas basicas de uma rede neural: parametros, entrada, transformacoes
e saida.

Deep learning entra logo depois porque, em geral, estamos falando de redes com
varias etapas de transformacao. Cada camada transforma a representacao anterior
em outra representacao. Entao, quando falamos de Transformer, estamos falando
de uma arquitetura de deep learning que processa sequencias.

### Definicao cientifica

A referencia principal para rede neural e deep learning e:

- Goodfellow, Bengio e Courville, *Deep Learning*, MIT Press, 2016:
  <https://www.deeplearningbook.org/>.

No nosso uso, uma rede neural e um modelo parametrizado que aplica operacoes
lineares e nao lineares sobre uma entrada para produzir uma saida. Isso importa
porque o nosso bloco usa pesos treinaveis, como `torch.nn.Linear`, e operacoes
nao lineares, como `ReLU`.

### Por que isso importa no nosso trabalho

Se a gente nao separa "rede neural" de "Transformer", corremos o risco de
chamar qualquer NN de Transformer. Entao, primeiro aceitamos que o bloco atual e
uma rede neural. Depois, criamos testes adicionais para saber se ela pertence a
familia Transformer.

### Como implementamos

O nucleo atual esta em `validacao.transformer.SimplifiedTransformerBlock`.
Dentro dele ha camadas lineares, normalizacao e FFN:

```python
self.q_proj = torch.nn.Linear(d_model, d_model)
self.k_proj = torch.nn.Linear(d_model, d_model)
self.v_proj = torch.nn.Linear(d_model, d_model)
self.out_proj = torch.nn.Linear(d_model, d_model)
self.norm1 = torch.nn.LayerNorm(d_model)
self.ffn = torch.nn.Sequential(
    torch.nn.Linear(d_model, ffn_hidden_dim),
    torch.nn.ReLU(),
    torch.nn.Linear(ffn_hidden_dim, d_model),
)
self.norm2 = torch.nn.LayerNorm(d_model)
```

### Como testamos

Testamos se a rede executa com entrada sequencial e preserva shape:

```python
block = SimplifiedTransformerBlock(d_model=4, num_heads=2, ffn_hidden_dim=8).eval()
sequence = torch.randn(2, 3, 4)
output = block(sequence)

assert output.shape == sequence.shape
```

### Com o que comparamos

Aqui a comparacao ainda nao e com um paper de resultado. A comparacao e com a
definicao operacional de uma rede neural: existe entrada, existem parametros,
existe composicao de camadas e existe saida.

### O que podemos concluir

Podemos concluir que o objeto atual e uma rede neural profunda pequena e
controlada.

### O que ainda nao podemos concluir

Ainda nao podemos concluir que ela e Transformer so por ser uma NN. Por conta
disso, vem o proximo passo: validar os criterios especificos de Transformer.

## Capitulo 2 - Transformer

### Ideia inicial

Um Transformer processa sequencias usando atencao. A ideia intuitiva e: cada
posicao da sequencia pode consultar outras posicoes e decidir quais informacoes
devem influenciar sua representacao.

Entao, se o nosso estudo fala em eficiencia de operacoes de Transformers, a NN
usada precisa ter pelo menos as pecas centrais dessa familia: entrada
sequencial, Q/K/V, attention, dependencia entre posicoes, informacao
posicional, residual, normalizacao e feed-forward.

### Definicao cientifica

A referencia principal e:

- Vaswani et al., *Attention Is All You Need*, 2017:
  <https://arxiv.org/abs/1706.03762>.

Esse trabalho define a arquitetura Transformer como uma arquitetura baseada em
mecanismos de atencao. A formula central da atencao escalonada usada no paper e:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

### Por que isso importa no nosso trabalho

Isso importa porque o nome "Transformer" carrega uma afirmacao cientifica. Se
usarmos uma NN comum e chamarmos de Transformer, o estudo perde validade. Entao,
em vez de confiar no nome da classe, criamos uma auditoria arquitetural.

### Como implementamos

A classe atual e um bloco pequeno:

```python
class SimplifiedTransformerBlock(torch.nn.Module):
    """Bloco pequeno para validacao: MHA + residual/norm + FFN."""
```

Ela nao se declara como Transformer completo. Ela se declara como bloco
simplificado. Isso e importante porque a implementacao nao tem uma pilha
completa de camadas, cabeca de tarefa e treinamento final.

### Como testamos

Usamos `validacao.auditoria_transformer.audit_transformer_module`:

```python
report = audit_transformer_module(block, d_model=4, seq_len=3)

assert report.architecture_label == "bloco_transformer_simplificado"
assert report.is_transformer_family
assert not report.can_be_called_transformer
assert report.can_be_called_simplified_block
```

### Com o que comparamos

Comparamos com uma lista de criterios derivados do Transformer original:

- entrada sequencial;
- projecoes Q/K/V;
- atencao escalonada;
- dependencia entre posicoes;
- informacao posicional;
- residual, normalizacao e FFN;
- saida deterministica em inferencia.

Tambem comparamos contra um controle negativo:

```python
plain_nn = torch.nn.Sequential(
    torch.nn.Linear(4, 4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 4),
).eval()

report = audit_transformer_module(plain_nn, d_model=4, seq_len=3)

assert report.architecture_label == "nao_aderente"
```

### O que podemos concluir

Podemos concluir que a NN atual pertence a familia Transformer e deve ser
chamada de:

```text
bloco Transformer simplificado
```

### O que ainda nao podemos concluir

Nao podemos chamar a implementacao atual de Transformer completo. Entao, no
texto do artigo, a formulacao correta e: "avaliamos um bloco Transformer
simplificado controlado".

### Ponte direta entre Vaswani e os nossos scripts

Entao, para nao ficar uma validacao apenas verbal, criamos uma ponte de
rastreabilidade. A logica e simples: primeiro olhamos para o artigo de Vaswani,
depois extraimos uma propriedade que pode aparecer no codigo, depois apontamos o
script Python que implementa essa propriedade, e por fim apontamos o teste que
confere se o comportamento bate com o esperado.

Essa ponte ficou documentada de forma completa em
`fases/01_validacao_conceitual/docs/validacao_transformer.md`. A ideia central dela e:

```text
Vaswani et al. (2017)
-> propriedade esperada
-> script Python
-> teste deterministico
-> limite da conclusao
```

Por exemplo, Vaswani define a attention escalonada como:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

Entao, por conta disso, o nosso script `fases/01_validacao_conceitual/validacao/attention.py` precisa fazer
essa conta explicitamente:

```python
scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)
weights = torch.softmax(scores, dim=-1)
return torch.matmul(weights, value)
```

Mas ainda nao basta o codigo "parecer" correto. Por conta disso, o teste
`test_scaled_dot_product_attention_matches_independent_numpy_reference` compara
a saida com um caso pequeno calculado de forma independente, e o teste
`test_scaled_dot_product_attention_matches_pytorch_reference` compara com
`torch.nn.functional.scaled_dot_product_attention`.

Entao, para esse trecho especifico, podemos dizer:

```text
A nossa scaled dot-product attention bate com a formula de Vaswani e com uma
implementacao de referencia do PyTorch.
```

Agora, isso ainda nao autoriza dizer que temos uma Transformer completa. Isso
autoriza dizer que temos o trecho de attention validado. Por isso fazemos a
mesma coisa para as outras pecas:

| Trecho do artigo | O que exigimos no codigo | Como testamos |
| --- | --- | --- |
| Entrada sequencial | Tensor `[batch, seq_len, d_model]`. | O bloco recebe a sequencia e preserva o shape. |
| Projecoes lineares | Operacao `Y = XW^T + b`. | Comparacao com valor manual conhecido. |
| Q, K e V | Tres projecoes separadas da sequencia. | Auditoria verifica `q_proj`, `k_proj` e `v_proj`. |
| Scaled attention | `softmax(QK^T / sqrt(d_k))V`. | Comparacao com NumPy, valor manual e PyTorch. |
| Multi-head | Separar heads, aplicar attention e recombinar. | Comparacao com SDPA aplicada por head. |
| Dependencia sequencial | Mudar um token pode mudar outra posicao. | Teste altera uma posicao e observa outra. |
| Posicao | Algum sinal posicional antes da attention. | Auditoria verifica positional encoding. |
| Residual/norm/FFN | Componentes de bloco Transformer. | Auditoria estrutural do bloco. |
| Controle negativo | NN comum deve ser rejeitada. | `Linear + ReLU + Linear` recebe `nao_aderente`. |

Essa tabela e importante para a apresentacao porque ela mostra que nao estamos
dizendo "e Transformer porque eu chamei de Transformer". Estamos dizendo:

```text
Este trecho pertence a familia Transformer porque a propriedade X aparece no
artigo, aparece no codigo, e passa no teste Y.
```

Com isso, a conclusao fica calibrada. Podemos afirmar que os scripts atuais
validam trechos centrais de Transformer e um bloco Transformer simplificado.
Ainda nao podemos afirmar que reproduzimos uma Transformer completa, porque a
arquitetura completa exigiria pilha de blocos, encoder/decoder ou pelo menos um
encoder completo, embeddings, cabeca de tarefa e validacao em tarefa real.

## Capitulo 3 - Bloco Transformer simplificado

### Ideia inicial

Depois de entender Transformer, a pergunta vira: qual parte do Transformer
vamos estudar? Como o projeto ainda esta em fase controlada, usamos um bloco.
Isso permite entender causa e efeito sem misturar muitas variaveis.

Entao, em vez de medir um modelo enorme logo de inicio, estudamos um bloco
pequeno onde conseguimos auditar cada parte.

### Definicao cientifica

O bloco se inspira no Transformer de Vaswani et al. Ele tem attention,
projecoes, residual, normalizacao e FFN. Como e uma simplificacao, ele nao e
uma arquitetura completa.

### Por que isso importa no nosso trabalho

Isso importa porque a validade cientifica vem do controle. Se usarmos um modelo
grande sem entender o caminho interno, talvez vejamos uma mudanca de latencia
sem saber se veio da attention, da quantizacao, do kernel, da memoria ou de
alguma otimizacao escondida.

### Como implementamos

O forward do bloco mostra a historia inteira:

```python
hidden = self.add_positional_information(sequence)
query, key, value = self.project_qkv(hidden)
attention = self.attention_core(query, key, value)
attention = combine_heads(attention)

hidden = self.norm1(hidden + self.out_proj(attention))
return self.norm2(hidden + self.ffn(hidden))
```

Entao, o fluxo e:

```text
sequencia
-> informacao posicional
-> Q/K/V
-> attention
-> recombinacao das heads
-> residual + norm
-> FFN
-> residual + norm
```

### Como testamos

O teste confere componentes e shape:

```python
assert output.shape == sequence.shape
assert hasattr(block, "q_proj")
assert hasattr(block, "k_proj")
assert hasattr(block, "v_proj")
assert hasattr(block, "norm1")
assert hasattr(block, "ffn")
```

E a auditoria confere o comportamento:

```python
assert report.has_qkv_projection
assert report.has_scaled_attention
assert report.has_cross_position_dependency
assert report.has_positional_information_or_justification
assert report.has_residual_norm_ffn_or_justification
```

### Com o que comparamos

Comparamos com os criterios arquiteturais do Transformer e com a API de
referencia do PyTorch para a attention interna.

### O que podemos concluir

Podemos concluir que o bloco e adequado para o recorte experimental.

### O que ainda nao podemos concluir

Ainda nao podemos concluir desempenho final de um Transformer real em producao,
porque isso exigiria arquitetura completa, tarefa, dataset e benchmark maior.

## Capitulo 4 - Self-attention

### Ideia inicial

Self-attention e a ideia de uma sequencia olhar para ela mesma. Entao, cada
posicao calcula o quanto deve considerar outras posicoes.

Isso importa porque, sem dependencia entre posicoes, uma operacao nao esta
capturando o aspecto sequencial que torna Transformers interessantes.

### Definicao cientifica

A referencia e Vaswani et al. (2017). No nosso recorte, self-attention significa
que Q, K e V sao derivados da mesma sequencia.

### Por que isso importa no nosso trabalho

Se mudamos um token e nenhuma outra posicao muda, entao a operacao nao esta
misturando informacao da sequencia. Por conta disso, testamos explicitamente se
alterar uma posicao pode alterar outra.

### Como implementamos

A attention central usa Q, K e V:

```python
scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)
weights = torch.softmax(scores, dim=-1)
return torch.matmul(weights, value)
```

### Como testamos

O teste cria duas versoes de `value`, alterando a segunda posicao:

```python
query = torch.zeros(1, 2, 2)
key = torch.zeros(1, 2, 2)
value_a = torch.tensor([[[1.0, 1.0], [3.0, 3.0]]])
value_b = torch.tensor([[[1.0, 1.0], [9.0, 9.0]]])

output_a = scaled_dot_product_attention_torch(query, key, value_a)
output_b = scaled_dot_product_attention_torch(query, key, value_b)

assert not torch.allclose(output_a[:, 0, :], output_b[:, 0, :])
```

### Com o que comparamos

Comparamos com a propriedade esperada de self-attention: uma posicao pode
depender de outra posicao da mesma sequencia.

### O que podemos concluir

Podemos concluir que a operacao permite dependencia entre posicoes.

### O que ainda nao podemos concluir

Isso nao prova qualidade de modelagem, nem desempenho, nem acuracia em tarefa.
Prova apenas a propriedade conceitual necessaria.

## Capitulo 5 - Scaled dot-product attention

### Ideia inicial

Scaled dot-product attention e a conta central. Ela calcula similaridade entre
queries e keys, normaliza essas similaridades e usa o resultado para combinar
values.

Entao, esse e um dos testes mais importantes: se essa formula estiver errada, o
resto do estudo perde base.

### Definicao cientifica

A formula vem de Vaswani et al.:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

### Por que isso importa no nosso trabalho

Essa formula e o nucleo matematico da attention usada no bloco. Por conta
disso, nao basta confiar no PyTorch. Implementamos uma versao manual e
comparamos com uma referencia independente.

### Como implementamos

Em PyTorch:

```python
d_k = query.shape[-1]
scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(d_k)
weights = torch.softmax(scores, dim=-1)
return torch.matmul(weights, value)
```

Em NumPy, a mesma ideia:

```python
scores = np.matmul(query, np.swapaxes(key, -1, -2)) / math.sqrt(d_k)
weights = softmax_numpy(scores, axis=-1)
return np.matmul(weights, value)
```

### Como testamos

Usamos tensores pequenos:

```python
query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
value = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
```

E exigimos o resultado esperado pela formula:

```python
expected_from_formula = torch.tensor(
    [[[1.6604769, 2.6604769], [2.3395231, 3.3395231]]]
)

assert torch.allclose(actual, expected_from_formula, atol=1e-6)
```

### Com o que comparamos

Comparamos com:

1. calculo NumPy independente;
2. resultado numerico derivado da formula;
3. `torch.nn.functional.scaled_dot_product_attention`;
4. `keras.ops.nn.dot_product_attention` com backend TensorFlow, usando os mesmos
   `Q`, `K`, `V` e mascara.

### O que podemos concluir

Podemos concluir que a attention manual esta matematicamente aderente ao
conceito e numericamente compativel com APIs publicas de dois frameworks. A
execucao canônica aprovou 9/9 comparacoes, com erro absoluto maximo de
`2.38418579e-07` para `atol=1e-6` e `rtol=1e-5`.

### O que ainda nao podemos concluir

Attention correta nao significa Transformer completo. Ela e uma parte
necessaria, mas nao suficiente.

## Capitulo 6 - Multi-head attention

### Ideia inicial

Multi-head attention divide a representacao em varias heads. Entao, em vez de
uma unica attention olhar para tudo de uma forma, varias attentions menores
processam partes da representacao.

### Definicao cientifica

A referencia continua sendo Vaswani et al. (2017). A arquitetura Transformer
usa varias heads para projetar, aplicar attention e recombinar informacoes.

### Por que isso importa no nosso trabalho

Se o bloco usa multi-head attention, precisamos saber se o split e o combine
estao corretos. Por conta disso, validamos shapes e comparamos com a referencia
do PyTorch.

### Como implementamos

O split:

```python
return tensor.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
```

O combine:

```python
return tensor.transpose(1, 2).contiguous().reshape(
    batch_size,
    seq_len,
    num_heads * head_dim,
)
```

E a attention multi-head transparente:

```python
query_heads = split_heads(query, num_heads)
key_heads = split_heads(key, num_heads)
value_heads = split_heads(value, num_heads)
attention_heads = scaled_dot_product_attention_torch(query_heads, key_heads, value_heads)
return combine_heads(attention_heads)
```

### Como testamos

Testamos shape e equivalencia:

```python
sequence = torch.randn(2, 3, 4)
num_heads = 2

actual = multi_head_attention_from_qkv(
    sequence,
    sequence,
    sequence,
    num_heads=num_heads,
)

assert actual.shape == sequence.shape
```

### Com o que comparamos

Comparamos com `torch.nn.functional.scaled_dot_product_attention` aplicada nas
heads ja separadas.

### O que podemos concluir

Podemos concluir que a mecanica multi-head esta correta no caso pequeno.

### O que ainda nao podemos concluir

Nao podemos concluir que o numero de heads escolhido e otimo. Isso seria uma
pergunta experimental diferente.

## Capitulo 7 - Projecao densa

### Ideia inicial

Projecao densa e uma camada linear. Ela pega um vetor e transforma em outro por
meio de uma matriz de pesos. Entao, antes de Q, K e V existirem, precisamos
projetar a entrada.

### Definicao cientifica

A operacao e:

```text
Y = XW^T + b
```

Ela aparece em redes neurais em Goodfellow et al. e no Transformer de Vaswani
et al., onde projecoes aprendidas geram Q, K, V e saidas de attention.

### Por que isso importa no nosso trabalho

Se a projecao densa estiver errada, Q/K/V tambem ficam errados. Entao, validamos
essa operacao separadamente antes de confiar no bloco.

### Como implementamos

```python
def dense_projection_torch(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight.transpose(-1, -2))
    if bias is not None:
        output = output + bias
    return output
```

### Como testamos

Usamos um exemplo pequeno:

```python
inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
weight = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
bias = torch.tensor([0.5, -0.5])

expected = torch.tensor([[[1.5, 3.5], [3.5, 7.5]]])

assert torch.allclose(actual, expected)
```

### Com o que comparamos

Comparamos com o resultado manual e com uma implementacao NumPy independente.

### O que podemos concluir

Podemos concluir que a projecao densa do projeto segue a formula linear
esperada.

### O que ainda nao podemos concluir

Projecao densa sozinha nao e Transformer. Ela e uma operacao auxiliar usada
dentro do bloco.

## Capitulo 8 - Informacao posicional

### Ideia inicial

Attention pura enxerga relacoes entre vetores, mas nao sabe automaticamente a
ordem dos tokens. Entao, Transformers usam alguma forma de informacao
posicional para indicar que a posicao 1 vem antes da posicao 2.

### Definicao cientifica

Vaswani et al. usam positional encoding no Transformer original. No nosso
recorte, usamos encoding sinusoidal, que adiciona sinais dependentes da posicao
aos vetores da sequencia.

### Por que isso importa no nosso trabalho

Se removemos informacao posicional sem justificativa, o bloco fica incompleto
para ser chamado de bloco Transformer. Por conta disso, a auditoria verifica
explicitamente se existe informacao posicional ou justificativa formal.

### Como implementamos

```python
position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
div_term = torch.exp(
    torch.arange(0, d_model, 2, device=device, dtype=dtype)
    * (-math.log(10000.0) / d_model)
)
encoding[:, 0::2] = torch.sin(position * div_term)
encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
```

E no bloco:

```python
hidden = self.add_positional_information(sequence)
```

### Como testamos

A auditoria verifica:

```python
assert report.has_positional_information_or_justification
```

### Com o que comparamos

Comparamos com o criterio arquitetural do Transformer: a sequencia precisa ter
alguma informacao de posicao, ou a ausencia precisa ser justificada.

### O que podemos concluir

Podemos concluir que o bloco atual tem informacao posicional.

### O que ainda nao podemos concluir

Nao podemos concluir que esse encoding e o melhor para qualquer tarefa. Ele e
adequado para validar o conceito.

## Capitulo 9 - Inferencia

### Ideia inicial

Inferencia e usar o modelo para produzir saidas, sem atualizar pesos. Entao,
quando medimos custo computacional, queremos medir execucao, nao treinamento.

### Definicao cientifica

Goodfellow et al. separam o uso de modelos para predicao da fase de treinamento.
No PyTorch, operacionalmente usamos `eval()` e `torch.inference_mode()`.

### Por que isso importa no nosso trabalho

Se os pesos mudam durante o experimento, a comparacao com baseline deixa de ser
justa. Entao, fixamos seed, colocamos modelo em `eval()` e usamos inferencia.

### Como implementamos

Na auditoria:

```python
module.eval()
with torch.inference_mode():
    return module(sequence)
```

No benchmark controlado:

```python
with torch.inference_mode():
    baseline_output = baseline_model(inputs).detach()
```

### Como testamos

A auditoria roda a mesma entrada duas vezes:

```python
first_output = _safe_forward(module, sequence)
second_output = _safe_forward(module, sequence)

deterministic_in_eval = torch.allclose(first_output, second_output, atol=atol)
```

### Com o que comparamos

Comparamos uma saida contra a outra, para verificar determinismo.

### O que podemos concluir

Podemos concluir que o bloco se comporta de forma deterministica em modo de
avaliacao.

### O que ainda nao podemos concluir

Isso nao garante reproducibilidade universal em qualquer hardware e qualquer
kernel. Por conta disso, o benchmark registra ambiente, seed e dispositivo.

## Capitulo 10 - KV cache

### Ideia inicial

KV cache e uma economia de recomputacao. Em geracao autoregressiva, quando o
modelo gera token por token, as chaves e valores dos tokens antigos nao precisam
ser recalculados toda vez.

Entao, o cache guarda K e V passados. A cada novo token, calculamos K/V apenas
do token novo e juntamos com o cache.

### Definicao cientifica

O conceito aparece em inferencia de Transformers autoregressivos e em trabalhos
de compressao de KV cache. Usamos como referencia moderna TurboQuant:

- Zandieh et al., *TurboQuant*, 2025:
  <https://arxiv.org/abs/2504.19874>.
- Google Research, *TurboQuant: Redefining AI efficiency with extreme
  compression*:
  <https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/>.

### Por que isso importa no nosso trabalho

KV cache e importante para memoria e latencia em inferencia. Mas, antes de medir
latencia, precisamos saber se o cache preserva a mesma saida.

### Como implementamos

O caminho com cache concatena K/V antigos com K/V do token novo:

```python
key_cache = key if key_cache is None else torch.cat([key_cache, key], dim=1)
value_cache = value if value_cache is None else torch.cat([value_cache, value], dim=1)
outputs.append(scaled_dot_product_attention_torch(query, key_cache, value_cache))
```

### Como testamos

Comparamos atencao causal completa contra atencao incremental com cache:

```python
full_output = causal_self_attention_full(sequence, identity, identity, identity)
cached = causal_self_attention_with_cache(sequence, identity, identity, identity)

assert torch.allclose(cached.output, full_output, atol=1e-6)
assert cached.cached_key_value_tokens == 4
assert cached.naive_key_value_tokens == 10
assert cached.reused_key_value_tokens == 6
```

### Com o que comparamos

Comparamos a saida com e sem cache. Se as saidas batem, o cache preserva o
comportamento. Tambem comparamos a quantidade de K/V calculados: 4 contra 10 no
caso pequeno.

### O que podemos concluir

Podemos concluir equivalencia algoritmica e reducao de recomputacao no exemplo.

### O que ainda nao podemos concluir

Ainda nao podemos concluir ganho de hardware. Para isso, precisamos medir
memoria, latencia e throughput em execucao controlada.

## Capitulo 11 - Pruning e sparsity

### Ideia inicial

Pruning e remover ou zerar partes do modelo. A intencao e criar esparsidade:
muitos zeros ou conexoes removidas. Entao, teoricamente, haveria menos trabalho
a fazer.

Mas existe um detalhe importante: zerar pesos nao significa necessariamente que
o hardware vai pular esses pesos. Se a multiplicacao continuar densa, a conta
ainda pode custar quase a mesma coisa.

### Definicao cientifica

Usamos:

- Tang et al., *A Survey on Transformer Compression*, 2024:
  <https://arxiv.org/abs/2402.05964>.
- PyTorch pruning tutorial:
  <https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html>.

### Por que isso importa no nosso trabalho

Como vamos comparar pruning com quantizacao, precisamos classificar pruning sem
exagero. Entao, se o pruning apenas cria mascara, classificamos como
conceitual/algoritmico.

### Como implementamos

A observacao de sparsity conta zeros:

```python
total = int(tensor.numel())
zeros = int((tensor == 0).sum().item())
ratio = 0.0 if total == 0 else zeros / total
```

E a classificacao considera hardware apenas se houver armazenamento e kernel
esparsos:

```python
if self.uses_sparse_storage and self.uses_sparse_kernel:
    return "hardware"
return "conceitual_algoritmico"
```

### Como testamos

Usamos uma camada com 8 pesos e aplicamos pruning de 50%:

```python
prune.l1_unstructured(layer, name="weight", amount=0.5)

observation = observe_sparsity(
    layer.weight,
    uses_sparse_storage=False,
    uses_sparse_kernel=False,
)

assert observation.zero_elements == 4
assert observation.sparsity_ratio == pytest.approx(0.5)
assert observation.validity_level == "conceitual_algoritmico"
assert hasattr(layer, "weight_mask")
```

### Com o que comparamos

Comparamos com a expectativa operacional: 50% de 8 pesos = 4 zeros. Tambem
verificamos a presenca de `weight_mask`, coerente com o mecanismo do PyTorch.

### O que podemos concluir

Podemos concluir que o pruning cria esparsidade conceitual/algoritmica.

### O que ainda nao podemos concluir

Nao podemos dizer que houve ganho real de hardware, porque o teste nao usa
storage esparso nem kernel esparso.

## Capitulo 12 - Quantizacao

### Ideia inicial

Quantizacao e representar numeros com menos precisao. Entao, em vez de guardar
tudo em `float32`, podemos representar pesos em `int8`, por exemplo.

Isso economiza memoria representacional, mas introduz erro numerico. Por conta
disso, precisamos medir duas coisas: se a representacao realmente ficou menor e
se o erro ficou controlado.

### Definicao cientifica

Usamos:

- Tang et al. (2024), sobre compressao de Transformers.
- Zandieh et al. (2025), TurboQuant.
- Documentacao `torchao`:
  <https://docs.pytorch.org/ao/stable/contributing/quantization_overview.html>.

### Por que isso importa no nosso trabalho

Quantizacao e uma das tecnicas centrais do estudo. Mas se quantizamos e logo
voltamos para `float32`, isso e apenas simulacao numerica. Entao, separamos
validade numerica de validade de hardware.

### Como implementamos

A quantizacao manual simetrica:

```python
max_abs = tensor.abs().max()
scale = max_abs / 127.0
quantized = torch.clamp(torch.round(tensor / scale), -127, 127).to(torch.int8)
dequantized = quantized.to(torch.float32) * scale
```

E com `torchao` v2:

```python
from torchao.quantization import Int8WeightOnlyConfig, quantize_

quantize_(model, Int8WeightOnlyConfig(version=2))
```

### Como testamos

Para a quantizacao manual:

```python
tensor = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
quantized, scale, dequantized = symmetric_int8_quantize(tensor)

assert quantized.dtype == torch.int8
assert q_observation.storage_bytes == 5
assert tensor.numel() * tensor.element_size() == 20
assert torch.max(torch.abs(tensor - dequantized)).item() <= scale.item()
```

Para `torchao`:

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    quantize_(model, Int8WeightOnlyConfig(version=2))

assert caught == []
assert type(model.weight).__name__ == "Int8Tensor"
assert model.weight.qdata.dtype == torch.int8
```

### Com o que comparamos

Comparamos:

- `float32`: 5 valores * 4 bytes = 20 bytes;
- `int8`: 5 valores * 1 byte = 5 bytes;
- erro maximo contra o limite dado pela escala;
- representacao interna `qdata` em `int8` no `torchao`.

### O que podemos concluir

Podemos concluir validade numerica e algoritmica da quantizacao no recorte.

### O que ainda nao podemos concluir

Ainda nao podemos dizer "ganho de hardware" sem demonstrar kernel/caminho de
execucao compativel.

## Capitulo 13 - Latencia, throughput, memoria e FLOPs

### Ideia inicial

Depois que os conceitos estao validados, queremos medir custo. Mas custo nao e
uma coisa so. Latencia e tempo por execucao. Throughput e quantidade processada
por tempo. Memoria e armazenamento usado. FLOPs e uma estimativa teorica de
operacoes.

Entao, se misturarmos tudo, a conclusao fica confusa. Por conta disso, medimos
cada dimensao separadamente.

### Definicao cientifica

Usamos Hennessy e Patterson, *Computer Architecture: A Quantitative Approach*,
como referencia geral para avaliacao quantitativa de desempenho, memoria,
latencia e throughput.

Tambem usamos Vaswani et al. para entender o custo de operacoes de attention em
sequencias.

### Por que isso importa no nosso trabalho

Uma tecnica pode reduzir memoria e nao reduzir latencia. Outra pode reduzir
FLOPs teoricos e nao acelerar no hardware real. Entao, cada resultado precisa
dizer exatamente qual metrica mudou.

### Como implementamos FLOPs

Para projecao densa:

```python
matmul_flops = 2 * batch_size * seq_len * in_features * out_features
bias_flops = batch_size * seq_len * out_features if bias else 0
return int(matmul_flops + bias_flops)
```

Para attention:

```python
score_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
weighted_value_flops = 2 * batch_size * num_heads * seq_len * seq_len * head_dim
return int(score_flops + weighted_value_flops)
```

### Como testamos FLOPs

```python
assert dense_projection_flops(
    batch_size=2,
    seq_len=3,
    in_features=4,
    out_features=5,
) == 270

assert attention_matmul_flops(
    batch_size=2,
    seq_len=3,
    d_model=4,
    num_heads=2,
) == 288
```

### Como implementamos benchmark controlado

O runner registra ambiente, seed, warmup, repeticoes e sincronizacao:

```python
metadata = {
    **env,
    "warmup": config.warmup,
    "repetitions": config.repetitions,
    "scenarios": list(config.scenarios),
    "used_cuda_synchronization": used_cuda_synchronization,
}
```

### Com o que comparamos

FLOPs sao comparados com formulas teoricas. Latencia, throughput e memoria
serao comparados contra baseline no CSV gerado pelo benchmark.

### O que podemos concluir

Podemos concluir que temos um protocolo de medicao pronto e validado por teste.

### O que ainda nao podemos concluir

Ainda nao podemos concluir desempenho final antes de rodar benchmarks amplos e
controlados.

## Capitulo 14 - MSE, MAE, R2 e similaridade de cosseno

### Ideia inicial

Quando aplicamos pruning ou quantizacao, a saida pode mudar. Entao, precisamos
medir quanto ela mudou em relacao ao baseline.

Por conta disso usamos quatro metricas: MSE, MAE, R2 e similaridade de cosseno.
Elas olham para erro e preservacao da direcao da representacao.

### Definicao cientifica

Usamos:

- Hastie, Tibshirani e Friedman, *The Elements of Statistical Learning*:
  <https://hastie.su.domains/ElemStatLearn/>.
- scikit-learn MSE:
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html>.
- scikit-learn MAE:
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html>.
- scikit-learn R2:
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html>.
- scikit-learn cosine similarity:
  <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html>.

### Por que isso importa no nosso trabalho

Se uma tecnica acelera mas destroi a saida, ela pode nao ser util. Entao,
desempenho precisa ser lido junto com fidelidade numerica.

### Como implementamos

MSE:

```python
return float(np.mean(np.square(reference_arr - candidate_arr)))
```

MAE:

```python
return float(np.mean(np.abs(reference_arr - candidate_arr)))
```

R2:

```python
return float(1.0 - (ss_res / ss_tot))
```

Cosseno:

```python
return float(np.dot(reference_arr, candidate_arr) / (reference_norm * candidate_norm))
```

### Como testamos

Para MSE, MAE e R2:

```python
reference = np.array([1.0, 2.0, 3.0])
candidate = np.array([1.0, 2.0, 4.0])

assert mean_squared_error(reference, candidate) == pytest.approx(1.0 / 3.0)
assert mean_absolute_error(reference, candidate) == pytest.approx(1.0 / 3.0)
assert r2_score(reference, candidate) == pytest.approx(0.5)
```

Para cosseno:

```python
reference = np.array([1.0, 1.0, 0.0])
candidate = np.array([1.0, 0.0, 0.0])

expected = 1.0 / math.sqrt(2.0)
assert cosine_similarity(reference, candidate) == pytest.approx(expected)
```

### Com o que comparamos

Comparamos com valores manuais conhecidos e com `scikit-learn`.

### O que podemos concluir

Podemos concluir que as metricas estao corretas para comparar saidas contra o
baseline.

### O que ainda nao podemos concluir

As metricas nao dizem, sozinhas, se uma tecnica e boa. Elas precisam ser
interpretadas junto com latencia, memoria, throughput e nivel de validade.

## Capitulo 15 - Protocolo experimental e matriz de validade

### Ideia inicial

Agora juntamos tudo. Se cada conceito tem teste, ainda precisamos garantir que
os resultados futuros sejam registrados de forma comparavel. Entao, criamos um
schema CSV e uma matriz de validacao.

### Definicao cientifica

Aqui a referencia e metodologica: experimentos precisam de rastreabilidade. No
nosso caso, rastreabilidade significa ligar cada conclusao a uma fonte, a um
teste e a um resultado.

### Por que isso importa no nosso trabalho

Sem matriz, podemos acabar escrevendo conclusoes mais fortes do que os dados
permitem. Entao, o pipeline obriga cada cenario a ter nivel de validade.

### Como implementamos

O schema CSV obrigatorio:

```python
CSV_COLUMNS = [
    "torch_version",
    "cuda_version",
    "gpu_name",
    "device",
    "model_kind",
    "scenario",
    "batch_size",
    "seq_len",
    "d_model",
    "num_heads",
    "dtype",
    "pruning_sparsity",
    "quantization_method",
    "validity_level",
    "latency_ms_mean",
    "latency_ms_p50",
    "latency_ms_p95",
    "throughput_tokens_s",
    "max_memory_bytes",
    "theoretical_flops",
    "mse",
    "mae",
    "r2",
    "cosine_similarity",
]
```

E a validacao rejeita colunas faltando, colunas extras e niveis invalidos.

### Como testamos

```python
validation = validate_benchmark_record(record)
invalid = validate_benchmark_record({"validity_level": "promessa_sem_evidencia"})

assert validation.is_valid
assert not invalid.is_valid
```

Tambem testamos que cenarios nao registrados sao rejeitados:

```python
with pytest.raises(ScenarioValidationError, match="cenario sem matriz"):
    validate_scenarios_registered(("baseline", "cenario_sem_evidencia"))
```

### Com o que comparamos

Comparamos cada resultado contra o baseline e contra a matriz de validade.

### O que podemos concluir

Podemos concluir que o estudo tem um caminho metodologico controlado para
aceitar ou rejeitar resultados.

### O que ainda nao podemos concluir

Ainda nao podemos concluir qual tecnica vence no desempenho final. O protocolo
so garante que, quando os benchmarks forem rodados, a conclusao sera
rastreavel.

## Fechamento - A frase metodologica do projeto

A historia inteira pode ser apresentada assim:

> O estudo nao mede desempenho antes de validar significado. Primeiro definimos
> cada conceito pela literatura. Entao transformamos essa definicao em uma
> propriedade observavel no codigo. Depois testamos essa propriedade com casos
> pequenos, deterministas e comparaveis com formulas ou bibliotecas de
> referencia. Por conta disso, quando formos medir pruning, quantizacao,
> latencia, memoria e throughput, saberemos exatamente o nivel de validade de
> cada conclusao.

Essa frase protege o trabalho de uma fragilidade comum: usar palavras fortes
sem evidencia proporcional. Entao, quando dissermos "bloco Transformer
simplificado", sabemos quais criterios ele passou. Quando dissermos
"quantizacao numerica", sabemos que houve `int8`, escala e erro medido. Quando
dissermos "pruning conceitual/algoritmico", sabemos que ha zeros e mascara, mas
nao necessariamente ganho de hardware.

O estado atual dos testes confirma essa base:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -W error
```

Resultado observado no ambiente de validacao cruzada em 24/08/2026:

```text
32 passed, 1 skipped
```

O teste ignorado exige CUDA. Os testes NumPy, PyTorch e TensorFlow/Keras
passaram sem `skip`.

Entao, o proximo passo do projeto nao e "inventar a metodologia"; a metodologia
ja esta desenhada. O proximo passo e usar esse protocolo para rodar benchmarks
controlados e interpretar os resultados sem ultrapassar o nivel de evidencia
observado.
