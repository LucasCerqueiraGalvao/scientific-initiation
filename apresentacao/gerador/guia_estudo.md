# Guia de Estudo do Tema

## 1. O que é um Transformer

Transformer é uma arquitetura de rede neural introduzida por Vaswani et al. em 2017.  
Ela se tornou central em IA porque substituiu estruturas recorrentes por operações matriciais paralelizáveis.  
Isso permitiu treinar e executar modelos muito maiores com melhor uso de hardware moderno.

Em termos simples, o Transformer recebe uma sequência de vetores e aprende quais partes dessa sequência devem influenciar outras partes.  
Ele faz isso principalmente por meio do mecanismo de atenção.

## 2. Por que a atenção custa caro

Na autoatenção, cada token pode ser comparado com muitos outros tokens da mesma sequência.  
Isso significa que, quando a sequência cresce, o número de comparações cresce bastante.  
Além disso, antes da atenção, o modelo ainda faz projeções lineares densas para gerar Q, K e V.

Então, o custo da inferência não vem de um único ponto, mas da combinação de:
- multiplicações matriciais densas
- comparação entre tokens
- armazenamento e leitura de memória intermediária

## 3. Diferença entre treino e inferência

Treino é a fase em que o modelo aprende ajustando seus pesos.  
Inferência é a fase em que o modelo já treinado é usado para gerar respostas.

No seu trabalho, o foco está na inferência.  
Isso é importante porque muitas otimizações relevantes para uso prático buscam tornar a inferência mais barata, rápida e escalável.

## 4. O que é quantização

Quantização é a redução da precisão numérica usada para representar dados.  
Por exemplo, em vez de armazenar valores em formatos de maior precisão, é possível usar 8 bits ou menos.

Benefícios esperados:
- menos memória
- menos tráfego de dados
- potencial aumento de velocidade

Risco:
- introdução de erro numérico

Por isso, quantização sempre envolve um trade-off entre eficiência e fidelidade.

## 5. O que é pruning

Pruning é a remoção ou desativação de pesos considerados menos relevantes.  
Em tese, isso reduz o tamanho efetivo do modelo e o número de operações.

Mas existe uma diferença importante em relação à quantização:
- quantização muda a representação dos valores
- pruning muda a estrutura efetiva do cálculo

Na prática, pruning pode exigir mais cuidado para realmente virar ganho de execução, porque hardware e bibliotecas nem sempre aproveitam bem esparsidade.

## 6. O que é KV cache

KV cache significa key-value cache.  
Durante a geração de tokens em modelos autoregressivos, o modelo guarda chaves e valores já calculados para não recomputar tudo a cada passo.

Isso acelera a inferência, mas tem um custo:
- conforme o contexto cresce, o KV cache ocupa cada vez mais memória

Por isso, técnicas que comprimem KV cache são muito relevantes para inferência eficiente.

## 7. O que o TurboQuant propõe

TurboQuant é um método recente de quantização online divulgado pelo Google Research em 24 de março de 2026.  
O paper correspondente está no arXiv desde 28 de abril de 2025.

Em alto nível, ele busca quantizar vetores com baixa distorção, incluindo aplicações em KV cache e busca vetorial.  
Na narrativa da sua apresentação, o ponto principal não é reproduzir toda a matemática do método, mas usar o trabalho como evidência de que quantização está no centro das soluções mais fortes para eficiência em inferência.

## 8. Por que quantização parece mais promissora aqui

No contexto do seu trabalho, quantização parece uma primeira hipótese mais forte por três motivos:

1. Ataca diretamente memória e largura de banda.
2. Tem apoio forte em literatura recente.
3. Pode ser comparada de forma relativamente controlada com baseline e pruning.

Isso não significa que pruning é irrelevante.  
Significa apenas que, para um primeiro ciclo experimental, quantização parece oferecer melhor custo-benefício científico.

## 9. O que significam as métricas

### Latência
Tempo gasto para executar uma operação ou cenário.

### Memória
Quanto espaço é necessário durante a execução.

### FLOPs
Estimativa do número de operações aritméticas de ponto flutuante.

### MSE
Erro quadrático médio. Penaliza mais fortemente erros grandes.

### MAE
Erro absoluto médio. Mede, em média, o quanto a saída se distancia da referência.

### R²
Coeficiente de determinação. Indica o quanto a saída candidata acompanha a referência.

### Similaridade de cosseno
Mede o alinhamento entre dois vetores.  
É útil quando interessa preservar direção e estrutura relativa da saída.

## 10. Como explicar seu experimento de forma clara

Uma forma simples de explicar é:

“Eu fixo os mesmos dados e os mesmos pesos-base, executo operações centrais do Transformer em três cenários diferentes e comparo custo computacional e qualidade de saída.”

Essa frase já resume:
- controle experimental
- comparação entre cenários
- foco em eficiência e fidelidade

## 11. Perguntas prováveis da banca

### “Por que você escolheu Transformer?”
Porque é a arquitetura dominante em IA atual e concentra desafios reais de inferência em tempo e memória.

### “Por que comparar quantização com pruning?”
Porque são duas estratégias clássicas e relevantes de otimização, mas atuam de formas diferentes: uma na representação numérica e outra na estrutura efetiva do modelo.

### “Por que dizer que quantização é melhor se você ainda não rodou o experimento?”
Eu não afirmo como conclusão do meu experimento. Eu apresento como hipótese inicial motivada por literatura recente e por características do problema.

### “Por que usar cenário controlado e modelo simplificado?”
Porque isso ajuda a isolar o efeito da técnica de otimização antes de ir para modelos completos mais complexos.

### “O que você espera encontrar?”
Espero observar redução de custo computacional nos cenários otimizados e depois comparar o quanto isso preserva ou degrada a qualidade das saídas em relação ao baseline.

## 12. Frase de segurança para a apresentação

Se em algum momento você travar, pode usar esta frase:

“Meu trabalho ainda está na fase inicial de execução, então hoje eu apresento principalmente a relevância do problema, a hipótese técnica e o desenho experimental que vai permitir validar essa hipótese com dados próprios.”
