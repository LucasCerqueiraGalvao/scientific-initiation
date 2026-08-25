# **📝 Observações**

jul. 31, 2026

## **Reunião em 31 de jul. de 2026 às 16:01 GMT-03:00**

Registros da reunião [Transcrição](https://docs.google.com/document/d/1WBgwj1gDAWh_7VGckHvPhRfSlcRzFIwL2JlnIDhRYbQ/edit?usp=drive_web&tab=t.v1ze8hoqd423) [Gravação](https://drive.google.com/file/d/1-FFyxhkpgkhlesSqULSIaujRSiDnT1UN/view?usp=drive_web) 

### **Resumo**

Discussão sobre validação do modelo Transformer, escopo de otimização no mecanismo de autoatenção e uso de modelos.

**Validação do modelo Transformer**  
A comparação com bibliotecas como o TensorFlow assegura a correção das implementações customizadas. Operações matemáticas fundamentais já validaram resultados específicos com sucesso.

**Escopo da otimização técnica**  
A otimização concentrada no mecanismo de autoatenção limita a complexidade do projeto ao custo computacional relevante. A arquitetura completa foi descartada por excesso de trabalho.

**Estratégia de testes externos**  
A utilização de modelos pré-treinados do Hugging Face para testes de inferência constitui uma abordagem válida. O cronograma de entrega permanece mantido para novembro.

### **Próximas etapas**

- [ ] \[Lucas Galvão\] Comparar Implementação: Comparar a implementacao local com funcoes padrao do TensorFlow e PyTorch para validar a integridade das operacoes.

- [ ] \[Lucas Galvão\] Criar Relatório: Elaborar um relatorio contendo tabelas e graficos comparativos dos resultados obtidos. Registrar o passo a passo das atividades e as referencias teoricas consultadas.

- [ ] \[Lucas Galvão\] Realizar Testes: Conduzir testes de benchmark para avaliar o desempenho do modelo desenvolvido.

### **Detalhes**

* **Validação do Modelo Transformer**: Lucas Galvão e Walter Silva Oliveira discutiram a validação de sua implementação do modelo Transformer. Walter Silva Oliveira sugeriu utilizar as camadas prontas do TensorFlow como base de comparação para validar o código desenvolvido. O objetivo é garantir que as implementações customizadas estejam alinhadas com as bibliotecas consolidadas antes de avançar para a aplicação de técnicas de otimização ([00:00:00](#00:00:00)).

* **Estratégia de Validação e Comparação**: Walter Silva Oliveira enfatizou a necessidade de garantir a correção do código comparando os resultados com bibliotecas como o TensorFlow ou Keras, utilizando a mesma entrada e semente de ruído. Lucas Galvão observou que comparações determinísticas já foram realizadas para operações específicas, como o produto interno (dot product) e as operações de Query, Key e Value (QKV), que não exigem validações arquiteturais adicionais por serem definidas por equações matemáticas claras ([00:01:28](#00:01:28)).

* **Documentação e Relatórios**: Walter Silva Oliveira recomendou que Lucas Galvão elabore gráficos e tabelas para documentar os resultados das comparações entre as implementações manuais e os resultados esperados. Esse relatório é considerado fundamental por Walter Silva Oliveira para consolidar o entendimento do trabalho e garantir que a base esteja sólida antes de aplicar técnicas avançadas como poda (pruning) ou quantização ([00:05:35](#00:05:35)).

* **Escopo da Otimização**: Lucas Galvão questionou sobre a definição formal de um Transformer e os limites da otimização. Walter Silva Oliveira esclareceu que o foco do projeto deve ser otimizar o mecanismo de autoatenção (self-attention), que possui o maior custo computacional, em vez de tentar otimizar a arquitetura completa do Transformer, o que seria excessivamente complexo e trabalhoso para o cronograma disponível ([00:09:18](#00:09:18)) ([00:11:27](#00:11:27)).

* **Testes com Modelos de Terceiros**: Lucas Galvão perguntou sobre a possibilidade de utilizar modelos pré-treinados, como os disponíveis no Hugging Face, para testar a inferência. Walter Silva Oliveira confirmou que essa é uma abordagem válida, embora o sucesso da implementação dependa do tempo restante até o final do projeto ([00:15:21](#00:15:21)).

* **Cronograma e Prazos**: Os participantes discutiram o cronograma, notando que o prazo para a entrega/apresentação é por volta de novembro. Walter Silva Oliveira mencionou que as reuniões institucionais e o retorno formal dos professores acontecerão na semana seguinte, o que permitirá definir melhor os próximos passos administrativos ([00:15:21](#00:15:21)).

* **Atividades Acadêmicas e Certificados**: Lucas Galvão expressou dificuldades em obter as 20 horas complementares exigidas pela faculdade, mencionando problemas com a aceitação de certificados de maratonas de programação. Walter Silva Oliveira orientou sobre a verificação dos limites de horas permitidos por categoria no sistema da instituição ([00:17:28](#00:17:28)).

* **Discussão Pessoal e Equilíbrio**: Lucas Galvão e Walter Silva Oliveira conversaram sobre interesses pessoais, incluindo jogos como League of Legends e Civilization, e sobre suas rotinas de trabalho e estudo. Lucas Galvão detalhou sua configuração de trabalho com desktop remoto, enquanto ambos discutiram o equilíbrio entre vida pessoal e obrigações acadêmicas ([00:18:20](#00:18:20)).

* **Encaminhamentos Finais**: Walter Silva Oliveira reforçou que Lucas Galvão deve começar a escrever o relatório e documentar os processos imediatamente para evitar sobrecarga futura. Lucas Galvão confirmou que iniciará os testes de benchmark e a estruturação da documentação conforme orientado ([00:23:13](#00:23:13)).

* **Reflexões sobre a Vida Universitária e Deslocamento**: A reunião terminou com uma conversa sobre as dificuldades de deslocamento entre Santos e São Paulo, as dinâmicas de amizade durante e após o período acadêmico e as expectativas para o final do semestre ([00:25:52](#00:25:52)).

*Revise as anotações do Gemini para checar se estão corretas. [Confira dicas e saiba como o Gemini faz anotações](https://support.google.com/meet/answer/14754931)*

*Como está a qualidade de **destas observações?** [Responda a uma breve pesquisa](https://google.qualtrics.com/jfe/form/SV_5bXzKQfylMIhSXc?confid=n3lUFOLoZPItvxYQB_PrDxIWOBABMgUIigIgABgFCA&detailLevel=standard&hasImages=False&entryPoint=footerMain&isGoogler=False) para nos dar seu feedback, incluindo o quanto as observações foram úteis para o que você precisa.*

# **📖 Transcrição**

jul. 31, 2026

## **Reunião em 31 de jul. de 2026 às 16:01 GMT-03:00 \- Transcrição**

### **00:00:00** {#00:00:00}

**Lucas Galvão:** Pronto. Daí depois eu posso usar o que você comentou. Hum.

**Walter Silva Oliveira:** Porque tá que você tava usando v que você usa as coisas do pai torche, mas você não lembra seu próprio p seu kerfow que já tem

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** o sistema de transformer pronto, não tem? Não sei se chegou a ver

**Lucas Galvão:** Eh,

**Walter Silva Oliveira:** isso.

**Lucas Galvão:** não, o transformer completo. Pronto.

**Walter Silva Oliveira:** Isso é porque você consegue fazer uma layer de transformer, pelo menos no temor flow, ela tem.

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** Tu acha que dá seria legal para validar também? Você tá usando o p o pche basicamente para criar os objetos.

**Lucas Galvão:** Угу.

**Walter Silva Oliveira:** mais criar as layers, talvez pensar como que a gente consegue compare esse cara com o que já é feito direto no nessas bibliotecas que se coloca quero uma lei de transformer e aí ele utiliza porque aí a gente garante, beleza? Então essas implementações que tem aqui estão batendo com a funçãozinha lá do Tens flow. Então a gente conseguindo validar esse ponto, o que que a gente pode fazer? a gente pode começar a brincar um pouco mais de como essas operações são feitas, antes elas com um pouco mais de detalhe e ver o impacto disso, entende?

### **00:01:28** {#00:01:28}

**Walter Silva Oliveira:** Só pra gente tem tem essas funções prontas, aparentemente elas estão OK. dei uma olhada por cima, mas só pra gente ter um caso base comparando com a biblioteca pura do densor flow,

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** por

**Lucas Galvão:** OK. É o interessante, dá para usar assim o Tflow,

**Walter Silva Oliveira:** exemplo.

**Lucas Galvão:** o que o Pyth aqui, eu acho que ele não tem o o não tem eh não tem o transformer completo, mas ele tem aquelas partes de dos tensores, né? Eu acho que tem até aqui.

**Walter Silva Oliveira:** Sim,

**Lucas Galvão:** Eh,

**Walter Silva Oliveira:** sim.

**Lucas Galvão:** ah, não lembro onde estava. É aqui, exatamente parte dos tensores que eh daí você acha que é interessante pegar tipo isso e comparar também com o desempenho do da própria do tenser flow para ter mais uma

**Walter Silva Oliveira:** É porque isso que beleza.

**Lucas Galvão:** validação?

**Walter Silva Oliveira:** Aparentemente tá funcionando, mas é aquela questão que a gente conversou lá no início,

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** como que você garante que o que você tem feito? Então beleza,

**Lucas Galvão:** Sim.

**Walter Silva Oliveira:** a ideia inicial era vamos entender um pouquinho mais como o transformer funciona,

**Lucas Galvão:** Угу.

**Walter Silva Oliveira:** como que a gente consegue otimizar esse transformer considerando que ele é um bloco que vai ser rodado recorrentemente, a gente quer de alguma maneira diminuir o tamanho desse cara.

### **00:02:53**

**Walter Silva Oliveira:** Aí, beleza,

**Lucas Galvão:** Угу.

**Walter Silva Oliveira:** a gente tem um monte de técnica que a gente pode utilizar para diminuir tamanho, fazer quantização, etc. Show. Questão é como esse, então a hipótese nossa é como que a gente vai avaliar impacto dessas técnicas no Transformer. Aí a ideia então foi, beleza, vamos estudar primeiro como ele funciona. A partir do momento que ele funcionou, a gente começa a aplicar essas técnicas. Aí agora o ponto é como que eu sei que esses códigos que eu desenvolvi eles

**Lucas Galvão:** Угу.

**Walter Silva Oliveira:** estão corretos? Então eu preciso de alguma maneira comparar o resultado desse cara com alguma questão da literatura. Então a minha hipót, eu acho que o que faz sentido é pensar em beleza,

**Lucas Galvão:** Угу.

**Walter Silva Oliveira:** eu sei que tensor flow ou que eras são bibliotex consolidad essa área de mach learning, então eu posso d mesma entrada, uma mesma semente de ruído eu tenho que ter a mesma saída nos dois.

**Lucas Galvão:** Угу.

**Walter Silva Oliveira:** Feito isso, beleza. Se então que todas essas questões ali de de metas,

**Lucas Galvão:** Вот.

**Walter Silva Oliveira:** na minha cash, tá tudo funcionando. Minhas operações são feitas corretamente. Aí feito isso, aí é a parte de brincar.

### **00:04:12**

**Walter Silva Oliveira:** Então, como que eu posso criar cenários? Aí a gente continua nessa parte de que pensar em cenários que a gente vai testar esse nosso Transformer.

**Lucas Galvão:** Sim.

**Walter Silva Oliveira:** Aí só pra gente garantir o que que quando a gente tem esse salto a gente saiba que o problema não tá na nossa implementação

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** inicial.

**Lucas Galvão:** Outra coisa que eh foi feita foi fazer a comparação de alguns processos com equações prontas, né? Que daí seria tipo aquelas equações que são tipo eh,

**Walter Silva Oliveira:** Угуm.

**Lucas Galvão:** que é tem aqueles QKV, né? aqui KT que daí, tipo, eles são operações prontas, daí eu fiz a comparação delas com as com as implementações que a gente fez aqui. Então, por exemplo, para esses para esse caso, não é necessário fazer uma mais comparações, né? Porque a gente já fez tipo uma comparação determinística com uma equação pronta, que seria a equação da definição lá, por exemplo, do que que é uma que que é um tensoro, o que que é um, qual o nome? Aquele produto pontual lá, dot product. Aí tem essas,

**Walter Silva Oliveira:** assim.

**Lucas Galvão:** é, sim, tem essas partes que elas são determinísticas, que elas são definidas por equações, não são por, por exemplo, não é arquitetura.

### **00:05:35** {#00:05:35}

**Lucas Galvão:** A parte da arquitetura é mais difícil de validar do que essa parte de eh multiplicação de matricial e tals.

**Walter Silva Oliveira:** Sim, sim, com certeza. Mas e exatamente,

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** por isso é mais complicado, porque que é importante a gente vai lhe dando passo a passo e talvez fazer isso,

**Lucas Galvão:** Uhum. Uhum.

**Walter Silva Oliveira:** talvez pegar, fazer uns gráficos, nem que seja uma tabelinha de resultados, só comparando. Beleza? 아 aqui tá o que eu o fazendo, digamos,

**Lucas Galvão:** Tá.

**Walter Silva Oliveira:** pegando a equação lá, fazendo multiplicação matricial, dado é a mesma entrada, tenho esse resultado. Qual eu falei na mão utilizando p torche, utilizando o tensor flow e aí a gente consegue

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** validar isso. Aí a gente começa a pensar nessa questão de como que a gente melhora o prone, como a gente melhora a aonar a quantização desses caras e ver como isso impacta.

**Lucas Galvão:** Uhum. Beleza. Legal. Eh.

**Walter Silva Oliveira:** Acho você já fez bastante coisa, então eu acho que é importante também começar a Claro está ainda numa etapa inicial,

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** mas vai pensar o que que é resultado, que que eu fiz, que que posso comparar, fazer uns gráficos, que eu até comentei lá, pegar esses, fazer meio que o relatóiozinho, que é ficar até mais fácil para depois você entender,

### **00:06:51**

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** tá? Que que eu fiz? Da onde que eu peguei as coisas para fazer um relatório para

**Lucas Galvão:** Uhum. Sim.

**Walter Silva Oliveira:** depois?

**Lucas Galvão:** Que gráficos que você acha que são interessantes? Por exemplo, eh, por exemplo, esse de, por exemplo, como se fosse um eixo é as variáveis e um eixo é o resultado das equações relacionadas às variáveis.

**Walter Silva Oliveira:** Isso passei o gráfico uma tabela, por exemplo. Digamos que,

**Lucas Galvão:** Pode

**Walter Silva Oliveira:** ó, se eu pegar fazer a conta ali novo no Python,

**Lucas Galvão:** ser?

**Walter Silva Oliveira:** vai dar X. Izando o método tal,

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** vai dar uma variância de vai dar o tal valor.

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** Aí, claro, depende muito do da maneira que você tá testando e do que você tá testando, né? Mas só pra gente ter esses esses estados palpáveis que fica até mais legal pra gente falar falar. Beleza? Ó, então ó,

**Lucas Galvão:** M.

**Walter Silva Oliveira:** a gente estudou essa parte teórica, entendemos como funciona, quais são as operações que estão sendo feitas e aí beleza? Então vamos implementar o nosso. Implementamos o nosso, ó, tô vendo aqui, ó.

### **00:08:01**

**Walter Silva Oliveira:** Se eu pegar a minha conta aqui do produto interno e eu pegar a funçãozinha lá do pai que faz o

**Lucas Galvão:** Угу.

**Walter Silva Oliveira:** dot product, deu a mesma coisa. E aí a gente vai escalando isso até, ó, se eu pegar essa mesma entrada, passar com o mesmo transforme, com o mesmo com minhas mesmas chaves, eu vou ter o mesmo resultado na saída. Aí, beleza? Então, eu sei que todo o nosso fluxo tá funcionando. Feito isso, aí a gente pode pensar, beleza, quais são as técnicas melhores? Aí a gente volta na segunda etapa lá da metodologia e pensar em como que a gente cria

**Lucas Galvão:** Угу.

**Walter Silva Oliveira:** para testar e otimizar o Transformer em si. Eu cheguei a ver que tinha até uns modelinhos de você rodar meio com uma Iá no próprio Transformer para você saber quais são os pesos que você tem da Prun. Tem tem um aí tem uma vida para pesquisar.

**Lucas Galvão:** Угу.

**Walter Silva Oliveira:** A gente pode pegar uma ou duas técnicas dessas da metodologia, aplicar testar o resultado.

**Lucas Galvão:** É, eh, sim, entendi. Uma coisa que que ficou bastante na minha cabeça era, por exemplo, o que que é um, assim, a definição formal de um transformer. Tipo assim, o que a gente tá pegando aqui?

### **00:09:18** {#00:09:18}

**Lucas Galvão:** Eh, eu acho que a gente não tá tipo replicando exatamente o que que o Atation isol e o Need fala que que é um transformer, que ele é muito, ele tem muitas etapas lá, mas a gente tá pegando operações que ocorrem dentro do Transformer, né?

**Walter Silva Oliveira:** É porque basicamente você tá fazendo a parte do selfation, né?

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** Então beleza. Então por enquanto a gente tá precisando só tr a ideia de transform inteiro é beleza. Vai pegar o self, passar um LP, fazer o encoding e tudo mais.

**Lucas Galvão:** Sim, sim.

**Walter Silva Oliveira:** Ei,

**Lucas Galvão:** Daí vai fazer o profet

**Walter Silva Oliveira:** só que essa própria questão do Censtion,

**Lucas Galvão:** lá.

**Walter Silva Oliveira:** ele acaba sendo uma matriz grande, né? Então, beleza, que a gente não chega no transform,

**Lucas Galvão:** Sim.

**Walter Silva Oliveira:** vamos tentar só como que a gente otimiza essa matriz de autens.

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** Acho que é isso. A gente tem um mundo para explorar. Só é importante sempre que a gente dá um passinho sabendo que a gente tem que terra firme, porque senão se a gente avançar muito rápido, a gente vai chegar lá na frente, putz, tá dando errado e aí onde é que tá errado?

### **00:10:29**

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** Só essa é a única preocupação. Mas assim, mas você tá já fez mais do que eu imaginava.

**Lucas Galvão:** Entendi.

**Walter Silva Oliveira:** Você tá indo no

**Lucas Galvão:** É, então a minha ideia era, na verdade era tipo fazer 90% da aí nesse mês, né? Mas acho que eu fiquei muito muito preguiçoso aí período de

**Walter Silva Oliveira:** não, mas não tem. É assim,

**Lucas Galvão:** férias.

**Walter Silva Oliveira:** claro, nas férias é o momento que você tem mais tempo para fazer as coisas, mas é isso, não tem por ter pressa. O que importa é validando as etapas, o início é a parte mais complicada,

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** porque você ainda não, pô, o que que é o K kv ali? Como é que funciona, como que eu faço con que tá,

**Lucas Galvão:** Sim.

**Walter Silva Oliveira:** é uma matriz, que operação que é essa? Depois que você internaliza isso,

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** as próximas etapas ficam muito mais fáceis. Então, início é a parte mais

**Lucas Galvão:** É sim. É sim. E assim,

**Walter Silva Oliveira:** complicada.

**Lucas Galvão:** o que eu achei especialmente complicado é essa parte de definição.

### **00:11:27** {#00:11:27}

**Lucas Galvão:** Tipo, por exemplo, o que que é transformer? Você vai pegando um artigo lá e ele não fala tipo explicitamente, ah, definição de transformer é tal, não. Ele só, tipo, fala assim, ah, estou implementando isso aqui, isso aqui é transformer, é tipo um monte de attentions lá. Daí, por exemplo, tanto que é o tanto que, tipo, então, por exemplo, como é que eu faço comparações para saber se as minhas operações elas estão corretas diante de dos artigos do que ele definá? Yeah. É fácil fazer paraas operações matemáticas, né, que é tipo fazer a atation lá, que é o QKV, eh, que é mais tranquilo a parte de projeção de produto lá, o dot pro, mas a parte de eh como esses itens se relacionam é muito, acho que é a parte mais complicada.

**Walter Silva Oliveira:** É bem

**Lucas Galvão:** E e eu não sei se, por exemplo, se dentro do escopo do nosso título lá, do nosso trabalho inicial, eu até revisei isso, talvez não seja necessário fazer esse tipo de revisão, tipo assim, de como os itens se relacionam, porque o nossa proposta é analisar eh estratégias de otimização dentro de operações específicas que estão dentro da Transformer, não otimização da Transformer em si, mas talvez fique meio É irrelevante fazer também a otimização

### **00:12:54**

**Walter Silva Oliveira:** Não,

**Lucas Galvão:** ou

**Walter Silva Oliveira:** o que você pode, você pode pensar é o seguinte,

**Lucas Galvão:** não?

**Walter Silva Oliveira:** beleza? Se a gente pensar como funciona, né, esse bloco, essa camada inteira, a gente vai ter a parte que mais tem custo computacional é a parte tal.

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** Então vamos focar só nessa parte que a priori lá semestre passado,

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** se a gente otimizar essa parte que é a mais relevante,

**Lucas Galvão:** Mum.

**Walter Silva Oliveira:** a gente consegue melhorar a performance. Então beleza, a gente pode até colocar isso como justificativa no faz a gente otimizar, otimizar o transforma inteiro, talvez seja algo que dê muito trabalho pro nosso tempo. Então a gente vai tem que focar nosso esforço em entender o mecanismo principal, que é o mecanismo de alta tensão.

**Lucas Galvão:** Sim,

**Walter Silva Oliveira:** E aí a partir desse pressuposto,

**Lucas Galvão:** sim.

**Walter Silva Oliveira:** tá? Como ele funciona com essas operações que ele faz e como o qual é o espaço que eu tenho para para otimizar. E desde as estratégias que tem na literatura, a gente não precisa inventar nada como elas funcionam e como elas se comparam.

**Lucas Galvão:** Mhm.

**Walter Silva Oliveira:** Então acho que é isso. Pensar, beleza, vamos focar só na parte de alta tensão que tem lá multiplicação, é só max para calcular a tensão.

### **00:14:09**

**Walter Silva Oliveira:** Beleza? Como que isso acontece?

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** Qual quais as técnicas que eu posso utilizar para melhorar isso? Aí a gente aplica essas técnicas e mexe formas.

**Lucas Galvão:** Sim. Perfeito, perfeito. É, mas você acha que antes de fazer isso tem uma etapa anterior, né, que era fazer a comparação lá com a outra biblioteca,

**Walter Silva Oliveira:** É,

**Lucas Galvão:** né,

**Walter Silva Oliveira:** e antes isso é talvez é que eu é agora

**Lucas Galvão:** outra que

**Walter Silva Oliveira:** parando para pensar o do tensor flowing pegar faz tanto tempo que ouvi

**Lucas Galvão:** Ehm

**Walter Silva Oliveira:** isso eh porque eu acho que não sensor flow. Eu só coloco o layer transforma ele inteiro. Não sei se a gente vai conseguir ver a parte de alta tensão interna ali da operação sendo

**Lucas Galvão:** M.

**Walter Silva Oliveira:** feita.

**Lucas Galvão:** Não deve ter sim.

**Walter Silva Oliveira:** É, mas é isso. É tipo pegar pega só vai uma um cálculo desses,

**Lucas Galvão:** Eh,

**Walter Silva Oliveira:** né? Uma interação e aí a gente compara, vê se o resultado que dá na saída dele é o resultado que a gente tem. É só isso que precisa fazer. para falar, beleza, a minha implementação ela

**Lucas Galvão:** certo.

### **00:15:21** {#00:15:21}

**Lucas Galvão:** Legal. Acho que dá fazer.

**Walter Silva Oliveira:** funciona.

**Lucas Galvão:** Você acha que vale a pena pegar eh modelos prontos? Tipo, quando a gente for testar, eh, vale a pena pegar, por exemplo, alguma coisa, um modelo já treinado para gente testar inferência em cima dele, tipo,

**Walter Silva Oliveira:** Pode pegar no Face.

**Lucas Galvão:** pegar um alguma coisa lá no huging face e isso.

**Walter Silva Oliveira:** Pode sim, sim, com certeza.

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** Se sobrar tempo, né? Mas acho que não é

**Lucas Galvão:** É, eu tô É,

**Walter Silva Oliveira:** assim.

**Lucas Galvão:** já tá meio apertado, né? É para setembro.

**Walter Silva Oliveira:** Eu tenho que ver a data exata, mas deve ser por aí,

**Lucas Galvão:** É.

**Walter Silva Oliveira:** porque aí vai a apresentação deve ser ou nacional do ano ou em dezembro.

**Lucas Galvão:** É,

**Walter Silva Oliveira:** Aí durante uns um, dois meses antes. Essa para aí setembro,

**Lucas Galvão:** é iss até que é quando?

**Walter Silva Oliveira:** outubro. aca em novembro,

**Lucas Galvão:** Em novembro. Ah, tá. Então tá tranquilo.

**Walter Silva Oliveira:** mas aí tem que mandar antes para ser aprovado. Não sei. Eh,

### **00:16:15**

**Lucas Galvão:** Não,

**Walter Silva Oliveira:** provavelmente isso vai ser discutido na semana que vem, né, que os professores a priori voltaram hoje. Hoje é o primeiro dia que de férias,

**Lucas Galvão:** não.

**Walter Silva Oliveira:** mas todas as reuniões com a reitoria e com o colegiado s semana que vem.

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** A gente deve discutir isso no próprio colegiado, como é que vai ficar essas atas?

**Lucas Galvão:** Uhum. Vocês chegam muitos alunos. nessas reuniões.

**Walter Silva Oliveira:** Só um só que merece.

**Lucas Galvão:** Beleza. Chegar lá o Felipe.

**Walter Silva Oliveira:** Não, nem fala,

**Lucas Galvão:** Lembra do Felipe?

**Walter Silva Oliveira:** nem fala esse nome. Pelo amor de Deus. Graças. Eu tô feliz quando eu vou dar lá para vocês só

**Lucas Galvão:** Que isso para você vai dar?

**Walter Silva Oliveira:** alegria.

**Lucas Galvão:** Eh, para aquele pessoal lá do primeiro ano de

**Walter Silva Oliveira:** Eu sou Não, não, só tô com quarto e sexto.

**Lucas Galvão:** novo.

**Walter Silva Oliveira:** Eu vou eu vou est aqui à noite, quer dizer, eu vou estar aqui terça, quarta, quer dizer, segunda, terça e quarta, mas segunda e quarta eu só dou aula de manhã. Aí vai ficar amanhã à tarde e terça ou à noite.

### **00:17:28** {#00:17:28}

**Lucas Galvão:** Uhum. Cara, não aguento mais a faculdade. Tem que acabar logo.

**Walter Silva Oliveira:** Tá acabando,

**Lucas Galvão:** Acabando.

**Walter Silva Oliveira:** tá acabando.

**Lucas Galvão:** Tá acabando, realmente.

**Walter Silva Oliveira:** Eu acho que o pior já passou. Só fechar essas horinhas aí. Você vai cheirar de letra e a

**Lucas Galvão:** É,

**Walter Silva Oliveira:** gente aí,

**Lucas Galvão:** eu tenho ainda as 20 horas complementares aí. Acho que eu vou doar sangue conta.

**Walter Silva Oliveira:** sei lá, vai doar sangue. Ai, sei lá,

**Lucas Galvão:** Acho que não.

**Walter Silva Oliveira:** pô. Na minha época, contava

**Lucas Galvão:** Acho que não. Tem que vou ter que pagar a lura lá e emitir um monte de certificado

**Walter Silva Oliveira:** Isso para loura tem um tem uns limites de horas tem as categorias lá e você

**Lucas Galvão:** lá.

**Walter Silva Oliveira:** tem é limitado a quantidade de horas por categoria. Então por mais que você faça isso de 200 horas você só consegue emitir tipo 15, tipo 10\.

**Lucas Galvão:** Sim, mas eu preciso só de 20\.

**Walter Silva Oliveira:** Tem que ver.

**Lucas Galvão:** Cara, você acredita que eu fiz eu fiz a maratona de programação todos os semestres

### **00:18:20** {#00:18:20}

**Walter Silva Oliveira:** Mas vai que eu acredito que você

**Lucas Galvão:** e eu não subi e eu não subi nenhuma

**Walter Silva Oliveira:** não mandou nenhuma. É, eu acredito.

**Lucas Galvão:** delas,

**Walter Silva Oliveira:** Você é mais comum do que deveria, Lucas.

**Lucas Galvão:** mano. É assim,

**Walter Silva Oliveira:** É mais comum do que

**Lucas Galvão:** e não,

**Walter Silva Oliveira:** deveria.

**Lucas Galvão:** o que é chato é eles não permitirem eu mandar o certificado

**Walter Silva Oliveira:** Você tem que mandar 15 dias depois.

**Lucas Galvão:** com absurdo. Deixa eu ver quantas horas eu preciso, mano. Você jogar LOL, professor. Sua

**Walter Silva Oliveira:** jogo, né? Jogo assim,

**Lucas Galvão:** cara,

**Walter Silva Oliveira:** fazia muito tempo quando eu jogava, eu vi, aí eu saí o LOL Classic lá, eu joguei umas três partidinhas com os meus amigos para relembrar os velhos tempos.

**Lucas Galvão:** não vou te adicionar pra gente jogar lá depois, pô.

**Walter Silva Oliveira:** Eu só jogar.

**Lucas Galvão:** Não, bora. Sou erra.

**Walter Silva Oliveira:** Perfeito.

**Lucas Galvão:** Qual o seu Qual seu nickname?

**Walter Silva Oliveira:** Eu vou, cadê? Vou mandar minha tag. Voltinho. V hasinho.

### **00:19:20**

**Walter Silva Oliveira:** Não, faltou um H. Pior que seas férias.

**Lucas Galvão:** Bom nick. Bom nick.

**Walter Silva Oliveira:** Eh, eu joguei Civilization com o Raul e com o Henrique. Tur de reserva.

**Lucas Galvão:** Quem é R Henrique?

**Walter Silva Oliveira:** é da outra turma

**Lucas Galvão:** Ah, tá, entendi. Pô, mas o o CV online é

**Walter Silva Oliveira:** aí. Ah,

**Lucas Galvão:** complicado.

**Walter Silva Oliveira:** é assim, mas é bom, é legal, é legal. 100% de

**Lucas Galvão:** Nossa,

**Walter Silva Oliveira:** rate.

**Lucas Galvão:** pô. Mas você assim, acho que eu nunca joguei uma partida do Civilization até o final. Meu limite,

**Walter Silva Oliveira:** É assim.

**Lucas Galvão:** quando eu jogava, meu limite era sempre quando eu descobria o satélite, que daí desbloqueava o mapa todo e meu PC não rodava mais ali, ó.

**Walter Silva Oliveira:** Uhum.

**Lucas Galvão:** Era o limite, dera o fim.

**Walter Silva Oliveira:** Ah, mas agora você tem uma Não, você tem uma placa de boa, pô.

**Lucas Galvão:** É, mas agora,

**Walter Silva Oliveira:** Agora vai lá agora.

**Lucas Galvão:** mas agora todos os meus amigos trabalham, fazem mestrado, namoram e ninguém quer ficar 36 horas numa parceira de

### **00:20:29**

**Walter Silva Oliveira:** É, não,

**Lucas Galvão:** C.

**Walter Silva Oliveira:** 36 não, mas a primeira partida foi foram 8 horas.

**Lucas Galvão:** É,

**Walter Silva Oliveira:** Aí a segunda foi mais rapidinho,

**Lucas Galvão:** então

**Walter Silva Oliveira:** acho que foi três. Não é tão ruim.

**Lucas Galvão:** não, não, nunca falei que é ruim, falei que é é longo.

**Walter Silva Oliveira:** Pronto, na durante as férias dá para jogar,

**Lucas Galvão:** Te adicionei aqui, ó.

**Walter Silva Oliveira:** tá? Não,

**Lucas Galvão:** Deixa eu ver seu perfil.

**Walter Silva Oliveira:** não consigo abrir aqui porque, né, a não deixa,

**Lucas Galvão:** Ah. Não acredito.

**Walter Silva Oliveira:** Mas agora chegar em casa.

**Lucas Galvão:** Não

**Walter Silva Oliveira:** É sério?

**Lucas Galvão:** acredito.

**Walter Silva Oliveira:** Você nunca tentou abrir? Se eu tent, ó, se eu tentar abrir aqui. Acho que não tem nesse celular não. Se eu tentar abrir aqui. Tem logo aqui? Não, não tem nesse computador. Graças a Deus.

**Lucas Galvão:** Nossa, você é muito mentiroso, professor. Olha isso. Você jogou três, seis.

**Walter Silva Oliveira:** Eu joguei ontem é umas três horinhas.

**Lucas Galvão:** Jogou, mas jogou 10 partidas.

### **00:21:24**

**Walter Silva Oliveira:** Jogando aranã. É rapidinho. Vê depois disso. A última vez que eu joguei, tinha jogado há uns três meses

**Lucas Galvão:** Não, daí Ah, não, o mês passado, pô.

**Walter Silva Oliveira:** atrás.

**Lucas Galvão:** É, realmente, teve um, é que foi dia 29, daí em cima tava dia 30, só que um é 29/06, outro é 30/07. Olha que coincidência,

**Walter Silva Oliveira:** Não, não é coincidência.

**Lucas Galvão:** não é coincidência.

**Walter Silva Oliveira:** Não é coincidência. Final do mês sempre mais tranquilo pro pessoal. Mas assim, eu jogo bem casualmente, não jogo

**Lucas Galvão:** que eu não jogava ranked desde 2022,

**Walter Silva Oliveira:** ranked.

**Lucas Galvão:** que foi o ano que eu parei de eh que eu comecei a faculdade, né? 2023 comecei a faculdade. Daí eu joguei bastante 2022\. Da. Tô jogando um pouquinho. Divertido. É divertido até não ficar mais assim, até você ficar no elo que você pertence e daí o jogo fica ruim de

**Walter Silva Oliveira:** Não é assim, eu eu tô com e você fica muito tempo para você jogar,

### **00:22:20**

**Lucas Galvão:** novo.

**Walter Silva Oliveira:** você joga, fala: "Pô, muito legal jogar lô".

**Lucas Galvão:** Até

**Walter Silva Oliveira:** Mas aí depois você joga dois dias,

**Lucas Galvão:** mesmo.

**Walter Silva Oliveira:** acabou, acabou o amor, esquece. Tem que ser assim.

**Lucas Galvão:** É, então é que tipo assim, quando você tá muito tempo sem jogar, toda partida que você joga você ganha mais de 10 kills,

**Walter Silva Oliveira:** É legal.

**Lucas Galvão:** né? Você pega tipo 15

**Walter Silva Oliveira:** É.

**Lucas Galvão:** kills.

**Walter Silva Oliveira:** E isso igual no Fortnite você tem que jogar só, tem que ganhar das criancinha. Aí quando vem o pessoal bom,

**Lucas Galvão:** Aham.

**Walter Silva Oliveira:** aí você vai

**Lucas Galvão:** Não, por isso que é bom jogar nas férias,

**Walter Silva Oliveira:** jogar.

**Lucas Galvão:** que daí tem as crianças às 10 horas da manhã. Ó,

**Walter Silva Oliveira:** É

**Lucas Galvão:** mostrar a tática que eu uso para trabalhar.

**Walter Silva Oliveira:** verdade.

**Lucas Galvão:** Tá vendo aqui, ó, que tá vendo esse aplicativo aqui?

**Walter Silva Oliveira:** Uhum.

**Lucas Galvão:** É um desk top remoto que eu uso pro meu notebook.

**Walter Silva Oliveira:** Sim.

**Lucas Galvão:** O notebook fica tipo atrás de mim aqui, daí eu consigo usar minha minha tela.

### **00:23:13** {#00:23:13}

**Lucas Galvão:** Daí fica tipo assim,

**Walter Silva Oliveira:** E aí? Aí fica só a área do trabalho.

**Lucas Galvão:** não fica é fica tudo do relacionado ao meu trabalho da

**Walter Silva Oliveira:** É, eu tenho dois computadores.

**Lucas Galvão:** empresa.

**Walter Silva Oliveira:** notebook uso para fazer coisa doutorado da Santos e me PC uso para parte

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** boa.

**Lucas Galvão:** Então daí é isso. Deixa eu ver algo mais interessante para ser falado aí.

**Walter Silva Oliveira:** Acho que é isso. Acho que isso aí é um caminho bom. Só fazer essa questão de já começar a escrever porque senão você vai esquecer. escrever texto mesmo, não só colocar os tópicos,

**Lucas Galvão:** Угу.

**Walter Silva Oliveira:** que é importante até para você começar a a lincar suas ideias, que são muitos conceitos. E aí como que uma coisa se encadeia na outra? Você vai acabar só entendendo isso melhor quando você começar a escrever. É uma parte importante até para não ficar quando chegar lá em setembro não ficar muito apertado nisso. E é isso. Fechar essas avaliações com resultados, fazer umas tabelinhas de resultados. os estados parciais em si. E aí a gente parte para continuar as

**Lucas Galvão:** Beleza.

### **00:24:34**

**Walter Silva Oliveira:** coisas. Deixa eu ver se tem mais alguma dizer. Eu acho que é isso, Lucas. Você tem mais alguma pergunta, alguma coisa que você quer conversar?

**Lucas Galvão:** Deixa eu ver. Não, conversar sempre bom, né? Deixa eu pensar alguma

**Walter Silva Oliveira:** Bom,

**Lucas Galvão:** coisa.

**Walter Silva Oliveira:** agora voltamos vem semana que vem eu tô eu tô aqui quase todo

**Lucas Galvão:** É. Ah, tem que aparecer lá para ver a

**Walter Silva Oliveira:** dia.

**Lucas Galvão:** calorada. É, mas agora você não tá dando mais a para calor, né? Só para quatro

**Walter Silva Oliveira:** Não, não, só quarto e sexto.

**Lucas Galvão:** semestre.

**Walter Silva Oliveira:** Problema é que a aula da noite é ruim agora porque é uma aula sua, é uma aula só. Então é tipo, papum, rapidão. Então nem dá. Quando são quatro aulas seguidas, eu dá para deixar um pouquinhos mais livre, mas quando é duas é

**Lucas Galvão:** Nossa, então eu gosto bem mais quando é aula dupla, porque tipo assim,

**Walter Silva Oliveira:** corrido.

**Lucas Galvão:** porque dá umas 9 da noite, o professor fala assim: "Ah, cansei, já podem ir embora". Agora, quando é tipo duas aulas seguidas de dois professores diferentes, o professor sempre dura até às 10\.

### **00:25:52** {#00:25:52}

**Walter Silva Oliveira:** né? Que eu tenho ficar mais curto que até vocês chegarem já é 9 horas e tem esse

**Lucas Galvão:** Ah,

**Walter Silva Oliveira:** ponto.

**Lucas Galvão:** nada a ver.

**Walter Silva Oliveira:** Aí você falou

**Lucas Galvão:** Eu não chego tarde. Eu não chego. Ah, sim. Eu só chego tarde quando o quando tá chovendo,

**Walter Silva Oliveira:** não.

**Lucas Galvão:** né? Daí não tem como. Aí é culpa culpa do trânsito.

**Walter Silva Oliveira:** Mas você mora onde?

**Lucas Galvão:** Hã,

**Walter Silva Oliveira:** Você mora onde?

**Lucas Galvão:** eu moro em Santos, só que às vezes eu trabalho em São Paulo e quando eu trabalho em São Paulo eu desço de

**Walter Silva Oliveira:** Ah,

**Lucas Galvão:** fretado,

**Walter Silva Oliveira:** tá.

**Lucas Galvão:** entendeu?

**Walter Silva Oliveira:** Fado da

**Lucas Galvão:** Eu pego no shopping é o

**Walter Silva Oliveira:** onde?

**Lucas Galvão:** Dourado,

**Walter Silva Oliveira:** Ah,

**Lucas Galvão:** que é uns 60 km,

**Walter Silva Oliveira:** sei, sei, sei. Já

**Lucas Galvão:** só que é muito trânsito. E a sua aula era bem de quinta,

**Walter Silva Oliveira:** é.

**Lucas Galvão:** quinta é o pior dia da semana.

**Walter Silva Oliveira:** Sei. Tô de olho.

**Lucas Galvão:** É sério?

### **00:26:48**

**Walter Silva Oliveira:** Tô de oi em você.

**Lucas Galvão:** O quê? É sério?

**Walter Silva Oliveira:** Tô de oi. Você ia na minha aula.

**Lucas Galvão:** Por quê? O quê?

**Walter Silva Oliveira:** Pode ser. Mas você ia na minha aula, tá tudo

**Lucas Galvão:** Sim, exatamente isso que eu tô falando, pô.

**Walter Silva Oliveira:** certo.

**Lucas Galvão:** Deixa eu ver. É, então, porque esse semestre, nossa, esse semestre eu quero ir o menos possível pra faculdade,

**Walter Silva Oliveira:** E você vai sentir falta

**Lucas Galvão:** cara.

**Walter Silva Oliveira:** depois.

**Lucas Galvão:** Vou pior que sabe o que eu fico pensando? Eh, queria ver se consigo manter o contato aí com os meus colegas mais próximos aí, tipo o Lucas e o Felipe e o João. Mas eu acho bem improvável. A vez o Lucas vai, mas os outros acho bem improvável.

**Walter Silva Oliveira:** Por que você diz isso?

**Lucas Galvão:** Ah, porque que parece porque eu não converso com eles em outra situação sem ser na faculdade. Что?

**Walter Silva Oliveira:** É assim, eu não pessoal que eu fiz a graduação poucos que a gente mantenho toda hora assim, mas pelo menos todo ano a gente se vê. Todo ano chega agosto que tem aniversário de uns quatro, aí a gente marca alguma coisa para comemorar e todo ano a gente se encontra.

### **00:28:07**

**Lucas Galvão:** Ah, entendi. Então,

**Walter Silva Oliveira:** Mas ficar todo dia junto assim tem problema.

**Lucas Galvão:** ah, ah, mas ah,

**Walter Silva Oliveira:** Não pode perder o vínculo de vez.

**Lucas Galvão:** mas por exemplo,

**Walter Silva Oliveira:** É normal. Vida

**Lucas Galvão:** é sim,

**Walter Silva Oliveira:** adulta.

**Lucas Galvão:** cara. Acredita que meus amigos, eles são meus amigos desde, sei lá,

**Walter Silva Oliveira:** 3

**Lucas Galvão:** maior parte, não,

**Walter Silva Oliveira:** anos.

**Lucas Galvão:** 2012, assim, são os mesmos amigos que eu tenho, as mesmas pessoas que eu saio até hoje assim. Então, tipo assim, daí eu penso, é, mas aí não tenho amigos novos, entendeu? Por exemplo, daí, por exemplo, é muito conveniente ter os mesmos grupos de amigos, porque você continua indo nos mesmos eventos

**Walter Silva Oliveira:** É,

**Lucas Galvão:** assim.

**Walter Silva Oliveira:** aí você tem que se forçar a manter a amizade, que manter a amizade é que dá trabalho,

**Lucas Galvão:** É muito difícil.

**Walter Silva Oliveira:** né?

**Lucas Galvão:** É, então acho que é isso. Tem alguma coisa?

**Walter Silva Oliveira:** Eu

**Lucas Galvão:** Tá vendo aqui a cidade que eu tô? Ó, aqui, ó. Tá vendo? Se você fosse paraa indústria, ó, você estaria aqui, ó. Dubai. Olha aqui, ó, a visão bonita.

**Walter Silva Oliveira:** tô bem aqui de livro para mim tá bom.

**Lucas Galvão:** Uhum.

**Walter Silva Oliveira:** Mas eu acho que é isso, Lucas.

**Lucas Galvão:** Tá bom.

**Walter Silva Oliveira:** Acho que tá tá indo bem.

**Lucas Galvão:** Beleza.

**Walter Silva Oliveira:** Qualquer coisa você me avisa.

**Lucas Galvão:** Vamos jogar uma aranha lá.

**Walter Silva Oliveira:** Não prometo, mas podemos jogar um aranzinho.

**Lucas Galvão:** Bora então. É isso. Valeu,

**Walter Silva Oliveira:** Cachorro.

**Lucas Galvão:** professor.

**Walter Silva Oliveira:** Vai loucas. Qualqu coisa você me avisa, hein?

**Lucas Galvão:** Falou. Beleza, vou progredir com o que você falou e vou progredir também com a

**Walter Silva Oliveira:** Valeu.

**Lucas Galvão:** eh eu acho que eu já vou começar a fazer uns testes de benchmark, já vou montar pelo menos os negócios.

**Walter Silva Oliveira:** Uhum. Vou já volta.

**Lucas Galvão:** Beleza,

**Walter Silva Oliveira:** Fechou.

**Lucas Galvão:** valeu,

**Walter Silva Oliveira:** Valeu,

### **A transcrição foi encerrada após 00:30:07**

*Esta transcrição editável foi gerada por computador e pode conter erros. As pessoas também podem alterar o texto depois que ele for criado.*

