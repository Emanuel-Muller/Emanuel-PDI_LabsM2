# Mini relatório — Laboratório M1, M2.1 e M2.2

## Identificação

- **Estudante:** Emanuel Müller Bueno
- **Matrícula:** 8113408
- **Laboratórios:** M1-1, M1-2, M1-3, M2.1 e M2.2
- **Linguagem:** Python

---

# 1. Objetivo

O objetivo dos laboratórios é aplicar os conceitos de processamento digital de imagens estudados em aula por meio da implementação de operações em Python. As atividades envolvem a leitura e análise de imagens, transformações de intensidade, manipulação de canais de cor, aplicação de filtros, segmentação e análise de componentes conectados.

Além de implementar as operações, busca-se compreender como os parâmetros utilizados influenciam os resultados, avaliar as diferenças entre os métodos e identificar as limitações encontradas durante o processamento de imagens reais.

# 2. Desenvolvimento — M1

## 2.1. Leitura e análise de imagens

A primeira etapa consiste em carregar uma imagem e obter informações sobre suas características, como dimensões, número de canais e tipo de representação dos pixels.

A operação `inspect` permite consultar essas informações sem modificar a imagem original. A operação `copy` permite gerar uma cópia da imagem, preservando seus dados.

Essas operações são importantes para verificar se a imagem foi carregada corretamente e para compreender seu formato antes de executar as demais transformações.

## 2.2. Separação de canais

As operações `channel_b`, `channel_g` e `channel_r` permitem separar os canais azul, verde e vermelho de uma imagem colorida.

Na representação BGR utilizada pelo OpenCV, a ordem dos canais é azul, verde e vermelho. A separação permite observar a contribuição de cada canal para a imagem e verificar como diferentes intensidades podem destacar ou ocultar determinadas regiões.

## 2.3. Conversão para tons de cinza

Foram implementadas duas abordagens para conversão de imagens coloridas para tons de cinza: média simples e média ponderada.

A operação `grayscale_average` utiliza a média aritmética dos três canais:

\[
I_{\text{cinza}}=\frac{R+G+B}{3}
\]

Já a operação `grayscale_weighted` utiliza pesos diferentes para cada canal, considerando a contribuição relativa das cores para a percepção de luminosidade:

\[
I_{\text{cinza}}=0{,}299R+0{,}587G+0{,}114B
\]

A média simples atribui a mesma importância a todos os canais, enquanto a média ponderada considera a maior contribuição do verde para a luminosidade percebida. Por isso, os dois métodos podem produzir resultados visualmente diferentes.

## 2.4. Quantização

A operação `quantize` reduz a quantidade de níveis de intensidade presentes na imagem.

Com menos níveis disponíveis, diferentes valores de pixels são agrupados em uma quantidade menor de tonalidades. Isso facilita a observação dos efeitos da redução de informação e pode produzir regiões com transições mais evidentes entre intensidades.

Entretanto, a quantização excessiva pode eliminar detalhes e produzir contornos artificiais.

## 2.5. Ajuste de brilho e contraste

A operação `brightness` modifica a intensidade dos pixels, permitindo clarear ou escurecer a imagem.

A operação `contrast` modifica a diferença entre as intensidades dos pixels, tornando as regiões claras e escuras mais ou menos distintas.

Essas operações são úteis para observar como as transformações de intensidade afetam a aparência da imagem. É importante considerar que valores muito elevados ou muito baixos podem causar saturação e perda de detalhes.

## 2.6. Negativo

A operação `negative` inverte as intensidades dos pixels. Para uma imagem em escala de cinza com valores entre 0 e 255, a transformação pode ser representada por:

\[
I_{\text{negativo}}=255-I
\]

Assim, regiões claras tornam-se escuras e regiões escuras tornam-se claras. Essa transformação permite visualizar informações de uma maneira diferente, embora não acrescente detalhes que não estavam presentes na imagem original.

## 2.7. Limiarização

A operação `threshold` compara a intensidade de cada pixel com um valor de limiar. Dependendo dessa comparação, o pixel é classificado como pertencente ao primeiro plano ou ao fundo.

Uma forma comum de representar a operação é:

\[
I'(x,y)=
\begin{cases}
255, & I(x,y)\geq T\\
0, & I(x,y)<T
\end{cases}
\]

em que \(T\) representa o limiar utilizado.

O resultado é uma imagem binária, na qual os pixels assumem dois valores. A qualidade da segmentação depende da escolha do limiar e da diferença de intensidade entre os objetos e o fundo.

## 2.8. Histograma

A operação `histogram` calcula a distribuição das intensidades dos pixels de uma imagem.

O histograma permite observar a frequência com que cada nível de intensidade aparece. Concentrações em determinadas regiões podem indicar predominância de áreas escuras ou claras, enquanto distribuições mais espalhadas podem representar maior variação tonal.

Essa análise auxilia na compreensão da imagem e na escolha de parâmetros para operações como limiarização e ajuste de contraste.

## 2.9. Convolução e filtros

A operação `convolution` aplica uma máscara ou kernel sobre a imagem para produzir uma nova representação dos pixels. O resultado depende dos valores presentes no kernel e da maneira como ele é aplicado.

Foram consideradas as seguintes operações de filtragem:

- **`mean_filter`:** utiliza a média dos valores em uma vizinhança para suavizar variações locais e reduzir ruídos.
- **`weighted_mean`:** aplica uma média ponderada, atribuindo importâncias diferentes aos pixels da vizinhança.
- **`laplacian`:** destaca variações de intensidade e pode ser utilizado para evidenciar bordas e detalhes.
- **`sobel`:** estima variações de intensidade nas direções horizontal e vertical, auxiliando na detecção de bordas.

Os filtros de suavização tendem a reduzir pequenas variações, mas também podem remover detalhes. Já os filtros de realce de bordas destacam mudanças bruscas de intensidade, embora possam responder a ruídos presentes na imagem.

# 3. Testes e resultados — M1

Os testes das operações M1 foram utilizados para verificar o funcionamento das transformações e observar suas consequências visuais.

| Operação | Resultado esperado e observado |
|---|---|
| `inspect` | Obtenção das informações da imagem carregada. |
| `copy` | Geração de uma cópia da imagem original. |
| `channel_b`, `channel_g`, `channel_r` | Separação dos canais de cor. |
| `grayscale_average` | Conversão para tons de cinza utilizando média simples. |
| `grayscale_weighted` | Conversão para tons de cinza utilizando pesos por canal. |
| `quantize` | Redução da quantidade de níveis de intensidade. |
| `brightness` | Clareamento ou escurecimento da imagem. |
| `contrast` | Modificação da diferença entre intensidades. |
| `negative` | Inversão das intensidades dos pixels. |
| `threshold` | Produção de uma imagem binária por limiarização. |
| `histogram` | Análise da distribuição de intensidades. |
| `convolution` | Aplicação de um kernel à imagem. |
| `mean_filter` | Suavização da imagem por média local. |
| `weighted_mean` | Suavização com pesos diferentes na vizinhança. |
| `laplacian` | Realce de variações locais de intensidade. |
| `sobel` | Detecção de bordas por gradientes. |

Os resultados permitem observar que cada operação modifica uma característica específica da imagem. A escolha da transformação adequada depende do objetivo do processamento e das propriedades da imagem de entrada.

# 4. Segmentação de imagens — M2.1

## 4.1. Objetivo

O objetivo da etapa M2.1 é separar objetos de interesse do fundo, produzindo máscaras que representem as regiões identificadas. Essa segmentação serve como preparação para as operações de análise de componentes conectados e morfologia da etapa M2.2.

Uma segmentação adequada deve preservar os objetos de interesse, reduzir a presença de regiões indevidas e, quando possível, manter os contornos dos objetos separados do fundo.

## 4.2. Imagens utilizadas

Foram utilizadas duas fotografias reais, denominadas `moedas1` e `moedas2`. Cada fotografia contém quatro moedas.

As imagens apresentam condições de iluminação, cores e contraste diferentes. Essas características influenciam a dificuldade de separar as moedas do fundo por meio de limiarização.

- **`moedas1`:** contém moedas pretas e amarelas sobre um fundo predominantemente amarelo. A semelhança de cor entre as moedas amarelas e o fundo dificulta a separação dessas regiões.
- **`moedas2`:** apresenta um fundo escuro, no qual somente duas moedas são claramente distinguíveis. As outras duas possuem pouco contraste e poucos detalhes observáveis.

## 4.3. Métodos de segmentação

A segmentação transforma a imagem original em uma máscara na qual os pixels são classificados em duas categorias: primeiro plano e fundo.

A limiarização depende da escolha de um valor de intensidade que separe as regiões. Métodos automáticos, como o limiar de Otsu, procuram determinar um limiar com base na distribuição dos níveis de intensidade. Métodos adaptativos calculam limiares de acordo com regiões locais da imagem, podendo ser úteis quando a iluminação não é uniforme.

A segmentação também pode considerar informações de cor, como os componentes de um espaço de cores. Independentemente do método escolhido, o resultado depende das características da imagem e da distinção entre os objetos e o fundo.

## 4.4. Análise visual

Na imagem `moedas1`, a presença de moedas amarelas sobre um fundo amarelo dificulta a determinação de limites confiáveis entre os objetos e o fundo. As moedas pretas apresentam diferença de cor mais evidente, mas a segmentação ainda pode produzir regiões conectadas ou fragmentadas.

Na imagem `moedas2`, o fundo escuro e a baixa visibilidade de duas moedas prejudicam a identificação de seus contornos. Quando os pixels dos objetos apresentam intensidades semelhantes às do fundo, a segmentação pode deixar partes das moedas de fora da máscara ou não identificá-las adequadamente.

Os resultados das máscaras binárias utilizadas na etapa seguinte confirmam que a segmentação apresenta limitações importantes, principalmente na primeira imagem, em que quase toda a área foi classificada como primeiro plano.

## 4.5. Discussão

A segmentação depende diretamente das características das imagens de entrada. Uma diferença significativa entre a intensidade ou a cor dos objetos e a do fundo facilita a classificação dos pixels. Quando existe semelhança entre essas regiões, um único limiar pode não ser suficiente para separar corretamente os objetos.

Além disso, a iluminação, as sombras e a ausência de detalhes visíveis podem prejudicar os contornos das moedas. Por esse motivo, a qualidade da máscara deve ser verificada visualmente antes de utilizar seus componentes para estimar a quantidade de objetos presentes.

Neste experimento, a segmentação não produziu máscaras suficientemente confiáveis para garantir a identificação automática das quatro moedas em ambas as fotografias.

# 5. Componentes conectados e morfologia — M2.2

## 5.1. Objetivo

A etapa M2.2 tem como objetivo identificar e analisar componentes conectados em imagens binárias, além de aplicar operações morfológicas para modificar regiões do primeiro plano.

A análise de componentes conectados permite identificar grupos de pixels que estão ligados entre si. Para cada componente, podem ser calculadas informações como área, posição, largura, altura e centroide.

As operações morfológicas permitem remover pequenas regiões, modificar contornos e preencher pequenas interrupções, dependendo da estrutura dos objetos e do elemento estruturante utilizado.

## 5.2. Conectividade

Foram consideradas conectividade 4 e conectividade 8.

Na conectividade 4, um pixel é considerado conectado aos pixels vizinhos nas direções horizontal e vertical. Na conectividade 8, também são considerados os vizinhos diagonais.

A conectividade escolhida pode alterar o número de componentes identificados. Regiões que estão ligadas somente pelos cantos podem ser consideradas separadas na conectividade 4, mas conectadas na conectividade 8.

Nas máscaras analisadas, os resultados iniciais foram:

| Imagem | Componentes com conectividade 4 | Componentes com conectividade 8 |
|---|---:|---:|
| `moedas1` | 1.677 | 966 |
| `moedas2` | 280 | 229 |

A redução no número de componentes ao utilizar conectividade 8 indica que algumas regiões estavam ligadas diagonalmente. Esses números representam componentes encontrados nas máscaras binárias, não a quantidade de moedas presentes nas fotografias.

## 5.3. Operações morfológicas

Foram aplicadas duas operações morfológicas em sequência: abertura e fechamento.

A **abertura** consiste em uma erosão seguida de uma dilatação. Ela pode remover pequenas regiões isoladas e reduzir detalhes que não sejam preservados pelo elemento estruturante.

O **fechamento** consiste em uma dilatação seguida de uma erosão. Essa operação pode conectar pequenas interrupções e preencher espaços estreitos dentro das regiões do primeiro plano.

Neste experimento, foi utilizado um elemento estruturante elíptico de tamanho 3 × 3, com uma iteração para cada operação. A área mínima configurada para a filtragem dos componentes foi de 20 pixels.

Essas operações modificam a máscara binária, mas não garantem que objetos que se tocam sejam separados ou que regiões de fundo classificadas incorretamente sejam corrigidas.

## 5.4. Resultados quantitativos

As imagens analisadas possuem resolução de 1200 × 1600 pixels, totalizando 1.920.000 pixels por imagem.

A tabela apresenta os resultados registrados pelo programa antes e depois da aplicação das operações morfológicas.

| Métrica | `moedas1` inicial | `moedas1` final | `moedas2` inicial | `moedas2` final |
|---|---:|---:|---:|---:|
| Componentes brutos | 966 | 355 | 229 | 109 |
| Componentes com área mínima de 20 pixels | 151 | 65 | 92 | 73 |
| Pixels do primeiro plano | 1.779.914 | 1.773.866 | 50.383 | 49.352 |
| Área total dos componentes mantidos | 1.776.065 | 1.771.565 | 49.811 | 49.009 |
| Proporção do primeiro plano | 92,70% | 92,39% | 2,62% | 2,57% |

A quantidade de pixels do primeiro plano representa a soma dos pixels classificados como pertencentes aos objetos na máscara binária. A proporção foi calculada dividindo essa quantidade pelo total de pixels da imagem.

Na imagem `moedas1`, o primeiro plano ocupa aproximadamente 92,39% da máscara final. Esse valor é muito elevado para uma máscara que deveria isolar somente quatro moedas e indica que uma grande parte do fundo foi classificada como primeiro plano.

Na imagem `moedas2`, o primeiro plano ocupa aproximadamente 2,57% da máscara final. Embora a proporção seja muito menor, isso não significa que as quatro moedas tenham sido identificadas corretamente, pois a máscara pode ter preservado somente partes dos objetos ou deixado regiões fora da segmentação.

## 5.5. Análise da imagem `moedas1`

A análise inicial identificou 966 componentes com conectividade 8. Após as operações morfológicas, foram identificados 355 componentes brutos, dos quais 65 permaneceram após a filtragem por área mínima.

Apesar da redução no número de componentes, a máscara final ainda apresenta um problema importante. O maior componente possui área de 1.763.622 pixels e ocupa a imagem inteira em sua caixa delimitadora, com largura de 1200 pixels e altura de 1600 pixels.

Esse resultado indica que uma região muito grande, incluindo áreas do fundo, foi agrupada em um único componente. Portanto, os componentes identificados não representam individualmente as quatro moedas.

Visualmente, é possível distinguir as quatro moedas na fotografia original. Entretanto, duas são pretas e duas são amarelas, em um fundo também amarelo. A semelhança entre as moedas amarelas e o fundo dificulta a separação dos contornos, enquanto a segmentação inadequada pode conectar regiões que deveriam permanecer separadas.

Assim, embora as operações morfológicas tenham reduzido o número de componentes detectados, elas não corrigiram o problema principal da máscara: a classificação incorreta de uma grande área da imagem como primeiro plano.

## 5.6. Análise da imagem `moedas2`

A análise inicial identificou 229 componentes com conectividade 8. Após as operações morfológicas, foram identificados 109 componentes brutos, dos quais 73 permaneceram após a filtragem por área mínima.

A área classificada como primeiro plano passou de 50.383 para 49.352 pixels. A proporção correspondente diminuiu de aproximadamente 2,62% para 2,57% da imagem.

Os maiores componentes finais apresentam áreas de 24.897, 8.886, 3.487 e 1.438 pixels. Entretanto, não é possível afirmar apenas com esses valores que cada componente corresponda a uma moeda. Para confirmar essa correspondência, seria necessário verificar visualmente a máscara e os contornos detectados.

Na fotografia original, somente duas moedas são claramente distinguíveis, enquanto as outras duas apresentam poucos detalhes em relação ao fundo escuro. Essa baixa diferença visual prejudica a segmentação e pode fazer com que partes das moedas não sejam classificadas como primeiro plano.

As operações morfológicas reduziram o número de componentes, mas a redução não comprova que os objetos tenham sido corretamente separados. A segmentação inicial continua sendo um fator determinante para a qualidade da análise.

## 5.7. Discussão dos resultados

Os resultados mostram que a análise de componentes conectados depende da qualidade da máscara binária produzida na etapa de segmentação. O algoritmo identifica regiões de pixels conectados, mas não reconhece semanticamente quais regiões correspondem a moedas.

Na imagem `moedas1`, o primeiro plano ocupa quase toda a imagem, o que demonstra que o fundo não foi separado adequadamente. Na imagem `moedas2`, a proporção de primeiro plano é muito menor, mas a baixa visibilidade de duas moedas impede concluir que todos os objetos foram segmentados corretamente.

A abertura e o fechamento alteraram a quantidade de componentes e a área total do primeiro plano. No entanto, essas operações não são suficientes para corrigir todos os problemas de contraste ou separar regiões que já estão conectadas de maneira inadequada.

Portanto, os números apresentados devem ser interpretados como propriedades das máscaras binárias, e não como uma contagem confirmada de moedas. Para obter uma contagem confiável, seria necessário melhorar a segmentação e verificar se cada moeda está representada por uma região distinta na máscara final.

# 6. Análise técnica geral

Os experimentos demonstram que as operações de processamento de imagens produzem resultados diferentes de acordo com seus parâmetros e com as características da imagem de entrada.

As transformações de intensidade, como brilho, contraste e negativo, modificam a representação visual dos pixels. Os filtros de suavização reduzem variações locais, enquanto os filtros de gradiente e Laplaciano destacam mudanças de intensidade.

A segmentação apresenta uma dificuldade adicional, pois depende da separação entre objetos e fundo. Uma máscara binária incorreta compromete as etapas seguintes, incluindo a análise de componentes conectados e as operações morfológicas.

Os resultados de M2.2 demonstram essa dependência. Mesmo quando o número de componentes diminui após o processamento, isso não significa necessariamente que a imagem tenha sido segmentada de forma mais correta. É necessário considerar também a área dos componentes, sua localização, seus contornos e sua correspondência com os objetos da fotografia original.

Dessa forma, os experimentos reforçam a importância de avaliar os resultados tanto quantitativamente quanto visualmente.

# 7. Limitações

Durante os experimentos, foram identificadas as seguintes limitações:

- Os resultados de algumas operações dependem diretamente dos parâmetros utilizados.
- A conversão para tons de cinza pode eliminar informações de cor que seriam úteis para distinguir objetos.
- A limiarização pode classificar incorretamente regiões quando existe pouca diferença de intensidade entre o objeto e o fundo.
- Variações de iluminação, sombras e baixo contraste podem prejudicar a segmentação de imagens reais.
- A análise de componentes conectados depende da qualidade da máscara binária e pode identificar fragmentos, ruídos ou regiões do fundo como componentes.
- A conectividade 4 ou 8 pode alterar a quantidade de componentes detectados.
- As operações morfológicas podem remover detalhes relevantes ou modificar os contornos dos objetos.
- A redução do número de componentes não garante que a segmentação tenha melhorado.
- Os resultados de `moedas1` e `moedas2` não permitem confirmar a contagem automática das quatro moedas em cada fotografia.

Uma possível melhoria seria revisar os parâmetros de segmentação, avaliar a polaridade das máscaras, explorar métodos de segmentação que considerem informações de cor e iluminação e inspecionar visualmente as máscaras antes da análise de componentes.

# 8. Conclusão

Os laboratórios permitiram aplicar conceitos de processamento digital de imagens utilizando Python, abrangendo operações de intensidade, manipulação de canais, filtragem, segmentação, morfologia e análise de componentes conectados.

As operações implementadas possibilitam modificar imagens e analisar suas características, mas os resultados dependem da escolha dos parâmetros e das propriedades das imagens utilizadas.

Nos experimentos com as fotografias de moedas, foi possível observar que a semelhança entre objetos e fundo e a baixa visibilidade de determinadas regiões dificultam a segmentação. Consequentemente, a análise de componentes conectados não conseguiu garantir a identificação individual das quatro moedas em ambas as imagens.

Conclui-se que o processamento morfológico pode modificar e simplificar máscaras binárias, mas não substitui uma segmentação adequada. A confiabilidade da análise depende da qualidade das etapas anteriores e da validação dos resultados por meio de medidas quantitativas e inspeção visual.

# 9. Referências

- GONZALEZ, Rafael C.; WOODS, Richard E. *Digital Image Processing*. Pearson.
- Documentação oficial do OpenCV. Disponível em: https://docs.opencv.org/
- Documentação oficial do NumPy. Disponível em: https://numpy.org/doc/

