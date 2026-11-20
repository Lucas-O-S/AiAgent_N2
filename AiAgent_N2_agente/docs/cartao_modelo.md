# Cartão do modelo — Previsão de churn (Telco Customer Churn)

## Identificação do modelo
- Modelo: Naive Bayes Gaussiano (`GaussianNB`) dentro de um `Pipeline` do scikit-learn.
- Arquivo: `Modelos/modelo_tabular.joblib`, salvo no Colab ao final do notebook `Telco_Customer_Churn.ipynb` (repositório da avaliação anterior), com scikit-learn 1.6.1. O script `scripts/avaliar_modelo.py` confere que ele reproduz a matriz de confusão do notebook.
- Treinado na avaliação anterior (N2). O agente apenas carrega o arquivo; não há retreino dentro do agente.
- Pré-processamento salvo junto no pipeline: atributos derivados (`serviceQuantity` = número de serviços adicionais contratados; `fibra_sem_suporte` = fibra óptica sem suporte técnico), `StandardScaler` nas numéricas e `OneHotEncoder(handle_unknown='ignore')` nas categóricas.

## Classes previstas
- Classe 1 = Cancela (Churn = Yes).
- Classe 0 = Não cancela (Churn = No).
- Saída do agente: classe prevista (limiar padrão de 0,5 do `predict`) e probabilidade de cancelar (`predict_proba`).

## Dados de treino e teste
- Base: 7043 clientes, 19 atributos de entrada (o `customerID` é descartado).
- Divisão estratificada 80/20 com `random_state=42`: 5634 clientes no treino e 1409 no teste.
- Taxa de churn: 26,54% (desbalanceamento moderado), igual no treino e no teste.

## Métricas no conjunto de teste
- Acurácia: 0,6934
- Precisão (classe Cancela): 0,4572
- Recall (classe Cancela): 0,8289
- F1 (classe Cancela): 0,5894
- ROC-AUC: 0,8086
- Matriz de confusão: TN = 667, FP = 368, FN = 64, TP = 310.
- Leitura: de cada 100 clientes que realmente cancelam, o modelo identifica cerca de 83 (recall alto). Em compensação, menos da metade dos clientes apontados como "Cancela" realmente cancela (precisão de 0,4572): há muitos falsos positivos.

## Por que este modelo foi escolhido
- O grupo atribuiu custo de R$ 10.000 a um falso negativo (cliente que cancela sem ser identificado) e R$ 100 a um falso positivo (ação de retenção oferecida a quem não ia cancelar).
- Com esses custos, o Naive Bayes teve o menor custo total no teste entre os 6 modelos treinados com todos os atributos: R$ 676.800.
- A Regressão Logística e a Árvore de Decisão têm F1 maior, mas deixam passar mais clientes que cancelam.

## Comparação com os outros modelos (teste, todos os atributos)
| Modelo | Acurácia | Precisão | Recall | F1 | FN | FP | Custo |
|---|---|---|---|---|---|---|---|
| Naive Bayes | 0,6934 | 0,4572 | 0,8289 | 0,5894 | 64 | 368 | R$ 676.800 |
| Regressão Logística | 0,7410 | 0,5079 | 0,7754 | 0,6138 | 84 | 281 | R$ 868.100 |
| Árvore de Decisão | 0,7551 | 0,5270 | 0,7567 | 0,6213 | 91 | 254 | R$ 935.400 |
| KNN | 0,7913 | 0,6198 | 0,5535 | 0,5847 | 167 | 127 | R$ 1.682.700 |
| MLP (rede neural) | 0,7949 | 0,6341 | 0,5374 | 0,5818 | 173 | 116 | R$ 1.741.600 |
| Perceptron | 0,7182 | 0,4716 | 0,5107 | 0,4904 | 183 | 214 | R$ 1.851.400 |

## Limitações do modelo
- Precisão baixa (0,4572): muitos alarmes falsos. A previsão "Cancela" deve ser lida como "cliente de risco, vale uma ação de retenção barata", não como certeza.
- O Naive Bayes supõe atributos independentes entre si, o que não é verdade aqui (por exemplo, tenure e TotalCharges têm correlação de 0,826). Por isso as probabilidades tendem a ficar extremas (muito perto de 0 ou de 1) e não são bem calibradas: uma probabilidade de 99% não significa 99% de chance real.
- A escolha do modelo depende dos custos supostos (R$ 10.000 por FN e R$ 100 por FP). Com outra relação de custos o melhor modelo pode ser outro.
- Clientes buscados por customerID podem ter sido usados no treino (80% da base). Para esses, o acerto tende a ser otimista; o agente informa se o cliente estava no treino ou no teste.
- Valores fora das faixas da base (por exemplo, tenure acima de 72 meses ou cobrança mensal acima de 118,75) são recusados pela validação, porque o modelo nunca viu esses casos.
- Base de uma única operadora, sem dimensão de tempo: não captura mudanças de preço, concorrência ou sazonalidade.

## Interpretação das previsões
- O Naive Bayes não fornece uma explicação simples por atributo. O agente mostra, ao lado da previsão, as taxas de churn históricas das categorias do cliente na base (por exemplo, a taxa de churn de quem tem contrato mensal).
- Essas taxas são associações observadas nos dados, não causas, e não são a conta interna do modelo.

## Riscos do agente (LLM)
- O LLM pode errar a rota ou extrair um campo errado do texto. Por isso a validação é determinística e o agente mostra os dados usados.
- O LLM nunca calcula a previsão: classe e probabilidade vêm da ferramenta e são escritas por código, não pelo LLM.
- Uma verificação automática descarta a resposta do LLM se ela citar números que não estão nos resultados das ferramentas nem neste cartão.
- O agente não analisa imagens: o grupo decidiu trabalhar só com a base tabular Telco.
