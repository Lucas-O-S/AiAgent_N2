# Dicionário de dados — Telco Customer Churn

## Visão geral da base
- 7043 clientes de uma operadora de telecomunicações, 21 colunas (customerID, 19 atributos e o alvo Churn).
- Sem valores ausentes nativos e sem linhas duplicadas.
- TotalCharges vinha como texto; 11 clientes com tenure = 0 (sem fatura fechada) ficaram vazios e foram preenchidos com 0.
- Alvo Churn: 73,46% não cancelaram e 26,54% cancelaram.

## Atributos de perfil do cliente
- gender: Female ou Male.
- SeniorCitizen: 1 se o cliente é idoso, 0 se não. Taxa de churn de idosos: 41,68%; de não idosos: 23,61%.
- Partner: Yes/No, tem cônjuge ou parceiro(a).
- Dependents: Yes/No, tem dependentes.

## Atributos de serviços contratados
- PhoneService: Yes/No, tem telefone.
- MultipleLines: Yes, No ou No phone service (quando não tem telefone).
- InternetService: DSL, Fiber optic ou No. Taxa de churn: fibra óptica 41,89%, DSL 18,96%, sem internet 7,40%.
- OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies: Yes, No ou No internet service (quando não tem internet).
- TechSupport: taxa de churn sem suporte técnico 41,64%, com suporte 15,17%.

## Atributos de contrato e pagamento
- Contract: Month-to-month, One year ou Two year. É o atributo que mais separa quem cancela. Taxa de churn: mensal 42,71%, um ano 11,27%, dois anos 2,83%.
- PaperlessBilling: Yes/No, fatura digital. Taxa de churn com fatura digital 33,57%, sem 16,33%.
- PaymentMethod: Electronic check, Mailed check, Bank transfer (automatic) ou Credit card (automatic). Taxa de churn com Electronic check 45,29%; nos demais métodos fica entre 15,24% e 19,11%.

## Atributos numéricos
- tenure: meses como cliente, de 0 a 72. Média de 17,98 meses entre quem cancelou e 37,57 entre quem ficou.
- MonthlyCharges: cobrança mensal, de 18,25 a 118,75. Média de 74,44 entre quem cancelou e 61,27 entre quem ficou.
- TotalCharges: cobrança acumulada, de 0 a 8684,80. Correlação com tenure de 0,826.

## Achado sobre fibra óptica e suporte técnico
- Clientes de fibra óptica cancelam mais. A hipótese de serem clientes mais novos foi descartada (tenure médio quase igual ao de DSL).
- O preço explica parte (fibra é mais cara), mas o fator mais forte é a falta de suporte técnico: entre clientes de fibra, o churn é 49,37% sem TechSupport e 22,63% com TechSupport.
- Por isso foi criado o atributo derivado fibra_sem_suporte.

## Regras de validação usadas pelo agente
- As 19 colunas de entrada são obrigatórias, com os valores categóricos exatamente como acima.
- Faixas numéricas: as mesmas observadas na base (tenure 0 a 72, MonthlyCharges 18,25 a 118,75, TotalCharges 0 a 8684,80).
- Coerência: sem internet implica "No internet service" nos 6 serviços de internet; sem telefone implica MultipleLines = "No phone service"; tenure = 0 implica TotalCharges = 0; TotalCharges precisa ser compatível com tenure × MonthlyCharges (razão entre 0,5 e 2).
- Quando InternetService = No ou PhoneService = No, o agente preenche os campos dependentes por coerência e avisa.
