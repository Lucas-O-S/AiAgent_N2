# Constantes do esquema do dataset Telco Customer Churn: colunas, valores aceitos, faixas e sinônimos.

OPCOES_SIM_NAO = ["Yes", "No"]
OPCOES_SERVICO_INTERNET = ["Yes", "No", "No internet service"]

# Colunas na ordem em que o pipeline salvo espera receber
COLUNAS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]

# Coluna categórica -> valores aceitos no dataset
COLUNAS_CATEGORICAS = {
    "gender": ["Female", "Male"],
    "Partner": OPCOES_SIM_NAO,
    "Dependents": OPCOES_SIM_NAO,
    "PhoneService": OPCOES_SIM_NAO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": OPCOES_SERVICO_INTERNET,
    "OnlineBackup": OPCOES_SERVICO_INTERNET,
    "DeviceProtection": OPCOES_SERVICO_INTERNET,
    "TechSupport": OPCOES_SERVICO_INTERNET,
    "StreamingTV": OPCOES_SERVICO_INTERNET,
    "StreamingMovies": OPCOES_SERVICO_INTERNET,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": OPCOES_SIM_NAO,
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}

# Coluna numérica -> tipo e faixa observada no dataset de treino (fora dela = fora da distribuição)
FAIXAS_NUMERICAS = {
    "SeniorCitizen": {"tipo": int, "min": 0, "max": 1},
    "tenure": {"tipo": int, "min": 0, "max": 72},
    "MonthlyCharges": {"tipo": float, "min": 18.25, "max": 118.75},
    "TotalCharges": {"tipo": float, "min": 0.0, "max": 8684.8},
}

SERVICOS_INTERNET = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                     "TechSupport", "StreamingTV", "StreamingMovies"]

DESCRICAO_CAMPOS = {
    "gender": "sexo do cliente",
    "SeniorCitizen": "é idoso? (1 = sim, 0 = não)",
    "Partner": "tem cônjuge/parceiro(a)?",
    "Dependents": "tem dependentes?",
    "tenure": "meses como cliente (0 a 72)",
    "PhoneService": "tem telefone fixo?",
    "MultipleLines": "tem múltiplas linhas?",
    "InternetService": "tipo de internet",
    "OnlineSecurity": "segurança online",
    "OnlineBackup": "backup online",
    "DeviceProtection": "proteção de aparelho",
    "TechSupport": "suporte técnico",
    "StreamingTV": "streaming de TV",
    "StreamingMovies": "streaming de filmes",
    "Contract": "tipo de contrato",
    "PaperlessBilling": "fatura digital?",
    "PaymentMethod": "forma de pagamento",
    "MonthlyCharges": "cobrança mensal (18,25 a 118,75)",
    "TotalCharges": "cobrança acumulada (0 a 8684,80)",
}

# Respostas de sim/não em português (comparadas já sem acento e em minúsculas)
RESPOSTAS_SIM = {"sim", "s", "yes", "y", "true", "verdadeiro", "tem", "possui", "1"}
RESPOSTAS_NAO = {"nao", "n", "no", "false", "falso", "nao tem", "nao possui", "0"}

_SINONIMOS_SEM_INTERNET = {"sem internet": "No internet service",
                           "sem servico de internet": "No internet service"}

# Coluna -> {sinônimo em português (sem acento, minúsculo): valor canônico do dataset}
SINONIMOS = {
    "gender": {"masculino": "Male", "homem": "Male", "m": "Male",
               "feminino": "Female", "mulher": "Female", "f": "Female"},
    "MultipleLines": {"sem telefone": "No phone service", "sem servico de telefone": "No phone service"},
    "InternetService": {"fibra": "Fiber optic", "fibra optica": "Fiber optic", "fiber": "Fiber optic",
                        "sem internet": "No", "nenhum": "No", "nenhuma": "No"},
    "Contract": {"mensal": "Month-to-month", "mes a mes": "Month-to-month",
                 "month to month": "Month-to-month",
                 "anual": "One year", "1 ano": "One year", "um ano": "One year",
                 "2 anos": "Two year", "dois anos": "Two year", "bienal": "Two year"},
    "PaymentMethod": {"cheque eletronico": "Electronic check", "boleto": "Electronic check",
                      "cheque pelo correio": "Mailed check", "cheque enviado pelo correio": "Mailed check",
                      "transferencia": "Bank transfer (automatic)",
                      "transferencia bancaria": "Bank transfer (automatic)",
                      "debito automatico": "Bank transfer (automatic)",
                      "cartao": "Credit card (automatic)", "cartao de credito": "Credit card (automatic)"},
    **{colunaServico: _SINONIMOS_SEM_INTERNET for colunaServico in SERVICOS_INTERNET},
}
