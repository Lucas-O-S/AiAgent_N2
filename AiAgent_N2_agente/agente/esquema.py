"""Esquema do dataset Telco Customer Churn e validação das entradas do agente.

Tudo aqui é determinístico (sem LLM): o LLM só extrai os valores do texto; quem decide
se a entrada é válida é este módulo, com base nas colunas, tipos e faixas do dataset.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

SIM_NAO = ["Yes", "No"]
SERV_INTERNET = ["Yes", "No", "No internet service"]

# Colunas na ordem em que o pipeline salvo espera receber
COLUNAS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges",
]

CATEGORICAS = {
    "gender": ["Female", "Male"],
    "Partner": SIM_NAO,
    "Dependents": SIM_NAO,
    "PhoneService": SIM_NAO,
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": SERV_INTERNET,
    "OnlineBackup": SERV_INTERNET,
    "DeviceProtection": SERV_INTERNET,
    "TechSupport": SERV_INTERNET,
    "StreamingTV": SERV_INTERNET,
    "StreamingMovies": SERV_INTERNET,
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": SIM_NAO,
    "PaymentMethod": ["Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"],
}

# Faixas observadas no dataset de treino (valores fora delas = fora da distribuição)
NUMERICAS = {
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


def _sem_acento(txt: str) -> str:
    txt = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()
    return " ".join(txt.lower().strip().split())


# Sinônimos em português -> valor canônico do dataset (comparação sem acento/maiúsculas)
_SIM = {"sim", "s", "yes", "y", "true", "verdadeiro", "tem", "possui", "1"}
_NAO = {"nao", "n", "no", "false", "falso", "nao tem", "nao possui", "0"}
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
}
for _c in SERVICOS_INTERNET:
    SINONIMOS[_c] = {"sem internet": "No internet service", "sem servico de internet": "No internet service"}


def normalizar_categorica(campo: str, valor) -> str | None:
    """Converte o valor para o rótulo canônico do dataset; None se não reconhecer."""
    if valor is None:
        return None
    permitidos = CATEGORICAS[campo]
    chave = _sem_acento(valor)
    for p in permitidos:
        if _sem_acento(p) == chave:
            return p
    if chave in SINONIMOS.get(campo, {}):
        return SINONIMOS[campo][chave]
    if "Yes" in permitidos and chave in _SIM:
        return "Yes"
    if "No" in permitidos and chave in _NAO:
        return "No"
    return None


def _para_numero(valor):
    if isinstance(valor, bool):
        return None
    if isinstance(valor, (int, float)):
        return float(valor)
    txt = _sem_acento(valor)
    if txt in _SIM:
        return 1.0
    if txt in _NAO:
        return 0.0
    txt = txt.replace("r$", "").replace(" ", "")
    if "," in txt:  # formato brasileiro 1.234,56
        txt = txt.replace(".", "").replace(",", ".")
    try:
        return float(txt)
    except ValueError:
        return None


@dataclass
class ResultadoValidacao:
    valido: bool
    dados: dict = field(default_factory=dict)        # dados normalizados prontos para o modelo
    faltando: list = field(default_factory=list)      # campos obrigatórios ausentes
    erros: list = field(default_factory=list)         # mensagens de erro (tipo/faixa/coerência)
    preenchidos_por_coerencia: list = field(default_factory=list)


def validar_cliente(dados_brutos: dict) -> ResultadoValidacao:
    """Valida colunas, tipos, faixas e coerência entre campos."""
    dados, erros, faltando, auto = {}, [], [], []
    brutos = {k: v for k, v in (dados_brutos or {}).items() if v not in (None, "")}

    # Preenchimento por coerência (regras do próprio dataset, não é imputação estatística)
    if normalizar_categorica("InternetService", brutos.get("InternetService")) == "No":
        for c in SERVICOS_INTERNET:
            if c not in brutos:
                brutos[c] = "No internet service"
                auto.append(c)
    if normalizar_categorica("PhoneService", brutos.get("PhoneService")) == "No" and "MultipleLines" not in brutos:
        brutos["MultipleLines"] = "No phone service"
        auto.append("MultipleLines")

    for campo in COLUNAS:
        if campo not in brutos:
            faltando.append(campo)
            continue
        valor = brutos[campo]
        if campo in CATEGORICAS:
            norm = normalizar_categorica(campo, valor)
            if norm is None:
                erros.append(f"{campo} = '{valor}' não é válido. Valores aceitos: {', '.join(CATEGORICAS[campo])}.")
            else:
                dados[campo] = norm
        else:
            regra = NUMERICAS[campo]
            num = _para_numero(valor)
            if num is None:
                erros.append(f"{campo} = '{valor}' deveria ser numérico.")
                continue
            if regra["tipo"] is int and abs(num - round(num)) > 1e-9:
                erros.append(f"{campo} = {valor} deveria ser um número inteiro.")
                continue
            if not (regra["min"] <= num <= regra["max"]):
                erros.append(f"{campo} = {valor} está fora da faixa do dataset "
                             f"({regra['min']} a {regra['max']}).")
                continue
            dados[campo] = int(round(num)) if regra["tipo"] is int else round(num, 2)

    # Coerência entre campos
    internet = dados.get("InternetService")
    for c in SERVICOS_INTERNET:
        v = dados.get(c)
        if internet == "No" and v in ("Yes", "No"):
            erros.append(f"{c} = '{v}' é incoerente com InternetService = 'No' (deveria ser 'No internet service').")
        if internet in ("DSL", "Fiber optic") and v == "No internet service":
            erros.append(f"{c} = 'No internet service' é incoerente com InternetService = '{internet}'.")
    tel, linhas = dados.get("PhoneService"), dados.get("MultipleLines")
    if tel == "No" and linhas in ("Yes", "No"):
        erros.append("MultipleLines deveria ser 'No phone service' quando PhoneService = 'No'.")
    if tel == "Yes" and linhas == "No phone service":
        erros.append("MultipleLines = 'No phone service' é incoerente com PhoneService = 'Yes'.")
    t, m, tot = dados.get("tenure"), dados.get("MonthlyCharges"), dados.get("TotalCharges")
    if None not in (t, m, tot):
        if t == 0 and tot > 0:
            erros.append("TotalCharges deveria ser 0 para cliente com tenure = 0 (sem fatura fechada).")
        elif t > 0:
            razao = tot / (m * t)
            if not (0.5 <= razao <= 2.0):
                erros.append(f"TotalCharges ({tot}) é incoerente com tenure × MonthlyCharges "
                             f"({t} × {m} = {round(t * m, 2)}).")

    valido = not erros and not faltando
    return ResultadoValidacao(valido, dados, faltando, erros, auto)
