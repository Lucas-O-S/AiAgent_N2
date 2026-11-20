"""Esquema do dataset Telco Customer Churn e validação das entradas do agente.

Tudo aqui é determinístico (sem LLM): o LLM só extrai os valores do texto; quem decide
se a entrada é válida é este módulo, com base nas colunas, tipos e faixas do dataset.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

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


def _normalizarTexto(texto: str) -> str:
    """Remove acentos, passa para minúsculas e junta espaços repetidos."""
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return " ".join(texto.lower().strip().split())


# Sinônimos em português -> valor canônico do dataset (comparação sem acento/maiúsculas)
_RESPOSTAS_SIM = {"sim", "s", "yes", "y", "true", "verdadeiro", "tem", "possui", "1"}
_RESPOSTAS_NAO = {"nao", "n", "no", "false", "falso", "nao tem", "nao possui", "0"}
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
for _colunaServico in SERVICOS_INTERNET:
    SINONIMOS[_colunaServico] = {"sem internet": "No internet service",
                                 "sem servico de internet": "No internet service"}


def normalizarValorCategorico(coluna: str, valor) -> str | None:
    """Converte o valor para o rótulo canônico do dataset; None se não reconhecer."""
    if valor is None:
        return None
    valoresPermitidos = COLUNAS_CATEGORICAS[coluna]
    valorNormalizado = _normalizarTexto(valor)
    for opcao in valoresPermitidos:
        if _normalizarTexto(opcao) == valorNormalizado:
            return opcao
    if valorNormalizado in SINONIMOS.get(coluna, {}):
        return SINONIMOS[coluna][valorNormalizado]
    if "Yes" in valoresPermitidos and valorNormalizado in _RESPOSTAS_SIM:
        return "Yes"
    if "No" in valoresPermitidos and valorNormalizado in _RESPOSTAS_NAO:
        return "No"
    return None


def _converterParaNumero(valor):
    """Aceita número, "sim/não" (1/0) e texto como "R$ 1.234,56"; None se não for numérico."""
    if isinstance(valor, bool):
        return None
    if isinstance(valor, (int, float)):
        return float(valor)
    texto = _normalizarTexto(valor)
    if texto in _RESPOSTAS_SIM:
        return 1.0
    if texto in _RESPOSTAS_NAO:
        return 0.0
    texto = texto.replace("r$", "").replace(" ", "")
    if "," in texto:  # formato brasileiro 1.234,56
        texto = texto.replace(".", "").replace(",", ".")
    try:
        return float(texto)
    except ValueError:
        return None


@dataclass
class ResultadoValidacao:
    valido: bool
    dadosNormalizados: dict = field(default_factory=dict)   # dados prontos para o modelo
    camposFaltando: list = field(default_factory=list)      # campos obrigatórios ausentes
    erros: list = field(default_factory=list)               # mensagens de erro (tipo/faixa/coerência)
    camposPreenchidosPorCoerencia: list = field(default_factory=list)


def validarCliente(dadosBrutos: dict) -> ResultadoValidacao:
    """Valida colunas, tipos, faixas e coerência entre campos."""
    dadosNormalizados, erros, camposFaltando, camposPreenchidos = {}, [], [], []
    valoresInformados = {coluna: valor for coluna, valor in (dadosBrutos or {}).items()
                         if valor not in (None, "")}

    # Preenchimento por coerência (regras do próprio dataset, não é imputação estatística)
    if normalizarValorCategorico("InternetService", valoresInformados.get("InternetService")) == "No":
        for colunaServico in SERVICOS_INTERNET:
            if colunaServico not in valoresInformados:
                valoresInformados[colunaServico] = "No internet service"
                camposPreenchidos.append(colunaServico)
    if (normalizarValorCategorico("PhoneService", valoresInformados.get("PhoneService")) == "No"
            and "MultipleLines" not in valoresInformados):
        valoresInformados["MultipleLines"] = "No phone service"
        camposPreenchidos.append("MultipleLines")

    for campo in COLUNAS:
        if campo not in valoresInformados:
            camposFaltando.append(campo)
            continue
        valor = valoresInformados[campo]
        if campo in COLUNAS_CATEGORICAS:
            valorNormalizado = normalizarValorCategorico(campo, valor)
            if valorNormalizado is None:
                erros.append(f"{campo} = '{valor}' não é válido. "
                             f"Valores aceitos: {', '.join(COLUNAS_CATEGORICAS[campo])}.")
            else:
                dadosNormalizados[campo] = valorNormalizado
        else:
            faixa = FAIXAS_NUMERICAS[campo]
            numero = _converterParaNumero(valor)
            if numero is None:
                erros.append(f"{campo} = '{valor}' deveria ser numérico.")
                continue
            if faixa["tipo"] is int and abs(numero - round(numero)) > 1e-9:
                erros.append(f"{campo} = {valor} deveria ser um número inteiro.")
                continue
            if not (faixa["min"] <= numero <= faixa["max"]):
                erros.append(f"{campo} = {valor} está fora da faixa do dataset "
                             f"({faixa['min']} a {faixa['max']}).")
                continue
            dadosNormalizados[campo] = int(round(numero)) if faixa["tipo"] is int else round(numero, 2)

    # Coerência entre campos
    tipoInternet = dadosNormalizados.get("InternetService")
    for colunaServico in SERVICOS_INTERNET:
        valorServico = dadosNormalizados.get(colunaServico)
        if tipoInternet == "No" and valorServico in ("Yes", "No"):
            erros.append(f"{colunaServico} = '{valorServico}' é incoerente com InternetService = 'No' "
                         "(deveria ser 'No internet service').")
        if tipoInternet in ("DSL", "Fiber optic") and valorServico == "No internet service":
            erros.append(f"{colunaServico} = 'No internet service' é incoerente com "
                         f"InternetService = '{tipoInternet}'.")
    servicoTelefone = dadosNormalizados.get("PhoneService")
    multiplasLinhas = dadosNormalizados.get("MultipleLines")
    if servicoTelefone == "No" and multiplasLinhas in ("Yes", "No"):
        erros.append("MultipleLines deveria ser 'No phone service' quando PhoneService = 'No'.")
    if servicoTelefone == "Yes" and multiplasLinhas == "No phone service":
        erros.append("MultipleLines = 'No phone service' é incoerente com PhoneService = 'Yes'.")
    meses = dadosNormalizados.get("tenure")
    cobrancaMensal = dadosNormalizados.get("MonthlyCharges")
    cobrancaTotal = dadosNormalizados.get("TotalCharges")
    if None not in (meses, cobrancaMensal, cobrancaTotal):
        if meses == 0 and cobrancaTotal > 0:
            erros.append("TotalCharges deveria ser 0 para cliente com tenure = 0 (sem fatura fechada).")
        elif meses > 0:
            razaoTotalPorEsperado = cobrancaTotal / (cobrancaMensal * meses)
            if not (0.5 <= razaoTotalPorEsperado <= 2.0):
                erros.append(f"TotalCharges ({cobrancaTotal}) é incoerente com tenure × MonthlyCharges "
                             f"({meses} × {cobrancaMensal} = {round(meses * cobrancaMensal, 2)}).")

    valido = not erros and not camposFaltando
    return ResultadoValidacao(valido, dadosNormalizados, camposFaltando, erros, camposPreenchidos)
