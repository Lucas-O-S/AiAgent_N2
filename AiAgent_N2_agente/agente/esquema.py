# Validação das entradas do agente com base no esquema do dataset Telco Customer Churn.
#
# Tudo aqui é determinístico (sem LLM): o LLM só extrai os valores do texto; quem decide
# se a entrada é válida é este módulo, com base nas colunas, tipos e faixas do dataset
# (definidos em esquema_constantes.py).
from __future__ import annotations

from dataclasses import dataclass, field

from .esquema_auxiliares import converterParaNumero, normalizarParaOpcoes
from .esquema_constantes import (COLUNAS, COLUNAS_CATEGORICAS, FAIXAS_NUMERICAS, OPCOES_SIM_NAO,
                                 SERVICOS_INTERNET, SINONIMOS)


@dataclass
class ResultadoValidacao:
    valido: bool
    dadosNormalizados: dict = field(default_factory=dict)   # dados prontos para o modelo
    camposFaltando: list = field(default_factory=list)      # campos obrigatórios ausentes
    erros: list = field(default_factory=list)               # mensagens de erro (tipo/faixa/coerência)
    camposPreenchidosPorCoerencia: list = field(default_factory=list)


# Converte o valor para o rótulo canônico da coluna no dataset; None se não reconhecer.
def normalizarValorCategorico(coluna: str, valor) -> str | None:
    return normalizarParaOpcoes(valor, COLUNAS_CATEGORICAS[coluna], SINONIMOS.get(coluna))


# Converte sim/não em português ou inglês para "Yes"/"No"; None se não reconhecer.
def normalizarSimNao(valor) -> str | None:
    return normalizarParaOpcoes(valor, OPCOES_SIM_NAO)


# Valida colunas, tipos, faixas e coerência entre campos.
def validarCliente(dadosBrutos: dict) -> ResultadoValidacao:
   
    valoresInformados = {coluna: valor for coluna, valor in (dadosBrutos or {}).items()
                         if valor not in (None, "")}
    
    camposPreenchidos = _preencherPorCoerencia(valoresInformados)

    dadosNormalizados, camposFaltando, erros = {}, [], []
    
    for campo in COLUNAS:
        
        if campo not in valoresInformados:
            camposFaltando.append(campo)
            continue
        
        valorNormalizado, erro = _validarCampo(campo, valoresInformados[campo])
        
        if erro:
            erros.append(erro)
        
        else:
            dadosNormalizados[campo] = valorNormalizado

    erros += _verificarCoerenciaEntreCampos(dadosNormalizados)
    
    valido = not erros and not camposFaltando
    
    return ResultadoValidacao(valido, dadosNormalizados, camposFaltando, erros, camposPreenchidos)


# Completa campos que só têm uma resposta possível (regras do próprio dataset, não é
# imputação estatística). Altera valoresInformados e devolve os campos preenchidos.
def _preencherPorCoerencia(valoresInformados: dict) -> list[str]:
    camposPreenchidos = []
    
    if normalizarValorCategorico("InternetService", valoresInformados.get("InternetService")) == "No":
    
        for colunaServico in SERVICOS_INTERNET:
    
            if colunaServico not in valoresInformados:
    
                valoresInformados[colunaServico] = "No internet service"
                camposPreenchidos.append(colunaServico)
    
    if (normalizarValorCategorico("PhoneService", valoresInformados.get("PhoneService")) == "No"
    
            and "MultipleLines" not in valoresInformados):
    
        valoresInformados["MultipleLines"] = "No phone service"
    
        camposPreenchidos.append("MultipleLines")
    
    if (
        normalizarValorCategorico("MultipleLines", valoresInformados.get("MultipleLines")) in ("Yes", "No")
        and not valoresInformados.get("PhoneService")
    ):
        valoresInformados["PhoneService"] = "Yes"
        camposPreenchidos.append("PhoneService")    
    
    return camposPreenchidos


# Devolve (valor normalizado, None) se o valor for válido, ou (None, mensagem de erro).
def _validarCampo(campo: str, valor):
    if campo in COLUNAS_CATEGORICAS:
        valorNormalizado = normalizarValorCategorico(campo, valor)
        if valorNormalizado is None:
            return None, (f"{campo} = '{valor}' não é válido. "
                          f"Valores aceitos: {', '.join(COLUNAS_CATEGORICAS[campo])}.")
        return valorNormalizado, None

    faixa = FAIXAS_NUMERICAS[campo]
    numero = converterParaNumero(valor)
    if numero is None:
        return None, f"{campo} = '{valor}' deveria ser numérico."
    if faixa["tipo"] is int and abs(numero - round(numero)) > 1e-9:
        return None, f"{campo} = {valor} deveria ser um número inteiro."
    if not (faixa["min"] <= numero <= faixa["max"]):
        return None, (f"{campo} = {valor} está fora da faixa do dataset "
                      f"({faixa['min']} a {faixa['max']}).")
    return (int(round(numero)) if faixa["tipo"] is int else round(numero, 2)), None


def _verificarCoerenciaEntreCampos(dados: dict) -> list[str]:
    erros = []

    # serviços de internet x tipo de internet
    tipoInternet = dados.get("InternetService")
    for colunaServico in SERVICOS_INTERNET:
        valorServico = dados.get(colunaServico)
        if tipoInternet == "No" and valorServico in ("Yes", "No"):
            erros.append(f"{colunaServico} = '{valorServico}' é incoerente com InternetService = 'No' "
                         "(deveria ser 'No internet service').")
        if tipoInternet in ("DSL", "Fiber optic") and valorServico == "No internet service":
            erros.append(f"{colunaServico} = 'No internet service' é incoerente com "
                         f"InternetService = '{tipoInternet}'.")

    # múltiplas linhas x telefone
    servicoTelefone, multiplasLinhas = dados.get("PhoneService"), dados.get("MultipleLines")
    if servicoTelefone == "No" and multiplasLinhas in ("Yes", "No"):
        erros.append("MultipleLines deveria ser 'No phone service' quando PhoneService = 'No'.")
    if servicoTelefone == "Yes" and multiplasLinhas == "No phone service":
        erros.append("MultipleLines = 'No phone service' é incoerente com PhoneService = 'Yes'.")

    # cobrança acumulada x meses × cobrança mensal
    meses, cobrancaMensal, cobrancaTotal = (dados.get("tenure"), dados.get("MonthlyCharges"),
                                            dados.get("TotalCharges"))
    if None not in (meses, cobrancaMensal, cobrancaTotal):
        if meses == 0 and cobrancaTotal > 0:
            erros.append("TotalCharges deveria ser 0 para cliente com tenure = 0 (sem fatura fechada).")
        elif meses > 0:
            razaoTotalPorEsperado = cobrancaTotal / (cobrancaMensal * meses)
            if not (0.5 <= razaoTotalPorEsperado <= 2.0):
                erros.append(f"TotalCharges ({cobrancaTotal}) é incoerente com tenure × MonthlyCharges "
                             f"({meses} × {cobrancaMensal} = {round(meses * cobrancaMensal, 2)}).")
    return erros
