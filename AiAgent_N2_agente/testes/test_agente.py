"""Testes automáticos SEM LLM (pytest). O LLM é substituído por respostas fixas, para testar
validação, ferramentas, guarda anti-alucinação e as arestas do grafo de forma reprodutível.
A avaliação do agente COM o LLM real é feita por testes/rodar_bateria.py.

Uso:  pytest -q
(o pytest só encontra funções cujo nome começa com "test")
"""
import sys
from pathlib import Path

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agente import ferramentas  # noqa: E402
from agente import nos  # noqa: E402
from agente.esquema import validarCliente  # noqa: E402
from agente.grafo import construirGrafo, perguntar  # noqa: E402

CLIENTE_VALIDO = ferramentas.buscarCliente("7590-VHVEG")["dados"]
SERVICOS_INTERNET = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                     "TechSupport", "StreamingTV", "StreamingMovies"]


# ------------------------------------------------------------ validação

def testClienteDaBaseEhValido():
    assert validarCliente(CLIENTE_VALIDO).valido


def testNormalizaSinonimosEmPortugues():
    dadosCliente = {**CLIENTE_VALIDO, "Contract": "mensal", "gender": "mulher", "Partner": "sim",
                    "PaymentMethod": "cheque eletrônico", "MonthlyCharges": "29,85"}
    resultado = validarCliente(dadosCliente)
    assert resultado.valido
    assert resultado.dadosNormalizados["Contract"] == "Month-to-month"
    assert resultado.dadosNormalizados["MonthlyCharges"] == 29.85


def testFaixaCategoriaEFaltaDeCampos():
    resultado = validarCliente({**CLIENTE_VALIDO, "tenure": -5, "Contract": "semanal"})
    assert not resultado.valido and len(resultado.erros) == 2
    resultado = validarCliente({"InternetService": "Fiber optic"})
    assert not resultado.valido and "tenure" in resultado.camposFaltando


def testCoerenciaEntreCampos():
    resultado = validarCliente({**CLIENTE_VALIDO, "InternetService": "No"})  # serviços de internet = No/Yes
    assert not resultado.valido and any("incoerente" in erro for erro in resultado.erros)
    resultado = validarCliente({**CLIENTE_VALIDO, "TotalCharges": 5000})      # tenure 1 x 29,85
    assert not resultado.valido


def testPreenchimentoPorCoerenciaSemInternet():
    dadosCliente = {coluna: valor for coluna, valor in CLIENTE_VALIDO.items() if coluna not in SERVICOS_INTERNET}
    dadosCliente["InternetService"] = "No"
    resultado = validarCliente(dadosCliente)
    assert resultado.valido and len(resultado.camposPreenchidosPorCoerencia) == 6


# ------------------------------------------------------------ ferramentas

def testPrevisaoIgualAoModeloSalvo():
    previsao = ferramentas.preverChurn(CLIENTE_VALIDO)
    probabilidadeDireta = ferramentas.carregarModelo().predict_proba(pd.DataFrame([CLIENTE_VALIDO]))[0][1]
    assert previsao["probabilidadeCancelar"] == round(float(probabilidadeDireta), 4)
    assert previsao["rotulo"] in ("Cancela", "Não cancela")


def testConsultaBase():
    resultado = ferramentas.consultarBase(
        "taxa_churn", [{"coluna": "Contract", "operador": "==", "valor": "mensal"}])
    assert resultado["resultado"] == 0.4271 and resultado["quantidadeClientes"] == 3875
    assert "erro" in ferramentas.consultarBase("taxa_churn", [{"coluna": "Inexistente", "valor": "x"}])


def testRagEncontraLimitacoes():
    trechos = ferramentas.buscarDocumentacao("quais as limitações do modelo?")
    assert trechos[0]["secao"] == "Limitações do modelo"


def testGuardaAntiAlucinacao():
    contexto = '{"taxa": 0.4271, "n": 3875}'
    assert nos.encontrarNumerosSemFonte("A taxa é de 42,7% em 3.875 clientes.", contexto) == []
    assert nos.encontrarNumerosSemFonte("A taxa é de 37,5%.", contexto) == ["37,5"]


# ------------------------------------------------------------ grafo com LLM falso

class _LlmEstruturadoFalso:
    def __init__(self, respostaFixa):
        self.respostaFixa = respostaFixa

    def invoke(self, _mensagens):
        return self.respostaFixa


@pytest.fixture
def llmFalso(monkeypatch):
    """Dicionário de respostas fixas: chave = nome do schema Pydantic, ou "texto" para o LLM livre."""
    respostasFixas = {}

    def obterLlmEstruturadoFalso(esquemaSaida):
        return _LlmEstruturadoFalso(respostasFixas[esquemaSaida.__name__].pop(0))

    class _LlmTextoFalso:
        def invoke(self, _mensagens):
            return AIMessage(content=respostasFixas["texto"].pop(0))

    monkeypatch.setattr(nos, "obterLlmEstruturado", obterLlmEstruturadoFalso)
    monkeypatch.setattr(nos, "obterLlm", lambda: _LlmTextoFalso())
    return respostasFixas


def testFluxoPrevisaoPorId(llmFalso):
    llmFalso["DecisaoRota"] = [nos.DecisaoRota(rota="previsao", justificativa="id")]
    llmFalso["DadosExtraidos"] = [nos.DadosExtraidos(idCliente="7590-VHVEG")]
    llmFalso["texto"] = ["**Interpretação**\nContrato mensal é um fator de risco."]
    estado = perguntar(construirGrafo(), "Qual a chance do cliente 7590-VHVEG cancelar?", "t1")
    assert estado["nosVisitados"][-4:] == ["validar", "prever", "recuperar_docs", "responder"]
    assert "Probabilidade de cancelar" in estado["resposta"] and not estado["guardaAcionada"]


def testGuardaDescartaNumeroInventado(llmFalso):
    llmFalso["DecisaoRota"] = [nos.DecisaoRota(rota="previsao", justificativa="id")]
    llmFalso["DadosExtraidos"] = [nos.DadosExtraidos(idCliente="7590-VHVEG")]
    llmFalso["texto"] = ["**Interpretação**\nO cliente tem 12,34% de chance real de sair."]
    estado = perguntar(construirGrafo(), "cliente 7590-VHVEG", "t2")
    assert estado["guardaAcionada"] and "12,34% de chance" not in estado["resposta"]


def testMemoriaPedidoDeCorrecaoEComplemento(llmFalso):
    agente = construirGrafo()
    dadosParciais = {coluna: valor for coluna, valor in CLIENTE_VALIDO.items() if coluna != "TotalCharges"}
    llmFalso["DecisaoRota"] = [nos.DecisaoRota(rota="previsao", justificativa="a"),
                               nos.DecisaoRota(rota="previsao", justificativa="b")]
    llmFalso["DadosExtraidos"] = [nos.DadosExtraidos(novoCliente=True, **dadosParciais),
                                  nos.DadosExtraidos(TotalCharges=29.85)]
    llmFalso["texto"] = ["**Interpretação**\nok"]
    primeiroTurno = perguntar(agente, "dados parciais", "t3")
    assert "pedir_correcao" in primeiroTurno["nosVisitados"] and "TotalCharges" in primeiroTurno["resposta"]
    segundoTurno = perguntar(agente, "o total é 29,85", "t3")
    assert segundoTurno["nosVisitados"][0] == "roteador" and "pedir_correcao" not in segundoTurno["nosVisitados"]
    assert "prever" in segundoTurno["nosVisitados"] and len(segundoTurno["historico"]) == 4


def testForaDoEscopo(llmFalso):
    llmFalso["DecisaoRota"] = [nos.DecisaoRota(rota="fora_escopo", justificativa="imagem")]
    estado = perguntar(construirGrafo(), "classifique esta imagem", "t4")
    assert estado["nosVisitados"] == ["roteador", "fora_escopo"]


def testConsultaComNovaTentativa(llmFalso):
    llmFalso["DecisaoRota"] = [nos.DecisaoRota(rota="consulta", justificativa="taxa")]
    llmFalso["PlanoConsulta"] = [
        nos.PlanoConsulta(tipo="estatistica", metrica="taxa_churn",
                          filtros=[nos.Filtro(coluna="Contrato", valor="mensal")]),   # coluna errada
        nos.PlanoConsulta(tipo="estatistica", metrica="taxa_churn",
                          filtros=[nos.Filtro(coluna="Contract", valor="Month-to-month")])]
    llmFalso["texto"] = ["**Resposta**\nA taxa é 42,71%."]
    estado = perguntar(construirGrafo(), "taxa de churn no contrato mensal?", "t5")
    assert estado["resultadoFerramentas"]["resultado"] == 0.4271 and not estado["guardaAcionada"]
