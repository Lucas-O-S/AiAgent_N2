"""Testes automáticos SEM LLM (pytest). O LLM é substituído por respostas fixas, para testar
validação, ferramentas, guarda anti-alucinação e as arestas do grafo de forma reprodutível.
A avaliação do agente COM o LLM real é feita por testes/rodar_bateria.py.

Uso:  pytest -q
"""
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agente import ferramentas as F  # noqa: E402
from agente import nos  # noqa: E402
from agente.esquema import validar_cliente  # noqa: E402
from agente.grafo import construir_grafo, perguntar  # noqa: E402

CLIENTE_OK = F.buscar_cliente("7590-VHVEG")["dados"]


# ------------------------------------------------------------ validação

def test_cliente_da_base_e_valido():
    assert validar_cliente(CLIENTE_OK).valido


def test_normaliza_sinonimos_em_portugues():
    d = {**CLIENTE_OK, "Contract": "mensal", "gender": "mulher", "Partner": "sim",
         "PaymentMethod": "cheque eletrônico", "MonthlyCharges": "29,85"}
    r = validar_cliente(d)
    assert r.valido and r.dados["Contract"] == "Month-to-month" and r.dados["MonthlyCharges"] == 29.85


def test_faixa_categoria_e_falta_de_campos():
    r = validar_cliente({**CLIENTE_OK, "tenure": -5, "Contract": "semanal"})
    assert not r.valido and len(r.erros) == 2
    r = validar_cliente({"InternetService": "Fiber optic"})
    assert not r.valido and "tenure" in r.faltando


def test_coerencia_entre_campos():
    r = validar_cliente({**CLIENTE_OK, "InternetService": "No"})  # serviços de internet = No/Yes
    assert not r.valido and any("incoerente" in e for e in r.erros)
    r = validar_cliente({**CLIENTE_OK, "TotalCharges": 5000})      # tenure 1 x 29,85
    assert not r.valido


def test_preenchimento_por_coerencia_sem_internet():
    d = {k: v for k, v in CLIENTE_OK.items() if k not in
         ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]}
    d["InternetService"] = "No"
    r = validar_cliente(d)
    assert r.valido and len(r.preenchidos_por_coerencia) == 6


# ------------------------------------------------------------ ferramentas

def test_previsao_igual_ao_modelo_salvo():
    import pandas as pd
    res = F.prever_churn(CLIENTE_OK)
    direto = F.carregar_modelo().predict_proba(pd.DataFrame([CLIENTE_OK]))[0][1]
    assert res["probabilidade_cancelar"] == round(float(direto), 4)
    assert res["rotulo"] in ("Cancela", "Não cancela")


def test_consulta_base():
    r = F.consultar_base("taxa_churn", [{"coluna": "Contract", "operador": "==", "valor": "mensal"}])
    assert r["resultado"] == 0.4271 and r["n_clientes"] == 3875
    assert "erro" in F.consultar_base("taxa_churn", [{"coluna": "Inexistente", "valor": "x"}])


def test_rag_encontra_limitacoes():
    assert F.buscar_documentacao("quais as limitações do modelo?")[0]["secao"] == "Limitações do modelo"


def test_guarda_anti_alucinacao():
    contexto = '{"taxa": 0.4271, "n": 3875}'
    assert nos.numeros_sem_fonte("A taxa é de 42,7% em 3.875 clientes.", contexto) == []
    assert nos.numeros_sem_fonte("A taxa é de 37,5%.", contexto) == ["37,5"]


# ------------------------------------------------------------ grafo com LLM falso

class _Estruturado:
    def __init__(self, saida):
        self.saida = saida

    def invoke(self, _msgs):
        return self.saida


@pytest.fixture
def llm_falso(monkeypatch):
    respostas = {}

    def estruturado(schema):
        return _Estruturado(respostas[schema.__name__].pop(0))

    class _Texto:
        def invoke(self, _msgs):
            return AIMessage(content=respostas["texto"].pop(0))

    monkeypatch.setattr(nos, "llm_estruturado", estruturado)
    monkeypatch.setattr(nos, "obter_llm", lambda: _Texto())
    return respostas


def test_fluxo_previsao_por_id(llm_falso):
    llm_falso["DecisaoRota"] = [nos.DecisaoRota(rota="previsao", justificativa="id")]
    llm_falso["DadosExtraidos"] = [nos.DadosExtraidos(customer_id="7590-VHVEG")]
    llm_falso["texto"] = ["**Interpretação**\nContrato mensal é um fator de risco."]
    out = perguntar(construir_grafo(), "Qual a chance do cliente 7590-VHVEG cancelar?", "t1")
    assert out["trace"][-4:] == ["validar", "prever", "recuperar_docs", "responder"]
    assert "Probabilidade de cancelar" in out["resposta"] and not out["guarda_acionada"]


def test_guarda_descarta_numero_inventado(llm_falso):
    llm_falso["DecisaoRota"] = [nos.DecisaoRota(rota="previsao", justificativa="id")]
    llm_falso["DadosExtraidos"] = [nos.DadosExtraidos(customer_id="7590-VHVEG")]
    llm_falso["texto"] = ["**Interpretação**\nO cliente tem 12,34% de chance real de sair."]
    out = perguntar(construir_grafo(), "cliente 7590-VHVEG", "t2")
    assert out["guarda_acionada"] and "12,34% de chance" not in out["resposta"]


def test_memoria_pedido_de_correcao_e_complemento(llm_falso):
    app = construir_grafo()
    parcial = {k: v for k, v in CLIENTE_OK.items() if k != "TotalCharges"}
    llm_falso["DecisaoRota"] = [nos.DecisaoRota(rota="previsao", justificativa="a"),
                                nos.DecisaoRota(rota="previsao", justificativa="b")]
    llm_falso["DadosExtraidos"] = [nos.DadosExtraidos(novo_cliente=True, **parcial),
                                   nos.DadosExtraidos(TotalCharges=29.85)]
    llm_falso["texto"] = ["**Interpretação**\nok"]
    t1 = perguntar(app, "dados parciais", "t3")
    assert "pedir_correcao" in t1["trace"] and "TotalCharges" in t1["resposta"]
    t2 = perguntar(app, "o total é 29,85", "t3")
    assert t2["trace"][0] == "roteador" and "pedir_correcao" not in t2["trace"]
    assert "prever" in t2["trace"] and len(t2["historico"]) == 4


def test_fora_do_escopo(llm_falso):
    llm_falso["DecisaoRota"] = [nos.DecisaoRota(rota="fora_escopo", justificativa="imagem")]
    out = perguntar(construir_grafo(), "classifique esta imagem", "t4")
    assert out["trace"] == ["roteador", "fora_escopo"]


def test_consulta_com_nova_tentativa(llm_falso):
    llm_falso["DecisaoRota"] = [nos.DecisaoRota(rota="consulta", justificativa="taxa")]
    llm_falso["PlanoConsulta"] = [
        nos.PlanoConsulta(tipo="estatistica", metrica="taxa_churn",
                          filtros=[nos.Filtro(coluna="Contrato", valor="mensal")]),   # coluna errada
        nos.PlanoConsulta(tipo="estatistica", metrica="taxa_churn",
                          filtros=[nos.Filtro(coluna="Contract", valor="Month-to-month")])]
    llm_falso["texto"] = ["**Resposta**\nA taxa é 42,71%."]
    out = perguntar(construir_grafo(), "taxa de churn no contrato mensal?", "t5")
    assert out["resultado_ferramentas"]["resultado"] == 0.4271 and not out["guarda_acionada"]
