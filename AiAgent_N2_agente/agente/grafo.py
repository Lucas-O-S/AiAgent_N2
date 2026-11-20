"""Montagem do grafo LangGraph (nós, arestas condicionais e memória por checkpoint).

Os nomes dos nós ("extrair_dados", "pedir_correcao"...) são rótulos do grafo: aparecem em
nosVisitados, nos logs, nos testes e no diagrama docs/grafo_agente.mmd.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import nos
from .estado import EstadoAgente


def _escolherNoAposRoteador(estado) -> str:
    return {"previsao": "extrair_dados", "consulta": "consultar"}.get(estado["rota"], "fora_escopo")


def _escolherNoAposValidacao(estado) -> str:
    return "prever" if estado.get("valido") else "pedir_correcao"


def construirGrafo(memoria=None):
    grafo = StateGraph(EstadoAgente)
    grafo.add_node("roteador", nos.roteador)
    grafo.add_node("extrair_dados", nos.extrairDados)
    grafo.add_node("validar", nos.validar)
    grafo.add_node("pedir_correcao", nos.pedirCorrecao)
    grafo.add_node("prever", nos.prever)
    grafo.add_node("consultar", nos.consultar)
    grafo.add_node("recuperar_docs", nos.recuperarDocumentacao)
    grafo.add_node("responder", nos.responder)
    grafo.add_node("fora_escopo", nos.foraEscopo)

    grafo.add_edge(START, "roteador")
    grafo.add_conditional_edges("roteador", _escolherNoAposRoteador,
                                {"extrair_dados": "extrair_dados", "consultar": "consultar",
                                 "fora_escopo": "fora_escopo"})
    grafo.add_edge("extrair_dados", "validar")
    grafo.add_conditional_edges("validar", _escolherNoAposValidacao,
                                {"prever": "prever", "pedir_correcao": "pedir_correcao"})
    grafo.add_edge("prever", "recuperar_docs")
    grafo.add_edge("consultar", "recuperar_docs")
    grafo.add_edge("recuperar_docs", "responder")
    grafo.add_edge("responder", END)
    grafo.add_edge("pedir_correcao", END)
    grafo.add_edge("fora_escopo", END)
    return grafo.compile(checkpointer=memoria if memoria is not None else MemorySaver())


def perguntar(agente, pergunta: str, idConversa: str = "padrao") -> dict:
    """Executa um turno da conversa (a memória fica no checkpoint da conversa)."""
    return agente.invoke(
        {"pergunta": pergunta, "nosVisitados": []},
        config={"configurable": {"thread_id": idConversa}},  # "thread_id" é a chave exigida pelo LangGraph
    )
