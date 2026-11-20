"""Montagem do grafo LangGraph (nós, arestas condicionais e memória por checkpoint)."""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import nos
from .estado import EstadoAgente


def _depois_do_roteador(estado) -> str:
    return {"previsao": "extrair_dados", "consulta": "consultar"}.get(estado["rota"], "fora_escopo")


def _depois_da_validacao(estado) -> str:
    return "prever" if estado.get("valido") else "pedir_correcao"


def construir_grafo(checkpointer=None):
    g = StateGraph(EstadoAgente)
    g.add_node("roteador", nos.roteador)
    g.add_node("extrair_dados", nos.extrair_dados)
    g.add_node("validar", nos.validar)
    g.add_node("pedir_correcao", nos.pedir_correcao)
    g.add_node("prever", nos.prever)
    g.add_node("consultar", nos.consultar)
    g.add_node("recuperar_docs", nos.recuperar_docs)
    g.add_node("responder", nos.responder)
    g.add_node("fora_escopo", nos.fora_escopo)

    g.add_edge(START, "roteador")
    g.add_conditional_edges("roteador", _depois_do_roteador,
                            {"extrair_dados": "extrair_dados", "consultar": "consultar",
                             "fora_escopo": "fora_escopo"})
    g.add_edge("extrair_dados", "validar")
    g.add_conditional_edges("validar", _depois_da_validacao,
                            {"prever": "prever", "pedir_correcao": "pedir_correcao"})
    g.add_edge("prever", "recuperar_docs")
    g.add_edge("consultar", "recuperar_docs")
    g.add_edge("recuperar_docs", "responder")
    g.add_edge("responder", END)
    g.add_edge("pedir_correcao", END)
    g.add_edge("fora_escopo", END)
    return g.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


def perguntar(app, pergunta: str, thread_id: str = "padrao") -> dict:
   
    """Executa um turno da conversa (a memória fica no checkpoint da thread)."""
   
    return app.invoke(
        {
            "pergunta": pergunta, "trace": []
        },
        config={
            "configurable": {
            "thread_id": thread_id
            }
        })
