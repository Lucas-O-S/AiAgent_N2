# Montagem do grafo LangGraph (nós, arestas condicionais e memória por checkpoint).
from __future__ import annotations

import uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from . import nos
from .estado import NO_INICIAL, EstadoAgente
from .registro import registrarNo

# Nome do nó no grafo -> função. O nome aparece em nosVisitados, nos logs, nos testes e no
# diagrama docs/grafo_agente.mmd; é definido só aqui.
NOS_DO_GRAFO = {
    NO_INICIAL: nos.roteador,
    "extrair_dados": nos.extrairDados,
    "validar": nos.validar,
    "pedir_correcao": nos.pedirCorrecao,
    "prever": nos.prever,
    "consultar": nos.consultar,
    "recuperar_docs": nos.recuperarDocumentacao,
    "responder": nos.responder,
    "fora_escopo": nos.foraEscopo,
}


def _escolherNoAposRoteador(estado) -> str:
    return {"previsao": "extrair_dados", "consulta": "consultar"}.get(estado["rota"], "fora_escopo")


def _escolherNoAposValidacao(estado) -> str:
    return "prever" if estado.get("valido") else "pedir_correcao"


def construirGrafo(memoria=None):
    grafo = StateGraph(EstadoAgente)
    for nomeNo, funcaoNo in NOS_DO_GRAFO.items():
        grafo.add_node(nomeNo, registrarNo(nomeNo)(funcaoNo))

    grafo.add_edge(START, NO_INICIAL)
    grafo.add_conditional_edges(NO_INICIAL, _escolherNoAposRoteador,
                                ["extrair_dados", "consultar", "fora_escopo"])
    grafo.add_edge("extrair_dados", "validar")
    grafo.add_conditional_edges("validar", _escolherNoAposValidacao, ["prever", "pedir_correcao"])
    grafo.add_edge("prever", "recuperar_docs")
    grafo.add_edge("consultar", "recuperar_docs")
    grafo.add_edge("recuperar_docs", "responder")
    grafo.add_edge("responder", END)
    grafo.add_edge("pedir_correcao", END)
    grafo.add_edge("fora_escopo", END)
    return grafo.compile(checkpointer=memoria if memoria is not None else MemorySaver())


# Identificador único de conversa (a memória do agente é separada por conversa).
def novoIdConversa(prefixo: str) -> str:
    return f"{prefixo}-{uuid.uuid4().hex[:8]}"


# Executa um turno da conversa (a memória fica no checkpoint da conversa).
def perguntar(agente, pergunta: str, idConversa: str = "padrao") -> dict:
    return agente.invoke(
        {"pergunta": pergunta, "nosVisitados": []},
        config={"configurable": {"thread_id": idConversa}},  # "thread_id" é a chave exigida pelo LangGraph
    )
