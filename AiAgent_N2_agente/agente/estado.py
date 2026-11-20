"""Estado tipado que circula entre os nós do grafo."""
from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, TypedDict

Rota = Literal["previsao", "consulta", "fora_escopo"]


def _trace_do_turno(atual: list, novo: list) -> list:
    """Acumula os nós visitados, recomeçando a lista a cada turno (o roteador é sempre o 1º nó)."""
    if novo and novo[0] == "roteador":
        return list(novo)
    return (atual or []) + (novo or [])


class EstadoAgente(TypedDict, total=False):
    # entrada do turno
    pergunta: str                       # mensagem do usuário neste turno
    historico: Annotated[list, operator.add]   # [{"papel": ..., "texto": ...}] (memória da conversa)

    # roteamento
    rota: Rota
    justificativa_rota: str

    # dados tabulares (persistem entre turnos para permitir correções e "e se...")
    dados_cliente: dict                 # valores brutos acumulados
    customer_id: Optional[str]
    cliente_base: Optional[dict]        # resultado da busca por customerID
    aguardando_correcao: bool

    # validação
    valido: bool
    erros_validacao: list
    campos_faltando: list
    preenchidos_por_coerencia: list

    # resultados das ferramentas e contexto recuperado
    resultado_ferramentas: dict
    contexto_docs: list

    # saída
    resposta: str
    guarda_acionada: bool               # True se a resposta do LLM foi descartada por citar números sem fonte
    trace: Annotated[list, _trace_do_turno]  # nós visitados neste turno (log)
