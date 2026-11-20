"""Estado tipado que circula entre os nós do grafo."""
from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional, TypedDict

Rota = Literal["previsao", "consulta", "fora_escopo"]


def _acumularNosDoTurno(nosAtuais: list, nosNovos: list) -> list:
    """Acumula os nós visitados, recomeçando a lista a cada turno (o roteador é sempre o 1º nó)."""
    if nosNovos and nosNovos[0] == "roteador":
        return list(nosNovos)
    return (nosAtuais or []) + (nosNovos or [])


class EstadoAgente(TypedDict, total=False):
    # entrada do turno
    pergunta: str                       # mensagem do usuário neste turno
    historico: Annotated[list, operator.add]   # [{"papel": ..., "texto": ...}] (memória da conversa)

    # roteamento
    rota: Rota
    justificativaRota: str

    # dados tabulares (persistem entre turnos para permitir correções e "e se...")
    dadosCliente: dict                  # valores brutos acumulados
    idCliente: Optional[str]            # customerID informado pelo usuário
    clienteDaBase: Optional[dict]       # resultado da busca por customerID
    aguardandoCorrecao: bool

    # validação
    valido: bool
    errosValidacao: list
    camposFaltando: list
    camposPreenchidosPorCoerencia: list

    # resultados das ferramentas e contexto recuperado
    resultadoFerramentas: dict
    trechosDocumentacao: list

    # saída
    resposta: str
    guardaAcionada: bool                # True se a resposta do LLM foi descartada por citar números sem fonte
    nosVisitados: Annotated[list, _acumularNosDoTurno]  # nós visitados neste turno (log)
