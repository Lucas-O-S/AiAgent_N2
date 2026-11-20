# Logs próprios: cada nó grava uma linha JSON em logs/execucoes.jsonl.
# (LangSmith também funciona: basta definir LANGSMITH_TRACING=true e LANGSMITH_API_KEY no .env.)
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

ARQUIVO_LOG = Path(__file__).resolve().parent.parent / "logs" / "execucoes.jsonl"


def _resumirJson(objeto, limite=600):
    texto = json.dumps(objeto, ensure_ascii=False, default=str)
    return texto if len(texto) <= limite else texto[:limite] + "…"


# Envolve a função de um nó: grava o log e acrescenta o nome do nó em nosVisitados.
# Aplicado em grafo.py ao registrar cada nó.
def registrarNo(nomeNo):
    def decorador(funcaoNo):
        # O parâmetro precisa se chamar "config": é por esse nome que o LangGraph o preenche.
        # Não trocar a cópia do __name__ (abaixo) por functools.wraps: o wraps faz o LangGraph
        # enxergar a assinatura de funcaoNo, que não tem "config", e o thread_id some do log.
        def executarComRegistro(estado, config=None):
            inicio = time.perf_counter()
            erro = None
            try:
                saida = funcaoNo(estado)
            except Exception as excecao:  # registra e propaga
                erro = repr(excecao)
                raise
            finally:
                idConversa = ((config or {}).get("configurable") or {}).get("thread_id")
                ARQUIVO_LOG.parent.mkdir(exist_ok=True)
                with ARQUIVO_LOG.open("a", encoding="utf-8") as arquivo:
                    arquivo.write(json.dumps({
                        "quando": datetime.now().isoformat(timespec="seconds"),
                        "thread_id": idConversa, "no": nomeNo,
                        "ms": round((time.perf_counter() - inicio) * 1000),
                        "entrada": _resumirJson({"pergunta": estado.get("pergunta")}),
                        "saida": None if erro else _resumirJson(
                            {chave: valor for chave, valor in saida.items() if chave != "nosVisitados"}),
                        "erro": erro,
                    }, ensure_ascii=False) + "\n")
            saida = dict(saida)
            saida["nosVisitados"] = [nomeNo]
            return saida
        executarComRegistro.__name__ = funcaoNo.__name__
        return executarComRegistro
    return decorador
