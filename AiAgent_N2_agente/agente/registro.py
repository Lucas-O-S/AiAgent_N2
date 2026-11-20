"""Logs próprios: cada nó grava uma linha JSON em logs/execucoes.jsonl.
(LangSmith também funciona: basta definir LANGSMITH_TRACING=true e LANGSMITH_API_KEY no .env.)"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

ARQUIVO_LOG = Path(__file__).resolve().parent.parent / "logs" / "execucoes.jsonl"


def _resumo(obj, limite=600):
    txt = json.dumps(obj, ensure_ascii=False, default=str)
    return txt if len(txt) <= limite else txt[:limite] + "…"


def registrar_no(nome):
    def deco(func):
        def wrapper(estado, config=None):
            ini = time.perf_counter()
            erro = None
            try:
                saida = func(estado)
            except Exception as e:  # registra e propaga
                erro = repr(e)
                raise
            finally:
                thread = ((config or {}).get("configurable") or {}).get("thread_id")
                ARQUIVO_LOG.parent.mkdir(exist_ok=True)
                with ARQUIVO_LOG.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "quando": datetime.now().isoformat(timespec="seconds"),
                        "thread_id": thread, "no": nome,
                        "ms": round((time.perf_counter() - ini) * 1000),
                        "entrada": _resumo({"pergunta": estado.get("pergunta")}),
                        "saida": None if erro else _resumo({k: v for k, v in saida.items() if k != "trace"}),
                        "erro": erro,
                    }, ensure_ascii=False) + "\n")
            saida = dict(saida)
            saida["trace"] = [nome]
            return saida
        wrapper.__name__ = func.__name__
        return wrapper
    return deco
