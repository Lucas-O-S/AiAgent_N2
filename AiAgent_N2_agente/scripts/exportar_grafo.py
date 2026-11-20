"""Exporta o diagrama do grafo para docs/grafo_agente.mmd (Mermaid).
Se houver internet, também tenta gerar docs/grafo_agente.png."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from agente.grafo import construir_grafo  # noqa: E402

grafo = construir_grafo().get_graph()
(RAIZ / "docs" / "grafo_agente.mmd").write_text(grafo.draw_mermaid(), encoding="utf-8")
print("docs/grafo_agente.mmd gerado")
try:
    (RAIZ / "docs" / "grafo_agente.png").write_bytes(grafo.draw_mermaid_png())
    print("docs/grafo_agente.png gerado")
except Exception as e:
    print(f"PNG não gerado ({type(e).__name__}); cole o .mmd em https://mermaid.live para exportar.")
