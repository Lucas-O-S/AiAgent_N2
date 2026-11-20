# Exporta o diagrama do grafo para docs/grafo_agente.mmd (Mermaid).
# Se houver internet, também tenta gerar docs/grafo_agente.png.
import sys
from pathlib import Path

PASTA_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PASTA_RAIZ))
from agente.grafo import construirGrafo  # noqa: E402

diagrama = construirGrafo().get_graph()
(PASTA_RAIZ / "docs" / "grafo_agente.mmd").write_text(diagrama.draw_mermaid(), encoding="utf-8")
print("docs/grafo_agente.mmd gerado")
try:
    (PASTA_RAIZ / "docs" / "grafo_agente.png").write_bytes(diagrama.draw_mermaid_png())
    print("docs/grafo_agente.png gerado")
except Exception as erro:
    print(f"PNG não gerado ({type(erro).__name__}); cole o .mmd em https://mermaid.live para exportar.")
