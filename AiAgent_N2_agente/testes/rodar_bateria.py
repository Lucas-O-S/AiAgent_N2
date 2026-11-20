"""Roda a bateria de testes com o LLM configurado no .env e grava:
- testes/resultados_bateria.csv  (uma linha por caso)
- testes/resultados_bateria.md   (tabela + taxas de acerto, para o README/apresentação)

Uso:  python testes/rodar_bateria.py
"""
import csv
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from agente.grafo import construir_grafo, perguntar  # noqa: E402
from agente.llm import nome_modelo, provedor  # noqa: E402

DESFECHO_POR_NO = {"prever": "previsao", "pedir_correcao": "pedido_correcao",
                   "consultar": "consulta", "fora_escopo": "recusa"}


def desfecho(trace):
    for no, nome in DESFECHO_POR_NO.items():
        if no in trace:
            return nome
    return "?"


def main():
    casos = json.loads((RAIZ / "testes" / "casos_teste.json").read_text(encoding="utf-8"))
    app = construir_grafo()
    linhas = []
    for caso in casos:
        thread = f"teste-{caso['id']}-{uuid.uuid4().hex[:6]}"
        ini = time.perf_counter()
        try:
            for msg in caso["turnos"]:
                estado = perguntar(app, msg, thread)
            erro = ""
        except Exception as e:
            estado, erro = {}, repr(e)
        seg = round(time.perf_counter() - ini, 1)
        rota = estado.get("rota", "erro")
        desf = desfecho(estado.get("trace", []))
        res_json = json.dumps(estado.get("resultado_ferramentas", {}), ensure_ascii=False, default=str)
        esperado_txt = caso.get("verificar_resultado_contem")
        ferramenta_ok = "" if not esperado_txt else ("sim" if esperado_txt in res_json else "não")
        linhas.append({
            "id": caso["id"], "categoria": caso["categoria"], "descricao": caso["descricao"],
            "entrada": " ⏎ ".join(caso["turnos"]),
            "rota_esperada": caso["rota_esperada"], "rota_obtida": rota,
            "rota_ok": rota == caso["rota_esperada"],
            "desfecho_esperado": caso["desfecho_esperado"], "desfecho_obtido": desf,
            "desfecho_ok": desf == caso["desfecho_esperado"],
            "resultado_ferramenta_ok": ferramenta_ok,
            "guarda_acionada": estado.get("guarda_acionada", ""),
            "trace": " → ".join(estado.get("trace", [])),
            "segundos": seg, "erro": erro,
            "resposta": estado.get("resposta", ""),
        })
        print(f"{caso['id']}: rota {rota} ({'ok' if rota == caso['rota_esperada'] else 'ERRO'}) | "
              f"desfecho {desf} | {seg}s {erro}")

    n = len(linhas)
    taxa_rota = sum(l["rota_ok"] for l in linhas) / n
    taxa_desf = sum(l["desfecho_ok"] for l in linhas) / n
    with (RAIZ / "testes" / "resultados_bateria.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0]))
        w.writeheader()
        w.writerows(linhas)

    md = [f"# Resultados da bateria de testes",
          f"- Execução: {datetime.now():%d/%m/%Y %H:%M} | LLM: `{provedor()}` / `{nome_modelo()}`",
          f"- **Taxa de roteamento correto: {sum(l['rota_ok'] for l in linhas)}/{n} = {taxa_rota:.0%}**",
          f"- Taxa de desfecho correto (rota + validação): {sum(l['desfecho_ok'] for l in linhas)}/{n} = {taxa_desf:.0%}",
          "", "| ID | Categoria | Rota esperada | Rota obtida | Desfecho esperado | Desfecho obtido | Ferramenta ok | Guarda | Tempo (s) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for l in linhas:
        md.append(f"| {l['id']} | {l['categoria']} | {l['rota_esperada']} | {l['rota_obtida']} "
                  f"{'✅' if l['rota_ok'] else '❌'} | {l['desfecho_esperado']} | {l['desfecho_obtido']} "
                  f"{'✅' if l['desfecho_ok'] else '❌'} | {l['resultado_ferramenta_ok']} | "
                  f"{l['guarda_acionada']} | {l['segundos']} |")
    md += ["", "## Respostas completas", ""]
    for l in linhas:
        md += [f"### {l['id']} — {l['descricao']}", f"**Entrada:** {l['entrada']}", "",
               f"**Trace:** {l['trace']}", "", l["resposta"] or f"_erro: {l['erro']}_", ""]
    md += ["## Análise das falhas", "", "_(preencher pelo grupo: para cada ❌, explicar a causa e o que foi feito)_"]
    (RAIZ / "testes" / "resultados_bateria.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nRoteamento correto: {taxa_rota:.0%} | Desfecho correto: {taxa_desf:.0%}")
    print("Arquivos: testes/resultados_bateria.csv e testes/resultados_bateria.md")


if __name__ == "__main__":
    main()
