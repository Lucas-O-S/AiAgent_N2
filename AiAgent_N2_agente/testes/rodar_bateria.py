# Roda a bateria de testes com o LLM configurado no .env e grava:
# - testes/resultados_bateria.csv  (uma linha por caso)
# - testes/resultados_bateria.md   (tabela + taxas de acerto, para o README/apresentação)
#
# Uso:  python testes/rodar_bateria.py
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

PASTA_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PASTA_RAIZ))

from agente.grafo import construirGrafo, novoIdConversa, perguntar  # noqa: E402
from agente.llm import obterNomeModelo, obterProvedor  # noqa: E402

# nó do grafo -> desfecho esperado em casos_teste.json
DESFECHO_POR_NO = {"prever": "previsao", "pedir_correcao": "pedido_correcao",
                   "consultar": "consulta", "fora_escopo": "recusa"}


def identificarDesfecho(nosVisitados):
    for nomeNo, desfecho in DESFECHO_POR_NO.items():
        if nomeNo in nosVisitados:
            return desfecho
    return "?"


def executar():
    casos = json.loads((PASTA_RAIZ / "testes" / "casos_teste.json").read_text(encoding="utf-8"))
    agente = construirGrafo()
    # as chaves de cada linha viram as colunas do CSV: mantidas como estavam
    linhas = []
    for caso in casos:
        idConversa = novoIdConversa(f"teste-{caso['id']}")
        inicio = time.perf_counter()
        try:
            for mensagem in caso["turnos"]:
                estado = perguntar(agente, mensagem, idConversa)
            erro = ""
        except Exception as excecao:
            estado, erro = {}, repr(excecao)
        segundos = round(time.perf_counter() - inicio, 1)
        rota = estado.get("rota", "erro")
        desfecho = identificarDesfecho(estado.get("nosVisitados", []))
        resultadoJson = json.dumps(estado.get("resultadoFerramentas", {}), ensure_ascii=False, default=str)
        textoEsperado = caso.get("verificar_resultado_contem")
        ferramentaOk = "" if not textoEsperado else ("sim" if textoEsperado in resultadoJson else "não")
        linhas.append({
            "id": caso["id"], "categoria": caso["categoria"], "descricao": caso["descricao"],
            "entrada": " ⏎ ".join(caso["turnos"]),
            "rota_esperada": caso["rota_esperada"], "rota_obtida": rota,
            "rota_ok": rota == caso["rota_esperada"],
            "desfecho_esperado": caso["desfecho_esperado"], "desfecho_obtido": desfecho,
            "desfecho_ok": desfecho == caso["desfecho_esperado"],
            "resultado_ferramenta_ok": ferramentaOk,
            "guarda_acionada": estado.get("guardaAcionada", ""),
            "trace": " → ".join(estado.get("nosVisitados", [])),
            "segundos": segundos, "erro": erro,
            "resposta": estado.get("resposta", ""),
        })
        print(f"{caso['id']}: rota {rota} ({'ok' if rota == caso['rota_esperada'] else 'ERRO'}) | "
              f"desfecho {desfecho} | {segundos}s {erro}")

    totalCasos = len(linhas)
    acertosRota = sum(linha["rota_ok"] for linha in linhas)
    acertosDesfecho = sum(linha["desfecho_ok"] for linha in linhas)
    taxaRotaCorreta = acertosRota / totalCasos
    taxaDesfechoCorreto = acertosDesfecho / totalCasos
    with (PASTA_RAIZ / "testes" / "resultados_bateria.csv").open("w", newline="", encoding="utf-8") as arquivo:
        escritorCsv = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritorCsv.writeheader()
        escritorCsv.writerows(linhas)

    linhasMarkdown = [
        "# Resultados da bateria de testes",
        f"- Execução: {datetime.now():%d/%m/%Y %H:%M} | LLM: `{obterProvedor()}` / `{obterNomeModelo()}`",
        f"- **Taxa de roteamento correto: {acertosRota}/{totalCasos} = {taxaRotaCorreta:.0%}**",
        f"- Taxa de desfecho correto (rota + validação): {acertosDesfecho}/{totalCasos} = {taxaDesfechoCorreto:.0%}",
        "", "| ID | Categoria | Rota esperada | Rota obtida | Desfecho esperado | Desfecho obtido "
            "| Ferramenta ok | Guarda | Tempo (s) |",
        "|---|---|---|---|---|---|---|---|---|"]
    for linha in linhas:
        linhasMarkdown.append(
            f"| {linha['id']} | {linha['categoria']} | {linha['rota_esperada']} | {linha['rota_obtida']} "
            f"{'✅' if linha['rota_ok'] else '❌'} | {linha['desfecho_esperado']} | {linha['desfecho_obtido']} "
            f"{'✅' if linha['desfecho_ok'] else '❌'} | {linha['resultado_ferramenta_ok']} | "
            f"{linha['guarda_acionada']} | {linha['segundos']} |")
    linhasMarkdown += ["", "## Respostas completas", ""]
    for linha in linhas:
        linhasMarkdown += [f"### {linha['id']} — {linha['descricao']}", f"**Entrada:** {linha['entrada']}", "",
                           f"**Trace:** {linha['trace']}", "", linha["resposta"] or f"_erro: {linha['erro']}_", ""]
    linhasMarkdown += ["## Análise das falhas", "",
                       "_(preencher pelo grupo: para cada ❌, explicar a causa e o que foi feito)_"]
    (PASTA_RAIZ / "testes" / "resultados_bateria.md").write_text("\n".join(linhasMarkdown), encoding="utf-8")
    print(f"\nRoteamento correto: {taxaRotaCorreta:.0%} | Desfecho correto: {taxaDesfechoCorreto:.0%}")
    print("Arquivos: testes/resultados_bateria.csv e testes/resultados_bateria.md")


if __name__ == "__main__":
    executar()
