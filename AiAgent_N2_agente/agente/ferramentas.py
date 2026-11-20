"""Ferramentas determinísticas do agente (nenhuma usa LLM).

- preverChurn: carrega o pipeline salvo (joblib) e devolve classe + probabilidade.
- buscarCliente: motor de busca por customerID na base Telco.
- calcularFatoresAssociados: taxas de churn históricas das categorias do cliente (interpretação).
- consultarBase: estatísticas sobre a base (taxa de churn, contagem, média) com filtros.
- buscarDocumentacao: recuperação (RAG) por TF-IDF no cartão do modelo e no dicionário.
"""
from __future__ import annotations

import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from .esquema import COLUNAS, COLUNAS_CATEGORICAS, FAIXAS_NUMERICAS, normalizarValorCategorico

PASTA_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PASTA_RAIZ / "Modelos"))  # para o joblib achar telco_churn_features.DerivatedAtributes

CAMINHO_MODELO = PASTA_RAIZ / "Modelos" / "modelo_tabular.joblib"
CAMINHO_METRICAS = PASTA_RAIZ / "Modelos" / "metricas_modelo_tabular.json"
CAMINHO_CSV = PASTA_RAIZ / "dados" / "Telco Customer Churn.csv"
PASTA_DOCUMENTACAO = PASTA_RAIZ / "docs"
DOCUMENTOS_RAG = ["cartao_modelo.md", "dicionario_dados.md"]

ROTULOS_CLASSE = {0: "Não cancela", 1: "Cancela"}


# ---------------------------------------------------------------- carregamento

@lru_cache(maxsize=1)
def carregarModelo():
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {CAMINHO_MODELO}.")
    return joblib.load(CAMINHO_MODELO)


@lru_cache(maxsize=1)
def carregarMetricas() -> dict:
    return json.loads(CAMINHO_METRICAS.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def carregarBase() -> pd.DataFrame:
    """Base com a mesma limpeza do notebook + marcação de treino/teste (mesmo split)."""
    base = pd.read_csv(CAMINHO_CSV)
    base["TotalCharges"] = pd.to_numeric(base["TotalCharges"], errors="coerce").fillna(0)
    churnBinario = base["Churn"].map({"No": 0, "Yes": 1})
    indicesTreino, _ = train_test_split(base.index, test_size=0.2, random_state=42, stratify=churnBinario)
    base["conjunto"] = "teste"
    base.loc[indicesTreino, "conjunto"] = "treino"
    base["churnBinario"] = churnBinario
    return base


# ---------------------------------------------------------------- ferramentas

def preverChurn(dadosCliente: dict) -> dict:
    """Recebe um cliente JÁ VALIDADO e devolve a previsão do modelo salvo."""
    modelo = carregarModelo()
    entradaModelo = pd.DataFrame([{coluna: dadosCliente[coluna] for coluna in COLUNAS}])
    probabilidades = modelo.predict_proba(entradaModelo)[0]
    classePrevista = int(modelo.predict(entradaModelo)[0])
    indiceClasseCancela = list(modelo.classes_).index(1)
    probabilidadeCancelar = float(probabilidades[indiceClasseCancela])
    return {
        "modelo": "Naive Bayes (GaussianNB)",
        "classePrevista": classePrevista,
        "rotulo": ROTULOS_CLASSE[classePrevista],
        "probabilidadeCancelar": round(probabilidadeCancelar, 4),
        "probabilidadeNaoCancelar": round(1 - probabilidadeCancelar, 4),
        "confianca": round(max(probabilidadeCancelar, 1 - probabilidadeCancelar), 4),
    }


def buscarCliente(idCliente: str) -> dict | None:
    """Motor de busca: encontra o cliente pelo customerID. None se não existir."""
    if not idCliente:
        return None
    base = carregarBase()
    linhasEncontradas = base[base["customerID"].str.upper() == idCliente.strip().upper()]
    if linhasEncontradas.empty:
        return None
    registro = linhasEncontradas.iloc[0]
    return {
        "customerID": registro["customerID"],
        "dados": {coluna: (registro[coluna].item() if hasattr(registro[coluna], "item") else registro[coluna])
                  for coluna in COLUNAS},
        "churnRealNaBase": registro["Churn"],
        "conjunto": registro["conjunto"],  # treino ou teste no split 80/20 do notebook
    }


def calcularFatoresAssociados(dadosCliente: dict, quantidade: int = 3) -> dict:
    """Taxas de churn HISTÓRICAS (na base) das categorias que o cliente possui.
    É uma associação observada nos dados, não a explicação interna do Naive Bayes."""
    base = carregarBase()
    taxaChurnGeral = float(base["churnBinario"].mean())
    fatores = []
    for coluna in COLUNAS_CATEGORICAS:
        valor = dadosCliente.get(coluna)
        clientesDoGrupo = base[base[coluna] == valor]
        if len(clientesDoGrupo) >= 30:
            fatores.append({"atributo": coluna, "valor": valor,
                            "taxaChurnGrupo": round(float(clientesDoGrupo["churnBinario"].mean()), 4),
                            "quantidadeClientes": int(len(clientesDoGrupo))})
    fatores.sort(key=lambda fator: fator["taxaChurnGrupo"], reverse=True)
    fatoresRisco = [fator for fator in fatores if fator["taxaChurnGrupo"] > taxaChurnGeral][:quantidade]
    fatoresProtecao = sorted([fator for fator in fatores if fator["taxaChurnGrupo"] < taxaChurnGeral],
                             key=lambda fator: fator["taxaChurnGrupo"])[:quantidade]
    if dadosCliente.get("InternetService") == "Fiber optic":
        clientesFibra = base[(base["InternetService"] == "Fiber optic")
                             & (base["TechSupport"] == dadosCliente.get("TechSupport"))]
        fibraSuporte = {"atributo": "fibra + TechSupport", "valor": dadosCliente.get("TechSupport"),
                        "taxaChurnGrupo": round(float(clientesFibra["churnBinario"].mean()), 4),
                        "quantidadeClientes": int(len(clientesFibra))}
    else:
        fibraSuporte = None
    return {"taxaChurnGeral": round(taxaChurnGeral, 4), "maiorRisco": fatoresRisco,
            "menorRisco": fatoresProtecao, "fibraSuporte": fibraSuporte}


OPERADORES_VALIDOS = {"==", "!=", ">", "<", ">=", "<="}
METRICAS_VALIDAS = {"taxa_churn", "contagem", "media"}


def consultarBase(metrica: str, filtros: list[dict] | None = None,
                  agruparPor: str | None = None, colunaMedia: str | None = None) -> dict:
    """Executa uma consulta estruturada na base. Valida colunas/valores antes."""
    base = carregarBase()
    colunasValidas = set(COLUNAS) | {"Churn"}
    if metrica not in METRICAS_VALIDAS:
        return {"erro": f"Métrica '{metrica}' inválida. Use: {sorted(METRICAS_VALIDAS)}."}
    filtrosAplicados = []
    for filtro in filtros or []:
        coluna, operador, valor = filtro.get("coluna"), filtro.get("operador", "=="), filtro.get("valor")
        if coluna not in colunasValidas:
            return {"erro": f"Coluna '{coluna}' não existe na base."}
        if operador not in OPERADORES_VALIDOS:
            return {"erro": f"Operador '{operador}' inválido."}
        if coluna in COLUNAS_CATEGORICAS:
            valorNormalizado = normalizarValorCategorico(coluna, valor)
            if valorNormalizado is None:
                return {"erro": f"Valor '{valor}' inválido para {coluna}. "
                                f"Aceitos: {COLUNAS_CATEGORICAS[coluna]}."}
            valor = valorNormalizado
            if operador not in ("==", "!="):
                return {"erro": f"Operador '{operador}' não se aplica à coluna categórica {coluna}."}
        elif coluna == "Churn":
            valor = normalizarValorCategorico("Partner", valor)  # Yes/No
            if valor is None:
                return {"erro": "Churn só aceita Yes ou No."}
        else:
            try:
                valor = float(str(valor).replace(",", "."))
            except ValueError:
                return {"erro": f"Valor '{valor}' deveria ser numérico para {coluna}."}
        valoresColuna = base[coluna]
        mascara = {"==": valoresColuna == valor, "!=": valoresColuna != valor,
                   ">": valoresColuna > valor, "<": valoresColuna < valor,
                   ">=": valoresColuna >= valor, "<=": valoresColuna <= valor}[operador]
        base = base[mascara]
        filtrosAplicados.append(f"{coluna} {operador} {valor}")
    if base.empty:
        return {"filtros": filtrosAplicados, "quantidadeClientes": 0, "resultado": "nenhum cliente encontrado"}
    if agruparPor and agruparPor not in colunasValidas:
        return {"erro": f"Coluna de agrupamento '{agruparPor}' não existe."}
    if metrica == "media":
        if colunaMedia not in FAIXAS_NUMERICAS:
            return {"erro": f"Média só para colunas numéricas: {list(FAIXAS_NUMERICAS)}."}
        valoresAlvo = base[colunaMedia]
    else:
        valoresAlvo = base["churnBinario"]
    agregacao = {"taxa_churn": "mean", "contagem": "size", "media": "mean"}[metrica]
    if agruparPor:
        grupos = valoresAlvo.groupby(base[agruparPor])
        resultadoPorGrupo = getattr(grupos, agregacao)().round(4)
        resultado = {str(grupo): (int(valorGrupo) if agregacao == "size" else float(valorGrupo))
                     for grupo, valorGrupo in resultadoPorGrupo.items()}
        quantidadeClientes = {str(grupo): int(tamanho) for grupo, tamanho in grupos.size().items()}
    else:
        resultado = (int(valoresAlvo.size) if agregacao == "size"
                     else round(float(getattr(valoresAlvo, agregacao)()), 4))
        quantidadeClientes = int(len(base))
    return {"metrica": metrica if metrica != "media" else f"media de {colunaMedia}",
            "filtros": filtrosAplicados, "agrupadoPor": agruparPor,
            "resultado": resultado, "quantidadeClientes": quantidadeClientes}


# ---------------------------------------------------------------- RAG

@lru_cache(maxsize=1)
def _construirIndiceDocumentos():
    """Quebra os documentos em seções (## / ###) e monta o índice TF-IDF."""
    trechos = []
    for nomeArquivo in DOCUMENTOS_RAG:
        caminho = PASTA_DOCUMENTACAO / nomeArquivo
        if not caminho.exists():
            continue
        texto = caminho.read_text(encoding="utf-8")
        for secao in re.split(r"\n(?=#{2,3} )", texto):
            secao = secao.strip()
            if len(secao.splitlines()) > 1 and len(secao) > 40:  # ignora títulos sem conteúdo
                titulo = secao.splitlines()[0].lstrip("# ").strip()
                trechos.append({"fonte": nomeArquivo, "secao": titulo, "texto": secao})
    vetorizador = TfidfVectorizer(strip_accents="unicode", lowercase=True, sublinear_tf=True)
    matrizTfidf = vetorizador.fit_transform([trecho["texto"] for trecho in trechos])
    return trechos, vetorizador, matrizTfidf


def buscarDocumentacao(pergunta: str, quantidade: int = 3, similaridadeMinima: float = 0.05) -> list[dict]:
    trechos, vetorizador, matrizTfidf = _construirIndiceDocumentos()
    similaridades = cosine_similarity(vetorizador.transform([pergunta]), matrizTfidf)[0]
    indicesMaisSimilares = similaridades.argsort()[::-1][:quantidade]
    return [{**trechos[indice], "similaridade": round(float(similaridades[indice]), 3)}
            for indice in indicesMaisSimilares if similaridades[indice] >= similaridadeMinima]
