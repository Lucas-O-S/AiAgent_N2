"""Ferramentas determinísticas do agente (nenhuma usa LLM).

- prever_churn: carrega o pipeline salvo (joblib) e devolve classe + probabilidade.
- buscar_cliente: motor de busca por customerID na base Telco.
- fatores_associados: taxas de churn históricas das categorias do cliente (interpretação).
- consultar_base: estatísticas sobre a base (taxa de churn, contagem, média) com filtros.
- buscar_documentacao: recuperação (RAG) por TF-IDF no cartão do modelo e no dicionário.
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

from .esquema import CATEGORICAS, COLUNAS, NUMERICAS, normalizar_categorica

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "Modelos"))  # para o joblib achar telco_churn_features.DerivatedAtributes

CAMINHO_MODELO = RAIZ / "Modelos" / "modelo_tabular.joblib"
CAMINHO_METRICAS = RAIZ / "Modelos" / "metricas_modelo_tabular.json"
CAMINHO_CSV = RAIZ / "dados" / "Telco Customer Churn.csv"
PASTA_DOCS = RAIZ / "docs"
DOCS_RAG = ["cartao_modelo.md", "dicionario_dados.md"]

ROTULOS = {0: "Não cancela", 1: "Cancela"}


# ---------------------------------------------------------------- carregamento

@lru_cache(maxsize=1)
def carregar_modelo():
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em {CAMINHO_MODELO}.")
    return joblib.load(CAMINHO_MODELO)


@lru_cache(maxsize=1)
def carregar_metricas() -> dict:
    return json.loads(CAMINHO_METRICAS.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def carregar_base() -> pd.DataFrame:
    """Base com a mesma limpeza do notebook + marcação de treino/teste (mesmo split)."""
    d = pd.read_csv(CAMINHO_CSV)
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0)
    y = d["Churn"].map({"No": 0, "Yes": 1})
    idx_treino, _ = train_test_split(d.index, test_size=0.2, random_state=42, stratify=y)
    d["conjunto"] = "teste"
    d.loc[idx_treino, "conjunto"] = "treino"
    d["churn_bin"] = y
    return d


# ---------------------------------------------------------------- ferramentas

def prever_churn(dados: dict) -> dict:
    """Recebe um cliente JÁ VALIDADO e devolve a previsão do modelo salvo."""
    modelo = carregar_modelo()
    X = pd.DataFrame([{c: dados[c] for c in COLUNAS}])
    proba = modelo.predict_proba(X)[0]
    classe = int(modelo.predict(X)[0])
    idx = list(modelo.classes_).index(1)
    p_churn = float(proba[idx])
    return {
        "modelo": "Naive Bayes (GaussianNB)",
        "classe_prevista": classe,
        "rotulo": ROTULOS[classe],
        "probabilidade_cancelar": round(p_churn, 4),
        "probabilidade_nao_cancelar": round(1 - p_churn, 4),
        "confianca": round(max(p_churn, 1 - p_churn), 4),
    }


_RE_ID = re.compile(r"\b\d{4}-[A-Z]{5}\b")


def buscar_cliente(customer_id: str) -> dict | None:
    """Motor de busca: encontra o cliente pelo customerID. None se não existir."""
    if not customer_id:
        return None
    d = carregar_base()
    linha = d[d["customerID"].str.upper() == customer_id.strip().upper()]
    if linha.empty:
        return None
    r = linha.iloc[0]
    return {
        "customerID": r["customerID"],
        "dados": {c: (r[c].item() if hasattr(r[c], "item") else r[c]) for c in COLUNAS},
        "churn_real_na_base": r["Churn"],
        "conjunto": r["conjunto"],  # treino ou teste no split 80/20 do notebook
    }


def fatores_associados(dados: dict, top: int = 3) -> dict:
    """Taxas de churn HISTÓRICAS (na base) das categorias que o cliente possui.
    É uma associação observada nos dados, não a explicação interna do Naive Bayes."""
    d = carregar_base()
    geral = float(d["churn_bin"].mean())
    itens = []
    for col in CATEGORICAS:
        val = dados.get(col)
        grupo = d[d[col] == val]
        if len(grupo) >= 30:
            itens.append({"atributo": col, "valor": val,
                          "taxa_churn_grupo": round(float(grupo["churn_bin"].mean()), 4),
                          "n_clientes": int(len(grupo))})
    itens.sort(key=lambda x: x["taxa_churn_grupo"], reverse=True)
    risco = [i for i in itens if i["taxa_churn_grupo"] > geral][:top]
    protecao = sorted([i for i in itens if i["taxa_churn_grupo"] < geral],
                      key=lambda x: x["taxa_churn_grupo"])[:top]
    if dados.get("InternetService") == "Fiber optic":
        g = d[(d["InternetService"] == "Fiber optic") & (d["TechSupport"] == dados.get("TechSupport"))]
        risco_fibra = {"atributo": "fibra + TechSupport", "valor": dados.get("TechSupport"),
                       "taxa_churn_grupo": round(float(g["churn_bin"].mean()), 4), "n_clientes": int(len(g))}
    else:
        risco_fibra = None
    return {"taxa_churn_geral": round(geral, 4), "maior_risco": risco,
            "menor_risco": protecao, "fibra_suporte": risco_fibra}


OPERADORES = {"==", "!=", ">", "<", ">=", "<="}
METRICAS = {"taxa_churn", "contagem", "media"}


def consultar_base(metrica: str, filtros: list[dict] | None = None,
                   agrupar_por: str | None = None, coluna_media: str | None = None) -> dict:
    """Executa uma consulta estruturada na base. Valida colunas/valores antes."""
    d = carregar_base()
    colunas_validas = set(COLUNAS) | {"Churn"}
    if metrica not in METRICAS:
        return {"erro": f"Métrica '{metrica}' inválida. Use: {sorted(METRICAS)}."}
    filtros_aplicados = []
    for f in filtros or []:
        col, op, val = f.get("coluna"), f.get("operador", "=="), f.get("valor")
        if col not in colunas_validas:
            return {"erro": f"Coluna '{col}' não existe na base."}
        if op not in OPERADORES:
            return {"erro": f"Operador '{op}' inválido."}
        if col in CATEGORICAS:
            val_n = normalizar_categorica(col, val)
            if val_n is None:
                return {"erro": f"Valor '{val}' inválido para {col}. Aceitos: {CATEGORICAS[col]}."}
            val = val_n
            if op not in ("==", "!="):
                return {"erro": f"Operador '{op}' não se aplica à coluna categórica {col}."}
        elif col == "Churn":
            val = normalizar_categorica("Partner", val)  # Yes/No
            if val is None:
                return {"erro": "Churn só aceita Yes ou No."}
        else:
            try:
                val = float(str(val).replace(",", "."))
            except ValueError:
                return {"erro": f"Valor '{val}' deveria ser numérico para {col}."}
        serie = d[col]
        mask = {"==": serie == val, "!=": serie != val, ">": serie > val,
                "<": serie < val, ">=": serie >= val, "<=": serie <= val}[op]
        d = d[mask]
        filtros_aplicados.append(f"{col} {op} {val}")
    if d.empty:
        return {"filtros": filtros_aplicados, "n_clientes": 0, "resultado": "nenhum cliente encontrado"}
    if agrupar_por and agrupar_por not in colunas_validas:
        return {"erro": f"Coluna de agrupamento '{agrupar_por}' não existe."}
    if metrica == "media":
        if coluna_media not in NUMERICAS:
            return {"erro": f"Média só para colunas numéricas: {list(NUMERICAS)}."}
        alvo = d[coluna_media]
    else:
        alvo = d["churn_bin"]
    func = {"taxa_churn": "mean", "contagem": "size", "media": "mean"}[metrica]
    if agrupar_por:
        g = alvo.groupby(d[agrupar_por])
        res = getattr(g, func)().round(4)
        resultado = {str(k): (int(v) if func == "size" else float(v)) for k, v in res.items()}
        tamanhos = {str(k): int(v) for k, v in g.size().items()}
    else:
        resultado = int(alvo.size) if func == "size" else round(float(getattr(alvo, func)()), 4)
        tamanhos = int(len(d))
    return {"metrica": metrica if metrica != "media" else f"media de {coluna_media}",
            "filtros": filtros_aplicados, "agrupado_por": agrupar_por,
            "resultado": resultado, "n_clientes": tamanhos}


# ---------------------------------------------------------------- RAG

@lru_cache(maxsize=1)
def _indice_docs():
    trechos = []
    for nome in DOCS_RAG:
        caminho = PASTA_DOCS / nome
        if not caminho.exists():
            continue
        texto = caminho.read_text(encoding="utf-8")
        for bloco in re.split(r"\n(?=#{2,3} )", texto):
            bloco = bloco.strip()
            if len(bloco.splitlines()) > 1 and len(bloco) > 40:  # ignora títulos sem conteúdo
                titulo = bloco.splitlines()[0].lstrip("# ").strip()
                trechos.append({"fonte": nome, "secao": titulo, "texto": bloco})
    vet = TfidfVectorizer(strip_accents="unicode", lowercase=True, sublinear_tf=True)
    matriz = vet.fit_transform([t["texto"] for t in trechos])
    return trechos, vet, matriz


def buscar_documentacao(pergunta: str, k: int = 3, limiar: float = 0.05) -> list[dict]:
    trechos, vet, matriz = _indice_docs()
    sims = cosine_similarity(vet.transform([pergunta]), matriz)[0]
    ordem = sims.argsort()[::-1][:k]
    return [{**trechos[i], "score": round(float(sims[i]), 3)} for i in ordem if sims[i] >= limiar]
