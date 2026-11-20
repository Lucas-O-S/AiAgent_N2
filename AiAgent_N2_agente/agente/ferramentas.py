# Ferramentas determinísticas do agente (nenhuma usa LLM).
#
# - preverChurn: carrega o pipeline salvo (joblib) e devolve classe + probabilidade.
# - buscarCliente: motor de busca por customerID na base Telco.
# - calcularFatoresAssociados: taxas de churn históricas das categorias do cliente (interpretação).
# - consultarBase: estatísticas sobre a base (taxa de churn, contagem, média) com filtros.
# - buscarDocumentacao: recuperação (RAG) por TF-IDF no cartão do modelo e no dicionário.
from __future__ import annotations

import json
import re
import sys
from functools import lru_cache

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

from .esquema import normalizarSimNao, normalizarValorCategorico
from .esquema_constantes import COLUNAS, COLUNAS_CATEGORICAS, FAIXAS_NUMERICAS
from .ferramentas_constantes import (AGREGACAO_POR_METRICA, CAMINHO_CSV, CAMINHO_METRICAS, CAMINHO_MODELO,
                                     DOCUMENTOS_RAG, FUNCAO_POR_OPERADOR, MINIMO_CLIENTES_POR_GRUPO,
                                     NOME_MODELO, PASTA_DOCUMENTACAO, PASTA_MODELOS, PROPORCAO_TESTE,
                                     ROTULOS_CLASSE, SEMENTE_ALEATORIA)


# ---------------------------------------------------------------- carregamento

@lru_cache(maxsize=1)
def carregarModelo():
    if not CAMINHO_MODELO.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {CAMINHO_MODELO}.")
    # o pipeline salvo referencia telco_churn_features.DerivatedAtributes, que está na pasta Modelos
    if str(PASTA_MODELOS) not in sys.path:
        sys.path.insert(0, str(PASTA_MODELOS))
    return joblib.load(CAMINHO_MODELO)


@lru_cache(maxsize=1)
def carregarMetricas() -> dict:
    return json.loads(CAMINHO_METRICAS.read_text(encoding="utf-8"))


# CSV com a mesma limpeza do notebook + coluna churnBinario (0/1).
def lerBaseLimpa() -> pd.DataFrame:
    base = pd.read_csv(CAMINHO_CSV)
    base["TotalCharges"] = pd.to_numeric(base["TotalCharges"], errors="coerce").fillna(0)
    base["churnBinario"] = base["Churn"].map({"No": 0, "Yes": 1})
    return base


# Mesma divisão treino/teste do notebook (usada pelo agente e por scripts/avaliar_modelo.py).
def dividirTreinoTeste(*dados, alvo):
    return train_test_split(*dados, test_size=PROPORCAO_TESTE, random_state=SEMENTE_ALEATORIA, stratify=alvo)


# Base limpa + coluna "conjunto" dizendo se o cliente ficou no treino ou no teste do modelo.
@lru_cache(maxsize=1)
def carregarBase() -> pd.DataFrame:
    base = lerBaseLimpa()
    indicesTreino, _ = dividirTreinoTeste(base.index, alvo=base["churnBinario"])
    base["conjunto"] = "teste"
    base.loc[indicesTreino, "conjunto"] = "treino"
    return base


# ---------------------------------------------------------------- ferramentas

# Recebe um cliente JÁ VALIDADO e devolve a previsão do modelo salvo.
def preverChurn(dadosCliente: dict) -> dict:
    modelo = carregarModelo()
    entradaModelo = pd.DataFrame([{coluna: dadosCliente[coluna] for coluna in COLUNAS}])
    probabilidades = modelo.predict_proba(entradaModelo)[0]
    classePrevista = int(modelo.predict(entradaModelo)[0])
    probabilidadeCancelar = float(probabilidades[list(modelo.classes_).index(1)])
    return {
        "modelo": NOME_MODELO,
        "classePrevista": classePrevista,
        "rotulo": ROTULOS_CLASSE[classePrevista],
        "probabilidadeCancelar": round(probabilidadeCancelar, 4),
        "probabilidadeNaoCancelar": round(1 - probabilidadeCancelar, 4),
        "confianca": round(max(probabilidadeCancelar, 1 - probabilidadeCancelar), 4),
    }


# Motor de busca: encontra o cliente pelo customerID. None se não existir.
def buscarCliente(idCliente: str) -> dict | None:
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


def _descreverGrupo(atributo: str, valor, clientesDoGrupo: pd.DataFrame) -> dict:
    return {"atributo": atributo, "valor": valor,
            "taxaChurnGrupo": round(float(clientesDoGrupo["churnBinario"].mean()), 4),
            "quantidadeClientes": int(len(clientesDoGrupo))}


# Taxas de churn HISTÓRICAS (na base) das categorias que o cliente possui.
# É uma associação observada nos dados, não a explicação interna do Naive Bayes.
def calcularFatoresAssociados(dadosCliente: dict, quantidade: int = 3) -> dict:
    base = carregarBase()
    taxaChurnGeral = float(base["churnBinario"].mean())
    fatores = []
    for coluna in COLUNAS_CATEGORICAS:
        valor = dadosCliente.get(coluna)
        clientesDoGrupo = base[base[coluna] == valor]
        if len(clientesDoGrupo) >= MINIMO_CLIENTES_POR_GRUPO:
            fatores.append(_descreverGrupo(coluna, valor, clientesDoGrupo))
    fatores.sort(key=lambda fator: fator["taxaChurnGrupo"], reverse=True)
    fatoresRisco = [fator for fator in fatores if fator["taxaChurnGrupo"] > taxaChurnGeral][:quantidade]
    fatoresProtecao = sorted([fator for fator in fatores if fator["taxaChurnGrupo"] < taxaChurnGeral],
                             key=lambda fator: fator["taxaChurnGrupo"])[:quantidade]
    fibraSuporte = None
    if dadosCliente.get("InternetService") == "Fiber optic":
        clientesFibra = base[(base["InternetService"] == "Fiber optic")
                             & (base["TechSupport"] == dadosCliente.get("TechSupport"))]
        fibraSuporte = _descreverGrupo("fibra + TechSupport", dadosCliente.get("TechSupport"), clientesFibra)
    return {"taxaChurnGeral": round(taxaChurnGeral, 4), "maiorRisco": fatoresRisco,
            "menorRisco": fatoresProtecao, "fibraSuporte": fibraSuporte}


# Devolve (valor convertido, None) ou (None, mensagem de erro).
def _prepararValorDoFiltro(coluna: str, operador: str, valor):
    if coluna in COLUNAS_CATEGORICAS:
        valorNormalizado = normalizarValorCategorico(coluna, valor)
        if valorNormalizado is None:
            return None, f"Valor '{valor}' inválido para {coluna}. Aceitos: {COLUNAS_CATEGORICAS[coluna]}."
        if operador not in ("==", "!="):
            return None, f"Operador '{operador}' não se aplica à coluna categórica {coluna}."
        return valorNormalizado, None
    if coluna == "Churn":
        valorNormalizado = normalizarSimNao(valor)
        if valorNormalizado is None:
            return None, "Churn só aceita Yes ou No."
        return valorNormalizado, None
    try:
        return float(str(valor).replace(",", ".")), None
    except ValueError:
        return None, f"Valor '{valor}' deveria ser numérico para {coluna}."


# Executa uma consulta estruturada na base. Valida colunas/valores antes.
def consultarBase(metrica: str, filtros: list[dict] | None = None,
                  agruparPor: str | None = None, colunaMedia: str | None = None) -> dict:
    base = carregarBase()
    colunasValidas = set(COLUNAS) | {"Churn"}
    if metrica not in AGREGACAO_POR_METRICA:
        return {"erro": f"Métrica '{metrica}' inválida. Use: {sorted(AGREGACAO_POR_METRICA)}."}

    filtrosAplicados = []
    for filtro in filtros or []:
        coluna, operador = filtro.get("coluna"), filtro.get("operador", "==")
        if coluna not in colunasValidas:
            return {"erro": f"Coluna '{coluna}' não existe na base."}
        if operador not in FUNCAO_POR_OPERADOR:
            return {"erro": f"Operador '{operador}' inválido."}
        valor, erro = _prepararValorDoFiltro(coluna, operador, filtro.get("valor"))
        if erro:
            return {"erro": erro}
        base = base[FUNCAO_POR_OPERADOR[operador](base[coluna], valor)]
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

    agregacao = AGREGACAO_POR_METRICA[metrica]
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

# Quebra os documentos em seções (## / ###) e monta o índice TF-IDF.
@lru_cache(maxsize=1)
def _construirIndiceDocumentos():
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
