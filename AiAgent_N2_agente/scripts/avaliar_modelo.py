"""Confere o modelo salvo no Colab (Modelos/modelo_tabular.joblib) SEM retreinar:
refaz a mesma divisão treino/teste do notebook, avalia o arquivo no teste, confere a matriz
de confusão com a impressa no notebook e grava Modelos/metricas_modelo_tabular.json
(métricas usadas pelo agente e pelo cartão do modelo).

Uso:  python scripts/avaliar_modelo.py
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

PASTA_RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PASTA_RAIZ / "Modelos"))

CAMINHO_MODELO = PASTA_RAIZ / "Modelos" / "modelo_tabular.joblib"
CAMINHO_METRICAS = PASTA_RAIZ / "Modelos" / "metricas_modelo_tabular.json"
CAMINHO_CSV = PASTA_RAIZ / "dados" / "Telco Customer Churn.csv"
MATRIZ_DO_NOTEBOOK = [[667, 368], [64, 310]]  # Naive Bayes, célula de custos do notebook


def executar():
    base = pd.read_csv(CAMINHO_CSV)
    base["TotalCharges"] = pd.to_numeric(base["TotalCharges"], errors="coerce").fillna(0)
    base = base.drop(columns=["customerID"])
    base["Churn"] = base["Churn"].map({"No": 0, "Yes": 1})
    atributos, alvo = base.drop(columns=["Churn"]), base["Churn"]
    atributosTreino, atributosTeste, alvoTreino, alvoTeste = train_test_split(
        atributos, alvo, test_size=0.2, random_state=42, stratify=alvo)

    modelo = joblib.load(CAMINHO_MODELO)
    previsoes = modelo.predict(atributosTeste)
    probabilidadesCancelar = modelo.predict_proba(atributosTeste)[:, 1]
    matrizConfusao = confusion_matrix(alvoTeste, previsoes)
    situacao = "idêntica" if matrizConfusao.tolist() == MATRIZ_DO_NOTEBOOK else "DIFERENTE"
    print(f"Matriz de confusão {situacao} à do notebook: {matrizConfusao.tolist()}")

    verdadeirosNegativos, falsosPositivos, falsosNegativos, verdadeirosPositivos = matrizConfusao.ravel()
    # as chaves abaixo formam o arquivo JSON lido pelo agente: não renomear
    metricas = {
        "modelo": "Naive Bayes (GaussianNB) — todos os atributos + derivados",
        "arquivo": "Modelos/modelo_tabular.joblib (salvo no Colab)",
        "versao_sklearn": sklearn.__version__,
        "classes": {"0": "Não cancela (Churn = No)", "1": "Cancela (Churn = Yes)"},
        "colunas_entrada": list(atributos.columns),
        "n_treino": int(len(atributosTreino)), "n_teste": int(len(atributosTeste)),
        "taxa_churn_teste": round(float(alvoTeste.mean()), 4),
        "teste": {
            "acuracia": round(accuracy_score(alvoTeste, previsoes), 4),
            "precisao": round(precision_score(alvoTeste, previsoes), 4),
            "recall": round(recall_score(alvoTeste, previsoes), 4),
            "f1": round(f1_score(alvoTeste, previsoes), 4),
            "roc_auc": round(roc_auc_score(alvoTeste, probabilidadesCancelar), 4),
            "matriz_confusao": {"TN": int(verdadeirosNegativos), "FP": int(falsosPositivos),
                                "FN": int(falsosNegativos), "TP": int(verdadeirosPositivos)},
        },
    }
    CAMINHO_METRICAS.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metricas["teste"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    executar()
