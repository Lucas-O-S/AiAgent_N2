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

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "Modelos"))

MODELO = RAIZ / "Modelos" / "modelo_tabular.joblib"
SAIDA = RAIZ / "Modelos" / "metricas_modelo_tabular.json"
CSV = RAIZ / "dados" / "Telco Customer Churn.csv"
MATRIZ_DO_NOTEBOOK = [[667, 368], [64, 310]]  # Naive Bayes, célula de custos do notebook


def main():
    data = pd.read_csv(CSV)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)
    data = data.drop(columns=["customerID"])
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})
    X, y = data.drop(columns=["Churn"]), data["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = joblib.load(MODELO)
    preds = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, preds)
    status = "idêntica" if cm.tolist() == MATRIZ_DO_NOTEBOOK else "DIFERENTE"
    print(f"Matriz de confusão {status} à do notebook: {cm.tolist()}")

    tn, fp, fn, tp = cm.ravel()
    metricas = {
        "modelo": "Naive Bayes (GaussianNB) — todos os atributos + derivados",
        "arquivo": "Modelos/modelo_tabular.joblib (salvo no Colab)",
        "versao_sklearn": sklearn.__version__,
        "classes": {"0": "Não cancela (Churn = No)", "1": "Cancela (Churn = Yes)"},
        "colunas_entrada": list(X.columns),
        "n_treino": int(len(X_train)), "n_teste": int(len(X_test)),
        "taxa_churn_teste": round(float(y_test.mean()), 4),
        "teste": {
            "acuracia": round(accuracy_score(y_test, preds), 4),
            "precisao": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "matriz_confusao": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        },
    }
    SAIDA.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metricas["teste"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
