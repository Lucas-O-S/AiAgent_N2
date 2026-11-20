# Confere o modelo salvo no Colab (Modelos/modelo_tabular.joblib) SEM retreinar:
# refaz a mesma divisão treino/teste do notebook, avalia o arquivo no teste, confere a matriz
# de confusão com a impressa no notebook e grava Modelos/metricas_modelo_tabular.json
# (métricas usadas pelo agente e pelo cartão do modelo).
#
# Uso:  python scripts/avaliar_modelo.py
import json
import sys
from pathlib import Path

import sklearn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agente.ferramentas import carregarModelo, dividirTreinoTeste, lerBaseLimpa  # noqa: E402
from agente.ferramentas_constantes import CAMINHO_METRICAS  # noqa: E402

MATRIZ_DO_NOTEBOOK = [[667, 368], [64, 310]]  # Naive Bayes, célula de custos do notebook


def executar():
    base = lerBaseLimpa()
    atributos = base.drop(columns=["customerID", "Churn", "churnBinario"])
    alvo = base["churnBinario"]
    atributosTreino, atributosTeste, alvoTreino, alvoTeste = dividirTreinoTeste(atributos, alvo, alvo=alvo)

    modelo = carregarModelo()
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
