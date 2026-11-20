# Constantes das ferramentas: caminhos dos arquivos, divisão treino/teste e operações aceitas nas consultas.
import operator
from pathlib import Path
from typing import Literal

PASTA_RAIZ = Path(__file__).resolve().parent.parent
PASTA_MODELOS = PASTA_RAIZ / "Modelos"
CAMINHO_MODELO = PASTA_MODELOS / "modelo_tabular.joblib"
CAMINHO_METRICAS = PASTA_MODELOS / "metricas_modelo_tabular.json"
CAMINHO_CSV = PASTA_RAIZ / "dados" / "Telco Customer Churn.csv"
PASTA_DOCUMENTACAO = PASTA_RAIZ / "docs"
DOCUMENTOS_RAG = ["cartao_modelo.md", "dicionario_dados.md"]

# Divisão treino/teste: precisa ser a mesma do notebook que treinou o modelo
PROPORCAO_TESTE = 0.2
SEMENTE_ALEATORIA = 42

NOME_MODELO = "Naive Bayes (GaussianNB)"
ROTULOS_CLASSE = {0: "Não cancela", 1: "Cancela"}

# Tamanho mínimo de um grupo para a taxa de churn dele ser mostrada como fator associado
MINIMO_CLIENTES_POR_GRUPO = 30

# Operadores aceitos nos filtros: o Literal é o que o LLM pode escolher; o dicionário executa a comparação
Operador = Literal["==", "!=", ">", "<", ">=", "<="]
FUNCAO_POR_OPERADOR = {"==": operator.eq, "!=": operator.ne, ">": operator.gt,
                       "<": operator.lt, ">=": operator.ge, "<=": operator.le}

# Métricas aceitas nas consultas -> agregação do pandas usada para calculá-las
Metrica = Literal["taxa_churn", "contagem", "media"]
AGREGACAO_POR_METRICA = {"taxa_churn": "mean", "contagem": "size", "media": "mean"}
