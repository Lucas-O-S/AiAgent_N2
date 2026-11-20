# Agente de IA — Churn de clientes Telco (N2)

Trabalho da disciplina de Inteligência Artificial (EC09), Grupo 6 — Prof. Marcones Cleber Brito da Silva.

Agente construído com **LangGraph** que responde perguntas em português sobre churn de clientes. Ele usa o modelo Naive Bayes treinado na avaliação anterior, carregado do disco, sem retreino. O escopo é só a base tabular **Telco Customer Churn**; a CNN de imagens é opcional e ficou de fora.

**Todo o projeto está na pasta [`AiAgent_N2_agente/`](AiAgent_N2_agente/).** O [README completo](AiAgent_N2_agente/README.md) traz a arquitetura, as instruções de execução, os testes e a documentação.

## Execução rápida

```bash
cd AiAgent_N2_agente
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                 # depois, preencha GROQ_API_KEY no .env
python app_gradio.py                 # interface web em http://127.0.0.1:7860
```

## Onde está cada entrega

| Entrega | Arquivo |
|---|---|
| Código do agente (estado, nós, grafo) | [`AiAgent_N2_agente/agente/`](AiAgent_N2_agente/agente/) |
| Modelo salvo | [`AiAgent_N2_agente/Modelos/modelo_tabular.joblib`](AiAgent_N2_agente/Modelos/) |
| Diagrama do grafo (Mermaid) | [`AiAgent_N2_agente/docs/grafo_agente.mmd`](AiAgent_N2_agente/docs/grafo_agente.mmd) |
| Cartão do modelo | [`AiAgent_N2_agente/docs/cartao_modelo.md`](AiAgent_N2_agente/docs/cartao_modelo.md) |
| Bateria de testes e resultados | [`AiAgent_N2_agente/testes/resultados_bateria.md`](AiAgent_N2_agente/testes/resultados_bateria.md) |
| Análise das falhas | [`AiAgent_N2_agente/testes/analise_falhas.md`](AiAgent_N2_agente/testes/analise_falhas.md) |
| Logs de execução | [`AiAgent_N2_agente/logs/execucoes.jsonl`](AiAgent_N2_agente/logs/execucoes.jsonl) |
| Log de uso de IA | [`AiAgent_N2_agente/LOG_USO_IA.md`](AiAgent_N2_agente/LOG_USO_IA.md) |

## Autores
- Adriana Kaori Kakazu — 082220004
- Beatriz dos Santos Buglio — 082220028
- Lucas Oliveira Silva — 082220019
- Vitoria Kaori Kuriyama — 082220005
