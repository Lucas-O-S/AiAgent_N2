# Log de uso de IA generativa

Exigência do enunciado (item 4): no mínimo **8 interações**, com os principais prompts, o que foi aceito, rejeitado ou corrigido e por quê, e ao menos **2 situações em que a IA errou** e como o grupo percebeu e corrigiu.

| # | Data | Ferramenta | Integrante | Prompt (resumo) | Resultado | Aceito / rejeitado / corrigido — por quê |
|---|---|---|---|---|---|---|
| 1 | 20/11/2026 | Claude | Adriana | Enviou o enunciado da N2 e o link do repositório; pediu um agente (não ML novo) que, a partir de uma pergunta, consulte o modelo para gerar respostas, como motor de busca/interpretação, só com a base Telco, sem imagens. | Estrutura inicial do agente LangGraph: estado, roteador, extração, validação, ferramentas (previsão, busca por customerID, consulta à base, RAG), memória, Gradio, cartão do modelo, testes e este log. | _(preencher: o que o grupo revisou, manteve e mudou)_ |
| 2 | 20/11/2026 | Claude | Adriana | Enviou o link do repositório do agente (AiAgent_N2), que já tinha o modelo original exportado do Colab. | Agente adaptado à estrutura do repositório (`Modelos/`, `dados/`), usando o arquivo original do Colab; script de conferência do modelo sem retreino. | Aceito |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
| 7 | | | | | | |
| 8 | | | | | | |

## Situações em que a IA errou ou sugeriu algo inadequado

Erros reais que apareceram na interação nº 1, detectados ao executar o código:

1. **Parâmetro removido do Gradio.** A IA escreveu `gr.Chatbot(type="messages")`, que não existe mais no Gradio 6. *Como foi percebido:* ao importar `app_gradio.py` deu `TypeError: unexpected keyword argument 'type'`. *Correção:* parâmetro removido (no Gradio 6 o formato de mensagens já é o padrão).
2. **Tipo errado no schema de extração.** `SeniorCitizen` foi declarado como texto, mas o dataset (e o LLM) usam 0/1 numérico. *Como foi percebido:* um teste do `pytest` falhou com `ValidationError: Input should be a valid string`. *Correção:* validador que converte números/booleanos em texto antes da validação.
3. **RAG devolvendo um trecho vazio.** A divisão dos documentos por títulos criava um trecho só com o título do cartão, que aparecia em 1º lugar para "qual o recall do modelo". *Como foi percebido:* teste manual da busca. *Correção:* trechos de uma linha descartados e TF-IDF com `sublinear_tf`; a seção de métricas passou a ser a 1ª.

4. **Recriou o modelo em vez de usar o original e fixou a versão errada do scikit-learn.** Na 1ª versão, como o `.joblib` não estava no repositório antigo, a IA retreinou o pipeline fora do Colab e fixou `scikit-learn==1.9.1`. *Como foi percebido:* ao usar o arquivo original do Colab (repositório AiAgent_N2), ele não abria na 1.9.1 (`AttributeError: _RemainderColsList`). *Correção:* versão fixada em 1.6.1 (a do Colab), o agente passou a usar o arquivo original e o script de retreino foi trocado por `scripts/avaliar_modelo.py`, que só confere o modelo (matriz de confusão idêntica à do notebook).

_(O grupo deve acrescentar os erros que encontrar nos próprios testes, por exemplo rotas erradas do LLM na bateria de testes.)_
