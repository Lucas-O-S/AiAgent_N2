"""Interface web (Gradio) para a demonstração ao vivo.

    python app_gradio.py   ->  abre em http://127.0.0.1:7860
"""
import json
import uuid

import gradio as gr

from agente.grafo import construirGrafo, perguntar

AGENTE = construirGrafo()

EXEMPLOS = [
    "Qual a chance do cliente 7590-VHVEG cancelar o contrato?",
    "E se ele tivesse contrato de dois anos?",
    "Qual a taxa de churn dos clientes com contrato mensal?",
    "Qual o recall do modelo e quais são as limitações dele?",
    "Um cliente com fibra óptica e contrato mensal vai cancelar?",
    "Classifique esta imagem de célula: está infectada por malária?",
]


def novoIdConversa() -> str:
    return f"web-{uuid.uuid4().hex[:8]}"


def responderMensagem(mensagem, historicoChat, idConversa):
    estado = perguntar(AGENTE, mensagem, idConversa)
    # "role" e "content" são as chaves exigidas pelo gr.Chatbot
    historicoChat = historicoChat + [{"role": "user", "content": mensagem},
                                     {"role": "assistant", "content": estado.get("resposta", "")}]
    detalhes = {
        "rota": estado.get("rota"), "justificativa": estado.get("justificativaRota"),
        "nosVisitados": " → ".join(estado.get("nosVisitados", [])),
        "dadosCliente": estado.get("dadosCliente"),
        "errosValidacao": estado.get("errosValidacao"),
        "resultadoFerramentas": estado.get("resultadoFerramentas"),
        "guardaAcionada": estado.get("guardaAcionada"),
    }
    return "", historicoChat, json.dumps(detalhes, ensure_ascii=False, indent=2, default=str)


def iniciarNovaConversa():
    return [], novoIdConversa(), ""


with gr.Blocks(title="Agente de churn — Telco") as interface:
    gr.Markdown("## Agente de churn — base Telco Customer Churn\n"
                "Previsões com o modelo salvo (Naive Bayes), consultas à base e ao cartão do modelo.")
    idConversa = gr.State(novoIdConversa)

    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=520)  # Gradio 6: formato de mensagens é o padrão
            caixaMensagem = gr.Textbox(placeholder="Pergunte algo…", label="Mensagem")
            gr.Examples(EXEMPLOS, inputs=caixaMensagem)
            gr.Button("Nova conversa").click(iniciarNovaConversa, outputs=[chat, idConversa, caixaMensagem])
        with gr.Column(scale=2):
            painelEstado = gr.Code(label="Estado do agente neste turno (rota, nós visitados, ferramentas)",
                                   language="json")

    caixaMensagem.submit(responderMensagem, [caixaMensagem, chat, idConversa],
                         [caixaMensagem, chat, painelEstado])

if __name__ == "__main__":
    interface.launch()
