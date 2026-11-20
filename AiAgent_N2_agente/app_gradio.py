"""Interface web (Gradio) para a demonstração ao vivo.

    python app_gradio.py   ->  abre em http://127.0.0.1:7860
"""
import json
import uuid

import gradio as gr

from agente.grafo import construir_grafo, perguntar

APP = construir_grafo()

EXEMPLOS = [
    "Qual a chance do cliente 7590-VHVEG cancelar o contrato?",
    "E se ele tivesse contrato de dois anos?",
    "Qual a taxa de churn dos clientes com contrato mensal?",
    "Qual o recall do modelo e quais são as limitações dele?",
    "Um cliente com fibra óptica e contrato mensal vai cancelar?",
    "Classifique esta imagem de célula: está infectada por malária?",
]


def responder(msg, historico, thread):
    estado = perguntar(APP, msg, thread)
    historico = historico + [{"role": "user", "content": msg},
                             {"role": "assistant", "content": estado.get("resposta", "")}]
    detalhes = {
        "rota": estado.get("rota"), "justificativa": estado.get("justificativa_rota"),
        "trace": " → ".join(estado.get("trace", [])),
        "dados_cliente": estado.get("dados_cliente"),
        "erros_validacao": estado.get("erros_validacao"),
        "resultado_ferramentas": estado.get("resultado_ferramentas"),
        "guarda_acionada": estado.get("guarda_acionada"),
    }
    return "", historico, json.dumps(detalhes, ensure_ascii=False, indent=2, default=str)


def nova_conversa():
    return [], f"web-{uuid.uuid4().hex[:8]}", ""


with gr.Blocks(title="Agente de churn — Telco") as demo:
    
    gr.Markdown("## Agente de churn — base Telco Customer Churn\n"
                "Previsões com o modelo salvo (Naive Bayes), consultas à base e ao cartão do modelo.")
    
    thread = gr.State(lambda: f"web-{uuid.uuid4().hex[:8]}")
    
    with gr.Row():
    
        with gr.Column(scale=3):
            chat = gr.Chatbot(height=520)  # Gradio 6: formato de mensagens é o padrão
            caixa = gr.Textbox(placeholder="Pergunte algo…", label="Mensagem")
            gr.Examples(EXEMPLOS, inputs=caixa)
            gr.Button("Nova conversa").click(nova_conversa, outputs=[chat, thread, caixa])
    
        with gr.Column(scale=2):
            painel = gr.Code(label="Estado do agente neste turno (rota, trace, ferramentas)", language="json")
    
    caixa.submit(responder, [caixa, chat, thread], [caixa, chat, painel])

if __name__ == "__main__":
    demo.launch()
