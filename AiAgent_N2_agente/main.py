"""Interface de linha de comando do agente.

    python main.py                       # conversa interativa (com memória)
    python main.py "pergunta"            # um único turno
    python main.py --mostrar-estado ...  # também imprime rota, trace e resultados das ferramentas
"""
import argparse
import json
import uuid

from agente.grafo import construir_grafo, perguntar


def mostrar(estado, detalhes):
    
    print("\n" + estado.get("resposta", "") + "\n")
    
    if detalhes:
    
        print(f"[rota: {estado.get('rota')} — {estado.get('justificativa_rota')}]")
    
        print(f"[trace: {' → '.join(estado.get('trace', []))}]")
    
        print("[ferramentas: " + json.dumps(estado.get("resultado_ferramentas", {}),
                                            ensure_ascii=False, default=str)[:800] + "]\n")


def main():

    argumentParser = argparse.ArgumentParser()
    
    argumentParser.add_argument("pergunta", nargs="*")
    
    argumentParser.add_argument("--mostrar-estado", action="store_true")
    
    args = argumentParser.parse_args()
    
    app = construir_grafo()
    
    thread = f"cli-{uuid.uuid4().hex[:8]}"
    
    if args.pergunta:
    
        mostrar(perguntar(app, " ".join(args.pergunta), thread), args.mostrar_estado)
        return
    
    print("Agente de churn Telco — digite 'sair' para encerrar, 'nova' para nova conversa.")
    
    while True:
        try:
            msg = input("você> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if msg.lower() in ("sair", "exit", "quit"):
            break
        if msg.lower() == "nova":
            thread = f"cli-{uuid.uuid4().hex[:8]}"
            print("(nova conversa)")
            continue
        if msg:
            mostrar(perguntar(app, msg, thread), args.mostrar_estado)


if __name__ == "__main__":
    main()
