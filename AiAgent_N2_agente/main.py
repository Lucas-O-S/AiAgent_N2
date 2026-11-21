# Interface de linha de comando do agente.
#
#     python main.py                       # conversa interativa (com memória)
#     python main.py "pergunta"            # um único turno
#     python main.py --mostrar-estado ...  # também imprime rota, nós visitados e resultados das ferramentas
import argparse
import json

from agente.grafo import construirGrafo, novoIdConversa, perguntar


def mostrarResultado(estado, mostrarDetalhes):
   
    print("\n" + estado.get("resposta", "") + "\n")
   
    if mostrarDetalhes:
        print(f"[rota: {estado.get('rota')} — {estado.get('justificativaRota')}]")
        print(f"[nós visitados: {' → '.join(estado.get('nosVisitados', []))}]")
        print("[ferramentas: " + json.dumps(estado.get("resultadoFerramentas", {}),
                                            ensure_ascii=False, default=str)[:800] + "]\n")


def executar():
   
    leitorArgumentos = argparse.ArgumentParser()
   
    leitorArgumentos.add_argument("pergunta", nargs="*")
   
    leitorArgumentos.add_argument("--mostrar-estado", dest="mostrarEstado", action="store_true")
   
    argumentos = leitorArgumentos.parse_args()

    agente = construirGrafo()
    
    idConversa = novoIdConversa("cli")

    if argumentos.pergunta:
        mostrarResultado(
            perguntar(
                agente,
                " ".join(argumentos.pergunta),
                idConversa
            ),
            argumentos.mostrarEstado)
        return

    print("Agente de churn Telco — digite 'sair' para encerrar, 'nova' para nova conversa.")
    
    while True:
        try:
            mensagem = input("você> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if mensagem.lower() in ("sair", "exit", "quit"):
            break
        
        if mensagem.lower() == "nova":
            idConversa = novoIdConversa("cli")
            print("(nova conversa)")
            continue
        
        if mensagem:
            mostrarResultado(perguntar(agente, mensagem, idConversa), argumentos.mostrarEstado)


if __name__ == "__main__":
    executar()
