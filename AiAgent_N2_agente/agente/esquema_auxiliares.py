# Funções auxiliares do esquema: normalização de texto e conversão de valores digitados pelo usuário.
from __future__ import annotations

import unicodedata

from .esquema_constantes import RESPOSTAS_NAO, RESPOSTAS_SIM


# Remove acentos, passa para minúsculas e junta espaços repetidos.
def normalizarTexto(texto) -> str:
    texto = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return " ".join(texto.lower().strip().split())


# Devolve a opção (grafia do dataset) que corresponde ao valor; None se não reconhecer.
# Aceita a própria opção sem acento/maiúsculas, um sinônimo ou uma resposta de sim/não.
def normalizarParaOpcoes(valor, opcoes: list[str], sinonimos: dict | None = None) -> str | None:
    if valor is None:
        return None
    valorNormalizado = normalizarTexto(valor)
    for opcao in opcoes:
        if normalizarTexto(opcao) == valorNormalizado:
            return opcao
    if sinonimos and valorNormalizado in sinonimos:
        return sinonimos[valorNormalizado]
    if "Yes" in opcoes and valorNormalizado in RESPOSTAS_SIM:
        return "Yes"
    if "No" in opcoes and valorNormalizado in RESPOSTAS_NAO:
        return "No"
    return None


# Aceita número, "sim/não" (1/0) e texto como "R$ 1.234,56"; None se não for numérico.
def converterParaNumero(valor):
    if isinstance(valor, bool):
        return None
    if isinstance(valor, (int, float)):
        return float(valor)
    texto = normalizarTexto(valor)
    if texto in RESPOSTAS_SIM:
        return 1.0
    if texto in RESPOSTAS_NAO:
        return 0.0
    texto = texto.replace("r$", "").replace(" ", "")
    if "," in texto:  # formato brasileiro 1.234,56
        texto = texto.replace(".", "").replace(",", ".")
    try:
        return float(texto)
    except ValueError:
        return None
