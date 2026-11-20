# Funções auxiliares dos nós: formatação, histórico da conversa, guarda anti-alucinação
# e montagem das respostas determinísticas.
from __future__ import annotations

import json
import re

from .nos_constantes import PADRAO_NUMERO


# 0.2654 -> "26,5%". Evita mostrar 0,0% ou 100,0% quando o valor não é exatamente 0 ou 1.
def formatarPercentual(proporcao: float) -> str:
    if 0 < proporcao < 0.001:
        return "< 0,1%"
    if 0.999 < proporcao < 1:
        return "> 99,9%"
    return f"{proporcao * 100:.1f}%".replace(".", ",")


def ultimaMensagemDoAgente(estado, limiteCaracteres: int = 800) -> str:
    for mensagem in reversed(estado.get("historico") or []):
        if mensagem["papel"] == "agente":
            return mensagem["texto"][:limiteCaracteres]
    return "(nenhuma)"


# Retorno padrão dos nós que encerram o turno: resposta + registro no histórico.
def montarRespostaDoAgente(texto: str, **outrosCampos) -> dict:
    return {"resposta": texto, "historico": [{"papel": "agente", "texto": texto}], **outrosCampos}


# Alguns provedores devolvem o conteúdo como lista de partes em vez de texto.
def extrairTextoDaResposta(conteudo) -> str:
    if isinstance(conteudo, list):
        return "".join(parte.get("text", "") if isinstance(parte, dict) else str(parte) for parte in conteudo)
    return conteudo


# ---------------------------------------------------------------- guarda anti-alucinação

# Lê "1.234,56", "3.875" ou "0.4271". Devolve (valor, casas decimais) ou (None, 0).
def _converterTextoEmNumero(texto: str):
    if "," in texto:
        texto = texto.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+", texto):
        texto = texto.replace(".", "")
    try:
        valor = float(texto)
    except ValueError:
        return None, 0
    casasDecimais = len(texto.split(".")[1]) if "." in texto else 0
    return valor, casasDecimais


# Números do texto do LLM que não aparecem no contexto
# (aceitando arredondamento e conversão proporção <-> porcentagem).
def encontrarNumerosSemFonte(texto: str, contexto: str) -> list[str]:
    numerosPermitidos = set()
    
    for numeroTexto in PADRAO_NUMERO.findall(contexto):
        
        leituras = [_converterTextoEmNumero(numeroTexto)[0]]

        if "," not in numeroTexto:
            
            try:
                leituras.append(float(numeroTexto))
                
            except ValueError:
                pass
        
        for valor in leituras:
            if valor is not None:
                numerosPermitidos.update({valor, valor * 100, valor / 100})


    numerosSuspeitos = []
    
    for numeroTexto in PADRAO_NUMERO.findall(texto):
        valor, casasDecimais = _converterTextoEmNumero(numeroTexto)
        if valor is None or (casasDecimais == 0 and valor <= 12):
            continue
        tolerancia = 0.5 * 10 ** (-casasDecimais) + 1e-9
       
        if not any(abs(permitido - valor) <= tolerancia for permitido in numerosPermitidos):
            numerosSuspeitos.append(numeroTexto)
    
    return numerosSuspeitos


# ---------------------------------------------------------------- respostas montadas por código

# Classe e probabilidade escritas por código (o LLM não as repete).
def montarCabecalhoPrevisao(resultado) -> str:
    previsao = resultado["previsao"]
    linhas = [f"**Previsão do modelo — {previsao['modelo']}: {previsao['rotulo']}**",
              f"- Probabilidade de cancelar: **{formatarPercentual(previsao['probabilidadeCancelar'])}** "
              f"(não cancelar: {formatarPercentual(previsao['probabilidadeNaoCancelar'])})"]
    clienteDaBase = resultado.get("clienteDaBase")
    if clienteDaBase:
        linha = (f"- Cliente {clienteDaBase['customerID']}: na base consta Churn = "
                 f"{clienteDaBase['churnRealNaBase']} "
                 f"(estava no conjunto de **{clienteDaBase['conjuntoNoSplit']}** do modelo)")
        if clienteDaBase["atributosAlterados"]:
            alteracoes = ", ".join(f"{atributo} = {valor}"
                                   for atributo, valor in clienteDaBase["atributosAlterados"].items())
            linha += f"; simulação com: {alteracoes}"
        linhas.append(linha)
    if resultado.get("camposPreenchidosPorCoerencia"):
        linhas.append(f"- Preenchidos por coerência: {', '.join(resultado['camposPreenchidosPorCoerencia'])}")
    return "\n".join(linhas)


# Resposta 100% determinística, usada se o LLM falhar ou citar números sem fonte.
def montarRespostaDeterministica(resultado) -> str:
    if "previsao" in resultado:
        fatores = resultado["fatores"]
        linhas = [f"**Interpretação** (taxas históricas na base; churn geral = "
                  f"{formatarPercentual(fatores['taxaChurnGeral'])})"]
        linhas += [f"- {fator['atributo']} = {fator['valor']}: "
                   f"{formatarPercentual(fator['taxaChurnGrupo'])} de churn"
                   for fator in fatores["maiorRisco"] + fatores["menorRisco"]]
        return "\n".join(linhas)
    return ("**Resposta**\n```json\n"
            + json.dumps(resultado, ensure_ascii=False, indent=2, default=str) + "\n```")
