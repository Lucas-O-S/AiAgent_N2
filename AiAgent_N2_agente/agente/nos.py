# Nós do grafo. Os que usam LLM: roteador, extrairDados, consultar (planejamento) e responder.
# Os demais são determinísticos. O nome de cada nó e o registro em log são definidos em grafo.py.
from __future__ import annotations

import json
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from . import ferramentas
from .esquema import validarCliente
from .esquema_constantes import COLUNAS, COLUNAS_CATEGORICAS, DESCRICAO_CAMPOS
from .estado import Rota, valoresIniciaisDoTurno
from .ferramentas_constantes import Metrica, Operador
from .llm import obterLlm, obterLlmEstruturado
from .nos_auxiliares import (encontrarNumerosSemFonte, extrairTextoDaResposta, montarCabecalhoPrevisao,
                             montarRespostaDeterministica, montarRespostaDoAgente, ultimaMensagemDoAgente)
from .nos_constantes import (CONTEXTO_AGUARDANDO_CORRECAO, FORMATO_CONSULTA, FORMATO_PREVISAO,
                             MENSAGEM_FORA_ESCOPO, PADRAO_ID_CLIENTE, PROMPT_CONSULTA, PROMPT_EXTRACAO,
                             PROMPT_RESPOSTA, PROMPT_ROTEADOR, TERMOS_BUSCA_PREVISAO)

# =============================================================== formatos de saída pedidos ao LLM


class DecisaoRota(BaseModel):
    
    rota: Rota = Field(
        description="previsao = prever churn de um cliente (por customerID ou atributos), "
                    "inclusive correções/complementos de dados e perguntas 'e se'; "
                    "consulta = perguntas sobre a base Telco, estatísticas, variáveis, o modelo, "
                    "métricas ou limitações; fora_escopo = qualquer outra coisa (inclui imagens).")
    justificativa: str = Field(description="Uma frase curta explicando a escolha.")


# Campo que o LLM pode deixar null; a descrição vai no schema enviado ao LLM.
def _campoOpcional(descricao):
    return Field(default=None, description=descricao)


# Campos de um cliente mencionados EXPLICITAMENTE pelo usuário. Não invente valores.
class DadosExtraidos(BaseModel):
   
    idCliente: Optional[str] = _campoOpcional("customerID no formato 1234-ABCDE, se citado")
   
    novoCliente: bool = Field(default=False, description=(
        "true se o usuário está descrevendo um cliente DIFERENTE do anterior; "
        "false se está corrigindo/complementando ou perguntando 'e se' sobre o mesmo cliente"))

    # nomes iguais às colunas do dataset (não renomear: o modelo e a validação dependem deles)
    gender: Optional[str] = _campoOpcional("Female ou Male")
   
    SeniorCitizen: Optional[str] = _campoOpcional("1 se idoso, 0 se não")
   
    Partner: Optional[str] = _campoOpcional("Yes/No")
   
    Dependents: Optional[str] = _campoOpcional("Yes/No")
   
    tenure: Optional[float] = _campoOpcional("meses como cliente")
   
    PhoneService: Optional[str] = _campoOpcional("Yes/No")
   
    MultipleLines: Optional[str] = _campoOpcional("Yes/No/No phone service")
   
    InternetService: Optional[str] = _campoOpcional("DSL/Fiber optic/No")
   
    OnlineSecurity: Optional[str] = _campoOpcional("Yes/No/No internet service")
   
    OnlineBackup: Optional[str] = _campoOpcional("Yes/No/No internet service")
   
    DeviceProtection: Optional[str] = _campoOpcional("Yes/No/No internet service")
   
    TechSupport: Optional[str] = _campoOpcional("Yes/No/No internet service")
   
    StreamingTV: Optional[str] = _campoOpcional("Yes/No/No internet service")
   
    StreamingMovies: Optional[str] = _campoOpcional("Yes/No/No internet service")
   
    Contract: Optional[str] = _campoOpcional("Month-to-month/One year/Two year")
   
    PaperlessBilling: Optional[str] = _campoOpcional("Yes/No")
   
    PaymentMethod: Optional[str] = _campoOpcional(
        "Electronic check/Mailed check/Bank transfer (automatic)/Credit card (automatic)")
   
    MonthlyCharges: Optional[float] = _campoOpcional("cobrança mensal")
   
    TotalCharges: Optional[float] = _campoOpcional("cobrança acumulada")

    @field_validator("*", mode="before")
    @classmethod
    def _converterNumeroParaTexto(cls, valor, info):
       
        # o LLM às vezes devolve 0/1 ou true/false em campos de texto: converte para texto
        if info.field_name not in ("tenure", "MonthlyCharges", "TotalCharges", "novoCliente") \
                and isinstance(valor, (int, float, bool)):
            return str(int(valor))
        
        return valor


class Filtro(BaseModel):
    coluna: str
    operador: Operador = "=="
    valor: str


class PlanoConsulta(BaseModel):
    tipo: Literal["estatistica", "documentacao"] = Field(
        description="estatistica = precisa calcular algo na base; documentacao = sobre o modelo, "
                    "métricas, variáveis ou limitações (responde com o cartão do modelo)")
    metrica: Optional[Metrica] = None
    filtros: list[Filtro] = Field(default_factory=list)
    agruparPor: Optional[str] = None
    colunaMedia: Optional[str] = Field(default=None, description="tenure, MonthlyCharges ou TotalCharges")


# =============================================================== nós


def roteador(estado):
    
    pergunta = estado["pergunta"]
    
    contextoExtra = ""
    
    if estado.get("aguardandoCorrecao"):
        contextoExtra = CONTEXTO_AGUARDANDO_CORRECAO.format(mensagemAnterior=ultimaMensagemDoAgente(estado))
        
    try:
        
        decisao = obterLlmEstruturado(DecisaoRota).invoke(
            [("system", PROMPT_ROTEADOR.format(contexto=contextoExtra)), ("user", pergunta)]
        )
        
        rota, justificativa = decisao.rota, decisao.justificativa
        
    except Exception as erro:
        rota, justificativa = "fora_escopo", f"falha do LLM no roteamento: {erro!r}"
    return {"rota": rota, "justificativaRota": justificativa, **valoresIniciaisDoTurno(),
            "historico": [{"papel": "usuario", "texto": pergunta}]}


def extrairDados(estado):
    
    pergunta = estado["pergunta"]
    
    try:
        dadosExtraidos = obterLlmEstruturado(DadosExtraidos).invoke(
            [("system", PROMPT_EXTRACAO.format(mensagemAnterior=ultimaMensagemDoAgente(estado))),
             ("user", pergunta)])
    
    except Exception as erro:
        return {"errosValidacao": [f"Não consegui interpretar os dados da mensagem ({type(erro).__name__}). "
                                   "Tente escrever os campos no formato campo = valor."]}
    
    camposInformados = dadosExtraidos.model_dump(include=set(COLUNAS), exclude_none=True)

    # customerID: o regex garante que um ID escrito na mensagem não se perca
    idEncontrado = PADRAO_ID_CLIENTE.search(pergunta)
    
    idCliente = ((idEncontrado.group(0).upper() if idEncontrado else None)
                or (dadosExtraidos.idCliente or "").strip().upper() or None)

    if idCliente and idCliente != estado.get("idCliente"):    # busca um cliente na base
      
        clienteEncontrado = ferramentas.buscarCliente(idCliente)
      
        if clienteEncontrado is None:
      
            return {"idCliente": idCliente, "clienteDaBase": None, "dadosCliente": {},
                    "errosValidacao": [f"customerID '{idCliente}' não foi encontrado na base Telco."]}
      
        return {"idCliente": idCliente, "clienteDaBase": clienteEncontrado,
                "dadosCliente": {**clienteEncontrado["dados"], **camposInformados}}
    if dadosExtraidos.novoCliente and not idCliente:          # outro cliente, descrito por atributos
    
        return {"idCliente": None, "clienteDaBase": None, "dadosCliente": camposInformados}
    
    # mesmo cliente: correção, complemento ou "e se"
    return {"dadosCliente": {**(estado.get("dadosCliente") or {}), **camposInformados}}


def validar(estado):
    if estado.get("errosValidacao"):        # erro já detectado (ex.: customerID inexistente)
        return {"valido": False}
    
    resultadoValidacao = validarCliente(estado.get("dadosCliente") or {})
    
    return {
            "valido": resultadoValidacao.valido,
            "dadosCliente": 
                {**estado.get("dadosCliente", {}), **resultadoValidacao.dadosNormalizados},
            "errosValidacao": resultadoValidacao.erros,
            "camposFaltando": resultadoValidacao.camposFaltando,
            "camposPreenchidosPorCoerencia": resultadoValidacao.camposPreenchidosPorCoerencia
        }


def pedirCorrecao(estado):
    
    linhas = ["Não consegui rodar o modelo porque a entrada precisa de ajustes:"]
    
    for erro in estado.get("errosValidacao") or []:
    
        linhas.append(f"- ❌ {erro}")
    
    camposFaltando = estado.get("camposFaltando") or []
    
    if camposFaltando:
    
        linhas.append(f"- Faltam {len(camposFaltando)} campo(s):")
    
        for campo in camposFaltando:
    
            opcoesAceitas = (f" — opções: {', '.join(COLUNAS_CATEGORICAS[campo])}"
                             if campo in COLUNAS_CATEGORICAS else "")
    
            linhas.append(f"  - `{campo}` ({DESCRICAO_CAMPOS[campo]}){opcoesAceitas}")
    
    linhas.append("\nEnvie só o que falta ou o que precisa ser corrigido (eu guardo o resto), "
                  "ou informe um customerID existente na base.")
    
    return montarRespostaDoAgente("\n".join(linhas), aguardandoCorrecao=True)


def prever(estado):
    
    dadosCliente = estado["dadosCliente"]
    
    resultado = {
        "previsao": ferramentas.preverChurn(dadosCliente),
        "fatores": ferramentas.calcularFatoresAssociados(dadosCliente)
    }
    
    clienteDaBase = estado.get("clienteDaBase")
    
    if clienteDaBase:
    
        atributosAlterados = {
            coluna: dadosCliente[coluna] for coluna in COLUNAS
                if dadosCliente.get(coluna) != clienteDaBase["dados"].get(coluna)
            }
        
        resultado["clienteDaBase"] = {
            "customerID": clienteDaBase["customerID"],
            "churnRealNaBase": clienteDaBase["churnRealNaBase"],
            "conjuntoNoSplit": clienteDaBase["conjunto"],
            "atributosAlterados": atributosAlterados
        }
    
    if estado.get("camposPreenchidosPorCoerencia"):
        resultado["camposPreenchidosPorCoerencia"] = estado["camposPreenchidosPorCoerencia"]
    
    return {"resultadoFerramentas": resultado, "aguardandoCorrecao": False}


def consultar(estado):
    pergunta, mensagemErro = estado["pergunta"], ""
    
    for _ in range(2):  # uma nova tentativa se a consulta planejada for inválida
      
        plano = obterLlmEstruturado(PlanoConsulta).invoke([
      
            ("system", PROMPT_CONSULTA.format(
                colunas=", ".join(COLUNAS), erroAnterior=mensagemErro,
                valoresCategoricos=json.dumps(COLUNAS_CATEGORICAS, ensure_ascii=False))),
            ("user", pergunta)])
        
        if plano.tipo == "documentacao":
            
            return {"resultadoFerramentas": {
                "tipo": "documentacao",
                "metricasModelo": ferramentas.carregarMetricas()
            }}
        
        resultadoConsulta = ferramentas.consultarBase(
            plano.metrica or "taxa_churn",
            [filtro.model_dump() for filtro in plano.filtros],
            plano.agruparPor, plano.colunaMedia
        )
        
        
        if "erro" not in resultadoConsulta:
        
            break
        
        mensagemErro = f"A consulta anterior falhou: {resultadoConsulta['erro']} Corrija o plano."
    
    return {
            "resultadoFerramentas": {
                "tipo": "estatistica",
                "plano": plano.model_dump(),
                **resultadoConsulta
            },
            "aguardandoCorrecao": False
        }


def recuperarDocumentacao(estado):
    textoBusca = estado["pergunta"]

    if estado.get("rota") == "previsao":
        textoBusca += TERMOS_BUSCA_PREVISAO

    return {"trechosDocumentacao": ferramentas.buscarDocumentacao(textoBusca)}


def responder(estado):

    resultado = estado.get("resultadoFerramentas") or {}

    trechos = estado.get("trechosDocumentacao") or []

    ehPrevisao = "previsao" in resultado
    cabecalho = montarCabecalhoPrevisao(resultado) if ehPrevisao else ""

    resultadosTexto = json.dumps(resultado, ensure_ascii=False, default=str)

    documentacaoTexto = "\n---\n".join(trecho["texto"] for trecho in trechos) or "(nenhum trecho relevante)"

    promptSistema = PROMPT_RESPOSTA.format(
        formato=FORMATO_PREVISAO if ehPrevisao else FORMATO_CONSULTA,
        pergunta=estado["pergunta"], resultados=resultadosTexto,
        documentacao=documentacaoTexto
    )
    
    guardaAcionada = False
    
    try:
        corpo = extrairTextoDaResposta(
            obterLlm().invoke([("system", promptSistema), ("user", estado["pergunta"])]).content)
    
        numerosSuspeitos = encontrarNumerosSemFonte(
            corpo, resultadosTexto + documentacaoTexto + cabecalho + estado["pergunta"]
        )
        
        if numerosSuspeitos:
        
            guardaAcionada = True
        
            corpo = (montarRespostaDeterministica(resultado)
                     + f"\n\n_(resposta do LLM descartada: citou números sem fonte: {numerosSuspeitos})_")
    
    except Exception as erro:
    
        guardaAcionada = True
    
        corpo = montarRespostaDeterministica(resultado) + f"\n\n_(LLM indisponível: {erro!r})_"
    
    fontes = sorted({f"{trecho['fonte']} › {trecho['secao']}" for trecho in trechos})
    
    rodape = f"\n\n<sub>Fontes: ferramentas do agente{'; ' + '; '.join(fontes) if fontes else ''}</sub>"
    
    texto = "\n\n".join(parte for parte in (cabecalho, corpo.strip()) if parte) + rodape
    
    return montarRespostaDoAgente(texto, guardaAcionada=guardaAcionada)


def foraEscopo(estado):
    return montarRespostaDoAgente(MENSAGEM_FORA_ESCOPO)
