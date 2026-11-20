"""Configuração do LLM. Escolha o provedor por variáveis de ambiente (arquivo .env):

    LLM_PROVIDER = ollama | groq | openai | google
    LLM_MODEL    = nome do modelo (ex.: qwen2.5:7b, llama-3.3-70b-versatile, gpt-4o-mini, gemini-2.0-flash)

Chaves de API ficam SOMENTE no .env (que está no .gitignore): GROQ_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()

PADROES = {
    "ollama": "qwen2.5:7b",
    "groq": "llama-3.3-70b-versatile",
    "openai": "gpt-4o-mini",
    "google": "gemini-2.0-flash",
}


def provedor() -> str:
    return os.getenv("LLM_PROVIDER", "ollama").lower()


def nome_modelo() -> str:
    return os.getenv("LLM_MODEL") or PADROES.get(provedor(), "")


@lru_cache(maxsize=1)
def obter_llm():
    p, m = provedor(), nome_modelo()
    if p == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=m, temperature=0,
                          base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    if p == "groq":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=m, temperature=0, api_key=os.environ["GROQ_API_KEY"],
                          base_url="https://api.groq.com/openai/v1")
    if p == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=m, temperature=0)
    if p == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=m, temperature=0)
    raise ValueError(f"LLM_PROVIDER desconhecido: {p}")


def llm_estruturado(schema):
    """LLM que responde obrigatoriamente no formato do schema Pydantic."""
    metodo = {"ollama": "json_schema", "groq": "function_calling",
              "openai": "function_calling"}.get(provedor())
    llm = obter_llm()
    return llm.with_structured_output(schema, method=metodo) if metodo else llm.with_structured_output(schema)
