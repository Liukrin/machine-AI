"""LLM backend configuration. To switch providers, change only this file."""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

LLM_CONFIG = {
    "base_url": "https://api.deepseek.com",
    "model": "deepseek-chat",
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "temperature": 0.0,
}


def get_llm():
    """Return a configured ChatOpenAI instance."""
    if not LLM_CONFIG["api_key"]:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set. Add your key to .env, then restart."
        )
    return ChatOpenAI(
        base_url=LLM_CONFIG["base_url"],
        model=LLM_CONFIG["model"],
        api_key=LLM_CONFIG["api_key"],
        temperature=LLM_CONFIG["temperature"],
    )
