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


def max_tokens_body(max_tokens: int) -> dict:
    """生成上限的请求体参数。

    langchain-openai 1.x 会把 max_tokens 参数改名为 max_completion_tokens 发出，
    DeepSeek 只认 max_tokens，收到 max_completion_tokens 会忽略（上限等于没设）。
    所以不走 langchain 的 max_tokens，而是经 extra_body 原样传 max_tokens。
    """
    return {"extra_body": {"max_tokens": max_tokens}}


def get_llm(max_tokens: int | None = None):
    """Return a configured ChatOpenAI instance（max_tokens 为生成上限，None 则用服务端默认值）。"""
    if not LLM_CONFIG["api_key"]:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set. Add your key to .env, then restart."
        )
    return ChatOpenAI(
        base_url=LLM_CONFIG["base_url"],
        model=LLM_CONFIG["model"],
        api_key=LLM_CONFIG["api_key"],
        temperature=LLM_CONFIG["temperature"],
        **(max_tokens_body(max_tokens) if max_tokens is not None else {}),
    )
