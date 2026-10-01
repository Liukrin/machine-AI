"""LLM backend configuration. To switch providers, change only this file."""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

LLM_CONFIG = {
    "base_url": "https://api.deepseek.com",
    # DeepSeek-V4.1-Flash。旧名 deepseek-chat 已公告停用，它对应的就是 Flash 的非思考模式（2026-09-30 起改用新名）
    "model": "deepseek-flash",
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "temperature": 0.0,
    "timeout": 60,        # 单次请求超时（秒）：连不上或长时间收不到数据就报错，不无限等待
    "max_retries": 2,     # 连接出错、超时、限流、服务端 5xx 时自动重试的次数
}

# deepseek-flash 默认开启思考模式（先输出一段推理再作答，更慢、更贵）。本项目一律用非思考模式，与旧名 deepseek-chat 的行为一致
THINKING_OFF = {"thinking": {"type": "disabled"}}


def client_kwargs(temperature: float | None = None, **body) -> dict:
    """ChatOpenAI 的公共参数（get_llm、评审模型、旧评测脚本共用）。body 里的键经 extra_body 原样发给 DeepSeek。

    max_tokens 要放进 body：langchain-openai 1.x 会把 max_tokens 参数改名为 max_completion_tokens 发出，
    DeepSeek 只认 max_tokens，收到 max_completion_tokens 会忽略（上限等于没设）。
    """
    if not LLM_CONFIG["api_key"]:
        raise RuntimeError("DEEPSEEK_API_KEY not set. Add your key to .env, then restart.")
    return {
        "base_url": LLM_CONFIG["base_url"],
        "model": LLM_CONFIG["model"],
        "api_key": LLM_CONFIG["api_key"],
        "temperature": LLM_CONFIG["temperature"] if temperature is None else temperature,
        "timeout": LLM_CONFIG["timeout"],
        "max_retries": LLM_CONFIG["max_retries"],
        "extra_body": {**THINKING_OFF, **body},
    }


def get_llm(max_tokens: int | None = None, stream_usage: bool = False):
    """Return a configured ChatOpenAI instance（max_tokens 为生成上限，None 则用服务端默认值）。

    stream_usage=True 时流式调用也返回 token 用量（Agent 在 LangGraph 里流式调用模型时需要）。
    """
    body = {} if max_tokens is None else {"max_tokens": max_tokens}
    return ChatOpenAI(**client_kwargs(**body), **({"stream_usage": True} if stream_usage else {}))
