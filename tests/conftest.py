"""pytest 公共设置。

两类测试：
- 不带标记的：只用仓库里的代码、配置和评测集，CI 上跑（GitHub Actions 没有知识库，也没有 API Key）。
- @pytest.mark.kb：要用本地知识库（data/、chroma_db/、models/，都不入库）。缺文件时自动跳过，并在汇总里写明原因。
大模型一律不真调用：要走 Agent 图的测试用 fake_llm 替换模型，按剧本返回回答或工具调用。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in ("src", "src/s3_eval", "src/s4_agent", "src/s5_app", "src/answer_eval"):
    sys.path.insert(0, str(ROOT / _p))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

KB_FILES = ["data/chunks/chunks.jsonl", "data/chunks/bm25_index.pkl", "chroma_db/chroma.sqlite3",
            "models/bge-small-zh-v1.5/config.json"]
MISSING_KB = [f for f in KB_FILES if not (ROOT / f).exists()]


def pytest_configure(config):
    config.addinivalue_line("markers", "kb: 要用本地知识库（data/、chroma_db/、models/，不入库），缺了自动跳过")


def pytest_collection_modifyitems(config, items):
    if not MISSING_KB:
        return
    skip = pytest.mark.skip(reason=f"没有本地知识库（缺 {', '.join(MISSING_KB)}），先按 README「构建知识库」建好")
    for item in items:
        if "kb" in item.keywords:
            item.add_marker(skip)


class FakeLLM:
    """按剧本依次返回 AIMessage 的假模型：替换 agent._llm(forced) 的返回值，记录每次收到的消息。"""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[list] = []

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls.append(list(messages))
        if not self.replies:
            raise AssertionError("FakeLLM 的剧本用完了：模型被多调用了一次")
        reply = self.replies.pop(0)
        if isinstance(reply, str):
            reply = AIMessage(content=reply)
        reply.usage_metadata = reply.usage_metadata or {"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}
        reply.response_metadata = {"finish_reason": "stop", "model_name": "fake-llm", **(reply.response_metadata or {})}
        return reply


@pytest.fixture
def fake_llm(monkeypatch):
    """用法：llm = fake_llm(["初稿", "改写稿"])；之后跑 Agent 图，模型调用按顺序取剧本。"""
    def install(replies):
        import agent

        llm = FakeLLM(replies)
        monkeypatch.setattr(agent, "_llm", lambda forced: llm)
        return llm
    return install


def tool_call(name: str, args: dict, call_id: str = "call_1"):
    """剧本里的一步工具调用（模型返回的 AIMessage 带 tool_calls）。"""
    from langchain_core.messages import AIMessage

    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])
