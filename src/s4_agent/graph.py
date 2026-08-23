"""S4 任务一：LangGraph 三节点骨架（retrieve -> generate -> format）。

只求跑通，不加拒答、不加引用校验、不做多轮。检索复用 hybrid_retrieval 的
hybrid_retrieve，LLM 复用 llm_config.get_llm。

用法：
    python src/s4_agent/graph.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "s3_eval"))

from langgraph.graph import StateGraph, END  # noqa: E402
from langchain_core.messages import SystemMessage, HumanMessage  # noqa: E402
import yaml  # noqa: E402
from llm_config import get_llm  # noqa: E402
from hybrid_retrieval import hybrid_retrieve, build_search_text, vector_top1_distance  # noqa: E402
from verify import verify_citation  # noqa: E402

SYSTEM_PROMPT = (
    "你是设备维修手册问答助手。请严格依据检索到的内容回答。\n"
    "1. 只根据检索到的内容回答，不得补充原文没有的信息；\n"
    "2. 答案末尾必须标注引用的 chunk_id；\n"
    "3. 若检索内容不足以回答，直接说「检索内容不足」；\n"
    "4. 表格内容按原样呈现关键行，不要重述整张表。"
)


class AgentState(TypedDict):
    question: str
    retrieved: list
    retrieval_scores: list
    answer: str
    verification: dict
    status: str


_last_usage: dict = {}
_raw_answer: str = ""
_llm_calls: int = 0


def _load_tau() -> float:
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    return float(cfg["s4_agent"]["tau_distance"])


TAU_DISTANCE = _load_tau()


def retrieve(state: AgentState) -> dict:
    """混合检索 top_5，chunk 的 text/table_html 随 chunk dict 一并入 state。"""
    chunks, scores = hybrid_retrieve(state["question"], top_k=5)
    return {"retrieved": chunks, "retrieval_scores": scores, "status": "retrieved"}


def generate(state: AgentState) -> dict:
    global _last_usage, _raw_answer, _llm_calls
    _llm_calls += 1
    context = "\n\n".join(
        f"[{c['chunk_id']}]\n{build_search_text(c)}" for c in state["retrieved"]
    )
    user = f"问题：{state['question']}\n\n检索到的内容：\n{context}"
    llm = get_llm()
    msg = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)], max_tokens=500)
    answer = (msg.content or "").strip()
    _raw_answer = answer

    um = getattr(msg, "usage_metadata", None) or {}
    if not um:
        um = (getattr(msg, "response_metadata", {}) or {}).get("token_usage", {})
    _last_usage = {
        "input": um.get("input_tokens", um.get("prompt_tokens")),
        "output": um.get("output_tokens", um.get("completion_tokens")),
        "total": um.get("total_tokens"),
    }
    return {"answer": answer, "status": "generated"}


def verify_node(state: AgentState) -> dict:
    """引用校验：bigram 可疑句 + chunk_id 有效性 + 无引用句。"""
    v = verify_citation(state["answer"], state["retrieved"], extra_grounding="")
    return {"verification": v, "status": "verified"}


def format_node(state: AgentState) -> dict:
    """把答案、引用来源与校验结果整理成最终输出。"""
    sources = "、".join(f"{c['chunk_id']}" for c in state["retrieved"])
    v = state.get("verification") or {}
    lines = [state["answer"], "——", f"检索来源：{sources}"]
    if v.get("checked"):
        lines.append(f"引用校验：可疑 {v['suspicious_count']}/{v['total_sentences']} 句")
        if v.get("fabricated_ids"):
            lines.append("⚠️ 引用了未检索到的 chunk_id：" + "、".join(v["fabricated_ids"]))
        if v.get("suspicious"):
            lines.append("⚠️ 可疑句：" + " ｜ ".join(v["suspicious"]))
        if v.get("uncited_sentences"):
            lines.append("无引用句：" + " ｜ ".join(v["uncited_sentences"]))
    final = "\n\n".join(lines)
    return {"answer": final, "status": "done"}


def route_after_retrieve(state: AgentState) -> str:
    """拒答闸一：向量 top-1 distance 超过阈值则直接到 reject。"""
    d = vector_top1_distance(state["question"])
    return "reject" if d > TAU_DISTANCE else "generate"


def reject(state: AgentState) -> dict:
    return {"answer": "知识库无相关内容", "status": "rejected"}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("verify", verify_node)
    g.add_node("format", format_node)
    g.add_node("reject", reject)
    g.set_entry_point("retrieve")
    g.add_conditional_edges("retrieve", route_after_retrieve, {"reject": "reject", "generate": "generate"})
    g.add_edge("generate", "verify")
    g.add_edge("verify", "format")
    g.add_edge("format", END)
    g.add_edge("reject", END)
    return g.compile()


VERIFY_QUESTIONS = [
    "联轴器对中允差是多少",
    "轴承润滑脂多久更换一次",
    "挖掘机液压泵压力多少正常",
    "泵轴对应的部件编号是多少",
]


def main() -> None:
    graph = build_graph()
    for q in VERIFY_QUESTIONS:
        state = graph.invoke({
            "question": q, "retrieved": [], "retrieval_scores": [],
            "answer": "", "verification": {}, "status": "start",
        })
        print("=" * 70)
        print(f"问题：{q}")
        print(f"状态：{state['status']}")
        if state["status"] == "rejected":
            print(f"  答案：{state['answer']}")
            continue
        v = state.get("verification") or {}
        print(f"  检索 chunk：{', '.join(c['chunk_id'] for c in state['retrieved'])}")
        if v.get("checked"):
            print(f"  校验：总句 {v['total_sentences']}，可疑 {v['suspicious_count']}")
            print(f"    cited_ids = {v['cited_ids']}")
            print(f"    fabricated_ids = {v['fabricated_ids']}")
            print(f"    uncited_sentences = {v['uncited_sentences']}")
            if v["suspicious"]:
                print(f"    suspicious = {v['suspicious']}")


if __name__ == "__main__":
    main()
