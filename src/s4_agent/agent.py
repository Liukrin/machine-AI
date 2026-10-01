"""S4 Agent：LangGraph 工具调用图。线上 /api/ask（agent 模式）直接执行这张图并流式输出（见 src/s5_app/api.py）。

    presearch ──▶ agent ──(有工具调用)──▶ tools ──▶ agent …
                    │
                    └──(直接作答 / 达到轮数上限)──▶ check ──(数值核对不上，且还没改写过)──▶ repair ──▶ check
                                                     └──▶ 结束

- presearch：用用户原问题（有对话历史时拼上上一轮的问题）先检索一次，结果以一次 search_manuals 调用的
  形式放进对话。省掉模型决定「先搜什么」的那一次调用：多数问题看完这次结果就能作答，只调一次模型。
- agent：模型读对话和工具结果，决定作答还是继续调工具；工具轮数达到上限后以 tool_choice=none 强制作答。
- tools：执行模型请求的工具（tools.py）。单个工具有超时；参数错误、超时、片段不存在等都作为观察结果
  回给模型，由模型修正后重试，不中断回答。换算、核对的参数要对得上原文（numcheck.py）。
- check：以「知识库无相关内容」开头的回答按拒答处理、去掉引用；引用校验（verify.py）；数值核对——回答里的
  每个「数值 + 单位」要能在本次的资料（问题、工具返回的内容、换算/核对结果）里找到，单位也要一致（numcheck.py）。
- repair：数值核对不上时，把问题清单交给模型改写一次（不调工具）；改写后再核对，改得更差就退回初稿。

注入防护（agent.injection_guard）：手册原文在工具输出里包进 <片段原文> 标签，系统提示词说明标签内是资料、不是指令。

与 rag 模式（graph.py 的固定流水线）的区别：没有「向量距离超过阈值就直接拒答」的闸。距离仍然算，
在检索结果里标注「相关度低」提示模型换说法再查；查不到时由模型按系统提示词拒答。

Harness 约束都在 configs/config.yaml 的 agent 段：工具轮数、单轮调用数、工具超时、对话历史长度。
每一步通过 LangGraph 的 custom 流（get_stream_writer）实时发出，API 转成 SSE 的 step 事件，前端画成时间线。
"""
from __future__ import annotations

import operator
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated, TypedDict

ROOT = Path(__file__).resolve().parents[2]
for _p in ("src", "src/s3_eval", "src/s4_agent"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

import yaml  # noqa: E402
from langchain_core.messages import (  # noqa: E402
    AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langgraph.config import get_stream_writer  # noqa: E402
from langgraph.graph import END, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402

from llm_config import get_llm  # noqa: E402
from numcheck import Source, check_answer, strip_citations  # noqa: E402
from tools import TOOL_LABEL, Toolbox, ToolResult, get_corpus, run_tool, tool_specs  # noqa: E402
from verify import verify_citation  # noqa: E402

_CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
CFG: dict = _CFG["agent"]
TAU_DISTANCE = float(_CFG["s4_agent"]["tau_distance"])
PROMPT_VERSION = str(CFG["system_prompt"])
MAX_TOOL_ROUNDS = int(CFG["max_tool_rounds"])
MAX_CALLS_PER_ROUND = int(CFG["max_calls_per_round"])
TOOL_TIMEOUT_S = float(CFG["tool_timeout_s"])
REFUSAL_MARKERS = list(CFG["refusal_markers"])
REFUSAL_LEAD = "知识库无相关内容"       # 系统提示词规定的拒答开头
MAX_REPAIRS = int(CFG.get("max_repairs", 1))
DOCUMENTS: dict = (_CFG.get("s5_app") or {}).get("documents") or {}
CALC_TOOLS = ("convert_unit", "check_value")

CHUNK_ID_RE = re.compile(r"[A-Za-z0-9_]+_c\d{4}")
# 历史回答里的引用标记（[4_c0012]、（chunk_id: …）等）：去掉，免得模型把上一轮的片段当成本轮依据来引用
_HISTORY_CITATION_RE = re.compile(r"[（(]?\s*(?:引用\s*)?(?:chunk[ _-]?id\s*[:：]?\s*)?[\[【]?\s*[A-Za-z0-9_]+_c\d{4}\s*[\]】]?\s*[)）]?",
                                  re.I)


def _catalog() -> str:
    lines = []
    for doc_id, d in DOCUMENTS.items():
        title = d.get("title") or doc_id
        lines.append(f"- {doc_id}：{title}")
    return "\n".join(lines)


def render_system_prompt(guard: bool | None = None) -> str:
    """系统提示词；guard（默认取 config agent.injection_guard）为真时接上注入防护的说明。"""
    guard = bool(CFG.get("injection_guard", True)) if guard is None else guard
    raw = (ROOT / _CFG["prompts"][PROMPT_VERSION]).read_text(encoding="utf-8")
    if guard:
        raw += (ROOT / _CFG["prompts"]["agent_guard"]).read_text(encoding="utf-8")
    return raw.replace("{catalog}", _catalog()).replace("{max_rounds}", str(MAX_TOOL_ROUNDS))


SYSTEM_PROMPT = render_system_prompt()
REPAIR_PROMPT = (ROOT / _CFG["prompts"]["agent_repair"]).read_text(encoding="utf-8")
MANUAL_IDS = list(DOCUMENTS)


# --------------------------------------------------------------------------- 状态
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    history_question: str          # 上一轮用户的问题（预检索时拼在一起）
    llm_calls: int
    tool_rounds: int
    seen: dict                     # 证据池：本次给模型看过的片段 {chunk_id: {"distance": …, "shown": …}}，保持首次出现的顺序
    steps: Annotated[list, operator.add]
    usage: dict                    # 各次模型调用累计：input / output / cache_read
    answer: str
    finish_reason: str | None
    model_name: str | None
    forced_final: bool
    verification: dict
    repairs: int                   # 已经让模型改写过几次
    draft: dict | None             # 改写前的初稿及其核对结果（改写后改得更差就退回它）


def history_messages(history: list[dict]) -> tuple[list[AnyMessage], str]:
    """前端传来的对话历史 → 消息列表（只留最近几轮，回答去掉引用并截断）；同时返回上一轮用户的问题。"""
    turns = int(CFG["history_turns"])
    limit = int(CFG["history_answer_chars"])
    msgs: list[AnyMessage] = []
    clean = [h for h in history or [] if h.get("role") in ("user", "assistant") and (h.get("content") or "").strip()]
    # 只保留最近 turns 轮：从后往前数用户消息
    keep_from, n_user = 0, 0
    for i in range(len(clean) - 1, -1, -1):
        if clean[i]["role"] == "user":
            n_user += 1
            if n_user == turns:
                keep_from = i
                break
    last_q = ""
    for h in clean[keep_from:]:
        text = h["content"].strip()
        if h["role"] == "user":
            msgs.append(HumanMessage(content=text))
            last_q = text
        else:
            text = re.sub(r"[ \t]+\n", "\n", _HISTORY_CITATION_RE.sub("", text)).strip()
            if len(text) > limit:
                text = text[:limit] + "……"
            msgs.append(AIMessage(content=text))
    return msgs, last_q


def initial_state(question: str, history: list[dict] | None = None) -> dict:
    hist, last_q = history_messages(history or [])
    return {
        "messages": [*hist, HumanMessage(content=question)], "question": question, "history_question": last_q,
        "llm_calls": 0, "tool_rounds": 0, "seen": {}, "steps": [], "usage": {"input": 0, "output": 0, "cache_read": 0},
        "answer": "", "finish_reason": None, "model_name": None, "forced_final": False, "verification": {},
        "repairs": 0, "draft": None,
    }


def _user_text(state: AgentState) -> str:
    """本次对话里用户说过的话（当前问题和对话历史里的提问）：被换算值、被核对值、回答里的数可以出自这里。"""
    return "\n".join(m.content for m in state["messages"] if isinstance(m, HumanMessage) and isinstance(m.content, str))


def _calc_texts(steps: list[dict]) -> list[str]:
    """之前成功的换算、核对工具输出。"""
    return [s["output"] for s in steps if s.get("type") == "tool_end" and s.get("ok") and s["name"] in CALC_TOOLS]


def _toolbox(state: AgentState) -> Toolbox:
    return Toolbox(CFG, TAU_DISTANCE, state["seen"], context=_user_text(state), calc=_calc_texts(state["steps"]))


# --------------------------------------------------------------------------- 工具事件（工具的执行器 run_tool 在 tools.py，与 MCP Server 共用）
def _tool_events(writer, call_id: str, rnd: int, name: str, args: dict, res: ToolResult, ms: float,
                 auto: bool = False) -> dict:
    """工具结束事件（前端时间线、评测记录共用）。换算、核对的输出原文一并带上（评测做数值溯源）。"""
    ev = {"type": "tool_end", "id": call_id, "round": rnd, "name": name, "label": TOOL_LABEL.get(name, name),
          "args": args, "ok": res.ok, "summary": res.summary, "elapsed_ms": ms, "auto": auto,
          "chunks": res.chunks, "data": res.data,
          "output": res.content if name in ("convert_unit", "check_value") or not res.ok else None}
    writer(ev)
    return ev


# --------------------------------------------------------------------------- 节点
def presearch(state: AgentState) -> dict:
    """系统先用原问题检索一次（有历史时拼上上一问，追问里常省略设备型号）。"""
    writer = get_stream_writer()
    query = state["question"]
    if state.get("history_question"):
        query = f"{state['history_question']} {query}"
    tb = _toolbox(state)
    args = {"query": query}
    writer({"type": "tool_start", "id": "presearch", "round": 0, "name": "search_manuals",
            "label": TOOL_LABEL["search_manuals"], "args": args, "auto": True})
    res, ms = run_tool(tb, "search_manuals", args, TOOL_TIMEOUT_S)
    ev = _tool_events(writer, "presearch", 0, "search_manuals", args, res, ms, auto=True)
    call = {"name": "search_manuals", "args": args, "id": "presearch", "type": "tool_call"}
    return {"messages": [AIMessage(content="", tool_calls=[call]),
                         ToolMessage(content=res.content, tool_call_id="presearch", name="search_manuals",
                                     status="success" if res.ok else "error")],
            "seen": dict(tb.seen), "steps": [ev]}


@lru_cache(maxsize=2)
def _llm(forced: bool):
    llm = get_llm(max_tokens=int(CFG["max_tokens"]), stream_usage=True)
    return llm.bind_tools(tool_specs(MANUAL_IDS), tool_choice="none" if forced else "auto")


def _add_usage(state: AgentState, msg) -> tuple[dict, dict, int]:
    """把一次模型调用的用量累加进 state 的 usage。返回 (新的累计用量, 本次 usage_metadata, 本次缓存命中 token)。"""
    um = msg.usage_metadata or {}
    cache_read = ((um.get("input_token_details") or {}).get("cache_read")) or 0
    usage = {"input": state["usage"]["input"] + (um.get("input_tokens") or 0),
             "output": state["usage"]["output"] + (um.get("output_tokens") or 0),
             "cache_read": state["usage"]["cache_read"] + cache_read}
    return usage, um, cache_read


def agent(state: AgentState) -> dict:
    writer = get_stream_writer()
    rnd = state["llm_calls"] + 1
    forced = state["tool_rounds"] >= MAX_TOOL_ROUNDS
    writer({"type": "llm_start", "round": rnd, "forced": forced})
    msgs: list[AnyMessage] = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    if forced:
        msgs.append(HumanMessage(content=f"（系统提示：工具调用已达到 {MAX_TOOL_ROUNDS} 轮上限，请基于上面已有的资料直接作答；"
                                         "资料不足就按要求说明，不要再调用工具。）"))
    t0 = time.perf_counter()
    msg = _llm(forced).invoke(msgs)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    usage, um, cache_read = _add_usage(state, msg)
    meta = msg.response_metadata or {}
    calls = [] if forced else list(msg.tool_calls or [])
    if forced and msg.tool_calls:          # tool_choice=none 下理论上不会出现；出现了也不执行
        msg = AIMessage(content=msg.content or "", response_metadata=meta, usage_metadata=msg.usage_metadata)
    if calls and (msg.content or "").strip():
        writer({"type": "thought", "round": rnd, "text": msg.content})
    if forced:
        writer({"type": "forced_final", "round": rnd})
    step = {"type": "llm", "round": rnd, "forced": forced, "tool_calls": [c["name"] for c in calls],
            "elapsed_ms": ms, "input_tokens": um.get("input_tokens"), "output_tokens": um.get("output_tokens"),
            "cache_read_tokens": cache_read, "finish_reason": meta.get("finish_reason")}
    return {"messages": [msg], "llm_calls": rnd, "usage": usage, "steps": [step],
            "forced_final": forced or state["forced_final"], "finish_reason": meta.get("finish_reason"),
            "model_name": meta.get("model_name")}


def tools(state: AgentState) -> dict:
    writer = get_stream_writer()
    tb = _toolbox(state)
    rnd = state["llm_calls"]
    calls = state["messages"][-1].tool_calls
    out: list[AnyMessage] = []
    steps: list[dict] = []
    for i, call in enumerate(calls):
        name, args = call["name"], call.get("args") or {}
        writer({"type": "tool_start", "id": call["id"], "round": rnd, "name": name,
                "label": TOOL_LABEL.get(name, name), "args": args})
        if i >= MAX_CALLS_PER_ROUND:
            res, ms = ToolResult(False, f"工具调用失败：一轮最多执行 {MAX_CALLS_PER_ROUND} 个工具调用，这一个没有执行。",
                                 "超出单轮调用数上限"), 0.0
        else:
            res, ms = run_tool(tb, name, args, TOOL_TIMEOUT_S)
        steps.append(_tool_events(writer, call["id"], rnd, name, args, res, ms))
        out.append(ToolMessage(content=res.content, tool_call_id=call["id"], name=name,
                               status="success" if res.ok else "error"))
    return {"messages": out, "seen": dict(tb.seen), "tool_rounds": state["tool_rounds"] + 1, "steps": steps}


def is_refusal(answer: str) -> bool:
    """拒答：含拒答话术且没有引用任何片段（与答案级评测 scoring.is_refusal 同一口径）。"""
    text = "".join(answer.split())
    return any(m in text for m in REFUSAL_MARKERS) and not CHUNK_ID_RE.search(answer)


_CHUNK_BLOCK = re.compile(r"\n(?=\[[A-Za-z0-9_]+_c\d{4}\] )")


def _clean_refusal(answer: str) -> str:
    """拒答不带引用：系统提示词这样要求，模型偶尔仍会带上所查片段的编号。去掉引用标记和随之留下的空格。"""
    return re.sub(r"[ \t]+(?=[，。；：、,.;:）)])", "", strip_citations(answer)).strip()


def _answer_sources(state: AgentState) -> list[Source]:
    """核对回答里的数值用的资料：用户的话、系统提示词里的手册清单（「API 610」这类书名里的数），加上本次工具成功返回的内容。
    检索、查表、读片段的输出按片段切开（单位按段核对）；换算、核对的输出可按舍入匹配（3.333 → 3.33）。
    出错的工具输出不算（报错信息里会复述模型给的错误参数）。"""
    out = [Source(_user_text(state)), Source(_catalog())]
    for m in state["messages"]:
        if not isinstance(m, ToolMessage) or getattr(m, "status", "success") == "error" or not isinstance(m.content, str):
            continue
        if m.name in CALC_TOOLS:
            out.append(Source(m.content, kind="calc"))
        else:
            out += [Source(block) for block in _CHUNK_BLOCK.split(m.content) if block.strip()]
    return out


def _verify(state: AgentState) -> tuple[str, dict, int]:
    """核对最后一条回答：拒答处理、引用校验、数值核对。返回 (答案, 核对结果, 数值问题个数)。"""
    answer = (state["messages"][-1].content or "").strip()
    if re.sub(r"^[\s#>*_`\-]+", "", answer).startswith(REFUSAL_LEAD):
        answer = _clean_refusal(answer)
    corpus = get_corpus()
    evidence = [corpus.by_id[cid] for cid in state["seen"] if cid in corpus.by_id]
    # 换算、核对工具的输出也是合法依据（「约合 3.33 L/min」「高出上限 6℃」在手册原文里没有）
    v = verify_citation(answer, evidence, extra_grounding="\n".join(_calc_texts(state["steps"])))
    refused = is_refusal(answer)
    n_checked, issues = (0, []) if refused or not answer else check_answer(answer, _answer_sources(state))
    verification = {
        "suspicious_count": v.get("suspicious_count", 0), "suspicious": v.get("suspicious", []),
        "cited_ids": v.get("cited_ids", []), "fabricated_ids": v.get("fabricated_ids", []),
        "refused": refused,
        "numbers_checked": n_checked,
        "number_issues": [{"quantity": i.quantity, "status": i.status, "source_units": i.source_units,
                           "sentence": i.sentence, "detail": i.describe()} for i in issues],
        "repair": None,
    }
    return answer, verification, len(issues)


def check(state: AgentState) -> dict:
    answer, verification, n_issues = _verify(state)
    if state["repairs"] == 0:
        draft = {"answer": answer, "verification": verification, "issues": n_issues} if n_issues else None
        return {"answer": answer, "verification": verification, "draft": draft}
    # 改写之后再核对：改得更差（问题更多，或答案为空）就退回初稿
    draft = state["draft"]
    keep = bool(answer) and n_issues <= draft["issues"]
    final_answer, final_v = (answer, verification) if keep else (draft["answer"], draft["verification"])
    final_v = {**final_v, "repair": {"issues_before": draft["issues"], "issues_after": n_issues,
                                      "kept": "repaired" if keep else "draft",
                                      "draft_issues": [i["detail"] for i in draft["verification"]["number_issues"]]}}
    return {"answer": final_answer, "verification": final_v}


def repair(state: AgentState) -> dict:
    """把数值核对的问题清单交给模型改写一次（tool_choice=none，不调工具；说明只在这次调用里，不进对话记录）。"""
    writer = get_stream_writer()
    rnd = state["llm_calls"] + 1
    issues = state["verification"]["number_issues"]
    writer({"type": "repair", "round": rnd, "issues": [i["detail"] for i in issues]})
    writer({"type": "llm_start", "round": rnd, "forced": True})
    note = REPAIR_PROMPT.replace("{issues}", "\n".join(f"- {i['detail']}" for i in issues))
    t0 = time.perf_counter()
    msg = _llm(True).invoke([SystemMessage(content=SYSTEM_PROMPT), *state["messages"], HumanMessage(content=note)])
    ms = round((time.perf_counter() - t0) * 1000, 1)
    usage, um, cache_read = _add_usage(state, msg)
    meta = msg.response_metadata or {}
    step = {"type": "llm", "round": rnd, "forced": True, "repair": True, "tool_calls": [], "elapsed_ms": ms,
            "input_tokens": um.get("input_tokens"), "output_tokens": um.get("output_tokens"),
            "cache_read_tokens": cache_read, "finish_reason": meta.get("finish_reason")}
    return {"messages": [AIMessage(content=msg.content or "", response_metadata=meta, usage_metadata=msg.usage_metadata)],
            "llm_calls": rnd, "usage": usage, "steps": [step], "repairs": state["repairs"] + 1,
            "finish_reason": meta.get("finish_reason"), "model_name": meta.get("model_name")}


def _route(state: AgentState) -> str:
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls and not state["forced_final"] else "check"


def _route_check(state: AgentState) -> str:
    return "repair" if state.get("draft") and state["repairs"] < MAX_REPAIRS else END


@lru_cache(maxsize=1)
def build_agent_graph():
    g = StateGraph(AgentState)
    g.add_node("presearch", presearch)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("check", check)
    g.add_node("repair", repair)
    g.set_entry_point("presearch")
    g.add_edge("presearch", "agent")
    g.add_conditional_edges("agent", _route, {"tools": "tools", "check": "check"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("check", _route_check, {"repair": "repair", END: END})
    g.add_edge("repair", "check")
    return g.compile()


def estimate_cost(usage: dict, pricing: dict) -> float:
    """按 config llm_pricing 估算一次回答的费用（元）：缓存命中与未命中的输入分开计价。"""
    hit = usage.get("cache_read") or 0
    miss = max(0, (usage.get("input") or 0) - hit)
    return (hit * pricing["input_cache_hit"] + miss * pricing["input_cache_miss"]
            + (usage.get("output") or 0) * pricing["output"]) / 1e6


def main() -> None:
    """命令行试问：python src/s4_agent/agent.py 「问题」"""
    sys.stdout.reconfigure(encoding="utf-8")
    question = " ".join(sys.argv[1:]) or "Model 3700 运行中测得轴承温度 85℃，换算成华氏度是多少？是否超出手册给出的正常范围？"
    final = None
    for mode, payload in build_agent_graph().stream(initial_state(question), stream_mode=["custom", "values"]):
        if mode == "custom" and payload["type"] == "tool_end":
            print(f"[{payload['round']}] {payload['label']} {payload['args']} → {payload['summary']}（{payload['elapsed_ms']} ms）")
        elif mode == "custom" and payload["type"] in ("thought", "forced_final"):
            print(f"[{payload['round']}] {payload['type']}: {payload.get('text', '')}")
        elif mode == "custom" and payload["type"] == "repair":
            print(f"[{payload['round']}] 数值核对不上，改写一次：{'；'.join(payload['issues'])}")
        elif mode == "values":
            final = payload
    print("\n" + final["answer"])
    print(f"\n模型调用 {final['llm_calls']} 次，工具轮数 {final['tool_rounds']}，用量 {final['usage']}，"
          f"引用校验 {final['verification']}")


if __name__ == "__main__":
    main()
