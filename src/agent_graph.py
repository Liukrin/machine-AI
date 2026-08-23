"""LangGraph three-node agent: Watcher -> Diagnostician -> Reporter.

Node 1 (Watcher): deterministic Python, no LLM. Reads s_level/s_shape from
  app_data/demo_windows.npz, compares against tau thresholds.
Node 2 (Diagnostician): calls query_rag.retrieve_or_reject; if hit, calls
  DeepSeek (via llm_config) with retrieved context.
Node 3 (Reporter): formats work order as JSON + Markdown.
Post-reporter: verify_citation() — hard check that LLM output is grounded.
"""

import os, sys, json, re, time
from datetime import datetime
from typing import TypedDict, List, Optional, Dict, Any

import numpy as np
from langgraph.graph import StateGraph, END

# Project root for data access
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

# ---------- State ----------
class AgentState(TypedDict):
    # Input
    window_id: int
    component: str             # target component for diagnosis
    # Watcher
    alert_level: str          # "normal" | "warning" | "danger"
    s_level: float
    s_shape: float
    X_valve: int
    X_pump: int
    X_accum: int
    cooler_target: int
    verdict_original: str     # original verdict from demo data
    # Diagnostician
    retrieval_hit: bool
    retrieved: List[str]       # list of retrieved entry ids
    retrieved_texts: List[str] # list of full entry texts
    retrieval_score: float
    diagnosis_raw: str         # raw LLM output
    mapped_severity: str       # deterministic severity from severity_map
    retrieved_severity: str    # actual severity of top-1 retrieved entry
    severity_match: bool       # whether mapped == retrieved
    severity_fallback: bool    # whether severity had to fall back
    fallback_reason: str       # reason for fallback, if any
    severity_unavailable: bool # severity not available for this component
    # Reporter
    work_order_md: str
    work_order_json: str
    # Verification
    verification: Dict[str, Any]
    # Flow control
    status: str                # "completed" | "rejected" | "normal_skip"

# ---------- Load thresholds once ----------
DATA = np.load(
    os.path.join(os.path.dirname(__file__), "..", "app_data", "demo_windows.npz"),
    allow_pickle=True,
)
TAU_LEVEL = float(DATA["tau_level"])
TAU_SHAPE = float(DATA["tau_shape"])

# ---------- Node 1: Watcher (deterministic, no LLM) ----------
def watcher(state: AgentState) -> AgentState:
    wid = state["window_id"]
    # Read from demo data
    sl = float(DATA["s_level"][wid])
    ss = float(DATA["s_shape"][wid])
    state["s_level"] = sl
    state["s_shape"] = ss
    state["X_valve"] = int(DATA["X_valve"][wid])
    state["X_pump"] = int(DATA["X_pump"][wid])
    state["X_accum"] = int(DATA["X_accum"][wid])
    state["cooler_target"] = int(DATA["cooler_target"][wid])
    state["verdict_original"] = str(DATA["verdict"][wid])
    if not state.get("component"):
        state["component"] = "冷却器"  # default for demo data

    # Decision: s_level/s_shape rule (same as demo)
    if (sl > TAU_LEVEL) or (ss > TAU_SHAPE):
        state["alert_level"] = "danger"
        state["status"] = "running"
    else:
        state["alert_level"] = "normal"
        state["status"] = "completed"  # Bug 2 fix: normal path ends here
    return state

# ---------- Node 2: Diagnostician ----------
def diagnostician(state: AgentState) -> AgentState:
    from query_rag import retrieve_or_reject
    from severity_map import map_severity

    # Determine severity from s_level (deterministic, no LLM)
    comp = state.get("component", "冷却器")
    sev = map_severity(comp, state["s_level"]) if comp == "冷却器" else None

    # Build a query from the state
    query_parts = [f"冷却器状态{cooler_status(state['cooler_target'])}"]
    query_parts.append(
        f"工况: valve={state['X_valve']}, pump_leak={state['X_pump']}, "
        f"accum={state['X_accum']}bar"
    )
    if state["s_level"] > TAU_LEVEL:
        query_parts.append(f"温度基线偏移 {state['s_level']:.1f}°C")
    if state["s_shape"] > TAU_SHAPE:
        query_parts.append("压力波形畸变")
    query = "；".join(query_parts)

    result = retrieve_or_reject(query, component=state.get("component", "冷却器"), severity=sev)
    state["retrieval_hit"] = result["hit"]
    state["retrieval_score"] = result.get("best_score", None)
    state["mapped_severity"] = sev or "无分级依据"
    state["severity_fallback"] = result.get("severity_fallback", False)
    state["severity_unavailable"] = (sev is None)
    if state["severity_unavailable"]:
        state["fallback_reason"] = f"组件「{comp}」无实测分级阈值，未按严重度过滤"
    elif state["severity_fallback"]:
        state["fallback_reason"] = f"组件「{comp}」无严重度「{sev}」的条目，回退至同组件最近条目"

    if not result["hit"]:
        state["retrieved"] = []
        state["retrieved_texts"] = []
        state["retrieved_severity"] = ""
        state["severity_match"] = False
        state["diagnosis_raw"] = ""
        wo = {"conclusion": "知识库无对应条目，需人工介入"}
        state["work_order_json"] = json.dumps(wo, ensure_ascii=False, allow_nan=False)
        state["work_order_md"] = "## 诊断结论\n\n知识库无对应条目，需人工介入。"
        state["verification"] = {"checked": False, "reason": "retrieval rejected"}
        state["status"] = "rejected"
        return state

    # Collect retrieved texts + severity audit trail
    retrieved_docs = result["results"]
    state["retrieved"] = [doc.metadata["id"] for doc, score in retrieved_docs]
    state["retrieved_severity"] = retrieved_docs[0][0].metadata.get("severity", "未知")
    state["severity_match"] = (
        (state["mapped_severity"] == state["retrieved_severity"])
        if not state["severity_unavailable"] else False
    )
    raw_texts = []
    for doc, score in retrieved_docs:
        txt = (
            f"[条目 {doc.metadata['id']}] 组件:{doc.metadata['component']} | "
            f"严重度:{doc.metadata['severity']}\n{doc.page_content}"
        )
        raw_texts.append(txt)
    state["retrieved_texts"] = raw_texts

    # Call LLM
    try:
        from llm_config import get_llm
        llm = get_llm()
        system_prompt = (
            "你是一名液压系统故障诊断工程师。你只能引用下方检索到的原文内容，"
            "禁止补充任何未出现在原文中的维修建议或技术细节。"
            "若原文不足以支撑判断，直接回答「检索内容不足」。"
            "输出格式：先给出诊断结论（1-2句），再列出建议处置措施（编号列表）。"
        )
        user_prompt = f"故障描述：{query}\n\n检索到的参考条目：\n\n"
        for txt in raw_texts:
            user_prompt += f"{txt}\n\n"

        response = llm.invoke(
            [("system", system_prompt), ("user", user_prompt)]
        )
        state["diagnosis_raw"] = response.content
    except RuntimeError as e:
        state["diagnosis_raw"] = f"[LLM 调用失败: {e}]"
    except Exception as e:
        state["diagnosis_raw"] = f"[LLM 调用失败: {e}]"

    state["status"] = "diagnosed"
    return state

def cooler_status(target: int) -> str:
    if target == 100: return "正常"
    if target == 20: return "效率降低至20%"
    if target == 3: return "接近失效(3%)"
    return f"未知({target})"

# ---------- Node 3: Reporter ----------
def reporter(state: AgentState) -> AgentState:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wid = state["window_id"]
    alert = state["alert_level"]
    sl = state["s_level"]
    ss = state["s_shape"]

    # Build work order
    wo = {
        "工单编号": f"WO-{wid:04d}",
        "生成时间": now,
        "告警级别": alert,
        "工况": {
            "valve": state["X_valve"],
            "pump_leakage": state["X_pump"],
            "accumulator_bar": state["X_accum"],
            "cooler_target": state["cooler_target"],
        },
        "偏离指标": {
            "s_level_C": round(sl, 4),
            "tau_level": round(TAU_LEVEL, 4),
            "s_shape": round(ss, 4),
            "tau_shape": round(TAU_SHAPE, 4),
        },
        "检索命中": state["retrieval_hit"],
        "检索条目": state["retrieved"],
        "检索分数": round(state["retrieval_score"], 4) if state["retrieval_score"] is not None else None,
        "严重度审计": {
            "确定性映射": state.get("mapped_severity", ""),
            "检索到的严重度": state.get("retrieved_severity", ""),
            "两者一致": state.get("severity_match", False),
            "回退触发": state.get("severity_fallback", False),
            "回退原因": state.get("fallback_reason", ""),
            "分级不可用": state.get("severity_unavailable", False),
        },
        "诊断结论": state["diagnosis_raw"],
        "原始判决": state["verdict_original"],
    }

    state["work_order_json"] = json.dumps(wo, ensure_ascii=False, indent=2, allow_nan=False)

    # Markdown version
    md = f"# 液压泵站预防性维护工单\n\n"
    md += f"**工单编号:** WO-{wid:04d}  \n**生成时间:** {now}  \n**告警级别:** {alert}\n\n"
    md += f"## 工况参数\n\n"
    md += f"| 参数 | 值 |\n|------|-----|\n"
    md += f"| 阀开度 (valve) | {state['X_valve']} |\n"
    md += f"| 泵泄漏 (pump) | {state['X_pump']} |\n"
    md += f"| 蓄能器 (accum) | {state['X_accum']} bar |\n"
    md += f"| 冷却器档位 | {state['cooler_target']} |\n\n"
    md += f"## 偏离指标\n\n"
    md += f"| 指标 | 实测值 | 阈值 |\n|------|--------|------|\n"
    md += f"| s_level | {sl:.4f} °C | {TAU_LEVEL:.4f} |\n"
    md += f"| s_shape | {ss:.4f} | {TAU_SHAPE:.4f} |\n\n"
    md += f"## 检索结果\n\n"
    md += f"命中: {state['retrieval_hit']} | 条目: {', '.join(state['retrieved'])}\n\n"
    md += f"## 诊断结论\n\n{state['diagnosis_raw']}\n\n"
    md += f"## 引用校验\n\n"
    v = state.get("verification", {})
    if v.get("checked"):
        md += f"可疑句子数: {v.get('suspicious_count', 0)}/{v.get('total_sentences', 0)}\n\n"
        for s in v.get("suspicious", []):
            md += f"- ⚠️ {s}\n"
    else:
        md += f"状态: {v.get('reason', '未执行')}\n"

    state["work_order_md"] = md
    state["status"] = "completed"
    return state

# ---------- Citation verification (pure Python) ----------
def verify_citation(llm_output: str, retrieved_texts: List[str],
                   extra_grounding: str = "") -> Dict[str, Any]:
    """Check each sentence in LLM output for grounding in retrieved texts.

    Args:
        llm_output: raw LLM diagnosis text
        retrieved_texts: full texts of retrieved KB entries
        extra_grounding: additional verifiable content (e.g. measured values
            from the deterministic Watcher layer), treated as legitimate reference
    """
    if not llm_output or not retrieved_texts:
        return {"checked": False, "reason": "empty input"}

    # Combine retrieved texts with extra grounding
    reference = " ".join(retrieved_texts) + " " + extra_grounding

    # Sentence split: by period, exclamation, question mark, or newline
    sentences = re.split(r"[。！？\n]", llm_output)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    suspicious = []
    for sent in sentences:
        # Skip headings / short structural lines
        if len(sent) < 8:
            continue
        if sent.endswith("：") or sent.endswith(":"):
            continue
        # Skip pure number lines
        if re.fullmatch(r"[\d\s\.\-\+]+", sent):
            continue

        # Extract Chinese characters for bigram matching
        chars = re.findall(r"[一-鿿]", sent)
        if len(chars) < 4:
            continue

        # Build character bigrams
        bigrams = set()
        for i in range(len(chars) - 1):
            bigrams.add(chars[i] + chars[i + 1])

        if not bigrams:
            continue

        # Count how many bigrams appear in reference
        found = sum(1 for bg in bigrams if bg in reference)
        if found / len(bigrams) < 0.15:
            suspicious.append(sent)

    return {
        "checked": True,
        "total_sentences": len(sentences),
        "suspicious_count": len(suspicious),
        "suspicious": suspicious,
    }

def verify_node(state: AgentState) -> AgentState:
    # Build extra grounding from Watcher's deterministic measurements
    extra = (
        f"s_level:{state.get('s_level',0):.2f} "
        f"s_shape:{state.get('s_shape',0):.4f} "
        f"tau_level:{TAU_LEVEL:.4f} tau_shape:{TAU_SHAPE:.4f} "
        f"cooler_target:{state.get('cooler_target',0)} "
        f"alert_level:{state.get('alert_level','')}"
    )
    state["verification"] = verify_citation(
        state.get("diagnosis_raw", ""),
        state.get("retrieved_texts", []),
        extra_grounding=extra,
    )
    return state

# ---------- Routing ----------
def route_after_watcher(state: AgentState) -> str:
    if state["alert_level"] == "normal":
        return "END"
    return "diagnostician"

def route_after_diagnostician(state: AgentState) -> str:
    if not state.get("retrieval_hit", False):
        return "END"
    return "reporter"

# ---------- Build graph ----------
def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("watcher", watcher)
    workflow.add_node("diagnostician", diagnostician)
    workflow.add_node("reporter", reporter)
    workflow.add_node("verify", verify_node)

    workflow.set_entry_point("watcher")

    workflow.add_conditional_edges(
        "watcher",
        route_after_watcher,
        {"END": END, "diagnostician": "diagnostician"},
    )
    workflow.add_conditional_edges(
        "diagnostician",
        route_after_diagnostician,
        {"END": END, "reporter": "reporter"},
    )
    workflow.add_edge("reporter", "verify")
    workflow.add_edge("verify", END)

    return workflow.compile()

# ---------- Main: run ----------
if __name__ == "__main__":
    graph = build_graph()
    print("Graph structure:")
    print(graph.get_graph().draw_ascii())
