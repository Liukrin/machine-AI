"""调用被测系统并缓存它的输出。

被测系统就是线上问答流水线：这里直接驱动 src/s5_app/api.py 的 _sse_events（FastAPI 接口
背后的事件生成器），不经 HTTP。评测拿到的检索结果、工具调用、拒答、答案、token 与线上同源。

两种模式（system.json 的 mode）：
  rag    阶段 1 的固定流水线（检索 → 拒答闸 → 生成），不看对话历史；
  agent  工具调用 Agent（src/s4_agent/agent.py），多轮题把前文历史一起发过去，每一步工具调用都记下来。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in ("src", "src/s3_eval", "src/s4_agent", "src/s5_app"):
    sys.path.insert(0, str(ROOT / _p))

# 决定系统行为的源码：任何一个改了，缓存的答案都不能再用
_PIPELINE_FILES = [
    "src/s5_app/api.py", "src/s4_agent/graph.py", "src/s4_agent/verify.py",
    "src/s3_eval/hybrid_retrieval.py", "src/llm_config.py",
    "src/s4_agent/agent.py", "src/s4_agent/tools.py", "src/s4_agent/units.py", "src/s4_agent/tables.py",
]
MODES = ("agent", "rag")


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def system_info(corpus_fingerprint: str, mode: str) -> dict:
    """被测系统的指纹：模式、模型、提示词、检索与生成参数、Agent 的约束参数、语料、流水线源码。"""
    from llm_config import LLM_CONFIG

    code = hashlib.sha1()
    for rel in _PIPELINE_FILES:
        code.update((ROOT / rel).read_bytes().replace(b"\r\n", b"\n"))
    base = {"mode": mode, "model": LLM_CONFIG["model"], "temperature": LLM_CONFIG["temperature"]}
    if mode == "rag":
        import graph
        info = {**base, "prompt_version": graph.PROMPT_VERSION, "prompt_sha1": _sha1(graph.SYSTEM_PROMPT)[:12],
                "top_k": graph.TOP_K, "max_tokens": graph.MAX_TOKENS, "tau_distance": graph.TAU_DISTANCE}
    elif mode == "agent":
        import agent
        c = agent.CFG
        info = {**base, "prompt_version": agent.PROMPT_VERSION, "prompt_sha1": _sha1(agent.SYSTEM_PROMPT)[:12],
                "top_k": int(c["search_top_k"]), "max_tokens": int(c["max_tokens"]), "tau_distance": agent.TAU_DISTANCE,
                "max_tool_rounds": int(c["max_tool_rounds"]), "max_calls_per_round": int(c["max_calls_per_round"]),
                "tool_timeout_s": float(c["tool_timeout_s"]), "lookup_max_rows": int(c["lookup_max_rows"]),
                "history_turns": int(c["history_turns"]), "history_answer_chars": int(c["history_answer_chars"])}
    else:
        raise ValueError(f"未知模式：{mode}（可选 {MODES}）")
    info.update({"corpus": corpus_fingerprint, "code_sha1": code.hexdigest()[:12]})
    info["system_id"] = _sha1(json.dumps(info, sort_keys=True, ensure_ascii=False))[:12]
    return info


def ask(question: str, history: list[dict] | None = None, mode: str = "agent") -> dict:
    """问一次，收集事件流里的全部信息。耗时在调用方一侧计时。"""
    from api import _sse_events

    rec: dict = {
        "status": "answered", "answer": "", "retrieved": [], "top1_distance": None, "tau": None,
        "verification": None, "finish_reason": None,
        "input_tokens": None, "output_tokens": None, "total_tokens": None,
        "t_first_token_ms": None, "t_total_ms": None, "error": None,
        "mode": mode, "steps": [], "llm_calls": None, "tool_calls": None, "cache_read_tokens": None,
        "forced_final": None, "model_name": None, "cost_yuan": None,
    }
    parts: dict[int, list[str]] = {}
    first_token: dict[int, float] = {}
    thought_rounds: set[int] = set()
    final_answer = None
    t0 = time.perf_counter()
    for ev in _sse_events(question, history or [], mode):
        name, data = ev["event"], json.loads(ev["data"])
        if name == "retrieval":       # agent 模式下是累计的来源列表，取最后一次
            rec["retrieved"] = [{"chunk_id": c["chunk_id"], "distance": c["distance"]} for c in data["chunks"]]
        elif name == "rejected":
            rec["status"] = "rejected_gate"
            rec["top1_distance"], rec["tau"] = data["top1_distance"], data["tau"]
        elif name == "token":
            rnd = data.get("round", 0)
            first_token.setdefault(rnd, round((time.perf_counter() - t0) * 1000, 1))
            parts.setdefault(rnd, []).append(data["text"])
        elif name == "step":
            if data["type"] == "tool_end":
                rec["steps"].append({k: data.get(k) for k in
                                     ("round", "name", "args", "ok", "summary", "elapsed_ms", "auto", "data", "output")})
            elif data["type"] == "thought":
                thought_rounds.add(data["round"])
        elif name == "verification":
            rec["verification"] = data
        elif name == "done":
            for k in ("finish_reason", "input_tokens", "output_tokens", "total_tokens", "llm_calls", "tool_calls",
                      "cache_read_tokens", "forced_final", "model_name", "cost_yuan"):
                rec[k] = data.get(k)
            final_answer = data.get("answer")
        elif name == "error":
            rec["status"], rec["error"] = "error", data.get("message")
    rec["t_total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    answer_rounds = [r for r in parts if r not in thought_rounds]
    last = max(answer_rounds) if answer_rounds else None
    # 首 token：最终回答那一轮的第一个 token（中间轮次决定调工具时流出的文字不算回答）
    rec["t_first_token_ms"] = first_token.get(last) if last is not None else None
    rec["answer"] = (final_answer if final_answer is not None
                     else "".join(parts.get(last, [])) if last is not None else "").strip()
    # 拒答闸用的向量 top-1 距离：作答的题事件流里不带，这里补算一次（不计入耗时），用于分析闸的区分度
    from hybrid_retrieval import vector_top1_distance
    rec["gate_distance"] = round(vector_top1_distance(question), 6)
    return rec


def _key(item_id: str, variant: str) -> str:
    return item_id if variant == "main" else f"{item_id}@{variant}"


def load_answers(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {_key(r["id"], r.get("variant", "main")): r for r in rows}


def save_answers(path: Path, answers: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [answers[k] for k in sorted(answers)]
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def run_system(items: list[dict], answers_path: Path, system: dict, *,
               reuse_only: bool = False, force: bool = False) -> dict[str, dict]:
    """逐题取系统输出。已缓存且系统指纹、问句、对话历史都没变的直接复用；其余重新调用。

    多轮题（multi_turn）：agent 模式把前文历史一起发（main）；rag 模式不支持历史，只发最后一句。
    另外两种模式都用 standalone 改写再问一次（variant=standalone），作为「追问被完美改写」时的参考上限。
    """
    mode = system.get("mode", "rag")
    answers = load_answers(answers_path)
    todo: list[tuple[dict, str, str, list[dict]]] = []
    for it in items:
        history = it.get("history") or [] if mode == "agent" else []
        todo.append((it, "main", it["question"], history))
        if it["type"] == "multi_turn" and it.get("standalone"):
            todo.append((it, "standalone", it["standalone"], []))

    n_new = 0
    for i, (it, variant, question, history) in enumerate(todo, 1):
        key = _key(it["id"], variant)
        hist_sha = _sha1(json.dumps(history, ensure_ascii=False, sort_keys=True))[:12] if history else None
        old = answers.get(key)
        fresh = (old is not None and not force and old.get("system_id") == system["system_id"]
                 and old.get("question") == question and old.get("history_sha1") == hist_sha
                 and old.get("status") != "error")
        if fresh:
            continue
        if reuse_only:
            raise RuntimeError(f"--reuse 要求全部答案已缓存，但 {key} 没有可用的缓存"
                               "（系统指纹、问句或对话历史变了，或上次出错）。去掉 --reuse 重新生成。")
        rec = ask(question, history, mode)
        rec.update({"id": it["id"], "variant": variant, "question": question, "history_sha1": hist_sha,
                    "system_id": system["system_id"], "ran_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        answers[key] = rec
        n_new += 1
        brief = rec["status"] if rec["status"] != "answered" else f"{len(rec['answer'])} 字"
        if mode == "agent" and rec["status"] == "answered":
            tools = [s["name"] for s in rec["steps"] if not s.get("auto")]
            brief += f"，模型调用 {rec['llm_calls']} 次" + (f"，工具 {'、'.join(tools)}" if tools else "")
        print(f"[{i}/{len(todo)}] {key:<24} {rec['t_total_ms']:>7.0f} ms  {brief}", flush=True)
        if n_new % 10 == 0:
            save_answers(answers_path, answers)      # 中途落盘，失败重跑时不必从头来
    save_answers(answers_path, answers)
    print(f"系统输出：本次新调用 {n_new} 次，复用 {len(todo) - n_new} 次 → {answers_path}")
    return answers
