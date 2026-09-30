"""调用被测系统并缓存它的输出。

被测系统就是线上问答流水线：这里直接驱动 src/s5_app/api.py 的 _sse_events（FastAPI 接口
背后的事件生成器），不经 HTTP。评测拿到的检索结果、拒答、答案、token 与线上同源，
线上流程以后怎么改（比如换成工具调用），评测都跟着走，不需要另写一套。
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
]


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def system_info(corpus_fingerprint: str) -> dict:
    """被测系统的指纹：模型、提示词、检索与生成参数、语料、流水线源码。"""
    import graph
    from llm_config import LLM_CONFIG

    code = hashlib.sha1()
    for rel in _PIPELINE_FILES:
        code.update((ROOT / rel).read_bytes().replace(b"\r\n", b"\n"))
    info = {
        "model": LLM_CONFIG["model"],
        "temperature": LLM_CONFIG["temperature"],
        "prompt_version": graph.PROMPT_VERSION,
        "prompt_sha1": _sha1(graph.SYSTEM_PROMPT)[:12],
        "top_k": graph.TOP_K,
        "max_tokens": graph.MAX_TOKENS,
        "tau_distance": graph.TAU_DISTANCE,
        "corpus": corpus_fingerprint,
        "code_sha1": code.hexdigest()[:12],
    }
    info["system_id"] = _sha1(json.dumps(info, sort_keys=True, ensure_ascii=False))[:12]
    return info


def ask(question: str) -> dict:
    """问一次，收集事件流里的全部信息。耗时在调用方一侧计时。"""
    from api import _sse_events

    rec: dict = {
        "status": "answered", "answer": "", "retrieved": [], "top1_distance": None, "tau": None,
        "verification": None, "finish_reason": None,
        "input_tokens": None, "output_tokens": None, "total_tokens": None,
        "t_first_token_ms": None, "t_total_ms": None, "error": None,
    }
    parts: list[str] = []
    t0 = time.perf_counter()
    for ev in _sse_events(question):
        name, data = ev["event"], json.loads(ev["data"])
        if name == "retrieval":
            rec["retrieved"] = [{"chunk_id": c["chunk_id"], "distance": c["distance"]} for c in data["chunks"]]
        elif name == "rejected":
            rec["status"] = "rejected_gate"
            rec["top1_distance"], rec["tau"] = data["top1_distance"], data["tau"]
        elif name == "token":
            if rec["t_first_token_ms"] is None:
                rec["t_first_token_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            parts.append(data["text"])
        elif name == "verification":
            rec["verification"] = data
        elif name == "done":
            for k in ("finish_reason", "input_tokens", "output_tokens", "total_tokens"):
                rec[k] = data.get(k)
        elif name == "error":
            rec["status"], rec["error"] = "error", data.get("message")
    rec["t_total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    rec["answer"] = "".join(parts).strip()
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
    """逐题取系统输出。已缓存且系统指纹、问句都没变的直接复用；其余重新调用。

    多轮题（multi_turn）按线上现状只发最后一句；另外用 standalone 改写再问一次
    （variant=standalone），用来估计「如果系统会把追问改写成独立问题」能到什么水平。
    """
    answers = load_answers(answers_path)
    todo: list[tuple[dict, str, str]] = []
    for it in items:
        todo.append((it, "main", it["question"]))
        if it["type"] == "multi_turn" and it.get("standalone"):
            todo.append((it, "standalone", it["standalone"]))

    n_new = 0
    for i, (it, variant, question) in enumerate(todo, 1):
        key = _key(it["id"], variant)
        old = answers.get(key)
        fresh = (old is not None and not force and old.get("system_id") == system["system_id"]
                 and old.get("question") == question and old.get("status") != "error")
        if fresh:
            continue
        if reuse_only:
            raise RuntimeError(f"--reuse 要求全部答案已缓存，但 {key} 没有可用的缓存"
                               "（系统指纹或问句变了，或上次出错）。去掉 --reuse 重新生成。")
        rec = ask(question)
        rec.update({"id": it["id"], "variant": variant, "question": question,
                    "system_id": system["system_id"], "ran_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        answers[key] = rec
        n_new += 1
        brief = rec["status"] if rec["status"] != "answered" else f"{len(rec['answer'])} 字"
        print(f"[{i}/{len(todo)}] {key:<24} {rec['t_total_ms']:>7.0f} ms  {brief}", flush=True)
        if n_new % 10 == 0:
            save_answers(answers_path, answers)      # 中途落盘，失败重跑时不必从头来
    save_answers(answers_path, answers)
    print(f"系统输出：本次新调用 {n_new} 次，复用 {len(todo) - n_new} 次 → {answers_path}")
    return answers
