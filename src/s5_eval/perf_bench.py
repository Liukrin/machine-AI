"""S5 任务二：端到端性能与 Token 成本评测（不起 HTTP 服务，直连节点函数）。

复用生产路径同一套函数，逐段计时（毫秒）：
    t_retrieval  混合检索（graph.retrieve → hybrid_retrieve）
    t_distances  逐块距离计算（api._chunk_distances，前端展示用）
    t_gate       拒答闸（vector_top1_distance，单次 n_results=1）
    t_ttft       发起 LLM 流式请求 → 第一个非空 token
    t_generate   LLM 流式生成总耗时
    t_verify     引用校验（graph.verify_node）
    t_total      端到端（含组装开销）

另测：每题 input/output/total tokens；reject_questions 单独跑拒答路径并打印
实测 distance 与 τ 对比（未触发拒答的占位题单列表格、排除出拒答统计）；
单次 model.encode(单句) 耗时中位数（量化同题重复编码 3 次的冗余代价）。

路径与参数一律读 configs/config.yaml 的 s5_perf 段；任何一题失败记录异常并
继续，最后统计失败数，不静默吞掉。

用法：
    python src/s5_eval/perf_bench.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "s3_eval"))
sys.path.insert(0, str(ROOT / "src" / "s4_agent"))
sys.path.insert(0, str(ROOT / "src" / "s5_app"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from llm_config import LLM_CONFIG, get_llm  # noqa: E402
from graph import SYSTEM_PROMPT, TAU_DISTANCE, retrieve, verify_node  # noqa: E402
from hybrid_retrieval import _CTX, _load_ctx, build_search_text, vector_top1_distance  # noqa: E402
from api import _chunk_distances  # noqa: E402


def load_cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.open("r", encoding="utf-8"))


def pct(vals: list[float], p: float) -> float:
    return float(np.percentile(vals, p))


def stats(vals: list[float]) -> dict:
    """P50 / P90 / P95 / 均值 / max；空列表返回 None 值。"""
    if not vals:
        return {"p50": None, "p90": None, "p95": None, "mean": None, "max": None}
    return {
        "p50": pct(vals, 50), "p90": pct(vals, 90), "p95": pct(vals, 95),
        "mean": float(np.mean(vals)), "max": float(np.max(vals)),
    }


def ms(v) -> str:
    return "—" if v is None else f"{v:.1f}"


def run_once(question: str, max_tokens: int) -> dict:
    """按生产路径（api._sse_events）顺序走一遍，分段计时。拒答则不调 LLM。"""
    rec: dict = {"rejected": False, "top1_distance": None, "t_ttft": None}
    t_all0 = time.perf_counter()

    t0 = time.perf_counter()
    chunks = retrieve({"question": question})["retrieved"]
    rec["t_retrieval"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _chunk_distances(question, [c["chunk_id"] for c in chunks])
    rec["t_distances"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    top1 = vector_top1_distance(question)
    rec["t_gate"] = (time.perf_counter() - t0) * 1000
    rec["top1_distance"] = round(top1, 6)

    if top1 > TAU_DISTANCE:
        rec["rejected"] = True
        rec["t_ttft"] = None
        rec["t_generate"] = None
        rec["t_verify"] = None
        rec["input_tokens"] = None
        rec["output_tokens"] = None
        rec["total_tokens"] = None
        rec["t_total"] = (time.perf_counter() - t_all0) * 1000
        return rec

    context = "\n\n".join(f"[{c['chunk_id']}]\n{build_search_text(c)}" for c in chunks)
    user = f"问题：{question}\n\n检索到的内容：\n{context}"
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]
    llm = get_llm()

    answer_parts: list[str] = []
    usage: dict = {}
    t_first = None
    t_req = time.perf_counter()
    for chunk in llm.stream(messages, max_tokens=max_tokens, stream_usage=True):
        text = chunk.content or ""
        if text:
            if t_first is None:
                t_first = time.perf_counter()
            answer_parts.append(text)
        um = getattr(chunk, "usage_metadata", None)
        if um:
            usage = um
    t_end = time.perf_counter()

    rec["t_ttft"] = (t_first - t_req) * 1000 if t_first is not None else None
    rec["t_generate"] = (t_end - t_req) * 1000
    rec["input_tokens"] = usage.get("input_tokens")
    rec["output_tokens"] = usage.get("output_tokens")
    rec["total_tokens"] = usage.get("total_tokens")
    if rec["total_tokens"] is None and (rec["input_tokens"] or rec["output_tokens"]):
        rec["total_tokens"] = (rec["input_tokens"] or 0) + (rec["output_tokens"] or 0)

    answer = "".join(answer_parts).strip()
    t0 = time.perf_counter()
    verify_node({"answer": answer, "retrieved": chunks})
    rec["t_verify"] = (time.perf_counter() - t0) * 1000

    rec["t_total"] = (time.perf_counter() - t_all0) * 1000
    return rec


def warmup(question: str, runs: int, max_tokens: int) -> None:
    """完整正常路径预热，结果丢弃；预热后校验冷启动已全部排除。"""
    for i in range(runs):
        t0 = time.perf_counter()
        rec = run_once(question, max_tokens)
        print(f"预热 {i + 1}/{runs}: total={(time.perf_counter() - t0) * 1000:.0f}ms"
              f" rejected={rec['rejected']}（结果丢弃）")
    assert "model" in _CTX, "_load_ctx 冷启动未完成，统计将混入模型加载耗时"


def measure_encode(question: str, runs: int) -> tuple[float, list[float]]:
    """单次 model.encode(单句) 耗时，取中位数（量化重复编码冗余）。"""
    ctx = _load_ctx()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        ctx["model"].encode([question], normalize_embeddings=ctx["normalize"],
                            show_progress_bar=False)
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts)), ts


def stats_row(name: str, s: dict) -> str:
    return f"| {name} | {ms(s['p50'])} | {ms(s['p90'])} | {ms(s['p95'])} | {ms(s['mean'])} | {ms(s['max'])} |"


def main() -> None:
    cfg = load_cfg()
    sp = cfg["s5_perf"]
    max_tokens = int(sp["max_tokens"])
    warmup_runs = int(sp["warmup_runs"])
    repeat = int(sp["repeat"])
    encode_runs = int(sp["encode_measure_runs"])

    qa_set = [json.loads(line) for line in
              (ROOT / sp["qa_set"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"评测题数 {len(qa_set)}  warmup={warmup_runs}  repeat={repeat}  max_tokens={max_tokens}")

    # ---- 预热（排除 _load_ctx / jieba / 连接冷启动）----
    warmup(qa_set[0]["question"], warmup_runs, max_tokens)

    # ---- 单次编码耗时（重复编码冗余量化）----
    encode_median, encode_ts = measure_encode(qa_set[0]["question"], encode_runs)
    print(f"单次 encode 中位数 {encode_median:.1f}ms（{encode_runs} 次，"
          f"min={min(encode_ts):.1f} max={max(encode_ts):.1f}）")

    # ---- qa_set 正常路径计时 ----
    records: list[dict] = []
    failures: list[dict] = []
    n_runs = len(qa_set) * repeat
    done = 0
    for r in qa_set:
        for _ in range(repeat):
            done += 1
            try:
                rec = run_once(r["question"], max_tokens)
                rec.update({"qa_id": r["qa_id"], "chunk_type": r["chunk_type"],
                            "difficulty": r["difficulty"], "question": r["question"]})
                records.append(rec)
                if rec["rejected"]:
                    print(f"[{done}/{n_runs}] {r['qa_id']} 拒答 distance={rec['top1_distance']} "
                          f"total={rec['t_total']:.0f}ms")
                else:
                    print(f"[{done}/{n_runs}] {r['qa_id']} total={rec['t_total']:.0f}ms "
                          f"ttft={rec['t_ttft']:.0f}ms tokens={rec['total_tokens']}")
            except Exception as exc:
                failures.append({"qa_id": r["qa_id"],
                                 "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{done}/{n_runs}] {r['qa_id']} 失败：{type(exc).__name__}: {exc}")

    normal = [r for r in records if not r["rejected"]]
    rejected_in_qa = [r for r in records if r["rejected"]]

    # ---- reject_questions 拒答路径 ----
    reject_records: list[dict] = []
    not_rejected: list[dict] = []
    reject_failures: list[dict] = []
    for q in sp["reject_questions"]:
        try:
            rec = run_once(q, max_tokens)
            rec["question"] = q
            verdict = "触发拒答" if rec["rejected"] else "未触发（进入生成路径）"
            print(f"拒答题「{q}」 distance={rec['top1_distance']} τ={TAU_DISTANCE} → {verdict} "
                  f"total={rec['t_total']:.0f}ms")
            (reject_records if rec["rejected"] else not_rejected).append(rec)
        except Exception as exc:
            reject_failures.append({"question": q, "error": f"{type(exc).__name__}: {exc}"})
            print(f"拒答题「{q}」 失败：{type(exc).__name__}: {exc}")

    # ---- 汇总 ----
    stage_keys = [("t_retrieval", "混合检索"), ("t_distances", "逐块距离（前端展示）"),
                  ("t_gate", "拒答闸"), ("t_ttft", "首 token 延迟"),
                  ("t_generate", "LLM 流式生成"), ("t_verify", "引用校验"),
                  ("t_total", "端到端")]
    stage_stats = {k: stats([r[k] for r in normal if r.get(k) is not None]) for k, _ in stage_keys}

    tot_mean = stage_stats["t_total"]["mean"]
    gen_mean = stage_stats["t_generate"]["mean"]
    ttft_p50 = stage_stats["t_ttft"]["p50"]
    local_mean = sum(stage_stats[k]["mean"] for k in ("t_retrieval", "t_distances", "t_gate"))

    in_tokens = [r["input_tokens"] for r in normal if r.get("input_tokens") is not None]
    out_tokens = [r["output_tokens"] for r in normal if r.get("output_tokens") is not None]
    all_tokens = [r["total_tokens"] for r in normal if r.get("total_tokens") is not None]

    def token_stats(sub: list[dict]) -> dict:
        return {
            "n": len(sub),
            "input": stats([r["input_tokens"] for r in sub if r.get("input_tokens") is not None]),
            "output": stats([r["output_tokens"] for r in sub if r.get("output_tokens") is not None]),
            "total": stats([r["total_tokens"] for r in sub if r.get("total_tokens") is not None]),
        }

    tk_all = token_stats(normal)
    tk_text = token_stats([r for r in normal if r["chunk_type"] == "text"])
    tk_table = token_stats([r for r in normal if r["chunk_type"] == "table"])

    rej_totals = stats([r["t_total"] for r in reject_records])
    saving_mean = (tot_mean - rej_totals["mean"]) if (tot_mean is not None and rej_totals["mean"] is not None) else None

    # ---- 报告 ----
    L = ["# S5 端到端性能与 Token 成本评测", ""]
    L.append(f"> 生成脚本：src/s5_eval/perf_bench.py（直连节点函数，不起 HTTP 服务）；"
             f"LLM：{LLM_CONFIG['model']}（{LLM_CONFIG['base_url']}）")
    L.append(f"> 配置：warmup_runs={warmup_runs}  repeat={repeat}  max_tokens={max_tokens}  "
             f"τ={TAU_DISTANCE}；题数 {len(qa_set)} × repeat = {n_runs} 次，"
             f"成功 {len(records)}，失败 {len(failures)}")
    L.append("")

    L.append("## 1. 分段耗时（正常路径，单位 ms）\n")
    L.append("| 阶段 | P50 | P90 | P95 | 均值 | max |")
    L.append("|---|---|---|---|---|---|")
    for k, name in stage_keys:
        L.append(stats_row(name, stage_stats[k]))
    L.append("")

    L.append("## 2. 端到端：正常路径 vs 拒答路径\n")
    L.append("拒答路径包含三段：混合检索 + 逐块距离 + 拒答闸（无 LLM 生成、无引用校验）。\n")
    L.append("| 路径 | 样本数 | P50 | 均值 | max |")
    L.append("|---|---|---|---|---|")
    L.append(f"| 正常路径 | {len(normal)} | {ms(stage_stats['t_total']['p50'])} | "
             f"{ms(tot_mean)} | {ms(stage_stats['t_total']['max'])} |")
    L.append(f"| 拒答路径 | {len(reject_records)} | {ms(rej_totals['p50'])} | "
             f"{ms(rej_totals['mean'])} | {ms(rej_totals['max'])} |")
    if saving_mean is not None:
        L.append(f"\n拒答每题省下约 **{saving_mean:.0f} ms**"
                 f"（正常均值 {tot_mean:.0f} − 拒答均值 {rej_totals['mean']:.0f}），"
                 f"即省去全部 LLM 生成与校验耗时。")
    if rejected_in_qa:
        L.append(f"\n> 注意：{len(rejected_in_qa)} 道 qa_set 题触发了拒答闸，"
                 "已列入下方失败/异常清单之外的观察项："
                 + "、".join(r["qa_id"] for r in rejected_in_qa))
    if not_rejected:
        L.append("\n### 2.1 未触发拒答的占位题（已排除出拒答路径统计）\n")
        L.append("| 占位题 | 实测 distance | τ | 说明 |")
        L.append("|---|---|---|---|")
        for r in not_rejected:
            L.append(f"| {r['question']} | {r['top1_distance']} | {TAU_DISTANCE} | "
                     f"未触发拒答（走了生成路径，t_total={r['t_total']:.0f}ms），"
                     "复现 badcase 第 2 条「正负样本重叠」 |")
    if reject_failures:
        L.append("\n拒答路径失败：" + "；".join(f"「{f['question']}」{f['error']}" for f in reject_failures))
    L.append("")

    L.append("## 3. Token 消耗（正常路径）\n")
    L.append("| 分组 | 样本数 | 指标 | P50 | 均值 | max |")
    L.append("|---|---|---|---|---|---|")
    for gname, tk in (("总体", tk_all), ("text 题", tk_text), ("table 题", tk_table)):
        for mname in ("input", "output", "total"):
            s = tk[mname]
            p50 = "—" if s["p50"] is None else f"{s['p50']:.0f}"
            mean = "—" if s["mean"] is None else f"{s['mean']:.0f}"
            mx = "—" if s["max"] is None else f"{s['max']:.0f}"
            L.append(f"| {gname} | {tk['n']} | {mname} | {p50} | {mean} | {mx} |")
    L.append(f"\n本次运行总消耗：total_tokens {sum(all_tokens)}"
             f"（input {sum(in_tokens)} + output {sum(out_tokens)}）。")
    L.append("\n> 单价随时变动，本报告只记 token 数量，不折算费用。\n")

    L.append("## 4. 耗时构成占比（按各阶段均值 / t_total 均值）\n")
    L.append("| 阶段 | 均值 (ms) | 占 t_total |")
    L.append("|---|---|---|")
    accounted = 0.0
    for k, name in stage_keys[:-1]:
        if k == "t_ttft":
            continue  # ttft 是 t_generate 的子集，不参与占比加总
        m = stage_stats[k]["mean"]
        if m is None or tot_mean is None:
            continue
        accounted += m
        L.append(f"| {name} | {m:.1f} | {m / tot_mean * 100:.1f}% |")
    other = tot_mean - accounted if tot_mean is not None else None
    if other is not None:
        L.append(f"| 其他（消息组装等） | {other:.1f} | {other / tot_mean * 100:.1f}% |")
    L.append("")
    L.append("### 4.1 重复编码冗余（已知问题量化，不改代码）\n")
    L.append(f"- 单次 model.encode(单句) 中位数：**{encode_median:.1f} ms**（{encode_runs} 次实测）")
    L.append("- 生产路径每题 encode 3 次：retrieve 内 1 次、vector_top1_distance 1 次、_chunk_distances 1 次")
    save2 = encode_median * 2
    pct_save = save2 / tot_mean * 100 if tot_mean else None
    L.append(f"- 理论可节省（缓存复用后少 2 次）：**{save2:.1f} ms/题**，"
             f"占 t_total 均值 **{pct_save:.1f}%**" if pct_save is not None else
             f"- 理论可节省（缓存复用后少 2 次）：**{save2:.1f} ms/题**")
    L.append("")

    L.append("## 5. 结论（数字均来自本次运行）\n")
    if tot_mean and gen_mean is not None:
        L.append(f"1. **LLM 生成是端到端主要耗时**：t_generate 均值 {gen_mean:.0f} ms，"
                 f"占 t_total 均值（{tot_mean:.0f} ms）的 {gen_mean / tot_mean * 100:.1f}%；"
                 f"首 token 延迟 P50 {ttft_p50:.0f} ms。")
        L.append(f"2. **本地计算开销很小**：检索 + 逐块距离 + 拒答闸合计均值 {local_mean:.0f} ms，"
                 f"仅占 {local_mean / tot_mean * 100:.1f}%；其中重复编码 3 次理论可省 "
                 f"{save2:.1f} ms（{pct_save:.1f}%）。" if pct_save is not None else
                 f"2. **本地计算开销很小**：检索 + 逐块距离 + 拒答闸合计均值 {local_mean:.0f} ms，"
                 f"仅占 {local_mean / tot_mean * 100:.1f}%。")
    if tk_text["input"]["mean"] is not None and tk_table["input"]["mean"] is not None:
        L.append(f"3. **Token 以 input 为主**：总体 input 均值 {tk_all['input']['mean']:.0f} vs "
                 f"output 均值 {tk_all['output']['mean']:.0f}；table 题 input 均值 "
                 f"{tk_table['input']['mean']:.0f} vs text 题 {tk_text['input']['mean']:.0f}，"
                 f"{'表格题上下文更长' if tk_table['input']['mean'] > tk_text['input']['mean'] else '文本题上下文反而更长'}。")
    L.append("")

    L.append("## 6. 失败清单\n")
    if failures or reject_failures:
        for f in failures:
            L.append(f"- {f['qa_id']}：{f['error']}")
        for f in reject_failures:
            L.append(f"- 拒答题「{f['question']}」：{f['error']}")
    else:
        L.append("（无）")
    L.append("")

    report = ROOT / sp["report"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(L), encoding="utf-8")

    # ---- 控制台摘要（与报告逐条对账用）----
    print("=" * 70)
    print(f"成功 {len(normal)} / 失败 {len(failures)}（qa_set）；"
          f"拒答触发 {len(reject_records)} / 未触发 {len(not_rejected)} / 失败 {len(reject_failures)}（占位题）")
    for k, name in stage_keys:
        s = stage_stats[k]
        print(f"{name:<12} P50={ms(s['p50'])} P90={ms(s['p90'])} P95={ms(s['p95'])} "
              f"mean={ms(s['mean'])} max={ms(s['max'])}")
    if tot_mean:
        print(f"拒答路径均值 {ms(rej_totals['mean'])} ms，每题省下 {ms(saving_mean)} ms")
    print(f"tokens：input 合计 {sum(in_tokens)}，output 合计 {sum(out_tokens)}，"
          f"total 合计 {sum(all_tokens)}")
    print(f"报告已写入 {report}")


if __name__ == "__main__":
    main()
