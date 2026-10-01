"""把逐题评分汇总成指标与 Markdown 报告。不调用 LLM，可反复重算。"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

from dataset import TYPE_CN, TYPES_ANSWER, TYPES_REFUSE

SPLIT_CN = {"dev": "dev（开发集）", "test": "test（测试集）"}


# --------------------------------------------------------------------------- 小工具
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """比例的 Wilson 95% 置信区间。"""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def frac(k: int, n: int, ci: bool = False) -> str:
    if n == 0:
        return "—"
    s = f"{k / n:.3f}（{k}/{n}）"
    if ci:
        lo, hi = wilson(k, n)
        s += f" [{lo:.2f}, {hi:.2f}]"
    return s


def pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    v = sorted(vals)
    idx = p / 100 * (len(v) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    return v[lo] if lo == hi else v[lo] + (v[hi] - v[lo]) * (idx - lo)


def ms(v: float | None) -> str:
    return "—" if v is None else f"{v:.0f}"


# --------------------------------------------------------------------------- 汇总
def aggregate(rows: list[dict]) -> dict:
    """rows：同一划分下 main 变体的逐题评分（已并入评审结果）。"""
    ok = [r for r in rows if r["behavior"] != "error"]
    ans_items = [r for r in ok if r["expect"] == "answer"]
    ref_items = [r for r in ok if r["expect"] == "refuse"]
    answered = [r for r in ok if r["behavior"] == "answered"]
    answered_in = [r for r in ans_items if r["behavior"] == "answered"]

    m: dict = {"n": len(rows), "n_error": len(rows) - len(ok), "n_answered": len(answered),
               "n_answer_items": len(ans_items), "n_refuse_items": len(ref_items)}

    # ---- 应作答的题
    m["evidence_all"] = (sum(r["evidence_all"] for r in ans_items), len(ans_items))
    m["false_refusal"] = (sum(r["refused"] for r in ans_items), len(ans_items))
    m["false_refusal_gate"] = sum(r["behavior"] == "rejected_gate" for r in ans_items)
    m["false_refusal_model"] = sum(r["behavior"] == "refused_model" for r in ans_items)
    m["fact_recall"] = (sum(r["n_facts_matched"] for r in ans_items), sum(r["n_facts"] for r in ans_items))
    m["fact_recall_answered"] = (sum(r["n_facts_matched"] for r in answered_in),
                                 sum(r["n_facts"] for r in answered_in))
    m["strict_correct"] = (sum(r["all_facts"] for r in ans_items), len(ans_items))
    m["numeric"] = (sum(r["n_numeric_matched"] for r in ans_items), sum(r["n_numeric"] for r in ans_items))
    num_items = [r for r in ans_items if r["n_numeric"]]
    m["numeric_items"] = (sum(r["n_numeric_matched"] == r["n_numeric"] for r in num_items), len(num_items))
    # 带着引用说「资料不足」的回答按规则算作答（没答出事实），这里单列评审的看法，不改变上面的判定
    m["answered_but_declined"] = sum(1 for r in answered_in if (r.get("judge") or {}).get("declined"))
    m["cited_gold"] = (sum(bool(r["cited_gold"]) for r in answered_in), len(answered_in))
    m["wrong_manual"] = (sum(r["wrong_manual"] for r in answered_in), len(answered_in))
    m["strict_grounded"] = (sum(r["all_facts"] and bool(r["cited_gold"]) for r in ans_items), len(ans_items))

    # ---- 应拒答的题
    m["refusal_recall"] = (sum(r["refused"] for r in ref_items), len(ref_items))
    m["refusal_gate"] = sum(r["behavior"] == "rejected_gate" for r in ref_items)
    m["refusal_model"] = sum(r["behavior"] == "refused_model" for r in ref_items)
    # 评审认为「实质上是在说明资料不足」的作答，不算进确定性指标，只单列
    m["refusal_declined_by_judge"] = sum(
        1 for r in ref_items if r["behavior"] == "answered" and (r.get("judge") or {}).get("declined"))
    refused_all = [r for r in ok if r["refused"]]
    m["refusal_precision"] = (sum(r["expect"] == "refuse" for r in refused_all), len(refused_all))

    # ---- 答案质量（所有实际作答的题）
    m["uncited"] = (sum(r["n_uncited"] for r in answered), sum(r["n_sentences"] for r in answered))
    m["numbers_traced"] = (sum(r["n_numbers"] - r["n_untraced"] for r in answered),
                           sum(r["n_numbers"] for r in answered))
    ov = [x for r in answered for x in r["sentence_overlap"] if x is not None]
    m["overlap"] = {"median": pct(ov, 50), "p10": pct(ov, 10), "low": sum(x < 0.5 for x in ov), "n": len(ov)}
    m["fabricated_citations"] = sum(len(r["cited_fabricated"]) for r in answered)
    m["answers_citing_sample"] = sum(bool(r["cited_sample"]) for r in answered)
    m["truncated"] = sum(r.get("finish_reason") == "length" for r in answered)

    sup = Counter()
    strict = cited_ok = cited_n = 0
    judge_failed = judge_n = 0
    for r in answered:
        j = r.get("judge")
        if j is None:
            continue
        judge_n += 1
        if j.get("failed"):
            judge_failed += 1
            continue
        for s in j["sentences"]:
            sup[s["support"]] += 1
            strict += s["support"] == "full" and s["quote_ok"]
            if s["cited_ok"] is not None:
                cited_n += 1
                cited_ok += bool(s["cited_ok"])
    checkable = sup["full"] + sup["partial"] + sup["none"]
    m["judge_answers"] = judge_n
    m["judge_failed"] = judge_failed
    m["faithful_strict"] = (strict, checkable)
    m["faithful_full"] = (sup["full"], checkable)
    m["unsupported"] = (sup["none"], checkable)
    m["partial"] = (sup["partial"], checkable)
    m["citation_ok"] = (cited_ok, cited_n)

    # ---- 关键事实：正则 vs 评审
    agree = both = only_regex = only_judge = contradicted = judged_present = fact_judge_failed = 0
    for r in answered_in:
        fj = r.get("fact_judge")
        if fj is None:
            continue
        if fj.get("failed"):
            fact_judge_failed += 1
            continue
        for f, v in zip(r["facts"], fj["verdicts"]):
            present = v == "present"
            judged_present += present
            contradicted += v == "contradicted"
            agree += f["matched"] == present
            both += 1
            only_regex += f["matched"] and not present
            only_judge += present and not f["matched"]
    m["fact_agree"] = (agree, both)
    m["fact_only_regex"] = only_regex
    m["fact_only_judge"] = only_judge
    m["fact_contradicted"] = contradicted
    m["fact_judge_failed"] = fact_judge_failed
    m["fact_recall_judge"] = (judged_present, both)     # 评审口径的关键事实召回（只含实际作答且评审成功的题）

    # ---- 耗时与 token
    t_ans = [r["t_total_ms"] for r in answered if r.get("t_total_ms") is not None]
    t_first = [r["t_first_token_ms"] for r in answered if r.get("t_first_token_ms") is not None]
    t_rej = [r["t_total_ms"] for r in ok if r["behavior"] == "rejected_gate" and r.get("t_total_ms") is not None]
    m["latency"] = {"p50": pct(t_ans, 50), "p95": pct(t_ans, 95), "ttft_p50": pct(t_first, 50),
                    "reject_p50": pct(t_rej, 50)}
    tin = [r["input_tokens"] for r in answered if r.get("input_tokens") is not None]
    tout = [r["output_tokens"] for r in answered if r.get("output_tokens") is not None]
    m["tokens"] = {"input_mean": sum(tin) / len(tin) if tin else None,
                   "output_mean": sum(tout) / len(tout) if tout else None,
                   "input_sum": sum(tin), "output_sum": sum(tout)}
    # 成本：所有调用了模型的题（作答 + 模型自述拒答）；被拒答闸拦下的题没有调用模型，记 0 元
    upper = [r["cost_upper"] for r in ok if r.get("cost_upper") is not None]
    actual = [r["cost_yuan"] for r in ok if r.get("cost_yuan") is not None]
    m["cost"] = {"upper_mean": sum(upper) / len(upper) if upper else None, "n_upper": len(upper),
                 "actual_mean": sum(actual) / len(actual) if actual else None, "n_actual": len(actual)}

    # ---- 按题型（应作答的题），供 compare.py 并排对比
    m["by_type"] = {}
    for typ in TYPES_ANSWER:
        sub = [r for r in ans_items if r["type"] == typ]
        if sub:
            m["by_type"][typ] = {"n": len(sub), "strict_correct": sum(r["all_facts"] for r in sub),
                                 "false_refusal": sum(r["refused"] for r in sub),
                                 "evidence_all": sum(r["evidence_all"] for r in sub)}

    # ---- Agent 行为（rag 模式没有这些字段）
    agent_rows = [r for r in ok if r.get("mode") == "agent" and r.get("llm_calls") is not None]
    if agent_rows:
        calls = Counter(r["llm_calls"] for r in agent_rows)
        expected = [r for r in agent_rows if r["expect"] == "answer" and r.get("tools_expected")]
        m["agent"] = {
            "n": len(agent_rows),
            "llm_calls_mean": sum(r["llm_calls"] for r in agent_rows) / len(agent_rows),
            "llm_calls_dist": {k: calls[k] for k in sorted(calls)},
            "presearch_only": sum(r["llm_calls"] == 1 for r in agent_rows),
            "tool_calls_mean": sum(r["tool_calls"] for r in agent_rows) / len(agent_rows),
            "tool_usage": dict(Counter(t for r in agent_rows for t in r["tools_used"])),
            "tool_errors": sum(r["tool_errors"] for r in agent_rows),
            "tool_errors_items": sum(r["tool_errors"] > 0 for r in agent_rows),
            "forced_final": sum(r["forced_final"] for r in agent_rows),
            "tools_expected_ok": (sum(r["tools_expected_ok"] for r in expected), len(expected)),
            "cache_read_share": (sum(r.get("cache_read_tokens") or 0 for r in agent_rows),
                                 sum(r.get("input_tokens") or 0 for r in agent_rows)),
        }
        checked = [r for r in agent_rows if r.get("sys_numbers_checked") is not None]
        if checked:                       # 阶段 3 的数值核对与改写（之前的评测没有）
            m["agent"]["numcheck"] = {
                "n": len(checked),
                "numbers": sum(r["sys_numbers_checked"] for r in checked),
                "repairs": sum(1 for r in checked if r.get("repair")),
                "repairs_fixed": sum(1 for r in checked if r.get("repair_fixed")),
                "repairs_kept_draft": sum(1 for r in checked if r.get("repair") == "draft"),
                "issue_answers": sum(1 for r in checked if r.get("sys_number_issues")),
            }
    return m


def group_table(rows: list[dict], key_fn, order: list[str], label_fn) -> list[str]:
    """应作答的题按某个维度分组的指标表。"""
    L = ["| 分组 | 题数 | 证据全部召回 | 误拒 | 关键事实召回 | 完全答对 | 数值事实 |",
         "|---|---|---|---|---|---|---|"]
    for key in order:
        sub = [r for r in rows if r["expect"] == "answer" and r["behavior"] != "error" and key_fn(r) == key]
        if not sub:
            continue
        L.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            label_fn(key), len(sub),
            frac(sum(r["evidence_all"] for r in sub), len(sub)),
            frac(sum(r["refused"] for r in sub), len(sub)),
            frac(sum(r["n_facts_matched"] for r in sub), sum(r["n_facts"] for r in sub)),
            frac(sum(r["all_facts"] for r in sub), len(sub)),
            frac(sum(r["n_numeric_matched"] for r in sub), sum(r["n_numeric"] for r in sub))))
    return L


# --------------------------------------------------------------------------- 报告
def build_report(scores: list[dict], items: dict[str, dict], split: dict[str, str], splits: list[str],
                 meta: dict) -> tuple[list[str], dict]:
    main = [s for s in scores if s["variant"] == "main"]
    by_split = {sp: [s for s in main if split[s["id"]] == sp] for sp in splits}
    aggs = {sp: aggregate(by_split[sp]) for sp in splits}
    sysinfo = meta["system"]

    mode = sysinfo.get("mode", "rag")
    title = meta.get("set_title") or "答案级评测集"
    L = [f"# {title}评测报告（{meta['run']}，{mode} 模式）", ""]
    L.append(f"> 生成脚本：src/answer_eval/run.py；生成时间 {meta['generated_at']}")
    if mode == "agent":
        L.append(f"> 被测系统：工具调用 Agent（src/s4_agent/agent.py），{sysinfo['model']}，提示词 {sysinfo['prompt_version']}，"
                 f"每次检索 {sysinfo['top_k']} 个片段，工具轮数上限 {sysinfo['max_tool_rounds']}（每题最多 "
                 f"{sysinfo['max_tool_rounds'] + 1} 次模型调用），单次生成上限 {sysinfo['max_tokens']}，"
                 f"相关度提示阈值 τ={sysinfo['tau_distance']}（不拦截），temperature={sysinfo['temperature']}；"
                 f"系统指纹 {sysinfo['system_id']}（语料 {sysinfo['corpus']}，流水线源码 {sysinfo['code_sha1']}）")
    else:
        L.append(f"> 被测系统：固定流水线（rag 模式），{sysinfo['model']}，提示词 {sysinfo['prompt_version']} 组，"
                 f"top_k={sysinfo['top_k']}，max_tokens={sysinfo['max_tokens']}，τ={sysinfo['tau_distance']}，"
                 f"temperature={sysinfo['temperature']}；"
                 f"系统指纹 {sysinfo['system_id']}（语料 {sysinfo['corpus']}，流水线源码 {sysinfo['code_sha1']}）")
    L.append(f"> 评测集：{meta['items_path']}，共 {meta['n_items']} 题；本报告覆盖 "
             + "、".join(f"{sp} {len(by_split[sp])} 题" for sp in splits)
             + "。dev 用于调试评测脚本和后续迭代，test 只用来报数。")
    L.append(f"> 评审模型：{meta['judge_model']}（与生成模型相同，存在自评偏差；提示词指纹 {meta['judge_prompt']}）。"
             "关键事实、数值、拒答、证据命中均由确定性代码判定，评审模型只负责逐句忠实度与引用是否成立。")
    L.append("> 比例后的方括号是 Wilson 95% 置信区间。题量有限，区间普遍较宽，差几个百分点不足以说明优劣。")
    L.append("")
    for sp in splits:
        if aggs[sp]["n_error"]:
            L.append(f"**{sp} 有 {aggs[sp]['n_error']} 题系统调用出错，已排除出全部分母，本次结果不完整。**\n")

    # ---- 1. 总览
    L += ["## 1. 总览", ""]
    head = "| 指标 | " + " | ".join(SPLIT_CN[sp] for sp in splits) + " | 判定方式 |"
    L += [head, "|---|" + "---|" * (len(splits) + 1)]

    def row(name: str, fn, how: str) -> None:
        L.append(f"| {name} | " + " | ".join(fn(aggs[sp]) for sp in splits) + f" | {how} |")

    row("**应作答的题**", lambda m: str(m["n_answer_items"]), "")
    row("证据全部召回（前 k 条含全部所需片段）", lambda m: frac(*m["evidence_all"], ci=True), "确定性")
    row("误拒（该答却拒答）", lambda m: frac(*m["false_refusal"], ci=True)
        + f"；闸 {m['false_refusal_gate']} / 模型自述 {m['false_refusal_model']}", "确定性")
    row("关键事实召回", lambda m: frac(*m["fact_recall"]), "确定性（正则）")
    row("完全答对（关键事实全部答出）", lambda m: frac(*m["strict_correct"], ci=True), "确定性（正则）")
    row("完全答对且引用了证据片段", lambda m: frac(*m["strict_grounded"], ci=True), "确定性")
    row("数值事实答对", lambda m: frac(*m["numeric"]), "确定性（正则）")
    row("含数值的题全部数值答对", lambda m: frac(*m["numeric_items"], ci=True), "确定性（正则）")
    row("作答时引用了证据片段", lambda m: frac(*m["cited_gold"]), "确定性")
    row("串手册（引用的片段全部出自别的手册）", lambda m: frac(*m["wrong_manual"]), "确定性")
    row("**应拒答的题**", lambda m: str(m["n_refuse_items"]), "")
    row("拒答召回（库外题被拒）", lambda m: frac(*m["refusal_recall"], ci=True)
        + f"；闸 {m['refusal_gate']} / 模型自述 {m['refusal_model']}", "确定性")
    row("拒答精确率（被拒的题里确实该拒的）", lambda m: frac(*m["refusal_precision"]), "确定性")
    row("**实际作答的答案**", lambda m: str(m["n_answered"]), "")
    row("忠实度·严格（句子有依据且摘抄的原文对得上）", lambda m: frac(*m["faithful_strict"]), "评审 + 代码核对摘抄")
    row("忠实度·评审口径（评审判为有依据）", lambda m: frac(*m["faithful_full"]), "评审")
    row("无依据句占比", lambda m: frac(*m["unsupported"]), "评审")
    row("引用成立（句子标的片段确实支持它）", lambda m: frac(*m["citation_ok"]), "评审")
    row("句子与检索片段的字面重合度：中位数 / 低于 0.5 的句数",
        lambda m: "—" if not m["overlap"]["n"] else
        f"{m['overlap']['median']:.2f} / {m['overlap']['low']}（共 {m['overlap']['n']} 句）", "确定性（4 字片段）")
    row("无引用句占比", lambda m: frac(*m["uncited"]), "确定性")
    row("答案中的数字可在检索片段里找到", lambda m: frac(*m["numbers_traced"]), "确定性")
    row("虚构的 chunk_id 引用（条）", lambda m: str(m["fabricated_citations"]), "确定性")
    row("引用了自造样例手册的答案（条）", lambda m: str(m["answers_citing_sample"]), "确定性")
    row("被长度上限截断的答案（条）", lambda m: str(m["truncated"]), "finish_reason")
    row("作答耗时 P50 / P95（ms）", lambda m: f"{ms(m['latency']['p50'])} / {ms(m['latency']['p95'])}", "计时")
    row("首 token P50（ms）", lambda m: ms(m["latency"]["ttft_p50"]), "计时")
    row("单题 token 均值（输入 / 输出）", lambda m: "—" if m["tokens"]["input_mean"] is None else
        f"{m['tokens']['input_mean']:.0f} / {m['tokens']['output_mean']:.0f}", "接口返回")
    row("单题成本均值（元，输入全按未命中缓存计）", lambda m: "—" if m["cost"]["upper_mean"] is None else
        f"{m['cost']['upper_mean']:.5f}", "token × 单价")
    if any(aggs[sp]["cost"]["actual_mean"] is not None for sp in splits):
        row("单题成本均值（元，按接口返回的缓存命中）", lambda m: "—" if m["cost"]["actual_mean"] is None else
            f"{m['cost']['actual_mean']:.5f}", "token × 单价")
    L.append("")
    L.append("> Agent 的换算、核对结果以工具输出为依据（评审看到的 calc_ 编号）：这类句子「引用成立」的条件是"
             "所标片段为核对的限值出处，或含有被换算的原始数值（代码核对）；数字溯源也把工具输出算作出处，"
             "带小数的数是工具精确值的正确舍入也算。")
    L.append(f"> 成本按 config llm_pricing 的高峰时段单价估算（{meta.get('pricing_note', '')}），作答和模型自述拒答的题都计入，"
             "被拒答闸拦下的题没有调用模型、记 0 元；耗时与 token 只统计实际作答的题。")
    L.append("> 「关键事实召回」把被误拒的题记为一条都没答出；只看实际作答的题，召回为 "
             + "、".join(f"{sp} {frac(*aggs[sp]['fact_recall_answered'])}" for sp in splits) + "。")
    L.append("> 「误拒」只统计拒答闸拦截和不带引用的拒答话术。另有一些回答带着引用说明「片段里没有这项内容」，"
             "按规则算作答（关键事实未答出），评审认为它们实质上是在说资料不足的有 "
             + "、".join(f"{sp} {aggs[sp]['answered_but_declined']} 题" for sp in splits) + "（第 6 节有标注）。")
    L.append("")

    # ---- 2. 按题型 / 手册
    L += ["## 2. 应作答的题分组结果", ""]
    for sp in splits:
        L += [f"### {sp} · 按题型", ""]
        L += group_table(by_split[sp], lambda r: r["type"], TYPES_ANSWER, lambda k: f"{TYPE_CN[k]}（{k}）")
        docs = sorted({"+".join(r["docs"]) for r in by_split[sp] if r["expect"] == "answer"})
        L += ["", f"### {sp} · 按手册", ""]
        L += group_table(by_split[sp], lambda r: "+".join(r["docs"]), docs, lambda k: f"手册 {k}")
        L.append("")

    # ---- 3. 拒答
    L += ["## 3. 应拒答的题", ""]
    L += ["| 划分 | 类型 | 题数 | 拒答闸拦截 | 模型自述拒答 | 照常作答 | 其中评审认为实质是在说资料不足 |", "|---|---|---|---|---|---|---|"]
    for sp in splits:
        for typ in TYPES_REFUSE:
            sub = [r for r in by_split[sp] if r["type"] == typ and r["behavior"] != "error"]
            if not sub:
                continue
            c = Counter(r["behavior"] for r in sub)
            declined = sum(1 for r in sub if r["behavior"] == "answered" and (r.get("judge") or {}).get("declined"))
            L.append(f"| {sp} | {TYPE_CN[typ]} | {len(sub)} | {c['rejected_gate']} | {c['refused_model']} "
                     f"| {c['answered']} | {declined} |")
    L.append("")
    for sp in splits:
        leaked = [r for r in by_split[sp] if r["expect"] == "refuse" and r["behavior"] == "answered"]
        if leaked:
            L.append(f"**{sp} 中照常作答的库外题：**\n")
            for r in leaked:
                j = r.get("judge") or {}
                tag = "评审：实质是拒答" if j.get("declined") else "评审：给出了实质内容"
                if j.get("failed"):
                    tag = "评审失败"
                L.append(f"- `{r['id']}` {items[r['id']]['question']}（闸距离 {r['gate_distance']}；{tag}；"
                         f"引用 {r['cited'] or '无'}）")
            L.append("")
    L.append("拒答闸的区分度（向量 top-1 距离，τ 以上被拦）：\n" if mode != "agent" else
             "原问题的向量 top-1 距离分布（agent 模式不按它拦截，只在检索结果里提示相关度低，此处供对照）：\n")
    L += ["| 划分 | 题目 | 题数 | 最小 | 中位 | 最大 | 低于 τ 的题数 |", "|---|---|---|---|---|---|---|"]
    tau = sysinfo["tau_distance"]
    for sp in splits:
        for name, sub in (("应作答", [r for r in by_split[sp] if r["expect"] == "answer"]),
                          ("应拒答", [r for r in by_split[sp] if r["expect"] == "refuse"])):
            d = [r["gate_distance"] for r in sub if r.get("gate_distance") is not None]
            if d:
                L.append(f"| {sp} | {name} | {len(d)} | {min(d):.4f} | {pct(d, 50):.4f} | {max(d):.4f} "
                         f"| {sum(x <= tau for x in d)} |")
    L.append("")

    # ---- 4. 多轮
    mt = [s for s in scores if s["type"] == "multi_turn" and s["behavior"] != "error"]
    if mt:
        main_name = "带前文历史" if mode == "agent" else "只发最后一句"
        intro = ("agent 模式把前文（history）连同追问一起发给系统（main），由模型结合上文改写后检索；"
                 if mode == "agent" else "rag 模式不看对话历史，只把最后一句发给系统（main）；")
        L += [f"## 4. 多轮追问：{main_name} vs 改写成独立问题", "",
              intro + "standalone 是把同一道题人工改写成不依赖前文的问句再问一次，"
              "相当于「追问被完美改写」时的参考上限。", "",
              "| 划分 | 问法 | 题数 | 证据全部召回 | 误拒 | 关键事实召回 | 完全答对 |", "|---|---|---|---|---|---|---|"]
        for sp in splits:
            for variant, name in (("main", main_name), ("standalone", "改写成独立问题")):
                sub = [s for s in mt if s["variant"] == variant and split[s["id"]] == sp]
                if sub:
                    L.append(f"| {sp} | {name} | {len(sub)} | {frac(sum(r['evidence_all'] for r in sub), len(sub))} "
                             f"| {frac(sum(r['refused'] for r in sub), len(sub))} "
                             f"| {frac(sum(r['n_facts_matched'] for r in sub), sum(r['n_facts'] for r in sub))} "
                             f"| {frac(sum(r['all_facts'] for r in sub), len(sub))} |")
        L.append("")

    # ---- Agent 行为
    if any("agent" in aggs[sp] for sp in splits):
        L += ["## Agent 行为（工具调用）", "",
              "每题先由系统用原问题检索一次（预检索，不算模型的工具调用）；之后模型可以再调工具，"
              f"最多 {sysinfo.get('max_tool_rounds')} 轮，到上限后强制作答。", "",
              "| 划分 | 题数 | 模型调用次数均值 | 分布（1/2/3/4 次） | 只看预检索就作答 | 工具调用均值（不含预检索） "
              "| 有工具报错的题 | 达到轮数上限 | 输入 token 中缓存命中占比 | 期望的工具都用到了 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for sp in splits:
            a = aggs[sp].get("agent")
            if not a:
                continue
            dist = " / ".join(str(a["llm_calls_dist"].get(k, 0)) for k in range(1, 5))
            L.append(f"| {sp} | {a['n']} | {a['llm_calls_mean']:.2f} | {dist} | {frac(a['presearch_only'], a['n'])} "
                     f"| {a['tool_calls_mean']:.2f} | {a['tool_errors_items']}（共 {a['tool_errors']} 次） "
                     f"| {a['forced_final']} | {frac(*a['cache_read_share'])} | {frac(*a['tools_expected_ok'])} |")
        tools_all = sorted({t for sp in splits for t in (aggs[sp].get("agent") or {}).get("tool_usage", {})})
        if tools_all:
            L += ["", "| 工具 | " + " | ".join(f"{sp} 用到的题数" for sp in splits) + " |",
                  "|---|" + "---|" * len(splits)]
            for t in tools_all:
                L.append(f"| {t} | " + " | ".join(str((aggs[sp].get("agent") or {}).get("tool_usage", {}).get(t, 0))
                                                  for sp in splits) + " |")
        L += ["", "> 「期望的工具都用到了」只统计评测集里标了 tools 的题，只作参考、不计入对错："
              "答对的方式不止一种（比如查到的片段里已经同时写了两种单位，就不必再换算）。", ""]
        if any("numcheck" in (aggs[sp].get("agent") or {}) for sp in splits):
            L += ["### 数值核对与改写（系统自己的核对，见 src/s4_agent/numcheck.py）", "",
                  "回答里的每个「数值 + 单位」要能在本次的资料（问题、工具返回的内容、换算/核对结果）里找到、单位一致；"
                  f"对不上的交给模型改写一次（至多 {sysinfo.get('max_repairs')} 次），改得更差就退回初稿。"
                  "下表是系统自己的核对结果，与上面评测的「数字可溯源」不是同一套代码。", "",
                  "| 划分 | 题数 | 核对的数值 | 触发改写的题 | 改写后核对通过 | 退回初稿 | 最终仍有核对不上的数值的题 |",
                  "|---|---|---|---|---|---|---|"]
            for sp in splits:
                nc = (aggs[sp].get("agent") or {}).get("numcheck")
                if nc:
                    L.append(f"| {sp} | {nc['n']} | {nc['numbers']} | {nc['repairs']} | {nc['repairs_fixed']} "
                             f"| {nc['repairs_kept_draft']} | {nc['issue_answers']} |")
            L.append("")

    # ---- 5. 评审自身的可靠性
    L += ["## 5. 评审模型的可靠性线索", "",
          "| 划分 | 忠实度评审失败 | 判为有依据但摘抄对不上原文的句子 | 关键事实：正则与评审一致 | 仅正则判为答出 "
          "| 仅评审判为答出 | 评审判为与参考答案矛盾 | 关键事实评审失败 |", "|---|---|---|---|---|---|---|---|"]
    for sp in splits:
        m = aggs[sp]
        L.append(f"| {sp} | {m['judge_failed']}/{m['judge_answers']} "
                 f"| {m['faithful_full'][0] - m['faithful_strict'][0]} | {frac(*m['fact_agree'])} "
                 f"| {m['fact_only_regex']} | {m['fact_only_judge']} | {m['fact_contradicted']} "
                 f"| {m['fact_judge_failed']} |")
    L.append("")
    L.append("> 「仅评审判为答出」多半是正则没覆盖到的说法（指标偏严）；「仅正则判为答出」多半是答案里出现了"
             "关键词但意思不对（指标偏松）。这两类和「摘抄对不上」的句子是人工复核的重点"
             "（calibrate.py export 会把它们排在最前面）。")
    L.append("> 评审会不会把有问题的句子也判成有依据，见 judge_probe.md：往 dev 答案里注入改数字、换反义词、"
             "编造句三种错误后的检出率。")
    L.append("")

    # ---- 6. 逐题问题清单
    L += ["## 6. 没有完全答对的题", ""]
    for sp in splits:
        L += [f"### {sp}", ""]
        n_bad = 0
        for r in by_split[sp]:
            if r["expect"] != "answer" or r["behavior"] == "error" or r["all_facts"]:
                continue
            n_bad += 1
            if r["refused"]:
                why = "被拒答闸拦截" if r["behavior"] == "rejected_gate" else "模型自述资料不足"
                why += f"（闸距离 {r['gate_distance']}）"
            else:
                missing = [f["say"] for f in r["facts"] if not f["matched"]]
                why = "缺：" + "；".join(missing)
                if (r.get("judge") or {}).get("declined"):
                    why += "（评审：回答实质上是在说资料不足）"
            ev = "证据已全部召回" if r["evidence_all"] else f"证据召回 {r['evidence_hit']}/{r['evidence_groups']}"
            if r.get("wrong_manual"):
                ev += f"；串手册（引用的是手册 {'、'.join(r['cited_docs'])}）"
            L.append(f"- `{r['id']}`（{TYPE_CN[r['type']]}）{items[r['id']]['question']} —— {why}；{ev}")
        if not n_bad:
            L.append("（无）")
        L.append("")

    unsupported = []
    for sp in splits:
        for r in by_split[sp]:
            j = r.get("judge")
            if not j or j.get("failed") or r["behavior"] != "answered":
                continue
            for s_text, s in zip(r["sentence_texts"], j["sentences"]):
                if s["support"] == "none":
                    unsupported.append((sp, r["id"], s_text))
    L += ["## 7. 评审判为无依据的句子", ""]
    if unsupported:
        for sp, rid, text in unsupported:
            L.append(f"- [{sp}] `{rid}`：{text}")
    else:
        L.append("（无）")
    L.append("")

    untraced = [(sp, r["id"], r["untraced_numbers"]) for sp in splits for r in by_split[sp]
                if r["behavior"] == "answered" and r["untraced_numbers"]]
    L += ["## 8. 答案里在检索片段中找不到的数字", ""]
    if untraced:
        for sp, rid, nums in untraced:
            L.append(f"- [{sp}] `{rid}`：{', '.join(f'{x:g}' for x in nums)}")
    else:
        L.append("（无）")
    L.append("")

    L += ["## 9. 局限", "",
          "- 评测集由 Claude 依据手册解析文本编写并经脚本校验，尚未经人工逐条复核；参考答案与关键事实的正则可能有疏漏。",
          "- 关键事实用正则匹配：答案换了说法可能漏判，出现关键词但意思不对可能误判；第 5 节给出了它与评审模型的分歧数量。",
          "- 评审模型与生成模型相同，忠实度可能偏乐观；人工校准样本由 calibrate.py 导出，人工判定填写之前，"
          "不能把评审给出的数字当作人工结论。",
          "- 忠实度衡量的是「句子有没有检索片段作依据」；依据出自别的手册时句子照样算忠实，所以另有「串手册」一项。",
          "- 单次运行、temperature=0，没有重复采样；DeepSeek 在 temperature=0 下也不保证逐字可复现。",
          "- 拒答判定为「命中拒答话术且没有引用任何片段」，绕着说的拒答可能被记成作答（第 3 节单列了评审的看法）。",
          ""]
    return L, {sp: _jsonable(aggs[sp]) for sp in splits}


def _jsonable(m: dict) -> dict:
    out = {}
    for k, v in m.items():
        if isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
            out[k] = {"k": v[0], "n": v[1], "rate": (v[0] / v[1]) if v[1] else None}
        else:
            out[k] = v
    return out


def write_csv(scores: list[dict], items: dict[str, dict], split: dict[str, str], path: Path) -> None:
    headers = ["题目ID", "变体", "划分", "题型", "手册", "期望", "行为", "闸距离", "证据命中", "证据组数",
               "事实答出", "事实总数", "数值答出", "数值总数", "完全答对", "实质句数", "无引用句数",
               "评审有依据句数", "评审无依据句数", "答案数字数", "找不到出处的数字数", "引用样例手册",
               "耗时ms", "输入token", "输出token", "结束原因", "模式", "模型调用次数", "用到的工具", "工具报错次数",
               "成本元(未命中计)", "问题"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for s in scores:
            j = s.get("judge") or {}
            sup = Counter(x["support"] for x in j.get("sentences", [])) if not j.get("failed") else Counter()
            is_ans = s["expect"] == "answer"
            w.writerow([
                s["id"], s["variant"], split[s["id"]], s["type"], "+".join(s["docs"]), s["expect"], s["behavior"],
                s.get("gate_distance"), s.get("evidence_hit", "") if is_ans else "",
                s.get("evidence_groups", "") if is_ans else "",
                s.get("n_facts_matched", "") if is_ans else "", s.get("n_facts", "") if is_ans else "",
                s.get("n_numeric_matched", "") if is_ans else "", s.get("n_numeric", "") if is_ans else "",
                ("是" if s.get("all_facts") else "否") if is_ans else "",
                s["n_sentences"], s["n_uncited"], sup["full"] if j else "", sup["none"] if j else "",
                s["n_numbers"], s["n_untraced"], "是" if s["cited_sample"] else "",
                s.get("t_total_ms"), s.get("input_tokens"), s.get("output_tokens"), s.get("finish_reason") or "",
                s.get("mode") or "", s.get("llm_calls") if s.get("llm_calls") is not None else "",
                "、".join(s.get("tools_used") or []), s.get("tool_errors") or 0,
                "" if s.get("cost_upper") is None else round(s["cost_upper"], 6),
                items[s["id"]]["question"] if s["variant"] == "main" else items[s["id"]].get("standalone", ""),
            ])


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
