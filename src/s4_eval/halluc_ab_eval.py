"""S4 补充实验：引用标注对照评测（A 无引用要求 / B 末尾标注 / C 逐句标注）。

脚本名沿用早期「幻觉抑制」的叫法；指标「无引用句占比」衡量的是引用标注有没有覆盖到
每一句，不衡量所引片段是否支持该句，也不衡量答案对错。

三组提示词都读 configs/config.yaml prompts 段的文件（线上用的是其中一版，见
s4_agent.system_prompt），以 B 组为基准做漂移自检：
  - A 组 = B 组删掉「答案末尾必须标注引用的 chunk_id」这一条、其余行原文照抄；
  - C 组 = B 组仅把引用条款替换为「每一句末尾标注」、其余行原文照抄，用于消除
    「末尾标注」与逐句评测口径的错配；
  - 检索与上下文组装复用 graph/hybrid_retrieval 同一套函数；引用标记识别与 chunk_id
    有效性判定复用 verify 节点的解析逻辑（CHUNK_ID_RE / _move_citations_before_punct /
    verify_citation），不另写一套正则。

关键控制：36 题各跑一次混合检索并落盘缓存（含向量 top-1 distance 与候选 chunk 全集），
各组从同一份缓存取 context，只换系统提示词；temperature 从 configs/config.yaml
读取（固定 0），各组共用。

答案复用：磁盘 answers_raw.jsonl 里某组若对本题集全部存在且 prompt_hash / 候选集合
与当前一致，默认直接复用不调 LLM（增量跑 C 组不重复花 A/B 的钱）；
--reuse-answers 强制全组零调用，--force-llm 强制全部重跑。

产出：
  1. 无引用句占比 = 无引用实质性句子数 / 实质性句子总数（剔除 <10 字片段、纯过渡句、整句拒答）
  2. 虚构文档检出条数：从答案抽取来源标识（chunk_id），不在该题检索候选集合内则计 1 条
     —— 复用 verify.verify_citation 的 fabricated_ids，不重实现判定
  3. 拒答率（控制变量）：拒答闸命中（检索侧决定，两组必然相同）+ 模型自述拒答（可能不同）

用法：
    python src/s4_eval/halluc_ab_eval.py                  # 按 config groups 跑全量（可复用的组自动复用）
    python src/s4_eval/halluc_ab_eval.py --groups C        # 只跑/只统计 C 组
    python src/s4_eval/halluc_ab_eval.py --rebuild-cache  # 强制重建检索缓存
    python src/s4_eval/halluc_ab_eval.py --reuse-answers  # 全组从磁盘复算，零 LLM 调用
    python src/s4_eval/halluc_ab_eval.py --force-llm      # 忽略磁盘答案，全部重新生成
    python src/s4_eval/halluc_ab_eval.py --limit 4        # 冒烟：只跑前 4 题（缓存仍全量建）

失败处理：单题 LLM 调用按配置显式重试，耗尽后记为 failed 并写入失败清单；只要有 failed
就以非零码退出，汇总数字标注为不完整，绝不把失败题当作成功样本参与统计。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
for _p in (ROOT / "src", ROOT / "src" / "s3_eval", ROOT / "src" / "s4_agent"):
    sys.path.insert(0, str(_p))

import yaml  # noqa: E402
from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from llm_config import LLM_CONFIG, client_kwargs  # noqa: E402
from graph import PROMPT_VERSION, TAU_DISTANCE, load_prompt  # noqa: E402
from hybrid_retrieval import build_search_text, hybrid_retrieve, vector_top1_distance  # noqa: E402
from verify import CHUNK_ID_RE, _move_citations_before_punct, verify_citation  # noqa: E402

# verify_citation 内部按 [。！？\n] 切句但未导出该正则；此处取同一切分口径，
# 并由 check_sentence_split() 用探针与 verify 实测对账，防止两处漂移。
SENT_SPLIT_RE = re.compile(r"[。！？\n]")

GROUP_A, GROUP_B, GROUP_C = "A", "B", "C"
GROUP_DESC = {GROUP_A: "弱约束（无引用标注要求）", GROUP_B: "强约束 + 答案末尾标注",
              GROUP_C: "强约束 + 逐句末尾标注"}
LAYER_CN = {"text": "正文", "table": "表格"}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fmt(v: float | None, nd: int = 3) -> str:
    return "—" if v is None else f"{v:.{nd}f}"


def fmt_delta(new: float | None, old: float | None, nd: int = 3) -> str:
    if new is None or old is None:
        return "—"
    return f"{new - old:+.{nd}f}"


# --------------------------------------------------------------------------- 自检
def check_sentence_split(acfg: dict) -> None:
    """自检：本脚本的切句 + 引用识别与 verify 节点当前实现完全一致，否则显式报错。"""
    probe_chunks = [{"chunk_id": "1_c0001", "chunk_type": "text",
                     "text": "联轴器对中允差为 0.05mm，端面间隙同轴度见下表。"}]
    probe = ("联轴器对中允差为 0.05mm。[1_c0001]\n"
             "端面间隙要求不超过 0.03mm，需要用塞尺逐点测量确认。[1_c0001]")
    v = verify_citation(probe, probe_chunks)
    mine = [s for s in (x.strip() for x in SENT_SPLIT_RE.split(_move_citations_before_punct(probe)))
            if len(s) > 5]
    if len(mine) != v["total_sentences"]:
        raise RuntimeError(
            f"切句口径与 verify 不一致：本脚本 {len(mine)} 句 vs verify_citation "
            f"{v['total_sentences']} 句，请核对 verify 的切分正则是否已改动")
    for sent in v["uncited_sentences"]:
        if CHUNK_ID_RE.search(sent):
            raise RuntimeError(f"引用标记识别与 verify 不一致：{sent}")
    if verify_citation("端面间隙要求不超过 0.03mm。", probe_chunks)["fabricated_ids"]:
        raise RuntimeError("探针无引用标记却产出 fabricated_ids，verify 判定口径已变动")
    if verify_citation("参见 9_c9999。", probe_chunks)["fabricated_ids"] != ["9_c9999"]:
        raise RuntimeError("探针虚构 chunk_id 未被识别，verify 判定口径已变动")
    # 逐句标注风格探针（C 组形态）：句内带标记的整句必须全部判为有引用
    probe_c = ("联轴器对中允差为 0.05mm，需要用塞尺逐点测量确认 [1_c0001]。"
               "端面间隙要求不超过 0.03mm，复紧后需要再次测量验证 [1_c0001]。")
    sc = score_answer(probe_c, probe_chunks, acfg)
    if sc["n_substantive"] != 2 or sc["n_uncited"] != 0 or sc["n_fabricated"] != 0:
        raise RuntimeError(f"逐句标注探针判分异常：实质句 {sc['n_substantive']}（应 2），"
                           f"无引用 {sc['n_uncited']}（应 0），虚构 {sc['n_fabricated']}（应 0）")


def check_weak_prompt(base: str, weak: str, pattern: str, expect_dropped: int) -> list[str]:
    """自检：A 组提示词必须 = B 组提示词恰好删掉命中 pattern 的那条，其余行原文照抄。"""
    pl = [x.strip() for x in base.splitlines() if x.strip()]
    wl = [x.strip() for x in weak.splitlines() if x.strip()]
    dropped = [x for x in pl if x not in wl]
    if len(dropped) != expect_dropped:
        raise RuntimeError(f"A 组提示词应恰好删除 {expect_dropped} 行，实际删除 {len(dropped)} 行："
                           f"{dropped}；B 组提示词为：{pl}")
    if not all(pattern in d for d in dropped):
        raise RuntimeError(f"A 组删除的行未全部命中约束模式「{pattern}」：{dropped}")
    if wl != [x for x in pl if x not in dropped]:
        raise RuntimeError("A 组提示词除删除行外与 B 组不一致（顺序或文字被改动）："
                           f"A={wl} B={pl}")
    if wl[0] != pl[0]:
        raise RuntimeError(f"角色/任务首行被改动：A={wl[0]!r} B={pl[0]!r}")
    return dropped


def check_per_sentence_prompt(base: str, ctext: str, pattern: str, marker: str) -> list[str]:
    """自检：C 组提示词 = B 组提示词恰好把引用条款那一行替换为逐句口径，其余行原文照抄。"""
    pl = [x.strip() for x in base.splitlines() if x.strip()]
    cl = [x.strip() for x in ctext.splitlines() if x.strip()]
    if len(pl) != len(cl):
        raise RuntimeError(f"C 组提示词行数与 B 组不同（C={len(cl)} 行，B={len(pl)} 行），"
                           "应为原文替换恰好 1 行")
    diffs = [(x, y) for x, y in zip(pl, cl) if x != y]
    if len(diffs) != 1:
        raise RuntimeError(f"C 组提示词应恰好改动 1 行，实际 {len(diffs)} 行：{diffs}；"
                           f"B 组为：{pl}")
    p_line, c_line = diffs[0]
    if pattern not in p_line:
        raise RuntimeError(f"C 组改动行不是引用标注条款（B 组对应行未命中「{pattern}」）：{p_line}")
    if marker not in c_line:
        raise RuntimeError(f"C 组替换行未包含「{marker}」，不是逐句标注口径：{c_line}")
    if pl[0] != cl[0]:
        raise RuntimeError(f"角色/任务首行被改动：C={cl[0]!r} B={pl[0]!r}")
    return [p_line, c_line]


# --------------------------------------------------------------------------- 缓存
def fingerprint(qa_set: list[dict]) -> str:
    payload = "\n".join(f"{r['qa_id']}|{r['question']}" for r in qa_set)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_cache(qa_set: list[dict], top_k: int, cache_path: Path, meta_path: Path) -> dict:
    """36 题各跑一次混合检索并落盘，供 A/B 两组共享。"""
    rows = []
    for i, r in enumerate(qa_set, 1):
        chunks, scores = hybrid_retrieve(r["question"], top_k=top_k)
        dist = vector_top1_distance(r["question"])
        rows.append({"qa_id": r["qa_id"], "question": r["question"],
                     "top1_distance": round(dist, 6), "chunks": chunks, "rrf_scores": scores})
        print(f"检索 {i}/{len(qa_set)}  {r['qa_id']}  top1_dist={dist:.4f}  "
              f"候选={','.join(c['chunk_id'] for c in chunks)}")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n",
                          encoding="utf-8")
    meta_path.write_text(json.dumps({
        "fingerprint": fingerprint(qa_set), "top_k": top_k,
        "tau_distance": TAU_DISTANCE, "n_questions": len(qa_set), "built_at": time.strftime("%F %T"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"检索缓存已写入 {cache_path}\n")
    return {x["qa_id"]: x for x in rows}


def ensure_cache(qa_set: list[dict], top_k: int, cache_path: Path, meta_path: Path,
                 rebuild: bool) -> dict:
    if not rebuild and cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        problems = []
        if meta.get("fingerprint") != fingerprint(qa_set):
            problems.append("题目集或问题文本已变更")
        if meta.get("top_k") != top_k:
            problems.append(f"top_k 由 {meta.get('top_k')} 变为 {top_k}")
        if meta.get("n_questions") != len(qa_set):
            problems.append(f"缓存题数 {meta.get('n_questions')} != 当前题数 {len(qa_set)}")
        if problems:
            raise RuntimeError("检索缓存失效：" + "；".join(problems)
                               + "。请确认这些变更是有意的，然后加 --rebuild-cache 重建。")
        rows = load_jsonl(cache_path)
        print(f"复用检索缓存 {cache_path}（{len(rows)} 题，建于 {meta.get('built_at')}）\n")
        return {x["qa_id"]: x for x in rows}
    if cache_path.exists() and not rebuild:
        raise RuntimeError(f"检索缓存 {cache_path} 缺少指纹文件 {meta_path}，"
                           "无法确认与当前配置一致。请加 --rebuild-cache 重建。")
    return build_cache(qa_set, top_k, cache_path, meta_path)


# --------------------------------------------------------------------------- 生成
def make_llm(temperature: float, max_tokens: int):
    """生成上限经 extra_body 传 max_tokens（原因见 llm_config.client_kwargs）。"""
    if not LLM_CONFIG["api_key"]:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置，无法调用 LLM。请先配置 .env。")
    return ChatOpenAI(**client_kwargs(temperature=temperature, max_tokens=max_tokens))


def gen_answer(llm, system_prompt: str, question: str, chunks: list[dict],
               retries: int, retry_sleep: float) -> str:
    """上下文与用户消息与 graph.generate / api._sse_events 完全同构，只换系统提示词。"""
    context = "\n\n".join(f"[{c['chunk_id']}]\n{build_search_text(c)}" for c in chunks)
    user = f"问题：{question}\n\n检索到的内容：\n{context}"
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user)]
    last: Exception | None = None
    for attempt in range(1, retries + 2):
        try:
            return (llm.invoke(messages).content or "").strip()
        except Exception as exc:  # 显式打印后重试，耗尽则抛出，不静默
            last = exc
            print(f"  !! LLM 第 {attempt} 次调用失败：{type(exc).__name__}: {exc}", file=sys.stderr)
            if attempt <= retries:
                time.sleep(retry_sleep)
    raise RuntimeError(f"LLM 调用重试 {retries} 次后仍失败：{type(last).__name__}: {last}") from last


# --------------------------------------------------------------------------- 指标
def is_refusal(text: str, markers: list[str], max_chars: int) -> bool:
    t = "".join(text.split())
    return len(t) <= max_chars and any(m in t for m in markers)


def substantive_sentences(answer: str, acfg: dict) -> list[str]:
    """实质性句子：切句后剔除 <min_sentence_chars 的片段、纯过渡句、整句拒答。"""
    min_chars = int(acfg["min_sentence_chars"])
    end_markers = list(acfg["transition_end_markers"])
    trans_markers = list(acfg["transition_markers"])
    trans_max = int(acfg["transition_max_chars"])
    refusal_markers = list(acfg["refusal_markers"])
    refusal_max = int(acfg["refusal_max_chars"])
    out = []
    for s in (x.strip() for x in SENT_SPLIT_RE.split(_move_citations_before_punct(answer))):
        if len(s) < min_chars:
            continue
        if any(s.endswith(m) for m in end_markers):
            continue
        if len(s) <= trans_max and any(m in s for m in trans_markers):
            continue
        if is_refusal(s, refusal_markers, refusal_max):
            continue
        out.append(s)
    return out


def score_answer(answer: str, chunks: list[dict], acfg: dict) -> dict:
    """无引用句占比 + 虚构文档条数（来源标识抽取与判定全部复用 verify 的实现）。"""
    sents = substantive_sentences(answer, acfg)
    cited = [s for s in sents if CHUNK_ID_RE.search(s)]
    v = verify_citation(answer, chunks)          # 复用 verify：来源标识 ∈ 候选 chunk_id 集合
    fabricated = list(v["fabricated_ids"])       # 候选集合外的来源标识，逐个计 1 条
    return {
        "n_substantive": len(sents),
        "n_cited": len(cited),
        "n_uncited": len(sents) - len(cited),
        "uncited_ratio": (len(sents) - len(cited)) / len(sents) if sents else None,
        "n_fabricated": len(fabricated),
        "fabricated_ids": fabricated,
        "cited_ids": v["cited_ids"],
        "suspicious_count": v["suspicious_count"],
    }


def aggregate(rows: list[dict]) -> dict:
    """按指标定义聚合：拒答题整题排除，失败题排除出所有分母并单独计数。"""
    failed = [r for r in rows if r["status"] == "failed"]
    rejected = [r for r in rows if r["status"].startswith("rejected")]
    scored = [r for r in rows if r["status"] == "scored"]
    n_pop = len(scored) + len(rejected)
    sub = sum(r["n_substantive"] for r in scored)
    unc = sum(r["n_uncited"] for r in scored)
    fab = sum(r["n_fabricated"] for r in scored)
    return {
        "n_rows": len(rows), "n_failed": len(failed), "n_scored": len(scored),
        "n_rej_gate": sum(1 for r in rejected if r["status"] == "rejected_gate"),
        "n_rej_model": sum(1 for r in rejected if r["status"] == "rejected_model"),
        "reject_rate": (len(rejected) / n_pop) if n_pop else None,
        "n_substantive": sub, "n_cited": sub - unc, "n_uncited": unc,
        "uncited_ratio": (unc / sub) if sub else None,
        "n_fabricated": fab,
        "fabricated_per_q": (fab / len(scored)) if scored else None,
        "failed": failed,
    }


# --------------------------------------------------------------------------- 报告
def compare_table(buckets: list[tuple[str, dict[str, list]]], groups: list[str]) -> list[str]:
    cols = ["分组", "n"]
    cols += [f"{g} 无引用占比" for g in groups]
    cols += [f"Δ({g}−{groups[0]})" for g in groups[1:]]
    cols += [f"{g} 虚构条数" for g in groups]
    cols += [f"{g} 拒答率" for g in groups]
    L = ["| " + " | ".join(cols) + " |",
         "|" + "|".join(["---"] * len(cols)) + "|"]
    for name, per_group in buckets:
        aggs = {g: aggregate(per_group[g]) for g in groups}
        cells = [name, str(aggs[groups[0]]["n_rows"])]
        cells += [fmt(aggs[g]["uncited_ratio"]) for g in groups]
        cells += [fmt_delta(aggs[g]["uncited_ratio"], aggs[groups[0]]["uncited_ratio"])
                  for g in groups[1:]]
        cells += [str(aggs[g]["n_fabricated"]) for g in groups]
        cells += [fmt(aggs[g]["reject_rate"]) for g in groups]
        L.append("| " + " | ".join(cells) + " |")
    return L


def build_report(acfg: dict, rows: list[dict], dropped: list[str], c_diff: list[str],
                 groups: list[str], meta_note: str, max_tokens: int,
                 temperature: float, n_legacy: int = 0) -> tuple[list[str], dict]:
    qa_by_id = {r["qa_id"]: r for r in rows}
    per_group = {g: [r for r in rows if r["group"] == g] for g in groups}
    aggs = {g: aggregate(per_group[g]) for g in groups}
    base = groups[0]

    def bucket_rows(key_fn, order: list[str]):
        return [(k, {g: [r for r in per_group[g] if key_fn(r) == k] for g in groups})
                for k in order]

    layers = bucket_rows(lambda r: r["layer"], sorted({r["layer"] for r in qa_by_id.values()}))
    diffs = bucket_rows(lambda r: r["difficulty"],
                        [d for d in ("easy", "medium", "hard") if any(r["difficulty"] == d for r in qa_by_id.values())])

    title = " vs ".join(f"{g} {GROUP_DESC[g]}" for g in groups)
    L = [f"# S4 引用标注对照评测（{title}）", ""]
    L.append(f"> 生成脚本：src/s4_eval/halluc_ab_eval.py；LLM：{LLM_CONFIG['model']}"
             f"（temperature={temperature} 取自 configs/config.yaml s4_halluc_ab.temperature，"
             f"max_tokens={max_tokens}，top_k={acfg['top_k']}，τ={TAU_DISTANCE}）")
    if n_legacy:
        L.append(f"> 注意：本次复用的 {n_legacy} 条答案生成于 max_tokens 修复（2026-09-29）之前：当时 langchain-openai "
                 "把 max_tokens 改名为 max_completion_tokens 发出，DeepSeek 不识别，上述 max_tokens 上限并未生效。")
    L.append(f"> 检索：{meta_note}；{'、'.join(groups)} 各组每题取同一份缓存的候选 chunk，只换系统提示词。")
    if GROUP_A in groups and dropped:
        L.append(f"> 提示词差异：A 组相对 B 组仅删除 {len(dropped)} 行 → "
                 + "；".join(f"「{d}」" for d in dropped))
    if GROUP_C in groups and c_diff:
        L.append(f"> 提示词差异：C 组相对 B 组仅替换 1 行 → 「{c_diff[0]}」改为「{c_diff[1]}」")
    L.append(f"> 线上当前使用 {PROMPT_VERSION} 组提示词（configs/config.yaml s4_agent.system_prompt）。")
    L.append("> 口径：引用标记识别与来源标识有效性判定复用 verify 节点（CHUNK_ID_RE / "
             "_move_citations_before_punct / verify_citation），未重写正则。"
             "「无引用句占比」衡量引用标注的覆盖，不衡量所引片段是否支持该句，也不衡量答案对错。")
    L.append("")

    L.append("## 1. 总体\n")
    L.append(f"| 组别 | 说明 | 可打分题 | 实质句数 | 无引用句数 | 无引用句占比 | Δ占比(vs {base}) "
             "| 虚构检出条数 | 拒答(闸/自述) | 拒答率 | 失败 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for g in groups:
        a = aggs[g]
        delta = "—" if g == base else fmt_delta(a["uncited_ratio"], aggs[base]["uncited_ratio"])
        L.append(f"| {g} | {GROUP_DESC[g]} | {a['n_scored']} | {a['n_substantive']} | {a['n_uncited']} "
                 f"| **{fmt(a['uncited_ratio'])}** | {delta} | {a['n_fabricated']} "
                 f"| {a['n_rej_gate']}/{a['n_rej_model']} | {fmt(a['reject_rate'])} | {a['n_failed']} |")
    L.append("")
    L.append(f"> Δ占比为负表示该组无引用句更少（更好）；拒答率是控制变量，正值表示该组「少答」更多，"
             "解读时需在结论中扣减该成分。")
    L.append("")

    L.extend(["## 2. 按层（正文 / 表格）\n"])
    L.extend(compare_table(layers, groups))
    L.append("")
    L.extend(["## 3. 按难度\n"])
    L.extend(compare_table(diffs, groups))
    L.append("")

    L.append("## 4. 结论（数字均来自本次运行）\n")
    rats = {g: aggs[g]["uncited_ratio"] for g in groups}
    if all(v is not None for v in rats.values()):
        chain = " → ".join(f"{g} {rats[g]:.3f}" for g in groups)
        d_parts = "；".join(f"Δ({g}−{base}) {rats[g] - rats[base]:+.3f}"
                            for g in groups if g != base)
        extra = ""
        if GROUP_B in rats and GROUP_C in rats and rats[GROUP_B] is not None:
            extra = ("B 组残值说明：B 组提示词只要求「答案末尾标注」而非逐句标注，与本指标"
                     "「句内或句尾含标记」的口径天然不匹配，不能读成「大部分句子无依据」；"
                     "C 组即该口径错配的对照变体。")
        L.append(f"1. **无引用句占比**：{chain}（{d_parts}）。{extra}")
    has_cb = (GROUP_B in groups and GROUP_C in groups
              and rats.get(GROUP_B) is not None and rats.get(GROUP_C) is not None)
    if has_cb:
        d_cb = rats[GROUP_C] - rats[GROUP_B]
        L.append(f"2. **逐句标注的净效应（C−B）**：无引用句占比 {rats[GROUP_B]:.3f} → {rats[GROUP_C]:.3f}"
                 f"（Δ {d_cb:+.3f}）；实质句数 {aggs[GROUP_B]['n_substantive']} → "
                 f"{aggs[GROUP_C]['n_substantive']}（句数变化本身会影响 micro 占比，"
                 "需与 details.csv 逐题明细同读）。")
    idx = 3 if has_cb else 2
    rr = {g: aggs[g]["reject_rate"] for g in groups}
    if all(v is not None for v in rr.values()):
        parts = " / ".join(f"{g} {rr[g]:.3f}（闸 {aggs[g]['n_rej_gate']}+自述 {aggs[g]['n_rej_model']}）"
                           for g in groups)
        cmps = []
        for g in groups[1:]:
            d = rr[g] - rr[base]
            if abs(d) < 1e-9:
                cmps.append(f"{g} 与 {base} 持平，引用率改善不来自「少答」")
            elif d > 0:
                cmps.append(f"{g} 比 {base} 高 {d * 100:.1f} 个百分点，其引用率改善中混有"
                            "「少答」成分，解读时需扣减")
            else:
                cmps.append(f"{g} 比 {base} 低 {-d * 100:.1f} 个百分点，改善不是靠少答换来的")
        gates = {aggs[g]["n_rej_gate"] for g in groups}
        warn = "" if len(gates) == 1 else "；警告：各组拒答闸命中数不同，检索侧存在未受控变量，需排查"
        L.append(f"{idx}. **拒答率（控制变量）**：{parts} → {'；'.join(cmps)}{warn}。")
    idx += 1
    cited_total = {g: sum(len(r["cited_ids"]) for r in per_group[g] if r["status"] == "scored")
                   for g in groups}
    fab_parts = " / ".join(f"{g} 抽取 {cited_total[g]} 标识、检出 {aggs[g]['n_fabricated']} 条"
                           for g in groups)
    L.append(f"{idx}. **虚构文档检出条数**：{fab_parts}。口径为「来源标识（chunk_id）∉ 该题检索"
             "候选集合」；完全不打引用的组检出天然为 0，属分母效应，不能直接读成「该组不虚构」。")
    L.append("")

    fails = [r for r in rows if r["status"] == "failed"]
    L.append("## 5. 失败清单\n")
    if fails:
        for f in fails:
            L.append(f"- {f['group']} 组 {f['qa_id']}：{f['error']}")
        L.append("\n**本次结果不完整：以上题目失败并已排除出全部统计分母，不得视为成功样本。**")
    else:
        L.append("（无）")
    L.append("")

    L.append("## 6. 残留局限\n")
    L.append("- 36 题、temperature=0 的单次采样，无重复采样与显著性检验，差值只作方向性结论。")
    L.append("- 虚构文档口径为 chunk_id 集合比对，未纳入文件名/书名等非结构化来源标识，"
             "bigram 句级校验仅作旁证列出。")
    if GROUP_C in groups:
        live = ("C 组提示词已上线" if PROMPT_VERSION == GROUP_C
                else f"线上当前为 {PROMPT_VERSION} 组，C 组未上线")
        L.append("- 「答案末尾标注」与逐句口径的错配影响已由 C 组对照量化（见第 4 节）；"
                 f"{live}。无引用句占比低只说明每句都带了引用标记，所引片段是否真的支持该句、"
                 "答案是否正确，需要另做答案级评测。")
    else:
        L.append("- 「答案末尾标注」与逐句口径不匹配：若把线上提示词改为「逐句末尾标注」重跑，"
                 "B 组无引用句占比预期会进一台阶下降，本次未覆盖该变体。")
    L.append("- 无引用句占比按句池（micro）合并计算，长答案句数多的题权重更大；"
             "逐题明细见 details.csv，可按需改macro。")
    L.append("- 各组共用拒答闸，提示词无法影响闸内拒答，故各组的闸拒答必然相同。")
    L.append("")
    return L, aggs


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["题目ID", "问题", "层", "原始类型", "难度", "组别", "实质句数", "有引用句数",
               "无引用句数", "无引用句占比", "虚构文档数", "虚构来源标识", "是否拒答", "拒答来源",
               "状态", "错误"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            rejected = r["status"].startswith("rejected")
            w.writerow([
                r["qa_id"], r["question"], r["layer"], r["chunk_type"], r["difficulty"], r["group"],
                r["n_substantive"] if r["status"] == "scored" else "",
                r["n_cited"] if r["status"] == "scored" else "",
                r["n_uncited"] if r["status"] == "scored" else "",
                (f"{r['uncited_ratio']:.3f}" if r.get("uncited_ratio") is not None else ""),
                r["n_fabricated"] if r["status"] == "scored" else "",
                "、".join(r["fabricated_ids"]),
                "是" if rejected else "否",
                {"rejected_gate": "拒答闸", "rejected_model": "模型自述"}.get(r["status"], ""),
                r["status"], r.get("error") or "",
            ])


# --------------------------------------------------------------------------- 主流程
def resolve_groups(args, acfg) -> list[str]:
    raw = args.groups or ",".join(str(g) for g in acfg.get("groups", ["A", "B"]))
    gs = [x.strip().upper() for x in raw.split(",") if x.strip()]
    unknown = [g for g in gs if g not in GROUP_DESC]
    if unknown:
        raise RuntimeError(f"未知组别 {unknown}，可选：{'、'.join(sorted(GROUP_DESC))}")
    if not gs:
        raise RuntimeError("参与组别为空，检查 --groups 或 config s4_halluc_ab.groups。")
    if len(set(gs)) != len(gs):
        raise RuntimeError(f"组别重复：{gs}")
    return gs


def main() -> None:
    ap = argparse.ArgumentParser(description="幻觉抑制模块对照评测（不改生产代码）")
    ap.add_argument("--rebuild-cache", action="store_true", help="强制重建检索缓存")
    ap.add_argument("--reuse-answers", action="store_true",
                    help="全部组从 answers_raw.jsonl 复算，零 LLM 调用；任一组不可用则显式报错")
    ap.add_argument("--force-llm", action="store_true", help="忽略磁盘已有答案，所有组重新调用 LLM 生成")
    ap.add_argument("--groups", default="", help="参与组别，逗号分隔（如 A,B,C）；默认读 config groups")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用；缓存仍按全量题集构建）")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    acfg = cfg["s4_halluc_ab"]
    qa_path = ROOT / acfg["qa_set"]
    qa_full = load_jsonl(qa_path)
    cache_rows = ensure_cache(qa_full, int(acfg["top_k"]), ROOT / acfg["retrieval_cache"],
                             ROOT / acfg["retrieval_meta"], args.rebuild_cache)
    qa_run = qa_full[:args.limit] if args.limit else qa_full

    temperature = float(acfg["temperature"])
    max_tokens = int(acfg["max_tokens"])
    retries, retry_sleep = int(acfg["llm_retries"]), float(acfg["retry_sleep"])

    if args.reuse_answers and args.force_llm:
        raise RuntimeError("--reuse-answers（零调用复算）与 --force-llm（强制重跑）互斥。")
    groups = resolve_groups(args, acfg)

    # 提示词构建 + 漂移自检（未参与本轮的组也校验文件，防后续误用漂移版本）
    prompts = {g: load_prompt(g, cfg) for g in (GROUP_A, GROUP_B, GROUP_C)}
    dropped = check_weak_prompt(prompts[GROUP_B], prompts[GROUP_A],
                                acfg["citation_drop_pattern"], int(acfg["citation_drop_lines"]))
    c_diff = check_per_sentence_prompt(prompts[GROUP_B], prompts[GROUP_C],
                                       acfg["citation_drop_pattern"], acfg["per_sentence_marker"])
    check_sentence_split(acfg)
    print(f"自检通过：A 组 = B 组删 {len(dropped)} 行；C 组 = B 组替换 1 行为逐句口径；"
          f"切句与引用识别与 verify 一致；线上当前为 {PROMPT_VERSION} 组")
    print(f"题目数 {len(qa_run)}/{len(qa_full)}  组别 {'、'.join(groups)}  "
          f"temperature={temperature}  max_tokens={max_tokens}  τ={TAU_DISTANCE}\n")

    # 按组决策：磁盘答案完整且提示词/候选集合一致的组直接复用，不重复花 LLM 钱
    raw_path = ROOT / acfg["answers_raw"]
    disk: dict[str, dict[str, dict]] = {}
    for x in (load_jsonl(raw_path) if raw_path.exists() else []):
        disk.setdefault(str(x["group"]).upper(), {})[x["qa_id"]] = x
    cand_ids = {q["qa_id"]: [c["chunk_id"] for c in cache_rows[q["qa_id"]]["chunks"]]
                for q in qa_run}
    need_ids = sorted(q["qa_id"] for q in qa_run
                      if cache_rows[q["qa_id"]]["top1_distance"] <= TAU_DISTANCE)

    reuse_plan: dict[str, dict] = {}
    for g in groups:
        cur_hash = hashlib.sha1(prompts[g].encode("utf-8")).hexdigest()
        got = {qa: disk[g][qa] for qa in need_ids if qa in disk.get(g, {})}
        reason = ""
        hashes = {r.get("prompt_hash") for r in got.values()}
        if len(got) < len(need_ids):
            reason = f"磁盘缺 {len(need_ids) - len(got)} 题答案"
        elif hashes == {None}:
            print(f"提示：{g} 组磁盘记录为旧版无提示词哈希，按完整性与提示词自检结果复用，"
                  "本次写回时补齐哈希")
        elif hashes != {cur_hash}:
            reason = "磁盘记录的提示词哈希与当前提示词不符（提示词已改动或为旧版混合记录）"
        elif any(got[qa].get("candidate_chunk_ids") != cand_ids[qa] for qa in need_ids):
            reason = "磁盘记录的候选集合与当前检索缓存不一致"
        if args.force_llm:
            reason = reason or "--force-llm 强制重新生成"
        reuse_plan[g] = {"reuse": not reason, "reason": reason, "hash": cur_hash}
        print(f"组 {g}（{GROUP_DESC[g]}）："
              + (f"复用磁盘答案（{len(need_ids)} 题）" if reuse_plan[g]["reuse"]
                 else f"调用 LLM 生成（{len(need_ids)} 题，原因：{reason}）"))
    if args.reuse_answers:
        bad = [g for g in groups if not reuse_plan[g]["reuse"]]
        if bad:
            raise RuntimeError("--reuse-answers 要求全部组可从磁盘复算，但 " + "、".join(bad)
                               + " 不可（原因见上）。去掉该 flag 先补齐生成。")
    print()

    any_run = any(not p["reuse"] for p in reuse_plan.values())
    llm = None if not any_run else make_llm(temperature, max_tokens)

    rows: list[dict] = []
    answers: list[dict] = []
    n_legacy = 0   # 复用的、在 max_tokens 修复前生成的答案条数
    for group in groups:
        print(f"===== {group} 组：{GROUP_DESC[group]} =====")
        for i, q in enumerate(qa_run, 1):
            cr = cache_rows[q["qa_id"]]
            rec = {"qa_id": q["qa_id"], "question": q["question"], "group": group,
                   "chunk_type": q["chunk_type"], "layer": LAYER_CN[q["chunk_type"]],
                   "difficulty": q["difficulty"], "top1_distance": cr["top1_distance"],
                   "candidate_chunk_ids": [c["chunk_id"] for c in cr["chunks"]],
                   "n_substantive": 0, "n_cited": 0, "n_uncited": 0, "uncited_ratio": None,
                   "n_fabricated": 0, "fabricated_ids": [], "cited_ids": [],
                   "suspicious_count": 0, "answer": "", "error": None}
            if cr["top1_distance"] > TAU_DISTANCE:      # 拒答闸：各组必然同路径，不调 LLM
                rec["status"] = "rejected_gate"
                rows.append(rec)
                print(f"[{group} {i}/{len(qa_run)}] {q['qa_id']} 拒答（闸） "
                      f"dist={cr['top1_distance']:.4f}>τ")
                continue
            try:
                if reuse_plan[group]["reuse"]:
                    old = disk[group][q["qa_id"]]
                    rec["answer"] = old["answer"]
                    answers.append({**old, "prompt_hash": reuse_plan[group]["hash"]})
                    if "max_tokens" not in old:   # 修复前生成的答案不带该字段，当时上限未生效
                        n_legacy += 1
                else:
                    rec["answer"] = gen_answer(llm, prompts[group], q["question"], cr["chunks"],
                                               retries, retry_sleep)
                    answers.append({"qa_id": q["qa_id"], "group": group, "answer": rec["answer"],
                                    "candidate_chunk_ids": rec["candidate_chunk_ids"],
                                    "prompt_hash": reuse_plan[group]["hash"],
                                    "max_tokens": max_tokens})
            except Exception as exc:
                rec["status"] = "failed"
                rec["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(rec)
                print(f"[{group} {i}/{len(qa_run)}] {q['qa_id']} 失败：{rec['error']}",
                      file=sys.stderr)
                continue
            if is_refusal(rec["answer"], list(acfg["refusal_markers"]),
                          int(acfg["refusal_max_chars"])):
                rec["status"] = "rejected_model"       # 整题拒答：排除出引用统计
                rows.append(rec)
                print(f"[{group} {i}/{len(qa_run)}] {q['qa_id']} 拒答（模型自述）")
                continue
            rec.update(score_answer(rec["answer"], cr["chunks"], acfg))
            rec["status"] = "scored"
            rows.append(rec)
            ratio_s = "—" if rec["uncited_ratio"] is None else f"{rec['uncited_ratio']:.3f}"
            print(f"[{group} {i}/{len(qa_run)}] {q['qa_id']} 实质 {rec['n_substantive']} "
                  f"无引用 {rec['n_uncited']} 虚构 {rec['n_fabricated']} 无引用占比 {ratio_s}")

    if not rows:
        raise RuntimeError("没有任何题目产出结果，检查 --limit 与题集。")

    L, aggs = build_report(acfg, rows, dropped, c_diff, groups,
                           f"复用同一份落盘缓存（{len(cache_rows)} 题，top_k={acfg['top_k']}）",
                           max_tokens, temperature, n_legacy)

    detail_csv = ROOT / acfg["detail_csv"]
    summary_md = ROOT / acfg["summary_md"]
    write_csv(rows, detail_csv)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in answers) + "\n", encoding="utf-8")
    if args.limit:
        print(f"提示：--limit 仅覆盖前 {args.limit} 题，答案文件已按本次题集重写（{len(answers)} 条），"
              "全量结果请去掉 --limit 重跑")
    summary_md.parent.mkdir(parents=True, exist_ok=True)
    summary_md.write_text("\n".join(L), encoding="utf-8")

    print("\n" + "=" * 78)
    print("\n".join(L))
    print("=" * 78)
    print("关键数字｜无引用句占比 " + " → ".join(f"{g}={fmt(aggs[g]['uncited_ratio'])}" for g in groups))
    print("        ｜虚构文档检出 " + " / ".join(f"{g} {aggs[g]['n_fabricated']} 条" for g in groups))
    print("        ｜拒答率 " + " / ".join(f"{g}={fmt(aggs[g]['reject_rate'])}" for g in groups))
    print(f"明细 CSV：{detail_csv}")
    print(f"汇总报告：{summary_md}")
    print(f"原始答案：{raw_path}")

    n_failed = sum(1 for r in rows if r["status"] == "failed")
    if n_failed:
        print(f"\n本次有 {n_failed} 题失败并被排除，结果不完整。", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
