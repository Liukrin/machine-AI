"""确定性评分：系统行为（作答/拒答）、证据命中、关键事实、数值、引用。不调用 LLM。

数值是否答对、数字是否出自检索片段，全部由这里的代码判定（CLAUDE.md：涉及数值的判定不交给 LLM）。
"""
from __future__ import annotations

import re

from dataset import Corpus, _rounds_to, fact_matches
from textnorm import CHUNK_ID_RE, attach_trailing_citations, loose, norm, numbers_in, strip_citations

_SENT_SPLIT_RE = re.compile(r"[。！？\n]")      # 与 verify.verify_citation 的切句口径一致
_MARKDOWN_EDGE = "*#>-·• \t"                    # 句子两端的 Markdown 标记（**小标题：** 之类）
SAMPLE_PREFIX = "sample_"


# --------------------------------------------------------------------------- 行为
def is_refusal(answer: str, markers: list[str]) -> bool:
    """模型自述拒答：答案里有拒答话术，且没有引用任何片段。

    带了引用说明模型给出了依据手册的实质内容（哪怕同时说了「手册未给出具体数值」），不算拒答。
    """
    text = "".join(strip_citations(answer).split())
    return any(m in text for m in markers) and not CHUNK_ID_RE.search(answer)


def behavior(record: dict, markers: list[str]) -> str:
    if record["status"] in ("rejected_gate", "error"):
        return record["status"]
    if not record["answer"] or is_refusal(record["answer"], markers):
        return "refused_model"
    return "answered"


# --------------------------------------------------------------------------- 句子
def sentences(answer: str, scfg: dict) -> list[dict]:
    """把答案切成实质性句子：[{text 去掉引用后的句子, cited 该句引用的 chunk_id}]。

    切句与「实质性」的判定沿用引用对照实验（s4_halluc_ab 段）的参数：
    剔除过短片段、以冒号结尾的过渡句、纯过渡话术。有两处比 verify.py 更宽：
    写在句号后面的引用不带方括号也算这句话的引用；先去掉句子两端的 Markdown 标记再判断。
    """
    out = []
    for raw in _SENT_SPLIT_RE.split(attach_trailing_citations(answer)):
        s = raw.strip().strip(_MARKDOWN_EDGE)
        if len(s) < int(scfg["min_sentence_chars"]):
            continue
        if any(s.endswith(m) for m in scfg["transition_end_markers"]):
            continue
        if len(s) <= int(scfg["transition_max_chars"]) and any(m in s for m in scfg["transition_markers"]):
            continue
        text = strip_citations(s).strip(" -*·•|")
        if len(text) < 2:
            continue
        out.append({"text": text, "cited": list(dict.fromkeys(CHUNK_ID_RE.findall(s)))})
    return out


def overlap(sentence: str, chunk_ids: list[str], corpus: Corpus, n: int = 4) -> float | None:
    """句子与这些片段的字面重合度：句子里的 n 字片段有多大比例原样出现在片段中（忽略标点空白）。

    接近 1 说明这句话基本是照抄原文；偏低说明是改写或归纳，评审判断更容易出错，值得人工看一眼。
    """
    t = loose(sentence)
    grams = {t[i:i + n] for i in range(len(t) - n + 1)}
    if not grams:
        return None
    context = " ".join(corpus.loose(c) for c in chunk_ids if c in corpus.chunks)
    return round(sum(g in context for g in grams) / len(grams), 3)


# --------------------------------------------------------------------------- 单题评分
def score(item: dict, record: dict, groups: list[list[str]], corpus: Corpus,
          acfg: dict, scfg: dict) -> dict:
    """一道题的确定性评分。groups 是证据解析出的 chunk_id 分组（应拒答的题为空）。"""
    beh = behavior(record, list(acfg["refusal_markers"]))
    retrieved = [r["chunk_id"] for r in record["retrieved"]]
    answer = record["answer"] if beh == "answered" else ""
    cited = list(dict.fromkeys(CHUNK_ID_RE.findall(answer)))
    sents = sentences(answer, scfg) if answer else []

    s: dict = {
        "id": item["id"], "variant": record.get("variant", "main"), "type": item["type"],
        "expect": item["expect"], "docs": item["docs"], "behavior": beh,
        "refused": beh in ("rejected_gate", "refused_model"),
        "gate_distance": record.get("gate_distance"),
        "retrieved": retrieved,
        "retrieved_sample": [c for c in retrieved if c.startswith(SAMPLE_PREFIX)],
        "cited": cited,
        "cited_docs": sorted({corpus.doc[c] for c in cited if c in corpus.doc}),
        "cited_fabricated": [c for c in cited if c not in retrieved],
        "cited_sample": [c for c in cited if c.startswith(SAMPLE_PREFIX)],
        "answer": answer,
        "sentence_texts": [x["text"] for x in sents],
        "sentence_cited": [x["cited"] for x in sents],
        "sentence_overlap": [overlap(x["text"], retrieved, corpus) for x in sents],
        "n_sentences": len(sents),
        "n_uncited": sum(1 for x in sents if not x["cited"]),
        "finish_reason": record.get("finish_reason"),
        "t_total_ms": record.get("t_total_ms"), "t_first_token_ms": record.get("t_first_token_ms"),
        "input_tokens": record.get("input_tokens"), "output_tokens": record.get("output_tokens"),
    }

    # Agent 的工具调用（rag 模式和旧缓存没有 steps）
    steps = record.get("steps") or []
    # 换算、核对工具的输出：原文（给评审、做数值溯源）+ 输入数值与限值出处（核对引用是否标对了片段）
    calc_outputs = []
    for st in steps:
        if not (st.get("ok") and st.get("name") in ("convert_unit", "check_value") and st.get("output")):
            continue
        a = st.get("args") or {}
        if st["name"] == "convert_unit":
            calc_outputs.append({"text": st["output"], "inputs": [a.get("value")], "source": None})
        else:
            calc_outputs.append({"text": st["output"], "source": a.get("source_chunk_id"),
                                 "inputs": [x for x in (a.get("limit_min"), a.get("limit_max")) if x is not None]})
    used = {st["name"] for st in steps}
    s.update({
        "mode": record.get("mode") or "rag",
        "llm_calls": record.get("llm_calls"),
        "tool_calls": sum(1 for st in steps if not st.get("auto")),      # 不含系统自动做的预检索
        "tools_used": sorted({st["name"] for st in steps if not st.get("auto")}),
        "tool_errors": sum(1 for st in steps if not st.get("ok")),
        "forced_final": bool(record.get("forced_final")),
        "cost_yuan": record.get("cost_yuan"),
        "cache_read_tokens": record.get("cache_read_tokens"),
        "calc_outputs": calc_outputs,
        "tools_expected": item.get("tools") or [],
        "tools_expected_ok": all(t in used for t in item.get("tools") or []),
    })

    # 数值溯源：答案里的每个数，是否出现在本次检索到的片段、问句或换算/核对工具的输出里
    if answer:
        allowed: set[float] = {round(x, 6) for x in numbers_in(record["question"])}
        # 工具输出里的数（含精确值）；答案常把精确值再舍入一位（3.333 → 3.33），带小数的数按「是某个精确值的正确舍入」算可溯源
        calc_numbers = {round(x, 6) for c in calc_outputs for x in numbers_in(c["text"])}
        allowed |= calc_numbers
        cited_allowed = set(allowed)
        for cid in retrieved:
            allowed |= corpus.numbers(cid)
        for cid in cited:
            if cid in corpus.chunks:
                cited_allowed |= corpus.numbers(cid)

        def traced(x: float, pool: set[float]) -> bool:
            return x in pool or (x != int(x) and any(_rounds_to(repr(x), v) for v in calc_numbers))

        nums = [round(x, 6) for x in numbers_in(answer, answer=True)]
        s["n_numbers"] = len(nums)
        s["untraced_numbers"] = sorted({x for x in nums if not traced(x, allowed)})
        s["n_untraced"] = sum(1 for x in nums if not traced(x, allowed))
        s["n_untraced_in_cited"] = sum(1 for x in nums if not traced(x, cited_allowed))
    else:
        s.update(n_numbers=0, untraced_numbers=[], n_untraced=0, n_untraced_in_cited=0)

    if item["expect"] != "answer":
        return s

    # ---- 应作答的题
    gold = {cid for g in groups for cid in g}
    hits = [any(cid in retrieved for cid in g) for g in groups]
    s["evidence_groups"] = len(groups)
    s["evidence_hit"] = sum(hits)
    s["evidence_all"] = all(hits) if hits else False
    s["cited_gold"] = [c for c in cited if c in gold]
    # 串手册：作答了，但引用的片段没有一个出自题目问的那份手册
    s["wrong_manual"] = bool(cited) and not any(d in item["docs"] for d in s["cited_docs"])

    text = norm(strip_citations(answer))
    facts = []
    for f in item["facts"]:
        facts.append({"say": f.get("say") or f["any"][0], "numeric": f["numeric"],
                      "matched": bool(answer) and fact_matches(f, text)})
    s["facts"] = facts
    s["n_facts"] = len(facts)
    s["n_facts_matched"] = sum(f["matched"] for f in facts)
    s["n_numeric"] = sum(f["numeric"] for f in facts)
    s["n_numeric_matched"] = sum(f["matched"] for f in facts if f["numeric"])
    s["all_facts"] = s["n_facts_matched"] == s["n_facts"]
    return s
