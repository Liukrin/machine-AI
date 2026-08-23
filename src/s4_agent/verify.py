"""S4 任务三：引用校验。从 src/agent_graph.py 的 verify_citation 适配而来。

bigram 匹配逻辑沿用，参考文本改为本轮检索到的 chunk 的 text + table_html（剥标签）。
新增：chunk_id 有效性（fabricated_citation）与无引用句检测。
"""
from __future__ import annotations

import html as _html
import re
from typing import List, Dict, Any

# 匹配 chunk_id 形如 1_c0030 / 4_c0178 / sample_cooler_manual_c0003
CHUNK_ID_RE = re.compile(r"[A-Za-z0-9_]+_c\d{4}")


def _strip_html(s: str) -> str:
    return _html.unescape(re.sub(r"<[^>]+>", "", s))


# 匹配「句末标点 + 紧接其后的一个或多个 [chunk_id] 标记」（如「…0.05mm。[2_c0016]」），
# 一个句末可有多个标记。只捕获到最后一个标记的 ]，不吞后面的空白/换行。
CITATION_AFTER_PUNCT_RE = re.compile(
    r"([。！？\n])(\s*\[[A-Za-z0-9_]+_c\d{4}\](?:\s*\[[A-Za-z0-9_]+_c\d{4}\])*)"
)


def _move_citations_before_punct(text: str) -> str:
    """把紧跟句末标点之后的 [chunk_id] 标记移到该标点之前。

    LLM 常把引用标在句号之后（…0.05mm。[2_c0016]），按「。！？」切句时引用被切到
    下一碎片，导致主句被误判为「无引用」。切句前先做一次搬移。
    """
    def _repl(m):
        return m.group(2).strip() + m.group(1)
    return CITATION_AFTER_PUNCT_RE.sub(_repl, text)


def verify_citation(llm_output: str, retrieved_chunks: List[dict],
                    extra_grounding: str = "") -> Dict[str, Any]:
    """校验 LLM 输出里每句是否落在检索到的内容中，并检查 chunk_id 引用有效性。

    Args:
        llm_output: LLM 原始输出
        retrieved_chunks: 本轮检索到的 chunk dict（含 chunk_id / text / table_html）
        extra_grounding: 额外合法依据（本项目暂传空）
    """
    empty = {
        "checked": False, "total_sentences": 0, "suspicious_count": 0,
        "suspicious": [], "cited_ids": [], "fabricated_ids": [], "uncited_sentences": [],
    }
    if not llm_output or not retrieved_chunks:
        return empty

    valid_ids = {c.get("chunk_id") for c in retrieved_chunks}

    # 参考文本：chunk 的 text + table_html（HTML 剥标签）
    parts = []
    for c in retrieved_chunks:
        if c.get("text"):
            parts.append(c["text"])
        if c.get("table_html"):
            parts.append(_strip_html(c["table_html"]))
    reference = " ".join(parts) + " " + extra_grounding

    # 预处理：把紧跟句末标点之后的 [chunk_id] 移到标点之前，避免切句把引用切到下一碎片
    text = _move_citations_before_punct(llm_output)

    # 1. chunk_id 有效性
    cited = CHUNK_ID_RE.findall(text)
    cited_ids = list(dict.fromkeys(cited))  # 去重保序
    fabricated_ids = [cid for cid in cited_ids if cid not in valid_ids]

    # 2. 句子切分
    sentences = [s.strip() for s in re.split(r"[。！？\n]", text) if len(s.strip()) > 5]

    # 3. 无引用句：>15 字且无任何 chunk_id 标注
    uncited_sentences = [s for s in sentences if len(s) > 15 and not CHUNK_ID_RE.search(s)]

    # 4. bigram 可疑句（沿用旧逻辑）
    # 注意：否定式拒答句（如「未提供」「未列出编号」）的 bigram 天然不在参考文本里，
    # 会被判为 suspicious。这是「描述缺失」方向的安全标注，不是缺陷，故保留现状、不改阈值。
    suspicious = []
    for sent in sentences:
        if len(sent) < 8:
            continue
        if sent.endswith("：") or sent.endswith(":"):
            continue
        if re.fullmatch(r"[\d\s\.\-\+]+", sent):
            continue
        chars = re.findall(r"[一-鿿]", sent)
        if len(chars) < 4:
            continue
        bigrams = set()
        for i in range(len(chars) - 1):
            bigrams.add(chars[i] + chars[i + 1])
        if not bigrams:
            continue
        found = sum(1 for bg in bigrams if bg in reference)
        if found / len(bigrams) < 0.15:
            suspicious.append(sent)

    return {
        "checked": True,
        "total_sentences": len(sentences),
        "suspicious_count": len(suspicious),
        "suspicious": suspicious,
        "cited_ids": cited_ids,
        "fabricated_ids": fabricated_ids,
        "uncited_sentences": uncited_sentences,
    }
