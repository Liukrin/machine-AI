"""答案级评测集的读取、证据解析、dev/test 划分与校验。

评测集（eval/answer_set.yaml）里的证据写的是手册原文引文，不是 chunk_id：
每次评测时把引文解析成「包含这段话的 chunk」，切分策略变了评测集也不会失效。
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
sys.path.insert(0, str(ROOT / "src" / "s3_eval"))

from textnorm import compile_pattern, loose, macro_numbers, norm, number_set  # noqa: E402

TYPES_ANSWER = ["fact", "procedure", "table", "cross_chunk", "colloquial", "multi_turn"]
TYPES_REFUSE = ["oos_adjacent", "oos_in_domain", "oos_unrelated"]
TYPE_CN = {
    "fact": "正文事实", "procedure": "操作步骤", "table": "表格查询", "cross_chunk": "跨片段",
    "colloquial": "口语化", "multi_turn": "多轮追问",
    "oos_adjacent": "相近领域（库外）", "oos_in_domain": "手册未写", "oos_unrelated": "无关问题",
}
SPLITS = ("dev", "test")


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- 语料
class Corpus:
    """chunks.jsonl 的只读视图：原文、归一化文本、数值集合。"""

    def __init__(self, chunks: list[dict]):
        from hybrid_retrieval import build_search_text  # 与线上喂给 LLM 的文本同一个函数

        self.chunks = {c["chunk_id"]: c for c in chunks}
        self.text = {cid: build_search_text(c) for cid, c in self.chunks.items()}
        self.norm = {cid: norm(t) for cid, t in self.text.items()}
        self.doc = {cid: c["doc_id"] for cid, c in self.chunks.items()}
        self.kind = {cid: c["chunk_type"] for cid, c in self.chunks.items()}
        self._numbers: dict[str, set[float]] = {}
        self._loose: dict[str, str] = {}

    @classmethod
    def load(cls, cfg: dict | None = None) -> "Corpus":
        cfg = cfg or load_config()
        path = ROOT / cfg["s2_vector"]["input_jsonl"]
        chunks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return cls(chunks)

    def fingerprint(self) -> str:
        h = hashlib.sha1()
        for cid in sorted(self.text):
            h.update(cid.encode("utf-8"))
            h.update(self.text[cid].encode("utf-8"))
        return h.hexdigest()[:16]

    def numbers(self, chunk_id: str) -> set[float]:
        if chunk_id not in self._numbers:
            self._numbers[chunk_id] = number_set(self.text[chunk_id])
        return self._numbers[chunk_id]

    def loose(self, chunk_id: str) -> str:
        """只留汉字、字母、数字的片段文本（核对摘抄、算字面重合度用）。"""
        if chunk_id not in self._loose:
            self._loose[chunk_id] = loose(self.text[chunk_id])
        return self._loose[chunk_id]

    def find(self, quote: str, docs: list[str] | None = None) -> list[str]:
        """归一化后包含这段引文的 chunk_id（可限定手册）。"""
        q = norm(quote)
        return sorted(cid for cid, t in self.norm.items()
                      if q in t and (not docs or self.doc[cid] in docs))


# --------------------------------------------------------------------------- 评测集
def _as_list(v) -> list:
    if v is None:
        return []
    return list(v) if isinstance(v, (list, tuple)) else [v]


def load_items(path: Path) -> list[dict]:
    """读取评测集并补齐派生字段（docs、编译后的事实模式）。结构错误在 validate 里报。"""
    items = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    for it in items:
        it["docs"] = [str(d) for d in _as_list(it.get("doc"))]
        it.setdefault("history", [])
        for f in it.get("facts") or []:
            f["any"] = _as_list(f.get("any"))
            f["numeric"] = bool(f.get("numeric"))
    return items


@lru_cache(maxsize=None)
def _compiled(pattern: str) -> re.Pattern:
    return compile_pattern(pattern)


def fact_matches(fact: dict, normalized_text: str) -> bool:
    return any(_compiled(p).search(normalized_text) for p in fact["any"])


def resolve_evidence(item: dict, corpus: Corpus) -> list[list[str]]:
    """每条 evidence 解析成一组等价的 chunk_id；答案需要每一组里至少一个片段。"""
    groups = []
    for ev in item.get("evidence") or []:
        docs = [str(d) for d in _as_list(ev.get("doc"))] or item["docs"]
        quotes = _as_list(ev.get("any_quote")) or _as_list(ev.get("quote"))
        ids: set[str] = set()
        for q in quotes:
            ids.update(corpus.find(q, docs))
        groups.append(sorted(ids))
    return groups


# --------------------------------------------------------------------------- 划分
def load_split(path: Path) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def assign_split(items: list[dict], existing: dict[str, str]) -> dict[str, str]:
    """给尚未划分的题目分配 dev/test：按题型分层，层内先按手册、再按 id 的哈希排序后交替分配。

    这样每种题型、每份手册在两边都大致各占一半，且分配结果只取决于 id，不取决于谁来分。
    已有的划分不动（否则加题会让旧题在两边来回跑，前后结果没法比）；
    新题接着该题型当前较少的一边开始交替。
    """
    split = dict(existing)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        by_type[it["type"]].append(it)
    for typ in sorted(by_type):
        counts = Counter(split[it["id"]] for it in by_type[typ] if it["id"] in split)
        todo = sorted((it for it in by_type[typ] if it["id"] not in split),
                      key=lambda it: ("+".join(it["docs"]),
                                      hashlib.sha1(it["id"].encode("utf-8")).hexdigest()))
        for it in todo:
            side = "dev" if counts["dev"] <= counts["test"] else "test"
            split[it["id"]] = side
            counts[side] += 1
    return dict(sorted(split.items()))


# --------------------------------------------------------------------------- 校验
def _grounded(fact: dict, evidence_ids: list[str], corpus: Corpus) -> tuple[bool, str]:
    """数值事实的数字必须能在证据片段里找到（至少一种写法的全部数字都在）。"""
    if fact.get("src"):
        src = norm(fact["src"])
        ok = any(src in corpus.norm[cid] for cid in evidence_ids)
        return ok, f"src「{fact['src']}」不在证据片段里"
    numbers: set[float] = set()
    evidence_norm = ""
    for cid in evidence_ids:
        numbers |= corpus.numbers(cid)
        evidence_norm += corpus.norm[cid] + "\n"
    tried = []
    for pattern in fact["any"]:
        wanted = macro_numbers(pattern)
        if not wanted:
            continue
        missing = []
        for w in wanted:
            if "/" in w:
                if w not in evidence_norm:
                    missing.append(w)
            elif round(float(w.replace(",", "")), 6) not in numbers:
                missing.append(w)
        if not missing:
            return True, ""
        tried.append(f"{pattern} 缺 {missing}")
    if not tried:
        return False, "数值事实的模式里没有 {n:…} 宏，无法核对数字是否出自原文"
    return False, "证据片段里找不到这些数字：" + "；".join(tried)


def validate(items: list[dict], corpus: Corpus, split: dict[str, str]) -> tuple[list[str], list[str]]:
    """返回 (错误, 警告)。有错误的评测集不允许用来评测。"""
    errors: list[str] = []
    warnings: list[str] = []
    seen: set[str] = set()
    doc_ids = set(corpus.doc.values())

    for it in items:
        iid = it.get("id", "<无 id>")
        err = lambda msg: errors.append(f"[{iid}] {msg}")     # noqa: E731
        warn = lambda msg: warnings.append(f"[{iid}] {msg}")  # noqa: E731

        if iid in seen:
            err("id 重复")
        seen.add(iid)
        typ, expect = it.get("type"), it.get("expect")
        if typ not in TYPES_ANSWER + TYPES_REFUSE:
            err(f"未知 type：{typ}")
            continue
        if expect != ("answer" if typ in TYPES_ANSWER else "refuse"):
            err(f"type={typ} 与 expect={expect} 不匹配")
        if not (it.get("question") or "").strip():
            err("缺 question")
        if iid not in split:
            err("未划分 dev/test（运行 validate.py --assign）")
        elif split[iid] not in SPLITS:
            err(f"划分值非法：{split[iid]}")

        if expect == "refuse":
            docs = [str(it["absent_in"])] if it.get("absent_in") else None
            for term in _as_list(it.get("absent")):
                hits = corpus.find(term, docs)
                if hits:
                    err(f"应拒答题声明语料中没有「{term}」，但出现在 {hits[:4]}")
            continue

        # ---- 应作答的题
        for d in it["docs"]:
            if d not in doc_ids:
                err(f"doc={d} 不在语料里")
        if not it["docs"]:
            err("缺 doc")
        if not (it.get("reference") or "").strip():
            err("缺 reference")
        facts = it.get("facts") or []
        if not facts:
            err("缺 facts")
        if typ == "multi_turn":
            if not it["history"] or not all(h.get("role") in ("user", "assistant") and h.get("content")
                                            for h in it["history"]):
                err("multi_turn 题需要 history（role/content）")
            if not (it.get("standalone") or "").strip():
                err("multi_turn 题需要 standalone 改写")

        groups = resolve_evidence(it, corpus)
        if not groups:
            err("缺 evidence")
        for ev, ids in zip(it.get("evidence") or [], groups):
            label = ev.get("quote") or ev.get("any_quote")
            if not ids:
                err(f"证据引文在语料（doc={it['docs']}）里找不到：{label}")
            elif len(ids) > 4:
                warn(f"证据引文命中 {len(ids)} 个片段，不够具体：{label}")
        evidence_ids = sorted({cid for ids in groups for cid in ids})

        if typ == "cross_chunk":
            if len(groups) < 2:
                err("cross_chunk 题至少要两条 evidence")
            elif set.intersection(*(set(g) for g in groups)):
                warn("各条 evidence 能被同一个片段同时满足，不算真正跨片段")
        if typ == "table" and evidence_ids and not any(corpus.kind[c] == "table" for c in evidence_ids):
            warn("type=table，但证据片段都不是表格")
        if typ in ("fact", "procedure") and evidence_ids and all(corpus.kind[c] == "table" for c in evidence_ids):
            warn(f"type={typ}，但证据片段全是表格")

        reference = norm(it.get("reference") or "")
        for f in facts:
            name = f.get("say") or f["any"]
            if not f["any"]:
                err(f"事实「{name}」缺 any 模式")
                continue
            try:
                for p in f["any"]:
                    _compiled(p)
            except re.error as exc:
                err(f"事实「{name}」的正则无法编译：{exc}")
                continue
            if not fact_matches(f, reference):
                err(f"参考答案不满足自己的事实「{name}」：{f['any']}")
            if f["numeric"] and evidence_ids:
                ok, why = _grounded(f, evidence_ids, corpus)
                if not ok:
                    err(f"数值事实「{name}」未落地：{why}")

    for extra in sorted(set(split) - seen):
        warnings.append(f"[{extra}] 划分文件里有这道题，评测集里没有")
    return errors, warnings


def distribution(items: list[dict], split: dict[str, str]) -> list[str]:
    """题型 × 划分、手册 × 划分的计数表（Markdown 行）。"""
    lines = ["| 题型 | dev | test | 合计 |", "|---|---|---|---|"]
    total = Counter()
    for typ in TYPES_ANSWER + TYPES_REFUSE:
        c = Counter(split.get(it["id"], "?") for it in items if it["type"] == typ)
        total.update(c)
        lines.append(f"| {TYPE_CN[typ]}（{typ}） | {c['dev']} | {c['test']} | {sum(c.values())} |")
    lines.append(f"| **合计** | {total['dev']} | {total['test']} | {sum(total.values())} |")
    lines += ["", "| 手册 | dev | test | 合计 |", "|---|---|---|---|"]
    by_doc: dict[str, Counter] = defaultdict(Counter)
    for it in items:
        key = "+".join(it["docs"]) if it["docs"] else "（应拒答，无手册）"
        by_doc[key][split.get(it["id"], "?")] += 1
    for key in sorted(by_doc):
        c = by_doc[key]
        lines.append(f"| {key} | {c['dev']} | {c['test']} | {sum(c.values())} |")
    return lines
