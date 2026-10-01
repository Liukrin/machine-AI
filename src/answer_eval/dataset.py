"""答案级评测集的读取、证据解析、dev/test 划分与校验。

评测集（eval/answer_set.yaml）里的证据写的是手册原文引文，不是 chunk_id：
每次评测时把引文解析成「包含这段话的 chunk」，切分策略变了评测集也不会失效。
"""
from __future__ import annotations

import ast
import hashlib
import json
import operator
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

TYPES_ANSWER = ["fact", "procedure", "table", "cross_chunk", "colloquial", "multi_turn",
                # 以下题型只出现在 Agent 多步任务集（eval/agent_tasks.yaml）
                "convert", "check", "table_lookup", "multi_step"]
TYPES_REFUSE = ["oos_adjacent", "oos_in_domain", "oos_unrelated"]
TYPE_CN = {
    "fact": "正文事实", "procedure": "操作步骤", "table": "表格查询", "cross_chunk": "跨片段",
    "colloquial": "口语化", "multi_turn": "多轮追问",
    "convert": "单位换算", "check": "限值核对", "table_lookup": "表格精确查询", "multi_step": "多步/跨手册",
    "oos_adjacent": "相近领域（库外）", "oos_in_domain": "手册未写", "oos_unrelated": "无关问题",
}
SPLITS = ("dev", "test")
# 评测集里 tools 字段可写的工具名（Agent 的工具，见 src/s4_agent/tools.py）
KNOWN_TOOLS = ("search_manuals", "lookup_table", "read_section", "convert_unit", "check_value")


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def eval_set(acfg: dict, name: str) -> dict:
    """评测集的路径配置。main 是 answer_eval 段顶层的 items / split；其余写在 answer_eval.sets 下，
    结果放在评测目录的同名子目录里（主评测集直接放在评测目录下，与以前一致）。"""
    if name == "main":
        return {"name": "main", "items": acfg["items"], "split": acfg["split"], "subdir": "",
                "title": "答案级评测集"}
    sets = acfg.get("sets") or {}
    if name not in sets:
        raise SystemExit(f"config answer_eval.sets 里没有评测集「{name}」，可选：main、{'、'.join(sets)}")
    s = sets[name]
    return {"name": name, "items": s["items"], "split": s["split"], "subdir": name, "title": s.get("title") or name}


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

    def numbers_lenient(self, chunk_id: str) -> set[float]:
        """宽松的数值集合：在 numbers() 之外，把 MinerU 拆开的小数（「2. 5m/s」）也按合并后的读法收进来。"""
        key = "~" + chunk_id
        if key not in self._numbers:
            merged = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", self.text[chunk_id])
            self._numbers[key] = self.numbers(chunk_id) | number_set(merged)
        return self._numbers[key]

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
        it["tools"] = [str(t) for t in _as_list(it.get("tools"))]
        for f in it.get("facts") or []:
            f["any"] = _as_list(f.get("any"))
            f["numeric"] = bool(f.get("numeric"))
            f["calc"] = [str(x) for x in _as_list(f.get("calc"))]
    return items


# --------------------------------------------------------------------------- 推算数值
_CALC_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.Div: operator.truediv, ast.Pow: operator.pow}


def calc_value(expr: str) -> float:
    """计算评测集里 calc 字段的算式（只允许数字与 + - * / ** 和括号）。"""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in _CALC_OPS:
            return _CALC_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        raise ValueError(f"算式里有不允许的内容：{ast.dump(node)[:60]}")
    return ev(ast.parse(expr, mode="eval"))


def _rounds_to(written: str, value: float) -> bool:
    """written（如 3.33）是不是 value 按其小数位数四舍五入后的写法。"""
    w = written.replace(",", "")
    decimals = len(w.split(".")[1]) if "." in w else 0
    return abs(float(w) - value) <= 0.5 * 10 ** -decimals + 1e-9


def _check_calc(fact: dict) -> list[str]:
    """推算数值事实（换算、差值）：模式里每个 {n:…} 都必须是某个 calc 算式结果的正确舍入。

    这类数字不在手册原文里，没法做「数字出自证据」的检查，改为用独立写出的算式核对，
    算式与 Agent 的换算工具互不依赖（换算系数在评测集里另写一遍）。
    """
    errs = []
    try:
        values = [calc_value(e) for e in fact["calc"]]
    except (SyntaxError, ValueError, ZeroDivisionError) as exc:
        return [f"calc 算式无法计算：{exc}"]
    for pattern in fact["any"]:
        for w in macro_numbers(pattern):
            if "/" in w:
                continue
            if not any(_rounds_to(w, v) for v in values):
                errs.append(f"{pattern} 里的 {w} 不是 calc 结果 {[round(v, 6) for v in values]} 的正确舍入")
    if not any(macro_numbers(p) for p in fact["any"]):
        errs.append("推算数值事实的模式里没有 {n:…} 宏")
    return errs


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
        unknown_tools = [t for t in it["tools"] if t not in KNOWN_TOOLS]
        if unknown_tools:
            err(f"tools 里有未知的工具名：{unknown_tools}（可选 {', '.join(KNOWN_TOOLS)}）")
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
            if f["calc"]:
                if not f["numeric"]:
                    err(f"事实「{name}」写了 calc，应同时标 numeric: true")
                for why in _check_calc(f):
                    err(f"推算数值事实「{name}」：{why}")
            elif f["numeric"] and evidence_ids:
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
