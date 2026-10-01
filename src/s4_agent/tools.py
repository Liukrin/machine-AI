"""Agent 的五个工具：检索手册、按行查表、读相邻片段、单位换算、限值核对。都是确定性代码，不调用 LLM。

每个工具对应阶段 1 基线里的一类失分（见 docs/answer_eval.md）：
  search_manuals  混合检索（与 rag 模式同一套 BM25 + 向量 RRF），可按手册、正文/表格过滤。
                  证据不够时由模型换个说法或缩小范围再搜，而不是一个距离阈值直接拒答。
  lookup_table    表格按行精确查询（tables.py），解决零件编号、故障表、扭矩表这类「关键词只有几个字」的题。
  read_section    读某个片段前后相邻的片段，解决答案被切到两个片段里的问题。
  convert_unit    单位换算（units.py），模型不心算。
  check_value     实测值与手册限值的比较。结论由代码给出（CLAUDE.md：物理阈值判决由确定性 Python 执行）；
                  限值必须出自本次对话里检索到的某个片段，且限值数字要能在该片段原文里找到。

工具出错（参数不对、片段不存在、单位不认识）时返回 ok=False 的结果，错误信息原样作为观察结果回给模型，
由模型修正参数后重试；不抛异常、不中断回答。

参数定义（发给模型的 JSON Schema）在 tool_schema.py；执行器 run_tool（超时、出错转成结果）在本文件末尾，
Agent（agent.py）与 MCP Server（src/s5_app/mcp_server.py）共用。
"""
from __future__ import annotations

import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in ("src", "src/s3_eval", "src/s4_agent"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

import numpy as np  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from hybrid_retrieval import _load_ctx, build_search_text, tokenize  # noqa: E402
from numcheck import Quantity, Source, ground, unit_id  # noqa: E402
from tables import TableStore, table_rows  # noqa: E402
from tool_schema import TOOL_ARGS, TOOL_LABEL, tool_specs  # noqa: E402,F401  agent.py 也从这里导入
from units import UnitError, compare, convert, fmt, numbers_in_text, same_unit, unit_display  # noqa: E402

PREVIEW_CHARS = 100        # 前端来源卡片的预览长度
TABLE_MAX_CHARS = 2000     # 检索结果里单个表格片段最多给模型看多少字，超出截断并提示改用 lookup_table
SMALL_TABLE_ROWS = 6       # 查表命中小表（不超过这么多行）时整表返回：竖排的小表单看一行没有意义
SOURCE_TAG = "片段原文"     # 注入防护：手册原文包在 <片段原文 id="…"> … </片段原文> 里，提示词说明标签内是资料不是指令


# --------------------------------------------------------------------------- 结果
@dataclass
class ToolResult:
    ok: bool
    content: str                                        # 回给模型的观察结果（ToolMessage 的内容）
    summary: str                                        # 前端时间线上的一句话
    chunks: list[dict] = field(default_factory=list)    # 本次新给模型看的片段 [{chunk_id, distance}]，进入证据池
    data: dict = field(default_factory=dict)            # 结构化结果（换算、核对），评测做数值溯源用


def _error(msg: str) -> ToolResult:
    return ToolResult(ok=False, content=f"工具调用失败：{msg}", summary=msg)


# --------------------------------------------------------------------------- 语料视图
class Corpus:
    """chunks.jsonl 的只读视图：按手册排序的片段、展示信息、表格行库。进程内单例。"""

    def __init__(self, documents: dict):
        ctx = _load_ctx()
        self.ctx = ctx
        self.by_id: dict[str, dict] = ctx["chunk_by_id"]
        self.order: dict[str, list[str]] = {}
        for cid, c in self.by_id.items():          # chunks.jsonl 按手册、按顺序写入，字典保持该顺序
            self.order.setdefault(c["doc_id"], []).append(cid)
        self.pos = {cid: i for ids in self.order.values() for i, cid in enumerate(ids)}
        self.documents = documents
        self.tables = TableStore(list(self.by_id.values()))
        self._aliases = self._build_aliases()

    def _build_aliases(self) -> dict[str, str]:
        out = {}
        for doc_id, d in self.documents.items():
            for name in [doc_id, d.get("short"), d.get("title"), *(d.get("aliases") or [])]:
                if name:
                    out[re.sub(r"\s+", "", str(name)).lower()] = doc_id
        return out

    def resolve_manual(self, manual: str | None) -> str | None:
        """手册编号或名称 → doc_id。认不出就抛 ValueError（错误信息回给模型）。"""
        if manual is None or not str(manual).strip():
            return None
        key = re.sub(r"\s+", "", str(manual)).lower()
        if key in self._aliases:
            return self._aliases[key]
        hits = {d for name, d in self._aliases.items() if key in name or name in key}
        if len(hits) == 1:
            return hits.pop()
        raise ValueError(f"认不出手册「{manual}」，manual 请填手册编号：{', '.join(self.documents)}")

    def doc_short(self, doc_id: str) -> str:
        d = self.documents.get(doc_id) or {}
        return d.get("short") or d.get("title") or doc_id

    def body(self, c: dict) -> str:
        """片段正文：正文去掉开头的章节前缀；表格按行展开（比拍平的单元格更好读）。"""
        if c["chunk_type"] == "table":
            rows = table_rows(c.get("table_html") or "")
            return "\n".join(" | ".join(r) for r in rows) if rows else build_search_text(c)
        text = c.get("text") or ""
        hp = c.get("heading_path")
        prefix = f"{hp}\n\n" if hp else ""
        return text[len(prefix):] if prefix and text.startswith(prefix) else text

    def header(self, c: dict) -> str:
        """片段的一行说明：手册 · 章节 · 页码 · 正文/表格。"""
        hp = c.get("heading_path") or ""
        section = hp.split(" > ")[-1] if hp else "（无章节标题）"
        pr = c.get("page_range") or []
        page = f"第 {pr[0] + 1} 页" if len(pr) == 2 and pr[0] is not None else ""
        kind = "表格" if c["chunk_type"] == "table" else "正文"
        return " · ".join(x for x in (self.doc_short(c["doc_id"]), section, page, kind) if x)

    def preview(self, c: dict) -> str:
        text = " ".join(self.body(c).split())
        return text[:PREVIEW_CHARS] + ("…" if len(text) > PREVIEW_CHARS else "")


@lru_cache(maxsize=1)
def get_corpus() -> Corpus:
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    return Corpus((cfg.get("s5_app") or {}).get("documents") or {})


@lru_cache(maxsize=2048)
def chunk_source(cid: str) -> Source:
    """一个片段作为核对数值用的资料：说明行 + 正文（表格按行展开，与给模型看的写法相同）。"""
    corpus = get_corpus()
    c = corpus.by_id[cid]
    return Source(f"{corpus.header(c)}\n{corpus.body(c)}")


# --------------------------------------------------------------------------- 检索
def hybrid_search(query: str, top_k: int, doc_id: str | None = None,
                  kind: str | None = None) -> tuple[list[str], dict[str, float], float | None]:
    """混合检索（与 hybrid_retrieval.hybrid_retrieve 同一套做法：两路各取 top-N，RRF 融合），可按手册/类型过滤。

    不加过滤时结果与 hybrid_retrieve 完全一致（rag 模式的检索）。返回 (chunk_id 列表, 各自的向量距离, 过滤范围内向量 top-1 距离)。
    """
    ctx = _load_ctx()
    emb = ctx["model"].encode([query], normalize_embeddings=ctx["normalize"], show_progress_bar=False)[0]
    conds = [{"doc_id": doc_id}] if doc_id else []
    if kind:
        conds.append({"chunk_type": kind})
    where = None if not conds else (conds[0] if len(conds) == 1 else {"$and": conds})
    vres = ctx["collection"].query(query_embeddings=[emb.tolist()], n_results=ctx["per_top"],
                                   where=where, include=["distances"])
    v_ids, v_dist = vres["ids"][0], vres["distances"][0]
    scores = ctx["bm25"].get_scores(tokenize(query))
    by_id = ctx["chunk_by_id"]

    def allowed(cid: str) -> bool:
        c = by_id[cid]
        return (not doc_id or c["doc_id"] == doc_id) and (not kind or c["chunk_type"] == kind)

    b_ids = [cid for cid in (ctx["bm25_ids"][i] for i in np.argsort(scores)[::-1]) if allowed(cid)][:ctx["per_top"]]
    fused: dict[str, float] = {}
    for rank, cid in enumerate(v_ids, 1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (ctx["rrf_k"] + rank)
    for rank, cid in enumerate(b_ids, 1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (ctx["rrf_k"] + rank)
    ranked = [cid for cid, _ in sorted(fused.items(), key=lambda x: -x[1])[:top_k]]
    dist = dict(zip(v_ids, v_dist))
    missing = [cid for cid in ranked if cid not in dist]
    if missing:
        got = ctx["collection"].get(ids=missing, include=["embeddings"])
        for cid, e in zip(got["ids"], got["embeddings"]):
            dist[cid] = 1.0 - float(np.dot(emb, e))
    return ranked, {cid: round(float(dist[cid]), 6) for cid in ranked}, (float(v_dist[0]) if v_dist else None)


# --------------------------------------------------------------------------- 工具箱
_SHOWN_RANK = {"rows": 1, "head": 2, "full": 3}


class Toolbox:
    """一次回答里的工具执行环境。

    seen 是本次对话已经给模型看过的片段（证据池），跨轮次累积：{chunk_id: {"distance": …, "shown": …}}。
    shown 记给模型看了多少：full 全文；head 长表格只给了开头（截断）；rows 只给了查表命中的几行。
    片段再次出现时，full 只给一行「前面已给出全文」，head 只给一行提示，rows 照常给正文。
    abbreviate=False 时不做这种省略，重复出现的片段也给出正文：MCP 客户端自己管理上下文，
    之前给过的全文可能已被压缩掉（见 src/s5_app/mcp_server.py）。

    context 是用户的问题（Agent 模式下含对话历史里的提问；MCP 拿不到，为 None），calc 是之前换算、核对工具的输出。
    换算、核对的参数要对得上这些资料（numcheck.ground）：数值出现在原文里时，单位必须与原文一致；
    Agent 模式下被换算值、被核对值还必须出自问题、片段或之前的换算结果（MCP 无从核对，只查单位）。
    """

    def __init__(self, cfg: dict, tau: float, seen: dict[str, dict] | None = None, abbreviate: bool = True,
                 context: str | None = None, calc: list[str] | None = None):
        self.cfg = cfg
        self.tau = tau
        self.abbreviate = abbreviate
        self.guard = bool(cfg.get("injection_guard", True))
        self.context = context
        self.calc: list[str] = list(calc or [])
        self.corpus = get_corpus()
        self.seen: dict[str, dict] = {cid: dict(v) for cid, v in (seen or {}).items()}

    def fork(self) -> "Toolbox":
        """副本：工具在副本上执行，按时完成才并回（run_tool），超时还在跑的线程改不到证据池。"""
        return Toolbox(self.cfg, self.tau, self.seen, self.abbreviate, self.context, self.calc)

    def merge(self, other: "Toolbox") -> None:
        self.seen.update(other.seen)
        self.calc = list(other.calc)

    # ---- 参数核对：数值、单位对得上资料（规则见 numcheck.py）
    def _sources(self, chunk_ids: list[str] | None = None) -> list[Source]:
        out = [Source(self.context)] if self.context else []
        out += [chunk_source(cid) for cid in (self.seen if chunk_ids is None else chunk_ids) if cid in self.corpus.by_id]
        return out + [Source(t, kind="calc") for t in self.calc]

    def _grounding_error(self, what: str, value: float, unit: str, chunk_ids: list[str] | None = None,
                         require_found: bool = False) -> str | None:
        uid = unit_id(unit)
        if uid is None:                     # 不认识的单位：交给换算本身报错
            return None
        f = ground(Quantity(float(value), fmt(value), unit, uid), self._sources(chunk_ids))
        if f.status == "unit_mismatch":
            return (f"{what} {fmt(value)} 在原文里的单位是 {'、'.join(f.source_units)}，不是「{unit}」。单位请照抄原文"
                    "（原文写的单位换算工具不认识时，说明原文单位后作答，不要换成别的单位），请核对后重试")
        if f.status == "no_source" and require_found:
            return (f"{what} {fmt(value)} 在问题、本次检索到的片段和之前的换算结果里都找不到。"
                    "数值必须照抄这些地方写明的数，不要自己估算或心算换算，请核对后重试")
        return None

    def _wrap(self, cid: str, body: str) -> str:
        """注入防护：原文包进标签；原文里伪造的同名标签改成全角，起不了作用。"""
        if not self.guard:
            return body
        body = body.replace(f"<{SOURCE_TAG}", f"＜{SOURCE_TAG}").replace(f"</{SOURCE_TAG}", f"＜/{SOURCE_TAG}")
        return f'<{SOURCE_TAG} id="{cid}">\n{body}\n</{SOURCE_TAG}>'

    def _mark(self, cid: str, distance: float | None, shown: str, new: list[dict]) -> None:
        """记进证据池（新片段同时记进 new）；shown 只升不降：看过全文的片段，之后再查到几行也还算看过全文。"""
        old = self.seen.get(cid)
        if old is None:
            self.seen[cid] = {"distance": distance, "shown": shown}
            new.append({"chunk_id": cid, "distance": distance})
        elif _SHOWN_RANK[shown] > _SHOWN_RANK[old.get("shown", "full")]:
            self.seen[cid] = {**old, "shown": shown}

    # ---- 片段展示：已给过的只给一行，省 token
    def _show(self, cid: str, distance: float | None, new: list[dict]) -> str:
        c = self.corpus.by_id[cid]
        dist = "" if distance is None else f" · 距离 {distance:.3f}"
        shown = (self.seen.get(cid) or {}).get("shown") if self.abbreviate else None
        if shown == "full":
            return f"[{cid}] {self.corpus.header(c)}{dist}（前面已给出全文）"
        if shown == "head":
            return f"[{cid}] {self.corpus.header(c)}{dist}（前面已给出开头部分；表格较长，要查具体行请用 lookup_table）"
        body = self.corpus.body(c)
        part = "full"
        if c["chunk_type"] == "table" and len(body) > TABLE_MAX_CHARS:
            body = body[:TABLE_MAX_CHARS] + "\n……（表格较长，已截断；要查具体行请用 lookup_table）"
            part = "head"
        self._mark(cid, distance, part, new)
        return f"[{cid}] {self.corpus.header(c)}{dist}\n{self._wrap(cid, body)}"

    def search_manuals(self, query: str, manual: str | None = None, kind: str | None = None) -> ToolResult:
        doc_id = self.corpus.resolve_manual(manual)
        ids, dist, top1 = hybrid_search(query, int(self.cfg["search_top_k"]), doc_id, kind)
        scope = "、".join(x for x in (f"手册 {doc_id}" if doc_id else "", {"text": "正文", "table": "表格"}.get(kind or "", ""))
                          if x) or "全部手册"
        if not ids:
            return ToolResult(True, f"检索「{query}」（{scope}）：没有结果。", f"「{query}」无结果")
        low = top1 is not None and top1 > self.tau
        conf = (f"最相近片段的向量距离 {top1:.3f}，高于阈值 {self.tau}：结果可能与问题无关，请核实或换个说法再查"
                if low else f"最相近片段的向量距离 {top1:.3f}（低于阈值 {self.tau}）")
        new: list[dict] = []
        parts = [self._show(cid, dist[cid], new) for cid in ids]
        content = f"检索「{query}」（{scope}）返回 {len(ids)} 个片段；{conf}。\n\n" + "\n\n".join(parts)
        summary = f"{len(ids)} 个片段（新 {len(new)}）" + ("，相关度低" if low else "")
        return ToolResult(True, content, summary, chunks=new,
                          data={"top1_distance": top1, "low_confidence": low, "scope": scope})

    def lookup_table(self, keywords: str, manual: str | None = None) -> ToolResult:
        doc_id = self.corpus.resolve_manual(manual)
        kws = [k for k in re.split(r"[\s,，、;；]+", keywords) if k]
        if not kws:
            return _error("keywords 不能为空")
        limit = int(self.cfg["lookup_max_rows"])
        hits, total = self.corpus.tables.lookup(kws, doc_id, limit=10_000)
        scope = f"手册 {doc_id}" if doc_id else "全部手册"
        if not hits:
            return ToolResult(True, f"在{scope}的表格里没有同时包含「{' '.join(kws)}」的行。可以减少关键词、换个说法，"
                                    "或用 search_manuals 检索。", f"「{' '.join(kws)}」无命中")
        # 同一行内容在多张表里重复出现（零件图例常重复），只保留第一次
        groups: dict[str, list[dict]] = {}
        seen_rows: set[str] = set()
        dup = 0
        for h in hits:
            key = " | ".join(h["cells"])
            if key in seen_rows:
                dup += 1
                continue
            seen_rows.add(key)
            groups.setdefault(h["chunk_id"], []).append(h)
        new: list[dict] = []
        parts: list[str] = []
        shown = 0                                   # 已列出的命中行数
        for cid, rows in groups.items():
            if shown >= limit:
                break
            c = self.corpus.by_id[cid]
            n_rows = self.corpus.tables.n_rows_of(cid)
            self._mark(cid, None, "full" if n_rows <= SMALL_TABLE_ROWS else "rows", new)
            lines: list[str] = []
            if n_rows <= SMALL_TABLE_ROWS:          # 小表整表给出
                lines += [f"  {' | '.join(r)}" for r in table_rows(c["table_html"])]
                shown += len(rows)
            else:
                if rows[0]["row_idx"] != 0:
                    lines.append(f"  表头：{rows[0]['header']}")
                take = rows[: limit - shown]
                lines += [f"  第 {h['row_idx'] + 1} 行：{' | '.join(h['cells'])}" for h in take]
                shown += len(take)
            parts.append(f"[{cid}] {self.corpus.header(c)}（共 {n_rows} 行）\n" + self._wrap(cid, "\n".join(lines)))
        more = sum(len(r) for r in groups.values()) - shown
        tail = []
        if more > 0:
            tail.append(f"另有 {more} 行命中未显示，可加关键词或指定手册缩小范围")
        if dup:
            tail.append(f"另有 {dup} 行与上面内容完全相同（出现在其他表里），未重复列出")
        content = (f"在{scope}的表格里查「{' '.join(kws)}」：\n\n" + "\n\n".join(parts)
                   + ("\n\n（" + "；".join(tail) + "）" if tail else ""))
        return ToolResult(True, content, f"{len(parts)} 张表，{shown} 行", chunks=new,
                          data={"tables": [p.split("]")[0][1:] for p in parts]})

    def read_section(self, chunk_id: str, before: int = 1, after: int = 1) -> ToolResult:
        if chunk_id not in self.corpus.by_id:
            return _error(f"没有 chunk_id 为「{chunk_id}」的片段")
        if chunk_id not in self.seen:
            return _error(f"片段 {chunk_id} 还没有在本次对话中出现过，请先检索")
        c = self.corpus.by_id[chunk_id]
        ids = self.corpus.order[c["doc_id"]]
        i = self.corpus.pos[chunk_id]
        window = ids[max(0, i - before): i + after + 1]
        new: list[dict] = []
        parts = [self._show(cid, None, new) for cid in window]
        content = f"{chunk_id} 前后相邻的片段（按手册顺序）：\n\n" + "\n\n".join(parts)
        return ToolResult(True, content, f"{len(window)} 个片段（新 {len(new)}）", chunks=new)

    def convert_unit(self, value: float, from_unit: str, to_unit: str, is_difference: bool = False) -> ToolResult:
        try:
            out = convert(value, from_unit, to_unit, is_difference=is_difference)
        except UnitError as exc:
            return _error(str(exc))
        err = self._grounding_error("被换算的数值", value, from_unit, require_found=self.context is not None)
        if err:
            return _error(err)
        a = f"{fmt(value)} {unit_display(from_unit)}"
        b = f"{fmt(out)} {unit_display(to_unit)}"
        kind = "（温差）" if is_difference else ""
        content = f"换算结果{kind}：{a} = {b}（精确值 {out:.6g} {unit_display(to_unit)}）"
        self.calc.append(content)
        return ToolResult(True, content, f"{a} = {b}",
                          data={"value": value, "from_unit": from_unit, "to_unit": to_unit, "result": out,
                                "display": b, "is_difference": is_difference})

    def check_value(self, value: float, unit: str, source_chunk_id: str, limit_max: float | None = None,
                    limit_min: float | None = None, limit_unit: str | None = None) -> ToolResult:
        if source_chunk_id not in self.corpus.by_id:
            return _error(f"没有 chunk_id 为「{source_chunk_id}」的片段")
        if source_chunk_id not in self.seen:
            return _error(f"片段 {source_chunk_id} 还没有在本次对话中出现过：限值必须出自检索到的片段，请先检索")
        nums = numbers_in_text(build_search_text(self.corpus.by_id[source_chunk_id]))
        missing = [x for x in (limit_min, limit_max) if x is not None and round(float(x), 6) not in nums]
        if missing:
            return _error(f"限值 {', '.join(fmt(x) for x in missing)} 在片段 {source_chunk_id} 的原文里找不到。"
                          "限值必须照抄原文写明的数值（单位不同时填原文的单位，由工具换算），请核对后重试")
        try:
            r = compare(value, unit, limit_min, limit_max, limit_unit)
        except UnitError as exc:
            return _error(str(exc))
        for lim in (x for x in (limit_min, limit_max) if x is not None):
            err = self._grounding_error("限值", lim, limit_unit or unit, [source_chunk_id])
            if err:
                return _error(err)
        err = self._grounding_error("被核对值", value, unit, require_found=self.context is not None)
        if err:
            return _error(err)
        lu = r["limit_unit"]
        rng = " ~ ".join([fmt(limit_min) if limit_min is not None else "", fmt(limit_max) if limit_max is not None else ""])
        if limit_min is None:
            rng = f"≤ {fmt(limit_max)}"
        elif limit_max is None:
            rng = f"≥ {fmt(limit_min)}"
        lines = [f"核对结论：{r['verdict_cn']}",
                 f"被核对值：{fmt(value)} {r['unit']}"
                 + (f"（换算为 {fmt(r['value_in_limit_unit'])} {lu}）" if not same_unit(unit, lu) else ""),
                 f"手册限值：{rng} {lu}（出自 {source_chunk_id}，限值数字已在原文中核对到）"]
        if r["margin"]:
            word = "高出上限" if r["verdict"] == "above_max" else "低于下限"
            pct = f"（{r['margin_pct']:.1f}%）" if r["margin_pct"] is not None else ""
            lines.append(f"差值：{word} {fmt(r['margin'])} {lu}{pct}")
        lines.append("请按上面的核对结论回答，不要自行重新比较。")
        self.calc.append("\n".join(lines))
        return ToolResult(True, "\n".join(lines), f"{fmt(value)} {r['unit']} → {r['verdict_cn']}",
                          data={**r, "source_chunk_id": source_chunk_id})

    # ---- 分发
    def run(self, name: str, args: dict) -> ToolResult:
        """校验参数并执行一个工具；任何错误都变成 ok=False 的结果（错误信息回给模型）。"""
        model = TOOL_ARGS.get(name)
        if model is None:
            return _error(f"没有名为 {name} 的工具，可用：{', '.join(TOOL_ARGS)}")
        try:
            parsed = model.model_validate(args or {})
        except ValidationError as exc:
            detail = "；".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return _error(f"参数不合法（{detail}）")
        try:
            return getattr(self, name)(**parsed.model_dump())
        except ValueError as exc:      # 手册编号认不出等
            return _error(str(exc))


# --------------------------------------------------------------------------- 执行器（Agent 与 MCP Server 共用）
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")


def run_tool(tb: Toolbox, name: str, args: dict, timeout_s: float) -> tuple[ToolResult, float]:
    """执行一个工具调用，返回 (结果, 耗时 ms)。超时、意外异常都变成 ok=False 的结果（错误信息回给调用方），不抛出。

    在工具箱的副本上执行，按时完成才把新给出的片段并回证据池：超时后工作线程停不下来、会接着跑完，
    它改的只是副本，碰不到证据池，也不会和后面的调用同时改同一份数据。
    """
    t0 = time.perf_counter()
    work = tb.fork()
    fut = _POOL.submit(work.run, name, args)
    try:
        res = fut.result(timeout=timeout_s)
    except FuturesTimeout:
        res = ToolResult(False, f"工具调用失败：执行超过 {timeout_s:g} 秒仍未完成，已放弃。可以换个参数重试或直接作答。",
                         f"超时（>{timeout_s:g} 秒）")
    except Exception as exc:  # 工具内部的意外错误也作为结果返回，不中断调用方
        res = ToolResult(False, f"工具调用失败：{type(exc).__name__}: {exc}", f"{type(exc).__name__}")
    else:
        tb.merge(work)
    return res, round((time.perf_counter() - t0) * 1000, 1)

