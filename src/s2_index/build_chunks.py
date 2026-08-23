"""S2 任务二：合并碎块成 chunk。只合并、不向量化、不建索引。

读 data/chunks/blocks_clean.jsonl -> 输出 data/chunks/chunks.jsonl。
合并参数读 configs/config.yaml 的 s2_chunk 段。

用法：
    python src/s2_index/build_chunks.py
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter
from itertools import groupby
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

# 参与文本合并的块类型（修正1：header/footer 一律不参与，直接丢弃）
TEXT_TYPES = {"text", "aside_text", "code", "equation"}

# 修正4：联系方式特征检测
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
URL_RE = re.compile(r"(?:https?://|www\.)[\w./?=&#\-]+", re.I)
PHONE_RE = re.compile(r"(?:\+\d{1,3}[- ]?)?\d{3,4}(?:[- ]\d{3,4})+")
HOTLINE_RE = re.compile(r"热线")
ADDR_RE = re.compile(
    r"(straße|strasse|street|road|地址|邮编|postal|Dortmund|Germany|Deutschland"
    r"|GmbH|有限公司|集团|股份)", re.I,
)


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_blocks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def strip_invisible(s: str) -> str:
    """剔除 \x00 等控制/不可见字符，保留正常空白。"""
    out = []
    for ch in s:
        o = ord(ch)
        if o < 32 and ch not in "\n\r\t":
            continue
        if o == 127:
            continue
        if unicodedata.category(ch) in ("Cf", "Cs", "Co", "Cn"):
            continue
        out.append(ch)
    return "".join(out)


def clean_heading_path(hp: str | None) -> str | None:
    """修正3：剔除控制字符并逐段 strip，跳过清洗后为空的层级。"""
    if not hp:
        return None
    segments = [strip_invisible(s).strip() for s in hp.split(" > ")]
    segments = [s for s in segments if s]
    return " > ".join(segments) if segments else None


def contact_hits(body: str) -> int:
    """修正4：正文命中的联系方式特征类目数。"""
    hits = 0
    if EMAIL_RE.search(body):
        hits += 1
    if URL_RE.search(body):
        hits += 1
    if PHONE_RE.search(body):
        hits += 1
    if HOTLINE_RE.search(body):
        hits += 1
    if ADDR_RE.search(body):
        hits += 1
    return hits


def pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    v = sorted(vals)
    idx = p * (len(v) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return v[lo]
    return v[lo] + (v[hi] - v[lo]) * (idx - lo)


def chunk_text_items(items: list[dict], target: int, maxc: int, overlap: int):
    """把同 heading 的有序文本 item 切成 chunk，返回 list[dict]。"""
    out = []
    i, n = 0, len(items)
    while i < n:
        i_start = i
        j = i
        src_ids, texts, pages, clen = [], [], [], 0
        while j < n:
            it = items[j]
            if clen > 0 and clen + it["char_len"] > maxc:
                break
            src_ids.append(it["block_id"])
            texts.append(it["text"])
            pages.append(it["page_idx"])
            clen += it["char_len"]
            j += 1
            if clen >= target:
                break
        body = "\n\n".join(t for t in texts if t)
        out.append({
            "source_block_ids": src_ids,
            "body": body,
            "page_range": [min(pages), max(pages)],
        })
        if j >= n:
            break
        i = max(j - overlap, i_start + 1)
    return out


def main() -> None:
    cfg = load_config()
    sc = cfg["s2_chunk"]
    blocks = load_blocks(ROOT / sc["input_jsonl"])
    target = int(sc["target_chars"])
    maxc = int(sc["max_chars"])
    overlap = int(sc["overlap_blocks"])
    min_body_chars = int(sc["min_body_chars"])
    contact_hits_min = int(sc["contact_hits_min"])

    chunks: list[dict] = []
    drop_counts: Counter = Counter()
    normalized_headings: set = set()

    for doc_id, doc_blocks_iter in groupby(blocks, key=lambda r: r["doc_id"]):
        doc_blocks = list(doc_blocks_iter)
        seq = 0
        text_items: list[dict] = []
        cur_heading = None

        def emit_text_chunks(heading):
            nonlocal seq
            clean_heading = clean_heading_path(heading)
            if heading and clean_heading != heading:
                normalized_headings.add(heading)
            for part in chunk_text_items(text_items, target, maxc, overlap):
                body = part["body"]
                # 修正2：正文过短丢弃
                if len(body.strip()) < min_body_chars:
                    drop_counts["min_body"] += 1
                    continue
                # 修正4：联系方式过滤
                if contact_hits(body) >= contact_hits_min:
                    drop_counts["contact"] += 1
                    continue
                text = f"{clean_heading}\n\n{body}" if clean_heading else body
                chunks.append({
                    "chunk_id": f"{doc_id}_c{seq:04d}",
                    "doc_id": doc_id,
                    "chunk_type": "text",
                    "heading_path": clean_heading,
                    "text": text,
                    "table_html": None,
                    "source_block_ids": part["source_block_ids"],
                    "char_len": len(text),
                    "page_range": part["page_range"],
                })
                seq += 1

        for b in doc_blocks:
            t = b["type"]
            # 修正1：页眉页脚直接丢弃，不参与合并
            if t in ("header", "footer"):
                drop_counts["header_footer"] += 1
                continue

            if t == "table":
                if text_items:
                    emit_text_chunks(cur_heading)
                    text_items = []
                chunks.append({
                    "chunk_id": f"{doc_id}_c{seq:04d}",
                    "doc_id": doc_id,
                    "chunk_type": "table",
                    "heading_path": clean_heading_path(b.get("heading_path")),
                    "text": None,
                    "table_html": b.get("table_html"),
                    "source_block_ids": [b["block_id"]],
                    "char_len": len(b.get("table_html") or ""),
                    "page_range": [b.get("page_idx"), b.get("page_idx")],
                })
                seq += 1
                continue

            if t == "image":
                if not b.get("text"):
                    continue  # 无图注：丢弃
                hp = b.get("heading_path")
                if hp != cur_heading:
                    if text_items:
                        emit_text_chunks(cur_heading)
                        text_items = []
                    cur_heading = hp
                text_items.append({
                    "block_id": b["block_id"], "text": b["text"],
                    "char_len": b["char_len"], "page_idx": b["page_idx"],
                })
                continue

            if t in TEXT_TYPES:
                hp = b.get("heading_path")
                if hp != cur_heading:
                    if text_items:
                        emit_text_chunks(cur_heading)
                        text_items = []
                    cur_heading = hp
                text_items.append({
                    "block_id": b["block_id"], "text": b["text"],
                    "char_len": b["char_len"], "page_idx": b["page_idx"],
                })
                continue
            # 其余类型忽略

        if text_items:
            emit_text_chunks(cur_heading)

    # 写输出
    out_path = ROOT / sc["output_jsonl"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # ---- 报告 ----
    text_chunks = [c for c in chunks if c["chunk_type"] == "text"]
    table_chunks = [c for c in chunks if c["chunk_type"] == "table"]
    lens = [c["char_len"] for c in text_chunks]

    print("=" * 70)
    print("【各修正规则丢弃数量】")
    print(f"  修正1 页眉页脚剔除(块)       {drop_counts.get('header_footer', 0)}")
    print(f"  修正2 正文过短丢弃(chunk)    {drop_counts.get('min_body', 0)}")
    print(f"  修正3 heading归一化(非丢弃)   {len(normalized_headings)} 个 heading 被清洗")
    print(f"  修正4 联系方式丢弃(chunk)    {drop_counts.get('contact', 0)}")

    print("\n【chunk 总数】")
    print(f"  共 {len(chunks)}（text={len(text_chunks)}  table={len(table_chunks)}）")

    print("\n【text chunk 的 char_len 分布】")
    print(f"  min={min(lens)}  中位数={pct(lens, 0.5):.0f}  均值={sum(lens)/len(lens):.1f}"
          f"  P90={pct(lens, 0.9):.0f}  max={max(lens)}")
    buckets = [
        ("<10", lambda x: x < 10),
        ("10-50", lambda x: 10 <= x < 50),
        ("50-200", lambda x: 50 <= x < 200),
        ("200-500", lambda x: 200 <= x < 500),
        ("500-2000", lambda x: 500 <= x <= 2000),
        (">2000", lambda x: x > 2000),
    ]
    for name, fn in buckets:
        print(f"  {name:<10} {sum(1 for x in lens if fn(x))}")

    print("\n【按文档分布】")
    by_doc = Counter(c["doc_id"] for c in chunks)
    by_doc_t = Counter(c["doc_id"] for c in text_chunks)
    by_doc_tab = Counter(c["doc_id"] for c in table_chunks)
    print(f"  {'doc_id':<22}{'text':>6}{'table':>8}{'合计':>8}")
    for did in sorted(by_doc):
        print(f"  {did:<22}{by_doc_t[did]:>6}{by_doc_tab[did]:>8}{by_doc[did]:>8}")

    # 随机抽 5 个 text chunk 完整打印
    random.seed(42)
    sample = random.sample(text_chunks, min(5, len(text_chunks)))
    print("\n【随机抽 5 个 text chunk 完整内容】")
    for i, c in enumerate(sample, 1):
        print(f"\n----- 样本 {i}：{c['chunk_id']} | {c['doc_id']} | page {c['page_range']} | "
              f"char_len {c['char_len']} | heading {str(c['heading_path'])[:40]} -----")
        print(c["text"])

    print("\n" + "=" * 70)
    print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()
