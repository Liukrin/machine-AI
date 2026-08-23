"""S1 任务四：生成质检报告。只统计，不修改 blocks.jsonl、不清洗。

读 data/parsed_md/blocks.jsonl，输出 Markdown 报告到 configs/config.yaml 的
qc_report_md 路径。

用法：
    python src/s1_ingest/qc_report.py
"""
from __future__ import annotations

import datetime
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

MAX_LIST = 10  # 异常块清单每类最多列出的条数


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_blocks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(vals: list[float], p: float) -> float | None:
    """线性插值百分位数。"""
    if not vals:
        return None
    v = sorted(vals)
    idx = p * (len(v) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return v[lo]
    return v[lo] + (v[hi] - v[lo]) * (idx - lo)


def disp(s, n: int = 60) -> str:
    """用于报告表格单元格的文本展示：折叠换行、转义 |、替换控制字符。"""
    if s is None:
        return ""
    s = s.replace("\n", " ").replace("\r", " ").replace("|", "\\|")
    s = "".join(ch if ord(ch) >= 32 else "·" for ch in s)
    return s.strip()[:n]


def is_allowed_char(ch: str) -> bool:
    """判断字符是否属于「中英文 + 数字 + 标点/常见符号」，否则视为乱码候选。"""
    if ch.isspace():
        return True
    o = ord(ch)
    if o < 128:
        return o >= 32 and o != 127  # 可打印 ASCII；控制字符视为乱码
    if 0x00A0 <= o <= 0x00FF:  # Latin-1 补充（° × ÷ ℃ 等）
        return True
    if 0x2000 <= o <= 0x206F:  # 通用标点（… — "" '' 等）
        return True
    if 0x2100 <= o <= 0x214F:  # 字母式符号（℃ ℉ ™ 等）
        return True
    if 0x2200 <= o <= 0x22FF:  # 数学运算符（× ≠ ≤ ≥ 等）
        return True
    if 0x2500 <= o <= 0x257F:  # 制表符
        return True
    if 0x3000 <= o <= 0x303F:  # CJK 标点
        return True
    if 0x3400 <= o <= 0x4DBF:  # CJK 扩展 A
        return True
    if 0x4E00 <= o <= 0x9FFF:  # CJK 统一表意
        return True
    if 0xFF00 <= o <= 0xFFEF:  # 全角形式
        return True
    return False


def garbage_ratio(text: str) -> float:
    if not text:
        return 0.0
    bad = sum(1 for ch in text if not is_allowed_char(ch))
    return bad / len(text)


def table_rows(html: str) -> list[int]:
    """返回每行 <td>/<th> 单元格数；无 html 返回空列表。"""
    if not html:
        return []
    rows = re.findall(r"<tr>(.*?)</tr>", html, re.S)
    return [len(re.findall(r"<t[hd]", r)) for r in rows]


def section4(blocks, min_chars, max_chars):
    """返回 (markdown 字符串, 各类命中数)。"""
    cat = defaultdict(list)
    for r in blocks:
        text = r.get("text")
        bid = (r["block_id"], r["doc_id"], r["page_idx"], disp(text))
        if not text and not r.get("table_html") and not r.get("img_path"):
            cat["空块"].append(bid)
        if text and r["char_len"] < min_chars:
            cat["过短块"].append(bid)
        if r["char_len"] > max_chars:
            cat["超长块"].append(bid)
        if text and (text.count(".") + text.count("…")) >= 20:
            cat["疑似目录页"].append(bid)
        if text and garbage_ratio(text) > 0.30:
            cat["疑似乱码"].append(bid)

    labels = ["空块", "过短块", "超长块", "疑似目录页", "疑似乱码"]
    out = []
    for lab in labels:
        items = cat[lab]
        out.append(f"#### 4.{labels.index(lab) + 1} {lab}（{len(items)} 块）\n")
        if not items:
            out.append("(无)\n")
            continue
        out.append("| block_id | doc_id | page_idx | 文本（前 60 字） |")
        out.append("|---|---|---|---|")
        for bid, did, page, t in items[:MAX_LIST]:
            out.append(f"| {bid} | {did} | {page} | {t} |")
        if len(items) > MAX_LIST:
            out.append(f"| … | 其余 {len(items) - MAX_LIST} 条略 | | |")
        out.append("")
    return "\n".join(out), {lab: len(cat[lab]) for lab in labels}


def main() -> None:
    cfg = load_config()
    s1 = cfg["s1_ingest"]
    blocks_path = ROOT / s1["blocks_jsonl"]
    report_path = ROOT / s1["qc_report_md"]
    pp = s1["postprocess"]
    min_chars = int(pp["min_block_chars"])
    max_chars = int(pp["max_block_chars"])

    blocks = load_blocks(blocks_path)
    docs = sorted({r["doc_id"] for r in blocks})
    L = []  # 报告行

    # ---- 1. 总览 ----
    L.append("# S1 语料解析质检报告\n")
    L.append(f"> 生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    L.append("## 1. 总览\n")
    L.append(f"- 文档数：{len(docs)}")
    L.append(f"- 总块数：{len(blocks)}\n")
    type_counter = Counter(r["type"] for r in blocks)
    L.append("| type | 数量 |")
    L.append("|---|---|")
    for t, c in sorted(type_counter.items(), key=lambda x: -x[1]):
        L.append(f"| {t} | {c} |")
    L.append("")

    # ---- 2. 按文档明细 ----
    L.append("## 2. 按文档明细\n")
    all_types = sorted(type_counter)
    L.append("| doc_id | 块数 | 页数 | " + " | ".join(all_types) + " |")
    L.append("|---" * (len(all_types) + 3) + "|")
    for did in docs:
        sub = [r for r in blocks if r["doc_id"] == did]
        pages = max((r["page_idx"] or 0) for r in sub) + 1
        tc = Counter(r["type"] for r in sub)
        cells = [str(tc.get(t, 0)) for t in all_types]
        L.append(f"| {did} | {len(sub)} | {pages} | " + " | ".join(cells) + " |")
    L.append("")

    # ---- 3. 长度分布 ----
    lens = [r["char_len"] for r in blocks]
    L.append("## 3. 长度分布（char_len，覆盖全部块；table/image 等 text 为空 → 0）\n")
    L.append(f"- min = {min(lens)}  /  中位数 = {pct(lens, 0.5):.0f}  /  均值 = {sum(lens)/len(lens):.1f}"
             f"  /  P90 = {pct(lens, 0.9):.0f}  /  max = {max(lens)}\n")
    buckets = [
        ("<10", lambda x: x < 10),
        ("10-50", lambda x: 10 <= x < 50),
        ("50-200", lambda x: 50 <= x < 200),
        ("200-500", lambda x: 200 <= x < 500),
        ("500-2000", lambda x: 500 <= x <= 2000),
        (">2000", lambda x: x > 2000),
    ]
    L.append("| 桶 | 数量 |")
    L.append("|---|---|")
    for name, fn in buckets:
        L.append(f"| {name} | {sum(1 for x in lens if fn(x))} |")
    L.append("")

    # ---- 4. 异常块清单 ----
    L.append("## 4. 异常块清单\n")
    sec4, _ = section4(blocks, min_chars, max_chars)
    L.append(sec4)

    # ---- 5. 表格质检 ----
    tables = [r for r in blocks if r["type"] == "table"]
    L.append("## 5. 表格质检\n")
    L.append(f"- 表格总数：{len(tables)}\n")
    L.append("按文档分布：")
    L.append("| doc_id | 表格数 |")
    L.append("|---|---|")
    for did in docs:
        L.append(f"| {did} | {sum(1 for r in tables if r['doc_id'] == did)} |")
    L.append("")
    col_stats = [table_rows(r.get("table_html")) for r in tables]
    col_dist = Counter(max(rows) if rows else 0 for rows in col_stats)
    L.append("列数分布（每表取行内最大 `<td>` 数作为列数）：")
    L.append("| 列数 | 表数量 |")
    L.append("|---|---|")
    for c, n in sorted(col_dist.items()):
        L.append(f"| {c} | {n} |")
    L.append("")
    inconsistent = [r["block_id"] for r, rows in zip(tables, col_stats) if len(set(rows)) > 1]
    L.append(f"列数不一致（行内 `<td>` 数不唯一）的表格：{len(inconsistent)} 张")
    if inconsistent:
        L.append("`" + "` `".join(inconsistent) + "`")
    L.append("")

    # ---- 6. 图片质检 ----
    images = [r for r in blocks if r["type"] == "image"]
    with_cap = sum(1 for r in images if r.get("text"))
    L.append("## 6. 图片质检\n")
    L.append(f"- image 块总数：{len(images)}")
    L.append(f"- 有图注：{with_cap}")
    L.append(f"- 无图注：{len(images) - with_cap}\n")

    # ---- 7. heading_path 质检 ----
    L.append("## 7. heading_path 质检\n")
    empty_hp = [r for r in blocks if not r.get("heading_path")]
    depths = [r["heading_path"].count(" > ") + 1 if r.get("heading_path") else 0 for r in blocks]
    depth_dist = Counter(depths)
    L.append(f"- 为空块数：{len(empty_hp)}")
    L.append(f"- 最深层级数：{max(depths)}")
    L.append("各层级深度分布：")
    L.append("| 深度 | 块数 |")
    L.append("|---|---|")
    for d in sorted(depth_dist):
        L.append(f"| {d} | {depth_dist[d]} |")
    L.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(L), encoding="utf-8")
    print(f"已写入 {report_path}")
    print(f"文档 {len(docs)} 份 / 总块数 {len(blocks)}")
    print(f"空块={len([1 for r in blocks if not r.get('text') and not r.get('table_html') and not r.get('img_path')])}"
          f" 表格={len(tables)} 图片={len(images)} 超长块={sum(1 for r in blocks if r['char_len'] > max_chars)}")


if __name__ == "__main__":
    main()
