"""S2 任务一：清洗 blocks。只清洗、不合并、不切分、不改 blocks.jsonl。

读 data/parsed_md/blocks.jsonl -> 输出 data/chunks/blocks_clean.jsonl。
清洗规则读 configs/config.yaml 的 s2_clean 段，按 1~6 顺序执行。

用法：
    python src/s2_index/clean_blocks.py
"""
from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_blocks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def strip_invisible(s: str) -> str:
    """剔除 \x00 等不可见/控制/格式字符，保留正常空白（空格、\\n、\\r、\\t）。"""
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


def has_invisible(s: str) -> bool:
    return strip_invisible(s) != s


def is_cjk(ch: str) -> bool:
    o = ord(ch)
    return 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF


def has_cjk_text(s: str) -> bool:
    return any(is_cjk(ch) for ch in s)


def has_alnum(s: str) -> bool:
    """是否含英文/数字（含全角）。"""
    for ch in s:
        if ch.isascii() and ch.isalnum():
            return True
        if 0xFF10 <= ord(ch) <= 0xFF5A:  # 全角数字 0-9 + 全角字母 A-Z/a-z
            return True
    return False


def is_pure_symbol(text: str) -> bool:
    """去掉标点与空白后无中英文数字 -> 视为纯符号块。"""
    stripped = "".join(
        ch for ch in text
        if not (unicodedata.category(ch).startswith("P") or ch.isspace())
    )
    return not has_cjk_text(stripped) and not has_alnum(stripped)


def pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    v = sorted(vals)
    import math
    idx = p * (len(v) - 1)
    lo, hi = math.floor(idx), math.ceil(idx)
    if lo == hi:
        return v[lo]
    return v[lo] + (v[hi] - v[lo]) * (idx - lo)


def main() -> None:
    cfg = load_config()
    s1 = cfg["s1_ingest"]
    sc = cfg["s2_clean"]
    in_path = ROOT / sc["input_jsonl"]
    out_path = ROOT / sc["output_jsonl"]
    toc_dot_min = int(sc["toc_dot_min"])
    min_chars = int(sc["min_block_chars"])

    blocks = load_blocks(in_path)
    before_by_doc = Counter(r["doc_id"] for r in blocks)

    counts = Counter()
    kept: list[dict] = []

    for r in blocks:
        text = r.get("text")
        html = r.get("table_html")
        img = r.get("img_path")
        btype = r.get("type")

        # 1. 空块：三者皆空
        if not text and not html and not img:
            counts["drop_empty"] += 1
            continue

        # 2. 控制字符：剔除不可见字符；空则删，否则保留清洗文本
        flag = "keep"
        if text and has_invisible(text):
            cleaned = strip_invisible(text)
            if not cleaned.strip():
                counts["drop_control_chars"] += 1
                continue
            text = cleaned
            flag = "strip_control_chars"
            counts["strip_control_chars"] += 1

        # 3. 目录页块
        if text and (text.count(".") + text.count("…")) >= toc_dot_min:
            counts["drop_toc"] += 1
            continue

        # 4. 纯符号块
        if text and is_pure_symbol(text):
            counts["drop_pure_symbol"] += 1
            continue

        # 5. 空表格块
        if btype == "table" and not html and not img:
            counts["drop_empty_table"] += 1
            continue

        # 6. 过短块：保留（任务二合并）
        if text and len(text) < min_chars:
            counts["keep_short"] += 1
            if flag == "keep":
                flag = "keep_short"

        kept.append({
            **r,
            "text": text,
            "char_len": len(text) if text else 0,
            "clean_flag": flag,
        })

    # 写输出
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    deleted = len(blocks) - len(kept)
    after_by_doc = Counter(r["doc_id"] for r in kept)

    # ---- 报告 ----
    print("=" * 64)
    print("【各规则命中数】")
    rules = [
        ("1 空块(三者皆空)         drop_empty", "drop_empty"),
        ("2 控制字符->删           drop_control_chars", "drop_control_chars"),
        ("2 控制字符->清洗保留     strip_control_chars", "strip_control_chars"),
        ("3 目录页块               drop_toc", "drop_toc"),
        ("4 纯符号块               drop_pure_symbol", "drop_pure_symbol"),
        ("5 空表格块               drop_empty_table", "drop_empty_table"),
        ("6 过短块(保留)           keep_short", "keep_short"),
    ]
    for label, key in rules:
        print(f"  {label:<28} {counts.get(key, 0)}")
    print("-" * 64)
    print(f"删除总数：{deleted}")
    print(f"剩余块数：{len(kept)}（清洗前 {len(blocks)}）")

    print("\n【剩余块 char_len 分布】")
    lens = [r["char_len"] for r in kept]
    # char_len 仍按原字段，但 text 被清洗的块 char_len 需重算
    lens = [len(r["text"]) if r["text"] else 0 for r in kept]
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

    print("\n【按文档剩余块数对比】")
    print(f"  {'doc_id':<22}{'清洗前':>8}{'剩余':>8}{'删除':>8}")
    for did in sorted(set(before_by_doc) | set(after_by_doc)):
        b = before_by_doc.get(did, 0)
        a = after_by_doc.get(did, 0)
        print(f"  {did:<22}{b:>8}{a:>8}{b - a:>8}")

    print("=" * 64)
    print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()
