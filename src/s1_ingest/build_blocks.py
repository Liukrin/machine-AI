"""S1 任务三：把 MinerU 产物整理成结构化 blocks。

源头只读 content_list.json（排除 _v2），不读 .md。不切 chunk、不向量化、
不清洗文本、不修复表格错位。输出 data/parsed_md/blocks.jsonl。

用法：
    python src/s1_ingest/build_blocks.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

# 输出字段顺序，缺省写 null（不省略键）
FIELDS = [
    "block_id", "doc_id", "page_idx", "type", "text",
    "table_html", "img_path", "heading_path", "char_len",
]

HEADER_FOOTER_TYPES = {"header", "footer"}


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_content_lists(parsed_dir: Path) -> list[tuple[str, Path]]:
    """返回 [(doc_id, content_list.json 路径)]，按 doc_id 排序，排除 *_v2.json。"""
    found = []
    for auto_dir in sorted(parsed_dir.glob("*/auto")):
        doc_id = auto_dir.parent.name
        cl = None
        for p in sorted(auto_dir.glob("*_content_list.json")):
            if not p.name.endswith("_v2.json"):
                cl = p
                break
        if cl is not None:
            found.append((doc_id, cl))
    return found


def build_records(blocks: list[dict], doc_id: str, threshold: float):
    """把单个文档的 content_list 转成记录列表，返回 (records, dropped, total_pages)。

    dropped: dict[text -> 页出现占比]，用于打印被剔除的页眉页脚清单。
    """
    # 页眉页脚文本 -> 出现过的页集合
    total_pages = max((b.get("page_idx", 0) for b in blocks), default=-1) + 1
    occurrence: dict[str, set] = defaultdict(set)
    for b in blocks:
        if b.get("type") in HEADER_FOOTER_TYPES:
            t = b.get("text") or ""
            if t:
                occurrence[t].add(b.get("page_idx"))

    dropped: dict[str, float] = {}
    heading_stack: dict[int, str] = {}  # level -> 标题文本
    records: list[dict] = []

    for idx, b in enumerate(blocks):
        btype = b.get("type")
        if btype == "page_number":
            continue

        text = b.get("text") or ""
        page_idx = b.get("page_idx")
        img_path = b.get("img_path")

        # 标题层级：text 块且带 text_level 时更新章节路径
        if btype == "text" and "text_level" in b:
            level = int(b["text_level"])
            for lv in [lv for lv in heading_stack if lv >= level]:
                del heading_stack[lv]
            heading_stack[level] = text

        heading_path = " > ".join(heading_stack[lv] for lv in sorted(heading_stack)) if heading_stack else None

        # 页眉页脚剔除
        if btype in HEADER_FOOTER_TYPES:
            pages = len(occurrence.get(text, ())) if text else 0
            ratio = pages / total_pages if total_pages else 0.0
            if ratio >= threshold:
                dropped[text] = ratio
                continue
            # 低于阈值：保留为正文（type 不变，填 text）
            rec_type, rec_text, table_html, rec_img = btype, (text or None), None, img_path
        elif btype == "table":
            rec_type, rec_text = "table", None
            table_html = b.get("table_body") or None
            rec_img = img_path
        elif btype == "image":
            rec_type = "image"
            cap = b.get("image_caption") or []
            rec_text = " ".join(cap).strip() or None
            table_html, rec_img = None, img_path
        elif btype == "equation":
            rec_type = "equation"
            rec_text, table_html, rec_img = (text or None), None, img_path
        elif btype == "code":
            rec_type = "code"
            rec_text = b.get("code_body") or None
            table_html, rec_img = None, img_path
        else:  # text / aside_text / 其它
            rec_type, rec_text = btype, (text or None)
            table_html, rec_img = None, img_path

        records.append({
            "block_id": f"{doc_id}_{idx:04d}",
            "doc_id": doc_id,
            "page_idx": page_idx,
            "type": rec_type,
            "text": rec_text,
            "table_html": table_html,
            "img_path": rec_img,
            "heading_path": heading_path,
            "char_len": len(rec_text) if rec_text else 0,
        })

    return records, dropped, total_pages


def main() -> None:
    cfg = load_config()
    s1 = cfg["s1_ingest"]
    parsed_dir = ROOT / s1["parsed_md_dir"]
    out_path = ROOT / s1["blocks_jsonl"]
    pp = s1["postprocess"]
    hf_threshold = float(pp["header_footer_max_occurrence"])
    max_chars = int(pp["max_block_chars"])

    docs = find_content_lists(parsed_dir)
    print(f"parsed_md_dir = {parsed_dir}")
    print(f"输出 = {out_path}")
    print(f"header_footer_max_occurrence = {hf_threshold}  max_block_chars = {max_chars}")
    print(f"共 {len(docs)} 个文档\n")

    all_records: list[dict] = []
    per_doc: dict[str, int] = {}
    all_dropped: list[tuple[str, str, float]] = []  # (doc_id, text, ratio)

    for doc_id, cl_path in docs:
        blocks = json.loads(cl_path.read_text(encoding="utf-8"))
        records, dropped, total_pages = build_records(blocks, doc_id, hf_threshold)
        all_records.extend(records)
        per_doc[doc_id] = len(records)
        for t, r in dropped.items():
            all_dropped.append((doc_id, t, r))
        if dropped:
            print(f"[{doc_id}] 剔除页眉页脚 {len(dropped)} 条（共 {total_pages} 页）:")

    # 写 blocks.jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- 统计 ----
    type_counter = Counter(r["type"] for r in all_records)
    no_heading = sum(1 for r in all_records if not r["heading_path"])
    over_len = sum(1 for r in all_records if r["char_len"] > max_chars)
    total = len(all_records)

    print("\n" + "=" * 72)
    print("【被剔除的页眉页脚清单】(文本 | 页出现占比)")
    print("-" * 72)
    for doc_id, t, r in sorted(all_dropped, key=lambda x: -x[2]):
        t_show = t if len(t) <= 40 else t[:40] + "…"
        print(f"  [{doc_id}] {r:.2f}  {t_show}")
    if not all_dropped:
        print("  (无)")

    print("\n" + "=" * 72)
    print(f"总块数：{total}")
    print("-" * 72)
    print("各 type 数量：")
    for t, c in sorted(type_counter.items(), key=lambda x: -x[1]):
        print(f"  {t:<14} {c}")
    print("-" * 72)
    print("按文档分布：")
    for doc_id, c in sorted(per_doc.items()):
        print(f"  {doc_id:<24} {c}")
    print("-" * 72)
    print(f"heading_path 为空：{no_heading} 块，占比 {no_heading / total * 100:.1f}%")
    print(f"char_len 超过 {max_chars}：{over_len} 块")
    print("=" * 72)
    print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()
