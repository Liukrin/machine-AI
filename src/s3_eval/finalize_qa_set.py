"""S3 任务二：把候选集定稿为正式评测集。纯文件处理，不调 API、不做检索。

读 eval/qa_candidates.jsonl -> 输出 eval/qa_set.jsonl。

用法：
    python src/s3_eval/finalize_qa_set.py
"""
from __future__ import annotations

import html as _html
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

DROP_IDS = {"qa_025", "qa_032", "qa_036", "qa_040"}
QA001_NEW_GOLD = "1_c0030"
QA001_ALT = "1_c0029"


def strip_html(s: str) -> str:
    return _html.unescape(re.sub(r"<[^>]+>", "", s))


def gold_text(c: dict) -> str:
    if c["chunk_type"] == "table":
        return strip_html(c.get("table_html") or "")
    return c.get("text") or ""


def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for ch in text if ch.isdigit()) / len(text)


def difficulty(cand: dict, chunk_by_id: dict[str, dict]) -> str:
    """确定性判断，不用 LLM。优先级 hard > medium > easy。"""
    if cand["chunk_type"] == "table":
        gold = chunk_by_id.get(cand["gold_chunk_id"])
        if gold and digit_ratio(gold_text(gold)) > 0.30:
            return "hard"
    if cand.get("gold_chunk_neighbors"):
        return "medium"
    return "easy"


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open("r", encoding="utf-8"))
    cands = [json.loads(line) for line in
             (ROOT / "eval" / "qa_candidates.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    chunks = [json.loads(line) for line in
              (ROOT / cfg["s2_chunk"]["output_jsonl"]).read_text(encoding="utf-8").splitlines()
              if line.strip()]
    chunk_by_id = {c["chunk_id"]: c for c in chunks}

    out = []
    for cand in cands:
        qid = cand["qa_id"]
        if qid in DROP_IDS:
            continue

        gold_id = cand["gold_chunk_id"]
        gold_alt: list[str] = []
        if qid == "qa_001":
            gold_id = QA001_NEW_GOLD
            gold_alt = [QA001_ALT]

        rec = {
            "qa_id": qid,
            "question": cand["question"],
            "gold_chunk_id": gold_id,
            "gold_alt": gold_alt,
            "doc_id": cand["doc_id"],
            "chunk_type": cand["chunk_type"],
            "heading_path": cand["heading_path"],
            "difficulty": None,
        }
        rec["difficulty"] = difficulty({**cand, "gold_chunk_id": gold_id}, chunk_by_id)
        out.append(rec)

    out_path = ROOT / "eval" / "qa_set.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- 报告 ----
    diff = Counter(r["difficulty"] for r in out)
    ctype = Counter(r["chunk_type"] for r in out)
    by_doc = Counter(r["doc_id"] for r in out)

    print("=" * 60)
    print(f"最终条数：{len(out)}（删除 {len(DROP_IDS)} 条）")
    print(f"difficulty：hard={diff.get('hard', 0)}  medium={diff.get('medium', 0)}  easy={diff.get('easy', 0)}")
    print(f"chunk_type：text={ctype.get('text', 0)}  table={ctype.get('table', 0)}")
    print("按文档分布：")
    for d in sorted(by_doc):
        print(f"  doc {d}: {by_doc[d]}")
    print("=" * 60)
    print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()
