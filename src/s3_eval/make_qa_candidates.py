"""S3 任务一：生成评测集候选（供人工核对）。不做检索、不算指标。

读 data/chunks/chunks.jsonl，从真实手册（doc_id 1/2/3/4）分层抽样 40 条 chunk，
用 DeepSeek 逐条生成 1 个问题，输出 eval/qa_candidates.jsonl。

用法：
    python src/s3_eval/make_qa_candidates.py        # 完整 40 条
    python src/s3_eval/make_qa_candidates.py 3      # 先跑 3 条预览
"""
from __future__ import annotations

import html as _html
import json
import math
import random
import re
import sys
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
sys.path.insert(0, str(ROOT / "src"))

from llm_config import get_llm  # noqa: E402

REAL_DOCS = {"1", "2", "3", "4"}
TEXT_N = 30
TABLE_N = 10
SEED = 42


def load_chunks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def preferred(c: dict) -> bool:
    """优先抽 char_len 100–500 且 heading_path 非空的 chunk。"""
    return bool(c.get("heading_path")) and 100 <= c.get("char_len", 0) <= 500


def largest_remainder(counts: dict[str, int], total: int) -> dict[str, int]:
    """按文档比例分配 total 个名额（最大余数法）。"""
    keys = [k for k, v in counts.items() if v > 0]
    denom = sum(counts[k] for k in keys)
    exact = {k: total * counts[k] / denom for k in keys}
    alloc = {k: math.floor(exact[k]) for k in keys}
    rem = total - sum(alloc.values())
    order = sorted(keys, key=lambda k: -(exact[k] - alloc[k]))
    for i in range(rem):
        alloc[order[i % len(order)]] += 1
    return alloc


def alloc_table(counts: dict[str, int], total: int) -> dict[str, int]:
    """每本有表的文档至少 1 张，其余名额按比例分配。"""
    docs_with = [d for d in counts if counts[d] > 0]
    alloc = {d: 1 for d in docs_with}
    remaining = total - len(docs_with)
    if remaining > 0:
        rem_counts = {d: counts[d] - 1 for d in docs_with}
        rem_alloc = largest_remainder(rem_counts, remaining)
        for d in docs_with:
            alloc[d] += rem_alloc.get(d, 0)
    return alloc


def sample_by_doc(chunks: list[dict], alloc: dict[str, int]) -> list[dict]:
    rng = random.Random(SEED)
    by_doc: dict[str, list[dict]] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)
    picked = []
    for did, n in sorted(alloc.items()):
        pool = by_doc.get(did, [])
        pref = [c for c in pool if preferred(c)]
        other = [c for c in pool if not preferred(c)]
        rng.shuffle(pref)
        rng.shuffle(other)
        picked.extend((pref + other)[:n])
    return picked


def strip_html(s: str) -> str:
    return _html.unescape(re.sub(r"<[^>]+>", "", s))


def table_to_text(c: dict) -> str:
    html_text = c.get("table_html") or ""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html_text, re.S)
    lines = []
    for r in rows:
        cells = [strip_html(x).strip() for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
        cells = [x for x in cells if x]
        if cells:
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def chunk_content(c: dict) -> str:
    if c["chunk_type"] == "table":
        return table_to_text(c)
    return c.get("text") or ""


def make_prompt(c: dict) -> str:
    kind = "表格" if c["chunk_type"] == "table" else "正文"
    heading = c.get("heading_path") or ""
    content = chunk_content(c)
    return (
        f"下面是一段设备维修手册的{kind}内容。\n\n"
        f"【章节标题】\n{heading}\n\n"
        f"【内容】\n{content}\n\n"
        f"请站在设备维修人员的角度，针对这段内容提出 1 个问题。\n"
        f"要求：\n"
        f"1. 问题必须能被这段内容直接回答；\n"
        f"2. 不要出现「本文」「上述」「以下」等指代；\n"
        f"3. 不要照抄原文句子；\n"
        f"4. 问题不超过 20 个字。\n\n"
        f"只输出问题本身，不要任何解释、编号或引号。"
    )


def clean_question(q: str) -> str:
    q = (q or "").strip()
    return q.strip("「」『』\"'“”")


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else TEXT_N + TABLE_N
    cfg = yaml.safe_load(CONFIG_PATH.open("r", encoding="utf-8"))
    chunks = load_chunks(ROOT / cfg["s2_chunk"]["output_jsonl"])

    real = [c for c in chunks if c["doc_id"] in REAL_DOCS]
    text_chunks = [c for c in real if c["chunk_type"] == "text"]
    table_chunks = [c for c in real if c["chunk_type"] == "table"]

    text_counts = {d: sum(1 for c in text_chunks if c["doc_id"] == d) for d in REAL_DOCS}
    table_counts = {d: sum(1 for c in table_chunks if c["doc_id"] == d) for d in REAL_DOCS}
    text_alloc = largest_remainder(text_counts, TEXT_N)
    table_alloc = alloc_table(table_counts, TABLE_N)

    print(f"text 抽样分配：{text_alloc}")
    print(f"table 抽样分配：{table_alloc}")

    # heading_path -> 共享该路径的 chunk_id 列表（跨块答案判断用）
    neighbors_map: dict[str, list[str]] = {}
    for c in chunks:
        hp = c.get("heading_path") or ""
        neighbors_map.setdefault(hp, []).append(c["chunk_id"])

    sampled = sample_by_doc(text_chunks, text_alloc) + sample_by_doc(table_chunks, table_alloc)
    sampled = sampled[: limit]

    llm = get_llm()
    api_calls = 0
    results = []
    for i, c in enumerate(sampled, 1):
        prompt = make_prompt(c)
        msg = llm.invoke(prompt)
        api_calls += 1
        question = clean_question(msg.content)
        hp = c.get("heading_path") or ""
        neighbors = [cid for cid in neighbors_map.get(hp, []) if cid != c["chunk_id"]]
        results.append({
            "qa_id": f"qa_{i:03d}",
            "question": question,
            "gold_chunk_id": c["chunk_id"],
            "doc_id": c["doc_id"],
            "chunk_type": c["chunk_type"],
            "heading_path": hp,
            "chunk_text": chunk_content(c)[:200],
            "gold_chunk_neighbors": neighbors,
            "review_status": "pending",
        })

    # 写文件（仅完整 40 条）
    if limit >= TEXT_N + TABLE_N:
        out_path = ROOT / "eval" / "qa_candidates.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\n已写入 {out_path}（{len(results)} 条）")
    else:
        print(f"\n预览模式：仅打印前 {limit} 条，未写文件。")

    # 打印汇总表
    print("\n" + "=" * 70)
    print(f"{'qa_id':<8}{'question':<24}{'gold_chunk_id':<14}{'chunk_type':<8}")
    print("-" * 70)
    for rec in results:
        print(f"{rec['qa_id']:<8}{rec['question']:<24}{rec['gold_chunk_id']:<14}{rec['chunk_type']:<8}")
    print("-" * 70)
    print(f"本次实际 API 调用次数：{api_calls}")
    print("=" * 70)


if __name__ == "__main__":
    main()
