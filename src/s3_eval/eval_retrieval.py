"""S3 任务三：纯向量检索基线评测。不做优化，只测现状，不调 LLM。

读 eval/qa_set.jsonl，对 chroma_db/equipment_manual 逐条检索，输出报告
eval/reports/s3_baseline.md。

用法：
    python src/s3_eval/eval_retrieval.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer  # noqa: E402
import chromadb  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
QA_SET = ROOT / "eval" / "qa_set.jsonl"
REPORT = ROOT / "eval" / "reports" / "s3_baseline.md"
TOP_K = 10


def compute_metrics(ranks: list[int | None]) -> tuple[dict, float, float | None]:
    """ranks: 每题首个命中位次（1-based）或 None。返回 (recall, mrr, avg_rank)。"""
    n = len(ranks)
    recall = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in (1, 3, 5, 10)}
    mrr = sum(1.0 / r for r in ranks if r is not None) / n
    hits = [r for r in ranks if r is not None]
    avg_rank = sum(hits) / len(hits) if hits else None
    return recall, mrr, avg_rank


def fmt_recall(v: float) -> str:
    return f"{v:.3f}"


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open("r", encoding="utf-8"))
    sv = cfg["s2_vector"]
    qa_set = [json.loads(line) for line in QA_SET.read_text(encoding="utf-8").splitlines() if line.strip()]

    model = SentenceTransformer(str(ROOT / sv["model_path"]), device=sv["device"], local_files_only=True)
    client = chromadb.PersistentClient(path=str(ROOT / sv["chroma_dir"]))
    collection = client.get_collection(sv["collection_name"])

    questions = [r["question"] for r in qa_set]
    embs = model.encode(questions, normalize_embeddings=bool(sv["normalize_embeddings"]), show_progress_bar=False)

    results = []
    for r, emb in zip(qa_set, embs):
        res = collection.query(
            query_embeddings=[emb.tolist()], n_results=TOP_K,
            include=["metadatas", "distances"],
        )
        ids = res["ids"][0]
        dists = res["distances"][0]
        metas = res["metadatas"][0]
        target = {r["gold_chunk_id"]} | set(r.get("gold_alt") or [])
        first = next((i for i, cid in enumerate(ids, 1) if cid in target), None)
        top3 = [(ids[i], dists[i], (metas[i].get("heading_path") or "")) for i in range(min(3, len(ids)))]
        results.append({**r, "first_rank": first, "top3": top3})

    # ---- 分组指标 ----
    def group_rows(key_fn):
        groups = defaultdict(list)
        for r in results:
            groups[key_fn(r)].append(r["first_rank"])
        rows = []
        for key in sorted(groups, key=str):
            recall, mrr, avg = compute_metrics(groups[key])
            rows.append((key, len(groups[key]), recall, mrr, avg))
        return rows

    all_recall, all_mrr, all_avg = compute_metrics([r["first_rank"] for r in results])
    diff_rows = group_rows(lambda r: r["difficulty"])
    type_rows = group_rows(lambda r: r["chunk_type"])
    doc_rows = group_rows(lambda r: r["doc_id"])

    # ---- 生成报告 ----
    L: list[str] = []
    L.append("# S3 纯向量检索基线评测\n")

    L.append("## 1. 总览\n")
    L.append(f"- 题目数：{len(results)}  top_k={TOP_K}")
    L.append(f"- Recall@1 / @3 / @5 / @10：{fmt_recall(all_recall[1])} / {fmt_recall(all_recall[3])} / {fmt_recall(all_recall[5])} / {fmt_recall(all_recall[10])}")
    L.append(f"- MRR@10：{all_mrr:.3f}")
    L.append(f"- 平均命中位次（仅命中题）：{all_avg:.2f}" if all_avg is not None else "- 平均命中位次：N/A")
    L.append("")

    header = "| 分组 | n | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | 平均命中位次 |"
    sep = "|---|---|---|---|---|---|---|---|"

    def emit_rows(title, rows):
        L.append(f"## {title}\n")
        L.append(header)
        L.append(sep)
        for key, n, recall, mrr, avg in rows:
            avg_s = f"{avg:.2f}" if avg is not None else "-"
            L.append(f"| {key} | {n} | {fmt_recall(recall[1])} | {fmt_recall(recall[3])} | "
                     f"{fmt_recall(recall[5])} | {fmt_recall(recall[10])} | {mrr:.3f} | {avg_s} |")
        L.append("")

    emit_rows("2. 按 difficulty", diff_rows)
    emit_rows("3. 按 chunk_type", type_rows)
    emit_rows("4. 按 doc_id", doc_rows)

    # ---- 失败清单 ----
    misses = [r for r in results if r["first_rank"] is None]
    L.append(f"## 5. 失败清单（Recall@10 未命中，共 {len(misses)} 条）\n")
    if misses:
        for r in misses:
            L.append(f"### {r['qa_id']}  {r['question']}")
            L.append(f"- difficulty={r['difficulty']}  chunk_type={r['chunk_type']}  doc_id={r['doc_id']}")
            L.append(f"- gold={r['gold_chunk_id']}  gold_heading={r['heading_path']}")
            L.append("- 实际 top-3：")
            for cid, dist, hp in r["top3"]:
                L.append(f"  - `{cid}`  dist={dist:.4f}  heading=…{hp[:36]}")
            L.append("")
    else:
        L.append("(无，全部命中)\n")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")

    # 控制台摘要
    print(f"题目数 {len(results)}")
    print(f"Recall@1={all_recall[1]:.3f} @3={all_recall[3]:.3f} @5={all_recall[5]:.3f} @10={all_recall[10]:.3f}")
    print(f"MRR@10={all_mrr:.3f}  平均命中位次={all_avg:.2f}" if all_avg is not None else f"MRR@10={all_mrr:.3f}")
    print(f"未命中 {len(misses)} 条")
    print(f"已写入 {REPORT}")


if __name__ == "__main__":
    main()
