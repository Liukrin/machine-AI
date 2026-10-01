"""S3 任务五：混合召回 + CrossEncoder 精排评测。三方对比同口径，不调 LLM。

读 eval/qa_set.jsonl，召回复用 hybrid 的 RRF 混合（top_20），再用本地
bge-reranker-base 精排取 top_10，输出 eval/reports/s3_rerank.md。

用法：
    python src/s3_eval/rerank_retrieval.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent))  # src/s3_eval
from hybrid_retrieval import rrf_fuse, build_search_text, tokenize, compute_metrics  # noqa: E402

from sentence_transformers import SentenceTransformer, CrossEncoder  # noqa: E402
import chromadb  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def rank_str(r):
    return f"第{r}位" if r is not None else "未命中"


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open("r", encoding="utf-8"))
    sv = cfg["s2_vector"]
    sret = cfg["s3_retrieval"]
    srer = cfg["s3_rerank"]
    top_k = int(srer["top_k"])
    per_top = int(srer["per_path_top"])
    rrf_k = int(srer["rrf_k"])
    batch = int(srer["batch_size"])

    qa_set = [json.loads(line) for line in (ROOT / srer["qa_set"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks = [json.loads(line) for line in (ROOT / sv["input_jsonl"]).read_text(encoding="utf-8").splitlines() if line.strip()]

    # 向量
    model = SentenceTransformer(str(ROOT / sv["model_path"]), device=sv["device"], local_files_only=True)
    client = chromadb.PersistentClient(path=str(ROOT / sv["chroma_dir"]))
    collection = client.get_collection(sv["collection_name"])

    # BM25（复用缓存）
    with (ROOT / sret["bm25_index"]).open("rb") as f:
        cache = pickle.load(f)
    bm25, bm25_ids = cache["bm25"], cache["chunk_ids"]

    # rerank
    reranker = CrossEncoder(str(ROOT / srer["rerank_model"]), device="cpu", local_files_only=True)
    print("CrossEncoder 加载完成")

    search_text = {c["chunk_id"]: build_search_text(c) for c in chunks}

    questions = [r["question"] for r in qa_set]
    embs = model.encode(questions, normalize_embeddings=bool(sv["normalize_embeddings"]), show_progress_bar=False)

    rerank_total = 0.0
    per_q = []
    for idx, (r, emb) in enumerate(zip(qa_set, embs), 1):
        target = {r["gold_chunk_id"]} | set(r.get("gold_alt") or [])
        vres = collection.query(query_embeddings=[emb.tolist()], n_results=per_top,
                                include=["metadatas", "distances"])
        v_ids = vres["ids"][0]
        scores = bm25.get_scores(tokenize(r["question"]))
        b_ids = [bm25_ids[i] for i in np.argsort(scores)[::-1][:per_top]]
        recall_ids = rrf_fuse(v_ids, b_ids, rrf_k)[:per_top]

        base_rank = next((i for i, cid in enumerate(v_ids[:top_k], 1) if cid in target), None)
        hyb_rank = next((i for i, cid in enumerate(recall_ids[:top_k], 1) if cid in target), None)

        pairs = [[r["question"], search_text.get(cid, "")] for cid in recall_ids]
        t0 = time.time()
        r_scores = reranker.predict(pairs, batch_size=batch, show_progress_bar=False)
        rerank_total += time.time() - t0
        rerank_ids = [recall_ids[i] for i in np.argsort(r_scores)[::-1]][:top_k]
        rerank_rank = next((i for i, cid in enumerate(rerank_ids, 1) if cid in target), None)

        per_q.append({**r, "base_rank": base_rank, "hyb_rank": hyb_rank, "rerank_rank": rerank_rank})
        if idx % 8 == 0 or idx == len(qa_set):
            print(f"  rerank {idx}/{len(qa_set)}")

    # ---- 指标 ----
    def ranks(key):
        return [p[key] for p in per_q]

    L = ["# S3 混合+rerank 精排评测（三方对比）\n"]
    L.append(f"- 题目数 {len(qa_set)}  召回 top_{per_top}  精排 top_{top_k}  RRF k={rrf_k}  batch={batch}\n")

    L.append("## 1. 对比表（基线 / 混合 / 混合+rerank / 相对混合变化量）\n")
    L.append("| 分组 | 指标 | 基线 | 混合 | 混合+rerank | 相对混合变化 |")
    L.append("|---|---|---|---|---|---|")
    groups = {
        "总体": per_q,
        "text": [p for p in per_q if p["chunk_type"] == "text"],
        "table": [p for p in per_q if p["chunk_type"] == "table"],
        "hard": [p for p in per_q if p["difficulty"] == "hard"],
    }
    for gname, gitems in groups.items():
        brec, bmrr, _ = compute_metrics([p["base_rank"] for p in gitems])
        hrec, hmrr, _ = compute_metrics([p["hyb_rank"] for p in gitems])
        rrec, rmrr, _ = compute_metrics([p["rerank_rank"] for p in gitems])
        for mname, k in [("Recall@1", 1), ("Recall@3", 3), ("Recall@5", 5), ("Recall@10", 10)]:
            L.append(f"| {gname} | {mname} | {brec[k]:.3f} | {hrec[k]:.3f} | {rrec[k]:.3f} | {rrec[k]-hrec[k]:+.3f} |")
        L.append(f"| {gname} | MRR@10 | {bmrr:.3f} | {hmrr:.3f} | {rmrr:.3f} | {rmrr-hmrr:+.3f} |")
    L.append("")

    # ---- qa_037/038/039 ----
    L.append("## 2. qa_037 / qa_038 / qa_039 排名变化\n")
    L.append("| qa_id | 问题 | 基线 | 混合 | 混合+rerank |")
    L.append("|---|---|---|---|---|")
    for qid in ("qa_037", "qa_038", "qa_039"):
        p = next(x for x in per_q if x["qa_id"] == qid)
        L.append(f"| {qid} | {p['question']} | {rank_str(p['base_rank'])} | {rank_str(p['hyb_rank'])} | {rank_str(p['rerank_rank'])} |")
    L.append("")

    # ---- 因 rerank 变差 ----
    L.append("## 3. 因 rerank 变差的题（相对混合）\n")
    worse = [p for p in per_q
             if p["hyb_rank"] is not None and (p["rerank_rank"] is None or p["rerank_rank"] > p["hyb_rank"])]
    if worse:
        for p in worse:
            L.append(f"- {p['qa_id']}　{p['question']}　混合{rank_str(p['hyb_rank'])} → rerank{rank_str(p['rerank_rank'])}")
    else:
        L.append("(无)")
    L.append("")

    # ---- 耗时 ----
    L.append("## 4. rerank 耗时\n")
    L.append(f"- 总耗时：{rerank_total:.1f}s")
    L.append(f"- 单题平均：{rerank_total / len(qa_set):.2f}s")
    L.append("")

    report = ROOT / srer["report"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(L), encoding="utf-8")

    # 控制台摘要
    brec, bmrr, _ = compute_metrics([p["base_rank"] for p in per_q])
    hrec, hmrr, _ = compute_metrics([p["hyb_rank"] for p in per_q])
    rrec, rmrr, _ = compute_metrics([p["rerank_rank"] for p in per_q])
    print(f"基线  R@10={brec[10]:.3f} MRR={bmrr:.3f}")
    print(f"混合  R@10={hrec[10]:.3f} MRR={hmrr:.3f}")
    print(f"rerank R@10={rrec[10]:.3f} MRR={rmrr:.3f}")
    print(f"rerank 总耗时 {rerank_total:.1f}s，单题 {rerank_total/len(qa_set):.2f}s")
    print(f"已写入 {report}")


if __name__ == "__main__":
    main()
