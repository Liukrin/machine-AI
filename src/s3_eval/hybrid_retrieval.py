"""S3 任务四：BM25 + 向量混合检索，与基线同口径对比。不做 rerank、不调 LLM。

读 eval/qa_set.jsonl，向量侧复用 chroma_db，BM25 侧用 jieba 分词 + rank_bm25
对 chunks.jsonl 的检索文本建索引（缓存到 pkl），RRF 融合，输出
eval/reports/s3_hybrid.md。

用法：
    python src/s3_eval/hybrid_retrieval.py
"""
from __future__ import annotations

import html as _html
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
import jieba
from rank_bm25 import BM25Okapi

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer  # noqa: E402
import chromadb  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def strip_html(s: str) -> str:
    return _html.unescape(re.sub(r"<[^>]+>", "", s))


def table_to_searchable_text(c: dict) -> str:
    """与建库时一致：heading + 表头 + 全部单元格拼接。"""
    html_text = c.get("table_html") or ""
    hp = c.get("heading_path") or ""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html_text, re.S)
    cell_rows = []
    for r in rows:
        cells = [strip_html(x).strip() for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
        cells = [x for x in cells if x]
        if cells:
            cell_rows.append(cells)
    parts = []
    if hp:
        parts.append(hp)
    if cell_rows:
        parts.append("表头: " + " | ".join(cell_rows[0]))
    all_cells = [x for row in cell_rows for x in row]
    if all_cells:
        parts.append(" ".join(all_cells))
    return "\n".join(parts)


def build_search_text(c: dict) -> str:
    if c["chunk_type"] == "table":
        return table_to_searchable_text(c)
    return c.get("text") or ""


def tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text) if t.strip()]


def compute_metrics(ranks: list[int | None]) -> tuple[dict, float, float | None]:
    n = len(ranks)
    recall = {k: sum(1 for r in ranks if r is not None and r <= k) / n for k in (1, 3, 5, 10)}
    mrr = sum(1.0 / r for r in ranks if r is not None) / n
    hits = [r for r in ranks if r is not None]
    avg = sum(hits) / len(hits) if hits else None
    return recall, mrr, avg


def rrf_fuse(v_ids: list[str], b_ids: list[str], k: int) -> list[str]:
    scores: dict[str, float] = {}
    for rank, cid in enumerate(v_ids, 1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(b_ids, 1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]


def fmt(v: float) -> str:
    return f"{v:.3f}"


def delta(new: float, old: float) -> str:
    d = new - old
    return f"{d:+.3f}"


# ---- 可复用的混合检索接口（供 S4 agent 调用）----
_CTX: dict = {}


def _load_ctx() -> dict:
    """惰性加载并缓存检索上下文（模型/Chroma/BM25/全量 chunk）。"""
    if "model" in _CTX:
        return _CTX
    cfg = yaml.safe_load(CONFIG_PATH.open("r", encoding="utf-8"))
    sv = cfg["s2_vector"]
    sret = cfg["s3_retrieval"]
    model = SentenceTransformer(str(ROOT / sv["model_path"]), device=sv["device"], local_files_only=True)
    client = chromadb.PersistentClient(path=str(ROOT / sv["chroma_dir"]))
    collection = client.get_collection(sv["collection_name"])
    with (ROOT / sret["bm25_index"]).open("rb") as f:
        cache = pickle.load(f)
    chunks = [json.loads(line) for line in
              (ROOT / sv["input_jsonl"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    _CTX.update({
        "model": model,
        "collection": collection,
        "bm25": cache["bm25"],
        "bm25_ids": cache["chunk_ids"],
        "chunk_by_id": {c["chunk_id"]: c for c in chunks},
        "normalize": bool(sv["normalize_embeddings"]),
        "per_top": int(sret["per_path_top"]),
        "rrf_k": int(sret["rrf_k"]),
    })
    return _CTX


def hybrid_retrieve(question: str, top_k: int = 5) -> tuple[list[dict], list[float]]:
    """混合检索（BM25+向量 RRF 融合），返回 (top_k 个完整 chunk dict, 对应 RRF 分数)。"""
    ctx = _load_ctx()
    emb = ctx["model"].encode([question], normalize_embeddings=ctx["normalize"], show_progress_bar=False)[0]
    vres = ctx["collection"].query(query_embeddings=[emb.tolist()], n_results=ctx["per_top"],
                                   include=["metadatas", "distances"])
    v_ids = vres["ids"][0]
    scores = ctx["bm25"].get_scores(tokenize(question))
    b_ids = [ctx["bm25_ids"][i] for i in np.argsort(scores)[::-1][:ctx["per_top"]]]
    fused: dict[str, float] = {}
    for rank, cid in enumerate(v_ids, 1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (ctx["rrf_k"] + rank)
    for rank, cid in enumerate(b_ids, 1):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (ctx["rrf_k"] + rank)
    ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
    return [ctx["chunk_by_id"][cid] for cid, _ in ranked], [s for _, s in ranked]


def vector_top1_distance(question: str) -> float:
    """向量侧 top-1 的 cosine distance（有绝对意义，用于拒答阈值判断）。"""
    ctx = _load_ctx()
    emb = ctx["model"].encode([question], normalize_embeddings=ctx["normalize"], show_progress_bar=False)[0]
    vres = ctx["collection"].query(query_embeddings=[emb.tolist()], n_results=1, include=["distances"])
    return float(vres["distances"][0][0])


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.open("r", encoding="utf-8"))
    sv = cfg["s2_vector"]
    sr = cfg["s3_retrieval"]
    top_k = int(sr["top_k"])
    per_top = int(sr["per_path_top"])
    rrf_k = int(sr["rrf_k"])

    qa_set = [json.loads(line) for line in (ROOT / sr["qa_set"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks = [json.loads(line) for line in (ROOT / sv["input_jsonl"]).read_text(encoding="utf-8").splitlines() if line.strip()]

    # 向量侧
    model = SentenceTransformer(str(ROOT / sv["model_path"]), device=sv["device"], local_files_only=True)
    client = chromadb.PersistentClient(path=str(ROOT / sv["chroma_dir"]))
    collection = client.get_collection(sv["collection_name"])

    # BM25 侧（缓存）
    idx_path = ROOT / sr["bm25_index"]
    if idx_path.exists():
        with idx_path.open("rb") as f:
            cache = pickle.load(f)
        bm25, bm25_ids = cache["bm25"], cache["chunk_ids"]
        print(f"已加载 BM25 缓存：{idx_path}")
    else:
        tokenized = [tokenize(build_search_text(c)) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        bm25_ids = [c["chunk_id"] for c in chunks]
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        with idx_path.open("wb") as f:
            pickle.dump({"chunk_ids": bm25_ids, "bm25": bm25}, f)
        print(f"已构建并缓存 BM25 索引：{idx_path}（{len(chunks)} 篇）")

    questions = [r["question"] for r in qa_set]
    embs = model.encode(questions, normalize_embeddings=bool(sv["normalize_embeddings"]), show_progress_bar=False)

    per_q = []
    for r, emb in zip(qa_set, embs):
        target = {r["gold_chunk_id"]} | set(r.get("gold_alt") or [])
        vres = collection.query(query_embeddings=[emb.tolist()], n_results=per_top,
                                include=["metadatas", "distances"])
        v_ids = vres["ids"][0]
        v_dists = vres["distances"][0]
        v_meta = vres["metadatas"][0]

        scores = bm25.get_scores(tokenize(r["question"]))
        top_idx = np.argsort(scores)[::-1][:per_top]
        b_ids = [bm25_ids[i] for i in top_idx]

        hybrid_ids = rrf_fuse(v_ids, b_ids, rrf_k)[:top_k]

        base_rank = next((i for i, cid in enumerate(v_ids[:top_k], 1) if cid in target), None)
        hyb_rank = next((i for i, cid in enumerate(hybrid_ids, 1) if cid in target), None)
        per_q.append({**r, "base_rank": base_rank, "hyb_rank": hyb_rank,
                      "v_ids": v_ids, "v_dists": v_dists, "v_meta": v_meta, "hybrid_ids": hybrid_ids})

    # ---- 指标 ----
    base_ranks = [p["base_rank"] for p in per_q]
    hyb_ranks = [p["hyb_rank"] for p in per_q]

    # 报告
    L = ["# S3 BM25+向量混合检索 对比基线\n"]
    L.append(f"- 题目数 {len(qa_set)}  top_k={top_k}  两路各取 top_{per_top}  RRF k={rrf_k}\n")

    L.append("## 1. 对比表（基线 / 混合 / 变化量）\n")
    L.append("| 分组 | 指标 | 基线 | 混合 | 变化量 |")
    L.append("|---|---|---|---|---|")

    groups = {
        "总体": [p for p in per_q],
        "text": [p for p in per_q if p["chunk_type"] == "text"],
        "table": [p for p in per_q if p["chunk_type"] == "table"],
        "hard": [p for p in per_q if p["difficulty"] == "hard"],
    }
    for gname, gitems in groups.items():
        br = [p["base_rank"] for p in gitems]
        hr = [p["hyb_rank"] for p in gitems]
        brec, bmrr, _ = compute_metrics(br)
        hrec, hmrr, _ = compute_metrics(hr)
        for mname, k in [("Recall@1", 1), ("Recall@3", 3), ("Recall@5", 5), ("Recall@10", 10)]:
            L.append(f"| {gname} | {mname} | {fmt(brec[k])} | {fmt(hrec[k])} | {delta(hrec[k], brec[k])} |")
        L.append(f"| {gname} | MRR@10 | {fmt(bmrr)} | {fmt(hmrr)} | {delta(hmrr, bmrr)} |")
    L.append("")

    # ---- qa_038 / qa_039 新排名 ----
    L.append("## 2. qa_038 / qa_039 在混合检索中的新排名\n")
    for qid in ("qa_038", "qa_039"):
        p = next(x for x in per_q if x["qa_id"] == qid)
        rank_s = f"第 {p['hyb_rank']} 位" if p["hyb_rank"] else "仍未进入 top-10"
        L.append(f"- **{qid}**　{p['question']}　→　{rank_s}（基线 {'第 '+str(p['base_rank'])+' 位' if p['base_rank'] else '未命中'}）")
        L.append("  - 混合 top-3：")
        for cid in p["hybrid_ids"][:3]:
            L.append(f"    - `{cid}`")
    L.append("")

    # ---- 因混合而变差的题 ----
    L.append("## 3. 因混合而变差的题（基线位次更靠前）\n")
    worse = [p for p in per_q
             if p["base_rank"] is not None and (p["hyb_rank"] is None or p["hyb_rank"] > p["base_rank"])]
    if worse:
        for p in worse:
            b = str(p["base_rank"])
            h = "未命中" if p["hyb_rank"] is None else str(p["hyb_rank"])
            L.append(f"- {p['qa_id']}　{p['question']}　基线第{b}位 → 混合第{h}位")
    else:
        L.append("(无)")
    L.append("")

    report = ROOT / sr["report"]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(L), encoding="utf-8")

    # 控制台摘要
    brec, bmrr, bavg = compute_metrics(base_ranks)
    hrec, hmrr, havg = compute_metrics(hyb_ranks)
    print(f"基线  R@1={brec[1]:.3f} @3={brec[3]:.3f} @5={brec[5]:.3f} @10={brec[10]:.3f} MRR={bmrr:.3f}")
    print(f"混合  R@1={hrec[1]:.3f} @3={hrec[3]:.3f} @5={hrec[5]:.3f} @10={hrec[10]:.3f} MRR={hmrr:.3f}")
    print(f"变差题数：{len(worse)}  已写入 {report}")


if __name__ == "__main__":
    main()
