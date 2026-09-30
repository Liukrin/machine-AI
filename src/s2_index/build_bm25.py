"""S2 任务四：建 BM25 索引（jieba 分词 + rank_bm25），写入 data/chunks/bm25_index.pkl。

分词与检索文本复用 hybrid_retrieval.build_bm25，保证建索引与查询同一口径。
已存在则覆盖重建：hybrid_retrieval.main() 只在索引缺失时才构建，chunks 变了会读到
旧索引，所以重建知识库时统一走本脚本（scripts/build_index.py 会调用）。

用法：
    python src/s2_index/build_bm25.py
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"
sys.path.insert(0, str(ROOT / "src" / "s3_eval"))

from hybrid_retrieval import build_bm25  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    chunks_path = ROOT / cfg["s2_vector"]["input_jsonl"]
    out_path = ROOT / cfg["s3_retrieval"]["bm25_index"]

    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    bm25, chunk_ids = build_bm25(chunks)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump({"chunk_ids": chunk_ids, "bm25": bm25}, f)

    print(f"BM25 索引：{len(chunk_ids)} 篇，平均 {bm25.avgdl:.1f} 词元/篇，词表 {len(bm25.idf)} 项")
    print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()
