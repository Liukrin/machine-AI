"""S2 任务三：生成 embedding 并建 Chroma 向量库。

读 data/chunks/chunks.jsonl，用本地 models/bge-small-zh-v1.5（local_files_only、
cpu、归一化）建 chroma_db/ 下的 collection=equipment_manual。

用法：
    python src/s2_index/build_vectors.py
"""
from __future__ import annotations

import html as _html
import json
import os
import re
import sys
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8")

# 强制离线加载，禁止联网下载模型
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer  # noqa: E402
import chromadb  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_chunks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def strip_html(s: str) -> str:
    return _html.unescape(re.sub(r"<[^>]+>", "", s))


def table_to_searchable_text(c: dict) -> str:
    """table chunk 的 table_html 不能直接 embed，拼一段可检索文本。"""
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


def build_embed_text(c: dict) -> str:
    if c["chunk_type"] == "table":
        return table_to_searchable_text(c)
    return c.get("text") or ""


def build_metadata(c: dict) -> dict:
    meta = {
        "chunk_id": c["chunk_id"],
        "doc_id": c["doc_id"],
        "chunk_type": c["chunk_type"],
        "heading_path": c.get("heading_path") or "",
        "page_range": json.dumps(c.get("page_range") or []),
        "source_block_ids": json.dumps(c.get("source_block_ids") or []),
    }
    if c["chunk_type"] == "table":
        meta["table_html"] = c.get("table_html") or ""
    return meta


def main() -> None:
    cfg = load_config()
    sv = cfg["s2_vector"]
    chunks = load_chunks(ROOT / sv["input_jsonl"])
    model_path = ROOT / sv["model_path"]
    chroma_dir = ROOT / sv["chroma_dir"]
    collection_name = sv["collection_name"]
    device = sv["device"]
    normalize = bool(sv["normalize_embeddings"])
    batch_size = int(sv["batch_size"])

    print(f"chunk 数：{len(chunks)}")
    print(f"模型：{model_path}  device={device}  normalize={normalize}")

    # 加载本地模型（只读本地文件；加载失败直接报错，不退回联网下载）
    model = SentenceTransformer(str(model_path), device=device, local_files_only=True)

    # 建库（已存在则删除重建）
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # 批量 embedding + 入库
    embed_texts = [build_embed_text(c) for c in chunks]
    n = len(chunks)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_chunks = chunks[start:end]
        batch_texts = embed_texts[start:end]
        batch_emb = model.encode(
            batch_texts, normalize_embeddings=normalize,
            batch_size=batch_size, show_progress_bar=False,
        )
        collection.add(
            ids=[c["chunk_id"] for c in batch_chunks],
            embeddings=batch_emb.tolist(),
            documents=batch_texts,
            metadatas=[build_metadata(c) for c in batch_chunks],
        )
        print(f"  已入库 {end}/{n}")

    print(f"\ncollection '{collection_name}' 共 {collection.count()} 条")

    # ---- 自检：3 条查询 top-3 ----
    queries = ["联轴器对中允差是多少", "轴承润滑脂多久更换", "泵启动前要做哪些检查"]
    print("\n" + "=" * 80)
    for q in queries:
        q_emb = model.encode([q], normalize_embeddings=normalize, show_progress_bar=False)
        res = collection.query(query_embeddings=q_emb.tolist(), n_results=3,
                               include=["metadatas", "documents", "distances"])
        print(f"\n查询：{q}")
        for i in range(3):
            meta = res["metadatas"][0][i]
            dist = res["distances"][0][i]
            doc = res["documents"][0][i]
            hp = (meta.get("heading_path") or "")[:32]
            body = doc[:80].replace("\n", " ")
            print(f"  #{i + 1}  {meta['chunk_id']}  dist={dist:.4f}  hp=…{hp}")
            print(f"        正文：{body}")
    print("=" * 80)


if __name__ == "__main__":
    main()
