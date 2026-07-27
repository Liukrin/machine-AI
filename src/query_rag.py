"""Query the hydraulic knowledge RAG store."""
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

RAG_STORE_DIR = "rag_store"
MODEL_PATH = "models/bge-small-zh-v1.5"

embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    persist_directory=RAG_STORE_DIR,
    embedding_function=embeddings,
    collection_name="hydraulic_knowledge",
)

# --- 5 test queries ---
queries = [
    ("Q1: 冷却器效率严重下降，出口温度升高",         ["cooler-02", "cooler-03"]),
    ("Q2: 阀门开度损失，压降增大 0.8 bar",           ["valve-03"]),
    ("Q3: 泵内部泄漏严重，流量不足",                  ["pump-02"]),
    ("Q4: 蓄能器预充压力接近失效",                    ["accum-03"]),
    ("Q5: 停机后系统压力掉得很快，泵频繁启动",        ["accum-01", "accum-02", "accum-03"]),
]

hits = 0
for query_text, expected_ids in queries:
    results = vectorstore.similarity_search_with_score(query_text, k=2)
    top_ids = [doc.metadata["id"] for doc, score in results]
    top_scores = [score for doc, score in results]
    is_hit = any(eid in top_ids for eid in expected_ids)
    if is_hit:
        hits += 1
    status = "HIT" if is_hit else "MISS"
    print(f"\n{'='*60}")
    print(f"{status} | {query_text}")
    print(f"  Expected: {expected_ids}")
    print(f"  Top-2:    {top_ids}")
    print(f"  Scores:   [{top_scores[0]:.4f}, {top_scores[1]:.4f}]")
    for i, (doc, score) in enumerate(results):
        print(f"    [{i}] id={doc.metadata['id']}, component={doc.metadata['component']}, "
              f"severity={doc.metadata['severity']}")

print(f"\n{'='*60}")
print(f"Hits: {hits}/5  (threshold: >=4)")
print(f"PASS: {hits >= 4}")
