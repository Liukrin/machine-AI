"""Query the hydraulic knowledge RAG store with rejection logic + metadata gates.

Score direction: similarity_search_with_score returns cosine DISTANCE.
Lower = more similar. Self-match = 0.0.
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

RAG_STORE_DIR = "rag_store"
MODEL_PATH = "models/bge-small-zh-v1.5"

# Valid components in knowledge base
KNOWN_COMPONENTS = {"冷却器", "换向阀", "液压泵", "蓄能器", "系统级"}

# Threshold calibrated on 6 positive + 6 negative samples:
#   positive scores: [0.367, 0.583]
#   negative scores: [0.649, 0.846]
# Clean gap at 0.583-0.649 -> threshold = 0.62
TAU_RETRIEVAL = 0.62

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


def retrieve_or_reject(query: str, k: int = 2,
                       component: str = None, severity: str = None):
    """Top-k retrieval with metadata gates.

    Gate 1 (hard): if component not in KNOWN_COMPONENTS, reject immediately
                   without vector search.
    Gate 2 (hard): if severity is provided, filter by exact severity match.
                   If no match, fall back to nearest within same component.
    Gate 3 (distance): among filtered results, apply TAU_RETRIEVAL threshold.

    Returns dict:
      {"hit": True,  "results": [...], "best_score": float,
       "severity_fallback": bool (if severity was requested)}
      {"hit": False, "reason": "...", "best_score": null}
    """
    import json

    # Gate 1: component filter
    if component is not None and component not in KNOWN_COMPONENTS:
        return {
            "hit": False,
            "reason": f"该组件「{component}」无知识库条目",
            "best_score": None,
        }

    # Gate 2: hard severity filter via Chroma metadata
    severity_fallback = False
    if severity is not None and component is not None:
        where_filter = {"$and": [{"component": component}, {"severity": severity}]}
        results = vectorstore.similarity_search_with_score(
            query, k=max(k, 10), filter=where_filter
        )
        if not results:
            # Fallback: same component, any severity
            severity_fallback = True
            results = vectorstore.similarity_search_with_score(
                query, k=max(k, 10),
                filter={"component": component},
            )
        if not results:
            return {"hit": False, "reason": "无相关条目", "best_score": None}
        # Return only exact-severity matches (or fallback) — no mixing
        # Trim to k
        results = results[:k]
    elif component is not None:
        where_filter = {"component": component}
        results = vectorstore.similarity_search_with_score(
            query, k=k, filter=where_filter
        )
    else:
        results = vectorstore.similarity_search_with_score(query, k=k)

    if not results:
        return {"hit": False, "reason": "无相关条目", "best_score": None}

    # Gate 3: distance threshold
    best_score = float(results[0][1])
    if best_score < TAU_RETRIEVAL:
        out = {
            "hit": True,
            "results": results[:k],
            "best_score": best_score,
        }
        if severity is not None:
            out["severity_fallback"] = severity_fallback
        return out
    else:
        return {
            "hit": False,
            "reason": f"无相关条目（最佳距离 {best_score:.4f} > tau={TAU_RETRIEVAL}）",
            "best_score": best_score,
        }


# ============================================================
# Regression test
# ============================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from severity_map import map_severity

    print("=== Regression: retrieve_or_reject with metadata gates ===\n")

    # Test 1: A scenario (cooler, s_level=20.06 -> severe)
    sev_a = map_severity("cooler", 20.06)
    print(f"Test 1: cooler, s_level=20.06 -> severity={sev_a}")
    r1 = retrieve_or_reject("冷却器效率严重下降温度升高", component="冷却器", severity=sev_a)
    if r1["hit"]:
        ids = [d.metadata["id"] for d, s in r1["results"]]
        print(f"  hit=True, top-1={ids[0]}, score={r1['best_score']:.4f}")
        print(f"  expected: cooler-02")
    else:
        print(f"  hit=False: {r1['reason']}")

    print()

    # Test 2: C scenario (cooler, s_level=7.97 -> moderate)
    sev_c = map_severity("cooler", 7.97)
    print(f"Test 2: cooler, s_level=7.97 -> severity={sev_c}")
    r2 = retrieve_or_reject("冷却器效率下降", component="冷却器", severity=sev_c)
    if r2["hit"]:
        ids = [d.metadata["id"] for d, s in r2["results"]]
        print(f"  hit=True, top-1={ids[0]}, score={r2['best_score']:.4f}")
        print(f"  expected: cooler-01")
    else:
        print(f"  hit=False: {r2['reason']}")

    print()

    # Test 3: component not in knowledge base
    print("Test 3: component='齿轮箱' (not in KB)")
    r3 = retrieve_or_reject("齿轮箱打齿", component="齿轮箱")
    print(f"  hit={r3['hit']}, reason={r3['reason']}")
    print(f"  expected: hit=False, gate 1 rejection")

    print()
    if r1["hit"] and r2["hit"]:
        same = r1["results"][0][0].metadata["id"] == r2["results"][0][0].metadata["id"]
        print(f"A/C top-1 differ: {not same} (A={r1['results'][0][0].metadata['id']}, C={r2['results'][0][0].metadata['id']})")
