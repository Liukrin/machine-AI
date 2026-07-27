"""Build Chroma vector store from knowledge/ .md files.

Chunking strategy: one .md file = one chunk.
Reason: each file is a complete "phenomenon-cause-action" loop.
Token-level splitting would separate the phenomenon from the action,
making retrieval results non-actionable.
"""

import os, yaml, shutil
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- Config ---
KNOWLEDGE_DIR = "knowledge"
RAG_STORE_DIR = "rag_store"
MODEL_PATH = "models/bge-small-zh-v1.5"

# --- 1. Load .md files as Documents ---
documents = []
for fname in sorted(os.listdir(KNOWLEDGE_DIR)):
    if not fname.endswith(".md"):
        continue
    fpath = os.path.join(KNOWLEDGE_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split frontmatter and body
    parts = content.split("---", 2)
    if len(parts) >= 3:
        fm = yaml.safe_load(parts[1])
        body = parts[2].strip()
    else:
        fm = {}
        body = content.strip()

    doc = Document(
        page_content=body,
        metadata={
            "id": fm.get("id", ""),
            "component": fm.get("component", ""),
            "severity": fm.get("severity", ""),
            "source": fm.get("source", ""),
            "filename": fname,
        },
    )
    documents.append(doc)
    print(f"  Loaded: {fname} -> id={fm.get('id', '?')}")

print(f"  Total documents: {len(documents)}")

# --- 2. Build embedding function ---
embeddings = HuggingFaceEmbeddings(
    model_name=MODEL_PATH,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# --- 3. Build Chroma (overwrite if exists) ---
if os.path.exists(RAG_STORE_DIR):
    shutil.rmtree(RAG_STORE_DIR)
    print(f"  Removed existing {RAG_STORE_DIR}")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=RAG_STORE_DIR,
    collection_name="hydraulic_knowledge",
)
print(f"  Persisted {vectorstore._collection.count()} chunks to {RAG_STORE_DIR}")
print("  DONE")
