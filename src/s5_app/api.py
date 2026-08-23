"""S5 任务一：FastAPI 后端。

复用 S4 graph.py 的节点函数与拒答阈值（retrieve / verify_node / TAU_DISTANCE /
SYSTEM_PROMPT），不复制其检索与校验逻辑；仅 LLM 生成部分改为 stream=True 逐块
转发，以满足 SSE 的 token 事件。

接口：
    GET  /api/health   健康检查
    POST /api/ask      SSE 流式问答

用法：
    python src/s5_app/api.py     # 默认 127.0.0.1:8000
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "s4_agent"))
sys.path.insert(0, str(ROOT / "src" / "s3_eval"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from sse_starlette.sse import EventSourceResponse  # noqa: E402

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from llm_config import LLM_CONFIG, get_llm  # noqa: E402
from graph import SYSTEM_PROMPT, TAU_DISTANCE, retrieve, verify_node  # noqa: E402
from hybrid_retrieval import (  # noqa: E402
    _load_ctx,
    build_search_text,
    vector_top1_distance,
)

MAX_TOKENS = 500          # 与 graph.py generate 保持一致
PREVIEW_CHARS = 100       # retrieval 事件里 preview 截断长度
REJECT_REASON = "知识库无相关内容"

app = FastAPI(title="设备运维知识问答 RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))


def _chunk_count() -> int:
    """知识库 chunk 总数（与 Chroma 入库数量一致，读 chunks.jsonl 行数）。"""
    path = ROOT / _config()["s2_vector"]["input_jsonl"]
    return sum(1 for line in path.open("r", encoding="utf-8") if line.strip())


def _preview(chunk: dict) -> str:
    """chunk 检索文本的前 PREVIEW_CHARS 字，作为前端预览。"""
    text = " ".join(build_search_text(chunk).split())
    return text[:PREVIEW_CHARS] + ("…" if len(text) > PREVIEW_CHARS else "")


def _chunk_distances(question: str, chunk_ids: list[str]) -> dict[str, float]:
    """每个检索 chunk 的向量 cosine distance（低=更近，与拒答阈值同口径）。"""
    ctx = _load_ctx()
    emb = ctx["model"].encode(
        [question], normalize_embeddings=ctx["normalize"], show_progress_bar=False
    )[0]
    got = ctx["collection"].get(ids=chunk_ids, include=["embeddings"])
    id2emb = {cid: e for cid, e in zip(got["ids"], got["embeddings"])}
    return {cid: round(1.0 - float(np.dot(emb, id2emb[cid])), 6) for cid in chunk_ids}


def _chunk_payload(chunk: dict, distance: float) -> dict:
    return {
        "chunk_id": chunk["chunk_id"],
        "heading_path": chunk.get("heading_path"),
        "doc_id": chunk.get("doc_id"),
        "chunk_type": chunk.get("chunk_type"),
        "distance": distance,
        "preview": _preview(chunk),
    }


def _event(name: str, data: dict) -> dict:
    """构造 SSE 事件。sse-starlette 对 dict 走 str()（Python repr，非法 JSON），
    故先 json.dumps 成字符串。"""
    return {"event": name, "data": json.dumps(data, ensure_ascii=False)}


def _sse_events(question: str):
    """按序产出 SSE 事件：retrieval → (rejected | token* + verification + done) / error。"""
    t0 = time.perf_counter()
    try:
        # 1. 检索（复用 graph.retrieve）
        chunks = retrieve({"question": question})["retrieved"]
        dists = _chunk_distances(question, [c["chunk_id"] for c in chunks])
        yield _event(
            "retrieval",
            {"chunks": [_chunk_payload(c, dists[c["chunk_id"]]) for c in chunks]},
        )

        # 2. 拒答闸（复用 graph 的阈值语义）
        top1 = vector_top1_distance(question)
        if top1 > TAU_DISTANCE:
            yield _event(
                "rejected",
                {"reason": REJECT_REASON, "top1_distance": top1, "tau": TAU_DISTANCE},
            )
            return

        # 3. 生成（stream=True 逐块转发；graph.generate 为 invoke，此处不复用）
        context = "\n\n".join(f"[{c['chunk_id']}]\n{build_search_text(c)}" for c in chunks)
        user = f"问题：{question}\n\n检索到的内容：\n{context}"
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)]

        llm = get_llm()
        answer_parts: list[str] = []
        usage: dict = {}
        for chunk in llm.stream(messages, max_tokens=MAX_TOKENS, stream_usage=True):
            text = chunk.content or ""
            if text:
                answer_parts.append(text)
                yield _event("token", {"text": text})
            um = getattr(chunk, "usage_metadata", None)
            if um:
                usage = um
        answer = "".join(answer_parts).strip()

        # 4. 引用校验（复用 graph.verify_node）
        verification = verify_node({"answer": answer, "retrieved": chunks})["verification"]
        yield _event(
            "verification",
            {
                "suspicious_count": verification.get("suspicious_count", 0),
                "suspicious": verification.get("suspicious", []),
                "cited_ids": verification.get("cited_ids", []),
                "fabricated_ids": verification.get("fabricated_ids", []),
            },
        )

        # 5. 结束
        total = usage.get("total_tokens")
        if total is None:
            total = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
        yield _event(
            "done",
            {"total_tokens": total, "elapsed_ms": int((time.perf_counter() - t0) * 1000)},
        )
    except Exception as exc:  # LLM 失败等一律显式推送，不静默
        yield _event("error", {"message": f"{type(exc).__name__}: {exc}"})


class AskRequest(BaseModel):
    question: str


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "chunk_count": _chunk_count(),
        "model_name": LLM_CONFIG["model"],
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    return EventSourceResponse(_sse_events(req.question))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
