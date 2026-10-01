"""S5 任务一：FastAPI 后端。

/api/ask 有两种模式（请求的 mode 字段；不填用 config agent.default_mode）：

- agent（默认）：执行 src/s4_agent/agent.py 的 LangGraph 工具调用图。用 graph.stream 同时订阅三路流：
  messages（模型逐 token 输出）、custom（每个工具调用的开始/结束，图里用 get_stream_writer 发出）、
  values（最终状态：答案、引用校验、用量）。支持对话历史（history 字段），追问由模型结合上文改写后检索。
- rag：阶段 1 基线的固定流水线，复用 graph.py 的节点函数与参数（retrieve / verify_node / SYSTEM_PROMPT /
  MAX_TOKENS / TAU_DISTANCE）：检索 → 向量距离超过阈值直接拒答 → 流式生成 → 引用校验。不看对话历史。

done 事件带 finish_reason，值为 "length" 时表示回答被 max_tokens 截断，前端据此提示。

服务启动时预加载 embedding 模型、BM25、jieba 词典、表格行库和 Agent 图（_warm_up），首个请求不用再等加载。

接口：
    GET  /api/health             健康检查（含知识库文档清单、默认模式）
    POST /api/ask                SSE 流式问答
    GET  /api/chunks/{chunk_id}  单个 chunk 完整内容（前端来源详情按需拉取）

用法：
    python src/s5_app/api.py     # 默认 127.0.0.1:8000
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "s4_agent"))
sys.path.insert(0, str(ROOT / "src" / "s3_eval"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402
from typing import Literal  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from sse_starlette.sse import EventSourceResponse  # noqa: E402

from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage  # noqa: E402
from llm_config import LLM_CONFIG, get_llm  # noqa: E402
from graph import MAX_TOKENS, SYSTEM_PROMPT, TAU_DISTANCE, retrieve, verify_node  # noqa: E402
from agent import CFG as AGENT_CFG, build_agent_graph, estimate_cost, initial_state  # noqa: E402
from tools import get_corpus, hybrid_search  # noqa: E402
from hybrid_retrieval import (  # noqa: E402
    _load_ctx,
    build_search_text,
    vector_top1_distance,
)

PREVIEW_CHARS = 100       # retrieval 事件里 preview 截断长度
REJECT_REASON = "知识库无相关内容"


def _config() -> dict:
    return yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    _warm_up()
    yield


app = FastAPI(title="设备运维知识问答 RAG API", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=(_config().get("s5_app") or {}).get("cors_origins") or [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _chunk_index() -> dict[str, dict]:
    """chunk_id -> chunk（读 chunks.jsonl，与 Chroma 入库同源）。进程内缓存，重建索引后需重启服务。"""
    path = ROOT / _config()["s2_vector"]["input_jsonl"]
    with path.open("r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    return {c["chunk_id"]: c for c in chunks}


@lru_cache(maxsize=1)
def _documents() -> dict[str, dict]:
    """文档展示信息（config s5_app.documents）。"""
    return (_config().get("s5_app") or {}).get("documents") or {}


def _doc_meta(doc_id: str) -> dict:
    """文档展示名；未在 config 登记的 doc_id 直接显示 doc_id。"""
    doc = _documents().get(doc_id) or {}
    title = doc.get("title") or doc_id
    return {"doc_title": title, "doc_short": doc.get("short") or title, "is_sample": bool(doc.get("sample"))}


def _body_text(chunk: dict) -> str:
    """text chunk 的正文：去掉建库时拼在开头的 heading_path 前缀（见 build_chunks.py）。"""
    text = chunk.get("text") or ""
    hp = chunk.get("heading_path")
    prefix = f"{hp}\n\n" if hp else ""
    return text[len(prefix):] if prefix and text.startswith(prefix) else text


def _pages(chunk: dict) -> list[int] | None:
    """page_range 是 MinerU 的 0 起页码，展示时转成 1 起。"""
    pr = chunk.get("page_range") or []
    if len(pr) != 2 or pr[0] is None or pr[1] is None:
        return None
    return [int(pr[0]) + 1, int(pr[1]) + 1]


def _preview(chunk: dict) -> str:
    """前端卡片预览：正文取去掉 heading 前缀后的内容，表格取单元格拼接文本，截断到 PREVIEW_CHARS 字。"""
    if chunk.get("chunk_type") == "table":
        text = build_search_text(chunk)
        hp = chunk.get("heading_path")
        if hp and text.startswith(hp):
            text = text[len(hp):]
    else:
        text = _body_text(chunk)
    text = " ".join(text.split())
    return text[:PREVIEW_CHARS] + ("…" if len(text) > PREVIEW_CHARS else "")


def _chunk_meta(chunk: dict) -> dict:
    return {
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk.get("doc_id"),
        **_doc_meta(chunk.get("doc_id")),
        "chunk_type": chunk.get("chunk_type"),
        "heading_path": chunk.get("heading_path"),
        "pages": _pages(chunk),
    }


def _chunk_distances(question: str, chunk_ids: list[str]) -> dict[str, float]:
    """每个检索 chunk 的向量 cosine distance（低=更近，与拒答阈值同口径）。"""
    ctx = _load_ctx()
    emb = ctx["model"].encode(
        [question], normalize_embeddings=ctx["normalize"], show_progress_bar=False
    )[0]
    got = ctx["collection"].get(ids=chunk_ids, include=["embeddings"])
    id2emb = {cid: e for cid, e in zip(got["ids"], got["embeddings"])}
    return {cid: round(1.0 - float(np.dot(emb, id2emb[cid])), 6) for cid in chunk_ids}


def _chunk_payload(chunk: dict, distance: float | None) -> dict:
    """来源卡片数据。distance 是与检索问句的向量距离；查表、读相邻片段得到的片段没有，为 None。"""
    return {**_chunk_meta(chunk), "distance": distance, "preview": _preview(chunk)}


def _event(name: str, data: dict) -> dict:
    """构造 SSE 事件。sse-starlette 对 dict 走 str()（Python repr，非法 JSON），
    故先 json.dumps 成字符串。"""
    return {"event": name, "data": json.dumps(data, ensure_ascii=False)}


def _sse_events(question: str, history: list[dict] | None = None, mode: str | None = None):
    """SSE 事件流。mode 不填用 config agent.default_mode。评测脚本（src/answer_eval/runner.py）直接调用本函数。"""
    mode = mode or str(AGENT_CFG["default_mode"])
    if mode == "rag":
        yield from _rag_events(question)
    else:
        yield from _agent_events(question, history or [])


def _agent_events(question: str, history: list[dict]):
    """agent 模式的事件流（图的结构见 src/s4_agent/agent.py）：

    step（llm_start / tool_start / tool_end / thought / forced_final）与 retrieval（累计的来源片段）穿插出现，
    token 带 round（第几次模型调用）；之后是 verification、done，出错时是 error。
    某次模型调用如果最后决定调工具，它之前流出的文字会再以 thought 事件发出，前端把它从回答区挪到时间线。
    """
    t0 = time.perf_counter()
    try:
        index = _chunk_index()
        sources: list[dict] = []
        cur_round = 0
        final: dict | None = None
        stream = build_agent_graph().stream(initial_state(question, history),
                                            stream_mode=["messages", "custom", "values"])
        for kind, payload in stream:
            if kind == "messages":
                chunk, meta = payload
                if (meta.get("langgraph_node") == "agent" and isinstance(chunk, AIMessageChunk)
                        and isinstance(chunk.content, str) and chunk.content):
                    yield _event("token", {"text": chunk.content, "round": cur_round})
            elif kind == "custom":
                if payload.get("type") == "llm_start":
                    cur_round = payload["round"]
                yield _event("step", {k: v for k, v in payload.items() if k != "chunks"})
                new = payload.get("chunks") or []
                if payload.get("type") == "tool_end" and new:
                    sources += [_chunk_payload(index[c["chunk_id"]], c["distance"]) for c in new]
                    yield _event("retrieval", {"chunks": sources})
            else:
                final = payload
        if final is None:
            raise RuntimeError("Agent 没有产出最终状态")

        yield _event("verification", final["verification"])
        usage = final["usage"]
        yield _event("done", {
            "mode": "agent",
            "total_tokens": usage["input"] + usage["output"],
            "input_tokens": usage["input"], "output_tokens": usage["output"],
            "cache_read_tokens": usage["cache_read"],
            "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            "finish_reason": final["finish_reason"],
            "llm_calls": final["llm_calls"],
            "tool_calls": sum(1 for s in final["steps"] if s.get("type") == "tool_end"),
            "forced_final": final["forced_final"],
            "model_name": final["model_name"],
            "cost_yuan": round(estimate_cost(usage, _config()["llm_pricing"]), 6),
            "answer": final["answer"],
        })
    except Exception as exc:  # LLM 失败等一律显式推送，不静默
        yield _event("error", {"message": f"{type(exc).__name__}: {exc}"})


def _rag_events(question: str):
    """rag 模式：按序产出 retrieval → (rejected | token* + verification + done) / error。"""
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

        llm = get_llm(max_tokens=MAX_TOKENS)
        answer_parts: list[str] = []
        usage: dict = {}
        finish_reason = None  # "stop" 正常结束；"length" 被 max_tokens 截断
        for chunk in llm.stream(messages, stream_usage=True):
            text = chunk.content or ""
            if text:
                answer_parts.append(text)
                yield _event("token", {"text": text})
            um = getattr(chunk, "usage_metadata", None)
            if um:
                usage = um
            fr = (getattr(chunk, "response_metadata", None) or {}).get("finish_reason")
            if fr:
                finish_reason = fr
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
            {
                "mode": "rag",
                "total_tokens": total,
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
                "finish_reason": finish_reason,
            },
        )
    except Exception as exc:  # LLM 失败等一律显式推送，不静默
        yield _event("error", {"message": f"{type(exc).__name__}: {exc}"})


def _warm_up() -> None:
    """启动时预加载，并空跑一次检索（触发 embedding 模型、Chroma、BM25、jieba 词典的加载）。不调用大模型。

    不做的话这些都推迟到第一个请求里加载，首问的检索多等约 0.6 秒。
    """
    t0 = time.perf_counter()
    _chunk_index()
    get_corpus()
    build_agent_graph()
    hybrid_search("水泵", 1)
    print(f"预加载完成，用时 {(time.perf_counter() - t0) * 1000:.0f} ms", flush=True)


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # 本会话之前的问答（前端按时间顺序传来）；agent 模式只取最近 config agent.history_turns 轮
    history: list[HistoryItem] = Field(default_factory=list, max_length=40)
    mode: Literal["agent", "rag"] | None = None


@app.get("/api/health")
def health() -> dict:
    counts = Counter(c.get("doc_id") for c in _chunk_index().values())
    docs = [{"doc_id": d, **_doc_meta(d), "chunk_count": n} for d, n in counts.items()]
    docs.sort(key=lambda x: (x["is_sample"], x["doc_id"]))
    return {
        "status": "ok",
        "chunk_count": sum(counts.values()),
        "model_name": LLM_CONFIG["model"],
        "default_mode": AGENT_CFG["default_mode"],
        "documents": docs,
    }


@app.get("/api/chunks/{chunk_id}")
def chunk_detail(chunk_id: str) -> dict:
    """单个 chunk 的完整内容：正文去掉 heading 前缀，表格返回 MinerU 原始 HTML（前端解析后渲染）。"""
    chunk = _chunk_index().get(chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail=f"chunk 不存在：{chunk_id}")
    is_table = chunk.get("chunk_type") == "table"
    return {
        **_chunk_meta(chunk),
        "content": None if is_table else _body_text(chunk),
        "table_html": chunk.get("table_html") if is_table else None,
    }


@app.post("/api/ask")
def ask(req: AskRequest):
    history = [h.model_dump() for h in req.history]
    return EventSourceResponse(_sse_events(req.question, history, req.mode))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
