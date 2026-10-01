"""HTTP 接口（src/s5_app/api.py）：请求日志的包装与 /api/feedback。

不带标记的用例把事件流换成固定的几条事件，不跑启动预加载，CI 上也能跑；
kb 用例跑启动预加载和完整的 Agent 图，大模型用剧本代替（不花钱）。
"""
import json
import time

import anyio
import pytest
from fastapi.testclient import TestClient

import api
from request_log import RequestLog


def ev(name: str, data: dict) -> dict:
    return api._event(name, data)


def collect(agen) -> list[dict]:
    """把 _logged（异步生成器）跑完，收下全部事件。"""
    async def run():
        return [e async for e in agen]
    return anyio.run(run)


def parse_sse(text: str) -> list[tuple[str, dict]]:
    out = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        name, data = None, []
        for line in block.split("\n"):
            if line.startswith("event:"):
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip())
        if name:
            out.append((name, json.loads("\n".join(data))))
    return out


@pytest.fixture
def log(tmp_path, monkeypatch):
    lg = RequestLog(tmp_path / "requests.db")
    monkeypatch.setattr(api, "REQUEST_LOG", lg)
    return lg


AGENT_EVENTS = [
    ev("step", {"type": "llm_start", "round": 1, "forced": False}),
    ev("retrieval", {"chunks": [{"chunk_id": "4_c0151"}, {"chunk_id": "4_c0152"}]}),
    ev("token", {"text": "先查一下表。", "round": 1}),            # 调用工具前的说明，不是答案
    ev("token", {"text": "最终答案", "round": 2}),
    ev("verification", {"suspicious_count": 0, "suspicious": [], "cited_ids": ["4_c0151"], "fabricated_ids": [],
                        "refused": False, "numbers_checked": 3, "number_issues": [], "repair": {"kept": "repaired"}}),
    ev("done", {"mode": "agent", "answer": "最终答案 [4_c0151]", "elapsed_ms": 2100, "llm_calls": 3, "tool_calls": 2,
                "input_tokens": 3000, "output_tokens": 200, "cost_yuan": 0.0076, "finish_reason": "stop",
                "model_name": "deepseek-flash"}),
]


def test_logged_agent_stream(log):
    history = [{"role": "user", "content": "上一问"}, {"role": "assistant", "content": "上一答"}]
    out = collect(api._logged("rid1", "这一问", history, None, iter(AGENT_EVENTS)))
    assert [e["event"] for e in out] == [e["event"] for e in AGENT_EVENTS]
    assert json.loads(out[-1]["data"])["request_id"] == "rid1"            # 只加在结束事件上
    assert not any("request_id" in json.loads(e["data"]) for e in out[:-1])

    r = log.rows()[0]
    assert (r["status"], r["mode"], r["question"], r["history_turns"]) == ("done", "agent", "这一问", 1)
    assert r["answer"] == "最终答案 [4_c0151]"                             # 以 done.answer 为准，不拼 token
    assert (json.loads(r["sources"]), json.loads(r["cited"]), r["fabricated"]) == (["4_c0151", "4_c0152"], ["4_c0151"], 0)
    assert (r["numbers_checked"], r["number_issues"], r["repaired"], r["refused"]) == (3, 0, "repaired", 0)
    assert (r["elapsed_ms"], r["llm_calls"], r["tool_calls"], r["cost_yuan"]) == (2100, 3, 2, 0.0076)


def test_logged_rag_stream_joins_tokens(log):
    events = [ev("retrieval", {"chunks": [{"chunk_id": "1_c0001"}]}),
              ev("token", {"text": "知识库无相关内容"}), ev("token", {"text": "。"}),
              ev("verification", {"suspicious_count": 0, "suspicious": [], "cited_ids": [], "fabricated_ids": []}),
              ev("done", {"mode": "rag", "total_tokens": 900, "input_tokens": 850, "output_tokens": 50,
                          "elapsed_ms": 1500, "finish_reason": "stop"})]
    collect(api._logged("rid2", "问题", [], "rag", iter(events)))
    r = log.rows()[0]
    # rag 模式的 verification 没有拒答判定，按同一口径（拒答话术且没有引用）补上
    assert (r["mode"], r["answer"], r["refused"], r["numbers_checked"]) == ("rag", "知识库无相关内容。", 1, None)


def test_logged_rejected_and_error(log):
    collect(api._logged("r-rej", "q", [], "rag",
                        iter([ev("rejected", {"reason": "知识库无相关内容", "top1_distance": 0.5, "tau": 0.3567})])))
    collect(api._logged("r-err", "q", [], None, iter([ev("error", {"message": "APITimeoutError: timed out"})])))
    rows = {r["id"]: r for r in log.rows()}
    assert rows["r-rej"]["status"] == "rejected" and rows["r-rej"]["elapsed_ms"] is not None
    assert (rows["r-err"]["status"], rows["r-err"]["error"]) == ("error", "APITimeoutError: timed out")


def test_logged_records_client_disconnect(log):
    """客户端中途断开 = sse-starlette 取消正在等下一个事件的任务。部署验证时发现：同步生成器在这种情况下不会被关闭，
    断开的请求一条都没记下来；改成异步生成器后，等手头这一步跑完，finally 照样写一行 aborted。"""
    def slow_events():
        yield AGENT_EVENTS[0]
        time.sleep(0.6)                      # 模型还在生成
        yield from AGENT_EVENTS[1:]

    async def run():
        with anyio.move_on_after(0.2):       # 0.2 秒后取消，相当于客户端断开
            async for _ in api._logged("r-abort", "q", [], None, slow_events()):
                pass
    anyio.run(run)
    r = log.rows()[0]
    assert (r["id"], r["status"], r["answer"]) == ("r-abort", "aborted", None) and r["elapsed_ms"] >= 200


def test_ask_then_feedback(log, monkeypatch):
    monkeypatch.setattr(api, "_sse_events", lambda question, history, mode: iter(AGENT_EVENTS))
    client = TestClient(api.app)              # 不用 with：不跑启动预加载（CI 上没有知识库）
    res = client.post("/api/ask", json={"question": "Model 3700 轴承温度上限是多少？"})
    assert res.status_code == 200 and res.headers["content-type"].startswith("text/event-stream")
    name, done = parse_sse(res.text)[-1]
    assert name == "done" and done["answer"] == "最终答案 [4_c0151]"

    rid = done["request_id"]
    assert client.post("/api/feedback", json={"request_id": rid, "rating": -1, "comment": "上限写错了"}).json() == {"ok": True}
    assert (log.rows()[0]["rating"], log.rows()[0]["comment"]) == (-1, "上限写错了")
    assert client.post("/api/feedback", json={"request_id": "nope", "rating": 1}).status_code == 404
    assert client.post("/api/feedback", json={"request_id": rid, "rating": 5}).status_code == 422


def test_log_disabled(monkeypatch):
    monkeypatch.setattr(api, "REQUEST_LOG", RequestLog(None))
    monkeypatch.setattr(api, "_sse_events", lambda question, history, mode: iter(AGENT_EVENTS))
    client = TestClient(api.app)
    _, done = parse_sse(client.post("/api/ask", json={"question": "q"}).text)[-1]
    assert "request_id" not in done                                       # 前端据此不显示 👍/👎
    assert client.post("/api/feedback", json={"request_id": "x", "rating": 1}).status_code == 503


@pytest.mark.kb
def test_ask_through_agent_graph(log, fake_llm):
    answer = "Model 3700 的轴承温度应介于 49°C | 120°F 和 82°C | 180°F 之间 [4_c0151]。"
    fake_llm([answer])
    with TestClient(api.app) as client:      # with：跑启动预加载（模型、索引、Agent 图）
        assert client.get("/api/health").json()["status"] == "ok"
        events = parse_sse(client.post("/api/ask", json={"question": "Model 3700 的轴承温度正常范围是多少？"}).text)
    names = [n for n, _ in events]
    assert names[0] == "step" and "retrieval" in names and names[-2:] == ["verification", "done"]
    verification, done = events[-2][1], events[-1][1]
    assert verification["number_issues"] == [] and verification["fabricated_ids"] == []
    assert (done["answer"], done["llm_calls"], done["model_name"]) == (answer, 1, "fake-llm")
    r = log.rows()[0]
    assert (r["id"], r["status"], r["answer"]) == (done["request_id"], "done", answer)
    assert "4_c0151" in json.loads(r["sources"])
