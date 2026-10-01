"""问答请求日志（SQLite）：每次 /api/ask 记一行（问题、模式、答案、耗时、用量、核对结果），前端的 👍/👎 写回同一行。

只记录经 HTTP 进来的请求。答案级评测直接调用 api._sse_events，不经过这里，评测的题不会混进日志。
数据库路径见 config s5_app.request_log（默认 logs/requests.db，不入库；Docker 部署时在 request-logs 卷里），
置空则不记录。每次读写单独开连接：sse-starlette 在线程池里迭代事件流，连接不跨线程共用。

看汇总：python scripts/log_report.py
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id              TEXT PRIMARY KEY,   -- 请求编号：随 done / rejected / error 事件发给前端，反馈时带回
    created_at      TEXT NOT NULL,      -- ISO 8601，带时区
    mode            TEXT,               -- agent | rag
    question        TEXT NOT NULL,
    history_turns   INTEGER,            -- 带了几轮对话历史（一问一答算一轮）
    status          TEXT NOT NULL,      -- done | rejected（rag 模式被拒答闸拦下）| error | aborted（客户端中途断开）
    answer          TEXT,               -- 最终采用的答案（agent 模式取 done.answer，rag 模式为流式文字拼接）
    refused         INTEGER,            -- 模型拒答：含拒答话术且没有引用（与评测、前端同一口径）
    elapsed_ms      INTEGER,
    llm_calls       INTEGER,
    tool_calls      INTEGER,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    cost_yuan       REAL,               -- 按 config llm_pricing 估算（agent 模式）
    finish_reason   TEXT,
    model_name      TEXT,
    sources         TEXT,               -- 本次给出的来源片段 chunk_id（JSON 数组）
    cited           TEXT,               -- 回答引用的 chunk_id（JSON 数组）
    fabricated      INTEGER,            -- 引用了本次没检索到的片段的个数
    numbers_checked INTEGER,            -- 数值核对：核对了几个数
    number_issues   INTEGER,            -- 最终答案里仍核对不上的数
    repaired        TEXT,               -- 改写过时：repaired（采用改写稿）| draft（改得更差，退回初稿）
    error           TEXT,
    rating          INTEGER,            -- 用户反馈：1 👍，-1 👎，NULL 未评
    comment         TEXT,               -- 👎 时可填的说明
    rated_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_created ON requests(created_at);
"""

COLUMNS = ("id", "created_at", "mode", "question", "history_turns", "status", "answer", "refused", "elapsed_ms",
           "llm_calls", "tool_calls", "input_tokens", "output_tokens", "cost_yuan", "finish_reason", "model_name",
           "sources", "cited", "fabricated", "numbers_checked", "number_issues", "repaired", "error")


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def new_request_id() -> str:
    return uuid.uuid4().hex


class RequestLog:
    """path 为 None 时不记录（record 什么都不做，feedback 返回 False）。

    数据库在第一次写入时才建（导入 api.py 的评测脚本、测试不会顺手建出一个空库）。
    """

    def __init__(self, path: Path | None):
        self.path = path
        self._ready = False

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def _connect(self) -> sqlite3.Connection:
        if not self._ready:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=5)) as db:
                db.executescript(SCHEMA)
            self._ready = True
        return sqlite3.connect(self.path, timeout=5)

    def record(self, row: dict) -> None:
        """写入一次问答。row 的键取自 COLUMNS，缺的记 NULL；列表类字段存成 JSON。"""
        if self.path is None:
            return
        values = [json.dumps(row[c], ensure_ascii=False) if isinstance(row.get(c), (list, dict)) else row.get(c)
                  for c in COLUMNS]
        with closing(self._connect()) as db, db:
            db.execute(f"INSERT INTO requests ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                       values)

    def feedback(self, request_id: str, rating: int | None, comment: str | None = None) -> bool:
        """记下用户对某次回答的评价（rating 为 None 表示撤销）。没有这次请求的记录时返回 False。"""
        if self.path is None:
            return False
        comment = (comment or "").strip() or None
        with closing(self._connect()) as db, db:
            cur = db.execute("UPDATE requests SET rating = ?, comment = ?, rated_at = ? WHERE id = ?",
                             (rating, comment if rating is not None else None, now(), request_id))
            return cur.rowcount == 1

    def rows(self, since: str | None = None) -> list[dict]:
        """全部记录（按时间先后）；since 为 ISO 日期/时间前缀时只取这之后的。"""
        if self.path is None or not self.path.exists():
            return []
        with closing(self._connect()) as db:
            db.row_factory = sqlite3.Row
            sql = "SELECT * FROM requests" + (" WHERE created_at >= ?" if since else "") + " ORDER BY created_at"
            return [dict(r) for r in db.execute(sql, (since,) if since else ())]
