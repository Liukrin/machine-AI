"""请求日志（src/s5_app/request_log.py）与汇总脚本（scripts/log_report.py）。"""
import importlib.util

from conftest import ROOT
from request_log import RequestLog, new_request_id


def _row(rid: str, **kw) -> dict:
    return {"id": rid, "created_at": "2026-10-01T10:00:00+08:00", "mode": "agent", "question": "问题", "status": "done", **kw}


def test_record_and_feedback(tmp_path):
    path = tmp_path / "logs" / "requests.db"
    log = RequestLog(path)
    assert not path.exists() and log.rows() == []             # 第一次写入才建库（导入 api.py 不留空库）
    rid = new_request_id()
    log.record(_row(rid, answer="答案", sources=["1_c0001", "1_c0002"], cited=["1_c0001"], refused=False))
    r = log.rows()[0]
    assert (r["sources"], r["cited"], r["refused"], r["rating"]) == ('["1_c0001", "1_c0002"]', '["1_c0001"]', 0, None)

    assert log.feedback(rid, -1, "  数值写错了 ")
    r = log.rows()[0]
    assert (r["rating"], r["comment"]) == (-1, "数值写错了") and r["rated_at"]
    assert log.feedback(rid, None, "撤销时说明一起清掉")
    assert (log.rows()[0]["rating"], log.rows()[0]["comment"]) == (None, None)
    assert not log.feedback("没有这个编号", 1)


def test_disabled_log_does_nothing():
    log = RequestLog(None)
    log.record(_row("x"))
    assert (log.enabled, log.feedback("x", 1), log.rows()) == (False, False, [])


def test_rows_since(tmp_path):
    log = RequestLog(tmp_path / "r.db")
    log.record(_row("old", created_at="2026-09-30T23:00:00+08:00"))
    log.record(_row("new", created_at="2026-10-01T09:00:00+08:00"))
    assert [r["id"] for r in log.rows("2026-10-01")] == ["new"]


def test_log_report(tmp_path):
    spec = importlib.util.spec_from_file_location("log_report", ROOT / "scripts" / "log_report.py")
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)

    log = RequestLog(tmp_path / "r.db")
    log.record(_row("a", elapsed_ms=1000, cost_yuan=0.01, numbers_checked=3, number_issues=0, repaired="repaired"))
    log.record(_row("b", elapsed_ms=3000, cost_yuan=0.02, refused=True, cited=["4_c0151"]))
    log.record(_row("c", status="rejected", mode="rag"))
    log.feedback("a", 1)
    log.feedback("b", -1, "手册里其实有")
    text = "\n".join(report.summarize(log.rows()))
    for expected in ("请求 3 次", "模型拒答 1/2，拒答闸拦下 1", "P50 2000 ms", "合计 ¥0.0300",
                     "改写 1 次（采用改写稿 1，退回初稿 0）", "👍 1，👎 1", "说明：手册里其实有", "引用 4_c0151"):
        assert expected in text, expected
