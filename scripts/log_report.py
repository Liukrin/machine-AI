"""请求日志汇总：请求量、拒答、耗时、费用、数值核对、用户反馈，以及最近的 👎 明细（找 badcase 用）。

用法：
    python scripts/log_report.py                    # 全部记录
    python scripts/log_report.py --since 2026-10-01 # 只看某天之后
    python scripts/log_report.py --db logs/requests.db
Docker 部署时在容器里看：docker compose exec api python scripts/log_report.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "s5_app"))

from request_log import RequestLog  # noqa: E402


def pct(xs: list[float], p: float) -> float:
    """线性插值分位数（与评测报告同一算法）。"""
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo = int(k)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (k - lo)


def summarize(rows: list[dict]) -> list[str]:
    L: list[str] = []
    n = len(rows)
    if not n:
        return ["没有记录。"]
    status = Counter(r["status"] for r in rows)
    modes = Counter(r["mode"] for r in rows)
    L.append(f"请求 {n} 次（{rows[0]['created_at']} ～ {rows[-1]['created_at']}）")
    L.append("  状态：" + "，".join(f"{k} {v}" for k, v in status.most_common()))
    L.append("  模式：" + "，".join(f"{k} {v}" for k, v in modes.most_common()))

    done = [r for r in rows if r["status"] == "done"]
    if done:
        refused = sum(bool(r["refused"]) for r in done)
        L.append(f"  拒答：模型拒答 {refused}/{len(done)}，拒答闸拦下 {status.get('rejected', 0)}")
        ms = [r["elapsed_ms"] for r in done if r["elapsed_ms"] is not None]
        if ms:
            L.append(f"  耗时：P50 {pct(ms, 0.5):.0f} ms，P95 {pct(ms, 0.95):.0f} ms")
        cost = [r["cost_yuan"] for r in done if r["cost_yuan"] is not None]
        if cost:
            L.append(f"  费用：合计 ¥{sum(cost):.4f}，每次平均 ¥{sum(cost) / len(cost):.5f}（按 config llm_pricing 估算）")
        checked = [r for r in done if r["numbers_checked"]]
        if checked:
            bad = sum(1 for r in checked if r["number_issues"])
            rep = Counter(r["repaired"] for r in checked if r["repaired"])
            L.append(f"  数值核对：{len(checked)} 次回答含数值，最终仍有对不上的 {bad} 次；"
                     f"改写 {sum(rep.values())} 次（采用改写稿 {rep.get('repaired', 0)}，退回初稿 {rep.get('draft', 0)}）")

    rated = [r for r in rows if r["rating"] is not None]
    up = sum(r["rating"] == 1 for r in rated)
    L.append(f"  反馈：{len(rated)}/{n} 次有评价，👍 {up}，👎 {len(rated) - up}")
    downs = [r for r in rated if r["rating"] == -1]
    if downs:
        L.append("\n最近的 👎（最多 10 条）：")
        for r in downs[-10:]:
            tags = [r["mode"] or "", "拒答" if r["refused"] or r["status"] == "rejected" else "",
                    f"数值对不上 {r['number_issues']}" if r["number_issues"] else ""]
            L.append(f"- [{r['created_at']}] {r['question'][:60]}（{'，'.join(t for t in tags if t)}）")
            if r["comment"]:
                L.append(f"    说明：{r['comment']}")
            cited = json.loads(r["cited"]) if r["cited"] else []
            L.append(f"    请求 {r['id']}，引用 {'、'.join(cited) or '无'}")
    return L


def main() -> None:
    ap = argparse.ArgumentParser(description="请求日志汇总")
    ap.add_argument("--db", default="", help="数据库路径；默认 config s5_app.request_log")
    ap.add_argument("--since", default="", help="只看这之后的记录（ISO 日期或时间，如 2026-10-01）")
    args = ap.parse_args()
    if args.db:
        path = Path(args.db)
    else:
        cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
        rel = (cfg.get("s5_app") or {}).get("request_log")
        if not rel:
            raise SystemExit("config s5_app.request_log 为空：请求日志没有开启。")
        path = ROOT / rel
    if not path.exists():
        raise SystemExit(f"还没有日志：{path}（起服务、问过问题后才会生成）")
    print("\n".join(summarize(RequestLog(path).rows(args.since or None))))


if __name__ == "__main__":
    main()
