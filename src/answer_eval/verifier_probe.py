"""校验器灵敏度测试：把回答里核对得上的数改错数字、改错单位，看数值核对（src/s4_agent/numcheck.py）能拦下多少，
并与旧的引用校验（verify.py：句子与片段的二元组重合度）对比。不调用大模型。

资料按评测记录重建，与线上核对时用的资料同构：用户的话（问题 + 多轮历史里的提问）、本次检索到的片段
（说明行 + 正文，表格按行展开）、换算/核对工具的输出。

扰动（每个回答至多各一处，确定性选取，选取时不看核对器的判断，免得「拦截率」变成循环论证）：
- 改数字：回答里第一个核对得上、且不是 10 以下整数的数，改成它的 2 倍；
- 改单位：回答里第一个带单位且核对得上的数，换成同量纲的另一个单位（按 SWAP 的顺序取第一个不同的）。
改过的数一定是错的；核对器没拦下，多半是改后的数恰好在资料别处出现过，或资料里这个数没有单位可对。
原始回答被数值核对标出问题的比例，是误报的上限（被标出的不一定都是误报，报告里列出逐条供人看）。

用法：
    python src/answer_eval/verifier_probe.py                       # 默认读 agent_v1_flash 的主集和任务集
    python src/answer_eval/verifier_probe.py --run agent_v2
产物：eval/answer_eval/<评测名>/verifier_probe.md
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).resolve().parents[2]
for _p in ("src", "src/s3_eval", "src/s4_agent", "src/answer_eval"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

from agent import _catalog  # noqa: E402
from dataset import eval_set, load_config, load_items  # noqa: E402
from numcheck import _CITATION, _NUMBER, Quantity, Source, _unit_after, check_answer, ground, unit_id  # noqa: E402
from tools import chunk_source, get_corpus  # noqa: E402
from verify import verify_citation  # noqa: E402

CALC_TOOLS = ("convert_unit", "check_value")
REFUSAL_LEAD = "知识库无相关内容"
SWAP = {   # 改单位时依次尝试的同量纲单位
    "长度": ["mm", "cm", "m", "in"], "压力": ["MPa", "bar", "kPa", "psi"], "流量": ["m³/h", "L/min", "L/s"],
    "温度": ["°C", "°F"], "扭矩": ["N·m", "kgf·m", "daN·m"], "功率": ["kW", "hp"], "转速": ["r/min", "r/s"],
    "质量": ["kg", "g", "t"], "力": ["N", "kN", "daN"], "体积": ["L", "m³", "mL"], "时间": ["h", "min", "s"],
    "线速度": ["m/s", "ft/s"], "比例": ["%", "‰"],
}


def sources_for(rec: dict, item: dict) -> tuple[list[Source], list[dict], str]:
    """评测记录 → (数值核对用的资料, 旧校验器用的片段, 旧校验器用的换算/核对文本)。"""
    corpus = get_corpus()
    user = [h["content"] for h in (item.get("history") or []) if h.get("role") == "user"] if rec["variant"] == "main" else []
    srcs = [Source("\n".join([*user, rec["question"]])), Source(_catalog())]
    cids = [r["chunk_id"] for r in rec.get("retrieved") or [] if r["chunk_id"] in corpus.by_id]
    srcs += [chunk_source(cid) for cid in cids]
    calc = [s["output"] for s in rec.get("steps") or [] if s.get("ok") and s.get("name") in CALC_TOOLS and s.get("output")]
    srcs += [Source(t, kind="calc") for t in calc]
    return srcs, [corpus.by_id[cid] for cid in cids], "\n".join(calc)


def candidates(answer: str) -> list[tuple[int, int, Quantity, int, int]]:
    """回答里可改的数：(数字起, 数字止, Quantity, 单位起, 单位止)。跳过引用编号里的数字。"""
    cites = [m.span() for m in _CITATION.finditer(answer)]
    out = []
    for m in _NUMBER.finditer(answer):
        if any(a <= m.start() < b for a, b in cites):
            continue
        unit, end = _unit_after(answer, m.end())
        q = Quantity(float(m.group()), m.group(), unit, unit_id(unit))
        ustart = end - len(unit) if unit else m.end()
        out.append((m.start(), m.end(), q, ustart, end))
    return out


def _fmt_like(x: float, num: str) -> str:
    d = len(num.split(".")[1]) if "." in num else 0
    return f"{x:.{d}f}"


def flagged(answer: str, srcs: list[Source], num: str) -> bool:
    """改过的回答里，这个数被标出了问题（按数字比，问题清单里的单位是规整后的写法，如 m3/h）。"""
    _, issues = check_answer(answer, srcs)
    return any(i.quantity.split(" ")[0] == num for i in issues)


def probe(rec: dict, item: dict) -> dict | None:
    answer = rec.get("answer") or ""
    if rec.get("status") != "answered" or not answer or answer.lstrip().startswith(REFUSAL_LEAD):
        return None
    srcs, evidence, calc = sources_for(rec, item)
    n, issues = check_answer(answer, srcs)
    old = verify_citation(answer, evidence, extra_grounding=calc).get("suspicious_count", 0)
    out = {"key": f"{rec['id']}@{rec['variant']}", "numbers": n, "issues": [i.describe() for i in issues],
           "num": None, "unit": None}
    cands = [c for c in candidates(answer) if ground(c[2], srcs).status == "ok"]
    # 改数字：第一个可改的数 ×2
    for a, b, q, *_ in cands:
        if q.uid is None and "." not in q.num and q.value < 10:
            continue
        new = _fmt_like(q.value * 2, q.num)
        pert = answer[:a] + new + answer[b:]
        old2 = verify_citation(pert, evidence, extra_grounding=calc).get("suspicious_count", 0)
        out["num"] = {"from": q.show(), "to": f"{new} {q.unit}" if q.unit else new,
                      "new": flagged(pert, srcs, new), "old": old2 > old}
        break
    # 改单位：第一个带单位的数，换成同量纲的另一个单位
    for a, b, q, ua, ub in cands:
        alt = next((u for u in SWAP.get(q.dim, []) if unit_id(u) not in (None, q.uid)), None) if q.uid else None
        if not alt:
            continue
        pert = answer[:ua] + alt + answer[ub:]
        old2 = verify_citation(pert, evidence, extra_grounding=calc).get("suspicious_count", 0)
        out["unit"] = {"from": q.show(), "to": f"{q.num} {alt}", "new": flagged(pert, srcs, q.num), "old": old2 > old}
        break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="校验器灵敏度测试（改数字、改单位）")
    ap.add_argument("--run", default="agent_v1_flash")
    args = ap.parse_args()
    cfg = load_config()
    acfg = cfg["answer_eval"]
    run_dir = ROOT / acfg["out_dir"] / args.run
    rows = []
    for set_name in ("main", "tasks"):
        es = eval_set(acfg, set_name)
        items = {it["id"]: it for it in load_items(ROOT / es["items"])}
        path = (run_dir / es["subdir"] if es["subdir"] else run_dir) / "answers.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                r = probe(rec, items[rec["id"]])
                if r:
                    r["set"] = set_name
                    rows.append(r)

    def rate(xs, key):
        hit = sum(1 for x in xs if x[key])
        return f"{hit}/{len(xs)}（{hit / max(1, len(xs)):.0%}）"

    nums = [r["num"] for r in rows if r["num"]]
    units = [r["unit"] for r in rows if r["unit"]]
    flagged_orig = [r for r in rows if r["issues"]]
    L = ["# 校验器灵敏度测试", "",
         f"> 生成脚本：src/answer_eval/verifier_probe.py；数据：{args.run} 的主集与任务集里实际作答的回答（不含拒答），"
         "资料按评测记录重建。不调用大模型。", "",
         f"- 回答数：{len(rows)}，共核对数值 {sum(r['numbers'] for r in rows)} 个",
         f"- 原始回答被数值核对标出问题：{len(flagged_orig)}/{len(rows)}（误报的上限，逐条见文末）", "",
         "| 扰动 | 份数 | 数值核对（numcheck）拦下 | 旧的引用校验（verify.py）拦下 |", "|---|---|---|---|",
         f"| 改数字（改成 2 倍） | {len(nums)} | {rate(nums, 'new')} | {rate(nums, 'old')} |",
         f"| 改单位（换成同量纲的另一个单位） | {len(units)} | {rate(units, 'new')} | {rate(units, 'old')} |", ""]
    miss_units = [u for u in units if not u["new"]]
    if miss_units:
        L += ["## 改单位没拦下的", "",
              "资料里这个数没带单位、所在表格行也没写单位、同一段资料也没提到同量纲的单位时，无从核对（宁可漏判，不误判）。", ""]
        L += [f"- {u['from']} → {u['to']}" for u in miss_units[:20]]
        L.append("")
    if flagged_orig:
        L += ["## 原始回答被标出的问题（需人工判断是否误报）", ""]
        for r in flagged_orig:
            L.append(f"- {r['key']}：" + "；".join(r["issues"]))
        L.append("")
    out = run_dir / "verifier_probe.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"报告：{out}")


if __name__ == "__main__":
    main()
