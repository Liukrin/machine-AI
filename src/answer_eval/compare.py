"""把两次评测的关键指标并排对比（读各自的 metrics*.json，不调用 LLM）。

用法：
    python src/answer_eval/compare.py baseline agent_v1                    # 主评测集，两边都有的划分
    python src/answer_eval/compare.py baseline agent_v1 --set tasks        # Agent 多步任务集
    python src/answer_eval/compare.py baseline agent_v1 --split dev        # 只看 dev

结果打印出来，并写到 eval/answer_eval/compare_<A>_vs_<B>[_<集>].md。
差值只是两次单独运行的点估计之差：题量小，区间宽，几个百分点的差异不能说明孰优孰劣（报告里附了 Wilson 区间）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import ROOT, SPLITS, TYPE_CN, eval_set, load_config  # noqa: E402
from report import wilson  # noqa: E402


def _load(run_dir: Path, split: str) -> tuple[dict, dict] | None:
    """某次评测在某个划分上的指标：优先全量的 metrics.json，没有再找只评了这个划分的 metrics_<split>.json。"""
    for name in ("metrics.json", f"metrics_{split}.json"):
        p = run_dir / name
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if split in data["splits"]:
                return data["meta"], data["splits"][split]
    return None


def _rate(m: dict, key: str, ci: bool = True) -> tuple[str, float | None]:
    v = m.get(key)
    if not isinstance(v, dict) or not v.get("n"):
        return "—", None
    s = f"{v['rate']:.3f}（{v['k']}/{v['n']}）"
    if ci:
        lo, hi = wilson(v["k"], v["n"])
        s += f" [{lo:.2f}, {hi:.2f}]"
    return s, v["rate"]


def _delta(a: float | None, b: float | None, pct: bool = True) -> str:
    if a is None or b is None:
        return "—"
    d = b - a
    return f"{d * 100:+.1f} 个百分点" if pct else f"{d:+.0f}"


def main() -> None:
    ap = argparse.ArgumentParser(description="并排对比两次答案级评测")
    ap.add_argument("a", help="评测名 A（如 baseline）")
    ap.add_argument("b", help="评测名 B（如 agent_v1）")
    ap.add_argument("--set", default="main")
    ap.add_argument("--split", choices=list(SPLITS), default=None)
    args = ap.parse_args()

    acfg = load_config()["answer_eval"]
    es = eval_set(acfg, args.set)
    dirs = [ROOT / acfg["out_dir"] / r / es["subdir"] if es["subdir"] else ROOT / acfg["out_dir"] / r
            for r in (args.a, args.b)]
    splits = [args.split] if args.split else list(SPLITS)

    L = [f"# 评测对比：{args.a} vs {args.b}（{es['title']}）", "",
         "> 生成脚本：src/answer_eval/compare.py；数字取自两次评测各自的 metrics*.json。"
         "比例后的方括号是 Wilson 95% 置信区间；题量小，区间重叠时差值不足以说明优劣。", ""]
    any_split = False
    for sp in splits:
        got = [_load(d, sp) for d in dirs]
        if not all(got):
            continue
        any_split = True
        (ma, a), (mb, b) = got
        sa, sb = ma["system"], mb["system"]
        L += [f"## {sp}", "",
              f"- A = {args.a}：{sa.get('mode', 'rag')} 模式，提示词 {sa['prompt_version']}，系统指纹 {sa['system_id']}",
              f"- B = {args.b}：{sb.get('mode', 'rag')} 模式，提示词 {sb['prompt_version']}，系统指纹 {sb['system_id']}", "",
              f"| 指标 | A：{args.a} | B：{args.b} | B − A |", "|---|---|---|---|"]

        def row(name: str, key: str, ci: bool = True) -> None:
            x, rx = _rate(a, key, ci)
            y, ry = _rate(b, key, ci)
            L.append(f"| {name} | {x} | {y} | {_delta(rx, ry)} |")

        L.append(f"| **应作答的题** | {a['n_answer_items']} | {b['n_answer_items']} | |")
        row("完全答对（关键事实全部答出）", "strict_correct")
        row("证据全部召回", "evidence_all")
        row("误拒（该答却拒答）", "false_refusal")
        L.append(f"| 　其中：拒答闸 / 模型自述 | {a['false_refusal_gate']} / {a['false_refusal_model']} "
                 f"| {b['false_refusal_gate']} / {b['false_refusal_model']} | |")
        row("关键事实召回", "fact_recall", ci=False)
        row("数值事实答对", "numeric", ci=False)
        row("串手册", "wrong_manual", ci=False)
        L.append(f"| **应拒答的题** | {a['n_refuse_items']} | {b['n_refuse_items']} | |")
        row("拒答召回（库外题被拒）", "refusal_recall")
        row("拒答精确率", "refusal_precision", ci=False)
        L.append(f"| **答案质量**（实际作答的答案） | {a['n_answered']} | {b['n_answered']} | |")
        row("忠实度·严格", "faithful_strict", ci=False)
        row("无依据句占比", "unsupported", ci=False)
        row("引用成立", "citation_ok", ci=False)
        row("无引用句占比", "uncited", ci=False)
        row("答案中的数字可溯源（片段或换算/核对工具输出）", "numbers_traced", ci=False)
        lat_a, lat_b = a["latency"], b["latency"]
        L.append(f"| 作答耗时 P50 / P95（ms） | {lat_a['p50']:.0f} / {lat_a['p95']:.0f} | {lat_b['p50']:.0f} / {lat_b['p95']:.0f} | |")
        L.append(f"| 首 token P50（ms） | {lat_a['ttft_p50']:.0f} | {lat_b['ttft_p50']:.0f} | |")
        ta, tb = a["tokens"], b["tokens"]
        L.append(f"| 单题 token 均值（输入 / 输出） | {ta['input_mean']:.0f} / {ta['output_mean']:.0f} "
                 f"| {tb['input_mean']:.0f} / {tb['output_mean']:.0f} | |")
        ca, cb = (a.get("cost") or {}).get("upper_mean"), (b.get("cost") or {}).get("upper_mean")
        if ca is not None and cb is not None:
            L.append(f"| 单题成本均值（元，输入全按未命中计） | {ca:.5f} | {cb:.5f} | ×{cb / ca:.1f} |")
        ag = b.get("agent") or a.get("agent")
        if ag:
            L.append(f"| 模型调用次数均值 | {(a.get('agent') or {}).get('llm_calls_mean', 1.0):.2f} "
                     f"| {(b.get('agent') or {}).get('llm_calls_mean', 1.0):.2f} | |")
        L.append("")

        types = [t for t in (a.get("by_type") or {}) if t in (b.get("by_type") or {})]
        if types:
            L += ["按题型：完全答对 / 误拒", "", f"| 题型 | 题数 | A 完全答对 | B 完全答对 | A 误拒 | B 误拒 |",
                  "|---|---|---|---|---|---|"]
            for t in types:
                x, y = a["by_type"][t], b["by_type"][t]
                L.append(f"| {TYPE_CN.get(t, t)} | {x['n']} | {x['strict_correct']} | {y['strict_correct']} "
                         f"| {x['false_refusal']} | {y['false_refusal']} |")
            L.append("")
    if not any_split:
        raise SystemExit("两次评测没有共同的划分可比（先跑完 run.py）。")

    out = ROOT / acfg["out_dir"] / (f"compare_{args.a}_vs_{args.b}" + (f"_{es['name']}" if es["subdir"] else "")
                                    + (f"_{args.split}" if args.split else "") + ".md")
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"已写入 {out}")


if __name__ == "__main__":
    main()
