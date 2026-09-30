"""评审灵敏度测试：往 dev 的答案里注入已知错误，看评审模型能查出多少。

评审模型在正式评测里把绝大多数句子判为「有依据」。这可能是答案确实忠实，也可能是评审
太宽松——只看正式评测分不出来。本脚本对每个实际作答的 dev 答案构造三种篡改：
    number     把一句话里的一个数字改成检索片段里没有的数
    antonym    把一个关键词换成反义词（关闭↔打开、顺时针↔逆时针、至少↔至多 …）
    fabricate  在末尾追加一句手册里没有的操作要求
然后用同一套评审提示词重新判，统计被篡改的那一句有多少被判为 none / partial（检出）。
同时对照确定性的「数值溯源」规则能否发现 number 类篡改。
未篡改句子上的误报情况直接看正式报告（同一批答案）。

只用 dev：test 不参与任何调试。

用法：
    python src/answer_eval/probe.py [--run baseline] [--limit 40]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import ROOT, Corpus, load_config, load_items, load_split  # noqa: E402
from judge import Judge, run_jobs  # noqa: E402
from textnorm import numbers_in  # noqa: E402

KIND_CN = {"number": "改数字", "antonym": "换反义词", "fabricate": "追加编造句"}
ANTONYMS = [("关闭", "打开"), ("顺时针", "逆时针"), ("至少", "至多"), ("不得", "必须"), ("高于", "低于"),
            ("大于", "小于"), ("之前", "之后"), ("增加", "减少"), ("拧紧", "拧松"), ("最大", "最小"),
            ("左侧", "右侧"), ("提高", "降低"), ("添加", "取出"), ("禁止", "允许"), ("不要", "务必"),
            ("超过", "低于"), ("启动", "停止")]
FABRICATED = [
    "安装前必须将电动机加热到80摄氏度并保温两小时",
    "每次启动后都要用煤油冲洗叶轮至少10分钟",
    "地脚螺栓必须使用12.9级高强度螺栓并涂抹螺纹锁固胶",
    "累计运行满500小时后必须更换全部轴承",
    "首次运行前应向泵腔内注入2升防冻液",
]
# 可篡改的数字：不取型号里的数字（前面是字母）和句内的列表序号（「1. …；2) …」，改了也不算错）
_NUM_RE = re.compile(r"(?<![\d.A-Za-z_#])\d+(?:\.\d+)?(?![\d]|\.\d)(?!\s*[.)）、])")


def _format_like(old: str, value: float) -> str:
    decimals = len(old.split(".")[1]) if "." in old else 0
    return f"{value:.{decimals}f}"


def perturb_number(sentences: list[dict], allowed: set[float]) -> tuple[int, str, str] | None:
    """返回 (句子下标, 改后的句子, 说明)。新数字保证不在检索片段和问句里出现。"""
    for i, s in enumerate(sentences):
        for m in _NUM_RE.finditer(s["text"]):
            old = float(m.group(0))
            for cand in (old * 2, old + 7, old * 3, old + 13, old * 5 + 1):
                new = _format_like(m.group(0), cand)
                if round(float(new), 6) not in allowed and float(new) != old:
                    text = s["text"][:m.start()] + new + s["text"][m.end():]
                    return i, text, f"{m.group(0)} → {new}"
    return None


def perturb_antonym(sentences: list[dict]) -> tuple[int, str, str] | None:
    for i, s in enumerate(sentences):
        for a, b in ANTONYMS:
            for src, dst in ((a, b), (b, a)):
                if src in s["text"]:
                    return i, s["text"].replace(src, dst, 1), f"{src} → {dst}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="评审灵敏度测试（dev）")
    ap.add_argument("--run", default="")
    ap.add_argument("--limit", type=int, default=0, help="最多用多少条答案（0 为全部）")
    args = ap.parse_args()

    cfg = load_config()
    acfg = cfg["answer_eval"]
    run_dir = ROOT / acfg["out_dir"] / (args.run or acfg["run"])
    split = load_split(ROOT / acfg["split"])
    items = {it["id"]: it for it in load_items(ROOT / acfg["items"])}
    corpus = Corpus.load(cfg)
    scores_path = next((p for p in (run_dir / "scores.jsonl", run_dir / "scores_dev.jsonl") if p.exists()), None)
    if scores_path is None:
        raise SystemExit(f"{run_dir} 下没有评分结果，先运行 python src/answer_eval/run.py --split dev")
    rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if split[r["id"]] == "dev" and r["variant"] == "main"
            and r["behavior"] == "answered" and r["sentence_texts"]]
    if args.limit:
        rows = rows[:args.limit]

    judge = Judge(acfg["judge"], run_dir / "judge_probe_cache.jsonl")
    cases, jobs = [], []
    for n, r in enumerate(rows):
        question = items[r["id"]]["question"]
        sents = [{"text": t, "cited": c} for t, c in zip(r["sentence_texts"], r["sentence_cited"])]
        allowed = {round(x, 6) for x in numbers_in(question)}
        for cid in r["retrieved"]:
            allowed |= corpus.numbers(cid)

        variants = []
        hit = perturb_number(sents, allowed)
        if hit:
            variants.append(("number", *hit))
        hit = perturb_antonym(sents)
        if hit:
            variants.append(("antonym", *hit))
        fab = FABRICATED[n % len(FABRICATED)]
        variants.append(("fabricate", len(sents), fab, "追加：" + fab))

        for kind, idx, text, how in variants:
            modified = [dict(s) for s in sents]
            if kind == "fabricate":
                modified.append({"text": text, "cited": sents[-1]["cited"]})
            else:
                modified[idx]["text"] = text
            case = {"id": r["id"], "kind": kind, "index": idx, "how": how, "sentence": text,
                    "traced": None}
            if kind == "number":      # 确定性规则能否发现：改后的句子里有没有检索片段中不存在的数
                case["traced"] = any(round(x, 6) not in allowed for x in numbers_in(text, answer=True))
            cases.append(case)
            jobs.append(lambda ref=f"{r['id']}#probe-{kind}", q=question, m=modified, ret=r["retrieved"]:
                        judge.faithfulness(ref, q, m, ret, corpus))

    print(f"评审灵敏度测试：{len(rows)} 条 dev 答案，{len(jobs)} 个篡改样本，并发 {acfg['judge']['concurrency']} …")
    t0 = time.perf_counter()
    for case, result in zip(cases, run_jobs(jobs, int(acfg["judge"]["concurrency"]))):
        if result.get("failed"):
            case["verdict"] = "failed"
        else:
            case["verdict"] = result["sentences"][case["index"]]["support"]
    judge.save(prune=not args.limit)
    print(f"完成：新调用 {judge.n_calls} 次，复用缓存 {judge.n_cached} 次，用时 {time.perf_counter() - t0:.0f} 秒\n")

    L = ["# 评审灵敏度测试（dev，注入已知错误）", "",
         f"> 生成脚本：src/answer_eval/probe.py；评审模型 {judge.model}，提示词指纹 {judge.prompt_hash()}；"
         f"取 {len(rows)} 条 dev 上实际作答的答案，各注入一处错误后重新评审。",
         "> 「检出」指被篡改的那一句被评审判为 none 或 partial。篡改是机械生成的，个别改完仍然成立"
         "（比如换了反义词但片段里两种说法都有），所以检出率的上限略低于 100%。", "",
         "| 篡改方式 | 样本数 | 判为 none | 判为 partial | 漏判（full） | 判为 na | 评审失败 | 检出率 |",
         "|---|---|---|---|---|---|---|---|"]
    summary = {}
    for kind in KIND_CN:
        sub = [c for c in cases if c["kind"] == kind]
        v = Counter(c["verdict"] for c in sub)
        valid = len(sub) - v["failed"]
        detected = v["none"] + v["partial"]
        rate = f"{detected / valid:.3f}（{detected}/{valid}）" if valid else "—"
        L.append(f"| {KIND_CN[kind]} | {len(sub)} | {v['none']} | {v['partial']} | {v['full']} | {v['na']} "
                 f"| {v['failed']} | {rate} |")
        summary[kind] = {"n": len(sub), "none": v["none"], "partial": v["partial"], "full": v["full"],
                         "na": v["na"], "failed": v["failed"]}
    num = [c for c in cases if c["kind"] == "number"]
    traced = sum(bool(c["traced"]) for c in num)
    L += ["", f"确定性的数值溯源规则（答案里的数字必须出现在检索片段或问句中）对「改数字」的检出："
              f"{traced}/{len(num)}。", ""]
    summary["number_traced"] = {"k": traced, "n": len(num)}

    missed = [c for c in cases if c["verdict"] in ("full", "na")]
    L += ["## 评审没有查出的篡改", ""]
    if missed:
        for c in missed:
            L.append(f"- `{c['id']}`（{KIND_CN[c['kind']]}，{c['how']}；评审判为 {c['verdict']}）：{c['sentence']}")
    else:
        L.append("（无）")
    L.append("")

    (run_dir / "judge_probe.md").write_text("\n".join(L), encoding="utf-8")
    (run_dir / "judge_probe.json").write_text(
        json.dumps({"summary": summary, "cases": cases}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"已写入 {run_dir / 'judge_probe.md'}")
    if any(c["verdict"] == "failed" for c in cases):
        print("有评审任务失败，结果不完整。", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
