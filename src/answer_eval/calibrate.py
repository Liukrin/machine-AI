"""人工校准评审模型：导出待标注样本，标完后计算评审与人工的一致率。

评审模型和生成模型是同一个，它给出的忠实度不能直接当结论用；要先抽一批它的判断给人看，
算出一致率（和 Cohen's kappa）之后，才知道这些数字可信到什么程度。

用法：
    python src/answer_eval/calibrate.py export            # 从 dev 抽样，写出两份 CSV
    python src/answer_eval/calibrate.py export --split test --n 60
    python src/answer_eval/calibrate.py score             # 读回填好的 CSV，输出一致率并写 calibration.md

两份 CSV（在 eval/answer_eval/<评测名>/ 下，用 Excel 打开即可）：
    calibration_sentences.csv  逐句：这句话有没有片段依据 —— 在「人工判定」列填 full / partial / none / na
    calibration_facts.csv      逐条关键事实：回答里有没有 —— 在「人工判定」列填 present / absent / contradicted
抽样规则：评审判为 partial / none 的句子、摘抄对不上原文的句子、正则与评审不一致的事实全部入选；
其余句子里一半取字面重合度最低的（改写多、最容易判错），一半随机抽，补足到 --n 条（固定随机种子，可复现）。
已填写的人工判定在重新导出时会保留。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import ROOT, Corpus, load_config, load_items, load_split  # noqa: E402

SENT_HEADERS = ["样本ID", "题目ID", "问题", "句子", "句子标注的引用", "评审判定", "评审给的支持片段", "评审摘抄",
                "摘抄在原文里", "字面重合度", "片段原文", "人工判定", "备注"]
FACT_HEADERS = ["样本ID", "题目ID", "问题", "关键事实", "回答", "正则判定", "评审判定", "人工判定", "备注"]
SENT_LABELS = ("full", "partial", "none", "na")
FACT_LABELS = ("present", "absent", "contradicted")
CHUNK_CHARS = 1200


def load_scores(run_dir: Path, split_name: str, split: dict[str, str]) -> list[dict]:
    for name in ("scores.jsonl", f"scores_{split_name}.jsonl"):
        p = run_dir / name
        if p.exists():
            rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
            return [r for r in rows if split[r["id"]] == split_name and r["variant"] == "main"]
    raise SystemExit(f"{run_dir} 下没有 scores.jsonl，先运行 python src/answer_eval/run.py")


def read_labels(path: Path) -> dict[str, tuple[str, str]]:
    """已有 CSV 里填过的人工判定：样本ID -> (判定, 备注)。"""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {r["样本ID"]: (r.get("人工判定", "").strip(), r.get("备注", "")) for r in csv.DictReader(f)
                if r.get("人工判定", "").strip()}


def write_rows(path: Path, headers: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def export(run_dir: Path, split_name: str, n: int, seed: int) -> None:
    cfg = load_config()
    acfg = cfg["answer_eval"]
    items = {it["id"]: it for it in load_items(ROOT / acfg["items"])}
    split = load_split(ROOT / acfg["split"])
    corpus = Corpus.load(cfg)
    scores = load_scores(run_dir, split_name, split)
    rng = random.Random(seed)

    # ---- 逐句
    must, rest = [], []
    for s in scores:
        j = s.get("judge")
        if not j or j.get("failed") or s["behavior"] != "answered":
            continue
        for i, (text, row_cited, ov, r) in enumerate(
                zip(s["sentence_texts"], s["sentence_cited"], s["sentence_overlap"], j["sentences"]), 1):
            chunks = r["by"] or row_cited or s["retrieved"]
            body = "\n---\n".join(f"[{c}] {corpus.text[c][:CHUNK_CHARS]}" for c in chunks if c in corpus.text)
            row = {"样本ID": f"{s['id']}#{i}", "题目ID": s["id"], "问题": items[s["id"]]["question"], "句子": text,
                   "句子标注的引用": " ".join(row_cited), "评审判定": r["support"],
                   "评审给的支持片段": " ".join(r["by"]), "评审摘抄": r["quote"],
                   "摘抄在原文里": "是" if r["quote_ok"] else ("否" if r["quote"] else ""),
                   "字面重合度": "" if ov is None else f"{ov:.2f}",
                   "片段原文": body, "人工判定": "", "备注": ""}
            flagged = r["support"] in ("partial", "none") or (r["support"] == "full" and not r["quote_ok"])
            (must if flagged else rest).append(row)
    # 其余句子里，字面重合度低（改写、归纳多）的一半优先入选，另一半随机抽，兼顾「最可能判错的」和「一般情况」
    rest.sort(key=lambda row: float(row["字面重合度"] or 1))
    quota = max(0, n - len(must))
    low, others = rest[:quota // 2], rest[quota // 2:]
    rng.shuffle(others)
    sent_rows = must + low + others[:quota - len(low)]
    n_flagged = len(must)
    sent_path = run_dir / "calibration_sentences.csv"
    done = read_labels(sent_path)
    for row in sent_rows:
        row["人工判定"], row["备注"] = done.get(row["样本ID"], ("", ""))
    write_rows(sent_path, SENT_HEADERS, sent_rows)

    # ---- 逐条事实
    must, rest = [], []
    for s in scores:
        fj = s.get("fact_judge")
        if s["expect"] != "answer" or s["behavior"] != "answered" or not fj or fj.get("failed"):
            continue
        for i, (f, v) in enumerate(zip(s["facts"], fj["verdicts"]), 1):
            row = {"样本ID": f"{s['id']}@f{i}", "题目ID": s["id"], "问题": items[s["id"]]["question"],
                   "关键事实": f["say"], "回答": s["answer"],
                   "正则判定": "present" if f["matched"] else "absent", "评审判定": v, "人工判定": "", "备注": ""}
            (must if f["matched"] != (v == "present") else rest).append(row)
    rng.shuffle(rest)
    fact_rows = must + rest[:max(0, n - len(must))]
    fact_path = run_dir / "calibration_facts.csv"
    done = read_labels(fact_path)
    for row in fact_rows:
        row["人工判定"], row["备注"] = done.get(row["样本ID"], ("", ""))
    write_rows(fact_path, FACT_HEADERS, fact_rows)

    print(f"逐句样本 {len(sent_rows)} 条（其中评审判为 partial/none 或摘抄对不上的 {n_flagged} 条）→ {sent_path}")
    print(f"事实样本 {len(fact_rows)} 条（其中正则与评审不一致的 {len(must)} 条）→ {fact_path}")
    print("在「人工判定」列填写后，运行 python src/answer_eval/calibrate.py score")


def kappa(pairs: list[tuple[str, str]]) -> float | None:
    """Cohen's kappa：扣除碰巧一致之后的一致程度。"""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)


def agreement_block(name: str, pairs: list[tuple[str, str]], labels: tuple[str, ...]) -> list[str]:
    if not pairs:
        return [f"**{name}**：还没有人工判定。", ""]
    k = kappa(pairs)
    agree = sum(a == b for a, b in pairs)
    L = [f"**{name}**：{len(pairs)} 条已标注，一致 {agree} 条（{agree / len(pairs):.3f}），"
         f"Cohen's kappa {'—' if k is None else f'{k:.3f}'}", "",
         "| 机器 \\ 人工 | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    table = Counter(pairs)
    for a in labels:
        L.append(f"| {a} | " + " | ".join(str(table[(a, b)]) for b in labels) + " |")
    L.append("")
    return L


def score(run_dir: Path) -> None:
    L = ["# 评审模型人工校准", "",
         "> 行是机器的判定，列是人工判定。只统计已填写「人工判定」的样本；样本不是随机抽的"
         "（评审判为有问题的句子全部入选），所以这里的一致率不能直接当成全体答案上的准确率。", ""]
    sent_path, fact_path = run_dir / "calibration_sentences.csv", run_dir / "calibration_facts.csv"
    for path, headers in ((sent_path, SENT_HEADERS), (fact_path, FACT_HEADERS)):
        if not path.exists():
            raise SystemExit(f"没有 {path}，先运行 calibrate.py export")

    with sent_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["人工判定"].strip()]
    bad = [r["样本ID"] for r in rows if r["人工判定"].strip() not in SENT_LABELS]
    if bad:
        raise SystemExit(f"calibration_sentences.csv 的人工判定只能填 {SENT_LABELS}，这些样本不合法：{bad[:8]}")
    L += agreement_block("逐句忠实度：评审 vs 人工", [(r["评审判定"], r["人工判定"].strip()) for r in rows], SENT_LABELS)

    with fact_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["人工判定"].strip()]
    bad = [r["样本ID"] for r in rows if r["人工判定"].strip() not in FACT_LABELS]
    if bad:
        raise SystemExit(f"calibration_facts.csv 的人工判定只能填 {FACT_LABELS}，这些样本不合法：{bad[:8]}")
    L += agreement_block("关键事实：评审 vs 人工", [(r["评审判定"], r["人工判定"].strip()) for r in rows], FACT_LABELS)
    binary = [(r["正则判定"], "present" if r["人工判定"].strip() == "present" else "absent") for r in rows]
    L += agreement_block("关键事实：正则 vs 人工（人工的 contradicted 记为未答出）", binary, ("present", "absent"))

    out = run_dir / "calibration.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"已写入 {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="评审模型人工校准")
    ap.add_argument("action", choices=["export", "score"])
    ap.add_argument("--run", default="")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--n", type=int, default=50, help="每份 CSV 的样本数上限")
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()
    acfg = load_config()["answer_eval"]
    run_dir = ROOT / acfg["out_dir"] / (args.run or acfg["run"])
    if args.action == "export":
        export(run_dir, args.split, args.n, args.seed)
    else:
        score(run_dir)


if __name__ == "__main__":
    main()
