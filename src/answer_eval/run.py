"""答案级评测：一条命令跑完「取系统输出 → 确定性评分 → 评审模型 → 报告」。

评的是答得对不对：关键事实与数值、忠实度与引用、该拒的拒没拒、不该拒的有没有误拒，
外加耗时与 token。被测系统是线上问答流水线本身（见 runner.py）。

用法：
    python src/answer_eval/run.py                  # dev + test 全量
    python src/answer_eval/run.py --split dev      # 只评 dev（调试评测脚本、迭代系统时用）
    python src/answer_eval/run.py --run agent-v1   # 系统改动后换个评测名，结果放到单独目录，便于前后对比
    python src/answer_eval/run.py --reuse          # 只用缓存重算指标与报告，零 LLM 调用
    python src/answer_eval/run.py --ids d2_cross_01,mt_03   # 只评指定题目（调试用）
    python src/answer_eval/run.py --no-judge       # 跳过评审模型，只出确定性指标

产物在 eval/answer_eval/<评测名>/：report.md、metrics.json、details.csv、scores.jsonl，
以及可复算用的 answers.jsonl（系统原始输出）、judgments.jsonl（评审原始输出）、system.json。

有系统调用出错或评审失败的题时，以非零码退出，报告里标明结果不完整。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (  # noqa: E402
    ROOT, SPLITS, Corpus, load_config, load_items, load_split, resolve_evidence, validate,
)
from report import build_report, write_csv, write_json  # noqa: E402
from runner import run_system, system_info  # noqa: E402
from scoring import score  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="答案级评测（一条命令）")
    ap.add_argument("--run", default="", help="评测名（结果目录名）；默认读 config answer_eval.run")
    ap.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    ap.add_argument("--ids", default="", help="只评这些题（逗号分隔）")
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 题（冒烟用）")
    ap.add_argument("--reuse", action="store_true", help="系统输出与评审结果都只用缓存，缺了就报错")
    ap.add_argument("--force", action="store_true", help="忽略缓存，全部重新调用系统")
    ap.add_argument("--no-judge", action="store_true", help="跳过评审模型")
    args = ap.parse_args()
    if args.reuse and args.force:
        raise SystemExit("--reuse 与 --force 互斥。")

    import selfcheck
    print(f"评分逻辑自检通过（{selfcheck.run()} 条）")

    cfg = load_config()
    acfg, scfg = cfg["answer_eval"], cfg["s4_halluc_ab"]
    items_path, split_path = ROOT / acfg["items"], ROOT / acfg["split"]
    items = load_items(items_path)
    corpus = Corpus.load(cfg)
    split = load_split(split_path)

    errors, warnings = validate(items, corpus, split)
    if errors:
        print("\n".join(errors))
        raise SystemExit(f"评测集有 {len(errors)} 处错误，先运行 python src/answer_eval/validate.py 修正。")
    for w in warnings:
        print("评测集警告：" + w)

    splits = list(SPLITS) if args.split == "all" else [args.split]
    selected = [it for it in items if split[it["id"]] in splits]
    if args.ids:
        wanted = {x.strip() for x in args.ids.split(",") if x.strip()}
        unknown = wanted - {it["id"] for it in items}
        if unknown:
            raise SystemExit(f"没有这些题：{sorted(unknown)}")
        selected = [it for it in items if it["id"] in wanted]
        splits = sorted({split[it["id"]] for it in selected}, key=SPLITS.index)
    if args.limit:
        selected = selected[:args.limit]
    partial = bool(args.ids or args.limit)

    run_name = args.run or acfg["run"]
    run_dir = ROOT / acfg["out_dir"] / run_name
    system = system_info(corpus.fingerprint())
    sys_path = run_dir / "system.json"
    if sys_path.exists() and not args.force:
        old = json.loads(sys_path.read_text(encoding="utf-8"))
        if old.get("system_id") != system["system_id"]:
            diff = {k: (old.get(k), v) for k, v in system.items() if old.get(k) != v and k != "system_id"}
            raise SystemExit(
                f"评测「{run_name}」是用另一个系统版本跑的，差异（旧 → 新）：{diff}。\n"
                "系统改了就换个 --run 名字另存一份，前后才能对比；确实要覆盖这次评测请加 --force。")
    print(f"评测名 {run_name}；划分 {'、'.join(splits)}；题数 {len(selected)}；系统指纹 {system['system_id']}"
          f"（提示词 {system['prompt_version']}，top_k={system['top_k']}，max_tokens={system['max_tokens']}）\n")

    # 1. 系统输出（缓存）
    answers = run_system(selected, run_dir / "answers.jsonl", system,
                         reuse_only=args.reuse, force=args.force)
    write_json(sys_path, system)

    # 2. 确定性评分
    scores: list[dict] = []
    for it in selected:
        groups = resolve_evidence(it, corpus)
        variants = ["main"] + (["standalone"] if it["type"] == "multi_turn" and it.get("standalone") else [])
        for variant in variants:
            rec = answers[it["id"] if variant == "main" else f"{it['id']}@{variant}"]
            s = score(it, rec, groups, corpus, acfg, scfg)
            s["_sents"] = [{"text": t, "cited": c} for t, c in zip(s["sentence_texts"], s["sentence_cited"])]
            s["_rec"], s["_item"] = rec, it
            scores.append(s)

    # 3. 评审模型（缓存、并发）
    judge_model = judge_prompt = "未运行"
    if not args.no_judge:
        from judge import Judge, run_jobs

        judge = Judge(acfg["judge"], run_dir / "judgments.jsonl")
        judge_model, judge_prompt = judge.model, judge.prompt_hash()
        if args.reuse:
            judge.reuse_only = True
        jobs, targets = [], []
        for s in scores:
            if s["behavior"] != "answered":
                continue
            ref = f"{s['id']}@{s['variant']}"
            if s["_sents"]:
                jobs.append(lambda s=s, ref=ref: judge.faithfulness(
                    ref, s["_rec"]["question"], s["_sents"], s["retrieved"], corpus))
                targets.append((s, "judge"))
            if s["expect"] == "answer":
                jobs.append(lambda s=s, ref=ref: judge.facts(ref, s["_item"], s["_rec"]["answer"]))
                targets.append((s, "fact_judge"))
        print(f"\n评审：{len(jobs)} 个任务，并发 {acfg['judge']['concurrency']} …", flush=True)
        t0 = time.perf_counter()
        for (s, field), result in zip(targets, run_jobs(jobs, int(acfg["judge"]["concurrency"]))):
            s[field] = result
        judge.save(prune=(args.split == "all" and not partial))
        print(f"评审完成：新调用 {judge.n_calls} 次，复用缓存 {judge.n_cached} 次，"
              f"用时 {time.perf_counter() - t0:.0f} 秒")
    for s in scores:
        for k in ("_sents", "_rec", "_item"):
            s.pop(k)

    # 4. 报告
    by_id = {it["id"]: it for it in items}
    meta = {"run": run_name, "generated_at": time.strftime("%Y-%m-%d %H:%M"), "system": system,
            "items_path": acfg["items"], "n_items": len(items),
            "judge_model": judge_model, "judge_prompt": judge_prompt}
    L, metrics = build_report(scores, by_id, split, splits, meta)
    suffix = "_partial" if partial else ("" if args.split == "all" else f"_{args.split}")
    report_path = run_dir / f"report{suffix}.md"
    report_path.write_text("\n".join(L), encoding="utf-8")
    write_json(run_dir / f"metrics{suffix}.json", {"meta": meta, "splits": metrics})
    write_csv(scores, by_id, split, run_dir / f"details{suffix}.csv")
    (run_dir / f"scores{suffix}.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in scores) + "\n", encoding="utf-8")

    print("\n" + "=" * 78)
    print("\n".join(L))
    print("=" * 78)
    print(f"报告：{report_path}")
    if suffix:
        print(f"注意：本次只评了部分题目，产物带 {suffix} 后缀，不覆盖全量报告。")

    n_err = sum(s["behavior"] == "error" for s in scores)
    n_judge_fail = sum(bool((s.get(k) or {}).get("failed")) for s in scores for k in ("judge", "fact_judge"))
    if n_err or n_judge_fail:
        print(f"\n本次有 {n_err} 题系统调用出错、{n_judge_fail} 个评审任务失败，结果不完整。", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
