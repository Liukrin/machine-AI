"""答案级评测：一条命令跑完「取系统输出 → 确定性评分 → 评审模型 → 报告」。

评的是答得对不对：关键事实与数值、忠实度与引用、该拒的拒没拒、不该拒的有没有误拒，
外加耗时与 token。被测系统是线上问答流水线本身（见 runner.py）。

用法：
    python src/answer_eval/run.py                  # dev + test 全量
    python src/answer_eval/run.py --split dev      # 只评 dev（调试评测脚本、迭代系统时用）
    python src/answer_eval/run.py --run agent-v1   # 系统改动后换个评测名，结果放到单独目录，便于前后对比
    python src/answer_eval/run.py --reuse          # 只用缓存重算指标与报告，零 LLM 调用
    python src/answer_eval/run.py --reuse --check  # 只复算、与已有 metrics.json 比对，不写文件（pytest 用它核对已提交的评测）
    python src/answer_eval/run.py --ids d2_cross_01,mt_03   # 只评指定题目（调试用）
    python src/answer_eval/run.py --no-judge       # 跳过评审模型，只出确定性指标
    python src/answer_eval/run.py --set tasks      # 换一个评测集（Agent 多步任务集，见 config answer_eval.sets）
    python src/answer_eval/run.py --mode rag       # 被测系统的模式：agent（默认，config agent.default_mode）或 rag

产物在 eval/answer_eval/<评测名>/：report.md、metrics.json、details.csv、scores.jsonl，
以及可复算用的 answers.jsonl（系统原始输出）、judgments.jsonl（评审原始输出）、system.json。
其他评测集的产物放在同名子目录里（如 eval/answer_eval/<评测名>/tasks/），system.json 共用。

--reuse / --frozen 时用评测目录里记录的系统指纹（system.json），不按当前代码重算：系统输出只用缓存，
不调用被测系统，所以流水线代码后来改过也能重出旧评测的报告。两者的区别是 --frozen 允许调用评审模型
（比如给旧系统补评一个新评测集），--reuse 连评审也只用缓存。

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
    ROOT, SPLITS, Corpus, eval_set, load_config, load_items, load_split, resolve_evidence, validate,
)
from report import build_report, write_csv, write_json  # noqa: E402
from runner import run_system, system_info  # noqa: E402
from scoring import score  # noqa: E402


def cost_upper(s: dict, pricing: dict) -> float | None:
    """单题成本上限（元）：输入全按未命中缓存计价。两种模式同一口径；被拒答闸拦下的题没有调用模型，记 0。"""
    if s["behavior"] == "rejected_gate":
        return 0.0
    if s.get("input_tokens") is None:
        return None
    return (s["input_tokens"] * pricing["input_cache_miss"] + (s.get("output_tokens") or 0) * pricing["output"]) / 1e6


def recorded_judge_model(set_dir: Path) -> str | None:
    """评测目录里 metrics.json 记录的评审模型名；没有记录（或当时没跑评审）返回 None，用当前模型。"""
    path = set_dir / "metrics.json"
    if not path.exists():
        return None
    name = (json.loads(path.read_text(encoding="utf-8")).get("meta") or {}).get("judge_model")
    return name if name and name != "未运行" else None


def _diff(old, new, path: str = "") -> list[str]:
    """两份 JSON 的差异（字典逐键递归，其余整体比较）。"""
    if isinstance(old, dict) and isinstance(new, dict):
        out: list[str] = []
        for k in sorted(set(old) | set(new), key=str):
            out += _diff(old.get(k, "<缺>"), new.get(k, "<缺>"), f"{path}.{k}" if path else str(k))
        return out
    return [] if old == new else [f"{path}：{old!r} → {new!r}"]


def main() -> None:
    ap = argparse.ArgumentParser(description="答案级评测（一条命令）")
    ap.add_argument("--run", default="", help="评测名（结果目录名）；默认读 config answer_eval.run")
    ap.add_argument("--split", choices=[*SPLITS, "all"], default="all")
    ap.add_argument("--ids", default="", help="只评这些题（逗号分隔）")
    ap.add_argument("--limit", type=int, default=0, help="只评前 N 题（冒烟用）")
    ap.add_argument("--reuse", action="store_true", help="系统输出与评审结果都只用缓存，缺了就报错")
    ap.add_argument("--frozen", action="store_true",
                    help="系统输出只用缓存（按 system.json 记录的指纹），评审模型照常调用")
    ap.add_argument("--force", action="store_true", help="忽略缓存，全部重新调用系统")
    ap.add_argument("--check", action="store_true",
                    help="与 --reuse 同用：只复算，与已有的 metrics.json 逐项比对，不写任何文件；不一致时非零退出")
    ap.add_argument("--no-judge", action="store_true", help="跳过评审模型")
    ap.add_argument("--set", default="main", help="评测集：main（默认）或 config answer_eval.sets 下的名字")
    ap.add_argument("--mode", choices=["agent", "rag"], default=None,
                    help="被测系统的模式；默认 config agent.default_mode。--reuse 时以 system.json 记录的为准")
    args = ap.parse_args()
    if (args.reuse or args.frozen) and args.force:
        raise SystemExit("--reuse / --frozen 与 --force 互斥。")
    if args.check and not args.reuse:
        raise SystemExit("--check 要与 --reuse 同用（只用缓存复算）。")
    cached_only = args.reuse or args.frozen

    import selfcheck
    print(f"评分逻辑自检通过（{selfcheck.run()} 条）")

    cfg = load_config()
    acfg, scfg = cfg["answer_eval"], cfg["s4_halluc_ab"]
    es = eval_set(acfg, args.set)
    items_path, split_path = ROOT / es["items"], ROOT / es["split"]
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
    set_dir = run_dir / es["subdir"] if es["subdir"] else run_dir
    sys_path = run_dir / "system.json"
    if args.check and not sys_path.exists():
        raise SystemExit(f"没有这次评测的记录：{sys_path}")
    if cached_only and sys_path.exists():
        # 只用缓存的系统输出：系统身份以当时记录的指纹为准（代码后来改过也不影响）
        system = json.loads(sys_path.read_text(encoding="utf-8"))
    else:
        system = system_info(corpus.fingerprint(), args.mode or str(cfg["agent"]["default_mode"]))
        if sys_path.exists() and not args.force:
            old = json.loads(sys_path.read_text(encoding="utf-8"))
            if old.get("system_id") != system["system_id"]:
                diff = {k: (old.get(k), v) for k, v in system.items() if old.get(k) != v and k != "system_id"}
                raise SystemExit(
                    f"评测「{run_name}」是用另一个系统版本跑的，差异（旧 → 新）：{diff}。\n"
                    "系统改了就换个 --run 名字另存一份，前后才能对比；只想用缓存重出报告请加 --reuse；"
                    "确实要覆盖这次评测请加 --force。")
    print(f"评测名 {run_name}；评测集 {es['name']}（{es['items']}）；划分 {'、'.join(splits)}；题数 {len(selected)}；"
          f"系统指纹 {system['system_id']}（{system.get('mode', 'rag')} 模式，提示词 {system['prompt_version']}，"
          f"top_k={system['top_k']}，max_tokens={system['max_tokens']}）\n")

    # 1. 系统输出（缓存）
    answers = run_system(selected, set_dir / "answers.jsonl", system,
                         reuse_only=cached_only, force=args.force)
    if not (cached_only and sys_path.exists()):
        write_json(sys_path, system)

    # 2. 确定性评分
    pricing = cfg["llm_pricing"]
    scores: list[dict] = []
    for it in selected:
        groups = resolve_evidence(it, corpus)
        variants = ["main"] + (["standalone"] if it["type"] == "multi_turn" and it.get("standalone") else [])
        for variant in variants:
            rec = answers[it["id"] if variant == "main" else f"{it['id']}@{variant}"]
            s = score(it, rec, groups, corpus, acfg, scfg)
            s["cost_upper"] = cost_upper(s, pricing)
            s["_sents"] = [{"text": t, "cited": c} for t, c in zip(s["sentence_texts"], s["sentence_cited"])]
            s["_rec"], s["_item"] = rec, it
            scores.append(s)

    # 3. 评审模型（缓存、并发）
    judge_model = judge_prompt = "未运行"
    if not args.no_judge:
        from judge import Judge, run_jobs

        # --reuse 只查缓存：评审身份沿用这次评测当时记录的模型名（模型后来改过名也能零调用复算）
        judge = Judge(acfg["judge"], set_dir / "judgments.jsonl",
                      model=recorded_judge_model(set_dir) if args.reuse else None)
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
                    ref, s["_rec"]["question"], s["_sents"], s["retrieved"], corpus, calc=s["calc_outputs"]))
                targets.append((s, "judge"))
            if s["expect"] == "answer":
                jobs.append(lambda s=s, ref=ref: judge.facts(ref, s["_item"], s["_rec"]["answer"]))
                targets.append((s, "fact_judge"))
        print(f"\n评审：{len(jobs)} 个任务，并发 {acfg['judge']['concurrency']} …", flush=True)
        t0 = time.perf_counter()
        for (s, field), result in zip(targets, run_jobs(jobs, int(acfg["judge"]["concurrency"]))):
            s[field] = result
        if not args.check:
            judge.save(prune=(args.split == "all" and not partial))
        print(f"评审完成：新调用 {judge.n_calls} 次，复用缓存 {judge.n_cached} 次，"
              f"用时 {time.perf_counter() - t0:.0f} 秒")
    for s in scores:
        for k in ("_sents", "_rec", "_item"):
            s.pop(k)

    # 4. 报告
    by_id = {it["id"]: it for it in items}
    meta = {"run": run_name, "generated_at": time.strftime("%Y-%m-%d %H:%M"), "system": system,
            "items_path": es["items"], "set": es["name"], "set_title": es["title"], "n_items": len(items),
            "judge_model": judge_model, "judge_prompt": judge_prompt,
            "pricing_note": f"输入未命中 {pricing['input_cache_miss']}、命中 {pricing['input_cache_hit']}、"
                            f"输出 {pricing['output']} 元/百万 tokens"}
    L, metrics = build_report(scores, by_id, split, splits, meta)
    suffix = "_partial" if partial else ("" if args.split == "all" else f"_{args.split}")
    if args.check:
        path = set_dir / f"metrics{suffix}.json"
        shown = path.relative_to(ROOT).as_posix()
        old = json.loads(path.read_text(encoding="utf-8"))
        new = json.loads(json.dumps({"meta": meta, "splits": metrics}, ensure_ascii=False))
        same_id = {k: old["meta"].get(k) for k in ("judge_model", "judge_prompt")} == \
                  {k: new["meta"][k] for k in ("judge_model", "judge_prompt")}
        diffs = _diff(old["splits"], new["splits"]) + ([] if same_id else ["meta：评审模型或评审提示词不同"])
        if diffs:
            print(f"\n复算与 {shown} 不一致（{len(diffs)} 处，列前 20 处）：\n" + "\n".join(diffs[:20]))
            sys.exit(1)
        print(f"\n复算一致：{shown} 的 {len(metrics)} 个划分逐项相同（未写任何文件）")
        return
    report_path = set_dir / f"report{suffix}.md"
    report_path.write_text("\n".join(L), encoding="utf-8")
    write_json(set_dir / f"metrics{suffix}.json", {"meta": meta, "splits": metrics})
    write_csv(scores, by_id, split, set_dir / f"details{suffix}.csv")
    (set_dir / f"scores{suffix}.jsonl").write_text(
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
