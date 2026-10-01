"""校验答案级评测集：结构、证据引文、数值落地、参考答案自洽、dev/test 划分。不调用 LLM。

用法：
    python src/answer_eval/validate.py            # 校验；有错误时以非零码退出
    python src/answer_eval/validate.py --set tasks   # 校验 Agent 多步任务集（config answer_eval.sets）
    python src/answer_eval/validate.py --assign   # 给还没划分的题分配 dev/test 并写回划分文件
    python src/answer_eval/validate.py --show d2_cross_01   # 打印某道题解析出的证据片段
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import (  # noqa: E402
    ROOT, Corpus, assign_split, distribution, eval_set, load_config, load_items, load_split,
    resolve_evidence, validate,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="校验答案级评测集")
    ap.add_argument("--assign", action="store_true", help="给未划分的题分配 dev/test 并写回划分文件")
    ap.add_argument("--show", default="", help="打印某道题解析出的证据片段")
    ap.add_argument("--set", default="main", help="评测集：main（默认）或 config answer_eval.sets 下的名字")
    args = ap.parse_args()

    import selfcheck
    print(f"评分逻辑自检通过（{selfcheck.run()} 条）")

    cfg = load_config()
    es = eval_set(cfg["answer_eval"], args.set)
    items_path, split_path = ROOT / es["items"], ROOT / es["split"]
    items = load_items(items_path)
    corpus = Corpus.load(cfg)
    split = load_split(split_path)

    if args.assign:
        new_split = assign_split(items, split)
        added = len(new_split) - len(split)
        split_path.write_text(json.dumps(new_split, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"划分文件已写入 {split_path}（新分配 {added} 题，已有 {len(split)} 题未动）")
        split = new_split

    if args.show:
        item = next((it for it in items if it["id"] == args.show), None)
        if item is None:
            raise SystemExit(f"没有这道题：{args.show}")
        print(f"{item['id']}  {item['question']}")
        for ev, ids in zip(item.get("evidence") or [], resolve_evidence(item, corpus)):
            print(f"  引文：{ev.get('quote') or ev.get('any_quote')}")
            for cid in ids:
                print(f"    {cid}（{corpus.kind[cid]}）：{corpus.text[cid][:160]}…")
        return

    errors, warnings = validate(items, corpus, split)
    print(f"评测集 {items_path}：{len(items)} 题；语料 {len(corpus.chunks)} 个片段（指纹 {corpus.fingerprint()}）\n")
    print("\n".join(distribution(items, split)))
    n_facts = sum(len(it.get("facts") or []) for it in items)
    n_numeric = sum(1 for it in items for f in it.get("facts") or [] if f["numeric"])
    print(f"\n关键事实 {n_facts} 条，其中数值事实 {n_numeric} 条")
    if warnings:
        print(f"\n警告 {len(warnings)} 条：")
        print("\n".join("  " + w for w in warnings))
    if errors:
        print(f"\n错误 {len(errors)} 条：")
        print("\n".join("  " + e for e in errors))
        raise SystemExit(1)
    print("\n校验通过：证据引文都能在语料中找到，数值事实都出自证据原文，参考答案满足各自的关键事实。")


if __name__ == "__main__":
    main()
