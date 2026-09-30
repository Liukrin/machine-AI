"""一键重建知识库：S1 解析 → blocks → 质检 → S2 清洗 → 切块 → 向量库 → BM25。

各步骤就是原先需要手动逐个运行的脚本，这里按顺序在子进程里执行，任一步失败立即停止；
输入输出路径仍全部由 configs/config.yaml 决定。最后核对 chunks.jsonl、Chroma、BM25
三处的 chunk_id 是否完全一致。

MinerU（parse 步）要求 transformers<5，与应用环境的 sentence-transformers 6（要求
transformers>=5）冲突，需要单独的解析环境（requirements-parse.txt）：
  - batch_parse.py 会跳过已有解析产物的 PDF，全部解析过时用应用环境直接跑即可；
  - 有新 PDF 需要解析时，用 --parse-python 指定解析环境的 python。

用法：
    python scripts/build_index.py                         # 全流程（已解析的 PDF 自动跳过）
    python scripts/build_index.py --parse-python <解析环境的 python 路径>
    python scripts/build_index.py --from chunks           # 从某一步开始，如只改了切块参数

重建后需重启后端（API 进程内缓存了模型、索引与 chunk）。
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]

# (步骤名, 说明, 脚本)
STEPS = [
    ("parse", "S1 MinerU 解析 PDF", "src/s1_ingest/batch_parse.py"),
    ("blocks", "S1 整理结构化 blocks", "src/s1_ingest/build_blocks.py"),
    ("qc", "S1 解析质检报告", "src/s1_ingest/qc_report.py"),
    ("clean", "S2 清洗 blocks", "src/s2_index/clean_blocks.py"),
    ("chunks", "S2 合并切块", "src/s2_index/build_chunks.py"),
    ("vectors", "S2 向量化并写入 Chroma", "src/s2_index/build_vectors.py"),
    ("bm25", "S2 建 BM25 索引", "src/s2_index/build_bm25.py"),
]
STEP_NAMES = [s[0] for s in STEPS]


def run_step(i: int, name: str, desc: str, script: str, python: str) -> float:
    print(f"\n{'=' * 78}\n[{i}/{len(STEPS)}] {name}：{desc}（{script}）\n{'=' * 78}", flush=True)
    t0 = time.perf_counter()
    res = subprocess.run([python, script], cwd=ROOT)
    elapsed = time.perf_counter() - t0
    if res.returncode != 0:
        raise SystemExit(f"\n步骤 {name} 失败（退出码 {res.returncode}，{elapsed:.1f}s），后续步骤未执行。")
    return elapsed


def check_consistency() -> None:
    """chunks.jsonl / Chroma / BM25 三处的 chunk_id 必须完全一致，否则检索会取到不存在或过期的片段。"""
    import chromadb
    import yaml

    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    sv, sr = cfg["s2_vector"], cfg["s3_retrieval"]

    chunks = [json.loads(line) for line in
              (ROOT / sv["input_jsonl"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [c["chunk_id"] for c in chunks]
    collection = chromadb.PersistentClient(path=str(ROOT / sv["chroma_dir"])).get_collection(sv["collection_name"])
    chroma_ids = collection.get(include=[])["ids"]
    with (ROOT / sr["bm25_index"]).open("rb") as f:
        bm25_ids = pickle.load(f)["chunk_ids"]

    types = Counter(c["chunk_type"] for c in chunks)
    docs = Counter(c["doc_id"] for c in chunks)
    print(f"\nchunks.jsonl {len(ids)} 条（" + " / ".join(f"{k} {v}" for k, v in sorted(types.items()))
          + f"）；Chroma {len(chroma_ids)} 条；BM25 {len(bm25_ids)} 篇")
    print("按文档：" + "，".join(f"{d} {n}" for d, n in sorted(docs.items())))

    problems = []
    if len(set(ids)) != len(ids):
        problems.append(f"chunks.jsonl 内有 {len(ids) - len(set(ids))} 个重复 chunk_id")
    if set(chroma_ids) != set(ids):
        problems.append(f"Chroma 与 chunks.jsonl 不一致（多 {len(set(chroma_ids) - set(ids))}，"
                        f"少 {len(set(ids) - set(chroma_ids))}）")
    if bm25_ids != ids:
        problems.append("BM25 的 chunk_id 列表与 chunks.jsonl 不一致")
    if problems:
        raise SystemExit("一致性检查失败：" + "；".join(problems))
    print("一致性检查通过：三处 chunk_id 完全一致")


def main() -> None:
    ap = argparse.ArgumentParser(description="一键重建知识库（S1 解析 → S2 索引）")
    ap.add_argument("--from", dest="start", choices=STEP_NAMES, default=STEP_NAMES[0],
                    help="从哪一步开始（默认 parse，即全流程）")
    ap.add_argument("--parse-python", default=sys.executable,
                    help="执行 parse 步的 python（MinerU 所在的解析环境）；默认用当前解释器")
    args = ap.parse_args()

    start = STEP_NAMES.index(args.start)
    timings = []
    for i, (name, desc, script) in enumerate(STEPS, 1):
        if i - 1 < start:
            continue
        python = args.parse_python if name == "parse" else sys.executable
        timings.append((name, run_step(i, name, desc, script, python)))

    print(f"\n{'=' * 78}\n各步耗时：" + "，".join(f"{n} {t:.1f}s" for n, t in timings))
    check_consistency()
    print("重建完成。若后端正在运行，请重启以加载新索引。")


if __name__ == "__main__":
    main()
