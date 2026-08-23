"""S1 任务二：批量解析 PDF。

串行调用 MinerU（pipeline 后端 / CPU / 中文），把 raw_pdf_dir 下每份 PDF
转成 MinerU 原始产物，不做任何后处理、清洗或 Markdown 改写。

用法：
    python src/s1_ingest/batch_parse.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

# 固定 UTF-8 输出，避免 Windows 控制台默认 GBK 导致中文乱码
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

# 配置里的 language 用 zh，而 MinerU CLI 的 -l 用 ch（其余枚举值原样透传）
_LANG_TO_CLI = {"zh": "ch", "ch": "ch", "en": "en"}


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_content_list(auto_dir: Path) -> Path | None:
    """在 {输出目录}/auto/ 下找主 content_list.json（排除 _v2）。"""
    if not auto_dir.is_dir():
        return None
    for p in sorted(auto_dir.glob("*_content_list.json")):
        if not p.name.endswith("_v2.json"):
            return p
    return None


def count_types(content_list_path: Path) -> tuple[int, dict[str, int]]:
    """返回 (页数, {type: 数量})。页数 = 各 block 最大 page_idx + 1。"""
    data = json.loads(content_list_path.read_text(encoding="utf-8"))
    blocks = data if isinstance(data, list) else data.get("pdf_info", [])
    counts: dict[str, int] = {}
    max_page = -1
    for b in blocks:
        t = str(b.get("type", "unknown"))
        counts[t] = counts.get(t, 0) + 1
        p = b.get("page_idx")
        if isinstance(p, int):
            max_page = max(max_page, p)
    pages = max_page + 1 if max_page >= 0 else 0
    return pages, counts


def run_mineru(pdf: Path, out_dir: Path, backend: str, lang: str) -> int:
    """调用 MinerU CLI 解析单份 PDF，返回退出码。"""
    cmd = [
        sys.executable, "-m", "mineru.cli.client",
        "-p", str(pdf),
        "-o", str(out_dir),
        "-b", backend,
        "-l", lang,
    ]
    env = os.environ.copy()
    # 模型已本地化，强制直连，避免 requests 走 Windows 系统代理拖慢启动检查
    env["NO_PROXY"] = "*"
    env["no_proxy"] = "*"
    env.pop("HTTP_PROXY", None)
    env.pop("HTTPS_PROXY", None)
    env.pop("ALL_PROXY", None)
    return subprocess.run(
        cmd, cwd=str(ROOT), env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )


def main() -> None:
    cfg = load_config()
    s1 = cfg["s1_ingest"]
    raw_dir = ROOT / s1["raw_pdf_dir"]
    parsed_dir = ROOT / s1["parsed_md_dir"]
    mineru = s1["mineru"]
    backend = mineru["backend"]           # pipeline
    device = mineru["device"]             # cpu（由 CPU 版 torch 保证，CLI 无 -d 参数）
    lang = _LANG_TO_CLI.get(mineru["language"], mineru["language"])  # zh -> ch

    pdfs = sorted(raw_dir.glob("*.pdf"))
    print(f"raw_pdf_dir = {raw_dir}")
    print(f"parsed_md_dir = {parsed_dir}")
    print(f"backend={backend}  device={device}  lang={lang}  |  共 {len(pdfs)} 份 PDF，串行解析\n")

    rows = []  # (stem, status, elapsed, pages, counts, error)
    for pdf in pdfs:
        stem = pdf.stem
        auto_dir = parsed_dir / stem / "auto"
        existing = find_content_list(auto_dir)

        if existing:
            pages, counts = count_types(existing)
            rows.append((stem, "SKIP", None, pages, counts, ""))
            print(f"SKIP   {stem}  (已有 content_list.json)")
            continue

        t0 = time.time()
        try:
            res = run_mineru(pdf, parsed_dir, backend, lang)
            elapsed = time.time() - t0
            cl = find_content_list(auto_dir)
            if res.returncode == 0 and cl is not None:
                pages, counts = count_types(cl)
                rows.append((stem, "OK", elapsed, pages, counts, ""))
                print(f"OK     {stem}  ({elapsed:.1f}s, {pages} 页)")
            else:
                err = (res.stderr or res.stdout or "").strip().splitlines()
                err = err[-1] if err else f"returncode={res.returncode}"
                rows.append((stem, "FAIL", elapsed, 0, {}, err))
                print(f"FAIL   {stem}  -> {err}")
        except Exception as e:  # noqa: BLE001 —— 单份失败不中断整体
            elapsed = time.time() - t0
            rows.append((stem, "FAIL", elapsed, 0, {}, repr(e)))
            print(f"FAIL   {stem}  -> {e!r}")

    # ---- 汇总表 ----
    print("\n" + "=" * 110)
    print(f"{'文件':<26}{'状态':<6}{'耗时(s)':>9}{'页数':>6}  content_list 各 type 数量")
    print("-" * 110)
    for stem, status, elapsed, pages, counts, err in rows:
        dur = f"{elapsed:.1f}" if elapsed is not None else "-"
        tc = ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "(无)"
        line = f"{stem:<26}{status:<6}{dur:>9}{pages:>6}  {tc}"
        if status == "FAIL" and err:
            line += f"   [error] {err}"
        print(line)
    print("-" * 110)
    n_ok = sum(1 for r in rows if r[1] == "OK")
    n_skip = sum(1 for r in rows if r[1] == "SKIP")
    n_fail = sum(1 for r in rows if r[1] == "FAIL")
    print(f"合计 {len(rows)} 份：OK={n_ok}  SKIP={n_skip}  FAIL={n_fail}")


if __name__ == "__main__":
    main()
