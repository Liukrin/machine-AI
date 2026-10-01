"""已提交的评测结果可复算：用缓存的系统输出和评审输出重算指标，与仓库里的 metrics.json 逐项一致（零调用、不写文件）。
另跑一遍 MCP Server 自检（scripts/mcp_smoke.py）。都要用本地知识库（kb）。
"""
import subprocess
import sys

import pytest

from conftest import ROOT

pytestmark = pytest.mark.kb
RUNS = ["baseline", "agent_v1", "agent_v1_flash", "agent_v2"]


def _run(*args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout)


@pytest.mark.parametrize("eval_set", ["main", "tasks"])
@pytest.mark.parametrize("run", RUNS)
def test_committed_eval_recomputes(run, eval_set):
    p = _run("src/answer_eval/run.py", "--run", run, "--set", eval_set, "--reuse", "--check")
    assert p.returncode == 0, (p.stdout[-2000:] + p.stderr[-2000:])
    assert "复算一致" in p.stdout


def test_mcp_smoke():
    p = _run("scripts/mcp_smoke.py", timeout=180)
    assert p.returncode == 0, (p.stdout[-3000:] + p.stderr[-2000:])
