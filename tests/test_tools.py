"""工具层（src/s4_agent/tools.py），用本地知识库，不调用大模型：

- 换算、核对工具的参数要对得上原文（数在问题或资料里找得到，单位与原文一致）；
- 手册原文包在 <片段原文> 标签里（注入防护），关掉开关就不加；
- 证据池的「给过全文 / 只给过几行」只升不降，查表只给过几行的大表，再读时要给正文；
- 工具超时后，后台线程跑完也改不到这次回答的证据池。
"""
import time

import pytest

import agent
from tables import table_rows
from tools import SMALL_TABLE_ROWS, Toolbox, get_corpus, run_tool

pytestmark = pytest.mark.kb
CFG, TAU = agent.CFG, agent.TAU_DISTANCE
Q_MZ = "Wilo-WR 管径 200mm 时允许的力矩 Mz 换算成 N·m 是多少？"


def test_convert_unit_must_match_source():
    tb = Toolbox(CFG, TAU, {}, context=Q_MZ)
    r = tb.run("lookup_table", {"keywords": "Mz", "manual": "2"})
    assert r.ok and "2_c0051" in tb.seen
    assert '<片段原文 id="2_c0051">' in r.content and "</片段原文>" in r.content

    r = tb.run("convert_unit", {"value": 85, "from_unit": "kgf·m", "to_unit": "N·m"})
    assert not r.ok and "dan" in r.content.lower()          # 原文这一行的单位是 daN（阶段 2 出过的错）
    r = tb.run("convert_unit", {"value": 85, "from_unit": "daN·m", "to_unit": "N·m"})
    assert r.ok and "850" in r.content
    r = tb.run("convert_unit", {"value": 86, "from_unit": "daN·m", "to_unit": "N·m"})
    assert not r.ok and "找不到" in r.content               # 问题和资料里都没有的数
    assert tb.run("convert_unit", {"value": 850, "from_unit": "N·m", "to_unit": "kgf·m"}).ok   # 上一步的结果可以接着换算


def test_mcp_mode_has_no_question_context():
    """MCP 拿不到用户的问题：资料里没有的数无从核对、放行（可能来自用户）；资料里有的数照样核对单位。"""
    tb = Toolbox(CFG, TAU, {}, abbreviate=False)
    tb.run("lookup_table", {"keywords": "Mz", "manual": "2"})
    assert tb.run("convert_unit", {"value": 86, "from_unit": "daN·m", "to_unit": "N·m"}).ok
    assert not tb.run("convert_unit", {"value": 85, "from_unit": "kgf·m", "to_unit": "N·m"}).ok


def test_check_value_limit_and_measured_value():
    q = "Model 3700 轴承温度实测 185°F，在手册要求的范围内吗？"
    tb = Toolbox(CFG, TAU, {}, context=q)
    r = tb.run("search_manuals", {"query": q})
    assert r.ok and '<片段原文 id="4_c0151">' in r.content
    args = {"value": 185, "unit": "°F", "source_chunk_id": "4_c0151", "limit_max": 180}
    r = tb.run("check_value", {**args, "limit_unit": "°C"})
    assert not r.ok and "°F" in r.content                    # 限值 180 的原文单位是 °F
    r = tb.run("check_value", {**args, "limit_unit": "°F"})
    assert r.ok and "超出上限" in r.content

    # 资料里恰好有「185 (85)」（4_c0058 温度组别表），所以换一个资料里没有的数来测被核对值
    q2 = "Model 3700 轴承温度实测 190°F，在手册要求的范围内吗？"
    tb2 = Toolbox(CFG, TAU, {}, context=q2)
    tb2.run("search_manuals", {"query": q2})
    measured = {"unit": "°C", "source_chunk_id": "4_c0151", "limit_max": 82, "limit_unit": "°C"}
    r = tb2.run("check_value", {"value": 87.8, **measured})
    assert not r.ok and "被核对值" in r.content              # 模型心算的换算结果，没出处
    r = tb2.run("convert_unit", {"value": 190, "from_unit": "°F", "to_unit": "°C"})
    assert r.ok and "87.78" in r.content
    assert "超出上限" in tb2.run("check_value", {"value": 87.78, **measured}).content


def test_injection_guard_switch():
    q = "Model 3700 轴承温度正常范围？"
    off = Toolbox({**CFG, "injection_guard": False}, TAU, {}, context=q)
    r = off.run("search_manuals", {"query": q})
    assert r.ok and "<片段原文" not in r.content


def _pick_table(big: bool):
    """找一张大表（或小表）和一个只命中它的关键词。"""
    corpus = get_corpus()
    for cid, c in corpus.by_id.items():
        if c["chunk_type"] != "table":
            continue
        n = corpus.tables.n_rows_of(cid)
        if (n <= SMALL_TABLE_ROWS) == big or n < 2:
            continue
        for row in table_rows(c.get("table_html") or "")[1:]:
            for cell in row:
                kw = cell.strip()
                if 2 <= len(kw) <= 12 and " " not in kw:
                    hits, _ = corpus.tables.lookup([kw], None, limit=10_000)
                    if hits and {h["chunk_id"] for h in hits} == {cid}:
                        return cid, kw
    pytest.fail("知识库里没找到合适的表")


def test_big_table_rows_then_full_text():
    cid, kw = _pick_table(big=True)
    tb = Toolbox(CFG, TAU, {})
    tb.run("lookup_table", {"keywords": kw})
    assert tb.seen[cid]["shown"] == "rows"
    r = tb.run("read_section", {"chunk_id": cid, "before": 0, "after": 0})
    assert "前面已给出全文" not in r.content and len(r.content) > 80     # 只给过几行，再读要给正文
    assert r.chunks == []                                                  # 不重复计入来源
    assert tb.seen[cid]["shown"] in ("full", "head")
    assert len(tb.run("read_section", {"chunk_id": cid, "before": 0, "after": 0}).content) < 200


def test_small_table_given_whole():
    cid, kw = _pick_table(big=False)
    tb = Toolbox(CFG, TAU, {})
    tb.run("lookup_table", {"keywords": kw})
    r = tb.run("read_section", {"chunk_id": cid, "before": 0, "after": 0})
    assert tb.seen[cid]["shown"] == "full" and "前面已给出全文" in r.content


def test_shown_never_downgrades():
    cid, kw = _pick_table(big=True)
    tb = Toolbox(CFG, TAU, {cid: {"distance": None, "shown": "full"}})
    tb.run("lookup_table", {"keywords": kw})
    assert tb.seen[cid]["shown"] == "full"


def test_timeout_does_not_touch_evidence_pool(monkeypatch):
    orig = Toolbox.search_manuals

    def slow(self, *a, **k):
        time.sleep(1.0)
        return orig(self, *a, **k)

    monkeypatch.setattr(Toolbox, "search_manuals", slow)
    tb = Toolbox(CFG, TAU, {})
    res, _ = run_tool(tb, "search_manuals", {"query": "水泵与电动机同心度允差"}, 0.3)
    assert not res.ok and "超过" in res.content
    time.sleep(2.0)                     # 等后台线程跑完
    assert tb.seen == {}

    monkeypatch.setattr(Toolbox, "search_manuals", orig)
    res, _ = run_tool(tb, "search_manuals", {"query": "水泵与电动机同心度允差"}, 10.0)
    assert res.ok and len(tb.seen) == len(res.chunks) > 0
