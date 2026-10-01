"""数值核对（src/s4_agent/numcheck.py）：回答和工具参数里的数，要在资料里找得到、单位对得上。

前半部分用自造的片段（CI 上跑）；后半部分（kb）用知识库里的真实片段，是阶段 3 开发时的回归用例。
"""
import json

import pytest

from numcheck import Source, check_answer, ground, quantities
from conftest import ROOT


def g(text: str, sources: list[Source]):
    """核对 text 里最后一个数。"""
    return ground(quantities(text)[-1], sources)


# --------------------------------------------------------------------------- 抽取
@pytest.mark.parametrize("text, expected", [
    ("176 Nm130英尺-磅", [("176", "Nm"), ("130", "英尺-磅")]),
    ("轴承温度应介于 49–82 °C", [("49", "°C"), ("82", "°C")]),                 # 区间前一个数沿用后面的单位
    ("$0.05\\mathrm{mm}$ 和 $85^{\\circ}\\mathrm{C}$", [("0.05", "mm"), ("85", "°C")]),
    ("5 to 10 次", [("5", None), ("10", None)]),                                # 「to」不是吨
    ("0.20 m³/h（约 3.33 L/min），转速 1450 r/min，压力 0.5bar",
     [("0.20", "m3/h"), ("3.33", "L/min"), ("1450", "r/min"), ("0.5", "bar")]),
])
def test_quantities(text, expected):
    assert [(q.num, q.unit) for q in quantities(text)] == expected


# --------------------------------------------------------------------------- 单个数的出处
WATER = Source("轴封冲洗水量\n泵型号 | 冲洗水量（m³/h）\nWR-50 | 0.20\nWR-80 | 0.30")
BEARING = Source("轴承温度应介于 49°C | 120°F 和 82°C | 180°F 之间。")
NOZZLE = Source("管口允许负荷\n管径 | Fx | Fy | Fz\n200 | 230 | 150 | 190\n"
                "管径 | 力矩[daN] | Mx | My | Mz\n200 | 力矩[daN] | 110 | 90 | 85")


@pytest.mark.parametrize("text, source, status, units", [
    ("0.20 m³/h", WATER, "ok", []),                       # 单位在表头
    ("0.30 立方米每小时", WATER, "ok", []),
    ("0.20 L/min", WATER, "unit_mismatch", ["m3/h"]),
    ("0.25 m³/h", WATER, "no_source", []),
    ("180°F", BEARING, "ok", []),                         # °C 与 °F 并列：数和单位要配对
    ("180°C", BEARING, "unit_mismatch", ["°F"]),
    ("82 摄氏度", BEARING, "ok", []),
    ("49 华氏度", BEARING, "unit_mismatch", ["°C"]),
    ("85 daN·m", NOZZLE, "ok", []),                       # 手册把力矩的单位写成「力矩[daN]」：daN 是 daN·m 的组成部分
    ("85 kgf·m", NOZZLE, "unit_mismatch", ["dan"]),       # 阶段 2 出过的错：daN·m 被模型改填成 kgf·m
    ("85 N·m", NOZZLE, "unit_mismatch", ["dan"]),
])
def test_ground(text, source, status, units):
    f = g(text, [source])
    assert (f.status, f.source_units) == (status, units)


@pytest.mark.parametrize("text, status", [
    ("约 3.33 L/min", "ok"),          # 换算结果按舍入匹配
    ("约 3.3 L/min", "ok"),
    ("3.333 L/min", "ok"),
    ("约 3.4 L/min", "no_source"),
    ("3.33 m³/h", "no_source"),       # 数对、单位不对
])
def test_ground_calc_output(text, status):
    calc = Source("换算结果：0.2 m³/h = 3.333 L/min（精确值 3.33333 L/min）", kind="calc")
    assert g(text, [calc]).status == status


# --------------------------------------------------------------------------- 整条回答
QUESTION = Source("Model 3700 轴承温度实测 185°F，在手册要求的范围内吗？")
CHECK = Source("核对结论：超出上限\n被核对值：185 °F\n手册限值：≤ 180 °F（出自 4_c0151）\n差值：高出上限 5 °F（2.8%）", kind="calc")
ANSWER = ("不在范围内，实测值超出上限。\n\n1. 手册规定：轴承温度应介于 49°C | 120°F 和 82°C | 180°F 之间 [4_c0151]。\n"
          "2. 核对结论：185°F 超出上限 180°F，高出 5°F（约 2.8%）。\n3. 见手册第 52 页。")


def test_check_answer_passes_correct_answer():
    n, issues = check_answer(ANSWER, [QUESTION, BEARING, CHECK])
    assert n == 8 and issues == []


def test_check_answer_flags_wrong_number_and_unit():
    bad = ANSWER.replace("180°F 之间", "190°F 之间").replace("高出 5°F", "高出 5°C")
    _, issues = check_answer(bad, [QUESTION, BEARING, CHECK])
    assert [(i.quantity, i.status, i.source_units) for i in issues] == [
        ("190 °F", "no_source", []), ("5 °C", "unit_mismatch", ["°F"])]
    assert "单位是 °F" in issues[1].describe()


def test_check_answer_skips_list_markers_years_and_small_counts():
    n, issues = check_answer("1.5 mm 的间隙。\n1. 检查联轴器。", [Source("间隙 1.5 mm")])
    assert (n, issues) == (1, [])          # 行首的 1.5 mm 照常核对，序号「1.」不核对
    assert check_answer("2024 年版手册第 3 章：拧紧 2 圈 [1_c0001]。", [Source("无关")]) == (0, [])


# --------------------------------------------------------------------------- 真实片段（kb）
@pytest.fixture(scope="module")
def real():
    from hybrid_retrieval import build_search_text
    from tables import table_rows

    chunks = {}
    for line in (ROOT / "data" / "chunks" / "chunks.jsonl").open(encoding="utf-8"):
        c = json.loads(line)
        chunks[c["chunk_id"]] = c

    def src(cid: str) -> Source:
        c = chunks[cid]
        if c["chunk_type"] == "table":       # 与工具输出一样按行渲染
            rows = table_rows(c["table_html"])
            return Source((c.get("heading_path") or "") + "\n" + "\n".join(" | ".join(r) for r in rows))
        return Source(build_search_text(c))
    return src


@pytest.mark.kb
@pytest.mark.parametrize("cid, text, status", [
    ("2_c0044", "0.20 m³/h", "ok"),
    ("2_c0044", "0.20 L/min", "unit_mismatch"),
    ("2_c0044", "0.25 m³/h", "no_source"),
    ("4_c0151", "180°F", "ok"),
    ("4_c0151", "180°C", "unit_mismatch"),
    ("4_c0151", "82 摄氏度", "ok"),
    ("2_c0051", "85 daN·m", "ok"),
    ("2_c0051", "85 kgf·m", "unit_mismatch"),
])
def test_ground_real_chunks(real, cid, text, status):
    assert g(text, [real(cid)]).status == status
