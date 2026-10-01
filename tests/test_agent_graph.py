"""Agent 图（src/s4_agent/agent.py）：预检索 → 模型 ⇄ 工具 → 核对 ⇄ 改写。

检索和工具是真的（kb），大模型用剧本代替：剧本按顺序给出每次模型调用的返回（文字回答或工具调用）。
"""
import pytest

import agent
from conftest import tool_call

pytestmark = pytest.mark.kb

Q_TEMP = "Model 3700 的轴承温度正常范围是多少？"
RIGHT = "Model 3700 的轴承温度应介于 49°C | 120°F 和 82°C | 180°F 之间 [4_c0151]。"
WRONG = "Model 3700 的轴承温度应介于 49°C | 120°F 和 82°C | 190°F 之间 [4_c0151]。超过 82°F 时应停机检查 [4_c0151]。"


def run(fake_llm, question: str, replies: list):
    llm = fake_llm(replies)
    events, final = [], None
    for mode, payload in agent.build_agent_graph().stream(agent.initial_state(question),
                                                          stream_mode=["custom", "values"]):
        if mode == "custom":
            events.append(payload)
        else:
            final = payload
    return events, final, llm


def test_correct_answer_passes_without_repair(fake_llm):
    events, final, _ = run(fake_llm, Q_TEMP, [RIGHT])
    v = final["verification"]
    assert v["numbers_checked"] >= 4 and v["number_issues"] == [] and v["repair"] is None
    assert final["llm_calls"] == 1 and not any(e["type"] == "repair" for e in events)


def test_wrong_numbers_get_one_repair(fake_llm):
    events, final, llm = run(fake_llm, Q_TEMP, [WRONG, RIGHT])
    repairs = [e for e in events if e["type"] == "repair"]
    assert len(repairs) == 1 and len(repairs[0]["issues"]) == 2
    # 问题清单只放在改写那一次调用的最后一条消息里，不进对话记录
    note = llm.calls[1][-1].content
    assert "190 °F" in note and "82 °F" in note
    assert all(m.content != note for m in final["messages"])
    v = final["verification"]
    assert (v["repair"]["kept"], v["repair"]["issues_before"], v["repair"]["issues_after"]) == ("repaired", 2, 0)
    assert final["answer"] == RIGHT and final["llm_calls"] == 2


def test_worse_rewrite_falls_back_to_draft(fake_llm):
    draft = "Model 3700 的轴承温度上限为 190°F [4_c0151]。"
    worse = "Model 3700 的轴承温度上限为 195°F，下限为 30°C [4_c0151]。"
    _, final, _ = run(fake_llm, Q_TEMP, [draft, worse])
    v = final["verification"]
    assert (v["repair"]["kept"], v["repair"]["issues_before"], v["repair"]["issues_after"]) == ("draft", 1, 2)
    assert final["answer"] == draft and len(v["number_issues"]) == 1


def test_refusal_with_citations_is_cleaned(fake_llm):
    q = "Wilo-WR 泵运行时测得振动速度 4.5 mm/s，是否超出手册允许值？"
    refusal = "知识库无相关内容。手册只给出了联轴器对中跳动限值（轴向最大 0.05 mm）[2_c0016]，没有振动速度的允许值 [2_c0023]。"
    events, final, _ = run(fake_llm, q, [refusal])
    v = final["verification"]
    assert v["refused"] and "_c00" not in final["answer"] and final["answer"].startswith("知识库无相关内容")
    assert v["repair"] is None and not any(e["type"] == "repair" for e in events)


def test_tool_call_result_grounds_the_answer(fake_llm):
    q = "Model 3700 运行中测得轴承温度 185°F，换算成摄氏度是多少？"
    events, final, _ = run(fake_llm, q, [tool_call("convert_unit", {"value": 185, "from_unit": "°F", "to_unit": "°C"}),
                                         "185°F 约为 85°C。"])
    ends = [e for e in events if e["type"] == "tool_end" and not e.get("auto")]
    assert len(ends) == 1 and ends[0]["ok"] and "85" in ends[0]["output"]
    assert final["verification"]["number_issues"] == [] and final["llm_calls"] == 2
