"""评分逻辑自检：归一化、数字匹配、数字提取、引用剥离、拒答判定、切句。

评测开始前先跑一遍（run.py / validate.py 都会调用），任何一条不成立就报错中止，
避免评分规则悄悄出错而指标看不出来。不调用 LLM，也不读语料。

用法：
    python src/answer_eval/selfcheck.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from textnorm import compile_pattern, macro_numbers, norm, numbers_in, strip_citations  # noqa: E402


def _expect(cond: bool, what: str, failures: list[str]) -> None:
    if not cond:
        failures.append(what)


def run() -> int:
    """返回检查条数；有不成立的直接抛 RuntimeError。"""
    bad: list[str] = []
    n = 0

    def check(cond: bool, what: str) -> None:
        nonlocal n
        n += 1
        _expect(cond, what, bad)

    # ---- 归一化
    check(norm("不 超过 ７５℃ 。") == "不超过75°c。", "全角数字、℃、空白")
    check(norm("保留5\\~8mm间隙") == "保留5~8mm间隙", "MinerU 的 \\~ 转义")
    check(norm("流量： <sup>3</sup>2m /h") == "流量:32m/h", "残留标签")
    check(norm("0.20 m³/h（高于 0.5bar）") == "0.20m3/h(高于0.5bar)", "上标与全角括号")
    check(norm("100 0.15 125~200 0.20") == "100 0.15 125~200 0.20", "数字之间保留一个空格")
    check(norm("20～30mm") == "20~30mm" and norm("IEC/EN 60079–14") == "iec/en60079-14", "波浪号与连字符")

    # ---- 数字宏：边界
    def hit(pattern: str, text: str) -> bool:
        return bool(compile_pattern(pattern).search(norm(text)))

    check(hit("{n:0.1}{mm}", "允差 0.1 毫米") and hit("{n:0.1}{mm}", "允差0.10mm"), "0.1 与 0.10 等价")
    check(not hit("{n:0.1}{mm}", "允差 0.15 毫米"), "0.1 不应匹配 0.15")
    check(not hit("{n:0.1}{mm}", "允差 10.1 毫米"), "0.1 不应匹配 10.1")
    check(not hit("{n:5}倍", "15倍") and not hit("{n:5}倍", "2.5倍") and hit("{n:5}倍", "管径的5倍"), "整数边界")
    check(hit("{n:0.20}m3/h", "0.2 m³/h") and hit("{n:0.2}m3/h", "0.20m³/h"), "0.20 与 0.2 等价")
    check(hit("{n:2000}", "每 2,000 运行小时") and hit("{n:2000}", "2000小时") and not hit("{n:2000}", "12000"),
          "千分位逗号")
    check(hit("{n:25}{mm}", "最少 25.0 毫米") and not hit("{n:25}{mm}", "25.5 毫米"), "25 与 25.0 等价")
    check(hit("{n:1/3}{to}{n:1/2}", "1/3～1/2") and not hit("{n:1/3}", "11/3"), "分数")
    check(hit("{n:2}m/s", "最大流速2m/s") and not hit("{n:2}m/s", "最大流速2.5m/s"), "2 不应匹配 2.5")
    check(hit("{n:75}{C}", "不超过75℃") and hit("{n:75}{C}", "75 度") and hit("{n:75}{C}", "75°C"), "温度单位写法")
    check(hit("{n:5}{to}{n:8}{mm}", "保留5\\~8mm") and hit("{n:5}{to}{n:8}{mm}", "5 到 8 毫米")
          and hit("{n:5}{to}{n:8}{mm}", "5-8mm"), "范围连接符")
    check(hit("{n:240}dan", "∑F 为 240 daN") and not hit("{n:240}dan", "160 200 240 315"), "带单位才算")
    check(macro_numbers("{n:0.35}.{0,3}{to}.{0,1}{n:1.01}") == ["0.35", "1.01"], "取出宏里的数字")

    # ---- 引用剥离
    check(strip_citations("允差0.1毫米[1_c0030]。") == "允差0.1毫米。", "[id]")
    check(strip_citations("符合规格（chunk_id: 4_c0026）。") == "符合规格。", "（chunk_id: id）")
    check(strip_citations("要点。\n\n引用 chunk_id: [4_c0023]、[4_c0026]").strip() == "要点。", "末尾引用行")
    check(strip_citations("【2_c0044】【sample_pump_manual_c0003】") == "", "全角括号与样例 id")

    # ---- 数字提取
    check(numbers_in("允差0.1毫米[1_c0030]，间隙5~8mm[1_c0029]。", answer=True) == [0.1, 5.0, 8.0],
          "答案数字不含 chunk_id 里的数")
    check(numbers_in("1. 关闭阀门\n2. 断电\n0.5 毫米", answer=True) == [0.5], "行首列表序号不算数值")
    check(numbers_in("- 3） 每 6 个月加 20～30g", answer=True) == [6.0, 20.0, 30.0], "带项目符号的序号")
    check(numbers_in("预热到 $1 2 0 ^ { \\circ } \\texttt { C } | 2 5 0 ^ { \\circ }$") == [120.0, 250.0],
          "公式里被拆开的数字")
    check(numbers_in("100 125 150 200") == [100.0, 125.0, 150.0, 200.0], "表格单元格不合并")
    check(numbers_in("每 2,000 运行小时") == [2000.0] and numbers_in("管径100，125，150") == [100.0, 125.0, 150.0],
          "千分位只认半角逗号")
    check(numbers_in("最大流速 2. 5m/s") == [2.0, 5.0], "原文写坏的 2. 5 按原样拆开（由评测集的 src 字段兜底）")

    # ---- 拒答判定与切句（依赖 scoring，放在最后导入，避免循环）
    from scoring import is_refusal, sentences

    markers = ["检索内容不足", "知识库无相关内容", "无法回答", "未能找到相关"]
    check(is_refusal("检索内容不足。", markers), "标准拒答")
    check(is_refusal("检索内容不足，手册中没有关于空压机的内容，因此无法回答这个问题。", markers), "较长的拒答")
    check(not is_refusal("手册未给出具体寿命，取决于清洁度[4_c0189]。检索内容不足以给出小时数。", markers),
          "带引用的说明不算拒答")
    check(not is_refusal("轴承温度不应超过75℃[1_c0033]。", markers), "正常作答")

    scfg = {"min_sentence_chars": 10, "transition_end_markers": ["：", ":"],
            "transition_markers": ["以下是相关信息"], "transition_max_chars": 25}
    sents = sentences("根据检索到的内容，需要做以下检查：\n- 手动旋转轴，确保没有摩擦。[4_c0319]\n"
                      "- 打开隔离阀并检查泵是否泄漏（chunk_id: 4_c0319）。", scfg)
    check([s["text"] for s in sents] == ["手动旋转轴，确保没有摩擦", "打开隔离阀并检查泵是否泄漏"]
          and all(s["cited"] == ["4_c0319"] for s in sents), "切句、去引用、取每句的引用")

    sents = sentences("**起动前准备：**\n从联轴器端向泵看，泵为顺时针方向旋转。1_c0004\n\n"
                      "最大差异为 0.38 毫米。【4_c0069】轴承应使用锂基润滑脂，渗透率2～3。", scfg)
    check([s["cited"] for s in sents] == [["1_c0004"], ["4_c0069"], []]
          and sents[0]["text"] == "从联轴器端向泵看，泵为顺时针方向旋转",
          "句号后面的引用（带不带括号）归前一句；Markdown 小标题不算句子")

    sents = sentences("要求防水，渗透率2～3。2_c0042\n\n外径 250~400mm 时，间距 c 为 3~8mm。2_c0015", scfg)
    check([s["cited"] for s in sents] == [["2_c0042"], ["2_c0015"]]
          and sents[0]["text"].endswith("渗透率2～3") and sents[1]["text"].endswith("3~8mm"),
          "挪动不带括号的引用时不能和前面的数字、字母连成错误的 id")

    from textnorm import loose
    check(loose("用纸板盖住泵的进出口。存放在干燥清洁处") in loose("• 用纸板盖住泵的进出口。\n• 存放在干燥清洁处，防止霜冻"),
          "核对摘抄时忽略标点与项目符号")
    check(loose("允差0.5毫米") not in loose("不同心度允差0.1毫米"), "改了数字的摘抄对不上")

    if bad:
        raise RuntimeError(f"评分逻辑自检失败 {len(bad)}/{n} 条：" + "；".join(bad))
    return n


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"评分逻辑自检通过（{run()} 条）")
