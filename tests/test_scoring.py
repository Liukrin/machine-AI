"""答案级评测的评分逻辑自检（src/answer_eval/selfcheck.py）：归一化、数字匹配、拒答判定、结论判定等。

评测开始前 run.py 也会跑一遍；放进 pytest 是为了改评分代码时在 CI 上就能发现。
"""
import selfcheck


def test_scoring_selfcheck():
    assert selfcheck.run() >= 50       # 有不成立的条目会直接抛 RuntimeError 并列出是哪几条
