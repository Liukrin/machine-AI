"""答案级评测的文本归一化、数字提取与关键事实匹配模式。

手册原文（MinerU 解析产物）和模型答案在写法上差异很大：全角/半角、空格、
「5\\~8mm」这类转义、℃ 与 °C、m³ 与 m3……评分前两边都先过 norm()，再做包含判断
或正则匹配。所有规则都是确定性的，不调用 LLM。
"""
from __future__ import annotations

import re
import unicodedata

# 答案里的引用标记：[4_c0319]、（chunk_id: 4_c0026）、【2_c0044】、引用 chunk_id：…
CHUNK_ID = r"[A-Za-z0-9_]+_c\d{4}"
CHUNK_ID_RE = re.compile(CHUNK_ID)
_LABEL = r"(?:引用\s*的?\s*)?(?:chunk[ _-]?id\s*[:：]?\s*)?"
_ITEM = rf"[\[【]?\s*{CHUNK_ID}\s*[\]】]?"
_GROUP = rf"{_ITEM}(?:\s*[,，、;；]?\s*{_ITEM})*"
CITATION = rf"(?:[（(]\s*{_LABEL}{_GROUP}\s*[)）]|{_LABEL}{_GROUP})"
_CITATION_RE = re.compile(CITATION, re.I)
# 写在句末标点之后的引用（「…旋转。1_c0004」「…泄漏。[4_c0319]」）：切句前挪到标点之前，
# 否则引用会被切到下一段，这句话就被当成没有引用
_TRAILING_CITATION_RE = re.compile(rf"([。！？])(\s*{CITATION})", re.I)

_DASHES = dict.fromkeys(map(ord, "—–−‐‑‒―"), "-")


def norm(text: str) -> str:
    """归一化：NFKC（全角→半角、℃→°C、³→3、～→~）、去 MinerU 转义与残留标签、去空白、小写、统一连字符。

    两个数字之间的空白保留一个空格：表格拍平后是「100 0.15 125~200 0.20」，
    全部去掉会把相邻单元格的数字并成「1000.15」，数字边界就判断不了了。
    """
    s = unicodedata.normalize("NFKC", text or "")
    s = s.replace("\\~", "~").replace("\\", "")
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"(?<=[\d.])\s+(?=[\d.])", "\x00", s)
    s = re.sub(r"\s+", "", s).replace("\x00", " ")
    return s.lower().translate(_DASHES)


def attach_trailing_citations(text: str) -> str:
    """把紧跟在句末标点后面的引用挪到标点之前（带不带括号都算）。

    挪过去时前面补一个空格：不带括号的 chunk_id 直接贴在正文后面（「…3~8mm2_c0015」），
    会和前面的字母数字连成一个错误的 id。
    """
    return _TRAILING_CITATION_RE.sub(lambda m: " " + m.group(2).strip() + m.group(1), text or "")


_LOOSE_RE = re.compile(r"[^0-9a-z一-鿿]")


def loose(text: str) -> str:
    """只保留汉字、字母、数字：用来核对「摘抄是否出自原文」，忽略标点、空白和项目符号的差异。"""
    return _LOOSE_RE.sub("", norm(text))


def strip_citations(text: str) -> str:
    """去掉答案里的引用标记，避免 chunk_id 里的数字被当成答案里的数值。"""
    return _CITATION_RE.sub("", text or "")


# --------------------------------------------------------------------------- 模式宏
# 评测集里的关键事实用正则描述，作用在 norm() 之后的文本上。为了写起来不易出错，提供几个宏：
#   {n:0.1}  有边界的数字：不会匹配 10.1 或 0.15；允许小数末尾补零（0.10）和千分位逗号（2,000）
#   {to}     范围连接符：~ - 至 到
#   {mm} {C} {F} {in}  常见单位的几种写法
_UNITS = {
    "mm": r"(?:mm|毫米)",
    "C": r"(?:°c|摄氏度|度)",
    "F": r"(?:°f|华氏度)",
    "in": r"(?:英寸|\")",
    "to": r"(?:~|-|至|到)",
}
_NUM_MACRO_RE = re.compile(r"\{n:([0-9][0-9.,/]*)\}")


def _int_regex(digits: str) -> str:
    """整数部分：四位及以上允许千分位逗号。"""
    if len(digits) <= 3:
        return re.escape(digits)
    head = len(digits) % 3 or 3
    groups = [digits[:head]] + [digits[i:i + 3] for i in range(head, len(digits), 3)]
    return ",?".join(groups)


def _num_regex(value: str) -> str:
    v = value.replace(",", "")
    if "/" in v:                       # 分数原样匹配（1/3、1/16）
        return rf"(?<![\d./]){re.escape(v)}(?![\d/])"
    if "." in v:
        v = v.rstrip("0").rstrip(".")  # 0.20 与 0.2 视为同一个数
    if "." in v:
        whole, frac = v.split(".")
        body = _int_regex(whole) + r"\." + frac + "0*"
    else:
        body = _int_regex(v) + r"(?:\.0+)?"
    return rf"(?<![\d.]){body}(?![\d]|\.\d)"


def expand(pattern: str) -> str:
    """把模式里的宏展开成正则。"""
    out = _NUM_MACRO_RE.sub(lambda m: _num_regex(m.group(1)), pattern)
    for name, rx in _UNITS.items():
        out = out.replace("{" + name + "}", rx)
    return out


def macro_numbers(pattern: str) -> list[str]:
    """模式里用 {n:…} 声明的数字（用于检查这些数字确实出现在证据原文里）。"""
    return _NUM_MACRO_RE.findall(pattern)


def compile_pattern(pattern: str) -> re.Pattern:
    return re.compile(expand(pattern), re.I)


def number_pattern(value: str) -> re.Pattern:
    """单个数字的有边界匹配（作用在 norm() 之后的文本上）。"""
    return re.compile(_num_regex(value))


# --------------------------------------------------------------------------- 数字提取
_NUMBER_RE = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d]|\.\d)")
# 行首的列表序号：「1. 」「2）」「(3)」「4、」；「0.5 毫米」开头的行不算
_LIST_MARKER_RE = re.compile(r"(?m)^\s*(?:[-*•·]\s*)?[（(]?\d{1,2}(?:[)）、]|\.(?!\d))\s*")
# MinerU 把公式里的数字拆成「$1 2 0 ^ { \circ }$」，只在 $…$ 内把数字间的空格并回去
_LATEX_SPAN_RE = re.compile(r"\$[^$]*\$")
_SPACED_DIGITS_RE = re.compile(r"(?<=\d) (?=[\d.])|(?<=\.) (?=\d)")
# 千分位逗号只认 ASCII 逗号，且须在 NFKC 之前处理（否则「100，125，150」会被并成一个数）
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")


def numbers_in(text: str, *, answer: bool = False) -> list[float]:
    """提取文本中的数值（浮点）。answer=True 时先去掉引用标记和列表序号。"""
    s = text or ""
    if answer:
        s = _LIST_MARKER_RE.sub("", strip_citations(s))
    s = _LATEX_SPAN_RE.sub(lambda m: _SPACED_DIGITS_RE.sub("", m.group(0)), s)
    s = _THOUSANDS_RE.sub("", s)
    s = unicodedata.normalize("NFKC", s).replace("\\", "")
    s = re.sub(r"<[^>]+>", " ", s)
    return [float(x) for x in _NUMBER_RE.findall(s)]


def number_set(text: str) -> set[float]:
    return {round(x, 6) for x in numbers_in(text)}
