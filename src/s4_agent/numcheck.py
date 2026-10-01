"""数值与单位的出处核对（确定性代码，不调用 LLM）。两处用到：

1. 工具参数（tools.py）：convert_unit 的被换算值、check_value 的被核对值与限值单位，必须对得上原文。
   专门防「换算工具不认识 daN·m，模型就改填 kgf·m」这类换单位（process_log 第 15 条）。
2. 回答核对（agent.py 的 check 节点）：回答里每个「数值 + 单位」都要能在本次的资料（问句、工具返回的内容、
   换算/核对结果）里找到，单位也要一致；有问题的句子交给模型改写一次。

一个数「对得上」资料，指：
- 数值：资料里出现过这个数（宽松读法：MinerU 拆开的「2. 5」、千分位「1,450」都认）；
  或者它是某个换算/核对结果的正确舍入（3.333 → 3.33）。
- 单位：资料里这个数明确带着单位时，单位要相同（同量纲、同系数：m³/h 与 立方米/小时、N·m 与 Nm 算相同）。
  资料里这个数没带单位（如表格单元格）时：
  - 它所在的表格行写了单位（如「力矩[daN] | Mz | 36 | 49 | 62 | 85」），给定单位要与其中之一相同，
    或以它为组成部分（daN·m 含 daN，算对得上；kgf·m 不算）——手册把力矩写成 daN 这类笔误照样能接住；
  - 再看同一段资料提到过的同量纲单位（如表头的「轴封水量[m³/h]」）：提到过、但都不是给定单位，才算不符。
  两样都没有，无从核对，不算不符。
宁可漏判也不误判：误判会让正确的回答被打回改写。
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from units import _LINEAR, _TEMPERATURE, _key, numbers_in_text

# --------------------------------------------------------------------------- 单位写法
_EXTRA = {   # 文本里常见、换算接口用不到的写法 → 换算表里的写法
    "立方米每小时": "m3/h", "立方米/h": "m3/h", "升每分钟": "l/min", "升/min": "l/min", "米每秒": "m/s",
    "转每分钟": "r/min", "转每分": "r/min", "min-1": "r/min", "1/min": "r/min", "r.p.m": "rpm",
    "kgf/cm2g": "kgf/cm2", "公斤/厘米2": "kgf/cm2",
}


def _table() -> dict[str, tuple[str, object]]:
    out: dict[str, tuple[str, object]] = {}
    for dim, (_base, table) in _LINEAR.items():
        for k, f in table.items():
            out[k] = (dim, round(float(f), 12))
    for k, scale in _TEMPERATURE.items():
        out[k] = ("温度", scale)
    for k, target in _EXTRA.items():
        out[_key(k)] = out[target]
    return out


UNITS = _table()
# 紧跟在数字后面才认的写法：去掉太容易误认的（「3 C 级」「D 型」「打开」）
_AFTER_NUMBER = {k: v for k, v in UNITS.items() if k not in {"c", "f", "k", "开", "d"}}
# 在一段文字里任意位置都认的写法（用来看「这段资料提到过哪些单位」）：去掉单个字母、单个汉字和容易撞上普通词的写法
_ANYWHERE = {k: v for k, v in UNITS.items()
             if len(k) >= 2 and k not in {"in", "ps", "pa", "mil", "thou", "开尔文", "kelvin", "celsius", "fahrenheit"}}
_ASCII_LETTER = re.compile(r"[a-z]")
_ANYWHERE_RE = re.compile(
    r"(?<![a-z])(" + "|".join(re.escape(k) for k in sorted(_ANYWHERE, key=len, reverse=True)) + r")(?![a-z])")


def unit_id(unit: str | None) -> tuple | None:
    """单位的身份（量纲, 系数）；认不出返回 None。"""
    if not unit:
        return None
    return UNITS.get(_key(unit))


def unit_dim(unit: str | None) -> str | None:
    uid = unit_id(unit)
    return uid[0] if uid else None


# --------------------------------------------------------------------------- 文本规整
_LATEX_CMD = re.compile(r"\\(?:mathrm|text|textrm|operatorname|mathbf|mbox|rm)\s*")


def normalize(text: str, merge_split_decimals: bool = False) -> str:
    """全角转半角（℃→°C、³→3）、去掉 LaTeX 包装（$85^{\\circ}\\mathrm{C}$ → 85°C）、去千分位逗号。"""
    s = unicodedata.normalize("NFKC", text or "")
    s = s.replace("^{\\circ}", "°").replace("^\\circ", "°").replace("\\circ", "°").replace("\\degree", "°")
    s = s.replace("\\cdot", "·").replace("\\times", "×").replace("\\%", "%")
    s = _LATEX_CMD.sub("", s)
    s = s.replace("\\,", "").replace("\\ ", " ").replace("\\!", "")
    s = s.replace("$", "").replace("{", "").replace("}", "").replace("\\", "")
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    if merge_split_decimals:
        s = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", s)          # MinerU 把「2.5」拆成「2. 5」
    s = re.sub(r"\s*([·/\-])\s*(?=[^\d\s])", r"\1", s)     # 「daN · m」「m3 / h」→ 去掉单位内部的空格
    return s


# --------------------------------------------------------------------------- 数值 + 单位
_NUMBER = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d]|\.\d)")
_RANGE_SEP = re.compile(r"\s*(?:~|～|-|–|—|至|到)\s*")
_SOURCE_TAG = re.compile(r"</?片段原文[^<>]*>")    # tools.SOURCE_TAG：工具输出里包手册原文的标签


@dataclass
class Quantity:
    value: float
    num: str                    # 数字原样（用于判断小数位数）
    unit: str | None = None     # 单位原样
    uid: tuple | None = None    # 单位身份
    start: int = 0
    end: int = 0

    @property
    def dim(self) -> str | None:
        return self.uid[0] if self.uid else None

    def show(self) -> str:
        return f"{self.num} {self.unit}" if self.unit else self.num


def _unit_after(s: str, pos: int) -> tuple[str | None, int]:
    """pos 处（数字之后）紧跟的单位：允许一个空格；取最长的认得的写法。"""
    p = pos + 1 if pos < len(s) and s[pos] == " " else pos
    low = s[p:p + 16].lower()
    for n in range(len(low), 0, -1):
        cand = low[:n]
        if cand[-1].isspace():
            continue
        if _key(cand) in _AFTER_NUMBER:
            nxt = low[n:n + 1]
            if _ASCII_LETTER.match(cand[-1]) and nxt and _ASCII_LETTER.match(nxt):
                continue                                   # 「5 to」不是 5 吨，「5 mins」不认
            return s[p:p + n], p + n
    return None, pos


def quantities(text: str, merge_split_decimals: bool = False) -> list[Quantity]:
    """抽出「数值 + 单位」（单位可能缺省）。区间「49–82 °C」「120~180°F」前一个数沿用后一个数的单位。"""
    s = normalize(text, merge_split_decimals)
    out: list[Quantity] = []
    for m in _NUMBER.finditer(s):
        if out and m.start() < out[-1].end:
            continue                                       # 落在上一个数的单位里（m³/h 规整成 m3/h 后的 3）
        unit, end = _unit_after(s, m.end())
        q = Quantity(float(m.group()), m.group(), unit, unit_id(unit), m.start(), end)
        if out and q.uid and out[-1].uid is None and _RANGE_SEP.fullmatch(s[out[-1].end:m.start()]):
            out[-1].unit, out[-1].uid = unit, q.uid
        out.append(q)
    return out


def mentioned_units(text: str) -> dict[str, dict[tuple, str]]:
    """一段文字里提到过的单位，按量纲分组：{量纲: {单位身份: 原样写法}}。"""
    s = normalize(text).lower()
    out: dict[str, dict[tuple, str]] = {}
    for m in _ANYWHERE_RE.finditer(s):
        uid = UNITS.get(_key(m.group(1)))
        if uid:
            out.setdefault(uid[0], {}).setdefault(uid, m.group(1))
    for q in quantities(text, merge_split_decimals=True):
        if q.uid:
            out.setdefault(q.uid[0], {}).setdefault(q.uid, q.unit)
    return out


# --------------------------------------------------------------------------- 资料
@dataclass
class Source:
    """一段资料（问句、一个片段、一次换算结果）及其预先抽好的数值、单位。

    含「|」的行当作表格行（工具输出、Markdown 表格都是这种写法），单独记下每行的数值和单位，用于按行核对单位。
    """
    text: str
    kind: str = "doc"            # doc：手册片段或问句；calc：换算/核对工具的输出（允许按舍入匹配）
    numbers: set = field(default_factory=set)
    qty: list = field(default_factory=list)
    mentioned: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)          # [(该行的数值集合, 该行提到的全部单位 {身份: 写法})]

    def __post_init__(self):
        # 去掉片段编号和工具输出里包原文的标签（编号里的数字不算资料里的数）
        self.text = _SOURCE_TAG.sub(" ", strip_citations(self.text or ""))
        self.numbers = numbers_in_text(self.text)
        self.qty = quantities(self.text, merge_split_decimals=True)
        self.mentioned = mentioned_units(self.text)
        for line in (self.text or "").splitlines():
            if "|" in line:
                units = {uid: w for by_dim in mentioned_units(line).values() for uid, w in by_dim.items()}
                self.rows.append((numbers_in_text(line), units))


def _decimals(num: str) -> int:
    return len(num.split(".")[1]) if "." in num else 0


def _rounds_to(q: Quantity, exact: float) -> bool:
    """q 是 exact 按 q 的小数位数舍入的结果（3.333 → 3.33 / 3.3 / 3）。"""
    return abs(round(exact, _decimals(q.num)) - q.value) < 1e-9


@dataclass
class Finding:
    status: str                  # ok / no_source（资料里找不到这个数）/ unit_mismatch（单位与资料不符）
    source_units: list[str] = field(default_factory=list)


def _parts(unit: str) -> set[str]:
    return set(re.split(r"[·/.\-]", _key(unit)))


def _compatible(q: Quantity, uid: tuple, written: str) -> bool:
    """给定单位与表格行里写的单位对得上：同一个单位，或行里的单位是给定单位的组成部分（daN 之于 daN·m）。"""
    return uid == q.uid or _key(written) in _parts(q.unit or "")


def ground(q: Quantity, sources: list[Source]) -> Finding:
    """核对一个「数值 + 单位」能否在资料里找到出处（规则见模块说明）。"""
    v = round(q.value, 6)
    conflicts: list[str] = []
    for src in sources:
        if src.kind == "calc":
            for c in src.qty:
                if (abs(c.value - q.value) < 1e-9 or _rounds_to(q, c.value)) and \
                        (q.uid is None or c.uid is None or c.uid == q.uid):
                    return Finding("ok")
        if v not in src.numbers:
            continue
        if q.uid is None:
            return Finding("ok")
        same_dim = [c for c in src.qty if abs(c.value - q.value) < 1e-9 and c.dim == q.dim]
        if any(c.uid == q.uid for c in same_dim):
            return Finding("ok")
        if same_dim:                                   # 资料里这个数明确带着同量纲的别的单位
            conflicts += [c.unit for c in same_dim]
            continue
        # 这个数没带单位：先看它所在的表格行写的单位，再看这段资料提到过的同量纲单位
        row_units = {uid: w for nums, units in src.rows if v in nums for uid, w in units.items()}
        if row_units and not any(_compatible(q, uid, w) for uid, w in row_units.items()):
            conflicts += list(row_units.values())
            continue
        mentioned = src.mentioned.get(q.dim, {})
        if row_units or not mentioned or q.uid in mentioned:
            return Finding("ok")
        conflicts += list(mentioned.values())
    if conflicts:
        return Finding("unit_mismatch", sorted(set(conflicts)))
    return Finding("no_source")


# --------------------------------------------------------------------------- 回答
_CITATION = re.compile(r"[（(]?\s*(?:引用\s*)?(?:chunk[ _-]?id\s*[:：]?\s*)?[\[【]?\s*[A-Za-z0-9_]+_c\d{4}\s*[\]】]?\s*[)）]?",
                       re.I)
_LIST_MARK = re.compile(r"(?m)^\s*(?:[-*+>]\s*)?(?:\d+(?:[.、](?!\d)|[)）])|[（(]\d+[)）])\s*")
_ORDINAL = re.compile(r"第\s*\d+\s*[页章节条步项点次行列级部轮]|\d{4}\s*年")
_SENT_SPLIT = re.compile(r"(?<=[。！？；!?;])|\n")


def strip_citations(text: str) -> str:
    return _CITATION.sub("", text or "")


@dataclass
class Issue:
    sentence: str
    quantity: str
    status: str
    source_units: list[str]

    def describe(self) -> str:
        if self.status == "unit_mismatch":
            return f"「{self.quantity}」：资料里这个数的单位是 {'、'.join(self.source_units)}，不是回答里写的单位"
        return f"「{self.quantity}」：本次的资料和问题里都找不到这个数"


def check_answer(answer: str, sources: list[Source]) -> tuple[int, list[Issue]]:
    """逐句核对回答里的数值。返回 (核对的数值个数, 问题列表)。

    不核对：引用编号、列表序号、「第 3 步」「第 52 页」这类序数和年份、不带单位的 10 以下整数（多是个数、步数）。
    """
    text = _ORDINAL.sub(" ", _LIST_MARK.sub("", strip_citations(answer)))
    n, issues = 0, []
    for sent in _SENT_SPLIT.split(text):
        sent = sent.strip()
        if not sent:
            continue
        for q in quantities(sent):
            if q.uid is None and "." not in q.num and q.value < 10:
                continue
            n += 1
            f = ground(q, sources)
            if f.status != "ok":
                issues.append(Issue(sent[:120], q.show(), f.status, f.source_units))
    return n, issues
