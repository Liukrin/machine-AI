"""确定性的单位换算与限值比较，供 Agent 的 convert_unit / check_value 工具使用。不调用 LLM。

CLAUDE.md 规定物理阈值判决一律由确定性 Python 执行：模型只负责从手册里找到限值并把参数交给这里，
「是否超限」由本模块比较得出，模型按工具返回的结论表述。

换算系数取自国际单位制定义值（1 in = 25.4 mm、1 lb = 0.45359237 kg、g = 9.80665 m/s²、
1 US gal = 3.785411784 L 等），不做近似；显示时保留 4 位有效数字。
"""
from __future__ import annotations

import math
import re
import unicodedata

_PSI = 0.45359237 * 9.80665 / 0.0254 ** 2      # 1 psi = 6894.757… Pa
_GAL = 3.785411784e-3                           # 1 US gal = 3.785411784 L
_LBF = 0.45359237 * 9.80665                     # 1 lbf = 4.448… N

# 量纲 -> (基准单位, {单位写法: 1 单位等于多少基准单位})。写法一律是 _key() 归一化之后的形式。
_LINEAR: dict[str, tuple[str, dict[str, float]]] = {
    "长度": ("m", {
        "m": 1, "米": 1, "mm": 1e-3, "毫米": 1e-3, "cm": 1e-2, "厘米": 1e-2, "um": 1e-6, "μm": 1e-6,
        "微米": 1e-6, "丝": 1e-5, "in": 0.0254, "inch": 0.0254, "英寸": 0.0254, '"': 0.0254,
        "ft": 0.3048, "英尺": 0.3048, "mil": 2.54e-5, "密耳": 2.54e-5, "thou": 2.54e-5,
    }),
    "压力": ("Pa", {
        "pa": 1, "帕": 1, "kpa": 1e3, "千帕": 1e3, "mpa": 1e6, "兆帕": 1e6, "bar": 1e5, "巴": 1e5,
        "mbar": 100, "psi": _PSI, "psig": _PSI, "lbf/in2": _PSI, "磅/平方英寸": _PSI,
        "kgf/cm2": 98066.5, "kg/cm2": 98066.5, "公斤/平方厘米": 98066.5, "公斤力/平方厘米": 98066.5,
        "atm": 101325, "标准大气压": 101325, "mh2o": 9806.65, "米水柱": 9806.65, "mwc": 9806.65,
        "mmhg": 133.322387415, "毫米汞柱": 133.322387415,
    }),
    "流量": ("m3/s", {
        "m3/s": 1, "m3/h": 1 / 3600, "立方米/小时": 1 / 3600, "立方米/时": 1 / 3600,
        "l/s": 1e-3, "升/秒": 1e-3, "l/min": 1e-3 / 60, "lpm": 1e-3 / 60, "升/分": 1e-3 / 60,
        "升/分钟": 1e-3 / 60, "l/h": 1e-3 / 3600, "升/小时": 1e-3 / 3600, "lh": 1e-3 / 3600,
        "ml/h": 1e-6 / 3600, "ml/min": 1e-6 / 60, "gpm": _GAL / 60, "usgpm": _GAL / 60,
        "gal/min": _GAL / 60, "加仑/分钟": _GAL / 60,
    }),
    "体积": ("m3", {"m3": 1, "立方米": 1, "l": 1e-3, "升": 1e-3, "ml": 1e-6, "毫升": 1e-6,
                   "gal": _GAL, "加仑": _GAL}),
    "质量": ("kg", {"kg": 1, "千克": 1, "公斤": 1, "g": 1e-3, "克": 1e-3, "t": 1e3, "吨": 1e3,
                   "lb": 0.45359237, "磅": 0.45359237, "oz": 0.028349523125, "盎司": 0.028349523125}),
    "力": ("N", {"n": 1, "牛": 1, "kn": 1e3, "千牛": 1e3, "dan": 10, "kgf": 9.80665, "公斤力": 9.80665,
                "lbf": _LBF}),
    "扭矩": ("N·m", {
        "n·m": 1, "nm": 1, "n.m": 1, "n-m": 1, "牛·米": 1, "牛米": 1, "kn·m": 1e3, "kn.m": 1e3, "knm": 1e3,
        "dan·m": 10, "dan.m": 10, "danm": 10, "dan-m": 10, "n·cm": 0.01,
        "kgf·m": 9.80665, "kgf.m": 9.80665, "kg·m": 9.80665, "公斤力·米": 9.80665, "kgf·cm": 0.0980665,
        "lbf·ft": _LBF * 0.3048, "ft·lb": _LBF * 0.3048, "lb·ft": _LBF * 0.3048, "ft-lb": _LBF * 0.3048,
        "lbf-ft": _LBF * 0.3048, "英尺-磅": _LBF * 0.3048, "英尺磅": _LBF * 0.3048,
        "lbf·in": _LBF * 0.0254, "in·lb": _LBF * 0.0254, "in-lb": _LBF * 0.0254, "英寸-磅": _LBF * 0.0254,
    }),
    "功率": ("W", {"w": 1, "瓦": 1, "kw": 1e3, "千瓦": 1e3, "hp": 745.69987158227022, "英制马力": 745.69987158227022,
                  "ps": 735.49875, "马力": 735.49875}),
    "线速度": ("m/s", {"m/s": 1, "米/秒": 1, "ft/s": 0.3048, "英尺/秒": 0.3048, "ft/min": 0.3048 / 60,
                     "m/min": 1 / 60, "km/h": 1 / 3.6}),
    "转速": ("r/min", {"r/min": 1, "rpm": 1, "转/分": 1, "转/分钟": 1, "r/s": 60, "rad/s": 60 / (2 * math.pi)}),
    "时间": ("s", {"s": 1, "秒": 1, "min": 60, "分钟": 60, "h": 3600, "小时": 3600, "d": 86400, "天": 86400,
                  "周": 604800}),
    "比例": ("%", {"%": 1, "‰": 0.1}),
}
_TEMPERATURE = {
    "°c": "C", "c": "C", "摄氏度": "C", "degc": "C", "celsius": "C",
    "°f": "F", "f": "F", "华氏度": "F", "degf": "F", "fahrenheit": "F",
    "k": "K", "开": "K", "开尔文": "K", "kelvin": "K",
}
_TEMP_SHOW = {"C": "°C", "F": "°F", "K": "K"}


class UnitError(ValueError):
    """单位写法不认识或量纲不一致。错误信息直接回给模型，让它改参数再试。"""


def _key(unit: str) -> str:
    """单位写法归一化：全角转半角（℃→°C、³→3）、小写、去空白，乘号统一成 ·。"""
    s = unicodedata.normalize("NFKC", unit or "").lower()
    s = re.sub(r"\s+", "", s).replace("^", "").replace("*", "·").replace("×", "·")
    s = s.replace("º", "°").replace("˚", "°")
    return s


def _lookup(unit: str) -> tuple[str, float | str]:
    """返回 (量纲, 系数)；温度返回 ("温度", "C"|"F"|"K")。"""
    k = _key(unit)
    if k in _TEMPERATURE:
        return "温度", _TEMPERATURE[k]
    for dim, (_base, table) in _LINEAR.items():
        if k in table:
            return dim, table[k]
    raise UnitError(f"不认识的单位「{unit}」。可用的写法例如：mm、in、MPa、bar、psi、kgf/cm²、m³/h、L/min、gpm、"
                    "°C、°F、N·m、lbf·ft、kgf·m、kW、hp、m/s、ft/s、r/min、%")


def _to_kelvin(v: float, scale: str) -> float:
    return {"C": v + 273.15, "F": (v - 32) * 5 / 9 + 273.15, "K": v}[scale]


def _from_kelvin(v: float, scale: str) -> float:
    return {"C": v - 273.15, "F": (v - 273.15) * 9 / 5 + 32, "K": v}[scale]


def convert(value: float, from_unit: str, to_unit: str, *, is_difference: bool = False) -> float:
    """把 value 从 from_unit 换算到 to_unit。is_difference=True 表示温差/温升（°C→°F 只乘 9/5，不加 32）。"""
    if same_unit(from_unit, to_unit):
        return float(value)
    d1, f1 = _lookup(from_unit)
    d2, f2 = _lookup(to_unit)
    if d1 != d2:
        raise UnitError(f"「{from_unit}」是{d1}单位，「{to_unit}」是{d2}单位，量纲不同，不能换算")
    if d1 == "温度":
        if is_difference:
            per_kelvin = {"C": 1.0, "K": 1.0, "F": 9 / 5}
            return value / per_kelvin[f1] * per_kelvin[f2]
        return _from_kelvin(_to_kelvin(value, f1), f2)
    return value * f1 / f2


def unit_display(unit: str) -> str:
    """单位的展示写法：温度统一成 °C/°F/K，其余原样（去掉首尾空白；不认识的单位也原样）。"""
    try:
        dim, f = _lookup(unit)
    except UnitError:
        return unit.strip()
    return _TEMP_SHOW[f] if dim == "温度" else unit.strip()


def same_unit(a: str, b: str) -> bool:
    """两个单位写法是否相同（归一化后比较；℃ 与 °C 算相同）。"""
    return _key(a) == _key(b)


def fmt(x: float) -> str:
    """数值展示：4 位有效数字，不用科学计数法；绝对值 ≥ 1000 时保留 1 位小数。"""
    if x == 0 or not math.isfinite(x):
        return "0" if x == 0 else str(x)
    if abs(x) >= 1000:
        s = f"{x:.1f}"
    else:
        digits = 4 - int(math.floor(math.log10(abs(x)))) - 1
        s = f"{round(x, digits):.{max(digits, 0)}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


# --------------------------------------------------------------------------- 限值比较
VERDICT_CN = {
    "above_max": "超出上限", "below_min": "低于下限", "at_limit": "恰好等于限值（在允许范围内）",
    "within": "在允许范围内",
}


def compare(value: float, unit: str, limit_min: float | None, limit_max: float | None,
            limit_unit: str | None = None) -> dict:
    """把 value（unit）换算到限值单位后与 [limit_min, limit_max] 比较。结论只由这里的数值比较决定。"""
    if limit_min is None and limit_max is None:
        raise UnitError("limit_min 和 limit_max 至少要给一个")
    if limit_min is not None and limit_max is not None and limit_min > limit_max:
        raise UnitError(f"下限 {limit_min} 大于上限 {limit_max}，请检查")
    lu = limit_unit or unit
    # 单位相同就直接比，不需要认识这个单位（如「°C/min」「滴/分钟」）；不同才换算
    v = float(value) if same_unit(unit, lu) else convert(value, unit, lu)
    tol = 1e-9 * max(1.0, abs(v))
    if limit_max is not None and v > limit_max + tol:
        verdict, margin, base = "above_max", v - limit_max, limit_max
    elif limit_min is not None and v < limit_min - tol:
        verdict, margin, base = "below_min", limit_min - v, limit_min
    elif any(lim is not None and abs(v - lim) <= tol for lim in (limit_min, limit_max)):
        verdict, margin, base = "at_limit", 0.0, None
    else:
        verdict, margin, base = "within", None, None
    return {
        "verdict": verdict, "verdict_cn": VERDICT_CN[verdict],
        "value": value, "unit": unit_display(unit), "value_in_limit_unit": v, "limit_unit": unit_display(lu),
        "limit_min": limit_min, "limit_max": limit_max,
        "margin": margin, "margin_pct": (margin / abs(base) * 100) if margin and base else None,
    }


# --------------------------------------------------------------------------- 原文数字
_SUPERSCRIPT = {"²": "^2 ", "³": "^3 "}
_SPLIT_DECIMAL = re.compile(r"(?<=\d)\.\s+(?=\d)")          # MinerU 把「2.5」拆成「2. 5」
_LATEX = re.compile(r"\$\$.*?\$\$|\$[^$]*\$", re.S)     # 块级公式 $$…$$ 先认，不然会被当成两对空的 $$
_NUMBER = re.compile(r"(?<![\d.])\d+(?:\.\d+)?(?![\d]|\.\d)")


def numbers_in_text(text: str) -> set[float]:
    """原文里出现过的数值，用来核对限值、回答里的数是否真出自资料。

    宽松读法，几种都收：MinerU 把「2.5」拆成的「2. 5」按拆开、合并两种读；「3,000」按千分位（3000）和
    小数逗号（3.000，OCR 常把小数点认成逗号，如 Model 3700 磨损环间隙表）两种读。
    """
    s = text or ""
    for a, b in _SUPERSCRIPT.items():
        s = s.replace(a, b)                                 # 「kg/cm²100 psig」不能并成 2100
    s = _LATEX.sub(lambda m: re.sub(r"(?<=\d) (?=[\d.])|(?<=\.) (?=\d)", "", m.group(0)), s)
    s = unicodedata.normalize("NFKC", s).replace("\\", "")
    # 只去掉真正的 HTML 标签（标签名以字母开头）。原先的 <[^>]+> 会把「<2.000 … >」这种比较写法之间的整段当成标签删掉
    s = re.sub(r"</?[A-Za-z][^<>]*>", " ", s)
    comma_decimal = re.sub(r"(?<=\d),(?=\d)", ".", s)
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    found = {round(float(x), 6) for x in _NUMBER.findall(s)}
    found |= {round(float(x), 6) for x in _NUMBER.findall(_SPLIT_DECIMAL.sub(".", s))}
    found |= {round(float(x), 6) for x in _NUMBER.findall(comma_decimal)}
    return found
