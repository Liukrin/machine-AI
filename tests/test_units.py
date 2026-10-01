"""单位换算、限值比较、原文数字提取（src/s4_agent/units.py）。安全红线的判定只由这里的确定性代码做。"""
import pytest

from units import UnitError, compare, convert, fmt, numbers_in_text, same_unit, unit_display


@pytest.mark.parametrize("value, src, dst, expected", [
    (185, "°F", "°C", 85.0),
    (85, "℃", "°F", 185.0),
    (85, "daN·m", "N·m", 850.0),
    (0.2, "m³/h", "L/min", 3.33333),
    (1, "bar", "kPa", 100.0),
    (1, "in", "mm", 25.4),
])
def test_convert(value, src, dst, expected):
    assert convert(value, src, dst) == pytest.approx(expected, rel=1e-5)


def test_temperature_difference_has_no_offset():
    """温差、温升只乘 9/5，不加 32。"""
    assert convert(10, "°C", "°F", is_difference=True) == pytest.approx(18.0)


def test_convert_refuses_different_dimensions():
    with pytest.raises(UnitError, match="量纲不同"):
        convert(1, "mm", "°C")


def test_unit_spellings():
    assert same_unit("℃", "°C")
    assert unit_display("摄氏度") == unit_display("℃") == "°C"


@pytest.mark.parametrize("args, verdict, margin", [
    ((185, "°F", None, 180, "°F"), "above_max", 5.0),
    ((80, "°C", 49, 82, "°C"), "within", None),
    ((82, "°C", 49, 82, None), "at_limit", 0.0),
    ((40, "°C", 49, 82, "°C"), "below_min", 9.0),
    ((190, "°F", None, 82, "°C"), "above_max", 5.7778),     # 先换算成限值单位（87.78 °C）再比
])
def test_compare(args, verdict, margin):
    r = compare(*args)
    assert r["verdict"] == verdict
    assert r["margin"] == (None if margin is None else pytest.approx(margin, abs=1e-4))


@pytest.mark.parametrize("args, msg", [
    ((1, "°C", None, None, None), "至少要给一个"),
    ((1, "°C", 5, 3, None), "大于上限"),
])
def test_compare_rejects_bad_limits(args, msg):
    with pytest.raises(UnitError, match=msg):
        compare(*args)


@pytest.mark.parametrize("x, shown", [(3.33333, "3.333"), (850.0, "850"), (1234.56, "1234.6"), (0.0254, "0.0254"),
                                      (87.7777, "87.78"), (0, "0")])
def test_fmt(x, shown):
    assert fmt(x) == shown


def test_numbers_in_text_lenient_readings():
    # MinerU 把「2.5」拆成「2. 5」：拆开、合并两种读法都收；「3,000」按千分位和小数逗号两种读
    assert {2.5, 3000.0, 3.0} <= numbers_in_text("间隙 2. 5 mm，转速 3,000 r/min")
    # 上标不能和后面的数并成一个数
    assert numbers_in_text("kg/cm²100 psig") == {2.0, 100.0}
    # 「<2.000」不是 HTML 标签，不能和后面的结束标签之间的内容一起被删掉（阶段 3 的 bug，见 process_log 23）
    assert 2.0 in numbers_in_text("<2.000 及以上 </片段原文>")
    # LaTeX 块里被空格拆开的数字
    assert 12.5 in numbers_in_text("$$ 1 2 . 5 $$")
