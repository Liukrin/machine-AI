"""Agent 五个工具的参数定义与说明（发给模型的 JSON Schema），以及前端、MCP 用的中文名。

只依赖 pydantic，导入很快：MCP Server 列工具时不必等检索依赖（torch、chromadb 等）加载完。
工具的实现在 tools.py（它从这里导入 TOOL_ARGS 做参数校验）。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- 工具参数（发给模型的 JSON Schema）
class SearchManualsArgs(BaseModel):
    """在维修手册中检索（BM25 + 向量混合检索），返回最相关的若干片段及其 chunk_id。"""
    query: str = Field(description="检索问句或关键词。写全设备型号和部件名称，用手册里的说法，不要用代词或「那个」")
    manual: str | None = Field(None, description="只在某一本手册里检索，填手册编号（见手册清单）；不确定就不填")
    kind: Literal["text", "table"] | None = Field(None, description="只检索正文（text）或只检索表格（table）；不填则都检索")


class LookupTableArgs(BaseModel):
    """在手册的表格里按关键词查找行（零件编号、故障原因、扭矩、规格参数等），返回命中的行和表头。"""
    keywords: str = Field(description="关键词，多个用空格分开，命中的行须同时包含全部关键词（如「泵轴」「Fy」「电机 电力」）。"
                                      "关键词要短，用手册表格里会出现的字眼")
    manual: str | None = Field(None, description="只查某一本手册的表格，填手册编号；不确定就不填")


class ReadSectionArgs(BaseModel):
    """读取某个片段前后相邻的片段（同一本手册），用于答案被切到相邻片段、需要上下文的情况。"""
    chunk_id: str = Field(description="本次对话里出现过的 chunk_id")
    before: int = Field(1, ge=0, le=2, description="向前读几个片段（0~2）")
    after: int = Field(1, ge=0, le=2, description="向后读几个片段（0~2）")


class ConvertUnitArgs(BaseModel):
    """单位换算（确定性计算）。凡是答案里需要换算单位的数值，都必须用它算，不要心算。"""
    value: float = Field(description="要换算的数值")
    from_unit: str = Field(description="原单位，如 m³/h、bar、°C、N·m、mm")
    to_unit: str = Field(description="目标单位，如 L/min、psi、°F、kgf·m、in")
    is_difference: bool = Field(False, description="是否为温差/温升（如「高出 3℃」）：是则 °C→°F 只乘 9/5、不加 32")


class CheckValueArgs(BaseModel):
    """判断一个实测值或设定值是否在手册规定的范围内（确定性比较，结论以本工具为准）。
    限值必须是本次对话里某个片段原文写明的数值，并给出该片段的 chunk_id；单位不同会先换算再比较。"""
    value: float = Field(description="实测值或设定值")
    unit: str = Field(description="value 的单位")
    source_chunk_id: str = Field(description="写有该限值的片段 chunk_id（必须是本次对话里工具返回过的片段）")
    limit_max: float | None = Field(None, description="上限：照抄片段原文写明的数值（手册写「不超过」「最大」「≤」的那个数），"
                                                      "不要自己换算；没有上限就不填")
    limit_min: float | None = Field(None, description="下限：照抄片段原文写明的数值（手册写「不低于」「至少」「≥」的那个数），"
                                                      "不要自己换算；没有下限就不填")
    limit_unit: str | None = Field(None, description="限值在原文里的单位。与 unit 不同时由工具换算后再比较；相同可不填")


TOOL_ARGS: dict[str, type[BaseModel]] = {
    "search_manuals": SearchManualsArgs,
    "lookup_table": LookupTableArgs,
    "read_section": ReadSectionArgs,
    "convert_unit": ConvertUnitArgs,
    "check_value": CheckValueArgs,
}
TOOL_LABEL = {"search_manuals": "检索手册", "lookup_table": "查表", "read_section": "读相邻片段",
              "convert_unit": "单位换算", "check_value": "限值核对"}


def tool_specs(manual_ids: list[str]) -> list[dict]:
    """OpenAI 格式的工具定义（bind_tools 用）。manual 参数列出可选的手册编号。"""
    specs = []
    for name, model in TOOL_ARGS.items():
        schema = model.model_json_schema()
        schema.pop("title", None)
        schema.pop("description", None)          # 与函数描述重复
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
            if prop.get("anyOf") and any(x.get("type") == "null" for x in prop["anyOf"]):
                # Optional[X] 展开成 X，并标注可以不填（DeepSeek 对 anyOf 的支持不如普通类型稳定）
                inner = [x for x in prop.pop("anyOf") if x.get("type") != "null"][0]
                prop.update(inner)
                prop.pop("default", None)
        if "manual" in schema.get("properties", {}):
            schema["properties"]["manual"]["enum"] = manual_ids
        specs.append({"type": "function", "function": {
            "name": name, "description": (model.__doc__ or "").strip().replace("\n    ", ""), "parameters": schema}})
    return specs
