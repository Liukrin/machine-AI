"""表格按行展开与查表（src/s4_agent/tables.py），工具参数定义（src/s4_agent/tool_schema.py）。"""
import pytest
from pydantic import ValidationError

from tables import TableStore, table_rows
from tool_schema import TOOL_ARGS, tool_specs

TORQUE = ('<table><tr><td>零件</td><td colspan="2">扭矩</td></tr>'
          '<tr><td rowspan="2">泵轴螺母</td><td>M12</td><td>40 N·m</td></tr>'
          '<tr><td>M16</td><td>95 N·m</td></tr></table>')
CHUNKS = [
    {"chunk_id": "9_c0001", "doc_id": "9", "chunk_type": "table", "heading_path": "4.2 拧紧力矩", "table_html": TORQUE},
    {"chunk_id": "8_c0001", "doc_id": "8", "chunk_type": "table", "heading_path": "附录 允许负荷",
     "table_html": "<table><tr><td>管径</td><td>Mz</td></tr><tr><td>200</td><td>85</td></tr></table>"},
    {"chunk_id": "8_c0002", "doc_id": "8", "chunk_type": "text", "heading_path": "x", "text": "正文片段不入表格库"},
]


def test_table_rows_expands_rowspan_and_keeps_colspan_once():
    assert table_rows(TORQUE) == [["零件", "扭矩"], ["泵轴螺母", "M12", "40 N·m"], ["泵轴螺母", "M16", "95 N·m"]]


@pytest.fixture(scope="module")
def store():
    return TableStore(CHUNKS)


def test_store_counts(store):
    assert (store.n_tables, store.n_rows) == (2, 5)
    assert store.n_rows_of("9_c0001") == 3 and store.n_rows_of("不存在") == 0


@pytest.mark.parametrize("keywords, doc_id, hits", [
    (["M16"], None, [("9_c0001", 2)]),
    (["泵轴螺母", "M12"], None, [("9_c0001", 1)]),     # rowspan 展开后每行都带上「泵轴螺母」
    (["扭矩", "M16"], None, [("9_c0001", 2)]),         # 关键词可以出现在表头
    (["拧紧力矩"], None, []),                          # 但至少一个要出现在行里
    (["200"], "8", [("8_c0001", 1)]),
    (["200"], "9", []),                                # 按手册过滤
    (["M16", "200"], None, []),                        # 全部关键词都要命中
])
def test_lookup(store, keywords, doc_id, hits):
    found, total = store.lookup(keywords, doc_id)
    assert [(h["chunk_id"], h["row_idx"]) for h in found] == hits and total == len(hits)


def test_tool_specs():
    specs = tool_specs(["1", "2", "4"])
    names = [s["function"]["name"] for s in specs]
    assert names == ["search_manuals", "lookup_table", "read_section", "convert_unit", "check_value"]
    assert sorted(TOOL_ARGS) == sorted(names)
    manual = specs[0]["function"]["parameters"]["properties"]["manual"]
    assert manual["enum"] == ["1", "2", "4"]           # 手册编号做成枚举，模型填不出不存在的手册


def test_tool_args_validation():
    TOOL_ARGS["convert_unit"](value=85, from_unit="daN·m", to_unit="N·m")
    with pytest.raises(ValidationError):
        TOOL_ARGS["convert_unit"](value=85, from_unit="daN·m")              # 缺 to_unit
    with pytest.raises(ValidationError):
        TOOL_ARGS["search_manuals"](query="泵", kind="image")               # kind 只能是 text / table
