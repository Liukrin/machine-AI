"""MCP Server 自检：用官方 Python SDK 的客户端以 stdio 方式启动 src/s5_app/mcp_server.py，
握手、列工具，再逐个调用工具核对结果（含应当报错的情况）。不调用大模型，不需要 API Key。

用法：
    python scripts/mcp_smoke.py
全部通过时退出码为 0。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "src" / "s5_app" / "mcp_server.py"
EXPECTED_TOOLS = {"search_manuals", "lookup_table", "read_section", "convert_unit", "check_value"}

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        failures.append(msg)


def text_of(result) -> str:
    return "\n".join(getattr(c, "text", "") for c in result.content)


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], cwd=str(ROOT))
    t0 = time.perf_counter()
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        init = await session.initialize()
        t_init = (time.perf_counter() - t0) * 1000
        check(init.server_info.name == "equip-manuals" and "{catalog}" not in (init.instructions or "")
              and "4：Model 3700" in (init.instructions or ""),
              f"握手 {t_init:.0f} ms：服务名 {init.server_info.name}，使用说明 {len(init.instructions or '')} 字（含手册清单）")

        t1 = time.perf_counter()
        tools = (await session.list_tools()).tools
        t_list = (time.perf_counter() - t1) * 1000
        names = {t.name for t in tools}
        check(names == EXPECTED_TOOLS, f"列出 5 个工具（{t_list:.0f} ms，不等检索依赖加载）：{sorted(names)}")
        check(all(t.annotations and t.annotations.read_only_hint and t.annotations.open_world_hint is False for t in tools),
              "工具都标注为只读、不访问外部系统")
        search = next(t for t in tools if t.name == "search_manuals")
        check("4" in search.input_schema["properties"]["manual"].get("enum", []), "manual 参数带手册编号枚举（与 Agent 同一份定义）")

        async def call(name: str, args: dict):
            t = time.perf_counter()
            r = await session.call_tool(name, args)
            return r, text_of(r), (time.perf_counter() - t) * 1000

        q = "Model 3700 轴承温度实测 185°F，在手册要求的范围内吗？"
        r, txt, ms = await call("search_manuals", {"query": q})
        check(not r.is_error and "[4_c0151]" in txt,
              f"search_manuals 检索到限值片段 4_c0151（首次调用 {ms:.0f} ms，含等待后台加载）")

        r, txt, ms = await call("search_manuals", {"query": "水泵与电动机同心度允差"})
        check(not r.is_error and "片段" in txt, f"加载完成后再检索一次（{ms:.0f} ms）")

        r, txt, ms = await call("check_value", {"value": 185, "unit": "°F", "source_chunk_id": "4_c0151",
                                                "limit_max": 180, "limit_unit": "°F"})
        check(not r.is_error and "超出上限" in txt, f"check_value 185 °F 对上限 180 °F → 超出上限（{ms:.0f} ms）")

        r, txt, _ = await call("check_value", {"value": 185, "unit": "°F", "source_chunk_id": "4_c0151",
                                               "limit_max": 200, "limit_unit": "°F"})
        check(r.is_error and "找不到" in txt, "check_value 用原文里没有的限值 200 → 报错（限值必须出自片段原文）")

        r, txt, _ = await call("read_section", {"chunk_id": "1_c0001"})
        check(r.is_error and "还没有在本次对话中出现过" in txt, "read_section 读本会话没出现过的片段 → 报错")

        r, txt, _ = await call("read_section", {"chunk_id": "4_c0151", "before": 0, "after": 1})
        check(not r.is_error and "[4_c0152]" in txt, "read_section 读到 4_c0151 的后一个片段")

        r, txt, _ = await call("convert_unit", {"value": 0.2, "from_unit": "m³/h", "to_unit": "L/min"})
        check(not r.is_error and "3.333" in txt, "convert_unit 0.2 m³/h → 3.333 L/min")

        r, txt, _ = await call("lookup_table", {"keywords": "轴封水量", "manual": None})
        check(not r.is_error and "[2_c0044]" in txt and "125~200 | 0.20" in txt,
              "lookup_table 查「轴封水量」命中 2_c0044 的「125~200 | 0.20」行（manual 传 null 也接受）")

        r, txt, _ = await call("search_manuals", {"query": q})
        check(not r.is_error and "前面已给出全文" not in txt and "[4_c0151]" in txt, "同一检索再来一次：仍给出正文（不省略）")

        r, txt, _ = await call("convert_unit", {"value": 1, "from_unit": "furlong", "to_unit": "m"})
        check(r.is_error and "工具调用失败" in txt, "convert_unit 不认识的单位 → 报错，错误信息回给模型")

        r, txt, _ = await call("no_such_tool", {})
        check(r.is_error, "调用不存在的工具 → 报错")

    print(f"\n{'全部通过' if not failures else f'{len(failures)} 项失败'}（共用时 {time.perf_counter() - t0:.1f} 秒）")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    anyio.run(main)
