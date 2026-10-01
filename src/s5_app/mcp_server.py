"""MCP Server：把 Agent 的五个工具（检索手册、按行查表、读相邻片段、单位换算、限值核对）用 MCP 暴露出来，
Claude Code、Cursor、Claude Desktop 等客户端可以直接查手册（stdio 传输，官方 Python SDK 2.x 的底层 Server）。

- 与 Agent 共用同一份工具定义和执行器：参数说明、手册编号枚举来自 tools.tool_specs，执行走 tools.run_tool
  （参数校验、超时、出错作为结果返回）。工具都是确定性代码，不调用大模型，不需要 DeepSeek API Key。
- 一个客户端会话对应一个服务进程，进程内保留证据池：check_value 的 source_chunk_id、read_section 的 chunk_id
  必须是本会话里工具返回过的片段，限值数字还要能在该片段原文里找到（与 Agent 相同）。
- 与 Agent 的差别：重复出现的片段也给出正文（abbreviate=False）。客户端自己管理上下文，之前给过的全文可能已被压缩掉。
- 工具出错时返回 isError=true 的结果，错误信息原样给客户端的模型，由它修正参数后重试。
- 检索依赖（torch、embedding 模型、Chroma、BM25）在后台线程加载：握手、列工具都不用等（工具定义来自只依赖
  pydantic 的 tool_schema.py），第一次调用工具时才等加载完成。
- stdout 只用于 MCP 协议：SDK 的 stdio_server 服务期间把 1 号句柄指向 stderr，依赖库的输出写不进协议流；
  所以后台加载放在进入 stdio_server 之后再开始，日志一律写 stderr。

用法（通常由 MCP 客户端启动，见仓库根目录 .mcp.json 与 README「MCP Server」一节）：
    python src/s5_app/mcp_server.py
自检：python scripts/mcp_smoke.py
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
for _p in ("src", "src/s3_eval", "src/s4_agent"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

import anyio  # noqa: E402
import mcp.types as types  # noqa: E402
import yaml  # noqa: E402
from mcp.server import Server, ServerRequestContext  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402

from tool_schema import TOOL_LABEL, tool_specs  # noqa: E402

SERVER_NAME = "equip-manuals"
_CFG = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
AGENT_CFG: dict = _CFG["agent"]
TAU_DISTANCE = float(_CFG["s4_agent"]["tau_distance"])
TOOL_TIMEOUT_S = float(AGENT_CFG["tool_timeout_s"])
LOAD_TIMEOUT_S = 300.0        # 等后台加载的上限（秒）；正常十秒以内完成
DOCUMENTS: dict = (_CFG.get("s5_app") or {}).get("documents") or {}
MANUAL_IDS = list(DOCUMENTS)


def log(msg: str) -> None:
    print(f"[{SERVER_NAME}] {msg}", file=sys.stderr, flush=True)


def render_instructions() -> str:
    """握手时发给客户端的使用说明（configs/prompts/mcp.txt，{catalog} 处填手册清单）。"""
    catalog = "\n".join(f"- {doc_id}：{d.get('title') or doc_id}" for doc_id, d in DOCUMENTS.items())
    raw = (ROOT / _CFG["prompts"]["mcp"]).read_text(encoding="utf-8")
    return raw.replace("{catalog}", catalog).strip()


def _mcp_tools() -> list[types.Tool]:
    """Agent 的工具定义（OpenAI 格式）→ MCP Tool。都只读本地知识库：不改任何东西、不访问外部系统。"""
    out = []
    for spec in tool_specs(MANUAL_IDS):
        fn = spec["function"]
        title = TOOL_LABEL.get(fn["name"])
        out.append(types.Tool(
            name=fn["name"], title=title, description=fn["description"], input_schema=fn["parameters"],
            annotations=types.ToolAnnotations(title=title, read_only_hint=True, destructive_hint=False,
                                              idempotent_hint=True, open_world_hint=False)))
    return out


MCP_TOOLS = _mcp_tools()


# --------------------------------------------------------------------------- 后台加载
_ready = threading.Event()
_state: dict = {}              # tools 模块、工具箱（证据池）；加载失败时为 error
_call_lock = threading.Lock()  # 工具调用逐个执行：证据池只有一份


def _load() -> None:
    t0 = time.perf_counter()
    try:
        import tools                      # 检索依赖较重（torch、sentence-transformers、chromadb），放在后台导入
        tools.get_corpus()
        tools.hybrid_search("水泵", 1)     # 空跑一次检索：加载 embedding 模型、Chroma、BM25、jieba 词典
        _state.update(tools=tools, toolbox=tools.Toolbox(AGENT_CFG, TAU_DISTANCE, abbreviate=False))
        log(f"知识库加载完成，用时 {(time.perf_counter() - t0) * 1000:.0f} ms")
    except Exception as exc:  # 加载失败也要让等待方醒来，并把原因报给客户端
        _state["error"] = f"{type(exc).__name__}: {exc}"
        log(f"知识库加载失败：{_state['error']}")
    finally:
        _ready.set()


async def _wait_ready() -> None:
    if not _ready.is_set():
        await anyio.to_thread.run_sync(_ready.wait, LOAD_TIMEOUT_S)
    if not _ready.is_set():
        raise RuntimeError(f"知识库加载超过 {LOAD_TIMEOUT_S:g} 秒仍未完成")
    if "error" in _state:
        raise RuntimeError(f"知识库加载失败：{_state['error']}")


def _call(name: str, arguments: dict):
    with _call_lock:
        return _state["tools"].run_tool(_state["toolbox"], name, arguments, TOOL_TIMEOUT_S)


# --------------------------------------------------------------------------- MCP 处理函数
async def on_list_tools(ctx: ServerRequestContext, params: types.PaginatedRequestParams | None) -> types.ListToolsResult:
    return types.ListToolsResult(tools=MCP_TOOLS)


async def on_call_tool(ctx: ServerRequestContext, params: types.CallToolRequestParams) -> types.CallToolResult:
    """参数由 Toolbox.run 用 Pydantic 校验（与 Agent 一致：可选参数传 null 也接受，错误信息是中文）。"""
    try:
        await _wait_ready()
    except RuntimeError as exc:
        return types.CallToolResult(content=[types.TextContent(type="text", text=str(exc))], is_error=True)
    args = params.arguments or {}
    res, ms = await anyio.to_thread.run_sync(_call, params.name, args)
    log(f"{params.name} {json.dumps(args, ensure_ascii=False)} → {res.summary}（{ms:g} ms）")
    return types.CallToolResult(content=[types.TextContent(type="text", text=res.content)], is_error=not res.ok)


server = Server(SERVER_NAME, version="0.1.0", title="泵类设备维修手册",
                description="检索泵类设备维修手册、按行查表、单位换算与限值核对（确定性工具，只读）",
                instructions=render_instructions(), on_list_tools=on_list_tools, on_call_tool=on_call_tool)


async def _serve() -> None:
    async with stdio_server() as (read_stream, write_stream):
        # 进入 stdio_server 后 1 号句柄已指向 stderr，这时再开始加载，依赖库的任何输出都写不进协议流
        threading.Thread(target=_load, name="kb-load", daemon=True).start()
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    anyio.run(_serve)


if __name__ == "__main__":
    main()
