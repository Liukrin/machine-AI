# 前端（Vite + React + TypeScript + Tailwind CSS）

设备运维知识问答的单页前端，开发期通过 Vite 代理对接后端 `http://localhost:8000`。

## 功能

- **两种模式**：输入框下方切换「Agent」（默认，工具调用 Agent）与「直接检索」（rag 模式，阶段 1 的固定流水线）；选择保存在本浏览器
- **查阅过程时间线**（Agent 模式）：实时显示每一步——系统预检索（标「自动」）、检索、查表、读相邻片段、单位换算、限值核对，含参数、结果摘要与耗时；工具出错标红并给出原因；回答结束后折叠成一行摘要，可点开
- **流式问答**：先展示检索到的手册片段（手册名 / 章节 / 页码；Agent 模式下随每次工具调用累计），再逐字输出回答，可随时停止生成
- **引用角标**：回答里的 chunk_id 引用（`[4_c0319]`、`（chunk_id: 4_c0026）`、`【2_c0044】` 等写法）渲染成来源序号角标，悬停高亮对应卡片，点击打开原文；表格按原始行列结构还原
- **状态区分**：拒答闸拦截（rag 模式）、资料不足（Agent 模式会附上查过什么）、请求失败分别展示，不混进普通答案；回答被生成长度上限截断（`finish_reason=length`）时额外提示
- **引用校验**：展示引用是否都指向本次检索结果、低重合句提示，以及模型调用次数、token 数与耗时
- **数值核对**（Agent 模式）：回答底部显示「N 个数值均在资料中核对到」或哪几个数核对不上（悬停看原因）；数值对不上触发了改写时，查阅过程里多一行说明，回答区换成改写后的版本，以后端最终采用的答案为准
- **👍/👎**：每个回答（包括拒答）下方可以评价，点 👎 会弹出一个可选的说明框；评价写进后端的请求日志，后端没开日志时不显示
- **自动重连**：打开页面时连不上后端（比如后端还在加载模型），按 1、2、4 秒递增重试，最长间隔 10 秒，连上后自动恢复
- **对话历史**：保存在浏览器 localStorage，刷新不丢；Agent 模式下把本会话最近的问答一起发给后端，追问可以省略设备型号（后端只取最近 3 轮）

## 依赖安装

```bash
npm install
```

## 启动

```bash
npm run dev       # http://localhost:5173，/api 自动代理到 http://localhost:8000
npm run typecheck # TypeScript 类型检查（CI 和前端镜像构建时都会跑）
npm run build     # 产物输出到 dist/
npm run preview   # 预览构建产物
```

后端先起：`python -m uvicorn src.s5_app.api:app --host 127.0.0.1 --port 8000`（在项目根目录执行）。

## 目录

```text
src/
├── App.tsx               # 会话状态、流式请求、整体布局
├── api.ts                # 接口类型、SSE 流解析、片段原文拉取
├── types.ts
├── lib/
│   ├── citations.ts      # 引用标记改写为角标；模型自述拒答识别
│   ├── format.ts         # 章节标题、页码、时间等展示格式
│   └── storage.ts        # 对话历史本地存储
└── components/
    ├── Sidebar.tsx       # 对话历史 + 知识库状态
    ├── TopBar.tsx
    ├── Welcome.tsx       # 空状态：示例问题、已收录手册
    ├── TurnView.tsx      # 单轮问答：来源、回答、校验、拒答 / 错误状态
    ├── Steps.tsx         # Agent 的查阅过程时间线
    ├── Sources.tsx       # 来源卡片横向列表
    ├── SourceDrawer.tsx  # 原文详情抽屉
    ├── Composer.tsx      # 输入框（兼容中文输入法回车选词）+ 模式切换
    └── icons.tsx
```

## 接口

| 接口 | 用途 |
|---|---|
| `GET /api/health` | 服务状态、片段总数、生成模型、默认模式、已收录文档清单 |
| `POST /api/ask` | SSE 流式问答，请求体 `{question, history, mode}`。rag 模式事件顺序 `retrieval → (rejected \| token* + verification + done) / error`；Agent 模式 `step`（模型调用开始、工具开始/结束、中间说明、改写）与 `retrieval`（累计来源）穿插，`token` 带 `round`，之后 `verification`、`done`。结束事件带 `request_id`（后端开了请求日志时） |
| `POST /api/feedback` | 👍/👎：`{request_id, rating: 1 \| -1 \| null, comment?}`，`null` 为撤销 |
| `GET /api/chunks/{chunk_id}` | 单个片段原文（正文或表格 HTML），打开来源详情时按需拉取 |

表格 HTML 来自 MinerU 解析结果，前端用 `DOMParser` 只提取单元格文本与合并信息后重建表格，不直接注入原始 HTML。

## SSE 解析要点

后端 `POST /api/ask` 返回 SSE 流，但浏览器 `EventSource` 不支持 POST，须用 `fetch` + `response.body.getReader()` 手动读流。三个关键点：

1. **按事件边界切分**。事件以空行分隔；后端 sse-starlette 用 `\r\n` 行尾。切分须兼容 `\r\n` / `\n` / `\r` 三种行尾（正则 `/\r?\n\r?\n/`），**不要硬编码 `\n\n`**——否则 CRLF 流里永远匹配不到，事件会积压到流结束才一次性派发（曾导致只渲染出残缺 done 行、拒答路径崩溃）。

2. **缓冲区拼接半行**。`TextDecoder.decode(value, { stream: true })` 增量解码，跨 chunk 截断的多字节字符与半行留在缓冲区，读到完整空行才派发一个事件块；流结束再用 `decode()`（不带 stream）flush 残余字节。

3. **多行 `data:` 合并**。一个事件块可含多行 `data:`，需 `join('\n')` 后再 `JSON.parse`。

解析实现在 [src/api.ts](src/api.ts)。
