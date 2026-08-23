# 前端（Vite + React + TypeScript + Tailwind CSS）

设备运维知识问答的单页前端，对接后端 `http://localhost:8000` 的 SSE 流式接口。

## 依赖安装

```bash
npm install
```

## 启动

```bash
npm run dev     # http://localhost:5173，/api 自动代理到 http://localhost:8000
npm run build   # 产物输出到 dist/
npm run preview # 预览构建产物
```

后端先起：`python -m uvicorn src.s5_app.api:app --host 127.0.0.1 --port 8000`（在项目根目录执行）。

## SSE 解析要点

后端 `POST /api/ask` 返回 SSE 流，但浏览器 `EventSource` 不支持 POST，须用 `fetch` + `response.body.getReader()` 手动读流。三个关键点：

1. **按事件边界切分**。事件以空行分隔；后端 sse-starlette 用 `\r\n` 行尾。切分须兼容 `\r\n` / `\n` / `\r` 三种行尾（正则 `/\r?\n\r?\n/`），**不要硬编码 `\n\n`**——否则 CRLF 流里永远匹配不到，事件会积压到流结束才一次性派发（曾导致只渲染出残缺 done 行、拒答路径崩溃）。

2. **缓冲区拼接半行**。`TextDecoder.decode(value, { stream: true })` 增量解码，跨 chunk 截断的多字节字符与半行留在缓冲区，读到完整空行才派发一个事件块；流结束再用 `decode()`（不带 stream）flush 残余字节。

3. **多行 `data:` 合并**。一个事件块可含多行 `data:`，需 `join('\n')` 后再 `JSON.parse`。

事件顺序：`retrieval → (rejected | token* + verification + done) / error`。解析实现在 [src/api.ts](src/api.ts)。
