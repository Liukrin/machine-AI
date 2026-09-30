// 后端接口类型 + POST-SSE 流式读取（EventSource 不支持 POST，需 fetch 手动解析）

export type DocumentInfo = {
  doc_id: string;
  doc_title: string;
  doc_short: string;
  is_sample: boolean;
  chunk_count: number;
};

export type Health = {
  status: string;
  chunk_count: number;
  model_name: string;
  documents: DocumentInfo[];
};

export type ChunkMeta = {
  chunk_id: string;
  doc_id: string;
  doc_title: string;
  doc_short: string;
  is_sample: boolean;
  chunk_type: 'text' | 'table';
  heading_path: string | null;
  /** 1 起页码 [起, 止] */
  pages: [number, number] | null;
};

/** retrieval 事件里的检索结果（不含全文，全文走 /api/chunks/{id} 按需拉取） */
export type ChunkRef = ChunkMeta & {
  /** 向量 cosine distance，越小越相关，与拒答阈值同口径 */
  distance: number;
  preview: string;
};

export type ChunkDetail = ChunkMeta & {
  content: string | null;
  table_html: string | null;
};

export type Verification = {
  suspicious_count: number;
  suspicious: string[];
  cited_ids: string[];
  fabricated_ids: string[];
};

export type Rejected = {
  reason: string;
  top1_distance: number;
  tau: number;
};

export type Done = {
  total_tokens: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  elapsed_ms: number;
  /** 模型结束原因："stop" 正常结束，"length" 被生成长度上限截断（旧会话记录里没有该字段） */
  finish_reason?: string | null;
};

/** 事件顺序：retrieval → (rejected | token* + verification + done) / error */
export type StreamEvent =
  | { event: 'retrieval'; data: { chunks: ChunkRef[] } }
  | { event: 'rejected'; data: Rejected }
  | { event: 'token'; data: { text: string } }
  | { event: 'verification'; data: Verification }
  | { event: 'done'; data: Done }
  | { event: 'error'; data: { message: string } };

export async function fetchHealth(): Promise<Health> {
  const res = await fetch('/api/health');
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

const chunkCache = new Map<string, Promise<ChunkDetail>>();

/** 拉取单个 chunk 全文；同一 id 只请求一次，失败时移出缓存以便重试。 */
export function fetchChunk(chunkId: string): Promise<ChunkDetail> {
  let p = chunkCache.get(chunkId);
  if (!p) {
    p = fetch(`/api/chunks/${encodeURIComponent(chunkId)}`).then((res) => {
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json() as Promise<ChunkDetail>;
    });
    p.catch(() => chunkCache.delete(chunkId));
    chunkCache.set(chunkId, p);
  }
  return p;
}

/** 解析一段 SSE 事件块（event: + data: 多行），交给 onEvent。 */
function dispatchRawEvent(raw: string, onEvent: (e: StreamEvent) => void): void {
  let event = 'message';
  const dataLines: string[] = [];
  for (const line of raw.split('\n')) {
    const ln = line.replace(/\r$/, '');
    if (ln.startsWith('event:')) {
      event = ln.slice('event:'.length).trim();
    } else if (ln.startsWith('data:')) {
      dataLines.push(ln.slice('data:'.length).trimStart());
    }
  }
  if (dataLines.length === 0) return;
  const payload = dataLines.join('\n');
  let data: unknown;
  try {
    data = JSON.parse(payload);
  } catch {
    data = payload;
  }
  onEvent({ event, data } as StreamEvent);
}

/**
 * POST /api/ask 的流式读取。手动读 body，用缓冲区处理跨 chunk 截断的半行数据，
 * 按空行切分事件块。signal 中止时 fetch / read 会抛 AbortError。
 */
export async function askStream(
  question: string,
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  // SSE 事件以空行分隔；后端用 \r\n（CRLF）行尾，兼容 \n / \r 三种行尾
  const EVENT_SEP = /\r?\n\r?\n/;

  const drain = () => {
    let m: RegExpExecArray | null;
    while ((m = EVENT_SEP.exec(buffer)) !== null) {
      const raw = buffer.slice(0, m.index);
      buffer = buffer.slice(m.index + m[0].length);
      if (raw.trim()) dispatchRawEvent(raw, onEvent);
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    drain();
  }

  // 流结束后 flush：多字节字符在最后一个 chunk 可能被截断，需不带 stream 再 decode 一次
  buffer += decoder.decode();
  drain();
  if (buffer.trim()) dispatchRawEvent(buffer, onEvent);
}
