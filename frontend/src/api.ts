// 后端接口类型 + POST-SSE 流式读取（EventSource 不支持 POST，需 fetch 手动解析）

export type Health = {
  status: string;
  chunk_count: number;
  model_name: string;
};

export type ChunkRef = {
  chunk_id: string;
  heading_path: string | null;
  doc_id: string;
  chunk_type: 'text' | 'table';
  distance: number;
  preview: string;
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
  elapsed_ms: number;
};

// 事件回调：event 名 + 已 JSON.parse 的 data
export type SSECallback = (event: string, data: any) => void;

export async function fetchHealth(): Promise<Health> {
  const res = await fetch('/api/health');
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/** 解析一段 SSE 事件块（event: + data: 多行），交给 onEvent。 */
function dispatchRawEvent(raw: string, onEvent: SSECallback): void {
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
  let data: any;
  try {
    data = JSON.parse(payload);
  } catch {
    data = payload;
  }
  onEvent(event, data);
}

/**
 * POST /api/ask 的流式读取。手动读 body，用缓冲区处理跨 chunk 截断的半行数据，
 * 按空行（\n\n）切分事件块。
 */
export async function askStream(question: string, onEvent: SSECallback): Promise<void> {
  const res = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
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
