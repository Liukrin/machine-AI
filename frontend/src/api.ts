// 后端接口类型 + POST-SSE 流式读取（EventSource 不支持 POST，需 fetch 手动解析）

export type DocumentInfo = {
  doc_id: string;
  doc_title: string;
  doc_short: string;
  is_sample: boolean;
  chunk_count: number;
};

export type Mode = 'agent' | 'rag';

export type Health = {
  status: string;
  chunk_count: number;
  model_name: string;
  /** /api/ask 不指定 mode 时的默认模式（旧后端没有该字段） */
  default_mode?: Mode;
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

/** retrieval 事件里的来源片段（不含全文，全文走 /api/chunks/{id} 按需拉取） */
export type ChunkRef = ChunkMeta & {
  /** 与检索问句的向量 cosine distance，越小越相关；Agent 查表、读相邻片段得到的片段没有，为 null */
  distance: number | null;
  preview: string;
};

export type ChunkDetail = ChunkMeta & {
  content: string | null;
  table_html: string | null;
};

/** 回答里核对不上的一个数值 */
export type NumberIssue = {
  quantity: string;
  /** no_source：本次的资料和问题里找不到这个数；unit_mismatch：资料里这个数的单位不是回答里写的单位 */
  status: 'no_source' | 'unit_mismatch';
  source_units: string[];
  sentence: string;
  detail: string;
};

export type Verification = {
  suspicious_count: number;
  suspicious: string[];
  cited_ids: string[];
  fabricated_ids: string[];
  /** 后端判定的拒答（含拒答话术且没有引用任何片段）；rag 模式与旧会话记录没有该字段 */
  refused?: boolean;
  /** agent 模式的数值核对：核对了几个数、最终答案里仍对不上的数、是否改写过（旧记录没有这些字段） */
  numbers_checked?: number;
  number_issues?: NumberIssue[];
  repair?: { issues_before: number; issues_after: number; kept: 'repaired' | 'draft'; draft_issues?: string[] } | null;
};

export type Rejected = {
  reason: string;
  top1_distance: number;
  tau: number;
};

export type Done = {
  mode?: Mode;
  total_tokens: number | null;
  input_tokens?: number | null;
  output_tokens?: number | null;
  /** 输入 token 中命中服务端缓存的部分（agent 模式） */
  cache_read_tokens?: number | null;
  elapsed_ms: number;
  /** 模型结束原因："stop" 正常结束，"length" 被生成长度上限截断（旧会话记录里没有该字段） */
  finish_reason?: string | null;
  /** agent 模式：模型调用次数、工具调用次数（含系统预检索）、是否因达到轮数上限被强制作答 */
  llm_calls?: number;
  tool_calls?: number;
  forced_final?: boolean;
  model_name?: string | null;
  /** 按 config llm_pricing 估算的本次费用（元） */
  cost_yuan?: number;
  /** agent 模式：最终采用的答案（数值核对后可能经过改写；拒答会去掉引用），以它为准 */
  answer?: string;
};

/** Agent 的步骤事件（step）：模型开始一次调用、工具开始/结束、中间轮次的说明文字、达到轮数上限 */
export type StepEvent =
  | { type: 'llm_start'; round: number; forced: boolean }
  | { type: 'tool_start'; id: string; round: number; name: string; label: string; args: Record<string, unknown>; auto?: boolean }
  | {
      type: 'tool_end';
      id: string;
      round: number;
      name: string;
      label: string;
      args: Record<string, unknown>;
      ok: boolean;
      summary: string;
      elapsed_ms: number;
      auto?: boolean;
      output?: string | null;
    }
  | { type: 'thought'; round: number; text: string }
  | { type: 'forced_final'; round: number }
  /** 回答里的数值核对不上，接下来让模型改写一次（issues 是逐条说明） */
  | { type: 'repair'; round: number; issues: string[] };

/**
 * rag 模式：retrieval → (rejected | token* + verification + done) / error
 * agent 模式：step / retrieval（累计的来源列表）穿插 → token*（带 round）→ verification → done / error
 */
export type StreamEvent =
  | { event: 'retrieval'; data: { chunks: ChunkRef[] } }
  | { event: 'rejected'; data: Rejected }
  | { event: 'step'; data: StepEvent }
  | { event: 'token'; data: { text: string; round?: number } }
  | { event: 'verification'; data: Verification }
  | { event: 'done'; data: Done }
  | { event: 'error'; data: { message: string } };

export type HistoryMessage = { role: 'user' | 'assistant'; content: string };

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
  req: { question: string; history: HistoryMessage[]; mode: Mode },
  onEvent: (e: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
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
