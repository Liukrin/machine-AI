import { useEffect, useRef, useState } from 'react';
import {
  askStream,
  fetchHealth,
  type ChunkRef,
  type Done,
  type Health,
  type Rejected,
  type Verification,
} from './api';

type Turn = {
  id: string;
  question: string;
  status: 'loading' | 'done' | 'rejected' | 'error';
  chunks: ChunkRef[];
  answer: string;
  verification: Verification | null;
  rejected: Rejected | null;
  done: Done | null;
  error: string | null;
};

let idSeq = 0;

function fmtMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

function SourceCard({ c }: { c: ChunkRef }) {
  return (
    <div className="flex flex-col gap-1 rounded border border-neutral-200 bg-neutral-50 px-3 py-2">
      <div className="flex items-center gap-2">
        <span className="font-mono text-xs font-semibold text-neutral-800">{c.chunk_id}</span>
        <span
          className={`rounded px-1.5 py-0.5 text-[11px] font-medium ${
            c.chunk_type === 'table' ? 'bg-blue-100 text-blue-700' : 'bg-neutral-200 text-neutral-600'
          }`}
        >
          {c.chunk_type}
        </span>
        <span className="font-mono text-[11px] text-neutral-400">doc {c.doc_id}</span>
        <span className="ml-auto font-mono text-[11px] text-neutral-500">
          d={c.distance.toFixed(4)}
        </span>
      </div>
      {c.heading_path && (
        <div className="text-xs font-medium text-neutral-700">{c.heading_path}</div>
      )}
      <div className="text-xs leading-relaxed text-neutral-600">{c.preview}</div>
    </div>
  );
}

function VerificationBlock({ v }: { v: Verification }) {
  const ok = v.suspicious_count === 0 && v.fabricated_ids.length === 0;
  return (
    <div
      className={`rounded border px-3 py-2 ${
        ok ? 'border-emerald-200 bg-emerald-50' : 'border-amber-200 bg-amber-50'
      }`}
    >
      <div className={`flex items-center gap-2 text-sm font-semibold ${ok ? 'text-emerald-700' : 'text-amber-700'}`}>
        <span>{ok ? '✓' : '⚠'}</span>
        <span>引用校验：可疑 {v.suspicious_count} 句</span>
        {v.cited_ids.length > 0 && (
          <span className="ml-auto font-mono text-[11px] font-normal text-neutral-500">
            引用 {v.cited_ids.join('、')}
          </span>
        )}
      </div>
      {v.fabricated_ids.length > 0 && (
        <div className="mt-1 rounded bg-red-100 px-2 py-1 text-xs font-medium text-red-700">
          引用未检索到的 chunk_id：{v.fabricated_ids.join('、')}
        </div>
      )}
      {v.suspicious.length > 0 && (
        <ul className="mt-1 list-disc pl-5 text-xs text-amber-800">
          {v.suspicious.map((s, i) => (
            <li key={i}>{s}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function RejectedCard({ r }: { r: Rejected }) {
  return (
    <div className="rounded border border-amber-200 bg-amber-50 px-3 py-2">
      <div className="flex items-center gap-2 text-sm font-semibold text-amber-700">
        <span>⛔</span>
        <span>{r.reason}</span>
      </div>
      <div className="mt-1 font-mono text-xs text-neutral-600">
        top1_distance = {r.top1_distance.toFixed(4)} &gt; τ = {r.tau}
      </div>
    </div>
  );
}

function TurnView({ turn }: { turn: Turn }) {
  return (
    <div className="rounded border border-neutral-200 bg-white p-4 shadow-sm">
      <div className="mb-3 flex items-baseline gap-2 border-b border-neutral-100 pb-2">
        <span className="text-[11px] font-semibold uppercase tracking-wide text-neutral-400">Q</span>
        <h3 className="text-sm font-semibold text-neutral-900">{turn.question}</h3>
      </div>

      {turn.status === 'loading' && (
        <div className="flex items-center gap-2 text-sm text-neutral-500">
          <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-neutral-300 border-t-neutral-600" />
          检索中…
        </div>
      )}

      {turn.chunks.length > 0 && (
        <div className="mb-3">
          <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-neutral-400">
            检索来源（{turn.chunks.length}）
          </div>
          <div className="flex flex-col gap-1.5">
            {turn.chunks.map((c) => (
              <SourceCard key={c.chunk_id} c={c} />
            ))}
          </div>
        </div>
      )}

      {turn.answer && (
        <div className="mb-3">
          <div className="mb-1.5 text-[11px] font-semibold uppercase tracking-wide text-neutral-400">
            答案
          </div>
          <div className="whitespace-pre-wrap text-sm leading-relaxed text-neutral-800">
            {turn.answer}
          </div>
        </div>
      )}

      {turn.verification && (
        <div className="mb-3">
          <VerificationBlock v={turn.verification} />
        </div>
      )}

      {turn.status === 'rejected' && turn.rejected && <RejectedCard r={turn.rejected} />}

      {turn.status === 'error' && (
        <div className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
          {turn.error}
        </div>
      )}

      {turn.done && (
        <div className="mt-3 border-t border-neutral-100 pt-2 text-xs text-neutral-500">
          完成 · {turn.done.total_tokens ?? '—'} tokens · {fmtMs(turn.done.elapsed_ms)}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch((e) => setHealthError(String(e)));
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [turns]);

  const submit = () => {
    const q = input.trim();
    if (!q || busy) return;
    setInput('');
    setBusy(true);

    const id = String(++idSeq);
    const turn: Turn = {
      id,
      question: q,
      status: 'loading',
      chunks: [],
      answer: '',
      verification: null,
      rejected: null,
      done: null,
      error: null,
    };
    setTurns((prev) => [...prev, turn]);

    const patch = (p: Partial<Turn>) =>
      setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...p } : t)));

    let answer = '';
    askStream(q, (event, data) => {
      switch (event) {
        case 'retrieval':
          patch({ chunks: data.chunks ?? [] });
          break;
        case 'rejected':
          patch({ status: 'rejected', rejected: data as Rejected });
          break;
        case 'token':
          answer += data.text ?? '';
          patch({ answer });
          break;
        case 'verification':
          patch({ verification: data as Verification });
          break;
        case 'done':
          patch({ status: 'done', done: data as Done });
          break;
        case 'error':
          patch({ status: 'error', error: data.message ?? '未知错误' });
          break;
      }
    })
      .catch((e) => patch({ status: 'error', error: String(e) }))
      .finally(() => setBusy(false));
  };

  const lastDone = [...turns].reverse().find((t) => t.status === 'done' && t.done);

  return (
    <div className="flex h-full flex-col bg-neutral-100 text-neutral-900">
      {/* 顶部：标题 + 健康状态 */}
      <header className="flex items-center gap-3 border-b border-neutral-200 bg-white px-5 py-3">
        <h1 className="text-base font-bold">设备运维知识问答</h1>
        <div className="ml-auto flex items-center gap-2 text-xs text-neutral-500">
          {health ? (
            <>
              <span
                className={`inline-block h-2 w-2 rounded-full ${
                  health.status === 'ok' ? 'bg-emerald-500' : 'bg-red-500'
                }`}
              />
              <span>
                {health.status === 'ok'
                  ? `${health.chunk_count} 条 chunk · ${health.model_name}`
                  : '后端异常'}
              </span>
            </>
          ) : healthError ? (
            <span className="text-red-600">健康检查失败：{healthError}</span>
          ) : (
            <span>连接后端…</span>
          )}
        </div>
      </header>

      {/* 中部：问答区 */}
      <main ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-5 py-4">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              submit();
            }}
            className="mb-4 flex gap-2"
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={busy}
              placeholder="输入设备运维问题，回车提交"
              className="flex-1 rounded border border-neutral-300 bg-white px-3 py-2 text-sm outline-none focus:border-neutral-500 disabled:bg-neutral-100 disabled:text-neutral-400"
            />
            <button
              type="submit"
              disabled={busy || !input.trim()}
              className="rounded bg-neutral-900 px-4 py-2 text-sm font-medium text-white hover:bg-neutral-700 disabled:cursor-not-allowed disabled:bg-neutral-300"
            >
              {busy ? '查询中…' : '提交'}
            </button>
          </form>

          <div className="flex flex-col gap-3">
            {turns.length === 0 && (
              <div className="rounded border border-dashed border-neutral-300 bg-white px-4 py-10 text-center text-sm text-neutral-400">
                尚未提问。示例：泵启动前要做哪些检查
              </div>
            )}
            {turns.map((t) => (
              <TurnView key={t.id} turn={t} />
            ))}
          </div>
        </div>
      </main>

      {/* 底部：done 统计 */}
      <footer className="border-t border-neutral-200 bg-white px-5 py-2 text-xs text-neutral-500">
        {lastDone ? (
          <span>
            最近完成：<span className="font-mono">{lastDone.done!.total_tokens ?? '—'}</span> tokens ·{' '}
            <span className="font-mono">{fmtMs(lastDone.done!.elapsed_ms)}</span>
          </span>
        ) : (
          <span>等待查询</span>
        )}
      </footer>
    </div>
  );
}
