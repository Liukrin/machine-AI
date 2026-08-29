import { useEffect, useRef, useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
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
  createdAt: number;
};

type Session = {
  id: string;
  title: string;
  turns: Turn[];
  createdAt: number;
};

let idSeq = 0;
let sessionSeq = 0;

function fmtMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

function fmtTime(ts: number): string {
  return new Date(ts).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
}

/* ─── 检索来源卡片 ─── */
function SourceCard({ c }: { c: ChunkRef }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div
      className="glass-card cursor-pointer px-3 py-2 transition-all"
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center gap-2">
        <span className="font-mono text-[11px] font-semibold text-indigo-400">{c.chunk_id}</span>
        <span
          className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
            c.chunk_type === 'table'
              ? 'bg-blue-500/20 text-blue-400'
              : 'bg-indigo-500/20 text-indigo-400'
          }`}
        >
          {c.chunk_type}
        </span>
        <span className="font-mono text-[10px] text-[var(--text-muted)]">doc {c.doc_id}</span>
        <span className="ml-auto font-mono text-[10px] text-[var(--text-muted)]">
          d={c.distance.toFixed(4)}
        </span>
        <span className="text-[10px] text-[var(--text-muted)]">{expanded ? '▲' : '▼'}</span>
      </div>
      {c.heading_path && (
        <div className="mt-1 text-xs font-medium text-[var(--text-secondary)]">{c.heading_path}</div>
      )}
      {expanded && (
        <div className="mt-2 text-xs leading-relaxed text-[var(--text-secondary)] border-t border-[var(--border-subtle)] pt-2">
          {c.preview}
        </div>
      )}
    </div>
  );
}

/* ─── 引用校验 ─── */
function VerificationBlock({ v }: { v: Verification }) {
  const ok = v.suspicious_count === 0 && v.fabricated_ids.length === 0;
  return (
    <div
      className={`rounded-lg border px-3 py-2.5 ${
        ok
          ? 'border-emerald-500/30 bg-emerald-500/5'
          : 'border-amber-500/30 bg-amber-500/5'
      }`}
    >
      <div className={`flex items-center gap-2 text-xs font-semibold ${ok ? 'text-emerald-400' : 'text-amber-400'}`}>
        <span>{ok ? '✓' : '⚠'}</span>
        <span>引用校验：可疑 {v.suspicious_count} 句</span>
        {v.cited_ids.length > 0 && (
          <span className="ml-auto font-mono text-[10px] font-normal text-[var(--text-muted)]">
            引用 {v.cited_ids.join('、')}
          </span>
        )}
      </div>
      {v.fabricated_ids.length > 0 && (
        <div className="mt-1.5 rounded bg-red-500/10 px-2 py-1 text-xs font-medium text-red-400">
          引用未检索到的 chunk_id：{v.fabricated_ids.join('、')}
        </div>
      )}
      {v.suspicious.length > 0 && (
        <ul className="mt-1.5 list-disc pl-5 text-xs text-amber-300/80">
          {v.suspicious.map((s, i) => (
            <li key={i}>{s}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* ─── 拒答卡片 ─── */
function RejectedCard({ r }: { r: Rejected }) {
  return (
    <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2.5">
      <div className="flex items-center gap-2 text-sm font-semibold text-amber-400">
        <span>⛔</span>
        <span>{r.reason}</span>
      </div>
      <div className="mt-1 font-mono text-xs text-[var(--text-muted)]">
        top1_distance = {r.top1_distance.toFixed(4)} &gt; τ = {r.tau}
      </div>
    </div>
  );
}

/* ─── 单轮对话视图 ─── */
function TurnView({ turn }: { turn: Turn }) {
  const isStreaming = turn.status === 'loading' && turn.answer.length > 0;

  return (
    <div className="flex flex-col gap-4">
      {/* 用户问题 */}
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-tr-sm bg-indigo-500/15 border border-indigo-500/20 px-4 py-2.5">
          <p className="text-sm text-[var(--text-primary)]">{turn.question}</p>
        </div>
      </div>

      {/* AI 回答区 */}
      <div className="flex gap-3">
        {/* AI 头像 */}
        <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center text-white text-xs font-bold shadow-lg shadow-indigo-500/20">
          AI
        </div>
        <div className="flex-1 min-w-0 flex flex-col gap-3">
          {/* 检索来源 */}
          {turn.chunks.length > 0 && (
            <div>
              <div className="mb-1.5 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                📚 检索来源（{turn.chunks.length}）
              </div>
              <div className="flex flex-col gap-1.5">
                {turn.chunks.map((c) => (
                  <SourceCard key={c.chunk_id} c={c} />
                ))}
              </div>
            </div>
          )}

          {/* 加载中 */}
          {turn.status === 'loading' && !turn.answer && (
            <div className="flex items-center gap-2 text-sm text-[var(--text-muted)]">
              <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-indigo-500/30 border-t-indigo-500" />
              <span>正在检索知识库…</span>
            </div>
          )}

          {/* 答案（Markdown 渲染） */}
          {turn.answer && (
            <div className="glass-card px-4 py-3">
              <div className={`markdown-body text-sm leading-relaxed ${isStreaming ? 'typing-cursor' : ''}`}>
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {turn.answer}
                </ReactMarkdown>
              </div>
            </div>
          )}

          {/* 校验结果 */}
          {turn.verification && <VerificationBlock v={turn.verification} />}

          {/* 拒答 */}
          {turn.status === 'rejected' && turn.rejected && <RejectedCard r={turn.rejected} />}

          {/* 错误 */}
          {turn.status === 'error' && (
            <div className="rounded-lg border border-red-500/30 bg-red-500/5 px-3 py-2.5 text-sm text-red-400">
              {turn.error}
            </div>
          )}

          {/* 完成统计 */}
          {turn.done && (
            <div className="text-[11px] text-[var(--text-muted)] flex items-center gap-2">
              <span>✓ 完成</span>
              <span>·</span>
              <span className="font-mono">{turn.done.total_tokens ?? '—'} tokens</span>
              <span>·</span>
              <span className="font-mono">{fmtMs(turn.done.elapsed_ms)}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ─── 侧边栏 ─── */
function Sidebar({
  sessions,
  activeId,
  onSelect,
  onNew,
  onClose,
}: {
  sessions: Session[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onClose: () => void;
}) {
  return (
    <div className="sidebar-enter fixed inset-0 z-50 flex">
      {/* 遮罩 */}
      <div className="flex-1 bg-black/50 backdrop-blur-sm" onClick={onClose} />
      {/* 侧边栏 */}
      <div className="w-72 h-full flex flex-col" style={{ background: 'var(--bg-secondary)' }}>
        <div className="flex items-center gap-2 p-4 border-b border-[var(--border-color)]">
          <h2 className="text-sm font-semibold text-[var(--text-primary)] flex-1">对话历史</h2>
          <button
            onClick={onNew}
            className="rounded-lg bg-indigo-500/15 px-3 py-1.5 text-xs font-medium text-indigo-400 hover:bg-indigo-500/25 transition-colors"
          >
            + 新对话
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2">
          {sessions.length === 0 && (
            <div className="px-3 py-8 text-center text-xs text-[var(--text-muted)]">
              暂无对话记录
            </div>
          )}
          {sessions.map((s) => (
            <button
              key={s.id}
              onClick={() => onSelect(s.id)}
              className={`w-full text-left rounded-lg px-3 py-2.5 mb-1 transition-colors ${
                s.id === activeId
                  ? 'bg-indigo-500/15 border border-indigo-500/30'
                  : 'hover:bg-[var(--bg-card)] border border-transparent'
              }`}
            >
              <div className="text-sm font-medium text-[var(--text-primary)] truncate">
                {s.title}
              </div>
              <div className="mt-0.5 text-[10px] text-[var(--text-muted)]">
                {s.turns.length} 条消息 · {fmtTime(s.createdAt)}
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ─── 主组件 ─── */
export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // 初始化一个新会话
  const createSession = useCallback(() => {
    const id = String(++sessionSeq);
    const session: Session = {
      id,
      title: '新对话',
      turns: [],
      createdAt: Date.now(),
    };
    setSessions((prev) => [session, ...prev]);
    setActiveSessionId(id);
    return id;
  }, []);

  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch((e) => setHealthError(String(e)));
    createSession();
  }, []);

  const activeSession = sessions.find((s) => s.id === activeSessionId);
  const turns = activeSession?.turns ?? [];

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
  }, [turns]);

  const updateSession = (sessionId: string, updater: (s: Session) => Session) => {
    setSessions((prev) => prev.map((s) => (s.id === sessionId ? updater(s) : s)));
  };

  const submit = () => {
    const q = input.trim();
    if (!q || busy || !activeSessionId) return;
    setInput('');
    setBusy(true);

    const turnId = String(++idSeq);
    const turn: Turn = {
      id: turnId,
      question: q,
      status: 'loading',
      chunks: [],
      answer: '',
      verification: null,
      rejected: null,
      done: null,
      error: null,
      createdAt: Date.now(),
    };

    const sid = activeSessionId;

    // 更新会话标题（取第一个问题）
    updateSession(sid, (s) => ({
      ...s,
      title: s.turns.length === 0 ? q.slice(0, 30) + (q.length > 30 ? '…' : '') : s.title,
      turns: [...s.turns, turn],
    }));

    const patch = (p: Partial<Turn>) =>
      updateSession(sid, (s) => ({
        ...s,
        turns: s.turns.map((t) => (t.id === turnId ? { ...t, ...p } : t)),
      }));

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

  return (
    <div className="flex h-full flex-col" style={{ background: 'var(--bg-primary)' }}>
      {/* 顶部导航栏 */}
      <header className="flex items-center gap-3 border-b border-[var(--border-color)] px-5 py-3" style={{ background: 'var(--bg-secondary)' }}>
        <button
          onClick={() => setSidebarOpen(true)}
          className="rounded-lg p-2 text-[var(--text-muted)] hover:bg-[var(--bg-card)] hover:text-[var(--text-primary)] transition-colors"
          title="对话历史"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="3" y1="6" x2="21" y2="6" />
            <line x1="3" y1="12" x2="21" y2="12" />
            <line x1="3" y1="18" x2="21" y2="18" />
          </svg>
        </button>

        <div className="flex items-center gap-2">
          <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
          </div>
          <h1 className="text-sm font-bold text-[var(--text-primary)]">设备运维知识问答</h1>
        </div>

        <div className="ml-auto flex items-center gap-2 text-xs">
          {health ? (
            <>
              <span
                className={`inline-block h-2 w-2 rounded-full ${
                  health.status === 'ok' ? 'bg-emerald-400 shadow-sm shadow-emerald-400/50' : 'bg-red-400'
                }`}
              />
              <span className="text-[var(--text-muted)]">
                {health.status === 'ok'
                  ? `${health.chunk_count} chunks · ${health.model_name}`
                  : '后端异常'}
              </span>
            </>
          ) : healthError ? (
            <span className="text-red-400">连接失败</span>
          ) : (
            <span className="text-[var(--text-muted)]">连接中…</span>
          )}
        </div>
      </header>

      {/* 主内容区 */}
      <main ref={scrollRef} className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-5 py-6">
          {/* 空状态 */}
          {turns.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20">
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-indigo-500/20 to-purple-600/20 border border-indigo-500/20 flex items-center justify-center mb-4">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-indigo-400">
                  <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" strokeLinecap="round" strokeLinejoin="round" />
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="17" x2="12.01" y2="17" strokeLinecap="round" />
                </svg>
              </div>
              <h2 className="text-lg font-semibold text-[var(--text-primary)] mb-2">设备运维知识问答</h2>
              <p className="text-sm text-[var(--text-muted)] mb-6">输入设备相关问题，AI 将基于知识库为您解答</p>
              <div className="flex flex-wrap gap-2 justify-center">
                {['泵启动前要做哪些检查', '阀门维护保养规程', '冷却系统常见故障'].map((q) => (
                  <button
                    key={q}
                    onClick={() => { setInput(q); }}
                    className="rounded-full border border-[var(--border-color)] bg-[var(--bg-card)] px-4 py-2 text-xs text-[var(--text-secondary)] hover:bg-[var(--bg-card-hover)] hover:text-[var(--text-primary)] transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* 对话列表 */}
          <div className="flex flex-col gap-6">
            {turns.map((t) => (
              <TurnView key={t.id} turn={t} />
            ))}
          </div>
        </div>
      </main>

      {/* 底部输入区 */}
      <footer className="border-t border-[var(--border-color)] p-4" style={{ background: 'var(--bg-secondary)' }}>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
          className="mx-auto max-w-3xl flex gap-3"
        >
          <div className="flex-1 relative">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={busy}
              placeholder="输入设备运维问题，回车提交…"
              className="w-full rounded-xl border border-[var(--border-color)] bg-[var(--bg-card)] px-4 py-3 text-sm text-[var(--text-primary)] placeholder-[var(--text-muted)] outline-none focus:border-indigo-500/50 focus:ring-2 focus:ring-indigo-500/10 disabled:opacity-50 transition-all"
            />
          </div>
          <button
            type="submit"
            disabled={busy || !input.trim()}
            className="glow-btn rounded-xl px-5 py-3 text-sm font-medium text-white"
          >
            {busy ? (
              <span className="flex items-center gap-2">
                <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
              </span>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            )}
          </button>
        </form>
      </footer>

      {/* 侧边栏 */}
      {sidebarOpen && (
        <Sidebar
          sessions={sessions}
          activeId={activeSessionId}
          onSelect={(id) => {
            setActiveSessionId(id);
            setSidebarOpen(false);
          }}
          onNew={() => {
            createSession();
            setSidebarOpen(false);
          }}
          onClose={() => setSidebarOpen(false)}
        />
      )}
    </div>
  );
}
