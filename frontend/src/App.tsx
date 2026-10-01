import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  askStream,
  fetchHealth,
  sendFeedback,
  type ChunkRef,
  type Health,
  type HistoryMessage,
  type Mode,
  type Rating,
} from './api';
import { Composer } from './components/Composer';
import { Sidebar } from './components/Sidebar';
import { SourceDrawer } from './components/SourceDrawer';
import { TopBar } from './components/TopBar';
import { TurnView } from './components/TurnView';
import { Welcome } from './components/Welcome';
import { newId } from './lib/format';
import { loadMode, loadSessions, saveMode, saveSessions } from './lib/storage';
import type { AgentStep, OpenSource, Session, Turn } from './types';

const APP_NAME = '设备运维知识助手';
const TITLE_MAX = 28;
// 发给后端的对话历史最多几轮（后端按 config agent.history_turns 再截一次）
const HISTORY_TURNS = 6;

/** 本会话之前已完成的问答 → 对话历史（出错、停止、被拒答闸拦下的轮次不带） */
function buildHistory(turns: Turn[]): HistoryMessage[] {
  return turns
    .filter((t) => t.status === 'done' && t.answer.trim())
    .slice(-HISTORY_TURNS)
    .flatMap((t) => [
      { role: 'user' as const, content: t.question },
      { role: 'assistant' as const, content: t.answer },
    ]);
}

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [sessions, setSessions] = useState<Session[]>(loadSessions);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [openSource, setOpenSource] = useState<OpenSource | null>(null);
  const [mode, setMode] = useState<Mode | null>(loadMode);

  const busyRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const streamingSidRef = useRef<string | null>(null);
  const scrollRef = useRef<HTMLElement>(null);
  const stickToBottom = useRef(true);
  // ask 回调里读最新的会话（拼对话历史），不把 sessions 放进依赖，免得每个 token 都重建回调
  const sessionsRef = useRef(sessions);
  useEffect(() => {
    sessionsRef.current = sessions;
  }, [sessions]);

  // 这里只做幂等的读取：StrictMode 下 effect 会执行两次，会话在首次提问时才创建。
  // 连不上就隔几秒重试（1、2、4…最长 10 秒）：前后端同时启动时，后端还在预加载模型，第一次检查必然失败
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let delay = 1000;
    const check = () => {
      fetchHealth()
        .then((h) => {
          if (stopped) return;
          setHealth(h);
          setHealthError(null);
        })
        .catch((e) => {
          if (stopped) return;
          setHealthError(String(e));
          timer = setTimeout(check, delay);
          delay = Math.min(delay * 2, 10_000);
        });
    };
    check();
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, []);

  useEffect(() => {
    const t = setTimeout(() => saveSessions(sessions), 400);
    return () => clearTimeout(t);
  }, [sessions]);

  // 用户没选过模式时，跟随后端的默认模式
  const effectiveMode: Mode = mode ?? health?.default_mode ?? 'agent';
  const changeMode = (m: Mode) => {
    setMode(m);
    saveMode(m);
  };

  const active = sessions.find((s) => s.id === activeId) ?? null;
  const turns = active?.turns ?? [];
  const sortedSessions = useMemo(() => [...sessions].sort((a, b) => b.updatedAt - a.updatedAt), [sessions]);

  useEffect(() => {
    document.title = active ? `${active.title} · ${APP_NAME}` : APP_NAME;
  }, [active?.title]);

  // 用户停留在底部时跟随流式输出滚动；向上翻看历史时不打扰
  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [turns]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (el) stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 96;
  };

  const patchTurn = useCallback((sid: string, tid: string, patch: Partial<Turn>) => {
    setSessions((prev) =>
      prev.map((s) =>
        s.id !== sid
          ? s
          : { ...s, updatedAt: Date.now(), turns: s.turns.map((t) => (t.id === tid ? { ...t, ...patch } : t)) },
      ),
    );
  }, []);

  const ask = useCallback(
    (raw: string) => {
      const question = raw.trim();
      if (!question || busyRef.current) return;

      const now = Date.now();
      const turn: Turn = {
        id: newId(),
        question,
        status: 'retrieving',
        mode: effectiveMode,
        chunks: [],
        answer: '',
        steps: [],
        verification: null,
        rejected: null,
        done: null,
        error: null,
        createdAt: now,
      };
      const sid = activeId ?? newId();
      const history = buildHistory(sessionsRef.current.find((s) => s.id === sid)?.turns ?? []);
      const title = question.length > TITLE_MAX ? `${question.slice(0, TITLE_MAX)}…` : question;
      setSessions((prev) =>
        prev.some((s) => s.id === sid)
          ? prev.map((s) => (s.id === sid ? { ...s, updatedAt: now, turns: [...s.turns, turn] } : s))
          : [{ id: sid, title, turns: [turn], createdAt: now, updatedAt: now }, ...prev],
      );
      setActiveId(sid);
      setInput('');
      setBusy(true);
      busyRef.current = true;
      streamingSidRef.current = sid;
      stickToBottom.current = true;

      const controller = new AbortController();
      abortRef.current = controller;
      const patch = (p: Partial<Turn>) => patchTurn(sid, turn.id, p);
      let answer = '';
      let answerRound: number | undefined;
      let steps: AgentStep[] = [];
      let finished = false;

      askStream(
        { question, history, mode: effectiveMode },
        (ev) => {
          switch (ev.event) {
            case 'retrieval':
              patch({ status: 'generating', chunks: ev.data.chunks ?? [] });
              break;
            case 'step': {
              const d = ev.data;
              if (d.type === 'llm_start') {
                patch({ status: 'generating', thinking: true });
              } else if (d.type === 'tool_start') {
                steps = [
                  ...steps,
                  { kind: 'tool', id: d.id, round: d.round, name: d.name, label: d.label, args: d.args, auto: !!d.auto, status: 'running' },
                ];
                patch({ steps, thinking: false });
              } else if (d.type === 'tool_end') {
                steps = steps.map((s) =>
                  s.kind === 'tool' && s.id === d.id
                    ? { ...s, status: d.ok ? 'ok' : 'error', summary: d.summary, elapsed_ms: d.elapsed_ms, output: d.output }
                    : s,
                );
                patch({ steps });
              } else if (d.type === 'thought') {
                // 这一轮流出的文字其实是调用工具前的说明，从回答区挪到步骤里
                steps = [...steps, { kind: 'thought', round: d.round, text: d.text }];
                if (answerRound === d.round) {
                  answer = '';
                  answerRound = undefined;
                }
                patch({ steps, answer, answerRound, thinking: false });
              } else if (d.type === 'forced_final') {
                steps = [...steps, { kind: 'forced', round: d.round }];
                patch({ steps });
              } else if (d.type === 'repair') {
                // 数值核对不上：接下来的一轮是改写后的回答（token 换了 round，回答区会自动换成新的一轮）
                steps = [...steps, { kind: 'repair', round: d.round, issues: d.issues }];
                patch({ steps });
              }
              break;
            }
            case 'rejected':
              finished = true;
              patch({ status: 'rejected', rejected: ev.data, requestId: ev.data.request_id });
              break;
            case 'token': {
              const r = ev.data.round ?? 0;
              if (answerRound !== r) {
                answer = '';
                answerRound = r;
              }
              answer += ev.data.text ?? '';
              patch({ status: 'generating', answer, answerRound, thinking: false });
              break;
            }
            case 'verification':
              patch({ verification: ev.data });
              break;
            case 'done':
              finished = true;
              // agent 模式以后端最终采用的答案为准（改写更差时退回初稿、拒答去掉引用，与流式文字可能不同）
              patch({
                status: 'done',
                done: ev.data,
                thinking: false,
                requestId: ev.data.request_id,
                ...(typeof ev.data.answer === 'string' ? { answer: ev.data.answer } : {}),
              });
              break;
            case 'error':
              finished = true;
              patch({ status: 'error', error: ev.data.message || '未知错误', thinking: false });
              break;
          }
        },
        controller.signal,
      )
        .then(() => {
          if (!finished) patch({ status: 'error', error: '连接中断，回答可能不完整', thinking: false });
        })
        .catch((e) => {
          if (controller.signal.aborted) patch({ status: 'stopped', thinking: false });
          else patch({ status: 'error', error: e instanceof Error ? e.message : String(e), thinking: false });
        })
        .finally(() => {
          busyRef.current = false;
          streamingSidRef.current = null;
          abortRef.current = null;
          setBusy(false);
        });
    },
    [activeId, patchTurn, effectiveMode],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  // 👍/👎：先写回后端，成功了再记到本地会话里（失败时抛给按钮显示）。不改会话的更新时间，免得侧栏顺序跳动
  const giveFeedback = useCallback(async (turn: Turn, rating: Rating | null, comment?: string) => {
    if (!turn.requestId) return;
    await sendFeedback(turn.requestId, rating, comment);
    const feedback = rating ? { rating, comment: comment || undefined } : undefined;
    setSessions((prev) =>
      prev.map((s) =>
        s.turns.some((t) => t.id === turn.id)
          ? { ...s, turns: s.turns.map((t) => (t.id === turn.id ? { ...t, feedback } : t)) }
          : s,
      ),
    );
  }, []);

  const newChat = () => {
    setActiveId(null);
    setSidebarOpen(false);
  };

  const selectSession = (id: string) => {
    stickToBottom.current = true;
    setActiveId(id);
    setSidebarOpen(false);
  };

  const deleteSession = (id: string) => {
    if (streamingSidRef.current === id) abortRef.current?.abort();
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeId === id) setActiveId(null);
  };

  const openSourceDrawer = useCallback((chunk: ChunkRef, index: number) => setOpenSource({ chunk, index }), []);
  const closeSourceDrawer = useCallback(() => setOpenSource(null), []);

  return (
    <div className="flex h-full overflow-hidden bg-white text-slate-800">
      <Sidebar
        sessions={sortedSessions}
        activeId={activeId}
        health={health}
        healthError={healthError}
        mobileOpen={sidebarOpen}
        onSelect={selectSession}
        onNew={newChat}
        onDelete={deleteSession}
        onCloseMobile={() => setSidebarOpen(false)}
      />

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar
          title={active?.title ?? '新对话'}
          health={health}
          healthError={healthError}
          onMenu={() => setSidebarOpen(true)}
        />
        <main ref={scrollRef} onScroll={onScroll} className="scroll-thin min-h-0 flex-1 overflow-y-auto">
          {turns.length === 0 ? (
            <Welcome documents={health?.documents ?? []} onAsk={ask} />
          ) : (
            <div className="mx-auto max-w-3xl space-y-10 px-4 pb-6 pt-8">
              {turns.map((t) => (
                <TurnView key={t.id} turn={t} onOpenSource={openSourceDrawer} onRetry={ask} onFeedback={giveFeedback} />
              ))}
            </div>
          )}
        </main>
        <Composer
          value={input}
          busy={busy}
          mode={effectiveMode}
          onModeChange={changeMode}
          onChange={setInput}
          onSubmit={() => ask(input)}
          onStop={stop}
        />
      </div>

      <SourceDrawer source={openSource} onClose={closeSourceDrawer} />
    </div>
  );
}
