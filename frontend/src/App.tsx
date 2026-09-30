import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { askStream, fetchHealth, type ChunkRef, type Health } from './api';
import { Composer } from './components/Composer';
import { Sidebar } from './components/Sidebar';
import { SourceDrawer } from './components/SourceDrawer';
import { TopBar } from './components/TopBar';
import { TurnView } from './components/TurnView';
import { Welcome } from './components/Welcome';
import { newId } from './lib/format';
import { loadSessions, saveSessions } from './lib/storage';
import type { OpenSource, Session, Turn } from './types';

const APP_NAME = '设备运维知识助手';
const TITLE_MAX = 28;

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [sessions, setSessions] = useState<Session[]>(loadSessions);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [input, setInput] = useState('');
  const [busy, setBusy] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [openSource, setOpenSource] = useState<OpenSource | null>(null);

  const busyRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const streamingSidRef = useRef<string | null>(null);
  const scrollRef = useRef<HTMLElement>(null);
  const stickToBottom = useRef(true);

  // 这里只做幂等的读取：StrictMode 下 effect 会执行两次，会话在首次提问时才创建
  useEffect(() => {
    fetchHealth()
      .then(setHealth)
      .catch((e) => setHealthError(String(e)));
  }, []);

  useEffect(() => {
    const t = setTimeout(() => saveSessions(sessions), 400);
    return () => clearTimeout(t);
  }, [sessions]);

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
        chunks: [],
        answer: '',
        verification: null,
        rejected: null,
        done: null,
        error: null,
        createdAt: now,
      };
      const sid = activeId ?? newId();
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
      let finished = false;

      askStream(
        question,
        (ev) => {
          switch (ev.event) {
            case 'retrieval':
              patch({ status: 'generating', chunks: ev.data.chunks ?? [] });
              break;
            case 'rejected':
              finished = true;
              patch({ status: 'rejected', rejected: ev.data });
              break;
            case 'token':
              answer += ev.data.text ?? '';
              patch({ answer });
              break;
            case 'verification':
              patch({ verification: ev.data });
              break;
            case 'done':
              finished = true;
              patch({ status: 'done', done: ev.data });
              break;
            case 'error':
              finished = true;
              patch({ status: 'error', error: ev.data.message || '未知错误' });
              break;
          }
        },
        controller.signal,
      )
        .then(() => {
          if (!finished) patch({ status: 'error', error: '连接中断，回答可能不完整' });
        })
        .catch((e) => {
          if (controller.signal.aborted) patch({ status: 'stopped' });
          else patch({ status: 'error', error: e instanceof Error ? e.message : String(e) });
        })
        .finally(() => {
          busyRef.current = false;
          streamingSidRef.current = null;
          abortRef.current = null;
          setBusy(false);
        });
    },
    [activeId, patchTurn],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

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
                <TurnView key={t.id} turn={t} onOpenSource={openSourceDrawer} onRetry={ask} />
              ))}
            </div>
          )}
        </main>
        <Composer value={input} busy={busy} onChange={setInput} onSubmit={() => ask(input)} onStop={stop} />
      </div>

      <SourceDrawer source={openSource} onClose={closeSourceDrawer} />
    </div>
  );
}
