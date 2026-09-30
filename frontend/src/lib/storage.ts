import type { Session } from '../types';

// 对话历史只存在当前浏览器（localStorage），不上传后端
const KEY = 'equip-rag.sessions.v1';
const MAX_SESSIONS = 30;

export function loadSessions(): Session[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // 刷新前未结束的流已断开，统一标记为已停止
    return (parsed as Session[]).map((s) => ({
      ...s,
      turns: s.turns.map((t) =>
        t.status === 'retrieving' || t.status === 'generating' ? { ...t, status: 'stopped' } : t,
      ),
    }));
  } catch {
    return [];
  }
}

export function saveSessions(sessions: Session[]): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(sessions.slice(0, MAX_SESSIONS)));
  } catch {
    // 隐私模式 / 配额不足时只保留内存中的会话
  }
}
