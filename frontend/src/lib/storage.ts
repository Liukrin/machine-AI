import type { Mode } from '../api';
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
        t.status === 'retrieving' || t.status === 'generating' ? { ...t, status: 'stopped', thinking: false } : t,
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

// 问答模式只是本浏览器的偏好；没选过时返回 null，由后端的默认模式决定
const MODE_KEY = 'equip-rag.mode.v1';

export function loadMode(): Mode | null {
  try {
    const v = localStorage.getItem(MODE_KEY);
    return v === 'agent' || v === 'rag' ? v : null;
  } catch {
    return null;
  }
}

export function saveMode(mode: Mode): void {
  try {
    localStorage.setItem(MODE_KEY, mode);
  } catch {
    // 存不了就只在本次页面里生效
  }
}
