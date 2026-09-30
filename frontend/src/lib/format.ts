import type { ChunkMeta, ChunkRef } from '../api';

// MinerU 常把「注意：/警告：」这类提示框标签识别成标题，单独展示没有信息量
const CALLOUT_RE = /^(注意|警告|小心|危险|提示|说明|注)\s*[:：]?$/;

export function headingSegments(c: ChunkMeta): string[] {
  return (c.heading_path ?? '')
    .split(' > ')
    .map((s) => s.trim())
    .filter(Boolean);
}

/** 章节路径：第一段通常是手册名（与 doc_title 重复），多于一段时去掉 */
export function sectionPath(c: ChunkMeta): string[] {
  const segs = headingSegments(c);
  return segs.length > 1 ? segs.slice(1) : segs;
}

/** 预览文本去掉开头重复的末级标题 */
export function snippet(c: ChunkRef): string {
  let t = c.preview;
  const segs = headingSegments(c);
  const leaf = segs[segs.length - 1];
  if (leaf && t.startsWith(leaf)) t = t.slice(leaf.length);
  return t.replace(/^[\s:：·•]+/, '');
}

/** 来源卡片标题：末两级章节；末级是提示框标签时改用「上级章节 · 标签」或「标签：正文开头」 */
export function sectionTitle(c: ChunkRef): string {
  const path = sectionPath(c);
  if (path.length === 0) return c.doc_short;
  const leaf = path[path.length - 1];
  if (CALLOUT_RE.test(leaf)) {
    const label = leaf.replace(/[:：]\s*$/, '');
    const parent = path.length > 1 ? path[path.length - 2] : '';
    return parent ? `${parent} · ${label}` : `${label}：${snippet(c)}`;
  }
  return path.slice(-2).join(' › ');
}

export function pagesLabel(c: ChunkMeta, short = false): string {
  if (!c.pages) return '';
  const [a, b] = c.pages;
  if (short) return a === b ? `P${a}` : `P${a}–${b}`;
  return a === b ? `第 ${a} 页` : `第 ${a}–${b} 页`;
}

export function fmtMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)} 秒` : `${ms} 毫秒`;
}

export function fmtRelative(ts: number): string {
  const diff = Date.now() - ts;
  if (diff < 60_000) return '刚刚';
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} 分钟前`;
  const d = new Date(ts);
  const now = new Date();
  if (d.toDateString() === now.toDateString()) {
    return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
  }
  return `${d.getMonth() + 1}月${d.getDate()}日`;
}

export function newId(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 8)}`;
}
