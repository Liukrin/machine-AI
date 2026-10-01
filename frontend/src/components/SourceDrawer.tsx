import { useEffect, useMemo, useState } from 'react';
import { fetchChunk, type ChunkDetail } from '../api';
import { pagesLabel, sectionPath, sectionTitle } from '../lib/format';
import type { OpenSource } from '../types';
import { IconAlert, IconFileText, IconTable, IconX } from './icons';

export function SourceDrawer({ source, onClose }: { source: OpenSource | null; onClose: () => void }) {
  const chunkId = source?.chunk.chunk_id;
  const [detail, setDetail] = useState<ChunkDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!chunkId) return;
    let alive = true;
    setDetail(null);
    setError(null);
    fetchChunk(chunkId)
      .then((d) => alive && setDetail(d))
      .catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
  }, [chunkId]);

  useEffect(() => {
    if (!source) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [source, onClose]);

  if (!source) return null;
  const { chunk: c, index } = source;
  const isTable = c.chunk_type === 'table';
  const similarity = c.distance == null ? null : Math.max(0, Math.min(1, 1 - c.distance));
  const path = sectionPath(c);

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      <div className="anim-fade absolute inset-0 bg-slate-900/25" onClick={onClose} />
      <aside className="anim-drawer-right relative flex h-full w-full max-w-[540px] flex-col bg-white shadow-2xl">
        <header className="flex items-start gap-3 border-b border-slate-200 px-5 py-4">
          <span className="mt-0.5 inline-flex h-6 min-w-6 items-center justify-center rounded-md bg-teal-600 px-1.5 text-xs font-semibold text-white">
            {index + 1}
          </span>
          <div className="min-w-0 flex-1">
            <h2 className="text-[15px] font-semibold leading-6 text-slate-900">{sectionTitle(c)}</h2>
            <p className="mt-0.5 text-xs text-slate-500">{c.doc_title}</p>
          </div>
          <button
            type="button"
            onClick={onClose}
            title="关闭（Esc）"
            className="rounded-lg p-1.5 text-slate-400 transition hover:bg-slate-100 hover:text-slate-600"
          >
            <IconX className="h-4 w-4" />
          </button>
        </header>

        <dl className="grid grid-cols-[4.5rem_1fr] gap-x-3 gap-y-2 border-b border-slate-100 px-5 py-3.5 text-xs">
          {c.pages && (
            <>
              <dt className="text-slate-400">页码</dt>
              <dd className="text-slate-700">{pagesLabel(c)}</dd>
            </>
          )}
          <dt className="text-slate-400">类型</dt>
          <dd className="flex items-center gap-1 text-slate-700">
            {isTable ? <IconTable className="h-3.5 w-3.5" /> : <IconFileText className="h-3.5 w-3.5" />}
            {isTable ? '表格' : '正文'}
          </dd>
          <dt className="text-slate-400">相似度</dt>
          {similarity == null || c.distance == null ? (
            <dd className="text-slate-500">（由查表或读取相邻片段得到，没有检索相似度）</dd>
          ) : (
            <dd className="flex items-center gap-2 text-slate-700">
              <span className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-100">
                <span className="block h-full rounded-full bg-teal-500" style={{ width: `${similarity * 100}%` }} />
              </span>
              {similarity.toFixed(3)}
              <span className="text-slate-400">（向量距离 {c.distance.toFixed(4)}）</span>
            </dd>
          )}
          {path.length > 1 && (
            <>
              <dt className="text-slate-400">章节</dt>
              <dd className="text-slate-700">{path.join(' › ')}</dd>
            </>
          )}
          <dt className="text-slate-400">片段 ID</dt>
          <dd className="font-mono text-slate-500">{c.chunk_id}</dd>
        </dl>

        {c.is_sample && (
          <div className="mx-5 mt-4 flex gap-2 rounded-lg bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-800 ring-1 ring-amber-200">
            <IconAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            该片段来自自造示例手册（S1 阶段用于打通 PDF 解析流程），内容为虚构，不应作为维修依据。
          </div>
        )}

        <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-5 py-4">
          {error ? (
            <p className="text-sm text-rose-600">原文加载失败：{error}</p>
          ) : !detail ? (
            <div className="space-y-2.5">
              {[92, 100, 84, 96, 70].map((w, i) => (
                <div key={i} className="skeleton h-3.5 rounded" style={{ width: `${w}%` }} />
              ))}
            </div>
          ) : detail.table_html ? (
            <HtmlTable html={detail.table_html} />
          ) : (
            <TextContent text={detail.content ?? ''} leaf={path[path.length - 1]} />
          )}
        </div>
      </aside>
    </div>
  );
}

/** 正文按空行分段；首段与末级标题相同（已在抽屉标题展示）时省略 */
function TextContent({ text, leaf }: { text: string; leaf?: string }) {
  const paras = text
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter(Boolean);
  const body = leaf && paras[0] === leaf ? paras.slice(1) : paras;
  if (body.length === 0) return <p className="text-sm text-slate-500">（该片段没有正文内容）</p>;
  return (
    <div className="space-y-3 text-sm leading-7 text-slate-700">
      {body.map((p, i) => (
        <p key={i} className="whitespace-pre-wrap">
          {p}
        </p>
      ))}
    </div>
  );
}

type Cell = { text: string; rowSpan: number; colSpan: number };

const clampSpan = (v: string | null) => Math.min(Math.max(Number(v) || 1, 1), 100);

/** 表格 HTML 来自 MinerU 解析；只提取单元格文本与合并信息重建成 React 表格，不注入原始 HTML。 */
function parseTable(html: string): Cell[][] {
  const doc = new DOMParser().parseFromString(html, 'text/html');
  return Array.from(doc.querySelectorAll('tr')).map((tr) =>
    Array.from(tr.children)
      .filter((el) => el.tagName === 'TD' || el.tagName === 'TH')
      .map((el) => ({
        text: (el.textContent ?? '').trim(),
        rowSpan: clampSpan(el.getAttribute('rowspan')),
        colSpan: clampSpan(el.getAttribute('colspan')),
      })),
  );
}

function HtmlTable({ html }: { html: string }) {
  const rows = useMemo(() => parseTable(html), [html]);
  if (rows.length === 0) return <p className="text-sm text-slate-500">（表格内容为空）</p>;
  return (
    <div className="scroll-thin overflow-x-auto rounded-lg ring-1 ring-slate-200">
      <table className="min-w-full border-collapse text-xs leading-5">
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className={i === 0 ? 'bg-slate-50 font-medium text-slate-700' : 'text-slate-600 even:bg-slate-50/50'}>
              {row.map((cell, j) => (
                <td
                  key={j}
                  rowSpan={cell.rowSpan}
                  colSpan={cell.colSpan}
                  className="border border-slate-200 px-2.5 py-1.5 align-top"
                >
                  {cell.text}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
