import { useCallback, useEffect, useRef, useState } from 'react';
import type { ChunkRef } from '../api';
import { pagesLabel, sectionTitle } from '../lib/format';
import { IconChevronDown, IconFileText, IconLayers, IconTable } from './icons';

type StripProps = {
  chunks: ChunkRef[];
  citedIds?: Set<string>;
  hoverId?: string | null;
  dimmed?: boolean;
  onHover?: (id: string | null) => void;
  onOpen: (chunk: ChunkRef, index: number) => void;
};

export function SourceStrip({ chunks, citedIds, hoverId, dimmed, onHover, onOpen }: StripProps) {
  const citedCount = citedIds ? chunks.filter((c) => citedIds.has(c.chunk_id)).length : 0;
  const scrollerRef = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ left: false, right: false });

  // 卡片横向滚动：隐藏原生滚动条，按是否还能滚动显示两侧渐隐与箭头
  const updateEdges = useCallback(() => {
    const el = scrollerRef.current;
    if (!el) return;
    setEdges({ left: el.scrollLeft > 4, right: el.scrollLeft + el.clientWidth < el.scrollWidth - 4 });
  }, []);

  useEffect(() => {
    const el = scrollerRef.current;
    if (!el) return;
    updateEdges();
    const ro = new ResizeObserver(updateEdges);
    ro.observe(el);
    return () => ro.disconnect();
  }, [updateEdges, chunks.length]);

  const scrollBy = (dir: 1 | -1) => scrollerRef.current?.scrollBy({ left: dir * 372, behavior: 'smooth' });

  return (
    <section>
      <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-slate-500">
        <IconLayers className="h-3.5 w-3.5" />
        {dimmed ? '最相近的片段' : '参考来源'}
        <span className="font-normal text-slate-400">
          {chunks.length}
          {citedCount > 0 && ` · 已引用 ${citedCount}`}
        </span>
      </div>
      <div className="relative">
        <div ref={scrollerRef} onScroll={updateEdges} className="no-scrollbar -mx-1 flex snap-x gap-2 overflow-x-auto px-1 py-0.5">
          {chunks.map((c, i) => (
            <SourceCard
              key={c.chunk_id}
              chunk={c}
              index={i}
              cited={citedIds?.has(c.chunk_id) ?? false}
              highlighted={hoverId === c.chunk_id}
              dimmed={dimmed}
              onHover={onHover}
              onOpen={onOpen}
            />
          ))}
        </div>
        {edges.left && <ScrollButton side="left" onClick={() => scrollBy(-1)} />}
        {edges.right && <ScrollButton side="right" onClick={() => scrollBy(1)} />}
      </div>
    </section>
  );
}

function ScrollButton({ side, onClick }: { side: 'left' | 'right'; onClick: () => void }) {
  const left = side === 'left';
  return (
    <div
      className={`pointer-events-none absolute inset-y-0 flex w-14 items-center ${
        left ? '-left-1 justify-start bg-gradient-to-r' : '-right-1 justify-end bg-gradient-to-l'
      } from-white via-white/80 to-transparent`}
    >
      <button
        type="button"
        onClick={onClick}
        title={left ? '向左查看' : '查看更多来源'}
        className="pointer-events-auto flex h-7 w-7 items-center justify-center rounded-full bg-white text-slate-500 shadow-md ring-1 ring-slate-200 transition hover:text-teal-700"
      >
        <IconChevronDown className={`h-4 w-4 ${left ? 'rotate-90' : '-rotate-90'}`} />
      </button>
    </div>
  );
}

type CardProps = {
  chunk: ChunkRef;
  index: number;
  cited: boolean;
  highlighted: boolean;
  dimmed?: boolean;
  onHover?: (id: string | null) => void;
  onOpen: (chunk: ChunkRef, index: number) => void;
};

function SourceCard({ chunk, index, cited, highlighted, dimmed, onHover, onOpen }: CardProps) {
  const isTable = chunk.chunk_type === 'table';
  return (
    <button
      type="button"
      onClick={() => onOpen(chunk, index)}
      onMouseEnter={() => onHover?.(chunk.chunk_id)}
      onMouseLeave={() => onHover?.(null)}
      title={`${chunk.doc_title}\n${chunk.heading_path ?? ''}`}
      className={`flex w-[178px] shrink-0 snap-start flex-col rounded-xl border bg-white p-3 text-left transition ${
        highlighted
          ? 'border-teal-400 shadow-[0_0_0_3px_rgba(20,184,166,0.15)]'
          : 'border-slate-200 hover:border-slate-300 hover:shadow-sm'
      } ${dimmed ? 'opacity-75' : ''}`}
    >
      <span className="flex items-center gap-1.5 text-[11px] text-slate-500">
        <span
          className={`inline-flex h-[18px] min-w-[18px] items-center justify-center rounded-md px-1 font-semibold transition ${
            cited ? 'bg-teal-600 text-white' : 'bg-slate-100 text-slate-500'
          }`}
        >
          {index + 1}
        </span>
        {isTable ? <IconTable className="h-3.5 w-3.5" /> : <IconFileText className="h-3.5 w-3.5" />}
        {isTable ? '表格' : '正文'}
        {chunk.pages && <span className="ml-auto text-slate-400">{pagesLabel(chunk, true)}</span>}
      </span>
      <span className="mt-2 line-clamp-2 text-[13px] font-medium leading-5 text-slate-800">{sectionTitle(chunk)}</span>
      <span className="mt-auto flex items-center gap-1 pt-2 text-[11px] text-slate-500">
        <span className="truncate">{chunk.doc_short}</span>
        {chunk.is_sample && (
          <span className="shrink-0 rounded bg-amber-50 px-1 text-amber-700 ring-1 ring-amber-200">示例</span>
        )}
      </span>
    </button>
  );
}

export function SourceSkeleton() {
  return (
    <section>
      <div className="mb-2 flex items-center gap-1.5 text-xs font-medium text-slate-400">
        <IconLayers className="h-3.5 w-3.5" />
        参考来源
      </div>
      <div className="flex gap-2 overflow-hidden pb-1.5">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="h-[92px] w-[178px] shrink-0 rounded-xl border border-slate-100 bg-white p-3">
            <div className="skeleton h-3 w-16 rounded" />
            <div className="skeleton mt-3 h-3 w-32 rounded" />
            <div className="skeleton mt-2 h-3 w-24 rounded" />
          </div>
        ))}
      </div>
    </section>
  );
}
