import { memo, useMemo, useState, type ReactNode } from 'react';
import Markdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import type { ChunkRef, Rating } from '../api';
import { CITE_PREFIX, extractCitedIds, isModelRefusal, linkCitations } from '../lib/citations';
import { fmtMs, pagesLabel, sectionTitle } from '../lib/format';
import type { Turn } from '../types';
import {
  IconAlert,
  IconCheck,
  IconChevronDown,
  IconCopy,
  IconRefresh,
  IconShieldAlert,
  IconShieldCheck,
  IconSparkle,
  IconThumbDown,
  IconThumbUp,
  LogoMark,
} from './icons';
import { SourceSkeleton, SourceStrip } from './Sources';
import { StepTimeline } from './Steps';

/** 写回 👍/👎（rating 为 null 表示撤销）；失败时抛错，由按钮提示 */
type FeedbackFn = (turn: Turn, rating: Rating | null, comment?: string) => Promise<void>;

type Props = {
  turn: Turn;
  onOpenSource: (chunk: ChunkRef, index: number) => void;
  onRetry: (question: string) => void;
  onFeedback: FeedbackFn;
};

export const TurnView = memo(function TurnView({ turn, onOpenSource, onRetry, onFeedback }: Props) {
  const [hoverId, setHoverId] = useState<string | null>(null);
  const citedIds = useMemo(() => extractCitedIds(turn.answer), [turn.answer]);
  const streaming = turn.status === 'generating';
  const waiting = (turn.status === 'retrieving' || turn.status === 'generating') && !turn.answer;
  // 后端给出的拒答判定优先（agent 模式）；旧会话记录按前端规则判断
  const refused = turn.status === 'done' && (turn.verification?.refused ?? isModelRefusal(turn.answer));
  const isAgent = turn.mode === 'agent';

  return (
    <div className="anim-fade-up space-y-5">
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-slate-100 px-4 py-2.5 text-[15px] leading-7 text-slate-800">
          {turn.question}
        </div>
      </div>

      <div className="flex gap-3">
        <div className="pt-0.5">
          <LogoMark size="sm" />
        </div>
        <div className="min-w-0 flex-1 space-y-4">
          {turn.status === 'rejected' && turn.rejected ? (
            <>
              <RejectedNotice turn={turn} onOpenSource={onOpenSource} />
              {turn.requestId && <FeedbackRow turn={turn} onFeedback={onFeedback} />}
            </>
          ) : (
            <>
              {isAgent && <StepTimeline turn={turn} />}
              {turn.status === 'retrieving' && <SourceSkeleton />}
              {turn.chunks.length > 0 && (
                <SourceStrip
                  chunks={turn.chunks}
                  citedIds={citedIds}
                  hoverId={hoverId}
                  onHover={setHoverId}
                  onOpen={onOpenSource}
                />
              )}
              {waiting && !(isAgent && turn.steps?.length) && <Progress turn={turn} />}
              {refused ? (
                <InsufficientNotice detail={isAgent ? turn.answer : undefined} />
              ) : (
                turn.answer && (
                  <Answer
                    text={turn.answer}
                    streaming={streaming}
                    chunks={turn.chunks}
                    onHover={setHoverId}
                    onOpen={onOpenSource}
                  />
                )
              )}
              {turn.status === 'done' && turn.done?.finish_reason === 'length' && <TruncatedNotice />}
              {turn.status === 'done' && !refused && <AnswerFooter turn={turn} onFeedback={onFeedback} />}
              {/* 拒答也要能评：误拒（手册里其实有）是最想收集的反馈之一 */}
              {turn.status === 'done' && refused && turn.requestId && <FeedbackRow turn={turn} onFeedback={onFeedback} />}
            </>
          )}
          {turn.status === 'stopped' && <p className="text-xs text-slate-400">已停止生成</p>}
          {turn.status === 'error' && (
            <ErrorNotice message={turn.error ?? '未知错误'} onRetry={() => onRetry(turn.question)} />
          )}
        </div>
      </div>
    </div>
  );
});

function Progress({ turn }: { turn: Turn }) {
  const text =
    turn.status === 'retrieving'
      ? '正在检索手册…'
      : `已找到 ${turn.chunks.length} 个相关片段，正在组织回答…`;
  return (
    <div className="flex items-center gap-2 text-sm text-slate-500">
      <span className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-teal-500/25 border-t-teal-600" />
      {text}
    </div>
  );
}

type AnswerProps = {
  text: string;
  streaming: boolean;
  chunks: ChunkRef[];
  onHover: (id: string | null) => void;
  onOpen: (chunk: ChunkRef, index: number) => void;
};

function Answer({ text, streaming, chunks, onHover, onOpen }: AnswerProps) {
  const markdown = useMemo(() => linkCitations(text), [text]);
  const components = useMemo<Components>(() => {
    const indexById = new Map(chunks.map((c, i) => [c.chunk_id, i]));
    return {
      a({ href, children }) {
        if (href?.startsWith(CITE_PREFIX)) {
          const id = decodeURIComponent(href.slice(CITE_PREFIX.length));
          const idx = indexById.get(id);
          if (idx === undefined) {
            return (
              <span className="cite-chip cite-chip-bad" title={`引用了本次未检索到的片段 ${id}`}>
                ?
              </span>
            );
          }
          const c = chunks[idx];
          return (
            <button
              type="button"
              className="cite-chip"
              title={`${c.doc_short} · ${sectionTitle(c)}${c.pages ? ` · ${pagesLabel(c)}` : ''}`}
              onMouseEnter={() => onHover(id)}
              onMouseLeave={() => onHover(null)}
              onClick={() => onOpen(c, idx)}
            >
              {idx + 1}
            </button>
          );
        }
        return (
          <a href={href} target="_blank" rel="noreferrer">
            {children}
          </a>
        );
      },
    };
  }, [chunks, onHover, onOpen]);

  return (
    <section>
      <div className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-slate-500">
        <IconSparkle className="h-3.5 w-3.5" />
        回答
      </div>
      <div className={`prose-answer ${streaming ? 'is-streaming' : ''}`}>
        <Markdown remarkPlugins={[remarkGfm]} components={components}>
          {markdown}
        </Markdown>
      </div>
    </section>
  );
}

function AnswerFooter({ turn, onFeedback }: { turn: Turn; onFeedback: FeedbackFn }) {
  const [copied, setCopied] = useState(false);
  const [showSuspicious, setShowSuspicious] = useState(false);
  const v = turn.verification;

  let tone: 'ok' | 'warn' | 'bad' = 'ok';
  let label: ReactNode = null;
  if (v) {
    if (v.fabricated_ids.length > 0) {
      tone = 'bad';
      label = `引用了 ${v.fabricated_ids.length} 个本次未检索到的片段：${v.fabricated_ids.join('、')}`;
    } else if (v.cited_ids.length === 0) {
      tone = 'warn';
      label = '回答没有标注引用，请对照来源核实';
    } else if (v.suspicious_count > 0) {
      tone = 'warn';
      label = (
        <button type="button" onClick={() => setShowSuspicious((s) => !s)} className="inline-flex items-center gap-1 hover:underline">
          {v.suspicious_count} 句与原文重合度低，建议核对
          <IconChevronDown className={`h-3 w-3 transition ${showSuspicious ? 'rotate-180' : ''}`} />
        </button>
      );
    } else {
      label = `${v.cited_ids.length} 处引用均指向本次检索到的片段`;
    }
  }

  const toneClass = {
    ok: 'text-teal-700',
    warn: 'text-amber-700',
    bad: 'text-rose-700',
  }[tone];
  const ToneIcon = tone === 'ok' ? IconShieldCheck : IconAlert;

  // agent 模式的数值核对：回答里的数值能否在本次资料里找到、单位是否一致（rag 模式与旧记录没有）
  const issues = v?.number_issues ?? [];
  const checked = v?.numbers_checked ?? 0;
  const repaired = v?.repair?.kept === 'repaired' ? '（改写过一次）' : '';
  const numLabel =
    checked > 0
      ? issues.length > 0
        ? { ok: false, text: `${issues.length} 个数值在资料里核对不上：${issues.map((i) => i.quantity).join('、')}` }
        : { ok: true, text: `${checked} 个数值均在资料中核对到${repaired}` }
      : null;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(turn.answer);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // 剪贴板不可用（非安全上下文）时静默
    }
  };

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-slate-100 pt-3 text-xs">
        {label && (
          <span className={`flex items-center gap-1.5 ${toneClass}`}>
            <ToneIcon className="h-4 w-4 shrink-0" />
            {label}
          </span>
        )}
        {numLabel && (
          <span
            className={`flex items-center gap-1.5 ${numLabel.ok ? 'text-teal-700' : 'text-amber-700'}`}
            title={issues.map((i) => i.detail).join('\n') || undefined}
          >
            {numLabel.ok ? <IconShieldCheck className="h-4 w-4 shrink-0" /> : <IconAlert className="h-4 w-4 shrink-0" />}
            {numLabel.text}
          </span>
        )}
        <span className="ml-auto flex items-center gap-3 text-slate-400">
          {turn.done && (
            <span title={turn.done.cost_yuan != null ? `按高峰时段单价估算约 ¥${turn.done.cost_yuan.toFixed(4)}` : undefined}>
              {turn.done.llm_calls != null && `模型调用 ${turn.done.llm_calls} 次 · `}
              {turn.done.total_tokens != null && `${turn.done.total_tokens.toLocaleString()} tokens · `}
              {fmtMs(turn.done.elapsed_ms)}
            </span>
          )}
          {turn.requestId && <FeedbackButtons turn={turn} onFeedback={onFeedback} />}
          <button
            type="button"
            onClick={copy}
            className="flex items-center gap-1 rounded-md px-1.5 py-1 text-slate-500 transition hover:bg-slate-100 hover:text-slate-700"
            title="复制回答"
          >
            {copied ? <IconCheck className="h-3.5 w-3.5 text-teal-600" /> : <IconCopy className="h-3.5 w-3.5" />}
            {copied ? '已复制' : '复制'}
          </button>
        </span>
      </div>
      {showSuspicious && v && v.suspicious.length > 0 && (
        <ul className="space-y-1 rounded-lg bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-800 ring-1 ring-amber-200">
          {v.suspicious.map((s, i) => (
            <li key={i}>· {s}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

/** 拒答（模型拒答、rag 模式的拒答闸）下方的一行评价 */
function FeedbackRow({ turn, onFeedback }: { turn: Turn; onFeedback: FeedbackFn }) {
  return (
    <div className="flex items-center justify-end gap-2 text-xs text-slate-400">
      这个结果有帮助吗？
      <FeedbackButtons turn={turn} onFeedback={onFeedback} />
    </div>
  );
}

/** 👍/👎：再点一次撤销；点 👎 后弹出一个可选的说明框（「数值错了」「手册里其实有」），写进请求日志 */
function FeedbackButtons({ turn, onFeedback }: { turn: Turn; onFeedback: FeedbackFn }) {
  const [pending, setPending] = useState(false);
  const [failed, setFailed] = useState(false);
  const [asking, setAsking] = useState(false);
  const [comment, setComment] = useState('');
  const current = turn.feedback?.rating ?? null;

  const send = async (rating: Rating | null, text?: string) => {
    setPending(true);
    setFailed(false);
    try {
      await onFeedback(turn, rating, text);
    } catch {
      setFailed(true);
    } finally {
      setPending(false);
    }
  };

  const click = (rating: Rating) => {
    if (pending) return;
    if (current === rating) {
      setAsking(false);
      void send(null);
      return;
    }
    setAsking(rating === -1);
    void send(rating);
  };

  const submitComment = () => {
    setAsking(false);
    if (comment.trim()) void send(-1, comment.trim());
  };

  const btn = (active: boolean, activeClass: string) =>
    `rounded-md p-1 transition disabled:opacity-50 ${active ? activeClass : 'text-slate-400 hover:bg-slate-100 hover:text-slate-600'}`;

  return (
    <span className="relative flex items-center gap-0.5">
      {failed && <span className="mr-1 text-rose-600">没发出去，再点一次</span>}
      <button
        type="button"
        onClick={() => click(1)}
        disabled={pending}
        title={current === 1 ? '已评价：有帮助（再点一次撤销）' : '有帮助'}
        className={btn(current === 1, 'bg-teal-50 text-teal-700')}
      >
        <IconThumbUp className="h-3.5 w-3.5" fill={current === 1 ? 'currentColor' : 'none'} />
      </button>
      <button
        type="button"
        onClick={() => click(-1)}
        disabled={pending}
        title={current === -1 ? `已评价：没帮助${turn.feedback?.comment ? `（${turn.feedback.comment}）` : ''}，再点一次撤销` : '没帮助'}
        className={btn(current === -1, 'bg-rose-50 text-rose-600')}
      >
        <IconThumbDown className="h-3.5 w-3.5" fill={current === -1 ? 'currentColor' : 'none'} />
      </button>
      {asking && (
        // 向上弹出：按钮在回答底部，往下弹会被滚动区域的底边挡住
        <div className="anim-fade-up absolute bottom-full right-0 z-20 mb-2 w-72 rounded-lg bg-white p-2.5 text-left shadow-lg ring-1 ring-slate-200">
          <div className="mb-1.5 text-xs font-medium text-slate-600">哪里不对？（可选）</div>
          <textarea
            autoFocus
            rows={2}
            maxLength={500}
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                submitComment();
              } else if (e.key === 'Escape') {
                setAsking(false);
              }
            }}
            placeholder="如：数值错了、漏了步骤、手册里其实有"
            className="w-full resize-none rounded-md border border-slate-200 px-2 py-1.5 text-xs leading-5 text-slate-700 outline-none focus:border-teal-400"
          />
          <div className="mt-1.5 flex justify-end gap-1.5">
            <button
              type="button"
              onClick={() => setAsking(false)}
              className="rounded-md px-2 py-1 text-xs text-slate-500 transition hover:bg-slate-100"
            >
              跳过
            </button>
            <button
              type="button"
              onClick={submitComment}
              className="rounded-md bg-teal-600 px-2.5 py-1 text-xs font-medium text-white transition hover:bg-teal-700"
            >
              提交
            </button>
          </div>
        </div>
      )}
    </span>
  );
}

function RejectedNotice({ turn, onOpenSource }: { turn: Turn; onOpenSource: Props['onOpenSource'] }) {
  const [showNearest, setShowNearest] = useState(false);
  const r = turn.rejected!;
  return (
    <div className="rounded-xl border border-amber-200 bg-amber-50/70 p-4">
      <div className="flex gap-3">
        <IconShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-amber-600" />
        <div className="min-w-0">
          <div className="text-sm font-semibold text-amber-900">手册中没有找到与该问题相关的内容</div>
          <p className="mt-1 text-sm leading-6 text-amber-900/80">
            最相近片段的向量距离为 <span className="font-mono">{r.top1_distance.toFixed(4)}</span>，高于拒答阈值{' '}
            <span className="font-mono">{r.tau}</span>。为避免编造答案，系统没有调用大模型生成回答。可以换个更具体的说法，或确认问题属于已收录手册的范围。
          </p>
          {turn.chunks.length > 0 && (
            <button
              type="button"
              onClick={() => setShowNearest((s) => !s)}
              className="mt-2 inline-flex items-center gap-1 text-xs font-medium text-amber-800 hover:underline"
            >
              {showNearest ? '收起' : '查看'}最相近的 {turn.chunks.length} 个片段
              <IconChevronDown className={`h-3 w-3 transition ${showNearest ? 'rotate-180' : ''}`} />
            </button>
          )}
        </div>
      </div>
      {showNearest && (
        <div className="mt-3">
          <SourceStrip chunks={turn.chunks} dimmed onOpen={onOpenSource} />
        </div>
      )}
    </div>
  );
}

function TruncatedNotice() {
  return (
    <div className="flex gap-2 rounded-lg bg-amber-50 px-3 py-2 text-xs leading-5 text-amber-800 ring-1 ring-amber-200">
      <IconAlert className="mt-0.5 h-3.5 w-3.5 shrink-0" />
      <span>回答达到生成长度上限被截断，后面的内容缺失。可以把问题拆细一些再问。</span>
    </div>
  );
}

function InsufficientNotice({ detail }: { detail?: string }) {
  // agent 模式的拒答带有「查过什么」的说明，原样展示；rag 模式只有一句拒答话术
  const explain = detail?.replace(/^\s*知识库无相关内容[。.，,：:]?\s*/, '').trim();
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
      <div className="text-sm font-semibold text-slate-800">手册里没有找到能回答这个问题的内容</div>
      <p className="mt-1 text-sm leading-6 text-slate-600">
        {explain || '模型判断上方片段中没有足够的依据，因此没有给出答案。'}
        {' '}可以补充设备型号或换个说法再试。
      </p>
    </div>
  );
}

function ErrorNotice({ message, onRetry }: { message: string; onRetry: () => void }) {
  return (
    <div className="rounded-xl border border-rose-200 bg-rose-50 p-4">
      <div className="flex items-center gap-2 text-sm font-semibold text-rose-800">
        <IconAlert className="h-4 w-4" />
        请求失败
      </div>
      <p className="mt-1 break-all font-mono text-xs leading-5 text-rose-700/90">{message}</p>
      <button
        type="button"
        onClick={onRetry}
        className="mt-3 inline-flex items-center gap-1.5 rounded-md bg-white px-2.5 py-1 text-xs font-medium text-rose-700 ring-1 ring-rose-200 transition hover:bg-rose-100"
      >
        <IconRefresh className="h-3.5 w-3.5" />
        重试
      </button>
    </div>
  );
}
