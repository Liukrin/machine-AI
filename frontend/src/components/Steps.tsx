import { useState, type ReactNode } from 'react';
import type { AgentStep, Turn } from '../types';
import { fmtMs } from '../lib/format';
import { IconAlert, IconChevronDown, IconFileText, IconGauge, IconSearch, IconSwap, IconTable } from './icons';

const TOOL_ICON: Record<string, ReactNode> = {
  search_manuals: <IconSearch className="h-3.5 w-3.5" />,
  lookup_table: <IconTable className="h-3.5 w-3.5" />,
  read_section: <IconFileText className="h-3.5 w-3.5" />,
  convert_unit: <IconSwap className="h-3.5 w-3.5" />,
  check_value: <IconGauge className="h-3.5 w-3.5" />,
};

const KIND_CN: Record<string, string> = { text: '只看正文', table: '只看表格' };
const QUERY_MAX = 32;

/** 检索词太长时截断显示（完整内容在悬停提示里） */
const clip = (s: unknown) => {
  const t = String(s ?? '');
  return t.length > QUERY_MAX ? `${t.slice(0, QUERY_MAX)}…` : t;
};

/** 工具参数的简短说明 */
function argsText(s: Extract<AgentStep, { kind: 'tool' }>): string {
  const a = s.args as Record<string, string | number | boolean | null | undefined>;
  const manual = a.manual ? ` · 手册 ${a.manual}` : '';
  switch (s.name) {
    case 'search_manuals':
      return `「${clip(a.query)}」${manual}${a.kind ? ` · ${KIND_CN[String(a.kind)] ?? a.kind}` : ''}`;
    case 'lookup_table':
      return `「${clip(a.keywords)}」${manual}`;
    case 'read_section':
      return `${a.chunk_id ?? ''} 前后`;
    case 'convert_unit':
      return `${a.value} ${a.from_unit} → ${a.to_unit}${a.is_difference ? '（温差）' : ''}`;
    case 'check_value': {
      const lim = [a.limit_min != null ? `≥ ${a.limit_min}` : '', a.limit_max != null ? `≤ ${a.limit_max}` : '']
        .filter(Boolean)
        .join('、');
      return `${a.value} ${a.unit} 对照 ${lim}${a.limit_unit ? ` ${a.limit_unit}` : ''}（${a.source_chunk_id}）`;
    }
    default:
      return JSON.stringify(a);
  }
}

/**
 * Agent 的查阅过程：每个工具调用一行（预检索、检索、查表、换算、核对），出错的标红并给出错误原因。
 * 进行中默认展开，结束后折叠成一行摘要，可点开。
 */
export function StepTimeline({ turn }: { turn: Turn }) {
  const steps = turn.steps ?? [];
  const running = turn.status === 'retrieving' || turn.status === 'generating';
  const [open, setOpen] = useState<boolean | null>(null);
  if (steps.length === 0) return null;
  const expanded = open ?? running;
  const tools = steps.filter((s): s is Extract<AgentStep, { kind: 'tool' }> => s.kind === 'tool');
  const errors = tools.filter((s) => s.status === 'error').length;
  const llmCalls = turn.done?.llm_calls;

  return (
    <section className="rounded-xl border border-slate-200 bg-slate-50/60">
      <button
        type="button"
        onClick={() => setOpen(!expanded)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-slate-500"
      >
        <span className="font-medium text-slate-600">查阅过程</span>
        <span>
          {tools.length} 步
          {llmCalls != null && ` · 模型调用 ${llmCalls} 次`}
          {errors > 0 && <span className="text-amber-700"> · {errors} 次工具报错后已重试或放弃</span>}
        </span>
        <IconChevronDown className={`ml-auto h-3.5 w-3.5 transition ${expanded ? 'rotate-180' : ''}`} />
      </button>
      {expanded && (
        <ol className="space-y-1.5 border-t border-slate-200/80 px-3 py-2.5">
          {steps.map((s, i) => (
            <StepRow key={s.kind === 'tool' ? s.id : `${s.kind}-${i}`} step={s} />
          ))}
          {running && turn.thinking && (
            <li className="flex items-center gap-2 text-xs text-slate-500">
              <span className="h-3 w-3 animate-spin rounded-full border-2 border-teal-500/25 border-t-teal-600" />
              模型正在阅读结果、决定下一步…
            </li>
          )}
        </ol>
      )}
    </section>
  );
}

function StepRow({ step }: { step: AgentStep }) {
  if (step.kind === 'thought') {
    return <li className="pl-5 text-xs italic leading-5 text-slate-500">模型说明：{step.text}</li>;
  }
  if (step.kind === 'forced') {
    return (
      <li className="flex items-center gap-1.5 text-xs text-amber-700">
        <IconAlert className="h-3.5 w-3.5" />
        已达到工具调用轮数上限，基于已查到的资料作答
      </li>
    );
  }
  const tone =
    step.status === 'error' ? 'text-rose-700' : step.status === 'running' ? 'text-slate-500' : 'text-slate-700';
  return (
    <li className={`text-xs leading-5 ${tone}`}>
      <div className="flex items-start gap-2">
        <span className={`mt-0.5 shrink-0 ${step.status === 'error' ? 'text-rose-500' : 'text-teal-600'}`}>
          {step.status === 'running' ? (
            <span className="block h-3.5 w-3.5 animate-spin rounded-full border-2 border-teal-500/25 border-t-teal-600" />
          ) : (
            (TOOL_ICON[step.name] ?? <IconSearch className="h-3.5 w-3.5" />)
          )}
        </span>
        <span className="min-w-0 flex-1">
          <span className="font-medium">{step.label}</span>
          {step.auto && <span className="ml-1 rounded bg-slate-200/70 px-1 text-[10px] text-slate-500">自动</span>}
          <span className="ml-1.5 break-all text-slate-500" title={JSON.stringify(step.args)}>
            {argsText(step)}
          </span>
          {step.summary && (
            <span className={step.status === 'error' ? 'block text-rose-600' : 'ml-1.5 text-slate-600'}>
              {step.status === 'error' ? `出错：${step.summary}` : `→ ${step.summary}`}
            </span>
          )}
        </span>
        {step.elapsed_ms != null && (
          <span className="shrink-0 tabular-nums text-slate-400">{fmtMs(Math.round(step.elapsed_ms))}</span>
        )}
      </div>
    </li>
  );
}
