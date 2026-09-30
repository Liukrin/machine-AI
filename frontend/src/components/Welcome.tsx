import type { ReactNode } from 'react';
import type { DocumentInfo } from '../api';
import { IconArrowRight, IconFileText, IconShieldCheck, IconSparkle, IconTable, LogoMark } from './icons';

// 示例问题均取自评测集或已验证可答的问题，覆盖正文 / 表格 / 多来源三种形态
const EXAMPLES: { tag: string; icon: ReactNode; question: string }[] = [
  { tag: '安装对中', icon: <IconFileText className="h-3.5 w-3.5" />, question: '水泵与电动机同心度允差是多少？' },
  { tag: '表格查询', icon: <IconTable className="h-3.5 w-3.5" />, question: '出水管径125时轴封水量是多少？' },
  { tag: '安全事项', icon: <IconShieldCheck className="h-3.5 w-3.5" />, question: '更换耐磨环前需注意什么安全事项？' },
  { tag: '多手册综合', icon: <IconSparkle className="h-3.5 w-3.5" />, question: '泵启动前要做哪些检查？' },
];

const OUT_OF_SCOPE = '如何配置 Nginx 反向代理？';

export function Welcome({ documents, onAsk }: { documents: DocumentInfo[]; onAsk: (q: string) => void }) {
  const manuals = documents.filter((d) => !d.is_sample);
  const samples = documents.filter((d) => d.is_sample);
  return (
    <div className="anim-fade-up mx-auto flex max-w-2xl flex-col items-center px-4 pb-12 pt-[10vh] text-center">
      <LogoMark size="lg" />
      <h1 className="mt-5 text-[26px] font-semibold tracking-tight text-slate-900">遇到设备问题，直接问手册</h1>
      <p className="mt-2 max-w-lg text-sm leading-6 text-slate-500">
        基于已收录的泵类设备手册回答安装、运行与维护问题。每条结论标注原文出处，点击角标即可查看原文；找不到依据时会直接说明。
      </p>

      <div className="mt-8 grid w-full gap-3 text-left sm:grid-cols-2">
        {EXAMPLES.map((ex) => (
          <button
            key={ex.question}
            type="button"
            onClick={() => onAsk(ex.question)}
            className="group flex flex-col rounded-xl border border-slate-200 bg-white p-4 transition hover:-translate-y-0.5 hover:border-teal-300 hover:shadow-[0_6px_20px_-8px_rgba(13,148,136,0.35)]"
          >
            <span className="flex items-center gap-1.5 text-xs font-medium text-teal-700">
              {ex.icon}
              {ex.tag}
            </span>
            <span className="mt-2 flex items-end justify-between gap-2 text-sm font-medium leading-6 text-slate-800">
              {ex.question}
              <IconArrowRight className="mb-1 h-4 w-4 shrink-0 text-slate-300 transition group-hover:translate-x-0.5 group-hover:text-teal-600" />
            </span>
          </button>
        ))}
      </div>

      {manuals.length > 0 && (
        <div className="mt-8 flex flex-wrap items-center justify-center gap-2 text-xs">
          <span className="text-slate-400">已收录</span>
          {manuals.map((d) => (
            <span key={d.doc_id} title={d.doc_title} className="rounded-full bg-slate-100 px-2.5 py-1 text-slate-600">
              {d.doc_short}
            </span>
          ))}
          {samples.length > 0 && (
            <span
              title={`另含 ${samples.length} 份自造示例手册（S1 阶段打通解析流程用）：${samples.map((d) => d.doc_title).join('、')}`}
              className="rounded-full border border-dashed border-slate-300 px-2.5 py-1 text-slate-400"
            >
              + {samples.length} 份示例
            </span>
          )}
        </div>
      )}

      <p className="mt-4 text-xs text-slate-400">
        超出手册范围的问题会被拒答，试试
        <button type="button" onClick={() => onAsk(OUT_OF_SCOPE)} className="ml-1 text-teal-700 underline-offset-2 hover:underline">
          「{OUT_OF_SCOPE}」
        </button>
      </p>
    </div>
  );
}
