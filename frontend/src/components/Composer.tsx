import { useLayoutEffect, useRef } from 'react';
import { IconArrowUp, IconStop } from './icons';

type Props = {
  value: string;
  busy: boolean;
  onChange: (v: string) => void;
  onSubmit: () => void;
  onStop: () => void;
};

const MAX_HEIGHT = 180;

export function Composer({ value, busy, onChange, onSubmit, onStop }: Props) {
  const ref = useRef<HTMLTextAreaElement>(null);

  // 输入框随内容自动增高，超过上限后内部滚动
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
    el.style.overflowY = el.scrollHeight > MAX_HEIGHT ? 'auto' : 'hidden';
  }, [value]);

  return (
    <div className="shrink-0 bg-gradient-to-t from-white via-white to-white/0 px-4 pb-4 pt-2">
      <form
        onSubmit={(e) => {
          e.preventDefault();
          onSubmit();
        }}
        className="mx-auto max-w-3xl"
      >
        <div className="flex items-end gap-2 rounded-2xl border border-slate-200 bg-white py-2 pl-4 pr-2 shadow-[0_2px_16px_-4px_rgba(15,23,42,0.08)] transition focus-within:border-teal-400 focus-within:shadow-[0_0_0_4px_rgba(20,184,166,0.12)]">
          <textarea
            ref={ref}
            rows={1}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={(e) => {
              // 输入法组字过程中的回车用于选词，不提交
              if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing && e.keyCode !== 229) {
                e.preventDefault();
                onSubmit();
              }
            }}
            placeholder="描述你遇到的设备问题…"
            className="scroll-thin flex-1 resize-none bg-transparent py-1.5 text-[15px] leading-6 text-slate-800 outline-none placeholder:text-slate-400"
          />
          {busy ? (
            <button
              type="button"
              onClick={onStop}
              title="停止生成"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-slate-900 text-white transition hover:bg-slate-700"
            >
              <IconStop className="h-3.5 w-3.5" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!value.trim()}
              title="发送（Enter）"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-teal-600 text-white transition hover:bg-teal-700 disabled:cursor-not-allowed disabled:bg-slate-100 disabled:text-slate-400"
            >
              <IconArrowUp className="h-4 w-4" />
            </button>
          )}
        </div>
        <p className="mt-2 text-center text-[11px] leading-4 text-slate-400">
          回答由大模型依据手册片段生成，涉及安全操作请以原手册为准 · 每个问题独立检索，暂不关联上文
        </p>
      </form>
    </div>
  );
}
