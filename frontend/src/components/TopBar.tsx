import type { Health } from '../api';
import { IconMenu } from './icons';

type Props = {
  title: string;
  health: Health | null;
  healthError: string | null;
  onMenu: () => void;
};

export function TopBar({ title, health, healthError, onMenu }: Props) {
  return (
    <header className="flex h-14 shrink-0 items-center gap-2 border-b border-slate-200/80 bg-white/90 px-4 backdrop-blur">
      <button
        type="button"
        onClick={onMenu}
        title="对话历史"
        className="-ml-1 rounded-lg p-1.5 text-slate-500 transition hover:bg-slate-100 lg:hidden"
      >
        <IconMenu className="h-5 w-5" />
      </button>
      <h1 className="min-w-0 flex-1 truncate text-sm font-medium text-slate-700">{title}</h1>
      {health ? (
        <span className="flex items-center gap-1.5 rounded-full bg-emerald-50 px-2.5 py-1 text-[11px] font-medium text-emerald-700 ring-1 ring-emerald-100">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
          {health.model_name}
        </span>
      ) : healthError ? (
        <span className="flex items-center gap-1.5 rounded-full bg-rose-50 px-2.5 py-1 text-[11px] font-medium text-rose-700 ring-1 ring-rose-100">
          <span className="h-1.5 w-1.5 rounded-full bg-rose-500" />
          后端未连接
        </span>
      ) : null}
    </header>
  );
}
