import type { Health } from '../api';
import { fmtRelative } from '../lib/format';
import type { Session } from '../types';
import { IconChat, IconManual, IconPlus, IconX, LogoMark } from './icons';

type Props = {
  sessions: Session[];
  activeId: string | null;
  health: Health | null;
  healthError: string | null;
  mobileOpen: boolean;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onCloseMobile: () => void;
};

export function Sidebar(props: Props) {
  const { mobileOpen, onCloseMobile } = props;
  return (
    <>
      <aside className="hidden w-[272px] shrink-0 border-r border-slate-200 bg-slate-50 lg:flex">
        <SidebarContent {...props} />
      </aside>
      {mobileOpen && (
        <div className="fixed inset-0 z-40 flex lg:hidden">
          <div className="absolute inset-0 bg-slate-900/30 backdrop-blur-[2px]" onClick={onCloseMobile} />
          <aside className="anim-drawer-left relative flex w-[280px] max-w-[85%] bg-slate-50 shadow-xl">
            <SidebarContent {...props} />
          </aside>
        </div>
      )}
    </>
  );
}

function SidebarContent({ sessions, activeId, health, healthError, onSelect, onNew, onDelete }: Props) {
  return (
    <div className="flex min-h-0 w-full flex-col">
      <div className="flex items-center gap-2.5 px-4 pb-4 pt-5">
        <LogoMark />
        <div className="min-w-0">
          <div className="truncate text-[15px] font-semibold tracking-tight text-slate-900">设备运维知识助手</div>
          <div className="truncate text-xs text-slate-500">泵类设备手册 · 检索增强问答</div>
        </div>
      </div>

      <div className="px-3">
        <button
          type="button"
          onClick={onNew}
          className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:border-teal-300 hover:text-teal-700"
        >
          <IconPlus className="h-4 w-4" />
          新对话
        </button>
      </div>

      <div className="mt-5 px-4 text-xs font-medium text-slate-400">历史对话</div>
      <nav className="mt-1.5 min-h-0 flex-1 space-y-0.5 overflow-y-auto px-2 pb-3">
        {sessions.length === 0 ? (
          <p className="px-3 py-6 text-center text-xs text-slate-400">还没有对话</p>
        ) : (
          sessions.map((s) => {
            const active = s.id === activeId;
            return (
              <div
                key={s.id}
                className={`group relative flex items-center rounded-lg transition ${
                  active ? 'bg-white shadow-sm ring-1 ring-slate-200' : 'hover:bg-slate-100'
                }`}
              >
                <button type="button" onClick={() => onSelect(s.id)} className="flex min-w-0 flex-1 items-start gap-2.5 px-3 py-2 text-left">
                  <IconChat className={`mt-0.5 h-4 w-4 shrink-0 ${active ? 'text-teal-600' : 'text-slate-400'}`} />
                  <span className="min-w-0">
                    <span className={`block truncate text-sm ${active ? 'font-medium text-slate-900' : 'text-slate-700'}`}>{s.title}</span>
                    <span className="mt-0.5 block text-[11px] text-slate-400">
                      {s.turns.length} 个问题 · {fmtRelative(s.updatedAt)}
                    </span>
                  </span>
                </button>
                <button
                  type="button"
                  onClick={() => onDelete(s.id)}
                  title="删除对话"
                  className="mr-1.5 hidden rounded-md p-1 text-slate-400 hover:bg-slate-200 hover:text-slate-600 group-hover:block"
                >
                  <IconX className="h-3.5 w-3.5" />
                </button>
              </div>
            );
          })
        )}
      </nav>

      <KnowledgeBaseCard health={health} healthError={healthError} />
    </div>
  );
}

function KnowledgeBaseCard({ health, healthError }: { health: Health | null; healthError: string | null }) {
  const manuals = health?.documents.filter((d) => !d.is_sample).length ?? 0;
  const samples = health?.documents.filter((d) => d.is_sample).length ?? 0;
  const state = health ? 'ok' : healthError ? 'down' : 'pending';
  return (
    <div className="border-t border-slate-200 p-3">
      <div className="rounded-xl bg-white p-3 ring-1 ring-slate-200">
        <div className="flex items-center gap-1.5 text-xs font-medium text-slate-700">
          <IconManual className="h-3.5 w-3.5 text-teal-600" />
          知识库
          <span className="ml-auto flex items-center gap-1.5 text-[11px] font-normal text-slate-500">
            <span
              className={`h-1.5 w-1.5 rounded-full ${
                state === 'ok' ? 'bg-emerald-500' : state === 'down' ? 'bg-rose-500' : 'bg-slate-300'
              }`}
            />
            {state === 'ok' ? '服务正常' : state === 'down' ? '后端未连接' : '连接中'}
          </span>
        </div>
        {health ? (
          <div className="mt-2 space-y-0.5 text-[11px] leading-5 text-slate-500">
            <div>
              {manuals} 份手册{samples > 0 && ` + ${samples} 份示例`} · {health.chunk_count} 个片段
            </div>
            <div>
              生成模型 <span className="font-mono text-slate-600">{health.model_name}</span>
            </div>
          </div>
        ) : (
          <div className="mt-2 text-[11px] leading-5 text-slate-400">
            {healthError ? '连不上后端，几秒后自动重试；没启动的话先启动后端（端口 8000）' : '正在连接后端…'}
          </div>
        )}
      </div>
    </div>
  );
}
