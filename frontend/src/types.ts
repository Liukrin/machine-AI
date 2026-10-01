import type { ChunkRef, Done, Mode, Rating, Rejected, Verification } from './api';

export type TurnStatus = 'retrieving' | 'generating' | 'done' | 'rejected' | 'error' | 'stopped';

/** Agent 的一步（工具调用或中间说明），由 step 事件合并而来 */
export type AgentStep =
  | {
      kind: 'tool';
      id: string;
      round: number;
      name: string;
      label: string;
      args: Record<string, unknown>;
      /** 系统自动做的预检索（不是模型发起的） */
      auto: boolean;
      status: 'running' | 'ok' | 'error';
      summary?: string;
      elapsed_ms?: number;
      output?: string | null;
    }
  | { kind: 'thought'; round: number; text: string }
  | { kind: 'forced'; round: number }
  | { kind: 'repair'; round: number; issues: string[] };

export type Turn = {
  id: string;
  question: string;
  status: TurnStatus;
  /** 旧会话记录没有该字段（当时只有 rag 模式） */
  mode?: Mode;
  chunks: ChunkRef[];
  answer: string;
  /** 当前回答文字属于第几次模型调用；新一轮开始输出时清空重来 */
  answerRound?: number;
  steps?: AgentStep[];
  /** 模型正在思考（已开始一次调用、还没有输出文字） */
  thinking?: boolean;
  verification: Verification | null;
  rejected: Rejected | null;
  done: Done | null;
  error: string | null;
  createdAt: number;
  /** 后端请求日志里的编号（日志开启时 done / rejected 事件带回；旧记录没有，不显示 👍/👎） */
  requestId?: string;
  /** 用户的评价（已写回后端） */
  feedback?: { rating: Rating; comment?: string };
};

export type Session = {
  id: string;
  title: string;
  turns: Turn[];
  createdAt: number;
  updatedAt: number;
};

/** 来源详情抽屉当前打开的片段（index 为该片段在本轮来源列表里的序号） */
export type OpenSource = { chunk: ChunkRef; index: number };
