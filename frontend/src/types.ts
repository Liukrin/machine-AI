import type { ChunkRef, Done, Rejected, Verification } from './api';

export type TurnStatus = 'retrieving' | 'generating' | 'done' | 'rejected' | 'error' | 'stopped';

export type Turn = {
  id: string;
  question: string;
  status: TurnStatus;
  chunks: ChunkRef[];
  answer: string;
  verification: Verification | null;
  rejected: Rejected | null;
  done: Done | null;
  error: string | null;
  createdAt: number;
};

export type Session = {
  id: string;
  title: string;
  turns: Turn[];
  createdAt: number;
  updatedAt: number;
};

/** 来源详情抽屉当前打开的片段（index 为该片段在本轮检索结果里的序号） */
export type OpenSource = { chunk: ChunkRef; index: number };
