// 把答案里的 chunk_id 引用（[4_c0319] / 【2_c0044】 / （chunk_id: 4_c0026） / 4_c0244, 4_c0227 等写法）
// 统一改写成 Markdown 链接 [cite](#cite-<id>)，由 Answer 组件渲染成可点击的来源角标。

const ID = String.raw`[A-Za-z0-9_]+_c\d{4}`;
// 引用内部只允许行内空白：换行要留给 Markdown（列表项、段落），不能被引用吞掉
const SP = String.raw`[ \t　]*`;
const ITEM = String.raw`[\[【]?${SP}${ID}${SP}[\]】]?`;
const GROUP = String.raw`${ITEM}(?:${SP}[,，、;；]?${SP}${ITEM})*`;
// 引用前常带的标签：「引用 chunk_id:」「chunk_id：」等
const LABEL = String.raw`(?:引用${SP}的?${SP})?chunk[ _-]?id${SP}[:：]?${SP}`;

// 被括号整体包住的引用组（可带标签）连同括号一起替换；否则替换「标签 + 引用组」或引用组本身。
// 单次扫描：替换结果里仍含 chunk_id，分多轮替换会重复改写。
const CITATION_RE = new RegExp(String.raw`[（(]${SP}(?:${LABEL})?(${GROUP})${SP}[)）]|(${LABEL})?(${GROUP})`, 'gi');
const ID_RE = new RegExp(ID, 'g');
// 末尾没写完的引用：流式输出中（「[4_c03」「（chunk_id: 4_c0」）或被 max_tokens 截断（「引用 chunk_id: [」），不展示；
// 末尾已是完整 chunk_id 的（「引用 chunk_id：4_c0027」）不算，交给 CITATION_RE 渲染
const DANGLING_RE = new RegExp(
  String.raw`(?:[（(]\s*)?(?:${LABEL}[\[【]?|[\[【])\s*(?!${ID}$)[A-Za-z0-9_]*$`,
  'i',
);

export const CITE_PREFIX = '#cite-';

export function extractCitedIds(text: string): Set<string> {
  return new Set(text.match(ID_RE) ?? []);
}

export function linkCitations(text: string): string {
  return text.replace(DANGLING_RE, '').replace(CITATION_RE, (_m, inParens?: string, label?: string, bare?: string) => {
    const ids = (inParens ?? bare ?? '').match(ID_RE) ?? [];
    const chips = ids.map((id) => `[cite](${CITE_PREFIX}${id})`).join('');
    // 答案末尾单独一行的「引用 chunk_id: …」展示成更自然的「来源：」
    return label?.includes('引用') ? `来源：${chips}` : chips;
  });
}

// 与评测脚本的模型自述拒答判定口径一致（configs/config.yaml s4_halluc_ab.refusal_*）
const REFUSAL_MARKERS = ['检索内容不足', '知识库无相关内容', '无法回答', '未能找到相关'];
const REFUSAL_MAX_CHARS = 40;

export function isModelRefusal(answer: string): boolean {
  const t = answer.replace(/\s+/g, '');
  return t.length <= REFUSAL_MAX_CHARS && REFUSAL_MARKERS.some((m) => t.includes(m));
}
