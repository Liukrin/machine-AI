"""评审模型（LLM-as-judge）：逐句忠实度与引用是否成立、关键事实是否答出。

评审模型与生成模型是同一个（llm_config 里的 DeepSeek 模型），存在自评偏差，所以做了三件事来约束它：
  1. 让它对每个「有依据」的判断摘抄原文，代码再核对这段原文确实在片段里
     （忽略标点、空白与项目符号后逐字包含），摘抄对不上的判断不计入「严格忠实度」；
  2. 关键事实以确定性正则匹配为主指标，评审结果只作对照，两者不一致的条目列出来供人复核；
  3. 输出必须是结构完整的 JSON，解析失败或条数不对按失败记，排除出分母并单独计数，不当成通过。
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402

from dataset import Corpus  # noqa: E402
from llm_config import LLM_CONFIG, client_kwargs  # noqa: E402
from textnorm import loose, strip_citations  # noqa: E402

SUPPORT_VALUES = ("full", "partial", "none", "na")
VERDICT_VALUES = ("present", "absent", "contradicted")


class Judge:
    def __init__(self, jcfg: dict, cache_path: Path, model: str | None = None):
        if not LLM_CONFIG["api_key"]:
            raise RuntimeError("DEEPSEEK_API_KEY 未设置，无法调用评审模型。")
        self.cfg = jcfg
        # 评审身份：写进缓存键和报告。--reuse 复算旧评测时传入当时记录的模型名（只查缓存、不调用），
        # 这样模型改名（deepseek-chat → deepseek-flash）之前的评审缓存仍然能用
        self.model = model or LLM_CONFIG["model"]
        self.llm = ChatOpenAI(**client_kwargs(temperature=float(jcfg["temperature"]), max_tokens=int(jcfg["max_tokens"]),
                                              response_format={"type": "json_object"}))
        self.prompts = {
            "faithfulness": (ROOT / jcfg["faithfulness_prompt"]).read_text(encoding="utf-8"),
            "facts": (ROOT / jcfg["facts_prompt"]).read_text(encoding="utf-8"),
        }
        self.cache_path = cache_path
        self.cache: dict[str, dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row
        self._lock = threading.Lock()
        self.reuse_only = False      # True 时只用缓存，缺了就报错（零 LLM 调用复算）
        self._loose_cache: dict[str, str] = {}
        self._used: set[str] = set()   # 本次用到的缓存条目；全量评测时据此清掉过期条目
        self.n_calls = 0
        self.n_cached = 0

    def _loose(self, corpus: Corpus, chunk_id: str) -> str:
        if chunk_id not in self._loose_cache:
            self._loose_cache[chunk_id] = loose(corpus.text[chunk_id])
        return self._loose_cache[chunk_id]

    def prompt_hash(self) -> str:
        h = hashlib.sha1()
        for k in sorted(self.prompts):
            h.update(self.prompts[k].encode("utf-8"))
        return h.hexdigest()[:12]

    def save(self, prune: bool = False) -> None:
        """写回缓存。prune=True 时只保留本次用到的条目（答案或评审提示词变了之后，旧条目就没用了）。"""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [self.cache[k] for k in sorted(self.cache) if not prune or k in self._used]
        self.cache_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
            encoding="utf-8")

    # ----------------------------------------------------------------------- 调用
    def _ask(self, kind: str, ref: str, user: str, check) -> dict:
        """调一次评审模型。check(parsed) 返回规整后的结果，结构不对时抛 ValueError。"""
        system = self.prompts[kind]
        key = hashlib.sha1("\x1f".join([kind, self.model, system, user]).encode("utf-8")).hexdigest()
        with self._lock:
            self._used.add(key)
            if key in self.cache:
                self.n_cached += 1
                return self.cache[key]["result"]
        if self.reuse_only:
            raise RuntimeError(f"--reuse 要求评审结果已缓存，但 {kind} {ref} 没有缓存"
                               "（答案、评审提示词或片段变了）。去掉 --reuse 重新评审。")
        last = ""
        for attempt in range(int(self.cfg["retries"]) + 1):
            try:
                # temperature=0 下原样重试只会得到同样的错误输出，所以重试时把上次的问题告诉它
                hint = (f"\n\n（上一次的输出不符合要求：{last}。请重新输出合法的 JSON，条目数与序号严格对应，"
                        "quote 里不要出现英文双引号。）") if last else ""
                msg = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=user + hint)])
                if (msg.response_metadata or {}).get("finish_reason") == "length":
                    raise ValueError("评审输出被 max_tokens 截断")
                result = check(json.loads(msg.content))
                with self._lock:
                    self.n_calls += 1
                    self.cache[key] = {"key": key, "kind": kind, "ref": ref, "result": result}
                return result
            except Exception as exc:  # 显式记录后重试；耗尽则按失败返回，不缓存、不当成通过
                last = f"{type(exc).__name__}: {exc}"
                print(f"  !! 评审 {kind} {ref} 第 {attempt + 1} 次失败：{last}", file=sys.stderr)
                time.sleep(float(self.cfg["retry_sleep"]))
        with self._lock:
            self.n_calls += 1
        return {"failed": True, "error": last}

    # ----------------------------------------------------------------------- 忠实度
    def faithfulness(self, ref: str, question: str, sentences: list[dict], retrieved: list[str],
                     corpus: Corpus, calc: list[dict] | None = None) -> dict:
        """逐句判断是否有片段依据。返回 {declined, sentences:[{support, by, quote, quote_ok, cited_ok}]}。

        calc 是 Agent 换算、核对工具的输出（scoring 里整理的 {text, inputs, source}）：原文编号成 calc_1、calc_2…
        和手册片段一起给评审，换算结果、核对结论以它为依据（这些数字手册原文里没有）。没有工具输出时，输入与以前完全相同。
        """
        calc = calc or []
        sources = {cid: corpus.text[cid] for cid in retrieved}
        calc_ids = [f"calc_{i}" for i in range(1, len(calc) + 1)]
        sources.update(zip(calc_ids, (c["text"] for c in calc)))
        calc_by_id = dict(zip(calc_ids, calc))
        context = "\n\n".join(f"[{cid}]\n{text}" for cid, text in sources.items())
        # 用「【句 i】」编号：答案句子里常有「1. …；2. …」这样的内部序号，用「1.」编号会被当成多句
        numbered = "\n".join(f"【句 {i}】{s['text']}" for i, s in enumerate(sentences, 1))
        note = ("（编号以 calc_ 开头的是计算工具的输出：单位换算结果、实测值与手册限值的比较结论，由确定性代码算出，"
                "可作为依据；手册限值本身仍须出自手册片段。）\n\n") if calc else ""
        user = (f"问题：{question}\n\n手册片段：\n{context}\n\n{note}"
                f"待核对的句子（共 {len(sentences)} 句，每个【句 i】是一句，引用标记已去掉）：\n{numbered}")

        def check(parsed: dict) -> dict:
            rows = parsed.get("sentences")
            if not isinstance(rows, list) or len(rows) != len(sentences):
                raise ValueError(f"句数不符：应 {len(sentences)}，得 {len(rows) if isinstance(rows, list) else rows!r}")
            by_i = {int(r["i"]): r for r in rows}
            if sorted(by_i) != list(range(1, len(sentences) + 1)):
                raise ValueError(f"句子序号不完整：{sorted(by_i)}")
            out = []
            for i in range(1, len(sentences) + 1):
                r = by_i[i]
                if r.get("support") not in SUPPORT_VALUES:
                    raise ValueError(f"第 {i} 句 support 非法：{r.get('support')!r}")
                out.append({"support": r["support"],
                            "by": [c for c in (r.get("by") or []) if isinstance(c, str)],
                            "quote": str(r.get("quote") or "")})
            return {"declined": bool(parsed.get("declined")), "sentences": out}

        result = self._ask("faithfulness", ref, user, check)
        if result.get("failed"):
            return result
        # 以下是确定性后处理：摘抄的原文是否真的在片段里；句子自己标的引用是否在支持片段之列
        out = []
        for s, r in zip(sentences, result["sentences"]):
            by = [c for c in r["by"] if c in sources]
            q = loose(r["quote"])
            where = by or list(sources)
            quote_ok = bool(q) and any(q in (self._loose(corpus, c) if c in corpus.text else loose(sources[c]))
                                       for c in where)
            cited_ok = None
            if s["cited"] and r["support"] != "na":
                # 依据是换算/核对工具输出时，Agent 按要求标的是原始数值（或限值）所在的片段，评审往往只列 calc 编号：
                # 由代码核对标的片段是不是核对的限值出处、或含有被换算的原始数值
                ok_chunks = {c for c in by if c in corpus.text}
                for cid in by:
                    info = calc_by_id.get(cid)
                    if not info:
                        continue
                    for c in s["cited"]:
                        if c == info.get("source") or (c in corpus.text and any(
                                v is not None and round(float(v), 6) in corpus.numbers_lenient(c) for v in info["inputs"])):
                            ok_chunks.add(c)
                cited_ok = r["support"] != "none" and any(c in ok_chunks for c in s["cited"])
            out.append({**r, "by": by, "quote_ok": quote_ok, "cited_ok": cited_ok})
        return {"declined": result["declined"], "sentences": out}

    # ----------------------------------------------------------------------- 关键事实
    def facts(self, ref: str, item: dict, answer: str) -> dict:
        """逐条判断关键事实是否在回答里。返回 {verdicts: [present|absent|contradicted]}。"""
        facts = item["facts"]
        numbered = "\n".join(f"{i}. {f.get('say') or f['any'][0]}" for i, f in enumerate(facts, 1))
        user = (f"问题：{item['question'] if item['type'] != 'multi_turn' else item['standalone']}\n\n"
                f"参考答案：{item['reference']}\n\n关键事实（共 {len(facts)} 条）：\n{numbered}\n\n"
                f"待评分的回答：\n{strip_citations(answer)}")

        def check(parsed: dict) -> dict:
            rows = parsed.get("facts")
            if not isinstance(rows, list) or len(rows) != len(facts):
                raise ValueError(f"事实条数不符：应 {len(facts)}，得 {len(rows) if isinstance(rows, list) else rows!r}")
            by_i = {int(r["i"]): r.get("verdict") for r in rows}
            if sorted(by_i) != list(range(1, len(facts) + 1)):
                raise ValueError(f"事实序号不完整：{sorted(by_i)}")
            bad = [v for v in by_i.values() if v not in VERDICT_VALUES]
            if bad:
                raise ValueError(f"verdict 非法：{bad}")
            return {"verdicts": [by_i[i] for i in range(1, len(facts) + 1)]}

        return self._ask("facts", ref, user, check)


def run_jobs(jobs: list, concurrency: int) -> list:
    """并发执行评审任务（每个 job 是无参函数），保持顺序返回。"""
    if not jobs:
        return []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(lambda fn: fn(), jobs))
