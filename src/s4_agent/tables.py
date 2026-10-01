"""表格按行入库：把 chunk 里 MinerU 的 HTML 表格展开成规整的行（处理 rowspan / colspan），
存进内存 SQLite，供 Agent 的 lookup_table 工具按关键词精确查行。

为什么要按行查：检索是按片段做的，一张大表（如 Model 3700 的故障排除表约 940 token）整块进上下文，
关键行淹没在里面，向量检索也常常排不到前面；零件编号这类题（「泵轴对应的零件编号」）
关键词只有两三个字，三种检索方式都找不到（见 eval/reports/s3_hybrid.md 的 qa_038）。
按行查表只返回命中的行和表头，跨行合并的单元格（如故障表里一个「症状」对应多个「原因」）展开后每行都带上。

只读 data/chunks/chunks.jsonl，不写 data/ 下任何文件；SQLite 建在内存里，进程启动后首次查询时构建。
"""
from __future__ import annotations

import html as _html
import json
import re
import sqlite3
import threading
import unicodedata
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- HTML → 网格
class _TableParser(HTMLParser):
    """收集 <tr> 里的 <td>/<th>：[(文本, rowspan, colspan), ...]。单元格里的其它标签只取文字。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[tuple[str, int, int]]] = []
        self._cell: list[str] | None = None
        self._span = (1, 1)

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.rows.append([])
        elif tag in ("td", "th"):
            a = dict(attrs)
            self._cell = []
            self._span = (_int(a.get("rowspan")), _int(a.get("colspan")))
            if not self.rows:
                self.rows.append([])
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._cell is not None:
            text = re.sub(r"\s+", " ", "".join(self._cell)).strip()
            self.rows[-1].append((text, *self._span))
            self._cell = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def _int(v) -> int:
    try:
        return max(1, int(str(v).strip('"\'')))
    except (TypeError, ValueError):
        return 1


def table_rows(table_html: str) -> list[list[str]]:
    """HTML 表格展开成行：rowspan 的单元格向下填充到每一行，colspan 的单元格只保留一次。"""
    p = _TableParser()
    p.feed(table_html or "")
    pending: dict[tuple[int, int], tuple[str, int]] = {}   # (行, 列) -> (文本, 来源单元格编号)
    out: list[list[str]] = []
    src_id = 0
    for r, row in enumerate(p.rows):
        cells: dict[int, tuple[str, int]] = {}
        col = 0
        for text, rs, cs in row:
            while (r, col) in pending:
                cells[col] = pending.pop((r, col))
                col += 1
            src_id += 1
            for dc in range(cs):
                cells[col + dc] = (text, src_id)
                for dr in range(1, rs):
                    pending[(r + dr, col + dc)] = (text, src_id)
            col += cs
        for key in sorted(k for k in pending if k[0] == r):
            cells[key[1]] = pending.pop(key)
        line, last_src = [], None
        for c in sorted(cells):
            text, sid = cells[c]
            if sid != last_src:              # 同一个 colspan 单元格只出现一次
                line.append(text)
            last_src = sid
        while line and not line[-1]:
            line.pop()
        if any(line):
            out.append(line)
    return out


def _norm(text: str) -> str:
    """匹配用的归一化：全角转半角、小写、去空白（与答案级评测的 textnorm.norm 同一思路）。"""
    s = unicodedata.normalize("NFKC", _html.unescape(text or "")).replace("\\", "")
    return re.sub(r"\s+", "", s).lower()


# --------------------------------------------------------------------------- SQLite
class TableStore:
    """内存 SQLite：tables（每张表一行：手册、章节、表头）与 table_rows（每个数据行一条）。"""

    def __init__(self, chunks: list[dict]):
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self.lock = threading.Lock()
        self.db.executescript("""
            CREATE TABLE tables (
                chunk_id TEXT PRIMARY KEY, doc_id TEXT, heading TEXT, header TEXT,
                context TEXT,      -- 章节标题 + 表头，归一化后用于关键词匹配
                n_rows INTEGER
            );
            CREATE TABLE table_rows (
                chunk_id TEXT, row_idx INTEGER,
                cells TEXT,        -- JSON 数组，展开后的单元格
                text TEXT          -- 归一化后的整行文本，用于关键词匹配
            );
            CREATE INDEX idx_rows_chunk ON table_rows(chunk_id);
        """)
        n_tables = n_rows = 0
        for c in chunks:
            if c.get("chunk_type") != "table" or not c.get("table_html"):
                continue
            rows = table_rows(c["table_html"])
            if not rows:
                continue
            header = " | ".join(rows[0])
            self.db.execute("INSERT INTO tables VALUES (?, ?, ?, ?, ?, ?)",
                            (c["chunk_id"], c["doc_id"], c.get("heading_path") or "", header,
                             _norm((c.get("heading_path") or "") + " " + header), len(rows)))
            self.db.executemany("INSERT INTO table_rows VALUES (?, ?, ?, ?)",
                                [(c["chunk_id"], i, json.dumps(r, ensure_ascii=False), _norm(" | ".join(r)))
                                 for i, r in enumerate(rows)])
            n_tables += 1
            n_rows += len(rows)
        self.db.commit()
        self.n_tables, self.n_rows = n_tables, n_rows

    @staticmethod
    def _like(keyword: str) -> str:
        k = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return f"%{k}%"

    def lookup(self, keywords: list[str], doc_id: str | None = None, limit: int = 12) -> tuple[list[dict], int]:
        """查同时含有全部关键词的行（关键词可出现在该行、表头或章节标题里，但至少一个要出现在行里）。

        返回 (命中行 [{chunk_id, doc_id, heading, header, row_idx, cells}], 命中总数)。
        """
        kws = [_norm(k) for k in keywords if _norm(k)]
        if not kws:
            return [], 0
        likes = [self._like(k) for k in kws]
        where = ["(? IS NULL OR t.doc_id = ?)"]
        params: list = [doc_id, doc_id]
        for p in likes:
            where.append("(r.text LIKE ? ESCAPE '\\' OR t.context LIKE ? ESCAPE '\\')")
            params += [p, p]
        where.append("(" + " OR ".join("r.text LIKE ? ESCAPE '\\'" for _ in likes) + ")")
        params += likes
        sql = ("SELECT r.chunk_id, t.doc_id, t.heading, t.header, r.row_idx, r.cells "
               "FROM table_rows r JOIN tables t ON r.chunk_id = t.chunk_id "
               f"WHERE {' AND '.join(where)} ORDER BY t.doc_id, r.chunk_id, r.row_idx")
        with self.lock:
            rows = self.db.execute(sql, params).fetchall()
        hits = [{"chunk_id": cid, "doc_id": d, "heading": h, "header": hd, "row_idx": i, "cells": json.loads(cells)}
                for cid, d, h, hd, i, cells in rows]
        return hits[:limit], len(hits)

    def n_rows_of(self, chunk_id: str) -> int:
        with self.lock:
            row = self.db.execute("SELECT n_rows FROM tables WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return row[0] if row else 0
