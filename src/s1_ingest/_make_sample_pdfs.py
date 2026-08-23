"""DEV-ONLY: generate sample Chinese PDFs into data/raw_pdf to exercise the S1 pipeline.

These are synthetic documents (plate cooler / hydraulic pump / solenoid valve O&M
manuals) used only to smoke-test MinerU parsing before the real corpus is dropped in.
Real corpus will replace them; this script can then be deleted.

Run with a python that has matplotlib:  python src/s1_ingest/_make_sample_pdfs.py
"""

import os
from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# --- register a CJK font once ---
FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"
if not os.path.exists(FONT_PATH):
    raise SystemExit(f"CJK font not found: {FONT_PATH}")
fm.fontManager.addfont(FONT_PATH)
matplotlib.rcParams["font.family"] = "SimHei"
matplotlib.rcParams["axes.unicode_minus"] = False

RAW_PDF_DIR = Path(__file__).resolve().parents[2] / "data" / "raw_pdf"


def _header_footer(pages, total):
    return f"某某装备运维手册 · 内部资料", f"第 {pages} 页 / 共 {total} 页"


def _write_pdf(path, blocks):
    """blocks: list of dicts:
       {kind: 'title'|'h1'|'h2'|'text'|'table'|'formula'|'fig', ...}
    """
    total_pages = len(blocks)
    with PdfPages(path) as pdf:
        for idx, blk in enumerate(blocks, start=1):
            fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            fig.suptitle(f"示例语料 —— {path.stem}", fontsize=9, y=0.985, color="gray")
            ax = fig.add_subplot(111)
            ax.axis("off")
            y = 0.92
            header, footer = _header_footer(idx, total_pages)

            # page header (repeated verbatim on every page -> Task 2 target)
            ax.text(0.06, 0.965, header, fontsize=8, color="gray",
                    transform=fig.transFigure)

            for item in blk:
                kind = item["kind"]
                if kind == "title":
                    ax.text(0.06, y, item["text"], fontsize=16, weight="bold",
                            transform=fig.transFigure, wrap=True)
                    y -= 0.055
                elif kind == "h1":
                    ax.text(0.06, y, item["text"], fontsize=13, weight="bold",
                            transform=fig.transFigure)
                    y -= 0.045
                elif kind == "h2":
                    ax.text(0.07, y, item["text"], fontsize=11, weight="bold",
                            transform=fig.transFigure)
                    y -= 0.040
                elif kind == "text":
                    ax.text(0.06, y, item["text"], fontsize=10,
                            transform=fig.transFigure, wrap=True,
                            linespacing=1.5)
                    y -= 0.035 * (1 + item["text"].count("\n"))
                elif kind == "formula":
                    ax.text(0.12, y, item["text"], fontsize=11,
                            transform=fig.transFigure)
                    y -= 0.045
                elif kind == "fig":
                    ax.figure.text(0.08, y - 0.02,
                                   item.get("caption", ""), fontsize=9,
                                   ha="left", color="black")
                    y -= 0.045
                elif kind == "table":
                    tbl = item["table"]
                    col_labels = tbl["col_labels"]
                    cell_text = tbl["rows"]
                    tab = ax.table(cellText=cell_text, colLabels=col_labels,
                                   loc="center", cellLoc="center",
                                   bbox=[0.06, y - 0.18, 0.88, 0.16])
                    tab.auto_set_font_size(False)
                    tab.set_fontsize(8)
                    y -= 0.22
                y -= 0.015
                if y < 0.08:
                    break

            ax.text(0.06, 0.02, footer, fontsize=8, color="gray",
                    transform=fig.transFigure)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"  wrote {path} ({len(blocks)} pages)")


def build_cooler():
    return [
        [
            {"kind": "title", "text": "板式冷却器维护手册（示例）"},
            {"kind": "text", "text": "本手册适用于板式冷却器的日常巡检、清洗与故障判断。适用于海水冷却回路，介质温度不超过 60℃。"},
            {"kind": "h1", "text": "第一章 结构与工作原理"},
            {"kind": "text", "text": "板式冷却器由一组波纹金属板片叠加而成，板片之间形成冷热双通道。冷介质（海水）与热介质（液压油）在相邻通道内逆流换热。"},
            {"kind": "h2", "text": "1.1 主要组成部件"},
            {"kind": "text", "text": "板式冷却器主要由固定压紧板、活动压紧板、换热板片、密封垫圈和紧固螺栓组成。板片数量根据换热面积需求确定，通常为 40 至 120 片。"},
            {"kind": "formula", "text": r"换热面积: $A = \frac{Q}{K \cdot \Delta T_{lm}}$"},
            {"kind": "table", "table": {"col_labels": ["部件", "材质", "数量"],
                                        "rows": [["换热板片", "304 不锈钢", "96"],
                                                 ["密封垫圈", "丁腈橡胶", "192"],
                                                 ["压紧板", "碳钢镀锌", "2"]]}},
        ],
        [
            {"kind": "h1", "text": "第二章 巡检与清洗"},
            {"kind": "text", "text": "日常巡检重点检查进出油口压差、油侧与海侧温差以及是否存在外漏。当压差超过 0.15 MPa 时应安排在线反冲清洗。"},
            {"kind": "h2", "text": "2.1 清洗周期"},
            {"kind": "text", "text": "海水侧每月进行一次反冲洗，油侧每半年进行一次化学清洗。化学清洗剂选用碱性清洗剂，禁用含氯制剂以免腐蚀板片。"},
            {"kind": "fig", "text": "", "caption": "图 1  板式冷却器结构示意图"},
            {"kind": "text", "text": "图 1 展示了板式冷却器的主要结构。实际拆检时应先记录板片排列方式，避免回装时顺序错误导致换热性能下降。"},
        ],
        [
            {"kind": "h1", "text": "第三章 常见故障与排除"},
            {"kind": "table", "table": {"col_labels": ["故障现象", "可能原因", "处置措施"],
                                        "rows": [["油温偏高", "板片结垢", "反冲洗"],
                                                 ["外漏", "密封圈老化", "更换垫圈"],
                                                 ["压差过大", "海水侧堵塞", "拆检清洗"]]}},
            {"kind": "formula", "text": r"传热系数: $K = \frac{1}{\frac{1}{\alpha_o}+\frac{\delta}{\lambda}+\frac{1}{\alpha_i}}$"},
            {"kind": "text", "text": "更换密封圈时应按对角线顺序均匀预紧螺栓，预紧扭矩按制造商规定值执行，禁止单侧过度紧固。"},
        ],
    ]


def build_pump():
    return [
        [
            {"kind": "title", "text": "液压齿轮泵检修规程（示例）"},
            {"kind": "text", "text": "本规程规定液压齿轮泵的拆检、测量与回装要求。适用于额定压力 16 MPa 以下的中压齿轮泵。"},
            {"kind": "h1", "text": "第一章 安全要求"},
            {"kind": "text", "text": "检修前必须泄压并断开动力源，确认泵体温度降至常温后方可拆卸。高处作业应系安全带。"},
        ],
        [
            {"kind": "h1", "text": "第二章 拆检与测量"},
            {"kind": "h2", "text": "2.1 齿顶间隙"},
            {"kind": "text", "text": "齿顶间隙使用塞尺测量，标准值为 0.02 至 0.06 mm。超过 0.10 mm 时应更换齿轮副。"},
            {"kind": "formula", "text": r"容积效率: $\eta_v = \frac{Q_{act}}{Q_{th}} \times 100\%$"},
            {"kind": "table", "table": {"col_labels": ["测量项目", "标准值", "报废值"],
                                        "rows": [["齿顶间隙", "0.02–0.06 mm", ">0.10 mm"],
                                                 ["轴径磨损", "<0.02 mm", ">0.05 mm"],
                                                 ["端面间隙", "0.03–0.08 mm", ">0.12 mm"]]}},
        ],
        [
            {"kind": "h1", "text": "第三章 回装与试车"},
            {"kind": "text", "text": "回装后应进行空载与负载试车，确认无异响、无泄漏、温升正常。油温稳定后测量出口压力是否达到额定值。"},
        ],
    ]


def build_valve():
    return [
        [
            {"kind": "title", "text": "电磁换向阀故障排查手册（示例）"},
            {"kind": "text", "text": "本手册用于电磁换向阀不动作、卡滞及内泄漏故障的现场排查。"},
            {"kind": "h1", "text": "第一章 阀芯卡滞排查"},
            {"kind": "text", "text": "阀芯卡滞多由油液污染或阀芯与阀套间隙过小引起。排查时先检查油液清洁度，再检查控制电流。"},
            {"kind": "table", "table": {"col_labels": ["检查项", "方法", "合格标准"],
                                        "rows": [["控制电压", "万用表测量", "DC 24V ±10%"],
                                                 ["油液清洁度", "取样送检", "NAS 9 级以内"],
                                                 ["阀芯行程", "千分尺", "3.5 mm ±0.1"]]}},
        ],
        [
            {"kind": "h1", "text": "第二章 电气检查"},
            {"kind": "text", "text": "测量线圈电阻值并与铭牌标称值比对。线圈短路时阻值明显偏小，断路时阻值为无穷大。"},
            {"kind": "formula", "text": r"电磁力: $F = \frac{B^2 A}{2\mu_0}$"},
            {"kind": "text", "text": "电磁阀换向时间不宜超过 0.1 秒，超过时应检查复位弹簧是否疲劳。"},
        ],
    ]


def main():
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)
    builders = [
        ("sample_cooler_manual", build_cooler),
        ("sample_pump_manual", build_pump),
        ("sample_valve_manual", build_valve),
    ]
    print(f"writing sample PDFs to {RAW_PDF_DIR}")
    for stem, builder in builders:
        _write_pdf(RAW_PDF_DIR / f"{stem}.pdf", builder())


if __name__ == "__main__":
    main()
