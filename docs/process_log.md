# 过程记录与失败日志

按时间顺序记录各阶段的关键失败点、现象与教训。

---

| # | 阶段 | 失败点 | 现象 | 教训 |
|---|---|---|---|---|
| 1 | 数据加载 | profile.txt 默认 header 读取 | pd.read_csv 未设置 header=None，第一行被当作列名，shape 变成 (2204,5) 而非 (2205,5) | 无表头文件必须显式 header=None，否则 pandas 静默吃掉一行 |
| 2 | 阶段一 | 随机分层划分 | 随机 StratifiedKFold 下 Accumulator F1=0.988，GroupKFold 分组留出后跌至 0.873，跌幅 11.65% | 工况配置重复出现导致标签泄漏，必须按 block_id 分组划分 |
| 3 | 阶段一 | CE/CP 作为特征 | η²(CE|cooler)=0.9983，几乎完全由标签决定 | 官方标注为 virtual 的传感器是标签泄漏源，禁止入模 |
| 4 | 阶段二 cooler | Chronos 置信带过窄 | 原始点级判决：B 类误报 68.3%、C 类 98.5%，几乎全部判异常。置信带宽度 0.066°C，健康块漂移 1.058°C，比值 0.062 | Chronos-Bolt-Small 的预测区间极度狭窄，不可直接用于点级阈值判决 |
| 5 | 阶段二 cooler | s_level/s_shape 分解挽救 | 改用残差分解 + 健康标定 P95 阈值后，B FPR=0、C FPR=0.067，A 检出率保持 1.00 | 偏离量分解（基线偏移 vs 波形畸变）比裸分位数判決更具区分力 |
| 6 | 阶段二 cooler | Chronos 未带来增益 | 平凡均值检测器与 Chronos 的 A/B/C 指标完全一致（1.00/0.00/0.067） | cooler 故障信号（~18.6°C）远超 Chronos 置信带的可区分量级，均值比较已足够 |
| 7 | 阶段二 valve | ps1_mean 检出率 0% | valve 三档 s_level 中位数仅 0.079 bar，远低于 τ=11.34 bar（被跨 cooler 漂移挟持） | ps1_mean 对 valve 无响应，其波动完全来自 cooler 档位变化 |
| 8 | 阶段二 valve | 预检 32 通道仅 ps2_std 通过 | 32 个候选通道中仅 1 个同时满足 sep≥3 且 sep≥2×drift | valve 故障在绝大多数通道上被 cooler/pump/accum 的生理波动覆盖 |
| 9 | 阶段二 valve | Chronos 残差框架下 ps2_std 检出率仍为 0% | 预检通过的通道，在实际 Chronos 逐点残差中无法与健康分离 | 通道可分（周期均值层面）≠ Chronos 残差可分（逐点预测层面） |
| 10 | 阶段二 valve | ΔP 最近配对阈值被 pump 挟持 | 最近健康块配对 100% 变更 pump 档位，d_drift 呈 bimodal 分布（0.17 vs 1.03 bar），τ=1.09 bar 超过所有故障信号 | 对照组的标签匹配决定了阈值上界——不控制其余标签的配对方式产生无意义的阈值 |
| 11 | 阶段二 valve | ΔP 匹配配对仍被 accumulator 挟持 | 固定 cooler+pump 后 d_drift 仍呈 bimodal（0.05 vs 0.89 bar），acc=130↔{100,115} 产生 0.86–0.90 bar 偏移 | dp12 不仅响应 valve，也强烈响应 accumulator 降解——这是物理通道的固有限制 |
| 12 | 阶段二 valve | ΔP 系统性对照仍 bimodal | 排除 acc=90 后，acc=130↔{100,115} 仍产生第二峰 | accumulator 最优态（130 bar）与降级态之间的 dp12 差异是 valve 故障信号的等量级 |
| 13 | 阶段二 valve | 检出率始终为零 | 五轮实验全部检出率 0% | valve 单通道检测因故障叠加而结构性不可行 |
| 14 | 阶段三 | npz 惰性句柄不可 pickle | `st.cache_data` 无法序列化 `np.load()` 返回的 NpzFile 对象，Streamlit Cloud 报 `UnserializableReturnValueError` | 缓存函数必须在 `with np.load() as z:` 的上下文内将数据读入 `{k: z[k]}` 普通 dict |
| 15 | 阶段三 | 构建张量时 np.unique(tuples) 返回 2D | `np.unique(list_of_tuples, return_inverse=True)` 将 4 元组解释为 (N,4) 数组，inverse 也是 2D | 多列组合 ID 应编码为单 int（如加权编码），避免 np.unique 的 2D 歧义 |
| 16 | 阶段四 | Chroma 距离 vs 相似度方向混淆 | 早期代码未区分距离和相似度，阈值方向错误 | `similarity_search_with_score` 返回余弦距离（越小越相似），自证 self-score=0 |
| 17 | 阶段四 | 拒答阈值留出验证未达标 | 6 个留出负样本中 2 个穿过 τ=0.62："冷却水塔风机"→cooler-02 (0.54)、"电磁阀线圈"→valve-03 (0.57) | 语义邻近导致的向量近似是 embedding 检索的固有局限，增大负样本集合建立多级阈值是改进方向 |
| 18 | 阶段五 | 知识库 cooler-01 数字错配 | cooler-01 描述 cooler=20 场景却引用了 18.6°C（cooler=3 的数字），来源标注为"本项目阶段一实测" | 知识库的数字必须有审计脚本（`test_knowledge.py`）做自动化回归，人工审阅不可靠 |
| 19 | 阶段五 | valve 三档 ΔP 四舍五入 | KB 写 0.20/0.48/0.83 bar，实测为 0.197/0.479/0.826 bar | 精确值应保留三位小数，四舍五入引入 1–2% 偏差 |
| 20 | 阶段五 | Chroma metadata filter $and 语法 | 初始使用 `{"component": comp, "severity": sev}` 被 Chroma 解释为 OR 而非 AND | Chroma 的 `$and` 操作符需要 `[{"key": "val"}, {"key2": "val2"}]` 格式 |
| 21 | 阶段五 | 严重度 map_severity 只接受英文名 | 组件名用中文"冷却器"传入时 `component == "cooler"` 为 False，返回 None | 函数需同时支持中英文名称 |
| 22 | 阶段五 | 引用校验器将 Watcher 实测值误标为幻觉 | LLM 输出中的 "20.1°C" 被标为 suspicious，但该值来自 Watcher 确定性层，是合法依据 | `verify_citation` 需接受 `extra_grounding` 参数，将 Watcher 实测值纳入合法参考 |
| 23 | 阶段五 | 断句器未跳过标题行 | "建议处置措施："被当作句子检查 bigram 匹配 | 以冒号结尾的行和长度 < 8 的行应在断句后跳过 |
| 24 | 阶段五 | LangGraph 正常路径状态残留 | alert_level="normal" 时 status 仍为 "running"，且空 .md 被输出 | 正常路径在 watcher→END 处应置 status="completed"，且跳过 .md 写入 |
| 25 | 阶段五 | json.dumps 输出 Infinity | `retrieval_score` 未命中时写入 `float("inf")`，JSON 标准不允许 Infinity | 未命中时应写 `None`，所有 `json.dumps` 加 `allow_nan=False` |
| 26 | 阶段五 | API key 占位符 | .env 初始为 `<your-api-key-here>`，LLM 调用返回 401 | 提示用户填入真实 key，不 fallback 到假响应 |
| 27 | 知识库 | cooler-03 的 18.6°C 在非冷却器语境下无实验依据 | "排除了冷却器本体故障后温度仍然偏高"却引用了 cooler=3 的 18.6°C 温升值 | 数字的语境归因必须与实测时的实验条件一致；非实验条件下的数字应标注为自编或删除 |
| 28 | 知识库 | system-01 的 10°C 阈值无实验依据 | 10°C 是人工设定的排查经验阈值，被标注为实测数字 | 经验阈值与实测数字必须严格区分——前者来源写"自编"，后者来源写"项目阶段X实测" |
| 29 | 部署 | GitHub 推送 HTTPS 认证失败 | `Recv failure: Connection was reset`，无 PAT/Credential Manager 配置 | 需用户手动配置 `gh auth login` 或 PAT |
| 30 | 部署 | 首次提交时 git 历史已是 init 后的 4535d3b 合并提交 | 项目根目录的 .git 记录显示之前已有一个 init commit | 首次提交需要处理已有的历史，不能直接 `git init` |

---

## 阶段一关键数字备忘

- profile.txt: (2205, 5), stable 行 1449, unstable 行 756
- block_id 1:1 combo_id (144 blocks)
- cooler 三段 stable_idx 长度: 480/480/489
- 四组件 GroupKFold Macro-F1 跌幅: Cooler 0%, Valve 4.5%, Pump 1.5%, Accumulator 11.7%

## 阶段二关键数字备忘

- cooler=3 vs 100 TS2 偏移: 18.6°C
- cooler=20 vs 100 TS2 偏移: 8.0°C
- valve=73/80/90 ΔP 偏移: 0.826/0.479/0.197 bar
- Chronos 置信带宽度: 0.066°C
- 健康块 ts2 漂移极差: 1.058°C
- τ_retrieval: 0.62 (距离)

## 审计记录

- 知识库数字审计: `tests/test_knowledge.py`（16 条断言，全部通过）
- 引用校验器单元测试: `tests/test_verifier.py`（5 条断言，全部通过）
- 留出验证: 4/6 通过，2 例漏网（冷却水塔风机 0.538, 电磁阀线圈 0.567）
