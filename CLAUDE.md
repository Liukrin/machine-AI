# 项目规则 · 复杂流体动力装备预测性维护平台 v4

## 坐标系（强制）
- raw_idx: 0–2204，原始文件行顺序，共 2205 行
- stable_idx: 0–1448，过滤 flag==1 并 reset_index 后，共 1449 行
- 任何涉及索引区间的输出必须显式标注属于哪一套坐标系

## 数据读取约定
- 数据根目录为 dataset/，不要另建 data/raw/
- 所有 .txt 传感器文件与 profile.txt 均无表头
- 统一读法：pd.read_csv(path, sep='\t', header=None)
- profile.txt 五列依次为：cooler, valve, pump_leakage, accumulator, stable_flag
- stable_flag：0 = 工况稳定（保留，1449 行）；1 = 未达稳态（过滤，756 行）

## 禁止清单
1. 禁止使用 CE、CP 作为特征或评估目标（官方标注 virtual，η²(CE|cooler)=0.9983，标签泄漏）
2. 禁止随机划分或随机分层抽样做训练/测试切分，必须按 block_id 分组划分
3. 禁止出现 RUL、剩余寿命、退化曲线、检测延迟相关的任何建模或指标
4. 禁止 SMOTE 或任何过采样（标签近似均匀，无此需要）
5. 禁止指标不理想时反复调参补救；如实记录后继续往下走
6. 禁止 fallback 到 dummy 模型；禁止运行时联网下载，模型加载一律 local_files_only=True
7. 禁止由 LLM 做物理阈值判决，安全红线一律由确定性 Python 执行
8. 禁止报告未标注坐标系的索引区间
9. 禁止用 pandas 默认 header 读取无表头文件，一律显式 header=None

## 工作方式
- 一次只做一个任务，跑完停下，不要自行推进到下一步
- 未被要求时不安装包、不下载模型、不修改 dataset/ 下任何文件
- 所有文字结论必须与本次运行打印出的数字逐条对账，不得凭印象总结
