# 液压系统冷却器故障检测 · 置信带偏离检出

基于 Chronos-Bolt 时序基础模型对健康工况波形的零样本延拓，
结合确定性物理规则（s_level / s_shape 分解）实现冷却器退化故障检测。

## 在线演示

Demo 链接：（部署后填写）

## 快速启动

```bash
pip install -r requirements.txt
streamlit run app.py
```

无需下载原始数据，预计算数据包已包含在仓库中。

## 数据说明

原始数据来自 UCI Machine Learning Repository 的
[Condition monitoring of hydraulic systems](https://archive.ics.uci.edu/dataset/447) 数据集，
包含 2205 个液压工作循环的多传感器时序记录。

因原始数据文件体积超过 GitHub 单文件 100 MB 限制，`dataset/` 目录不入仓库。
Demo 使用预计算数据包 `app_data/demo_windows.npz`，
包含 Chronos-Bolt 对全部 1680 个窗口的预测结果与判决标签，
无需下载原始数据即可运行交互界面。

## 目录结构

```
.
├── app.py                  Streamlit 交互 Demo 入口
├── app_data/               预计算数据包（Chronos 预测结果）
├── data/                   预处理后的秒级对齐张量
├── dataset/                原始 UCI 数据（不入仓库）
├── hydraulic_cleaning/     探索性分析 Notebook
├── models/                 模型权重存储目录
├── reports/                实验报告与图表
├── src/                    源代码模块
├── CLAUDE.md               项目施工手册（阶段一 ~ 阶段三）
├── requirements.txt        运行 Demo 的依赖
└── .gitignore
```
