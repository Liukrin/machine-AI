# S3 纯向量检索基线评测

## 1. 总览

- 题目数：36  top_k=10
- Recall@1 / @3 / @5 / @10：0.583 / 0.861 / 0.861 / 0.944
- MRR@10：0.720
- 平均命中位次（仅命中题）：1.97

## 2. 按 difficulty

| 分组 | n | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | 平均命中位次 |
|---|---|---|---|---|---|---|---|
| easy | 14 | 0.714 | 1.000 | 1.000 | 1.000 | 0.833 | 1.43 |
| hard | 5 | 0.200 | 0.400 | 0.400 | 0.600 | 0.333 | 3.00 |
| medium | 17 | 0.588 | 0.882 | 0.882 | 1.000 | 0.740 | 2.24 |

## 3. 按 chunk_type

| 分组 | n | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | 平均命中位次 |
|---|---|---|---|---|---|---|---|
| table | 7 | 0.143 | 0.571 | 0.571 | 0.714 | 0.357 | 2.80 |
| text | 29 | 0.690 | 0.931 | 0.931 | 1.000 | 0.807 | 1.83 |

## 4. 按 doc_id

| 分组 | n | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@10 | 平均命中位次 |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 0.333 | 0.667 | 0.667 | 1.000 | 0.556 | 3.00 |
| 2 | 5 | 0.600 | 1.000 | 1.000 | 1.000 | 0.767 | 1.60 |
| 3 | 2 | 0.000 | 0.500 | 0.500 | 1.000 | 0.217 | 6.50 |
| 4 | 26 | 0.654 | 0.885 | 0.885 | 0.923 | 0.768 | 1.54 |

## 5. 失败清单（Recall@10 未命中，共 2 条）

### qa_038  泵轴对应的部件编号是多少？
- difficulty=hard  chunk_type=table  doc_id=4
- gold=4_c0310  gold_heading=安装、运行与维护手册 Model 3700, API Type OH2 / ISO 13709 1st and 2nd Ed. / API 610 8/9/10/11th Ed. > 注意：
- 实际 top-3：
  - `4_c0048`  dist=0.2759  heading=…安装、运行与维护手册 Model 3700, API Type OH2 
  - `4_c0053`  dist=0.2877  heading=…安装、运行与维护手册 Model 3700, API Type OH2 
  - `4_c0055`  dist=0.2887  heading=…安装、运行与维护手册 Model 3700, API Type OH2 

### qa_039  拆卸密封腔盖时需先拆哪些部件？
- difficulty=hard  chunk_type=table  doc_id=4
- gold=4_c0209  gold_heading=安装、运行与维护手册 Model 3700, API Type OH2 / ISO 13709 1st and 2nd Ed. / API 610 8/9/10/11th Ed. > 6.4.9 拆卸密封腔盖
- 实际 top-3：
  - `4_c0208`  dist=0.2699  heading=…安装、运行与维护手册 Model 3700, API Type OH2 
  - `4_c0210`  dist=0.2806  heading=…安装、运行与维护手册 Model 3700, API Type OH2 
  - `4_c0211`  dist=0.3033  heading=…安装、运行与维护手册 Model 3700, API Type OH2 
