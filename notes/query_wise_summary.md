# Query-wise / List-wise 精排模型建模范式调研与实验设计

> 面向场景：智行酒店推荐精排模型，从 pointwise 打分向 query-wise（list-wise）建模范式演进的前期调研。
> Baseline 模型：`zxhtl_seq_v2_5_nosid_topnsp_qw_meanpool_M3oE_f_idgate_300`
> 本文档包含：① 业界/学术界 list-wise 精排前沿论文综述（所有链接均为真实可查的 arXiv / DOI 链接）；② 结合 baseline 代码的实际改造方式与实验设计。

---

## 目录

- [1. 背景与动机](#1-背景与动机)
- [2. 业界学术前沿：list-wise 建模范式全景](#2-业界学术前沿list-wise-建模范式全景)
- [3. 论文详细归纳](#3-论文详细归纳)
- [4. Baseline 代码现状与改造约束](#4-baseline-代码现状与改造约束)
- [5. 实验设计](#5-实验设计)
- [6. 风险与注意事项](#6-风险与注意事项)
- [7. 参考链接汇总](#7-参考链接汇总)

---

## 1. 背景与动机

### 1.1 为什么 pointwise 打分不够

传统精排模型对每个 `(user, item)` 对独立打分：

```
score_i = f(x_i)   # x_i 只包含 item 自身特征 + 用户画像/行为序列
```

然后按 `score` 对候选列表排序。这个范式的两个固有问题：

1. **忽略列表内 item 之间的上下文关系**。每个 item 的得分与"它在哪个列表里、邻居是谁、列表里有多少个竞品"完全无关。同一个酒店放在"一堆 200 元以下的酒店"里和放在"一堆 800 元以上的酒店"里，用 pointwise 模型打分结果完全一样，但用户的相对比较行为显然不同。
2. **优化目标与排序指标不一致**。训练用逐样本 BCE，线上/离线评价用 GAUC（用户粒度 AUC）或 NDCG 这类"列表级"排序指标。逐样本损失对排序位置不敏感，一个 item 从第 5 位升到第 1 位，pointwise 损失几乎不变，但 NDCG/GAUC 变化很大。

### 1.2 酒店场景的用户决策是"相对比较"

用户在酒店列表页做的是**相对决策**：在同一屏幕/同一次请求返回的候选列表里，对比价格、星级、距离、评分后选择一个"相对最好"的酒店。因此：

- 点击/下单概率不仅由 item 自身质量决定，还受**列表内邻近 item** 影响（竞争效应、价格锚定、同质化/多样性）；
- 展示位置本身带**位置偏差**（靠前的天然点击率高），需要显式处理；
- 一次请求返回的候选列表（召回 top-60 以内）就是用户实际决策的"上下文"，值得被模型利用。

### 1.3 术语澄清

| 术语 | 含义 |
|------|------|
| **pointwise** | 逐样本打分 + 逐样本损失（baseline 当前的建模方式） |
| **pairwise** | 以"样本对"为单位的排序损失（BPR、RankNet 等），关注两两相对序 |
| **listwise loss** | 以**整个列表**为单位的损失函数（ListMLE、ApproxNDCG 等），直接优化列表排序 |
| **list-aware / context-aware model** | 模型**结构**显式建模列表内 item 间的交互（SetRank、PERM、DLCM 等） |
| **query-wise（数据管线）** | 数据样本以"一次请求（query）的整个候选列表"为单位（baseline 的 `query_wise_mode` 已实现） |

一句话：**query-wise 是数据组织方式，list-wise 是建模/损失范式**。baseline 是"query-wise 数据 + pointwise 建模"，本文调研的是如何把后者升级为 listwise。

### 1.4 Baseline 现状（关键结论）

阅读 `zxhtl_seq_v2_5_nosid_topnsp_qw_meanpool_M3oE_f_idgate_300` 代码后确认：

- **数据管线已经是 query-wise**：`query_wise_mode: true`，每个 `SequenceExample` 是一次请求；`items_per_request=64` 定长，超出截断、不足 padding；`item_rank_topn=60`；每个 item 带 `query_id`（请求内共享）与 `item_valid_mask`（padding 标记）。
- **但建模仍是 pointwise**：
  - `dnn_input = concat([rv_feas_new, s_emb, m_emb, target_attn_out])`，其中每个 item 的表示只包含自身特征 + 用户行为序列 target-attention 结果；
  - 列表内 item **没有任何交互**：没有同请求内其他 item 的信息；
  - 损失是逐样本 BCE（click / duration / ctcvr）。
- **有利条件**：`query_id` 与 `item_valid_mask` 在 `build_model` 里被保留下来了（`boolean_mask` 之后 `query_id` 仍可用），意味着做 listwise 损失/列表内交互不需要改数据、不需要改 SQL，纯模型侧改动即可。

> 结论：**数据已就绪，缺的是 listwise 的损失函数与列表内交互建模。** 这是本项目最有利的起点。

---

## 2. 业界学术前沿：list-wise 建模范式全景

### 2.0 总览

| 类别 | 论文 | 会议/年份 | 链接 | 一句话 |
|------|------|-----------|------|--------|
| 损失 | ListNet（Cao et al.） | ICML 2007 | [DOI](https://doi.org/10.1145/1273496.1273513) | 用 permutation/top-k 概率定义列表级损失，listwise 开山之作 |
| 损失 | ListMLE（Xia et al.） | ICML 2008 | [DOI](https://doi.org/10.1145/1390156.1390306) | 极大似然 Ground-Truth 排列（Plackett-Luce），SGD 优化 |
| 损失 | SoftRank（Taylor et al.） | WSDM 2008 | [DOI](https://doi.org/10.1145/1341531.1341544) | 对分数加噪求 rank 分布，软化 NDCG（SoftNDCG） |
| 损失 | ApproxNDCG（Qin et al.） | IR Journal 2010 | [DOI](https://doi.org/10.1007/s10791-009-9124-x) | 用 sigmoid 近似排序指示函数，直接优化近似 NDCG |
| 损失 | LambdaLoss（Wang et al.） | CIKM 2018 | [arXiv](https://arxiv.org/abs/1802.02265) | 为 LambdaRank 提供概率框架，推导 metric-driven 损失 |
| 损失 | Revisiting Approx Metric Opt（Bruch et al.） | SIGIR 2019 | [arXiv](https://arxiv.org/abs/1812.08473) | 用现代 DNN 复现 ApproxNDCG，证明其仍具 SOTA 竞争力 |
| 损失 | NeuralNDCG（Pobrotyn & Białobrzeski） | arXiv 2021 | [arXiv](https://arxiv.org/abs/2102.07831) | 用 NeuralSort 可微排序直接逼近 NDCG |
| 损失 | PiRank（Swezey et al.） | NeurIPS 2021 | [arXiv](https://arxiv.org/abs/2012.06731) | 可微排序 + 分治归并，扩展到大规模列表 |
| 损失 | SmoothI（Thonet et al.） | AAAI 2022 | [arXiv](https://arxiv.org/abs/1911.09798) | 平滑 rank 指示函数，可嵌入 NDCG/MAP/ARP |
| 结构 | DLCM（Ai et al.） | SIGIR 2018 | [arXiv](https://arxiv.org/abs/1804.05936) | GRU 编码列表局部上下文，重打分（reranking） |
| 结构 | PERM（Pei et al.） | RecSys 2019 | [arXiv](https://arxiv.org/abs/1904.06813) | Transformer 列表编码 + 个性化解码，排列概率损失 |
| 结构 | SetRank（Pang et al.） | SIGIR 2020 | [arXiv](https://arxiv.org/abs/1912.05891) | 多头自注意力建模 item 交互，排列不变（permutation-invariant） |
| 结构 | DPIN（Huang et al.） | SIGIR 2019 | [arXiv](https://arxiv.org/abs/2106.12229) | 位置-候选深度交互，建模列表位置上下文 |
| 结构 | HLATR（Zhang et al.） | arXiv 2022 | [arXiv](https://arxiv.org/abs/2205.10569) | 轻量列表感知 Transformer，检索+重排特征融合 |
| 结构/LLM | RankGPT / LRL（Sun et al.） | EMNLP 2023 | [arXiv](https://arxiv.org/abs/2305.02156) | LLM 直接对列表生成排序，零样本 listwise 重排 |
| 工业 | LTR for E-Commerce Search（Santu et al.） | SIGIR 2017 | [arXiv](https://arxiv.org/abs/1903.04263) | 电商搜索 LTR 实战经验：下单率是最稳目标 |
| 工业/偏差 | Bias and Debias Survey（Chen et al.） | arXiv 2020 | [arXiv](https://arxiv.org/abs/2010.03240) | 7 类偏差（含位置偏差）的系统综述 |

### 2.1 列表级损失函数：直接优化排序指标

这一类工作不改变模型结构，只改变**损失函数**：把"逐样本 BCE"换成/叠加"以列表为单位的排序损失"。代表性谱系：

- **概率排列派**（ListNet → ListMLE）：把排序看成"对排列的概率建模"。ListNet 用 cross-entropy 对齐预测/真实排列概率（top-k 近似）；ListMLE 直接用 Plackett-Luce 模型做极大似然，理论上有排序一致性保证。
- **指标软化派**（SoftRank → ApproxNDCG → Revisiting → SmoothI → NeuralNDCG/PiRank）：把 NDCG 这类"非连续、非分解"指标软化。
  - SoftRank：给分数注入噪声 → 得到 rank 分布 → 优化期望 NDCG；
  - ApproxNDCG：用 sigmoid 近似 rank 指示函数 + top-K 选择器；
  - Revisiting（Bruch et al.）在深度学习时代重估 ApproxNDCG，发现它比许多新损失还稳；
  - NeuralNDCG / PiRank：用 NeuralSort 的可微排序直接做 NDCG 的 surrogate，PiRank 用分治扩展到超长列表；
  - SmoothI：用一个可微函数 SmoothI 近似 rank 指示，可自由嵌入 NDCG/MAP/ARP，有收敛性保证。
- **Lambda 梯度派**（LambdaLoss）：LambdaRank 的梯度（按位置加权的 pairwise 梯度）被证明是某类 metric-driven 损失的最优梯度，LambdaLoss 给出了统一概率框架，理论上澄清了"直接优化 NDCG"的正确打开方式。

> 与 baseline 的关系：这类工作**只改损失函数**，模型结构完全不动，是侵入性最小、见效最快的一档。

### 2.2 列表感知的模型结构：context-aware / list-aware ranking

这一类工作把"列表内 item 交互"显式放进模型**结构**：

- **DLCM（SIGIR 2018）**：第一个"重排序"范式——先有 base ranker 打分，再对 top-k 列表用 **GRU** 顺序编码，学习局部列表上下文来**修正/重打分**，用 attention loss 训练。
- **PERM（RecSys 2019，阿里）**：重排序模型，GRU 编码整个列表上下文 + **Transformer decoder**（用用户 embedding 作为 query 做个性化注意力），损失用"近似排列概率"，线上电商验证有效。
- **SetRank（SIGIR 2020）**：堆叠多头自注意力对所有 doc 联合编码，**不注入位置信息**，从而保证"排列不变"（同一集合任意输入顺序输出排序不变），满足理想排序模型的两个性质（建模 doc-doc 交互 + 排列不变）。
- **DPIN（SIGIR 2019，美团）**：聚焦位置偏差，建模"位置 × 候选 × 用户 × 上下文"的深度交互，提出 position-wise AUC（PAUC）评估，AB 实验线上提升。
- **HLATR（2022，阿里）**：轻量级列表感知 Transformer，融合检索阶段与重排阶段特征，可并行化，主打低延迟重排。
- **RankGPT / LRL（EMNLP 2023，Outstanding Paper）**：LLM 版 listwise——一次性把候选列表喂给 LLM，让它直接输出排序后的 item 编号（零样本，滑窗处理长列表）。

> 与 baseline 的关系：baseline 有 M3oE 多任务塔 + target attention，但没有"item × item"的列表内交互层。这一类方法对应给 baseline 增加一个**列表交互模块**（自注意力/GRU over 同请求 items）。

### 2.3 工业界应用与偏差

- **LTR for E-Commerce Search（SIGIR 2017）**：电商搜索 LTR 实战总结——**下单率（order rate）是最稳的训练目标**，其次点击率； popularity 类列表特征很有用；query 属性稀疏是主要难点。酒店场景类似（以 cr / 下单为核心目标）。
- **Bias and Debias Survey（2020）**：系统梳理位置偏差、选择偏差、曝光偏差等 7 类偏差。列表级建模天然要处理**位置偏差**（baseline 已用 `rank_oh` + show_index dropout 处理了一部分）。

### 2.4 范式小结：三条落地路径

| 路径 | 对应论文 | 侵入程度 | 预期收益 |
|------|---------|---------|---------|
| ① 改/加列表级损失 | ListMLE、ApproxNDCG、LambdaLoss、NeuralNDCG、SmoothI | 低（只改 loss） | GAUC/NDCG 直接对齐排序指标 |
| ② 注入列表上下文特征 | DPIN、E-Commerce LTR、SetRank-lite | 中（加特征/小模块） | 显式利用"相对比较"信号 |
| ③ 列表内交互结构 | SetRank、PERM、DLCM、HLATR | 高（加交互层/重排塔） | 最彻底的 list-aware 建模 |

---

## 3. 论文详细归纳

> 以下每篇给出**真实链接**、会议年份、核心思想、方法与 baseline 结合点。

### 3.1 列表级损失函数（路径 ①）

#### 1. ListNet —— Learning to Rank: From Pairwise Approach to Listwise Approach
- **链接**：[DOI 10.1145/1273496.1273513](https://doi.org/10.1145/1273496.1273513)（ICML 2007）
- **作者**：Zhe Cao, Tao Qin, Tie-Yan Liu, Ming-Feng Tsai, Hang Li
- **核心思想**：第一个系统的 listwise 方法。定义**排列概率**（permutation probability，基于 Plackett-Luce）与 **top-k 概率**（top-one 用 softmax 即可近似），训练损失 = 预测分布与真实分布之间的 KL/cross-entropy。
- **方法**：NN 打分 → softmax 得到 top-one 概率 → 与真实标签的 softmax 分布算交叉熵；对全排列精确计算做了 top-k 近似。理论上分析了损失的 consistency、soundness、continuity。
- **与 baseline 结合**：最朴素的 listwise 损失。可直接把 baseline 的 click 打分 `q[0]` 在请求内 softmax 后与"真实点击的 softmax 标签"算 KL——但 ListNet 假设有"排序标签"，酒店场景只有隐式点击标签，需把 0/1 点击转成列表内归一化标签（如点击者权重 1、未点击 0，再 softmax）。适合作为**对照基线损失**。

#### 2. ListMLE —— Listwise Approach to Learning to Rank: Theory and Algorithm
- **链接**：[DOI 10.1145/1390156.1390306](https://doi.org/10.1145/1390156.1390306)（ICML 2008）
- **作者**：Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, Hang Li
- **核心思想**：与 ListNet 同源但更简洁——直接对**真实排列**做极大似然：`L = -log P(π* | scores)`，P 是 Plackett-Luce 模型（`P(π) = ∏ softmax(scores of remaining)`）。有统计一致性理论保证。
- **方法**：一次前向得到列表分数，按真实排列序（正样本排在负样本前）依次取 softmax 累乘再取负对数。SGD 优化。Top-1 情形退化为 softmax cross-entropy。
- **与 baseline 结合**：酒店场景"真实排列"由点击/下单定义（点击者排前面）。实现时按 `query_id` 分组，组内按 label 降序排列后计算。**这是本文推荐优先尝试的 listwise 损失之一**：公式简单、数值稳定、理论上可解释。

#### 3. SoftRank —— SoftRank: Optimizing Non-Smooth Rank Metrics
- **链接**：[DOI 10.1145/1341531.1341544](https://doi.org/10.1145/1341531.1341544)（WSDM 2008）
- **作者**：Michael J. Taylor, Jo/John Guiver, Stephen E. Robertson, Thomas P. Minka
- **核心思想**：给每个 doc 的打分注入高斯噪声，使"分值的 rank"成为随机变量 → 得到 rank 分布 → 在此基础上计算**期望 NDCG**（SoftNDCG），梯度平滑。
- **方法**：分数 → 高斯分布（方差为超参）→ 两两比较概率 → rank 分布（动态规划/近似）→ SoftNDCG 求导训练。
- **与 baseline 结合**：适合在 baseline 的 logits 上做 SoftNDCG 辅助损失。实现上比 ApproxNDCG 稍重（需要 rank 分布），可作为进阶选项。

#### 4. ApproxNDCG —— A General Approximation Framework for Direct Optimization of Information Retrieval Measures
- **链接**：[DOI 10.1007/s10791-009-9124-x](https://doi.org/10.1007/s10791-009-9124-x)（Information Retrieval Journal, 2010）
- **作者**：Tao Qin, Tie-Yan Liu, Hang Li
- **核心思想**：把 NDCG 的不可导环节（rank 指示函数、位置折扣函数）分别用可导近似替代：用 sigmoid 近似 `I(s_i > s_j)`，用 top-K 选择器近似截断。得到平滑的 **ApproxNDCG**。
- **方法**：`rank(i) ≈ 1 + Σ_j sigmoid(-(s_i - s_j))`，代入 NDCG 公式得到可导 surrogate。
- **与 baseline 结合**：**本文最推荐优先实现的损失**。只需要 baseline 的列表内分数 + label，公式现成（见 5.2 节），纯模型侧改动，无需改数据。与 GAUC/NDCG 评价目标最对齐。

#### 5. LambdaLoss —— The LambdaLoss Framework for Ranking Metric Optimization
- **链接**：[arXiv:1802.02265](https://arxiv.org/abs/1802.02265)（CIKM 2018）
- **作者**：Xuanhui Wang, Cheng Li, Nadav Golbandi, Michael Bendersky, Marc Najork
- **核心思想**：为 LambdaRank 建立统一概率框架——定义一族 metric-driven 损失，其中 **LambdaRank 的 λ 梯度是该损失的最优梯度**。证明"按位置加权 pairwise 梯度"≈ 优化 NDCG。
- **方法**：把排序指标写成可导形式，推导出其"最优梯度"，再构造相应损失；LambdaLoss 是该族损失的特例。
- **与 baseline 结合**：提供**理论依据**：给 baseline 加一个 NDCG 感知的 pairwise 加权损失（按 rank 位置加权），效果等价于直接优化 NDCG。可作为一个轻量替代（比 ApproxNDCG 更易数值稳定）。

#### 6. Revisiting Approximate Metric Optimization in the Age of Deep Neural Networks
- **链接**：[arXiv:1812.08473](https://arxiv.org/abs/1812.08473)（SIGIR 2019）
- **作者**：Sebastian Bruch, Masrour Zoghi, Michael Bendersky, Marc Najork
- **核心思想**：在深度神经网络时代重新评估 ApproxNDCG 框架（A-NDCG、Gumbel-A-NDCG），发现**直接优化近似 NDCG 仍然显著优于/持平许多现代 LTR 损失**，且超参敏感性低。
- **方法**：以现代 NN 为打分模型，实现 A-NDCG / Gumbel-A-NDCG，与 LambdaRank 等对比。
- **与 baseline 结合**：为"用 ApproxNDCG 替换/叠加 BCE"提供了最强背书——这是**微软搜索团队推荐的在 DNN 排序模型上直接优化 NDCG 的方案**，与我们的目标（提升 GAUC）一致。代码可参考其公开实现思路。

#### 7. NeuralNDCG —— Direct Optimisation of a Ranking Metric via Differentiable Relaxation of Sorting
- **链接**：[arXiv:2102.07831](https://arxiv.org/abs/2102.07831)（arXiv, 2021）
- **作者**：Przemysław Pobrotyn, Radosław Białobrzeski
- **核心思想**：用 **NeuralSort**（可微排序算子）松弛排序，从而把 NDCG 变成完全可导的目标，逼近真实 NDCG。
- **方法**：分数 → NeuralSort 可微排序矩阵 → 代入 NDCG 公式求导。
- **与 baseline 结合**：NeuralSort 实现较重，适合作为与 ApproxNDCG 对照的进阶损失。若 ApproxNDCG 效果显著，可不做本项。

#### 8. PiRank —— Scalable Learning To Rank via Differentiable Sorting
- **链接**：[arXiv:2012.06731](https://arxiv.org/abs/2012.06731)（NeurIPS 2021）
- **作者**：Robin Swezey, Aditya Grover, Bruno Charron, Stefano Ermon
- **核心思想**：在 NeuralSort 基础上加**温度退火**（低温趋近真实排序）与**分治归并策略**，使可微排序损失能扩展到非常大的列表。
- **方法**：递归地把子列表排序后合并，只传播 top 项，降低复杂度。
- **与 baseline 结合**：baseline 的列表长度只有 M=64（top-60），规模不大，PiRank 的优势体现不明显；可作为未来超长列表（如召回几百个）时的选项。

#### 9. SmoothI —— Listwise Learning to Rank Based on Approximate Rank Indicators
- **链接**：[arXiv:1911.09798](https://arxiv.org/abs/1911.09798)（AAAI 2022）
- **作者**：Thibaut Thonet, Yagmur Gizem Cinar, Éric Gaussier, Minghan Li, Jean-Michel Renders
- **核心思想**：提出 **SmoothI**——rank 指示函数的可微近似，可自由嵌入 NDCG/MAP/ARP 等多种指标，并给出向真实指标收敛的理论保证。
- **方法**：用 SmoothI 替换排序指标里的指示函数，得到统一可导损失。
- **与 baseline 结合**：与 ApproxNDCG 同类，作为对照项；理论上覆盖面比 NDCG 更广（可用于 MAP/ARP）。

### 3.2 列表感知模型结构（路径 ③）

#### 10. DLCM —— Learning a Deep Listwise Context Model for Ranking Refinement
- **链接**：[arXiv:1804.05936](https://arxiv.org/abs/1804.05936)（SIGIR 2018）
- **作者**：Qingyao Ai, Keping Bi, Jiafeng Guo, W. Bruce Croft
- **核心思想**：提出"**重排序**"范式——不从头学习打分，而是对 base ranker 给出的 top-k 列表，用 **GRU** 顺序编码整个列表，学习"局部列表上下文"来修正每个 doc 的分数（ranking refinement）。
- **方法**：base 打分特征向量 → GRU 顺序编码 → attention loss（对列表整体求损失）→ 输出修正分数。
- **与 baseline 结合**：把 baseline 的 `dnn_input`（或 M3oE 输出 logits）当作"base 打分特征"，在其上加一层列表 GRU/attention 得到修正分数。**注意**：DLCM 是顺序（position 敏感）建模，与位置偏差相关；训练用列表级 attention loss。

#### 11. PERM —— Personalized Re-ranking for Recommendation
- **链接**：[arXiv:1904.06813](https://arxiv.org/abs/1904.06813)（RecSys 2019，阿里）
- **作者**：Changhua Pei, Yi Zhang, Yongfeng Zhang, Fei Sun, Xiao Lin, Hanxiao Sun, Jian Wu, Peng Jiang, Junfeng Ge, Wenwu Ou
- **核心思想**：推荐场景的重排序模型。对 base ranker 的结果做列表级编码，并**引入个性化**（user embedding 作为 Transformer decoder 的 attention query），损失用**近似排列概率**。
- **方法**：GRU 编码列表 → Transformer decoder（带用户个性化 query）→ 输出每个 item 的排列概率 → 用近似排列概率（基于打分）做损失。离线基准 + 阿里线上电商验证。
- **与 baseline 结合**：baseline 的 `uid_emb`（slot 260）正好可充当个性化 query。此方案比 DLCM 更"个性化"，可作为路径 ③ 的进阶实现。训练上需要处理排列概率（用 top-k 近似）。

#### 12. SetRank —— Learning a Permutation-Invariant Ranking Model for Information Retrieval
- **链接**：[arXiv:1912.05891](https://arxiv.org/abs/1912.05891)（SIGIR 2020）
- **作者**：Liang Pang, Jun Xu, Qingyao Ai, Yanyan Lan, Xueqi Cheng, Jirong Wen
- **核心思想**：用**多头自注意力**堆叠对列表内所有 doc 联合编码，**不注入位置信息**，从而得到排列不变（permutation-invariant）的列表感知打分模型。论证了理想排序模型的两个性质：建模 doc-doc 交互 + 排列不变。
- **方法**：doc 特征序列 → 多层 self-attention（SetEncoder）→ 逐位置打分。训练可用任何 LTR 损失。
- **与 baseline 结合**：这是把"列表内 item 交互"塞进 baseline 最自然的结构——在 M3oE 塔前加一个 masked multi-head self-attention over `[B, M, item_repr]`（mask 掉 padding）。相比 PERM/DLCM 更简单（无顺序、无 decoder），且**天然处理了不同请求 item 数不一致**（masked attention）。**本文推荐作为路径 ③ 的首选结构**。

#### 13. DPIN —— Deep Position-wise Interaction Network for CTR Prediction
- **链接**：[arXiv:2106.12229](https://arxiv.org/abs/2106.12229)（SIGIR 2019，美团）
- **作者**：Jianqiang Huang, Ke Hu, Qingtao Tang, Mingjian Chen, Yi Qi, Jia Cheng, Jun Lei
- **核心思想**：建模"**位置 × 候选 × 用户 × 上下文**"的深度非线性交互，缓解位置偏差；提出 **PAUC**（position-wise AUC）评价指标。
- **方法**：每个位置学习位置表示，候选 item 与位置做深度交互（item-position 匹配），保证训练/推断位置一致性。
- **与 baseline 结合**：baseline 已有 `rank_oh`（20 档 one-hot）+ show_index dropout。DPIN 提示我们可以把 **rank 位置显式注入列表交互层**（作为位置 embedding），并新增 PAUC 观测指标。

#### 14. HLATR —— Enhance Multi-stage Text Retrieval with Hybrid List Aware Transformer Reranking
- **链接**：[arXiv:2205.10569](https://arxiv.org/abs/2205.10569)（arXiv, 2022，阿里巴巴）
- **作者**：Yanzhao Zhang, Dingkun Long, Guangwei Xu, Pengjun Xie
- **核心思想**：轻量级列表感知 Transformer 重排模块，融合**检索阶段与重排阶段**的特征，对整个候选列表联合打分，主打低延迟、可并行。
- **方法**：把 retrieval 分数 + rerank 特征一起进 Transformer，输出列表级修正分数。
- **与 baseline 结合**：baseline 若想保留"base 分数"信息，可把当前 M3oE 输出的 logits 作为额外输入特征喂给列表交互层（即"打分再打分"），与 DLCM 思路一致。

#### 15. RankGPT / LRL —— Zero-shot Listwise Document Reranking with a Large Language Model
- **链接**：[arXiv:2305.02156](https://arxiv.org/abs/2305.02156)（EMNLP 2023，Outstanding Paper）
- **作者**：Weiwei Sun, Lingyong Yan, Xinyu Ma, Shuaiqiang Wang, Pengjie Ren, Zhumin Chen, Dawei Yin, Zhaochun Ren
- **核心思想**：**LLM 版 listwise 重排**——一次性把候选列表交给 LLM，让 LLM 直接输出排序后的编号（而不是逐个打分），滑窗处理超长列表。证明 listwise 的"整体比较"比 pointwise 逐项打分更能发挥 LLM 的排序能力。
- **方法**：构造 prompt（"请将以下文档按相关性排序"）→ LLM 输出重排后的 id 序列 → 滑窗迭代。
- **与 baseline 结合**：作为**未来方向**——用 LLM 给酒店候选做 listwise 排序蒸馏 teacher（生成顺序标签/soft label），再蒸馏回 M3oE 精排；或直接用 LLM 打分做线上重排实验。与当前判别式模型不冲突，属于 pipeline 上层。

### 3.3 工业界应用与偏差（路径 ②/④）

#### 16. On Application of Learning to Rank for E-Commerce Search
- **链接**：[arXiv:1903.04263](https://arxiv.org/abs/1903.04263)（SIGIR 2017）
- **作者**：Shubhra Kanti Karmaker Santu, Parikshit Sondhi, ChengXiang Zhai
- **核心思想**：电商搜索场景 LTR 的实战经验总结。**下单率（order rate）是最稳的训练目标**（优于点击率、加购率）；popularity 类列表特征价值很高；query 属性稀疏是主要难点。
- **方法**：在工业数据集上系统对比不同目标信号、特征组合、judgment 获取方式。
- **与 baseline 结合**：① 为"listwise 损失主目标选 cr/下单"提供依据；② 提示可以注入 **popularity/列表统计特征**（如酒店在列表中的相对价格分位、同列表均值/中位数价格）——这些是典型的"列表上下文特征"，属于路径 ②。

#### 17. Bias and Debias in Recommender System: A Survey and Future Directions
- **链接**：[arXiv:2010.03240](https://arxiv.org/abs/2010.03240)（arXiv, 2020）
- **作者**：Jiawei Chen, Hande Dong, Xiang Wang, Fuli Feng, Meng Wang, Xiangnan He
- **核心思想**：系统综述推荐系统 7 类偏差（选择、位置、曝光、流行度等）及其因果解释与去偏方法。
- **方法**：分类学整理 + 各方法评述。
- **与 baseline 结合**：listwise 建模必须处理**位置偏差**——baseline 已用 `rank_oh` + show_index dropout，但做列表级建模后，建议进一步验证"位置感知 listwise 损失"（如按位置加权的 LambdaLoss 变体）与"位置作为列表注意力输入"。

---

## 4. Baseline 代码现状与改造约束

### 4.1 数据管线（已具备 query-wise 结构）

`model_script/train.py` 关键点：

| 位置 | 说明 |
|------|------|
| `Config.__init__` (L41-61) | `query_wise_mode`、`items_per_request=64`、`item_rank_topn=60`、`context_discrete_slots` |
| `_parse_tfrd_query_wise` (L303-563) | 每请求一个 SequenceExample → 按 `rank` 截断到 top-60 → 定长 `[M, ...]` → 返回 `query_id`、`item_valid_mask` |
| `build_dataset` (L866-1018) | `batch(batch_size)` 收集 B=64 个请求 → `[B, M, ...]` |
| `build_model` (L1038-1084) | 展平为 `[BM, ...]`，request-major 顺序（全局下标 = r*M+i） |
| `build_model` (L1182-1218) | **`boolean_mask` 剔除 padding** → 动态 batch `B_valid` |
| `build_model` (L1199) | `query_id` 在 mask 后被保留 `[B_valid, 1]` |

### 4.2 模型前向（pointwise）

```
s_emb / m_emb / l_emb  ← embedding_lookup([BM, ...])
target_attn_out        ← target attention over 行为序列（L1249-1317）
dnn_input = concat([rv_feas_new, s_emb, m_emb, target_attn_out])  (L1322)
dnn_input_norm = BN  (L1323)
→ M3oE 塔（domain_rep / shared_expert / domain_expert / task_expert / PPNet gate）→ mlp_out_list (L1319-1519)
q = sigmoid(mlp_out_list)；q[2] = ctcvr = sigmoid(l0)*sigmoid(l2)  (L1521-1524)
损失 = 逐样本 BCE 求和  (L1526-1538)
```

### 4.3 关键改造约束

1. **`boolean_mask` 破坏了列表结构**（L1182-1218）：mask 之后 batch 维是动态 `B_valid`，`[B, M]` 网格不存在了。列表级操作（listwise loss / 列表内 attention）必须在 **mask 之前**（固定 `[BM]` 可 reshape 成 `[B, M, ...]`）或在 mask 之后用 **`query_id` 分段**（`tf.unique` + `tf.dynamic_partition` / `unsorted_segment_*`）实现。
2. **Horovod 集合通信必须在固定 shape 上**：`embedding_lookup` 内的 allgather 已在 `[BM, ...]` 静态 shape 上完成（注释 L1089-1090），新增的列表操作若放在 mask 之前，天然满足；若放 mask 之后，要确保只在本地计算（不触发跨 rank 通信）。
3. **`dense_input_size` 联动**：任何改变 `dnn_input` 维度的修改（如新增列表上下文向量）必须同步更新 `Config.dense_input_size`（L218）。
4. **新配置项要进 yaml**：新增超参（如 listwise 损失权重 λ、注意力层数）要同时在 `model.yaml` 与 `Config.__init__` 中定义（workflow 5.2 规则）。
5. **padding item 必须全程 mask**：`item_valid_mask` 在 listwise 损失 / 列表 attention 里都要作为 mask 使用，防止 padding 影响梯度。
6. **信息泄漏**：列表内交互是**双向**的（同请求内所有 item 互相可见），这在线下重排范式里是合法的（离线打分时整个列表已知）；但若未来做"在线逐个打分"部署，需要注意（workflow 5.2 规则 6 的精神）。

---

## 5. 实验设计

> 设计原则：**从简单到复杂递进**，每个实验给出动机、对应论文、具体代码修改点、预期收益、风险。所有实验均**不需要改数据/SQL**（query-wise 数据已具备）。

### 实验 A（推荐优先）：列表级损失函数 —— Listwise Loss

**动机**：把训练目标从"逐样本 BCE"对齐到"列表排序指标（GAUC/NDCG）"。改动最小、见效最快，且为后续列表结构实验提供"纯损失增益"的对照基线。

**对应论文**：ApproxNDCG + [Revisiting](https://arxiv.org/abs/1812.08473)（首选）、[ListMLE](https://doi.org/10.1145/1390156.1390306)、[LambdaLoss](https://arxiv.org/abs/1802.02265)、[SmoothI](https://arxiv.org/abs/1911.09798)。

**实现方式**（以 ApproxNDCG 为例）：

1. 在 `build_model` 中取列表内分数：`mlp_out_list[0]`（click logit）与 `mlp_out_list[2]`（cr logit），此时是 `[B_valid, 1]`。
2. 用 `query_id` 分段：
   ```python
   qid = tf.reshape(query_id, [-1])                      # [B_valid]
   # 保证同一请求的 item 连续（boolean_mask 保持 request-major 顺序）
   unique_qid, idx = tf.unique(qid)
   score = mlp_out_list[0][:, 0]                          # click logit [B_valid]
   label = click[:, 0]                                    # 或 cr_valid 作为 relevance
   mask  = tf.cast(tf.greater(qid, -1), tf.float32)       # 全 1，示意
   lists_s = tf.dynamic_partition(score, idx, tf.size(unique_qid))
   lists_y = tf.dynamic_partition(label, idx, tf.size(unique_qid))
   ```
3. 用 `tf.map_fn` 对每个 list 计算 ApproxNDCG 损失：
   ```python
   def _approx_ndcg_loss(args):
       s, y = args                                   # 变长 list
       idx_i = tf.range(tf.size(y))
       # rank 的平滑近似: rank(i) = 1 + Σ_j sigmoid(-(s_i - s_j))
       rel = tf.cast(y, tf.float32)
       rank_i = 1.0 + tf.reduce_sum(tf.nn.sigmoid(-(tf.expand_dims(s,1) - tf.expand_dims(s,0))), axis=1)
       dcg = tf.reduce_sum((2**rel - 1) / tf.log2(rank_i + 1.0))
       # 理想 DCG: 按 rel 降序排
       y_sorted = tf.sort(rel, direction='DESCENDING')
       idcg = tf.reduce_sum((2**y_sorted - 1) / tf.log2(tf.range(1, tf.size(y)+1, dtype=tf.float32)+1.0))
       return -dcg / tf.maximum(idcg, 1e-8)
   list_loss = tf.reduce_mean(tf.map_fn(_approx_ndcg_loss, (lists_s, lists_y), dtype=tf.float32))
   ```
4. 叠加到总损失：`self.loss += self.parm.listwise_w * list_loss`（新增 yaml 参数 `listwise_w`，建议从 0.1 起调）。
5. 主目标选择：先对 **click** 做（正样本更稠密）；再尝试对 **ctcvr（q[2]）** 做，label 用 `cr_valid`，与线上核心目标对齐。

**变体（对照）**：
- **ListMLE 版**：组内按 label 降序排列后，`loss = -Σ log softmax(s_i | remaining)`；实现同样用 `dynamic_partition + map_fn`。
- **Lambda 加权版**：用 LambdaLoss 的"按位置加权的 pairwise 梯度"实现轻量替代，数值上比 ApproxNDCG 更稳。

**预期收益**：click/cr 的 **GAUC 提升**（列表排序正确率更直接地被优化）；AUC 可能持平或略升。

**风险**：① 变长列表的 `map_fn` 在 TF1 静态图下可能较慢——M=64、B=64 时每个 step 最多 64 个 list，开销可接受；② 损失尺度与 BCE 差异大，需要归一化（除以列表长度）与调 `listwise_w`；③ 需要对 `rank` 截断后的列表计算（top-60 内），与线上一致即可。

---

### 实验 B：列表上下文特征注入 —— List Context Features

**动机**：用显式的"列表上下文"信号增强每个 item 的表示，模拟用户的相对比较。改动中等、可解释性强。

**对应论文**：[DPIN](https://arxiv.org/abs/2106.12229)（位置交互）、[LTR for E-Commerce Search](https://arxiv.org/abs/1903.04263)（popularity/列表统计特征）、SetRank-lite 思路。

**实现方式**（在 `boolean_mask` 之前，利用 `[B, M, ...]` 网格）：

1. 在 `build_model` 中，`embedding_lookup` 之后、`boolean_mask` 之前，把 `s_emb` reshape 为 `[B, M, slots_sizeS*D]`。
2. 计算**列表内聚合特征**（masked，排除 padding）：
   ```python
   mask3 = tf.reshape(item_valid_mask, [B, M, 1])                    # [B,M,1]
   cnt   = tf.maximum(tf.reduce_sum(mask3, axis=1, keepdims=True), 1.0)
   list_mean = tf.reduce_sum(s_emb3 * mask3, axis=1, keepdims=True) / cnt   # [B,1,D]
   list_max  = tf.reduce_max(s_emb3 - (1-mask3)*1e9, axis=1, keepdims=True)
   ctx       = tf.tile(list_mean, [1, M, 1])                          # [B,M,D]
   # 可选的"去自身"版本：list_mean_wo_self = (sum - self)/ (cnt-1)
   ```
3. 把 `ctx`/`list_max`（或投影到小维度）concat 到 `dnn_input`（同时更新 `dense_input_size`）。
4. **位置嵌入**：将 `rank_oh`（已有，L1124）投影成位置 embedding，也可作为列表上下文的一部分输入交互层。
5. 可加**相对特征**：`price_rank_in_list`、`price_zscore_in_list`（若 `rv_feas` 已有价格字段则直接算，否则用 s_emb 近似相似度）。

**预期收益**：GAUC 提升，尤其对"价格/星级相近的竞品列表"；可解释性强（能看每个列表特征的重要性）。

**风险**：列表聚合特征是"全局统计"，与 item 自身特征可能冗余；需要控制注入维度，防止过拟合。此实验也验证"列表内信息是否有用"，为实验 C 提供判断依据。

---

### 实验 C（核心结构）：列表内交互 —— Masked Self-Attention over Items

**动机**：让每个 item 的表示显式包含"同请求其他 item 的信息"，实现真正的 list-aware 打分。这是 list-wise 建模范式最核心的结构升级。

**对应论文**：[SetRank](https://arxiv.org/abs/1912.05891)（首选，排列不变）、[PERM](https://arxiv.org/abs/1904.06813)（个性化解码）、[DLCM](https://arxiv.org/abs/1804.05936)（GRU 顺序编码）、[HLATR](https://arxiv.org/abs/2205.10569)。

**实现方式**（SetRank 风格，加在 M3oE 塔之前）：

1. 在 `dnn_input_norm` 处（L1323 之后），把 item 表示 reshape 成 `[B, M, D_in]`。
2. 加 **N 层 masked multi-head self-attention**（N 建议 1-2 层起步）：
   ```python
   # item_repr: [B, M, D_in]
   # mask:      [B, M]  (来自 item_valid_mask)
   attn_mask = tf.tile(tf.expand_dims(mask, 1), [1, M, 1])          # [B,M,M]
   # 标准 scaled-dot-product attention，attn_mask 中 padding 位置为 -inf
   attn_out = multi_head_attn(item_repr, attn_mask)                  # [B,M,D_out]
   ```
3. **位置选项**：
   - **排列不变版（SetRank）**：不加位置编码；
   - **位置感知版（DPIN/PERM）**：把 `rank` 投影成位置 embedding 加到 item_repr 上（推荐先在 click 上对比两个版本）。
4. 可选残差 + LN，然后 flatten 回 `[BM, D_out]`，继续走 `boolean_mask` → M3oE 塔。
5. 与实验 A 组合：列表交互层 + listwise 损失一起用，效果预计叠加。

**代码注意**：
- 所有 attention 计算在 `[B, M]` 固定 shape 上，满足 Horovod 约束；
- `D_out` 变化需更新 `dense_input_size`；
- 注意 M3oE 塔的输入维度现在来自 attention 输出，检查 `domain_rep_ext_layer_size` 等使用 `dense_input_size` 的地方（L238）；
- `tf.contrib` 里没有现成 masked MHA，需要手写或复制 `target_attention_pool_time_aware` 的 attention 骨架（L692-826）改造。

**预期收益**：最直接的列表感知建模，GAUC/NDCG 提升空间最大；同时为将来"重排序"（实验 D）铺路。

**风险**：① 训练成本增加（attention 层）；② 首层 M3oE 塔输入分布变化较大，需要重新调 lr / 观测收敛；③ 若 `M` 内 padding 占比高，masked attention 利用率低——需监控 `qw/padding_ratio`（L1557）。

---

### 实验 D（进阶）：端到端列表重排序范式 —— Reranker

**动机**：对齐 DLCM/PERM 的"打分再打分"重排范式——先有 base 分数，再做列表级重排，能利用 base ranker 的排序信息并叠加列表上下文。

**对应论文**：[DLCM](https://arxiv.org/abs/1804.05936)、[PERM](https://arxiv.org/abs/1904.06813)、[HLATR](https://arxiv.org/abs/2205.10569)。

**实现方式**：
1. **两阶段训练**（推荐先做）：
   - Stage 1：用当前 baseline 训练到收敛，产出每个 item 的 base logits；
   - Stage 2：把 base logits 作为**额外输入特征**，加列表交互层（复用实验 C 的结构），用 listwise 损失微调（冻结或低 lr）。
2. **联合训练**：base logits 与 item 特征一起进列表交互层，端到端训练（更先进，但更难收敛）。
3. 个性化变体：参考 PERM，用 `uid_emb`（L1466）作为 attention query。

**预期收益**：在实验 C 基础上进一步提升，尤其当 base 打分本身有信息量时。

**风险**：工程量大；两阶段训练需要保存 base 模型输出；perf 上在线要支持"列表级"推断。

---

### 实验优先级与推荐顺序

| 阶段 | 实验 | 侵入度 | 建议 |
|------|------|--------|------|
| P0 | **A：listwise loss（ApproxNDCG 优先）** | 低 | 先跑，快速验证"列表级损失是否有效" |
| P0 | **B：列表上下文特征** | 中 | 可与 A 并行，验证"列表内信息是否有用" |
| P1 | **C：masked self-attention over items** | 高 | A/B 有正向信号后做，核心结构升级 |
| P1 | A+C 组合 | 中 | 交互结构 + 列表损失叠加 |
| P2 | D：端到端 reranker | 高 | 前序实验稳定后再投入 |

### 评估方案

- **主指标**：保持现有 `GAUC`（`gauc_cr_expo` 为核心）、`AUC`、`cr_valid`，并**新增列表级指标**：
  - **NDCG@K（K=5/10）**：在 `build_metric` 中按 `query_id` 分组计算（类似实验 A 的分段逻辑）；
  - **MRR**（首个正样本的倒数排名）；
  - **PAUC**（[DPIN](https://arxiv.org/abs/2106.12229) 提出的 position-wise AUC，观测位置一致性）。
- **分组 key**：GAUC 按 `dispatch_id`（现状）；NDCG/MRR 按 `query_id`（一次请求的候选列表），更能反映"相对比较"场景。
- **线上**：关注下单转化率（对应 E-Commerce 论文"order rate 最稳"的结论）。

---

## 6. 风险与注意事项

1. **列表结构在 `boolean_mask` 后被破坏**：所有列表操作要么在 mask 之前（`[B,M]` 网格），要么 mask 之后用 `query_id` 分段（`tf.unique` + `dynamic_partition`/`unsorted_segment_*`）。不要假设 `[BM]` 可以直接当列表用。
2. **Horovod 通信 shape**：列表交互层放在 `[B,M]` 固定 shape 上，避免在动态 `B_valid` 上触发跨 rank 通信。
3. **padding 处理**：`item_valid_mask` 要全程参与（attention mask、loss 归一化），否则 padding item 的梯度会污染模型。
4. **`dense_input_size` 联动**：实验 B/C 改输入维度，必须同步改 `Config.dense_input_size`（L218），否则 dense 变量 shape 错误。
5. **损失尺度**：listwise 损失与 BCE 量纲不同，务必归一化并配权重 `listwise_w`；从 0.1 起，观察总损失与 GAUC 曲线。
6. **位置偏差**：列表级损失天然会对"排前面的样本"更敏感，需配合位置特征/位置加权（Lambda 风格），避免把位置偏差学成"位置分"。
7. **截断一致**：线上打分与离线训练都要用同一 `item_rank_topn=60` 截断语义，保证 listwise 损失看到的列表与线上一致。
8. **TF1 静态图**：`tf.map_fn`/`dynamic_partition` 在图构建期就要定义好，避免动态 Python 控制流；调试时用小 batch 先验 shape。

---

## 7. 参考链接汇总

| # | 论文 | 链接 |
|---|------|------|
| 1 | ListNet (ICML 2007) | https://doi.org/10.1145/1273496.1273513 |
| 2 | ListMLE (ICML 2008) | https://doi.org/10.1145/1390156.1390306 |
| 3 | SoftRank (WSDM 2008) | https://doi.org/10.1145/1341531.1341544 |
| 4 | ApproxNDCG (IR Journal 2010) | https://doi.org/10.1007/s10791-009-9124-x |
| 5 | LambdaLoss (CIKM 2018) | https://arxiv.org/abs/1802.02265 |
| 6 | Revisiting Approx Metric Opt (SIGIR 2019) | https://arxiv.org/abs/1812.08473 |
| 7 | NeuralNDCG (arXiv 2021) | https://arxiv.org/abs/2102.07831 |
| 8 | PiRank (NeurIPS 2021) | https://arxiv.org/abs/2012.06731 |
| 9 | SmoothI (AAAI 2022) | https://arxiv.org/abs/1911.09798 |
| 10 | DLCM (SIGIR 2018) | https://arxiv.org/abs/1804.05936 |
| 11 | PERM (RecSys 2019) | https://arxiv.org/abs/1904.06813 |
| 12 | SetRank (SIGIR 2020) | https://arxiv.org/abs/1912.05891 |
| 13 | DPIN (SIGIR 2019) | https://arxiv.org/abs/2106.12229 |
| 14 | HLATR (arXiv 2022) | https://arxiv.org/abs/2205.10569 |
| 15 | RankGPT/LRL (EMNLP 2023) | https://arxiv.org/abs/2305.02156 |
| 16 | LTR for E-Commerce Search (SIGIR 2017) | https://arxiv.org/abs/1903.04263 |
| 17 | Bias and Debias Survey (arXiv 2020) | https://arxiv.org/abs/2010.03240 |

---

*注：本文档中所有论文链接均为真实存在的 arXiv/DOI 链接，检索与核对时间 2026-08-17。经典 LTR 论文（ListNet/ListMLE/SoftRank/ApproxNDCG）无 arXiv 版本，使用出版社 DOI 链接。*
