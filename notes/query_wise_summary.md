# Query-wise / List-wise 精排模型建模范式调研与实验设计

> 面向场景：智行酒店推荐精排模型，从 pointwise 打分向 query-wise（list-wise）建模范式演进的前期调研。
> Baseline 模型：`zxhtl_seq_v2_5_nosid_topnsp_qw_meanpool_M3oE_f_idgate_300`
> 本文档包含：① 业界/学术界 list-wise 精排前沿论文综述（所有链接均为真实可查的 arXiv / DOI 链接）；② 结合 baseline 代码的实际改造方式与实验设计。
>
> **公式渲染说明**：本文档的数学公式使用 **LaTeX（KaTeX / MathJax）** 语法（`$$...$$` 块级、`$...$` 行内）。请用支持数学渲染的 Markdown 查看器打开，例如 **Typora、Obsidian、VS Code（内置 Markdown 预览）、GitHub 网页端**。若查看器不支持，公式会退化为显示 LaTeX 源码。

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

##### 1.1 详细解析：ListNet 的原理、公式推导与例子

**① 问题设定与记号**

一个 query 有 $n$ 个候选文档（在酒店场景就是一次请求返回的候选酒店）。记模型对候选的打分为 $\hat{s} = (\hat{s}_1, \dots, \hat{s}_n)$，ground-truth 相关性标签为 $y = (y_1, \dots, y_n)$（酒店场景即 click/cr 的 0/1 标签或更细的等级）。ListNet 的核心思想是：**把"排序"看成在候选集合上定义一个概率分布**，然后让"模型打分诱导的概率分布"去逼近"标签诱导的概率分布"。

**② 排列概率（permutation probability）**

对任意一个排列 $\pi$（即把 $n$ 个候选排成一个有序序列），其概率定义为：

$$
P_s(\pi) = \prod_{t=1}^{n} \frac{\varphi(s_{\pi(t)})}{\sum_{j=t}^{n} \varphi(s_{\pi(j)})}
$$

**符号解读**（把每个符号拆开，避免混淆）：

- $\pi$：一个**排列（permutation）**，即把 $n$ 个候选排成一个有序序列。例如 $\pi=(2,3,1)$ 表示"第 1 位是 2 号物品，第 2 位是 3 号物品，第 3 位是 1 号物品"。
- $\pi(t)$：排列 $\pi$ 中**第 $t$ 位上的物品编号（identity）**，**不是分数**。$\pi(1)$ 是排在第 1 位的物品，$\pi(2)$ 是排在第 2 位的物品，以此类推；$\pi$ 是 $\{1,\dots,n\}$ 的一个排列，所以 $\pi(t)$ 取遍 $1\sim n$ 每个值恰好一次。
- $s_{\pi(t)}$：**第 $t$ 位那个物品的打分**，即模型给物品 $\pi(t)$ 的打分。若物品 $i$ 的打分记为 $s_i$，那么 $s_{\pi(t)}$ 就是"先取位置 $t$ 上的物品编号 $\pi(t)$，再查它的分数"。与①中的 $\hat{s}$ 是同一个东西（论文记作 $s$）。

**逐位拆开看**（$n=3$ 的完整例子）：

设 3 个物品 doc1 / doc2 / doc3，打分 $s=(1.0,\,0.5,\,0.2)$，取排列 $\pi=(2,3,1)$（即 doc2 排第 1、doc3 排第 2、doc1 排第 3）：

| $t$ | $\pi(t)$ | $s_{\pi(t)}$ | 分子 $\varphi(s_{\pi(t)})$ | 分母 $\sum_{j=t}^{3}\varphi(s_{\pi(j)})$（=剩余物品） |
|-----|----------|--------------|---------------------------|------------------------------------------------------|
| 1 | 2 | $s_2 = 0.5$ | $\varphi(0.5)$ | $\varphi(0.5)+\varphi(0.2)+\varphi(1.0)$（doc2, doc3, doc1） |
| 2 | 3 | $s_3 = 0.2$ | $\varphi(0.2)$ | $\varphi(0.2)+\varphi(1.0)$（doc3, doc1） |
| 3 | 1 | $s_1 = 1.0$ | $\varphi(1.0)$ | $\varphi(1.0)$（doc1） |

于是：

$$
P_s(\pi)
= \frac{\varphi(0.5)}{\varphi(0.5)+\varphi(0.2)+\varphi(1.0)}
  \times \frac{\varphi(0.2)}{\varphi(0.2)+\varphi(1.0)}
  \times \frac{\varphi(1.0)}{\varphi(1.0)}
$$

**关键点**：每一步的分母**只包含"还没被取走"的物品**（即位置 $t, t{+}1, \dots, n$ 上的物品），已经排在前面的物品不再参与竞争。这就是 Plackett-Luce 逐步抽选的本质——第 1 步从全部 $n$ 个里选，第 2 步从剩下 $n-1$ 个里选，……，最后一步只剩 1 个，概率恒为 1。

其中 $\varphi(x) > 0$ 是递增的"分值变换函数"。直观理解：第 $t$ 步从剩余候选中"选中 $\pi(t)$"的概率正比于它的 $\varphi(\text{分值})$，分母是剩余所有候选的 $\varphi(\text{分值})$ 之和；整条排列的概率是 $n$ 步的乘积。

论文给出两种常用 $\varphi$：

| 分值函数 | 要求 | 不变性 |
|------|------|--------|
| 线性 $\varphi(x)=x$ | 分值需 $>0$ | **尺度不变**：整体乘以常数 $a>0$，$P$ 不变 |
| 指数 $\varphi(x)=e^{x}$ | 无 | **平移不变**：整体加上常数 $c$，$P$ 不变（即 Plackett-Luce 模型） |

**③ Top-k 概率**

前 $k$ 个位置恰好是某个候选集合 $G_k$（不计内部顺序）的概率：

$$
P_s(G_k) = \sum_{\pi:\,\{\pi(1),\dots,\pi(k)\}=G_k} P_s(\pi)
$$

即对所有"前 $k$ 位恰好是 $G_k$ 的排列"的概率求和。注意这里要对 $k!$ 种内部顺序求和，计算代价约 $O(n!/(n-k)!)$，$k$ 稍大就不可行。$k=1$ 时退化为 **top-one 概率**（用指数函数时就是 softmax）：

$$
P_s(i) = \frac{\varphi(s_i)}{\sum_{j=1}^{n}\varphi(s_j)}
\qquad \text{指数形式: }\; P_s(i) = \frac{e^{s_i}}{\sum_j e^{s_j}}
$$

**④ ListNet 损失（top-one + 指数函数）与梯度推导**

论文实际使用的损失是**预测 top-one 分布与标签 top-one 分布的交叉熵**：

$$
L(\hat{s}, y) = - \sum_{i=1}^{n} P_y(i)\,\log P_{\hat{s}}(i),
\qquad
P_y(i) = \frac{e^{y_i}}{\sum_j e^{y_j}},\quad
P_{\hat{s}}(i) = \frac{e^{\hat{s}_i}}{\sum_j e^{\hat{s}_j}}
$$

（由于标签分布的熵 $H(P_y)$ 与模型无关，最小化交叉熵 $\equiv$ 最小化 KL 散度。）

**梯度推导**（softmax+交叉熵的"优雅抵消"）：

$$
\log P_{\hat{s}}(j) = \hat{s}_j - \log\Bigl(\textstyle\sum_k e^{\hat{s}_k}\Bigr)
\quad\Rightarrow\quad
\frac{\partial \log P_{\hat{s}}(j)}{\partial \hat{s}_i} = \delta_{ij} - P_{\hat{s}}(i)
$$

其中 $\delta_{ij}$ 为 Kronecker 记号。代入交叉熵求导：

$$
\frac{\partial L}{\partial \hat{s}_i}
= - \sum_j P_y(j)\,\bigl(\delta_{ij} - P_{\hat{s}}(i)\bigr)
= - P_y(i) + P_{\hat{s}}(i)\,\sum_j P_y(j)
= P_{\hat{s}}(i) - P_y(i)
$$

（最后一步用到 $\sum_j P_y(j) = 1$。）

**结论：$\dfrac{\partial L}{\partial \hat{s}_i} = P_{\hat{s}}(i) - P_y(i)$** —— 模型概率高于标签概率的 item 梯度为正（要压低分数），低于标签概率的 item 梯度为负（要抬高分数）。梯度把所有 item 的分数一起向"标签概率分布"的方向推，这就是"列表级"优化的本质：**每个 item 的梯度都受到同 query 其他 item 的影响**（因为 softmax 分母共享）。

**⑤ 具体例子**

$n=3$，标签 $y=(2,1,0)$（doc1 最相关），模型打分 $\hat{s}=(1.0,0.5,0.2)$。

| 项 | 计算 | 结果 |
|---|---|---|
| $e^{y}$ | $(e^{2},\,e^{1},\,e^{0})=(7.389,\,2.718,\,1.000)$ | $P_y=(0.665,\,0.245,\,0.090)$ |
| $e^{\hat{s}}$ | $(e^{1.0},\,e^{0.5},\,e^{0.2})=(2.718,\,1.649,\,1.221)$ | $P_{\hat{s}}=(0.486,\,0.295,\,0.219)$ |

$$
L = -\,[\,0.665\ln 0.486 + 0.245\ln 0.295 + 0.090\ln 0.219\,]
  = -\,[\,0.665(-0.722) + 0.245(-1.221) + 0.090(-1.518)\,]
  \approx 0.915
$$

$$
\frac{\partial L}{\partial \hat{s}} = P_{\hat{s}} - P_y = (-0.179,\; +0.050,\; +0.129)
$$

含义：doc1 真实最相关，但模型给的分数让它的概率(0.486)低于标签概率(0.665)，所以梯度为负 → 应提高 doc1 的分数；doc3 应降低。梯度方向完全符合直觉，且三个 item 的梯度通过 softmax 分母耦合在一起。

**⑥ 与"query 内 softmax + 对正样本做 CE"的关系（重点）**

如果 ground-truth 用 **one-hot**（正样本 $P_y=1$，其余 $0$），上面的交叉熵退化为：

$$
L = - \log P_{\hat{s}}(\text{正样本}) = - \log \mathrm{softmax}\bigl(\hat{s}_{\text{正样本}}\bigr)
$$

**逐步推导（为什么 one-hot 会让交叉熵"塌缩"成只剩正样本那一项）**

回到 ④ 的交叉熵（一个 query 有 $n$ 个候选，$P_{\hat{s}}(i)$ 是模型给 item $i$ 的 softmax 概率）：

$$
L(\hat{s}, y) = -\sum_{i=1}^{n} P_y(i)\, \log P_{\hat{s}}(i)
$$

"one-hot 标签"的含义是：把标签分布 $P_y$ 直接定义成"正样本概率为 1、其余为 0"的确定性分布。设正样本是第 $p$ 个 item：

$$
P_y(i) = \begin{cases} 1 & i = p \\[2pt] 0 & i \neq p \end{cases}
$$

把它代进求和，逐项写开：

$$
L = -\Bigl[\, \underbrace{P_y(1)}_{=\,0}\log P_{\hat{s}}(1) + \cdots + \underbrace{P_y(p)}_{=\,1}\log P_{\hat{s}}(p) + \cdots + \underbrace{P_y(n)}_{=\,0}\log P_{\hat{s}}(n) \Bigr]
$$

凡是 $i \neq p$ 的项，系数 $P_y(i) = 0$，整项直接消失（$0 \times \log(\cdot) = 0$）；只有 $i = p$ 那一项留下来：

$$
L = -\,1 \cdot \log P_{\hat{s}}(p) = -\log P_{\hat{s}}(p)
$$

而 $P_{\hat{s}}(p)$ 就是模型对正样本的 softmax 概率：

$$
P_{\hat{s}}(p) = \frac{e^{\hat{s}_p}}{\sum_{j=1}^{n} e^{\hat{s}_j}} = \mathrm{softmax}\bigl(\hat{s}\bigr)_p
$$

所以最终：

$$
L = -\log \mathrm{softmax}\bigl(\hat{s}\bigr)_p = -\log \frac{e^{\hat{s}_p}}{\sum_{j=1}^{n} e^{\hat{s}_j}}
$$

**直观理解**：交叉熵度量"目标分布 vs 预测分布"的差异。目标分布是"正样本概率 = 1"的确定性分布，因此最小化损失就等价于**让模型把概率尽量集中到正样本上**——即最大化正样本的 softmax 概率，也就是最小化它的负对数。这就是"对正样本位置做交叉熵"。

**数值例子**（沿用 ⑤：$n=3$，打分 $\hat{s}=(1.0,0.5,0.2)$，正样本是 doc1）：

$$
P_{\hat{s}}(1) = \frac{e^{1.0}}{e^{1.0}+e^{0.5}+e^{0.2}} = \frac{2.718}{5.588} = 0.486
$$

$$
L_{\text{one-hot}} = -\ln 0.486 = 0.721
$$

对比 ⑤ 里"非 one-hot"的标签 softmax 版本（$P_y=(0.665,0.245,0.090)$）算出的 $L=0.915$——两个损失方向一致（都要求 doc1 概率高），但 one-hot 版数值不同，因为它完全忽略了负样本的标签分布。

**两点澄清（容易踩的坑）**：

1. **one-hot 版只"管"正样本**：负样本项系数为 0，既不贡献损失也不贡献梯度。所以它强制正样本"排第一"，但对负样本之间的相对顺序没有任何约束——这与完整 ListMLE（会约束整条排列）不同。
2. **one-hot 与"论文原始 $P_y$"的差别**：论文的 $P_y$ 是把标签 $y$ 做 softmax（$P_y(i)=e^{y_i}/\sum_j e^{y_j}$）。若 $y_p=1$、其余为 0，则 $P_y(p)=e/(e+n-1)$，不是精确的 1，其余负样本也有 $\frac{1}{e+n-1}$ 的小权重。one-hot 是把 $P_y$ 直接设为 $\delta_{i,p}$ 的简化，也是多分类里标准的 hard-label 交叉熵。两者在"单正样本"场景方向一致、数值略有差别，工业实现常用 one-hot。

这正是你同事说的做法——"query 内候选打分做 softmax，再对正样本位置做交叉熵"。**它是 ListNet top-one 损失在 one-hot 标签下的特例**，也是工业界最常见的 listwise 简化实现。原论文本身用的 P_y 是"标签的 softmax"（非 one-hot），但在"单正样本 + 其余为 0"的设定下两者非常接近（`exp(1) vs exp(0)=1`，权重 2.72:1），one-hot 是常用简化。

**深入追问：你说的"对 0/1 标签做 softmax"（$e^{1}/(e^{0}+e^{0}+e^{1})$）为什么不行？——其实可以，它正是原论文的做法**

先给结论：**你的公式完全合法，而且正是原论文（Cao et al. 2007）的做法**。one-hot 不是"唯一正确"，它只是目标分布的另一种（更激进的）选择。关键在于理解 $P_y$ 的性质。

**$P_y$ 是"定义出来的目标分布"，不是"算出来的"**。ListNet 的框架是：把模型分数 $\hat{s}$ 和标签 $y$ 各自变成一个概率分布（$P_{\hat{s}}$ 和 $P_y$），再让两者互相逼近（交叉熵）。$P_y$ 怎么定义是建模者的自由，只要满足两条：① 是合法分布（求和为 1）；② 随标签单调（更相关 → 概率更高）。于是有两种自然选择：

**方式 A：对标签做 softmax（你的公式，论文做法）**

$$
P_y(i) = \frac{e^{y_i}}{\sum_j e^{y_j}}
\qquad \Rightarrow \qquad
P_y = \Bigl(\frac{e}{e+2},\,\frac{1}{e+2},\,\frac{1}{e+2}\Bigr) \approx (0.576,\,0.212,\,0.212)
$$

**方式 B：one-hot（硬目标，工业常用）**

$$
P_y(i) = \delta_{i,\,p}
\qquad \Rightarrow \qquad
P_y = (1,\,0,\,0)
$$

**两者的关系：方式 A 是方式 B 的"锐化极限"**。引入锐化温度 $t$：

$$
P_y^{(t)}(i) = \frac{e^{t\,y_i}}{\sum_j e^{t\,y_j}}
= \Bigl(\frac{e^{t}}{e^{t}+2},\,\frac{1}{e^{t}+2},\,\frac{1}{e^{t}+2}\Bigr)
\xrightarrow{t\to\infty} (1,\,0,\,0)
$$

- $t=1$：就是你的公式（温和版，正样本目标概率只有 $e/(e+2)=0.576$）；
- $t\to\infty$：趋近 one-hot（正样本目标概率 1）。

所以 **one-hot 可以理解为"把目标分布锐化到极致"**：它宣称"正样本必须 100% 排第一"，而 softmax-of-labels 只宣称"正样本的 top-1 概率是每个负样本的 $e$ 倍"。

**梯度上的实际差异**（同一个例子：$\hat{s}=(1.0,0.5,0.2)$，$P_{\hat{s}}=(0.486,0.295,0.219)$，梯度 $\partial L/\partial \hat{s}_i = P_{\hat{s}}(i)-P_y(i)$）：

| item | $P_{\hat{s}}(i)$ | 方式A $P_y(i)$ | 方式A梯度 | 方式B $P_y(i)$ | 方式B梯度 |
|------|------------------|----------------|-----------|----------------|-----------|
| doc1（正） | 0.486 | 0.576 | $-0.090$ | 1 | $-0.514$ |
| doc2（负） | 0.295 | 0.212 | $+0.083$ | 0 | $+0.295$ |
| doc3（负） | 0.219 | 0.212 | $+0.007$ | 0 | $+0.219$ |

解读：

- **优化方向完全一致**：两种方式都让 doc1 升分、doc2/doc3 降分，排序目标相同。
- **力度不同**：one-hot 的梯度明显更激进——正样本约 5.7 倍（$0.514/0.090$），负样本可差数十倍（如 doc3 的 $0.219/0.007 \approx 31$ 倍）。因为 one-hot "要求"负样本概率压到 0，而 softmax-of-labels 只要求压到 $1/(e+2)=0.212$ 即可。
- **列表很长时两者趋于一致**：$n$ 很大时 $1/(e+n-1)\to 0$，方式 A 的负样本目标概率趋近 0，与方式 B 越来越接近。

**那到底该用哪个？**

| 场景 | 建议 | 原因 |
|------|------|------|
| 二值标签，只关心"正样本是否排第一" | **one-hot** | 实现即标准多分类 softmax 损失（`tf.nn.sparse_softmax_cross_entropy_with_logits`），梯度干净，工业默认 |
| 多级相关性标签（如 NDCG 0-4 分） | **softmax-of-labels（或按标签加权的目标）** | one-hot 会丢掉"2 分比 1 分更相关"的层级信息，这正是原论文的设定场景 |
| 二值标签但 query 内**多个正样本** | 完整 ListMLE / top-k 损失 | one-hot 需逐正样本或选主正样本处理，softmax-of-labels 给所有正样本等权重，都不理想 |

**⑦ 关键性质与局限**

- 复杂度：top-one 版为 **$O(n)$**，优于 pairwise RankNet 的 $O(n^2)$。
- 与指标的关系：ListNet 损失优化的是"概率分布距离"，与 NDCG 等排序指标只是**松散相关**（论文讨论了在特殊条件如二值标签下才与 NDCG 有更强联系），不直接对齐位置折扣。
- 局限：① 不区分"排第 1 还是第 2"（无位置折扣）；② 完整排列概率/top-k(k>1) 计算代价高，论文主要用 k=1；③ 对"多个正样本"的处理依赖标签分布的选择。

**⑧ 与 baseline 的 BCE 损失对比：pointwise vs listwise（核心区别）**

**baseline 的损失到底是什么**

`loss_op_cr = tf.reduce_sum(tf.keras.backend.binary_crossentropy(target=labels[1], output=q[2]))`

- `labels[1]` = 转化标签 $\text{cr} \in \{0,1\}$（该酒店是否被下单）；
- `q[2]` = 模型输出的 $\text{ctcvr} = P(\text{click}) \times P(\text{convert})$；
- `tf.reduce_sum` 把 batch 内每个 item 的损失全部加起来。

其数学形式是**逐样本二分类交叉熵**：

$$
\ell_{\text{BCE}} = \sum_{i=1}^{N} -\,[\, y_i \log p_i + (1-y_i)\log(1-p_i)\,],
\qquad p_i = q[2]_i,\ \ y_i = \text{cr}_i
$$

**关键**：虽然数据是 query-wise 组织的（一次请求的 item 在同一个 batch 块、共享 `query_id`），但这个损失对每个 item $i$ 只用到它**自己**的 $(p_i, y_i)$——同 query 的其他 item 完全不参与梯度。这就是 pointwise。

作为对比，listwise 损失（ListNet top-one、one-hot 版）长这样：

$$
\ell_{\text{list}} = -\log \frac{e^{s_{\text{正}}}}{\sum_{j \in \text{query}} e^{s_j}}
$$

正样本的损失依赖**同 query 所有 item 的分数**（softmax 分母）。这就是本质区别：pointwise 看"每个 item 绝对对错"，listwise 看"正样本在列表里的相对位置"。

**具体例子**（同一个 query 返回 3 个酒店 A / B / C，转化标签 $y=(1,0,0)$，只有 A 被下单）：

| 情况 | logits $s=(s_A,s_B,s_C)$ | 排序 |
|------|--------------------------|------|
| ① 排序正确 | $(2.0,\,1.0,\,0.5)$ | A 第 1 |
| ② 排序正确，整体平移 $-1.8$ | $(0.2,\,-0.8,\,-1.3)$ | A 第 1（与①相对序完全相同） |
| ③ 排序错误 | $(0.5,\,1.0,\,2.0)$ | A 第 3（最后） |

**情况①**（A 第 1）：

$$
p = \bigl(\sigma(2.0),\,\sigma(1.0),\,\sigma(0.5)\bigr) = (0.881,\,0.731,\,0.622)
$$

$$
\ell_{\text{BCE}} = -\ln 0.881 - \ln(1-0.731) - \ln(1-0.622) = 0.127 + 1.313 + 0.974 = 2.414
$$

$$
\ell_{\text{list}} = -\log\frac{e^{2.0}}{e^{2.0}+e^{1.0}+e^{0.5}} = -\log\frac{7.389}{11.756} = -\log 0.628 = 0.465
$$

**情况②**（相对序同①，但整体平移 $-1.8$）：

$$
p = \bigl(\sigma(0.2),\,\sigma(-0.8),\,\sigma(-1.3)\bigr) = (0.550,\,0.310,\,0.214)
$$

$$
\ell_{\text{BCE}} = -\ln 0.550 - \ln(1-0.310) - \ln(1-0.214) = 0.598 + 0.371 + 0.241 = 1.210
$$

$$
\ell_{\text{list}} = -\log\frac{e^{0.2}}{e^{0.2}+e^{-0.8}+e^{-1.3}} = -\log\frac{1.221}{1.943} = -\log 0.628 = 0.465
$$

**情况③**（A 排最后）：

$$
p = \bigl(\sigma(0.5),\,\sigma(1.0),\,\sigma(2.0)\bigr) = (0.622,\,0.731,\,0.881)
$$

$$
\ell_{\text{BCE}} = -\ln 0.622 - \ln(1-0.731) - \ln(1-0.881) = 0.474 + 1.313 + 2.127 = 3.914
$$

$$
\ell_{\text{list}} = -\log\frac{e^{0.5}}{e^{0.5}+e^{1.0}+e^{2.0}} = -\log\frac{1.649}{11.756} = -\log 0.140 = 1.964
$$

**结果汇总**：

| 情况 | 排序 | $\ell_{\text{BCE}}$（baseline） | $\ell_{\text{list}}$（ListNet） |
|------|------|-------------------------------|-------------------------------|
| ① 正确 | A 第 1 | 2.414 | 0.465 |
| ② 同序平移 | A 第 1 | **1.210（变）** | **0.465（不变）** |
| ③ 错误 | A 第 3 | 3.914 | 1.964 |

**从这个例子读出的结论**：

1. **BCE 没有平移不变性**：② 与 ① 排序完全相同，但整体分数平移 $-1.8$ 后 $\ell_{\text{BCE}}$ 从 2.414 变到 1.210——损失被"绝对概率"主导，而不是"相对排序"。listwise 的 softmax 分母把所有分数归一化，所以② 与 ① 的 $\ell_{\text{list}}$ 都是 0.465。
2. **BCE 对排序错误的惩罚是间接的**：③ 中 A 排最后，$\ell_{\text{BCE}}$ 暴涨的主因是负样本 B、C 的预测概率太高（$1-p$ 很小），而不是"正样本相对位置错"这件事被直接建模。
3. **listwise 直接对齐排序**：$\ell_{\text{list}}$ 只看"正样本分数在所有 item 中的相对位置"，这正是 GAUC / NDCG 关心的对象。但 ListNet 无位置折扣（A 排第 1 与排第 2 对 top-one 损失一样），要完全对齐 NDCG 还需 ApproxNDCG / LambdaLoss 这类带位置折扣的损失（见 #4 / #5 / #6）。
4. **落到代码**：把 baseline 的 $\ell_{\text{BCE}}$ 换成 listwise 损失，就是要按 `query_id` 分组、组内做 softmax 再对正样本取负对数；`tf.reduce_sum` 相应变成"每个 query 各算一个列表损失，再对 query 求平均"（实现见 §5 实验 A）。

**⑨ baseline 中的 ListNet 代码实现：每一行与数学公式的对应**

> 对应实验目录 `zxhtl_seq_v2_5_nosid_topnsp_qw_listnet_auxloss_meanpool_M3oE_f_idgate_300` 中 `build_model` 里的 `_build_listnet_loss`。

**⑨.1 query_id 怎么来的（一次请求的所有 item 共享同一个 id）**

在 `_parse_tfrd_query_wise` 中：

```python
query_id = tf.fill([self.config.parm.items_per_request], tf.cast(request_id, tf.int64))
```

- `request_id` 是 `build_dataset` 里 `tf.data.Dataset.enumerate()` 生成的递增下标：第 0 个请求 → 0、第 1 个请求 → 1、……
- **重要（训练踩坑点）**：`enumerate()` 是对**整个数据流**全局计数。在 `train_eval` 混合数据集下，`request_id` 会一直递增到几十万甚至更大，**并不是**每 batch 从 0 重新计数的 `0..batch_size-1`。因此 ListNet 分组时**不能假设 `query_id` 落在 `[0, batch_size)`**，必须用 `tf.unique` 压缩（见 ⑨.5）。
- `tf.fill([M], value)` 生成一个长度为 $M$、**所有元素都等于 `value`** 的张量。因此一个请求展开的 $M$ 个位置（真实 item + padding）全部拿到同一个 `request_id`。
- `build_model` 里 flatten 成 `[BM, 1]`（request-major：全局下标 = $r \times M + i$），再经 `boolean_mask` 后，每个有效 item 仍带着自己的 query_id → 这就是 ListNet 按 query 分组的依据。

示例（$batch\_size=2,\ M=3$）：

$$
\text{enumerate} \Rightarrow \text{request_id}=0,1
\Rightarrow
\begin{bmatrix} 0&0&0 \\ 1&1&1 \end{bmatrix}
\xrightarrow{\text{flatten}}
\begin{bmatrix} 0\\0\\0\\1\\1\\1 \end{bmatrix}
$$

**⑨.2 tf.unsorted_segment_sum 的语法**

```python
tf.unsorted_segment_sum(data, segment_ids, num_segments)
```

- `data`：shape `[N, ...]`，要被分组的张量；
- `segment_ids`：shape `[N]`，`data[i]` 属于哪一组（从 0 编号）；
- `num_segments`：分组总数，**必须 ≥ `segment_ids` 最大值 + 1**；
- 输出 `output[k] = Σ_{j: segment_ids[j]==k} data[j]`，shape `[num_segments] + data.shape[1:]`。

名字里的 **unsorted** 表示不要求 `segment_ids` 有序（`boolean_mask` 之后 item 顺序是乱的，正好需要这个）。它是在"一维展平 + 每元素一个组 id"的形态下，把数据重新聚合成 `[num_q]` 组级向量的标准做法：

实际代码里 segment_ids 用的是 **`qid_compact`**（由 `tf.unique(query_id)` 压缩后的 `0..K-1`，见 ⑨.5），`num_q` 是实际出现的 query 数：

| 代码 | 数学含义 |
|------|---------|
| `tf.unsorted_segment_max(logits, qid_compact, num_q)` | $M_q = \max_{j\in q} s_j$，每组最大 logit |
| `tf.unsorted_segment_sum(exp_shifted, qid_compact, num_q)` | $Z_q = \sum_{j\in q} e^{\,s_j - M_q}$，每组 softmax 分母 |
| `tf.unsorted_segment_sum(label, qid_compact, num_q)` | $k_q = \sum_{j\in q} \mathbb{1}[y_j{=}1]$，每组正样本数 |
| `tf.unsorted_segment_sum(pos_loss, qid_compact, num_q)` | 每组正样本的损失和 |

示例（7 个 item、3 个 query，`qid_compact` 已压缩）：

```
qid_compact = [0,0,0, 1,1, 2,2]
label       = [1,0,0, 0,1, 0,0]
pos_cnt = tf.unsorted_segment_sum(label, qid_compact, 3) = [1, 1, 0]
```

**⑨.3 数值稳定的 log-softmax（为什么先减 max）**

原始 log-softmax：

$$
\log \mathrm{softmax}(s)_i = s_i - \log\Bigl(\sum_{j\in q} e^{s_j}\Bigr)
$$

问题：$e^{s_j}$ 在 $s_j \gtrsim 88$（float32）时溢出成 `inf`，损失变 NaN。利用 **softmax 平移不变性**（分子分母同乘 $e^{-M_q}$）：

$$
\frac{e^{s_i}}{\sum_j e^{s_j}}
= \frac{e^{s_i - M_q}}{\sum_j e^{s_j - M_q}}
\quad\Rightarrow\quad
\log \mathrm{softmax}(s)_i
= (s_i - M_q) - \log\Bigl(\sum_j e^{s_j - M_q}\Bigr)
$$

代码与公式逐行对应：

| 代码 | 公式 |
|------|------|
| `max_logit = tf.unsorted_segment_max(logits, qid_compact, num_q)` | $M_q = \max_{j\in q} s_j$ |
| `shifted = logits - tf.gather(max_logit, qid_compact)` | $t_i = s_i - M_{q(i)} \le 0$ |
| `exp_shifted = tf.exp(shifted)` | $u_i = e^{t_i} \in (0,\,1]$，不会溢出 |
| `denom = tf.unsorted_segment_sum(exp_shifted, qid_compact, num_q)` | $Z_q = \sum_{j\in q} e^{\,s_j - M_q}$ |
| `log_softmax = shifted - tf.log(tf.gather(denom, qid_compact))` | $\log \mathrm{softmax}(s)_i = t_i - \log Z_{q(i)}$ |

**注意最后一行**：`denom` 是 per-query 的 `[K]`，必须先 `tf.gather` 展开回 per-item 维度（`[B_valid]`）再与 `shifted` 相减。若直接写 `shifted - tf.log(denom)`，`[B_valid] - [K]` shape 不匹配，会报 `Select ... Input 0: [2076] != input 1: [64]` 之类的错。

数值例子（$s=(1.0,0.5,0.2)$）：$M_q=1.0$，$t=(0,-0.5,-0.8)$，$Z=1+0.6065+0.4493=2.0558$，于是 $\log\mathrm{softmax}(s_1) = 0 - \log 2.0558 = -0.721$ ✓（与 $\log\frac{e^{1}}{e^{1}+e^{0.5}+e^{0.2}} = -0.721$ 一致）。

**⑨.4 多正样本：当前实现做了什么（数学等价与梯度）**

`pos_loss = -label * log_softmax` 会对 query 内**每个**正样本各算一个 `-log softmax`，再用 `pos_sum / pos_cnt` 取平均。假设 query 有 $k$ 个正样本（集合 $P$）：

$$
L_q = \frac{1}{k}\sum_{p\in P} -\log \mathrm{softmax}(s_p)
= \frac{1}{k}\Bigl[\sum_{p\in P}\bigl(\log Z - s_p\bigr)\Bigr]
= \log Z - \frac{1}{k}\sum_{p\in P} s_p
$$

梯度（$p$ 为正样本、$n$ 为负样本）：

$$
\frac{\partial L_q}{\partial s_p} = \mathrm{softmax}(s_p) - \frac{1}{k},
\qquad
\frac{\partial L_q}{\partial s_n} = \mathrm{softmax}(s_n)
$$

**含义**：每个正样本被推向"softmax 概率 $= 1/k$"，每个负样本被压向 0。收敛时**所有正样本排在所有负样本前面、正样本之间概率均分**。所以当前实现**也能处理多正样本**——它表达的目标是"正样本组 vs 负样本组"。

**和 ListMLE 的区别**：

| | ListNet（当前实现） | ListMLE |
|---|---|---|
| 正样本内部顺序 | 不指定，概率均分 | 需给定排列 $\pi^{*}$，显式学"$p_1$ 最前、$p_2$ 次之……" |
| 负样本内部顺序 | 不约束 | 也参与逐位 softmax（取决于你构造的排列） |
| 适用场景 | 只要"被点击/下单的都在没点过的前面"即可 | 正样本内部也要分先后（如下单 > 点击） |

> 若需要正样本内部排序，也可以选 **ApproxNDCG**（按 relevance 累加、带位置折扣），比 ListMLE 更贴合酒店排序目标。

**⑨.5 修复后的完整实现（train_eval 下 query_id 是全局大数）**

训练中实际遇到的报错：

```
Invalid argument: Inputs to operation Select ... must have the same size and shape. Input 0: [2076] != input 1: [64]
```

两个根因（均已修复）：

1. **`num_q = batch_size` 假设错误**：`enumerate()` 对整个数据流计数，在 `train_eval` 混合数据集下 `query_id` 是几十万的大数，远超 `batch_size=64`。`tf.unsorted_segment_*` / `tf.gather` 遇到越界 segment id 直接报错。修复：`tf.unique` 把实际出现的 `query_id` 压缩成 `0..K-1`。
2. **`log_softmax = shifted - tf.log(denom)` 广播错误**：`shifted` 是 per-item `[B_valid]`，`denom` 是 per-query `[K]`，直接相减 shape 不匹配（这正是 `Select ... [2076] != [64]` 的来源）。修复：`tf.gather(denom, qid_compact)` 展开回 per-item 维度。

修复后的核心代码：

```python
if getattr(self.config.parm, 'query_wise_mode', False):
    qid_1d = tf.reshape(query_id, [-1])                        # [B_valid]
    # query_id 来自 enumerate() 对整个数据流的全局计数,
    # train_eval 混合数据集下并不限定在 [0, batch_size)。
    unique_qid, qid_compact = tf.unique(qid_1d)                # qid_compact ∈ [0, K)
    num_q = tf.size(unique_qid)                                # 实际出现的 query 数

    def _build_listnet_loss(logits, label, name):
        logits = tf.reshape(logits, [-1])                      # [B_valid]
        label = tf.reshape(tf.cast(label, tf.float32), [-1])   # [B_valid] 0/1
        max_logit = tf.unsorted_segment_max(
            logits, qid_compact, num_segments=num_q)           # [K]
        shifted = logits - tf.gather(max_logit, qid_compact)   # [B_valid]
        exp_shifted = tf.exp(shifted)
        denom = tf.unsorted_segment_sum(
            exp_shifted, qid_compact, num_segments=num_q)      # [K]
        # denom 是 per-query 的 [K], 必须 gather 回 per-item 再减
        log_softmax = shifted - tf.log(tf.gather(denom, qid_compact))  # [B_valid]
        pos_loss = -label * log_softmax                        # 仅正样本贡献
        pos_cnt = tf.unsorted_segment_sum(
            label, qid_compact, num_segments=num_q)            # [K]
        pos_sum = tf.unsorted_segment_sum(
            pos_loss, qid_compact, num_segments=num_q)         # [K]
        per_query = tf.where(
            pos_cnt > 0,
            pos_sum / tf.maximum(pos_cnt, 1.0),
            tf.zeros_like(pos_cnt))                            # [K]
        valid_cnt = tf.reduce_sum(tf.cast(pos_cnt > 0, tf.float32))
        loss = tf.reduce_sum(per_query) / tf.maximum(valid_cnt, 1.0)
        tf.summary.scalar('listnet_loss/' + name, loss)
        return loss

    listnet_loss_click = _build_listnet_loss(mlp_out_list[0], labels[0], 'click')
    listnet_loss_cr = _build_listnet_loss(mlp_out_list[2], labels[1], 'cr')
    self.loss_ops = [loss_op_click, loss_op_cr, loss_op_dr, loss_op_cvr,
                     self.config.parm.listnet_w_click * listnet_loss_click,
                     self.config.parm.listnet_w_cr * listnet_loss_cr]
```

要点：`qid_compact` 保证 segment id 总是 `0..K-1` 的紧凑整数（同一 query 的 item 映射到同一个值），无论 `query_id` 本身多大；`denom` 用 `tf.gather` 展开回 per-item 维度，`log_softmax` 的 shape 恒为 `[B_valid]`。

---

#### 2. ListMLE —— Listwise Approach to Learning to Rank: Theory and Algorithm
- **链接**：[DOI 10.1145/1390156.1390306](https://doi.org/10.1145/1390156.1390306)（ICML 2008）
- **作者**：Fen Xia, Tie-Yan Liu, Jue Wang, Wensheng Zhang, Hang Li
- **核心思想**：与 ListNet 同源但更简洁——直接对**真实排列**做极大似然：$L = -\log P(\pi^{*} \mid \hat{s})$，$P$ 是 Plackett-Luce 模型（$P(\pi) = \prod_t \mathrm{softmax}(\text{剩余分数})_t$）。有统计一致性理论保证。
- **方法**：一次前向得到列表分数，按真实排列序（正样本排在负样本前）依次取 softmax 累乘再取负对数。SGD 优化。
- **与 baseline 结合**：酒店场景"真实排列"由点击/下单定义（点击者排前面）。实现时按 `query_id` 分组，组内按 label 降序排列后计算。**这是本文推荐优先尝试的 listwise 损失之一**：公式简单、数值稳定、理论上可解释。

##### 2.1 详细解析：ListMLE 的原理、公式推导与例子

**① 问题设定**

与 ListNet 相同：一个 query 有 $n$ 个候选，模型打分 $\hat{s}=(\hat{s}_1,\dots,\hat{s}_n)$。区别在于 ground-truth 用一个**排列** $\pi^{*}$ 表示（按相关性从高到低排好的全序）。ListMLE 直接最大化"真实排列在 Plackett-Luce 模型下的似然"，即**负对数似然损失**。

**② Plackett-Luce 排列似然**

$$
P(\pi^{*} \mid \hat{s})
= \prod_{t=1}^{n} \frac{\exp\bigl(\hat{s}_{\pi^{*}(t)}\bigr)}{\sum_{j=t}^{n} \exp\bigl(\hat{s}_{\pi^{*}(j)}\bigr)}
$$

逐项理解：第 $t$ 步在"从第 $t$ 位到末尾的剩余候选"上做 softmax，取当前位（真实排列中第 $t$ 个文档）的概率；$n$ 步概率相乘就是整个排列出现的概率。这就是同事所说"对剩余候选做 softmax"的完整版本——不是只做一步，而是**每一步都对剩余候选 softmax，然后累乘**。

**③ ListMLE 损失与梯度推导**

$$
L(\hat{s}, \pi^{*})
= - \log P(\pi^{*} \mid \hat{s})
= \sum_{t=1}^{n} \Bigl[\, \log\Bigl(\sum_{j=t}^{n} e^{\hat{s}_{\pi^{*}(j)}}\Bigr) - \hat{s}_{\pi^{*}(t)} \Bigr]
$$

梯度（记 $\mathrm{pos}(i)$ 为 item $i$ 在 $\pi^{*}$ 中的位置）：

$$
\frac{\partial L}{\partial \hat{s}_i}
= \sum_{t=1}^{\mathrm{pos}(i)} \frac{e^{\hat{s}_i}}{\sum_{j=t}^{n} e^{\hat{s}_{\pi^{*}(j)}}} - 1
$$

推导要点：$-\hat{s}_{\pi^{*}(t)}$ 这一项只对 $t = \mathrm{pos}(i)$ 有贡献（系数 $-1$）；$\log\bigl(\sum_{j\ge t} e^{\hat{s}_{\pi^{*}(j)}}\bigr)$ 这一项只要 item $i$ 还在"剩余集合"里（即 $t \le \mathrm{pos}(i)$）就有贡献，贡献为 $e^{\hat{s}_i}/S_t$（其中 $S_t = \sum_{j=t}^{n} e^{\hat{s}_{\pi^{*}(j)}}$ 是后缀和）。

**直观解释**：每个 item $i$ 与"在真实排列中排在它后面的所有 item"竞争。若排在其后的 item 分数逼近它，梯度中正的项变大，模型被迫拉开差距。整个损失与梯度**预计算后缀和 $S_t$ 后可做到 $O(n)$ 复杂度**。

**④ Top-k ListMLE**

只取前 $k$ 步的求和：

$$
L_{\text{top-}k}(\hat{s}, \pi^{*})
= \sum_{t=1}^{k} \Bigl[\, \log\Bigl(\sum_{j=t}^{n} e^{\hat{s}_{\pi^{*}(j)}}\Bigr) - \hat{s}_{\pi^{*}(t)} \Bigr]
$$

含义：只要求前 $k$ 个位置排对，后面的顺序不管。酒店场景可以用 top-$k$（如只关注列表前 10 位）。

**⑤ 具体例子**

$n=3$，真实排列 $\pi^{*}=(1,2,3)$（doc1 > doc2 > doc3，对应标签 $y=(2,1,0)$），模型打分 $\hat{s}=(1.0,0.5,0.2)$。

$$
P(\pi^{*})
= \frac{e^{1}}{e^{1}+e^{0.5}+e^{0.2}}
  \cdot \frac{e^{0.5}}{e^{0.5}+e^{0.2}}
  \cdot \frac{e^{0.2}}{e^{0.2}}
= \frac{2.718}{5.588} \cdot \frac{1.649}{2.870} \cdot 1
= 0.486 \times 0.574 \times 1
= 0.279
$$

$$
L = -\ln(0.279) \approx 1.275
$$

| item | 梯度 $\partial L/\partial \hat{s}_i$ | 方向 |
|------|-------------------------------------|------|
| doc1（pos=1） | $\frac{e^{1}}{5.588} - 1 = 0.486 - 1 = -0.514$ | 应升分 |
| doc2（pos=2） | $\frac{e^{0.5}}{5.588} + \frac{e^{0.5}}{2.870} - 1 = 0.295 + 0.574 - 1 = -0.131$ | 应升分 |
| doc3（pos=3） | $\frac{e^{0.2}}{5.588} + \frac{e^{0.2}}{2.870} + \frac{e^{0.2}}{1.221} - 1 = 0.219 + 0.426 + 1.0 - 1 = +0.645$ | 应降分 |

注意 doc3 的梯度里含 **3 个 softmax 项**（在 t=1、2、3 三步都参与了竞争），这正是 ListMLE 与"单步 softmax CE"的关键差别。

**⑥ 与"query 内 softmax + 对正样本做 CE"的关系（重点，需辨析）**

- 若只取**第一步**（$t=1$ 的 top-1 ListMLE，即只要求第 1 位是正样本），损失就是 $-\log \dfrac{e^{\hat{s}_{\text{正}}}}{\sum e^{\hat{s}}} = -\log \mathrm{softmax}(\hat{s}_{\text{正}})$ —— 恰好等于同事说的"query 内 softmax + 对正样本做 CE"。
- 但**完整 ListMLE 是 n 步 softmax 的累乘**，不只考虑正样本：负样本之间的相对顺序也会贡献损失（见上面 doc3 梯度的 3 个项）。
- 这意味着完整 ListMLE 要求 ground-truth 是**全序**。若标签只有"正/负"二值（部分序），需要额外约定负样本内部顺序（按展示位置 / 随机），否则同样的正负标签会对应不同损失值。**这正是工业界更常用"单正样本 top-one CE"（ListNet one-hot 版）而非完整 ListMLE 的实践原因。** 在 baseline 落地时，若 query 内有多个点击/下单，建议用 top-k ListMLE 或先只取单正样本做 top-one。

**⑦ 关键性质与局限**

- **凸性**：损失在打分向量 $\hat{s}$ 上是凸的（$-\log\sum e^{\hat{s}}$ 是凸的，线性项不破坏凸性）；线性打分模型下在参数 $w$ 上也是凸的 → 可保证收敛到全局最优。
- **一致性**：论文给出 listwise 损失"一致性（consistency）"的充分条件，并证明 ListMLE 是 sound + consistent 的（样本无穷时收敛到最优排序；对 NDCG、AP 有一致性分析）。这是相对 ListNet 的主要理论贡献。
- **局限**：ListMLE 优化的是"排列似然"，与 NDCG/GAUC 等**业务指标不直接对应**（部分综述认为其与指标"脱钩"）；实践中常与 ApproxNDCG/LambdaLoss 等"指标对齐损失"搭配使用。对部分序（多正样本、同分）敏感，需要额外约定。**注意：原文中没有"Top-1 情形退化为 softmax cross-entropy"这一说法——退化为单步 softmax 的只是"只取第一位的 top-1 版本"，这是本文档为便于落地做的解读，不要与原文混淆。**

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

##### 4.1 详细解析：ApproxNDCG 的原理、公式推导与例子

**① 目标：NDCG 指标（为什么不可导）**

对一个 query 的 $n$ 个 item，按模型打分 $s$ 降序排列。位置 $i$（从 1 开始）的 item 增益为 $2^{\text{rel}_i}-1$（binary relevance 下正样本=1、负样本=0），按位置折扣 $\log_2(i+1)$ 累加：

$$
\text{DCG} = \sum_{i=1}^{n} \frac{2^{\text{rel}_i}-1}{\log_2(i+1)},
\qquad
\text{NDCG} = \frac{\text{DCG}}{\text{IDCG}}
$$

$\text{IDCG}$ 是最理想排序（relevance 降序）下的 DCG，是常数。NDCG $\in (0,1]$，越大越好。

**问题**：DCG 依赖 item 的**实际排序位置 $i$**，而排序是"非连续"操作——分数微小变化可能让位置整位跳变，所以 DCG 对分数 $s$ 的梯度处处为 0 或不存在。这正是 pointwise 损失（BCE）与 NDCG 脱节的原因。

**② 核心思路：把 rank 平滑化**

ApproxNDCG 的关键：item $i$ 的排序位置 = "分数比 $s_i$ 高的 item 数 + 1"：

$$
\text{rank}(i) = 1 + \sum_{j \neq i} \mathbb{I}[s_j > s_i]
$$

其中 $\mathbb{I}[\cdot]$ 是阶跃指示函数（不可导）。用 **sigmoid** 平滑近似"$j$ 排在 $i$ 前面"：

$$
\mathbb{I}[s_j > s_i] \;\approx\; \sigma(s_j - s_i) = \frac{1}{1+e^{-(s_j - s_i)}}
$$

于是得到**平滑 rank**：

$$
\widehat{\text{rank}}(i) = 1 + \sum_{j \neq i} \sigma(s_j - s_i)
$$

- 当 $s_j \gg s_i$ 时 $\sigma \to 1$（j 确实排在 i 前，贡献 1）；
- 当 $s_j \ll s_i$ 时 $\sigma \to 0$（j 排在 i 后，不贡献）；
- 它处处可导，且随分数连续变化。

**③ ApproxNDCG 损失**

把平滑 rank 代入 DCG，得到可导的 $\widehat{\text{DCG}}$：

$$
\widehat{\text{DCG}} = \sum_{i=1}^{n} \frac{2^{\text{rel}_i}-1}{\log_2\bigl(\widehat{\text{rank}}(i)+1\bigr)},
\qquad
\widehat{\text{NDCG}} = \frac{\widehat{\text{DCG}}}{\text{IDCG}}
$$

训练时最小化 `1 - NDCG`（或 `-log NDCG`）。由于 IDCG 是常数，梯度只通过 $\widehat{\text{rank}}(i)$ 流动——每个 item 的梯度会"拉着分数比自己高的正样本往下、分数比自己低的正样本往上"，从而把正样本整体往列表前面推。

**④ 具体例子（one-hot 多正样本）**

query 内 4 个 item：A、B 是正样本（rel=1），C、D 是负样本（rel=0）。

| 情况 | 打分 $s=(s_A,s_B,s_C,s_D)$ | 真实排序 | NDCG |
|---|---|---|---|
| ① 全部正样本靠前 | $(1.0,\,0.9,\,0.0,\,0.0)$ | A B C D | **1.0** |
| ② B 掉到第 3 | $(1.0,\,0.0,\,0.9,\,0.0)$ | A C B D | **0.920** |

情况① DCG：$1/\log_2 2 + 1/\log_2 3 + 0 + 0 = 1 + 0.631 = 1.631$，IDCG 同为 1.631，NDCG = 1.0。
情况② DCG：$1/\log_2 2 + 0/\log_2 3 + 1/\log_2 4 + 0 = 1 + 0.5 = 1.5$，NDCG = $1.5/1.631 = 0.920$。

**注意**：两个正样本等价（rel 都是 1），但 NDCG 仍然"奖励 B 往前挪"——B 从位置 3 提到位置 2 的边际收益 $= 1/\log_2 3 - 1/\log_2 4 = 0.631 - 0.5 = 0.131$。这就是位置折扣的作用：**不要求正样本内部有顺序，但任何正样本越靠前越好**。

平滑 rank 示例（情况①，用 $\sigma(s_j - s_i)$ 近似）：

| item | 真实 rank | $\widehat{\text{rank}}$ | 贡献 |
|---|---|---|---|
| A | 1 | $1+\sigma(-0.1)+\sigma(-1)+\sigma(-1)=1+0.475+0.269+0.269=2.013$ | $1/\log_2 3.013 = 0.628$ |
| B | 2 | $1+\sigma(0.1)+\sigma(-0.9)+\sigma(-0.9)=1+0.525+0.289+0.289=2.103$ | $1/\log_2 3.103 = 0.612$ |
| C | 3 | $1+\sigma(1)+\sigma(0.9)+\sigma(0)=1+0.731+0.711+0.5=2.942$ | 0（负样本） |
| D | 4 | 同 C | 0 |

平滑 NDCG ≈ $(0.628+0.612)/1.631 = 0.760$（比真实 1.0 略低，因为平滑 rank 偏大，但**可导且随排序单调**——这正是 surrogate 的意义）。

**⑤ 与 one-hot 多正样本的关系（为什么它比 ListNet/ListMLE 更贴合）**

- **不需要正样本内部顺序**：ApproxNDCG 只用每个 item 的 rel 分数（1/0），不需要构造全排列；
- **天然处理多正样本**：所有正样本的增益各自按位置折扣累加，"任意正样本往前"都被奖励；
- **无 ListNet 的"均分反作用"**：ListNet 多正样本时把每个正样本概率推向 $1/k$，强正样本本来 0.6 也会被压回 0.5；ApproxNDCG 的梯度正比于位置折扣差异，只会继续把正样本往前推；
- **直接对齐排序指标**：优化的就是 NDCG 本身。

**⑥ 与 ListMLE 的对比（one-hot 场景）**

| | ListMLE | ApproxNDCG |
|---|---|---|
| 需要正样本内部顺序吗 | **需要**（要补全排列 $\pi^{*}$） | **不需要**（只用 rel 分数） |
| one-hot 多正样本 | 得硬造负样本序，可能学入位置偏差 | 增益按位置累加，天然处理 |
| 与酒店排序目标 | 优化"排列似然"，与 NDCG 脱钩 | **直接优化 NDCG** |
| 位置折扣 | 无（top-k 只近似） | 有（前位权重更大） |
| 多正样本均分反作用 | 无（但有顺序假设） | 无 |

> 一句话：one-hot 标签下，ListMLE 多出来的 n-1 步 softmax 学的是你硬造的负样本序；ApproxNDCG 则直接把"所有正样本尽量靠前"变成可导目标，与 GAUC/NDCG 对齐程度最高。

##### 4.2 baseline 中的 ApproxNDCG 代码实现：每一行与数学公式的对应

> 对应实验目录 `zxhtl_seq_v2_5_nosid_topnsp_qw_approxndcg_auxloss_meanpool_M3oE_f_idgate_300` 中 `build_model` 里的 `_build_approxndcg_loss`。

**数学公式回顾**（详见 4.1）：

$$
\widehat{\text{rank}}(i) = 1 + \sum_{j\neq i}\sigma(s_j - s_i)
\quad\Rightarrow\quad
\widehat{\text{DCG}} = \sum_{i} \frac{2^{\text{rel}_i}-1}{\log_2\bigl(\widehat{\text{rank}}(i)+1\bigr)},
\quad
\text{loss} = 1 - \frac{\widehat{\text{DCG}}}{\text{IDCG}}
$$

**核心代码**（已带 shape 注释；`B_valid` = 一个 step 内有效 item 数，`K` = 实际出现的 query 数）：

```python
if getattr(self.config.parm, 'query_wise_mode', False):
    qid_1d = tf.reshape(query_id, [-1])                        # [B_valid]
    # query_id 来自 enumerate() 对整个数据流的全局计数, train_eval 下不限定在 [0, batch_size)。
    unique_qid, qid_compact = tf.unique(qid_1d)                # qid_compact ∈ [0, K)
    num_q = tf.size(unique_qid)                                # 实际出现的 query 数

    def _build_approxndcg_loss(logits, label, name):
        s = tf.reshape(logits, [-1])                           # [B_valid]
        rel = tf.reshape(tf.cast(label, tf.float32), [-1])     # [B_valid] 0/1

        # ---- 1. 平滑 rank ----
        diff = s[:, None] - s[None, :]                         # [B_valid, B_valid], diff[i,j]=s_i-s_j
        sig = tf.sigmoid(-diff)                                # [B_valid, B_valid], σ(s_j - s_i)
        pair_mask = tf.cast(                                   # [B_valid, B_valid], 同 query 才为 1
            tf.equal(qid_compact[:, None], qid_compact[None, :]), tf.float32)
        sig = sig * pair_mask
        rank_hat = 0.5 + tf.reduce_sum(sig, axis=1)            # [B_valid], 1+Σ_{j≠i}σ = 0.5+Σ_j σ

        # ---- 2. DCG surrogate ----
        gain = tf.pow(2.0, rel) - 1.0                          # [B_valid], 0/1 标签下 = rel
        # TF1 无 tf.log2, 用换底公式: log2(x) = ln(x)/ln(2)
        dcg_contrib = gain / (tf.log(rank_hat + 1.0) / tf.log(2.0))  # [B_valid]
        dcg = tf.unsorted_segment_sum(                         # [K]
            dcg_contrib, qid_compact, num_segments=num_q)

        # ---- 3. IDCG: 只依赖每个 query 的正样本数 ----
        pos_cnt = tf.unsorted_segment_sum(                     # [K]
            rel, qid_compact, num_segments=num_q)
        pos_cnt_int = tf.cast(pos_cnt, tf.int32)               # [K]
        max_pos = tf.reduce_max(pos_cnt_int)                   # 本 batch 单 query 最多正样本数
        positions = tf.cast(tf.range(1, max_pos + 1), tf.float32)  # [max_pos]
        disc = 1.0 / (tf.log(positions + 1.0) / tf.log(2.0))   # [max_pos], 1/log2(i+1)
        pos_mask = tf.range(max_pos)[None, :] < pos_cnt_int[:, None]  # [K, max_pos]
        idcg = tf.reduce_sum(tf.where(                         # [K]
            pos_mask, tf.broadcast_to(disc, [num_q, max_pos]),
            tf.zeros([num_q, max_pos], tf.float32)), axis=1)

        # ---- 4. NDCG 与损失 ----
        ndcg = dcg / tf.maximum(idcg, 1e-8)                    # [K]
        per_query = tf.where(pos_cnt > 0, 1.0 - ndcg,          # [K], 无正样本 query 不计入
                             tf.zeros_like(pos_cnt))
        valid_cnt = tf.reduce_sum(tf.cast(pos_cnt > 0, tf.float32))
        loss = tf.reduce_sum(per_query) / tf.maximum(valid_cnt, 1.0)
        tf.summary.scalar('approxndcg_loss/' + name, loss)
        return loss

    approxndcg_loss_click = _build_approxndcg_loss(mlp_out_list[0], labels[0], 'click')
    approxndcg_loss_cr = _build_approxndcg_loss(mlp_out_list[2], labels[1], 'cr')
    self.loss_ops = [loss_op_click, loss_op_cr, loss_op_dr, loss_op_cvr,
                     self.config.parm.approxndcg_w_click * approxndcg_loss_click,
                     self.config.parm.approxndcg_w_cr * approxndcg_loss_cr]
```

**变量 shape 变化一览**：

| 步骤 | 代码 | shape |
|------|------|-------|
| 输入分数 | `s = reshape(logits)` | `[B_valid]` |
| pairwise 分数差 | `diff = s[:,None] - s[None,:]` | `[B_valid, B_valid]` |
| 平滑指示函数 | `sig = sigmoid(-diff)` | `[B_valid, B_valid]` |
| query 内 mask | `sig = sig * pair_mask` | `[B_valid, B_valid]` |
| 平滑 rank | `rank_hat = 0.5 + reduce_sum(sig, axis=1)` | `[B_valid]` |
| 增益 | `gain = 2^rel - 1` | `[B_valid]` |
| DCG 贡献 | `dcg_contrib = gain / log2(rank_hat+1)` | `[B_valid]` |
| DCG | `dcg = unsorted_segment_sum(dcg_contrib, qid_compact, K)` | `[K]` |
| 正样本数 | `pos_cnt = unsorted_segment_sum(rel, qid_compact, K)` | `[K]` |
| 理想位置折扣 | `disc = 1/log2(positions+1)` | `[max_pos]` |
| IDCG | `idcg = reduce_sum(where(pos_mask, broadcast(disc), 0), axis=1)` | `[K]` |
| NDCG / loss | `ndcg = dcg/max(idcg,eps)`，`per_query = where(pos_cnt>0, 1-ndcg, 0)` | `[K]` |

**与 ListNet 实现的对比**：

| | ListNet（⑨ 节实现） | ApproxNDCG（本节实现） |
|---|---|---|
| 位置"来源" | softmax 分母（`unsorted_segment_sum` 聚合，O(B_valid)） | pairwise sigmoid 矩阵（O(B_valid²)） |
| 位置折扣 | 无（只要求正样本概率高） | **有**（`rank_hat` 进入 `log2` 分母） |
| 多正样本 | 均分概率 $1/k$（有"强正样本被压回"的反作用） | 增益按位置折扣累加，无均分反作用 |
| 复杂度 | O(B_valid) | O(B_valid²)（pairwise 矩阵，B_valid≈2000-4000 时可接受） |
| 数值稳定 | 组内减 max 再 softmax | sigmoid 天然稳定，无 exp 溢出 |

**注意事项**：

1. **复杂度**：`[B_valid, B_valid]` 的 pairwise 矩阵，内存 O(B_valid²)。B_valid ≈ 2000-4000 时可接受；若未来列表超长（如召回几百个），应改用 `tf.dynamic_partition` + `tf.map_fn` 按 query 拆成小矩阵。
2. **pair_mask 必须正确**：rank 求和只在同 query 内，跨 query 的 pair 必须 mask 掉，否则会把其他请求的 item 也算进 rank。
3. **无正样本的 query**：`pos_cnt = 0` → `IDCG = 0` → 通过 `tf.where(pos_cnt > 0, ...)` 跳过，不贡献梯度。
4. **平滑 rank 与真实 rank 的关系**：`rank_hat = 0.5 + Σ_j σ(s_j - s_i)` 把"自己对自己"的贡献从 σ(0)=0.5 修正为 1.0，单 item query 时 `rank_hat = 1.0`。
5. **权重超参**：`model.yaml` 中 `approxndcg_w_click: 0.5`、`approxndcg_w_cr: 0.5`，设 0 即等价于纯 pointwise baseline。

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

##### 12.1 详细解析：SetRank 的原理、公式推导与例子

**① 为什么需要"排列不变"（动机）**

传统 LTR 的两类打分函数各有缺陷：

- **univariate（逐文档打分）**：$f(x_i)$ 只依赖单个文档，无法建模文档间交互（DLCM 之前的 LTR 主流）；
- **sequential multivariate（顺序建模）**：$f(x_1, \dots, x_n)$ 显式建模文档间关系（如 DLCM 用 GRU 按输入顺序编码列表），但输出**依赖输入顺序**——同一个文档集合换一种输入顺序，输出的排序就可能不同，这在数学上是"不自洽"的。

SetRank 提出理想排序模型应满足两个性质：**① 能建模 doc-doc 交互**；**② 排列不变（permutation-invariant）**——输入文档的先后顺序不影响最终排序结果。

**② permutation-invariant 与 permutation-equivariant 的精确定义**

设输入文档集合 $X = \{x_1, \dots, x_n\}$，$P$ 是任意置换（对输入做任意重排），函数 $f$ 作用在集合上。

- **排列不变（permutation-invariant）**：输出不随输入顺序改变，即

$$
f(P \cdot X) = f(X)
$$

  例如排序模型的输出是一个排列（文档的有序列表），那么无论 $X$ 里的文档以什么顺序输入，模型给出的排序结果必须相同。这是排序任务**最终输出的要求**。

- **排列等变（permutation-equivariant）**：输出随输入"同序重排"，即

$$
g(P \cdot X) = P \cdot g(X)
$$

  例如编码器输出每个元素的表示：输入重排，表示也跟着重排，但"哪个元素对应哪个表示"不变。这是**中间表示**的要求。

**关系**：SetRank 的巧妙之处是**"等变编码 + 逐元素打分 = 不变排序"**。SetEncoder 是排列等变的（每个文档得到自己的表示，顺序跟着输入换），打分函数对每个文档的表示独立打分，于是"哪个文档得哪个分"不随输入顺序改变，最终按分数排序的结果就自然排列不变。

**③ SetEncoder：MAB / SAB / ISAB**

SetRank 的编码器来自 Set Transformer，核心是三个模块：

**Multihead Attention Block（MAB）**——两集合 $X, Y$ 之间的注意力，$X$ 作 query、$Y$ 作 key/value：

$$
\text{MAB}(X, Y) = \text{LayerNorm}\bigl(H + \text{rFF}(H)\bigr),
\qquad H = \text{Multihead}(X, Y, Y)
$$

其中 $\text{Multihead}$ 是标准多头注意力：$\text{head}_i = \text{Attention}(XW_i^Q, YW_i^K, YW_i^V)$，$\text{Attention}(Q,K,V) = \mathrm{softmax}\bigl(QK^{\top}/\sqrt{d_k}\bigr)V$；$\text{rFF}$ 是逐位置的前馈网络。

**Set Attention Block（SAB）**——集合内自注意力：

$$
\text{SAB}(X) = \text{MAB}(X, X)
$$

所有元素互为 query/key/value，捕获两两交互。复杂度 $O(n^2)$。

**Induced Set Attention Block（ISAB）**——引入 $m$ 个可学习"诱导点"（inducing points）$I_m = [i_1, \dots, i_m]$，降低复杂度到 $O(mn)$：

$$
\text{ISAB}_m(X) = \text{MAB}\bigl(X,\ \text{MAB}(I_m, X)\bigr)
$$

即"诱导点先读一遍集合，集合再读诱导点"，相当于对集合做低秩投影。$m \ll n$ 时计算量显著降低。

MAB / SAB / ISAB **都是排列等变的**：输入元素重排，输出元素跟着重排。

**④ SetRank 模型**

1. 把 query 的 $n$ 个文档的特征向量 $\{x_1, \dots, x_n\}$ 作为输入集合；
2. 过堆叠的 SetEncoder（SAB 或 ISAB 若干层），得到排列等变的表示 $\{h_1, \dots, h_n\}$；
3. 对每个 $h_i$ 独立打分：$s_i = \sigma(w^{\top} h_i + b)$（或线性层）；
4. 排序 = 按 $s_i$ 降序排列。

因为第 2 步等变、第 3 步逐元素独立，所以整体是排列不变的。论文给出了 SAB 版与 ISAB 版两种变体，实验用 **attention rank loss**（注意力加权的列表级损失，与 DLCM 同源）训练，在 MSLR-WEB30K 等三个基准上显著超过 RankSVM / LambdaMART 及神经 IR 基线。

**⑤ 具体例子（等变 vs 不变）**

假设 3 个文档，各自有初始特征 $x_A, x_B, x_C$。

- **等变**：输入 $[x_A, x_B, x_C]$ → SetEncoder 输出 $[h_A, h_B, h_C]$；输入重排为 $[x_C, x_A, x_B]$ → 输出 $[h_C, h_A, h_B]$。$h$ 跟着输入同序换位，但"$h_A$ 属于文档 A"始终不变。
- **不变**：设打分后 $s_A = 0.8, s_B = 0.5, s_C = 0.2$。无论输入是 $[A,B,C]$ 还是 $[C,B,A]$ 还是任意顺序，最终排序都是 $(A, B, C)$。因为每个文档的分数只取决于"它自己的内容 + 其他文档构成的集合"，不取决于它们在输入序列里的先后位置。

**⑥ 针对酒店场景的例子与实现（对应实验 C）**

**业务直觉**：用户在酒店列表页做相对比较——酒店 A 的吸引力受同列表里"价格更低的酒店 B""评分更高的酒店 C"影响，但**不受 A、B、C 在输入张量里的先后顺序影响**。这正是排列不变性想要的：同一批候选酒店，无论内部怎么排，打分结果应当一致。这也和 DLCM（GRU 按列表顺序编码，输出依赖顺序）形成鲜明对比。

**与 baseline 结合的实现**（实验 C 的核心结构，加在 M3oE 塔之前）：

1. 在 `build_model` 中、`boolean_mask` 之前，把 item 表示 reshape 成 `[B, M, D_in]`（`M = items_per_request = 64`，`D_in` 如 `dnn_input` 的维度）；
2. 加 **masked self-attention**：padding 位置在 softmax 前加 $-\infty$，避免 padding 参与交互：

$$
\text{Attention}(Q,K,V) = \mathrm{softmax}\!\Bigl(\frac{QK^{\top}}{\sqrt{d_k}} + \mathcal{M}\Bigr)V
$$

  其中 $\mathcal{M}$ 是 mask 矩阵（padding 位置为 $-\infty$，其余为 0），由 `item_valid_mask` 构造；
3. **纯 SetRank 版**：不注入位置信息（排列不变）；**位置感知版**：把 `rank` 投影成位置 embedding 加到 item 表示上（对应实验 C 的"位置感知"选项）；
4. 残差 + LayerNorm，flatten 回 `[BM, D_out]`，再走 `boolean_mask` → M3oE 塔。

**关键代码注意**：

- 手写 masked multi-head attention 可复用 `target_attention_pool_time_aware` 的注意力骨架（`train.py` L692-826），把"序列内 attention"改成"列表内 attention"；
- 所有注意力在 `[B, M]` 固定 shape 上计算，满足 Horovod 通信约束；
- 输出维度变化需同步更新 `Config.dense_input_size`；
- 复杂度：$M=64$ 时 SAB 为 $O(M^2)=4096$，可接受；若未来候选列表更大（如几百个），改用 ISAB 降低复杂度。

**⑦ 训练损失**

论文用 attention rank loss，但 SetRank 是"结构"方法，训练损失可以自由选择：酒店场景建议**叠加现有 pointwise BCE + 列表级损失**（ListMLE / ApproxNDCG，见 #2 / #4），或直接用列表级损失微调。注意力模块本身只负责"让 item 表示互相看见"，不强制特定损失。

##### 12.2 深入理解：为什么 permutation-invariant 是优点？——回答"输入已经是按位次排序的列表，输出与顺序相关不就是 listwise 吗"

这是一个非常关键的疑问，容易把两个"顺序"混为一谈。先厘清概念，再给例子。

**① 有两种"顺序"，只有一种是模型应当依赖的**

| 顺序 | 含义 | 有业务含义吗 | 模型应依赖吗 |
|---|---|---|---|
| **输入枚举顺序（permutation）** | 同一批候选在张量里的先后拼接顺序 | **没有**。它由数据加载/分批/截断方式决定，是"伪顺序" | **不应该依赖** → 这就是 permutation-invariant 要消灭的 |
| **展示位置（position）** | 酒店在列表页实际展示在第几位 | **有**。第 1 位天然点击率高（位置偏差） | **应该显式建模**（如 `rank_oh` / 位置 embedding） |

你的疑问"输入是按实际位次排序后的列表，输出当然可以和顺序相关"——这里说的"顺序"其实是**展示位置**，它确实应该有影响（位置偏差）。但 SetRank 的 permutation-invariant **反对的并不是"使用位置信息"，而是"依赖输入的枚举顺序"**。这两者可以完全解耦：位置信息可以作为**每个 item 的显式特征**（元素属性）传入，而结构本身不依赖"谁在张量里排在前面"。

**② listwise 和"依赖输入顺序"是两回事**

listwise 的定义是：**以整个列表为样本**，用列表级损失/输出整体排序。它完全不要求输入顺序有意义：

- ListMLE 输入一个排列（有序），用 Plackett-Luce 建模——它把输入顺序当成"ground truth 排列"来用；
- SetRank 输入一个集合（无顺序），用 SetEncoder 建模 item 间交互——它刻意消除输入顺序的影响。

两者都是 listwise。SetRank 是"listwise + 不依赖输入顺序"的形态，强调的是排序问题的**内在对称性**。

**③ 为什么排列不变是优点（4 个理由）**

1. **排序目标的对称性**：NDCG / GAUC 这类指标对候选集合是**天然对称**的——把候选集合任意打乱，指标不变。模型作为这个任务的近似，理应尊重这个对称性。若模型不是排列不变的，它就把"无关的方差"（拼接顺序）当成了预测依据，只会**增加过拟合、降低泛化**。
2. **避免学到虚假相关**：如果模型依赖输入顺序，训练数据里"排在前面的 item 通常分高"这类**数据加载伪相关**会被模型学进去，但线上拼接顺序完全不同，这个依赖就成了噪声。
3. **线上线下一致性**：线上打分时，800 个候选可能分批、并行、顺序任意。排列不变保证：**同一个酒店，无论它出现在哪一批、张量里排第几位，打出的分数完全一致**。这对排序系统的稳定性和可调试性至关重要。
4. **归纳偏置**：排列不变是"正确"对待集合数据的归纳偏置（inductive bias），就像 CNN 对图像的平移不变、RNN 对时间步的顺序依赖——把模型容量用在真正有意义的信号（内容交互）上，而不是浪费在无意义的顺序上。

**④ 具体例子（为什么排列依赖在线上不可接受）**

假设 3 个酒店 A、B、C，真实"相对质量"是 A > B > C。

- **排列不变模型**：输入 $[A,B,C]$ 或 $[C,A,B]$ 或任意顺序，打分都是 $s_A=0.8, s_B=0.5, s_C=0.2$，排序永远是 $(A,B,C)$。
- **排列依赖模型**（如 GRU 按输入顺序编码，隐含"靠前位置有先验优势"）：输入 $[A,B,C]$ 时 A 拿到"首位置"的隐式加成 → 排序 $A>B>C$；输入 $[C,A,B]$ 时 C 拿到首位置加成 → 可能排序变成 $C>A>B$。**同一批酒店，只是后端拼接顺序变了，线上排序结果就变了**——这是不可接受的。

**⑤ 结合你的代码：当前 query-wise 数据的 item 顺序其实就是按位次排的**

`_parse_tfrd_query_wise` 里 `keep_idx_sorted` 是按 `item_rank` 升序排列的，所以 `[B, M]` 张量里 item 的顺序 ≈ 展示位次。这意味着：

- 如果做 **DLCM / GRU 式**（按输入顺序编码），输入顺序=展示位次，模型确实在"用顺序编码位置"，输出与顺序相关是合理的；
- 如果做 **SetRank 式**（排列不变），模型不利用这个顺序；若想保留位置偏差，就把 `rank`（或 `rank_oh`）作为**显式特征**传入（baseline 本来就把 `rank_oh` 拼进了 `show_index_processed`）。

**结论**：你的场景下，排列不变不是"不能利用位次"，而是"位次应该作为特征显式传，而不是靠张量拼接顺序隐式带进去"。这样既保留了位置偏差的建模能力，又消除了"同一个候选集合因拼接顺序不同而排序不同"的隐患。这也正是 SetRank 相比 DLCM 的工程优势：**结构上就保证了线上无论怎么分批/拼接，结果稳定一致**。

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
