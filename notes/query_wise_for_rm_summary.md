# 推荐场景下的 Pair-wise / List-wise 损失函数调研与实验设计

> 面向场景：智行酒店推荐精排，从 pointwise 打分向 pairwise/list-wise 建模范式演进的调研。
> 与 `query_wise_summary.md`（搜索场景为主的 LTR 综述）互补，本文聚焦**推荐系统 / CTR 预估**场景下，以**损失函数改进**为主的 pairwise/list-wise 业界落地与研究。
> Baseline 模型：`zxhtl_seq_v2_5_nosid_topnsp_qw_meanpool_M3oE_f_idgate_300`（query-wise 数据管线 + pointwise 打分）

---

## 目录

- [1. 核心论文详解：腾讯 Understanding the Ranking Loss for Recommendation with Sparse User Feedback](#1-核心论文详解腾讯-understanding-the-ranking-loss-for-recommendation-with-sparse-user-feedback)
- [2. 引用网络：推荐/CTR 场景的 pairwise/list-wise 损失谱系](#2-引用网络推荐ctr-场景的-pairwiselist-wise-损失谱系)
  - [2.1 RankNet](#21-ranknet)
  - [2.2 Combined-Pair](#22-combined-pair)
  - [2.3 Combined-List](#23-combined-list)
  - [2.4 RCR（精简）](#24-rcr精简)
  - [2.5 JRC（精简）](#25-jrc精简)
- [3. 关键问题回答：pair-wise / list-wise 损失是否必须针对同一请求内？](#3-关键问题回答pair-wise--list-wise-损失是否必须针对同一请求内)
- [4. InfoNCE 损失：对比学习在推荐/CTR 的标准做法](#4-infonce-损失对比学习在推荐ctr-的标准做法)
- [5. Pairwise 在推荐/CTR 的业界落地](#5-pairwise-在推荐ctr-的业界落地)
- [6. 与 baseline 结合的建议](#6-与-baseline-结合的建议)
- [7. 参考链接汇总](#7-参考链接汇总)

---

## 1. 核心论文详解：腾讯 Understanding the Ranking Loss for Recommendation with Sparse User Feedback

- **链接**：[arXiv:2403.14144](https://arxiv.org/abs/2403.14144)（KDD 2024）
- **作者**：Zhutian Lin, Junwei Pan, Shangyu Zhang, Ximei Wang, Xi Xiao, Shudong Huang, Lei Xiao, Jie Jiang（腾讯）
- **一句话**：系统分析了"在稀疏正反馈下，BCE 单独训练的负样本梯度消失问题"，提出在 BCE 之上叠加辅助排序损失（Combined-Pair / Combined-List / Combined-Contrastive），并在腾讯广告两个场景线上 GMV 提升 0.70% 与 1.26%。

### 1.1 背景与动机

**CTR 预估的标准做法是 pointwise BCE**：把每个 (user, item) 对当作独立二分类样本，正样本=点击，负样本=曝光未点击。但论文指出一个被忽视的问题：

> **在稀疏正反馈场景下，BCE 对负样本的梯度会消失，导致模型"学不动"负样本。**

为什么？BCE 对负样本（$y=0$）的梯度为：

$$
\nabla_{z_i} L_{\text{BCE}} = \sigma(z_i) - y_i = \sigma(z_i)
$$

即负样本的梯度大小 = 它的预测点击率 pCTR = $\sigma(z_i)$。当正样本很稀疏时，模型很快学会把绝大多数样本预测为低点击率（$\sigma(z_i) \approx 0$），于是负样本的梯度 $\approx 0$——模型对"哪些负样本更接近正样本、需要被进一步压低"失去区分能力。这就是**负样本梯度消失**。

### 1.2 方法框架：BCE + 辅助排序损失

论文的框架非常简洁——**不替换 BCE，而是叠加一个辅助排序损失**：

$$
\mathcal{L} = \mathcal{L}_{\text{BCE}} + \alpha \cdot \mathcal{L}_{\text{Rank}}
$$

关键洞察：**排序损失对负样本产生更大的梯度**（见 1.4 梯度分析），从而缓解 BCE 的负样本梯度消失，同时不破坏 BCE 的校准能力（输出仍是可解释的点击率）。

论文重点对比了三种已有的"BCE+排序损失"组合，并提出一个新方法 Combined-Contrastive：

| 方法 | 辅助排序损失 | 引用 | 梯度特点 |
|------|------------|------|---------|
| Combined-Pair | pairwise（RankNet 式） | [Li et al. 2015](#22-combined-pair) | 负样本梯度 $\propto \sum \sigma(z^- - z^+)$ |
| Combined-List | listwise（ListNet 式） | [Yan et al. 2022](#23-combined-list) | 负样本梯度 $\propto \mathrm{softmax}(z^-)$ |
| JRC | listwise（双 logit 解耦） | [Sheng et al. 2023](#25-jrc精简) | 对非点击 logit 产生梯度 |
| **Combined-Contrastive** | contrastive（论文新提出） | 本文 | 类似 InfoNCE 的对比结构 |

### 1.3 各组合损失的公式

**① BCE（基准）**：

$$
\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N}\Bigl[y_i \log\sigma(z_i) + (1-y_i)\log\bigl(1-\sigma(z_i)\bigr)\Bigr]
$$

**② Combined-Pair**（BCE + pairwise rank loss）：

$$
\mathcal{L}_{\text{Rank}}^{\text{CP}} = \frac{1}{N^{+}N^{-}}\sum_{i \in N^{+}}\sum_{j \in N^{-}} \log\bigl(1 + e^{-(z_i - z_j)}\bigr)
$$

其中 $N^{+}$ 是正样本集合、$N^{-}$ 是负样本集合。这是 RankNet 式 pairwise logistic loss 的 batch 化版本（对 batch 内所有正负样本对求平均，**不需要同一请求**，见第 3 节）。

**③ Combined-List**（BCE + ListNet loss）：

$$
\mathcal{L}_{\text{Rank}}^{\text{List}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{z_i}}{\sum_{k=1}^{N} e^{z_k}}
$$

这就是把整个 batch 当成一个"列表"，对每个正样本位置做 softmax cross-entropy（ListNet top-one one-hot 版，对应 `query_wise_summary.md` 的 #1）。

**④ JRC**（decoupled 双 logit）：

$$
\mathcal{L}_{\text{Rank}}^{\text{JRC}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{z_i y_i}}{\sum_{k=1}^{N} e^{z_k y_k}}
$$

JRC 把 logit 解耦为"点击状态"和"非点击状态"两个 logit，用 $z_i y_i$ 的乘积形式让正样本的分数参与 softmax、负样本的贡献按标签加权（详见 2.5）。

**⑤ Combined-Contrastive**（论文新提出）：

$$
\mathcal{L}_{\text{CC}} = \alpha\, \mathcal{L}_{\text{BCE}} + (1-\alpha)\, \mathcal{L}_{\text{Contr}}
$$

其中对比项 $\mathcal{L}_{\text{Contr}}$ 采用 InfoNCE 风格：正样本（点击）作为 anchor，负样本（未点击）作为负例，让正样本的表示靠近、负样本远离（详见第 4 节 InfoNCE）。

### 1.4 梯度分析（论文核心理论贡献）

论文的核心不是提出新结构，而是**从梯度角度统一解释"为什么 BCE+排序损失有效"**。

- **BCE 的负样本梯度**：$\nabla_{z_j^{-}} \mathcal{L}_{\text{BCE}} = \sigma(z_j^{-}) \to 0$（稀疏正反馈下负样本 pCTR 很小 → 梯度消失）。
- **Combined-Pair 的负样本梯度**：

$$
\nabla_{z_j^{-}} \mathcal{L}_{\text{Rank}}^{\text{CP}} = \frac{1}{N^{+}}\sum_{i \in N^{+}} \sigma(z_j^{-} - z_i^{+})
$$

  负样本梯度 = 它对所有正样本的"相对分数差"的 sigmoid 之和。即使 $z_j^{-}$ 很小，只要它和某个正样本 $z_i^{+}$ 的差还不足以完全分开，梯度就非零——**对负样本始终有梯度**，缓解了消失问题。

- **Combined-List / JRC 的负样本梯度**：$\propto \mathrm{softmax}(z_j^{-})$（ListNet 式）或对非点击 logit 产生梯度（JRC 式），同样避免随 pCTR 趋零而消失。

**理论结论**：正样本越稀疏，加排序损失的提升越大。论文用不同正样本占比做了实验验证这一趋势。

### 1.5 实验与线上效果

- **离线**：在公开数据集上，BCE+排序损失相对纯 BCE 在 AUC 上一致提升；正样本越稀疏提升越大。
- **线上**：在腾讯广告两个主要场景集成 Combined-Pair（或 Combined-Contrastive），GMV 分别提升 **0.70%** 和 **1.26%**。
- **代码**：论文附有开源实现（GitHub），便于复现。

### 1.6 如何适用于我的酒店场景

**直接可借鉴的点**：

1. **当前 baseline 是纯 pointwise BCE**（`loss_op_click` / `loss_op_cr` / `loss_op_cvr`），与论文的出发点完全一致——正样本（点击/下单）稀疏，负样本梯度消失问题在酒店场景同样存在。叠加一个辅助排序损失是**侵入性最小、最贴近工业落地**的下一步。
2. **论文的梯度分析给了我们一个诊断工具**：如果你的模型负样本 pCTR 普遍很低（如 <0.01），那 BCE 的负样本梯度几乎消失，排序损失大概率有效；如果 pCTR 分布还比较健康，收益可能有限。
3. **Combined-Pair 的实现最简洁**（详见代码实现），而且**不需要 query_id 分组**（见第 3 节）——在现有 `[B_valid]` 展平数据上直接对正负样本对求 loss 即可，改动极小。

**代码实现（Combined-Pair，TF1，query-wise 下直接用 B_valid 全局配对）**：

```python
# 在 build_model 的 loss 区块中, 基于 mlp_out_list[0] (click logit) 或 mlp_out_list[2] (cr logit)
def _build_combined_pair_loss(logits, label, name):
    """logits/label: [B_valid, 1] -> 全局正负样本对 pairwise loss (不需要 query_id 分组)"""
    z = tf.reshape(logits, [-1])                            # [B_valid]
    y = tf.reshape(tf.cast(label, tf.float32), [-1])        # [B_valid] 0/1

    pos_z = tf.boolean_mask(z, tf.greater(y, 0.5))          # [N+]
    neg_z = tf.boolean_mask(z, tf.less(y, 0.5))             # [N-]

    # pairwise logistic loss: log(1 + exp(-(z+ - z-)))
    # 数值稳定: softplus(-(z+ - z-))
    diff = tf.expand_dims(pos_z, 1) - tf.expand_dims(neg_z, 0)   # [N+, N-]
    pair_loss = tf.nn.softplus(-diff)                            # [N+, N-]
    loss = tf.reduce_mean(pair_loss)                             # 标量
    tf.summary.scalar('combined_pair_loss/' + name, loss)
    return loss

combined_pair_click = _build_combined_pair_loss(mlp_out_list[0], labels[0], 'click')
combined_pair_cr = _build_combined_pair_loss(mlp_out_list[2], labels[1], 'cr')
# self.loss_ops += [w_pair_click * combined_pair_click, w_pair_cr * combined_pair_cr]
```

**注意事项**：
- 正样本集合 $N^{+}$ 与负样本集合 $N^{-}$ 的笛卡尔积 $N^{+} \times N^{-}$ 可能很大（如 $N^{+}=200, N^{-}=1800$ → 36 万对），可用采样控制（如每个正样本采样 K 个负样本，K 取 4~20）。
- 权重 $\alpha$（`w_pair_click` / `w_pair_cr`）建议从 0.1 起调（参考腾讯线上 0.70%/1.26% 的量级，辅助损失权重通常很小）。
- 若只想对"同一请求内"的 pair 建模（列表内排序），则需用 `query_id`/`dispatch_id` 分组，见第 3 节与 6 节。

---

## 2. 引用网络：推荐/CTR 场景的 pairwise/list-wise 损失谱系

腾讯这篇论文把 CTR 场景的"BCE+排序损失"谱系梳理得很清楚。本节按时间线详解这些方法（详略安排：Combined-Pair / RankNet / Combined-List 详细，RCR / JRC 精简）。

### 2.1 RankNet

- **链接**：ICML 2005，[Learning to Rank using Gradient Descent](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/)（Burges et al.）
- **核心思想**：pairwise 排序的经典之作。对一对文档 $(i,j)$（$i$ 比 $j$ 更相关），定义"$i$ 排在 $j$ 前的概率"：

$$
P_{ij} = \frac{1}{1 + e^{-(s_i - s_j)}}
$$

- **损失**：用交叉熵衡量"预测概率 $P_{ij}$"与"目标 $P_{ij}^{*}$（通常 1/0）"的差距：

$$
L = -P_{ij}^{*}\log P_{ij} - (1 - P_{ij}^{*})\log(1 - P_{ij})
$$

  等价于 `softplus(-(s_i - s_j))`（当目标 $P_{ij}^{*}=1$ 时）。

- **与 CTR 的关系**：RankNet 原生于搜索排序，但它的 pairwise logistic loss 是 Combined-Pair 的直接前身。腾讯论文中 Combined-Pair 的梯度 $\nabla = \frac{1}{N^{+}}\sum_{i}\sigma(z_j^- - z_i^+)$ 正是 RankNet 损失对负样本的梯度形式。
- **局限性**：RankNet 只保证"相对序"，输出分数不是校准的点击率（这也是为什么工业界用 BCE+排序损失的组合而非纯 RankNet）。

### 2.2 Combined-Pair

- **链接**：Li et al. 2015（Twitter 时间线广告），后被广泛引用为 Combined-Pair 来源。
- **核心思想**：在 pointwise BCE 之上叠加 pairwise ranking loss，**同时获得校准能力（BCE）与排序能力（pairwise）**：

$$
\mathcal{L} = \mathcal{L}_{\text{BCE}} + \alpha \cdot \mathcal{L}_{\text{Pairwise}}
$$

- **公式**（batch 化全局配对）：

$$
\mathcal{L}_{\text{Pairwise}} = \frac{1}{N^{+}N^{-}}\sum_{i \in N^{+}}\sum_{j \in N^{-}} \log\bigl(1 + e^{-(z_i - z_j)}\bigr)
$$

- **关键性质**（腾讯论文 1.4 分析的核心）：
  - BCE 负样本梯度 $\propto \sigma(z^-) \to 0$（梯度消失）；
  - Combined-Pair 负样本梯度 $\propto \frac{1}{N^{+}}\sum_{i}\sigma(z^- - z_i^+) \neq 0$（始终有梯度，除非 $z^-$ 已经比所有正样本小很多）。
- **落地**：被 Twitter / Google / Alibaba 等广泛采用。**注意：它是"全局配对"，不要求正负样本来自同一请求**（详见第 3 节）。

### 2.3 Combined-List

- **链接**：Yan et al. 2022（BCE + ListNet）。
- **核心思想**：把 pairwise 换成 listwise（ListNet top-one one-hot 版），对整个 batch 做 softmax 交叉熵：

$$
\mathcal{L}_{\text{List}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{z_i}}{\sum_{k=1}^{N}e^{z_k}}
$$

- **与 Combined-Pair 的对比**：
  - 两者都能给负样本非零梯度；
  - List 版把"整个 batch"当成一个列表，正样本被推向 softmax 概率高；Pair 版逐对比较。
  - 腾讯论文实验显示：Combined-Pair 通常略优于 Combined-List（pairwise 的逐对比较更细粒度），但 List 版实现更简单。
- **注意**：这里的 ListNet 是在 **batch 级**做的（把 batch 当列表），**不是 query 内**——这与搜索场景的"query 内列表"不同，是 CTR 场景的常见简化（见第 3 节）。

### 2.4 RCR（精简）

- **链接**：Bai et al. 2023（Google），Regression-Compatible Ranking。
- **核心思想**：让排序损失的**最小值与 BCE 的最小值对齐**（regression-compatible），避免"排序损失的最优解 ≠ 校准的点击率"。
- **做法**：用 `ListCE`（ListNet 的变体，对 logit 做"兼容"变换）作为辅助损失，使模型既能排序又能保持校准。被 Google 采用。
- **与腾讯论文的关系**：腾讯论文把 RCR 归入"BCE+List 式排序损失"一族，梯度分析与 Combined-List 类似。

### 2.5 JRC（精简）

- **链接**：Sheng et al. 2023（KDD 2023，Alibaba），Joint Ranking and Calibration。
- **核心思想**：把 logit **解耦成"点击 logit"与"非点击 logit"两个**，用减法控制校准、用对比控制排序：

$$
\mathcal{L}_{\text{Rank}}^{\text{JRC}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{z_i y_i}}{\sum_{k=1}^{N}e^{z_k y_k}}
$$

- **关键性质**：$z_i y_i$ 使**正样本**的分数进 softmax 分子、负样本按标签加权；对"非点击 logit"单独产生梯度，缓解梯度消失。在阿里展示广告落地，同时优化排序与校准。
- **与腾讯论文的关系**：腾讯论文把 JRC 作为"listwise-like"的一族，梯度分析同样适用。

---

## 3. 关键问题回答：pair-wise / list-wise 损失是否必须针对同一请求内？

**你的问题**：pairwise/listwise 损失是否必须针对一次请求内？比较正负样本 pair 时，是否要保证来自同一请求？我的代码是否要用 query_id 分组？

**直接回答：不一定。是否分组取决于你建模的目标（"比较单元"是什么），而不是 pairwise/listwise 本身。**

### 3.1 三个层次的"比较单元"

| 比较单元 | 典型方法 | 是否要求同一请求 |
|---------|---------|----------------|
| **Query（搜索）** | RankNet、ListNet、ListMLE（搜索 LTR） | **要求**：relevance 是 query-doc 属性，只有同 query 的 doc 才能比较 |
| **User（推荐）** | BPR（per-user 正负采样） | **不要求同一请求**：正样本=用户交互过的 item，负样本=采样 |
| **Batch / 全局（CTR）** | Combined-Pair、Combined-List（腾讯） | **不要求**：正样本=点击，负样本=曝光未点击/采样，全局配对 |

### 3.2 为什么 CTR/推荐场景通常不要求同一请求

搜索场景的 pairwise/listwise 必须同 query，是因为**相关性是 query-doc 对**——不同 query 的文档之间没有可比性。但推荐/CTR 场景：

1. **目标是"用户更可能点击/下单的 item 排前面"**，正负样本的语义是"用户偏好"，不需要同一次请求内的上下文；
2. **同一请求内往往没有足够多的正负样本**（比如一次请求只有几个正样本），全局/用户级配对能利用更多训练信号；
3. 腾讯 Combined-Pair 的梯度公式 $\nabla = \frac{1}{N^{+}}\sum_{i}\sigma(z^- - z_i^+)$ 是**对 batch 内所有正样本**求和的全局配对——工业 CTR 标准做法。

### 3.3 什么情况下才需要 query_id 分组

只有当你**明确想优化"同一请求/列表内的相对排序"**时才需要分组，典型场景：

- **列表重排（reranking）**：想让"用户在当前请求返回的候选列表里"点击/下单的酒店排前面——此时同一请求内的 item 共享展示上下文，应分组（`query_id`）；
- **与 GAUC 口径对齐**：如果你的评估指标是按 `dispatch_id` 分组算 GAUC，那训练损失按同口径分组更一致（这就是之前把 listnet 分组 key 改成 `dispatch_id` 的原因）。

### 3.4 对你的代码的具体结论

| 你想要的损失 | 是否用 query_id 分组 |
|-------------|---------------------|
| **CTR 全局排序**（正样本整体比负样本高，类似腾讯 Combined-Pair） | **不需要**。在 `[B_valid]` 上直接取 `pos_z` / `neg_z` 全局配对（见 1.6 代码） |
| **列表内排序**（同一请求候选列表内比较，重排目标） | **需要**。用 `query_id`（或 `dispatch_id`）分组后组内配对 |
| **InfoNCE**（对比学习，见第 4 节） | **不需要**。通常全局/用户级采样负样本 |

**建议**：酒店场景先做**全局 Combined-Pair**（侵入最小、贴合 CTR），若目标是"列表内相对排序"再做 query_id 分组的列表级损失（可参考 `query_wise_summary.md` 中 listnet 的 `tf.unique` 分组实现）。

---

## 4. InfoNCE 损失：对比学习在推荐/CTR 的标准做法

### 4.1 定义与公式

InfoNCE（Information Noise-Contrastive Estimation）来自 [Oord et al. 2018《Contrastive Predictive Coding》](https://arxiv.org/abs/1807.03748)。核心思想：给定一个 anchor（上下文 $c_t$），从候选集合中**找出那个真正的正样本**，其余是负样本：

$$
\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\Bigl[\log \frac{f(x^{+}, c)}{\sum_{x_j \in \{x^{+}\} \cup \mathcal{N}} f(x_j, c)}\Bigr]
$$

其中 $f(x, c) = e^{s(x, c)}$（用相似度分数的指数形式），分母对正样本 + 一组负样本求和。**最小化 InfoNCE = 最大化 anchor 与正样本的互信息下界**，等价于一个 $(|\mathcal{N}|+1)$-way softmax 分类。

### 4.2 为什么说它是推荐场景的标准做法

InfoNCE 在推荐/CTR 的"标准"地位来自它与几个经典方法的同构关系：

| 方法 | 结构 | 与 InfoNCE 的关系 |
|------|------|------------------|
| **Sampled Softmax**（YouTube/Meta） | $-\log \frac{e^{s(u,i^+)}}{\sum_{j \in \text{sample}} e^{s(u,j)}}$ | **完全同构**：就是 InfoNCE 把正样本+采样负样本做 softmax |
| **NCE**（Noise Contrastive Estimation） | 区分"真实对"与"噪声对" | InfoNCE 是 NCE 的"多分类"版本 |
| **BPR**（pairwise） | $-\log\sigma(s(u,i^+) - s(u,j^-))$ | InfoNCE 是 BPR 的"多负样本"扩展（softmax 归一化版） |
| **ListNet top-one**（listwise） | $-\log \frac{e^{z^+}}{\sum_k e^{z_k}}$ | 结构一致，但分母范围不同 |

**一句话**：InfoNCE ≈ "Sampled Softmax" ≈ "多负样本版 BPR" ≈ "单正样本版 ListNet"，区别只在**分母包含哪些负样本**。这就是为什么它在推荐中"标准"——它把 pair-wise 的单负样本比较推广到了多负样本、归一化的 softmax 比较。

### 4.3 在推荐/CTR 中的标准用法

1. **双塔/向量召回**：用户塔 $u$、物品塔 $v$，正样本=用户交互过的 item，负样本=in-batch 采样或全局采样，用 InfoNCE 拉近正对、推开负对（YouTube 双塔、各类向量召回的标准损失）。
2. **序列推荐**：对比学习做自监督增强（如 SASRec 的负采样）。
3. **作为辅助损失**：腾讯 Combined-Contrastive 就是在 BCE 上叠加 InfoNCE 式对比损失，利用对比项的"负样本恒有梯度"性质缓解梯度消失。

### 4.4 与腾讯论文 Combined-Contrastive 的关系

腾讯的 `L_CC = α·L_BCE + (1-α)·L_Contr`，其 $L_{\text{Contr}}$ 采用 InfoNCE 风格（anchor 为正样本，负样本为未点击）。论文认为 Combined-Contrastive 与 Combined-Pair 在"给负样本非零梯度"这一点上殊途同归，且对比学习的表示空间更利于区分难负样本。

### 4.5 代码实现建议（TF1，酒店场景）

```python
def _build_infonce_loss(anchor, pos, neg, name):
    """anchor/pos: [B, D]; neg: [B, N, D] -> InfoNCE 损失(与双塔/序列模型配合)"""
    # 相似度
    pos_sim = tf.reduce_sum(anchor * pos, axis=-1)                 # [B]
    neg_sim = tf.matmul(anchor, neg, transpose_b=True)            # [B, N]
    logits = tf.concat([tf.expand_dims(pos_sim, -1), neg_sim], axis=-1)  # [B, N+1]
    labels = tf.zeros([tf.shape(logits)[0]], dtype=tf.int32)      # 正样本在位置0
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels))
    tf.summary.scalar('infonce_loss/' + name, loss)
    return loss
```

> 注意：InfoNCE 通常用于**学习表示**（双塔/序列），而不是直接优化精排分数。在精排阶段更常见的是 Combined-Pair / Combined-List；InfoNCE 作为辅助损失时，需要先有"表示"（如双塔中间层输出），再在其上做对比。

---

## 5. Pairwise 在推荐/CTR 的业界落地

### 5.1 BPR（Bayesian Personalized Ranking）

- **链接**：[Rendle et al. 2009](https://arxiv.org/abs/1205.2618)（UAI 2009）
- **公式**（per-user 配对）：

$$
\mathcal{L}_{\text{BPR}} = -\sum_{(u,i,j)} \log\sigma\bigl(\hat{x}_{ui} - \hat{x}_{uj}\bigr)
$$

  对每个用户 $u$，正样本 $i$（交互过）与负样本 $j$（采样未交互）配对。
- **落地**：矩阵分解/embedding 召回的标准损失，被大量工业推荐系统采用（尤其在召回/粗排阶段）。
- **与精排的区别**：BPR 输出的是"相对分数"，不是校准点击率；精排场景通常要保留 BCE 的校准能力，因此用 Combined-Pair（BCE+pairwise）而非纯 BPR。

### 5.2 YouTube / Meta：Sampled Softmax 与 SBC

- **Sampled Softmax**（YouTube 视频推荐、双塔）：用 in-batch 负样本做 softmax，等价 InfoNCE。
- **SBC（Sampling-Bias-Corrected，[RecSys 2019](https://arxiv.org/abs/1906.07591)）**：指出 in-batch softmax 对热门 item 过度惩罚（采样偏差），提出用流式频率估计修正 logits（减去 $\log p_j$ 校正项）。**这是 pairwise/softmax 类损失在大规模推荐落地的关键工程细节**——负采样偏差校正比损失函数本身更影响效果。

### 5.3 大厂 CTR 精排的 pairwise/listwise 组合实践

| 公司 | 做法 | 阶段 |
|------|------|------|
| **Twitter** | Combined-Pair（pointwise+pairwise） | 广告精排 |
| **Google** | RCR（BCE+ListCE，校准兼容） | 精排 |
| **Alibaba** | JRC（双 logit 解耦 + listwise-like） | 展示广告精排 |
| **Tencent** | Combined-Pair / Combined-Contrastive | 广告精排，GMV +0.70%/1.26% |
| **Meta/YouTube** | Sampled Softmax / SBC（in-batch 负采样） | 召回/双塔 |

**共性**：都是在 pointwise BCE（保校准）之上叠加一个排序损失（保排序），且通常**全局/用户级配对而非 query 内配对**。负采样策略（比例、难度、偏差校正）是决定效果的关键工程点。

### 5.4 负采样策略要点

- **比例**：常见 1:3 ~ 1:10（正:负），pre-ranking 常用 1:4；
- **来源**：曝光未点击（最真实）、全局随机采样、in-batch 采样（配 SBC 校正）；
- **难度**：随机负样本 → 难负样本（hard negative，与正样本相似但未点击）能显著提升区分能力，但过难会导致训练不稳定。

---

## 6. 与 baseline 结合的建议

基于以上调研，给酒店精排的落地建议（由简到繁）：

| 优先级 | 方法 | 改动 | 预期收益 |
|--------|------|------|---------|
| P0 | **全局 Combined-Pair**（BCE+pairwise） | 只加一个辅助损失（1.6 代码），`[B_valid]` 全局配对，负采样 K=4~10，权重 α 从 0.1 起 | 缓解负样本梯度消失，AUC/GAUC 提升；**腾讯验证的工业标准做法** |
| P1 | **InfoNCE 辅助损失** | 需中间层表示（双塔/序列），在精排塔上叠加 | 表示更可分；与 Combined-Pair 互补 |
| P1 | **Combined-List**（batch 级 ListNet） | 替换辅助损失为 ListNet 式 | 实现更简单，可作对照 |
| P2 | **列表内排序**（query_id 分组） | 用 `query_id`/`dispatch_id` 分组做 pair/list | 若目标是"列表内相对排序/GAUC" |
| P2 | **JRC 式双 logit** | 结构改动大 | 排序+校准联合优化 |

**核心建议**：先做 P0（全局 Combined-Pair），它侵入最小、直接对应当前 baseline 的稀疏正反馈痛点，且有腾讯线上验证。负采样比例和偏差校正是重点调参项。

---

## 7. 参考链接汇总

| # | 论文/方法 | 链接 |
|---|----------|------|
| 1 | Understanding the Ranking Loss for Recommendation with Sparse User Feedback（腾讯，KDD 2024） | https://arxiv.org/abs/2403.14144 |
| 2 | RankNet: Learning to Rank using Gradient Descent（Burges et al., ICML 2005） | https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/ |
| 3 | Contrastive Predictive Coding（Oord et al., 2018，InfoNCE 来源） | https://arxiv.org/abs/1807.03748 |
| 4 | BPR: Bayesian Personalized Ranking from Implicit Feedback（Rendle et al., 2009） | https://arxiv.org/abs/1205.2618 |
| 5 | Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations（SBC，RecSys 2019） | https://arxiv.org/abs/1906.07591 |

> 注：Combined-Pair（Li et al. 2015）、Combined-List（Yan et al. 2022）、RCR（Bai et al. 2023）、JRC（Sheng et al. 2023）为论文引用网络中的方法，本文按腾讯论文及检索信息归纳；若需精确定位原文 DOI，建议以腾讯论文参考文献列表为准。

