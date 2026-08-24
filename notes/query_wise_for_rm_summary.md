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

#### 1.1.1 深度解读：为什么负样本梯度"必然"消失？——论文原话逐句拆解

你贴的这段话是论文 Finding 3 的完整论证，它比"负样本梯度 ∝ pCTR"更进一步：**即使模型是完美的无偏估计器，负样本梯度也会因为数据稀疏而天然很小**。这不是训练失败，而是数据统计特性决定的。逐句拆解：

> 原文：*"The gradients of negative samples are proportional to its pCTR value, $\hat{p}_j$. The expected value of $\hat{p}_j$ produced by an unbiased CTR estimation model with BCE loss is close to the underlying global CTR, which equals approximately to the proportion of click samples (i.e., positive feedback) to the total samples."*

**① 为什么负样本梯度 ∝ $\hat{p}_j$（Eq. 10）**

BCE 对负样本（$y_j = 0$）的梯度：

$$
\nabla_{z_j} L_{\text{BCE}} = \sigma(z_j) - y_j = \sigma(z_j) = \hat{p}_j
$$

$\hat{p}_j$ 就是模型给样本 $j$ 的预测点击率（pCTR）。

**② 为什么"无偏"模型的 $\hat{p}_j$ 期望 ≈ 全局 CTR？——BCE 的全局最优解就是条件概率**

BCE 损失对固定特征 $x$ 的样本，其全局最小值点满足 $\sigma(z) = P(y{=}1 \mid x) = E[y \mid x]$。证明：对固定 $x$，设真实点击概率为 $p = P(y{=}1\mid x)$，BCE 期望为 $-p\log\sigma - (1-p)\log(1-\sigma)$，对 $\sigma$ 求导令 0 得 $\sigma = p$。

所以一个"无偏 / 尺度校准（scale calibrated）"的模型，其输出就是真实条件概率 $\hat{p}_j \approx P(\text{click} \mid x_j)$。

**③ 为什么全局 CTR 低 → 负样本的 $\hat{p}_j$ 也低？**

对所有样本（正 + 负）的预测求期望：

$$
\mathbb{E}_{x}\bigl[\hat{p}_j\bigr] = \mathbb{E}_x\bigl[P(\text{click}\mid x)\bigr] = P(\text{click}) = \frac{N^{+}}{N}
$$

即"平均预测点击率 = 全局点击率 = 正样本占比"。**当正样本稀疏（如广告 CTR < 2%），随机抽一个样本（大概率是负样本），它的预测点击率平均只有 2% 左右。** 因为绝大多数样本是负样本，模型对它们的条件概率估计天然很低——这是数据的统计特性，与模型好坏无关。

**④ 结论（Finding 3）：负样本梯度 ∝ 很小的 $\hat{p}_j$ → 梯度消失**

$$
\nabla_{z_j} L_{\text{BCE}} = \hat{p}_j \approx \text{global CTR} < 2\% \approx 0
$$

模型对"典型负样本"几乎不产生梯度，无法区分"普通负样本"和"接近点击边界的难负样本"，负样本之间的相对排序学不到。

**一个数值例子**：假设 100 万曝光样本、2 万点击，全局 CTR = 2%。

- 完美校准模型对"典型负样本"预测 $\hat{p}_j \approx 0.02$ → 负样本梯度 $\approx 0.02$（极小）；
- 对正样本预测 $\hat{p}_i \approx 0.98$ → 正样本梯度 $\approx 0.98 - 1 = -0.02$（其实也不大，但方向明确）。

> 直觉：BCE 把每个样本当独立二分类，模型只要"把绝大多数样本预测为低概率"就能把损失降得很低。负样本已经"躺平"在低概率区，梯度极小，模型没有动力去精雕细琢"哪些负样本更接近正样本"。**排序损失（如 Combined-Pair）的负样本梯度 $\propto \sum \sigma(z^- - z_i^+)$ 不依赖 $\hat{p}_j$ 本身，而是依赖"负样本相对正样本的分数差"，因此始终有梯度**——这正是加排序损失能缓解梯度消失的根源。

**对你的场景的意义**：酒店点击/下单同样稀疏（下单率远低于 2%），baseline 的 `loss_op_cr`（下单）负样本梯度消失问题比 click 更严重。所以 Combined-Pair 这类辅助排序损失在 **cr 任务上预期收益更大**——这与此前"只加 `listnet_w_cr`"的建议一致。

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

- **链接**：[Click-through Prediction for Advertising in Twitter Timeline（Li et al., KDD 2015）](https://doi.org/10.1145/2783258.2788582)，后被广泛引用为 Combined-Pair 来源。
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

#### 2.2.1 代码实现与公式对应（Combined-Pair）

> 对应实验目录 `zxhtl_seq_v2_5_nosid_topnsp_qw_cp_meanpool_M3oE_f_idgate_300` 中 `build_model` 里的 `_build_combined_pair_loss`。
> 说明：代码实现了"按分组"的 pairwise 损失。`group_by_qid=true` 按 `query_id` 分组（组内正负配对）；`false` 全局配对（整个 batch 一组）。

**数学公式回顾**：

$$
\mathcal{L}_{\text{CP}} = \frac{1}{G}\sum_{g=1}^{G}\Bigl[\frac{1}{N^{+}_g N^{-}_g}\sum_{i\in N^{+}_g}\sum_{j\in N^{-}_g}\underbrace{\log\bigl(1+e^{-(z_i - z_j)}\bigr)}_{=\text{softplus}(-(z_i-z_j))}\Bigr]
$$

即：对每个分组 $g$，取其内部的正样本对负样本的所有配对，每个配对用 RankNet 式 logistic loss；组内平均后，再对**有配对的分组**取平均。

**核心代码（带 shape 注释）**：

```python
# group_by_qid=true 用 query_id 分组; false 全部视为一组
if getattr(self.config.parm, 'group_by_qid', True):
    gid_1d = tf.reshape(query_id, [-1])                  # [B_valid]
    unique_gid, gid_compact = tf.unique(gid_1d)          # gid_compact ∈ [0, K)
    num_g = tf.size(unique_gid)
else:
    b_valid = tf.shape(query_id)[0]
    gid_compact = tf.zeros([b_valid], dtype=tf.int32)
    num_g = 1

def _build_combined_pair_loss(logits, label, name):
    z = tf.reshape(logits, [-1])                         # [B_valid]
    y = tf.reshape(tf.cast(label, tf.float32), [-1])     # [B_valid] 0/1
    pos_mask = tf.cast(tf.greater(y, 0.5), tf.float32)   # [B_valid] 正样本 mask
    neg_mask = 1.0 - pos_mask                            # [B_valid] 负样本 mask
    diff = tf.expand_dims(z, 1) - tf.expand_dims(z, 0)   # [B, B] diff[i,j]=z_i-z_j
    pair_loss = tf.nn.softplus(-diff)                    # [B, B] log(1+exp(-(z_i-z_j)))
    same_group = tf.cast(tf.equal(gid_compact[:, None], gid_compact[None, :]), tf.float32)  # [B, B]
    pair_mask = pos_mask[:, None] * neg_mask[None, :] * same_group   # [B, B]
    weighted = pair_loss * pair_mask                     # [B, B] 只保留"正对负且同组"
    pos_cnt = tf.unsorted_segment_sum(pos_mask, gid_compact, num_segments=num_g)   # [K]
    neg_cnt = tf.unsorted_segment_sum(neg_mask, gid_compact, num_segments=num_g)   # [K]
    pair_cnt = pos_cnt * neg_cnt                         # [K] 每组正负对数
    pair_sum = tf.unsorted_segment_sum(tf.reduce_sum(weighted, axis=1), gid_compact, num_segments=num_g)  # [K]
    per_group = tf.where(pair_cnt > 0, pair_sum / tf.maximum(pair_cnt, 1.0), tf.zeros_like(pair_cnt))
    valid_cnt = tf.reduce_sum(tf.cast(pair_cnt > 0, tf.float32))
    loss = tf.reduce_sum(per_group) / tf.maximum(valid_cnt, 1.0)
    return loss

cp_loss_click = _build_combined_pair_loss(mlp_out_list[0], labels[0], 'click')
cp_loss_cr = _build_combined_pair_loss(mlp_out_list[2], labels[1], 'cr')
self.loss_ops = [loss_op_click, loss_op_cr, loss_op_dr, loss_op_cvr,
                 self.config.parm.cp_w_click * cp_loss_click,
                 self.config.parm.cp_w_cr * cp_loss_cr]
```

**逐行推导（代码 ↔ 公式）**：

| 代码 | shape | 数学含义 |
|------|-------|---------|
| `z = reshape(logits)` | `[B_valid]` | 第 $i$ 个样本的 logit $z_i$ |
| `y = reshape(label)` | `[B_valid]` | 标签 $y_i \in \{0,1\}$ |
| `pos_mask = (y>0.5)` | `[B_valid]` | 正样本指示 $\mathbb{1}[y_i{=}1]$ |
| `neg_mask = 1-pos_mask` | `[B_valid]` | 负样本指示 $\mathbb{1}[y_i{=}0]$ |
| `diff = z[:,None]-z[None,:]` | `[B, B]` | 两两分数差矩阵 $\text{diff}[i,j] = z_i - z_j$ |
| `pair_loss = softplus(-diff)` | `[B, B]` | 每对 $(i,j)$ 的 RankNet 损失 $\log(1+e^{-(z_i-z_j)})$ |
| `same_group` | `[B, B]` | $\mathbb{1}[g_i = g_j]$，是否同一分组 |
| `pair_mask = pos·neg·same_group` | `[B, B]` | 只保留"$i$ 正、$j$ 负、同组"的配对 |
| `pos_cnt / neg_cnt` | `[K]` | 每组正/负样本数 $N^{+}_g$ / $N^{-}_g$ |
| `pair_cnt = pos_cnt·neg_cnt` | `[K]` | 每组正负对数 $N^{+}_g \cdot N^{-}_g$ |
| `pair_sum` | `[K]` | 每组配对损失和 $\sum_{i\in N^{+}_g}\sum_{j\in N^{-}_g}\text{softplus}(\cdot)$ |
| `per_group = pair_sum/pair_cnt` | `[K]` | 组内平均 $\frac{1}{N^{+}_g N^{-}_g}\sum\cdots$ |
| `loss = mean(per_group)` | 标量 | 对所有**有配对**的分组取平均 |

**数值例子**（2 个 query，5 个 item）：

| query | item (label, logit) | 正负对 | 组内平均 loss |
|-------|--------------------|--------|--------------|
| q0 | (1,1.0) (0,0.5) (0,0.2) | 1 正 × 2 负 = 2 对 | $\frac{\log(1+e^{-0.5})+\log(1+e^{-0.8})}{2}$ |
| q1 | (1,0.8) (0,0.3) | 1 正 × 1 负 = 1 对 | $\log(1+e^{-0.5})$ |

最终 $\mathcal{L}_{\text{CP}} = \frac{1}{2}(\text{q0 平均} + \text{q1 平均})$（两 query 都有配对）。

**注意事项**：
- `pair_cnt = pos_cnt × neg_cnt` 用乘法是因为"组内正负对数 = 组内正样本数 × 组内负样本数"（笛卡尔积）；
- `group_by_qid=false` 时 `gid_compact` 全 0、`num_g=1`，退化为全局配对（腾讯论文原版 batch 化全局配对）；
- 若正负对数巨大（如 $N^{+}=200, N^{-}=1800 \to 36$ 万对），`[B, B]` 矩阵仍可接受，但可加负采样（每正样本采 K 个负样本）控制。

### 2.3 Combined-List

- **链接**：Yan et al. 2022（BCE + ListNet）。注：腾讯论文将"BCE + ListNet loss"记为 Combined-List 并引用 Yan et al. 2022，但该确切标题/DOI 未能在公开检索中精确定位（可能为内部或非 arXiv 工作），原始出处建议以腾讯论文参考文献为准。其基础 ListNet 见 [Cao et al., ICML 2007](https://doi.org/10.1145/1273496.1273513)。
- **核心思想**：把 pairwise 换成 listwise（ListNet top-one one-hot 版），对整个 batch 做 softmax 交叉熵：

$$
\mathcal{L}_{\text{List}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{z_i}}{\sum_{k=1}^{N}e^{z_k}}
$$

- **与 Combined-Pair 的对比**：
  - 两者都能给负样本非零梯度；
  - List 版把"整个 batch"当成一个列表，正样本被推向 softmax 概率高；Pair 版逐对比较。
  - 腾讯论文实验显示：Combined-Pair 通常略优于 Combined-List（pairwise 的逐对比较更细粒度），但 List 版实现更简单。
- **注意**：这里的 ListNet 是在 **batch 级**做的（把 batch 当列表），**不是 query 内**——这与搜索场景的"query 内列表"不同，是 CTR 场景的常见简化（见第 3 节）。

#### 2.3.1 代码实现与公式对应（Combined-List）

> 对应实验目录 `zxhtl_seq_v2_5_nosid_topnsp_qw_cl_meanpool_M3oE_f_idgate_300` 中 `build_model` 里的 `_build_combined_list_loss`。
> **与之前实现的 listnet 辅助损失（`query_wise_summary.md` ⑨ 节）逻辑完全一致**，只是新增了 `group_by_qid=false`（全局）分支。

**数学公式回顾**（ListNet top-one one-hot，按分组）：

$$
\mathcal{L}_{\text{CL}} = \frac{1}{G}\sum_{g=1}^{G}\Bigl[\frac{1}{N^{+}_g}\sum_{i\in N^{+}_g} -\log\underbrace{\frac{e^{z_i}}{\sum_{k\in g} e^{z_k}}}_{=\text{softmax}(z)_i}\Bigr]
$$

即：组内对每个**正样本**计算"它的分数在组内 softmax 中的概率"的负对数，正样本平均后，再对有正样本的分组取平均。

**核心代码（带 shape 注释）**：

```python
def _build_combined_list_loss(logits, label, name):
    logits = tf.reshape(logits, [-1])                      # [B_valid]
    label = tf.reshape(tf.cast(label, tf.float32), [-1])   # [B_valid] 0/1
    # 数值稳定: 组内减去最大 logit
    max_logit = tf.unsorted_segment_max(logits, gid_compact, num_segments=num_g)   # [K]
    shifted = logits - tf.gather(max_logit, gid_compact)   # [B_valid]
    exp_shifted = tf.exp(shifted)                          # [B_valid]
    denom = tf.unsorted_segment_sum(exp_shifted, gid_compact, num_segments=num_g)  # [K]
    log_softmax = shifted - tf.log(tf.gather(denom, gid_compact))  # [B_valid]
    pos_loss = -label * log_softmax                        # 仅正样本贡献
    pos_cnt = tf.unsorted_segment_sum(label, gid_compact, num_segments=num_g)   # [K]
    pos_sum = tf.unsorted_segment_sum(pos_loss, gid_compact, num_segments=num_g) # [K]
    per_group = tf.where(pos_cnt > 0, pos_sum / tf.maximum(pos_cnt, 1.0), tf.zeros_like(pos_cnt))
    valid_cnt = tf.reduce_sum(tf.cast(pos_cnt > 0, tf.float32))
    loss = tf.reduce_sum(per_group) / tf.maximum(valid_cnt, 1.0)
    return loss

cl_loss_click = _build_combined_list_loss(mlp_out_list[0], labels[0], 'click')
cl_loss_cr = _build_combined_list_loss(mlp_out_list[2], labels[1], 'cr')
self.loss_ops = [loss_op_click, loss_op_cr, loss_op_dr, loss_op_cvr,
                 self.config.parm.cl_w_click * cl_loss_click,
                 self.config.parm.cl_w_cr * cl_loss_cr]
```

**逐行推导（代码 ↔ 公式）**：

| 代码 | shape | 数学含义 |
|------|-------|---------|
| `logits = reshape(logits)` | `[B_valid]` | logit $z_i$ |
| `label = reshape(label)` | `[B_valid]` | 标签 $y_i \in \{0,1\}$ |
| `max_logit = unsorted_segment_max` | `[K]` | 每组最大 logit $M_g = \max_{k\in g} z_k$（数值稳定） |
| `shifted = logits - gather(max_logit)` | `[B_valid]` | $t_i = z_i - M_{g(i)} \le 0$ |
| `exp_shifted = exp(shifted)` | `[B_valid]` | $e^{t_i}$（不溢出） |
| `denom = unsorted_segment_sum` | `[K]` | 组内分母 $Z_g = \sum_{k\in g} e^{z_k - M_g}$ |
| `log_softmax = shifted - log(gather(denom))` | `[B_valid]` | $\log\text{softmax}(z)_i = t_i - \log Z_{g(i)}$ |
| `pos_loss = -label·log_softmax` | `[B_valid]` | 仅正样本项 $-\mathbb{1}[y_i{=}1]\log\text{softmax}(z)_i$ |
| `pos_cnt / pos_sum` | `[K]` | 每组正样本数 / 正样本损失和 |
| `per_group = pos_sum/pos_cnt` | `[K]` | 组内正样本平均 |
| `loss = mean(per_group)` | 标量 | 对有正样本的分组取平均 |

**与 CP 的对比（逐行差异）**：
- CP 用 `[B,B]` 两两配对矩阵；CL 用 `unsorted_segment_sum` 聚合（只需 `[B_valid]` 和 `[K]`），实现更轻量；
- CP 的负样本梯度 ∝ $\sum\sigma(z^- - z^+)$；CL 的负样本梯度 ∝ $\mathrm{softmax}(z^-)$（都非零，缓解梯度消失）；
- **`group_by_qid=false` 时**：`num_g=1`，softmax 在整个 batch 上做，即"把整个 batch 当一个列表"——这是 CTR 场景的常见简化（等价于腾讯论文的 batch 级 ListNet）。

### 2.4 RCR（精简）

- **链接**：[Regression Compatible Listwise Objectives for Calibrated Ranking with Binary Relevance（Bai et al., 2023，Google）](https://arxiv.org/abs/2211.01494)。
- **核心思想**：让排序损失的**最小值与 BCE 的最小值对齐**（regression-compatible），避免"排序损失的最优解 ≠ 校准的点击率"。
- **做法**：用 `ListCE`（ListNet 的变体，对 logit 做"兼容"变换）作为辅助损失，使模型既能排序又能保持校准。被 Google 采用。
- **与腾讯论文的关系**：腾讯论文把 RCR 归入"BCE+List 式排序损失"一族，梯度分析与 Combined-List 类似。

### 2.5 JRC（精简）

这块参考 https://zhuanlan.zhihu.com/p/638414676 这篇文章很详细
- **链接**：[Joint Optimization of Ranking and Calibration with Contextualized Hybrid Model（Sheng et al., KDD 2023，Alibaba）](https://arxiv.org/abs/2208.06164)。
- **核心思想**：把 logit **解耦成"点击 logit"与"非点击 logit"两个**，用减法控制校准、用对比控制排序：

$$
\mathcal{L}_{\text{Rank}}^{\text{JRC}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{e^{z_i y_i}}{\sum_{k=1}^{N}e^{z_k y_k}}
$$

- **关键性质**：$z_i y_i$ 使**正样本**的分数进 softmax 分子、负样本按标签加权；对"非点击 logit"单独产生梯度，缓解梯度消失。在阿里展示广告落地，同时优化排序与校准。
- **与腾讯论文的关系**：腾讯论文把 JRC 作为"listwise-like"的一族，梯度分析同样适用。

#### 2.5.1 代码实现与公式对应（JRC）

> 对应实验目录 `zxhtl_seq_v2_5_nosid_topnsp_qw_jrc_meanpool_M3oE_f_idgate_300` 中 `build_model` 里的 `_build_jrc_loss`。
> 说明：本实现**不改变主塔输出结构**（现有 logit 仍作 `z_click`），额外从 PPNet 塔 128 维表征生成 `z_nclick`，体现 JRC 的双 logit；核心排序损失用 JRC 的标签加权 listwise 公式。可选校准损失（`sigmoid(z_click - z_nclick)` 逼近 label）默认权重 0 关闭。

**数学公式回顾**（JRC 双 logit + 排序损失）：

$$
\text{双 logit: } \underbrace{z_i^{\text{click}}}_{\text{主塔 logit}} \oplus \underbrace{z_i^{\text{nclick}}}_{\text{PPNet 塔投影}}
\qquad
\text{排序损失: } \mathcal{L}_{\text{Rank}}^{\text{JRC}} = -\frac{1}{G}\sum_{g=1}^{G}\Bigl[\frac{1}{N^{+}_g}\sum_{i\in N^{+}_g}\log\frac{e^{z_i y_i}}{\sum_{k\in g}e^{z_k y_k}}\Bigr]
$$

**核心代码（带 shape 注释）**：

```python
def _build_jrc_loss(logits, cur_rep, label, name):
    z = tf.reshape(logits, [-1])                           # z_click [B_valid]
    y = tf.reshape(tf.cast(label, tf.float32), [-1])       # [B_valid] 0/1
    # ---- 生成非点击 logit: 从 PPNet 塔表征线性投影(体现双 logit) ----
    cur_rep = tf.cast(cur_rep, tf.float32)
    cur_rep_dim = cur_rep.get_shape().as_list()[-1]        # 128
    with tf.variable_scope('jrc_nclick_' + name, reuse=tf.AUTO_REUSE):
        w_nclick = tf.get_variable("W", [cur_rep_dim, 1],
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        b_nclick = tf.get_variable("b", [1, 1],
                                   initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        z_nclick = tf.add(tf.matmul(cur_rep, w_nclick), b_nclick)   # [B_valid, 1]
    z_nclick = tf.reshape(z_nclick, [-1])                  # [B_valid]
    # ---- JRC 排序损失: -log( exp(z_i*y_i) / Σ_k exp(z_k*y_k) ) ----
    shifted = z * y                                        # 标签加权 logit
    max_shifted = tf.unsorted_segment_max(shifted, gid_compact, num_segments=num_g)  # [K]
    exp_shifted = tf.exp(shifted - tf.gather(max_shifted, gid_compact))  # [B_valid]
    denom = tf.unsorted_segment_sum(exp_shifted, gid_compact, num_segments=num_g)    # [K]
    log_softmax = (shifted - tf.gather(max_shifted, gid_compact)
                   - tf.log(tf.gather(denom, gid_compact)))            # [B_valid]
    pos_mask = y                                        # 仅正样本贡献
    loss_per_item = -pos_mask * log_softmax              # [B_valid]
    pos_cnt = tf.unsorted_segment_sum(pos_mask, gid_compact, num_segments=num_g)   # [K]
    pos_sum = tf.unsorted_segment_sum(loss_per_item, gid_compact, num_segments=num_g) # [K]
    per_group = tf.where(pos_cnt > 0, pos_sum / tf.maximum(pos_cnt, 1.0), tf.zeros_like(pos_cnt))
    valid_cnt = tf.reduce_sum(tf.cast(pos_cnt > 0, tf.float32))
    rank_loss = tf.reduce_sum(per_group) / tf.maximum(valid_cnt, 1.0)
    # ---- 可选校准损失: sigmoid(z_click - z_nclick) 逼近 label ----
    cal_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=z - z_nclick, labels=y))
    return rank_loss, cal_loss

jrc_rank_click, jrc_cal_click = _build_jrc_loss(mlp_out_list[0], task_embeddings[0], labels[0], 'click')
jrc_rank_cr, jrc_cal_cr = _build_jrc_loss(mlp_out_list[2], task_embeddings[2], labels[1], 'cr')
self.loss_ops = [loss_op_click, loss_op_cr, loss_op_dr, loss_op_cvr,
                 self.config.parm.jrc_w_click * jrc_rank_click,
                 self.config.parm.jrc_w_cr * jrc_rank_cr,
                 self.config.parm.jrc_w_cal_click * jrc_cal_click,
                 self.config.parm.jrc_w_cal_cr * jrc_cal_cr]
```

**逐行推导（代码 ↔ 公式）**：

| 代码 | shape | 数学含义 |
|------|-------|---------|
| `z = reshape(logits)` | `[B_valid]` | 点击 logit $z_i^{\text{click}}$（主塔现有输出） |
| `z_nclick = matmul(cur_rep, W)+b` | `[B_valid]` | 非点击 logit $z_i^{\text{nclick}}$（从 PPNet 塔表征投影） |
| `shifted = z * y` | `[B_valid]` | 标签加权 logit $z_i y_i$（正样本= $z_i$，负样本=0） |
| `max_shifted` | `[K]` | 每组最大 $z_i y_i$（数值稳定） |
| `exp_shifted` | `[B_valid]` | $e^{z_i y_i - M_g}$ |
| `denom` | `[K]` | 组内分母 $\sum_{k\in g} e^{z_k y_k}$ |
| `log_softmax` | `[B_valid]` | $\log\frac{e^{z_i y_i}}{\sum_{k\in g}e^{z_k y_k}}$ |
| `loss_per_item = -y·log_softmax` | `[B_valid]` | 仅正样本的负对数概率 |
| `per_group` | `[K]` | 组内正样本平均 |
| `rank_loss = mean(per_group)` | 标量 | JRC 排序损失 |
| `cal_loss = BCE(z - z_nclick, y)` | 标量 | 校准：$p = \sigma(z^{\text{click}} - z^{\text{nclick}})$ 逼近标签 |

**关键推导：为什么 $z_i y_i$ 让"负样本只贡献常数"**：

当 $y \in \{0,1\}$ 时：
- 正样本：$e^{z_i \cdot 1} = e^{z_i}$（分数参与分子分母）；
- 负样本：$e^{z_i \cdot 0} = e^0 = 1$（**常数**，不随负样本 logit 变化）。

所以分母 $= \sum_{\text{正}}e^{z} + N^{-}$（正样本分数和 + 负样本个数）。负样本分数**不直接参与排序**（只以个数进入归一化），但对"非点击 logit"的校准路径（`z - z_nclick`）单独产生梯度——这就是 JRC"解耦排序与校准"的含义。

**注意事项**：
- `task_embeddings[0]` 是 click 塔的 PPNet 128 维表征，`task_embeddings[2]` 是 cr 塔的（在多塔循环里收集）；
- 主塔 logit 仍用于现有 BCE/ctcvr 损失，本实现是**附加**的 JRC 辅助损失，不改变主输出；
- `jrc_w_cal_*` 默认 0 关闭校准项；若开启，`sigmoid(z_click - z_nclick)` 会成为模型另一个可解释的校准概率。

### 2.6 Combined-Contrastive（腾讯新提出）

- **链接**：腾讯 [arXiv:2403.14144](https://arxiv.org/abs/2403.14144)（Combined-Contrastive，`L_CC = α·L_BCE + (1-α)·L_Contr`）。
- **核心思想**：用 InfoNCE 式对比损失作为辅助项，让同 label 的表征靠近、不同 label 的表征远离，同样缓解 BCE 的负样本梯度消失。对比项在**表示空间**（而不是 logit 空间）工作，与 Combined-Pair/List（logit 空间）互补。
- **与 InfoNCE 的关系**：`L_Contr` 就是第 4 节 InfoNCE 的推荐场景应用——锚点是当前 item 表征，正对是"同组同 label 的其他 item"，负对是"同组不同 label 的 item"（通过分母隐式包含）。

#### 2.6.1 代码实现与公式对应（Combined-Contrastive）

> 对应实验目录 `zxhtl_seq_v2_5_nosid_topnsp_qw_cc_meanpool_M3oE_f_idgate_300` 中 `build_model` 里的 `_build_combined_contrastive_loss`。
> 表征 `rep` 取多塔 PPNet 塔 128 维输出（`cur_input` / `mlp_out_list_1[0]`），通过 `task_embeddings` 收集。

**数学公式回顾**（组内 InfoNCE）：

$$
\mathcal{L}_{\text{Contr}} = -\frac{1}{|\mathcal{A}|}\sum_{i\in\mathcal{A}} \log\frac{\sum_{j\in\mathcal{P}_i} e^{\,\text{sim}(h_i,h_j)/\tau}}{\sum_{k\in g(i), k\ne i} e^{\,\text{sim}(h_i,h_k)/\tau}}
$$

其中 $h_i$ 是 item $i$ 的 128 维表征，$\mathcal{P}_i$ 是同组同 label 的其他 item，$\tau$ 是温度。分子=正对相似度，分母=同组所有非自身（含正对和负对）。

**核心代码（带 shape 注释）**：

```python
def _build_combined_contrastive_loss(rep, label, name, tau):
    rep = tf.cast(rep, tf.float32)                         # [B_valid, 128] cur_input 表征
    y = tf.reshape(tf.cast(label, tf.float32), [-1])       # [B_valid] 0/1
    b = tf.shape(rep)[0]
    rep_norm = tf.nn.l2_normalize(rep, axis=-1)            # [B_valid, 128] L2 归一化
    sim = tf.matmul(rep_norm, rep_norm, transpose_b=True) / tau  # [B, B] cosine/tau
    eye = tf.eye(b, dtype=tf.float32)
    same_group = tf.cast(tf.equal(gid_compact[:, None], gid_compact[None, :]), tf.float32)  # [B, B]
    same_label = tf.cast(tf.equal(tf.cast(y[:, None], tf.int32), tf.cast(y[None, :], tf.int32)), tf.float32)  # [B, B]
    pos_pair = same_group * same_label * (1.0 - eye)       # [B, B] 正对: 同组同label非自身
    denom = tf.reduce_sum(tf.exp(sim) * same_group * (1.0 - eye), axis=1)  # [B] 分母
    numer = tf.reduce_sum(tf.exp(sim) * pos_pair, axis=1)  # [B] 分子
    has_pos = tf.cast(tf.reduce_sum(pos_pair, axis=1) > 0, tf.float32)  # [B]
    loss_per_item = -tf.log(tf.maximum(numer, 1e-8) / tf.maximum(denom, 1e-8))  # [B]
    loss = tf.reduce_sum(loss_per_item * has_pos) / tf.maximum(tf.reduce_sum(has_pos), 1.0)
    return loss

cc_loss_click = _build_combined_contrastive_loss(task_embeddings[0], labels[0], 'click', self.config.parm.cc_tau)
cc_loss_cr = _build_combined_contrastive_loss(task_embeddings[2], labels[1], 'cr', self.config.parm.cc_tau)
self.loss_ops = [loss_op_click, loss_op_cr, loss_op_dr, loss_op_cvr,
                 self.config.parm.cc_w_click * cc_loss_click,
                 self.config.parm.cc_w_cr * cc_loss_cr]
```

**逐行推导（代码 ↔ 公式）**：

| 代码 | shape | 数学含义 |
|------|-------|---------|
| `rep_norm = l2_normalize(rep)` | `[B,128]` | 表征归一化 $h_i / \|h_i\|$（用于 cosine 相似度） |
| `sim = matmul(rep_norm, rep_norm^T)/tau` | `[B, B]` | 相似度矩阵 $\text{sim}(h_i,h_j)/\tau$ |
| `same_group` | `[B, B]` | 是否同组 $g_i = g_j$ |
| `same_label` | `[B, B]` | 是否同标签 $y_i = y_j$ |
| `pos_pair = same_group·same_label·(1-eye)` | `[B, B]` | 正对集合 $\mathcal{P}_i$：同组同 label 非自身 |
| `denom` | `[B]` | 分母 $\sum_{k\in g(i),k\ne i} e^{\text{sim}(h_i,h_k)/\tau}$（含正对与负对） |
| `numer` | `[B]` | 分子 $\sum_{j\in\mathcal{P}_i} e^{\text{sim}(h_i,h_j)/\tau}$ |
| `has_pos` | `[B]` | 该 item 是否有正对（组内同 label 数 ≥2） |
| `loss = mean over items with pos` | 标量 | 只对"存在正对"的 item 计算 InfoNCE |

**关键推导：为什么负样本"不用单独构造"**：

InfoNCE 的分母 `denom = Σ_{同组非自身} e^{sim}` 已经包含：
- **正对**（同 label）：通过 `numer` 显式利用；
- **负对**（不同 label）：在分母里**隐式参与**——`log(numer/denom)` 里，分母越大损失越大，而分母包含所有负样本，所以负样本越多/越相似，损失越大，模型就被推向"让不同 label 的表征远离"。

这就是为什么 `neg_pair` 变量是冗余的（分母已覆盖）。若组内只有 1 个正样本（无其他同 label item），`has_pos=0`，该 item 不贡献——避免稀疏下的退化。

**注意事项**：
- 表征来自 `cur_input`（PPNet 塔 128 维，未 concat 前），click 用 `task_embeddings[0]`、cr 用 `task_embeddings[2]`；
- `cc_tau=0.1` 是 InfoNCE 温度；`tau` 越小对比越"尖锐"，loss 越大（权重需相应调小，见 alpha 调整）；
- 若组内同 label 样本过少（稀疏），`has_pos` 多为 0，对比信号弱——可考虑跨组采样或降低对"正对"的依赖。

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

## 附录：LogLoss、BCE 与交叉熵的关系推导

> 概念澄清：LogLoss 到底是"$-log p$"还是"BCE"？答案是——**LogLoss 就是"真实类别的负对数概率"；在二分类下它恰好等于 BCE，在多分类下它恰好等于 Categorical Cross-Entropy**。下面是完整推导。

### A.1 LogLoss 的一般定义（负对数似然）

对单个样本，模型输出一个类别概率分布 $\hat{p} = [\hat{p}_1, \dots, \hat{p}_K]$，真实标签用 one-hot $y = [y_1, \dots, y_K]$ 表示。**LogLoss 定义为"真实类别对应的预测概率"的负对数**：

$$
L_{\text{LogLoss}} = -\log \hat{p}_{c}, \qquad \text{其中 } y_c = 1
$$

### A.2 多分类：LogLoss = Categorical Cross-Entropy（分类交叉熵）

softmax 把 logits $z$ 变成概率分布：

$$
\hat{p}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}
$$

分类交叉熵（CCE）：

$$
L_{\text{CCE}} = -\sum_{k=1}^{K} y_k \log \hat{p}_k
$$

由于 $y$ 是 one-hot（只有 $y_c = 1$，其余为 0），求和塌缩成一项：

$$
L_{\text{CCE}} = -\,0\cdot\log\hat{p}_1 - \cdots - 1\cdot\log\hat{p}_c - \cdots - 0\cdot\log\hat{p}_K = -\log \hat{p}_c
$$

**结论：多分类 LogLoss = Categorical Cross-Entropy = softmax cross-entropy**。✓

### A.3 二分类：LogLoss = BCE（二元交叉熵）

二分类用 sigmoid 给出两类概率：

$$
\hat{p}_1 = \sigma(z) = \frac{1}{1+e^{-z}}, \qquad \hat{p}_0 = 1 - \sigma(z)
$$

真实标签 $y \in \{0,1\}$。按 LogLoss 定义（真实类别的负对数）：

- 当 $y = 1$：$L = -\log\sigma(z)$
- 当 $y = 0$：$L = -\log\bigl(1-\sigma(z)\bigr)$

用 $y$ 把两种情况统一写成一个式子：

$$
L = -\bigl[y\log\sigma(z) + (1-y)\log\bigl(1-\sigma(z)\bigr)\bigr]
$$

这正是 BCE。**结论：二分类 LogLoss = Binary Cross-Entropy**。✓

### A.4 二分类用 softmax 也等价于 sigmoid

如果二分类也用 softmax（两个输出 logit $z_1, z_0$）：

$$
\hat{p}_1 = \frac{e^{z_1}}{e^{z_1}+e^{z_0}} = \frac{1}{1+e^{-(z_1 - z_0)}} = \sigma(z_1 - z_0)
$$

令 $z = z_1 - z_0$，就退化成 sigmoid。所以**二分类 softmax 交叉熵与 BCE 完全等价**，只是 logit 差了一个平移。

### A.5 与腾讯论文 / 你代码的关系

| 上下文 | 说的"logloss" | 本质 |
|--------|--------------|------|
| 腾讯论文（Finding 3） | logloss / BCE | 就是二分类 BCE（你贴的 Eq. 10 梯度推导基于它） |
| 你的 baseline `loss_op_click` / `loss_op_cr` / `loss_op_cvr` | BCE | `tf.keras.backend.binary_crossentropy` 就是 BCE / logloss |
| Combined-List / ListNet 的 softmax 项 | softmax cross-entropy | 多分类交叉熵在"列表"上的形式（正样本位置） |

**一句话**：LogLoss 是"真实类别的负对数概率"这个**定义**；BCE / CCE 是它在二分类 / 多分类下的**具体展开**。CTR 场景说的 logloss 就是 BCE。

---

## 7. 参考链接汇总

| # | 论文/方法 | 链接 |
|---|----------|------|
| 1 | Understanding the Ranking Loss for Recommendation with Sparse User Feedback（腾讯，KDD 2024） | https://arxiv.org/abs/2403.14144 |
| 2 | RankNet: Learning to Rank using Gradient Descent（Burges et al., ICML 2005） | https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/ |
| 3 | Contrastive Predictive Coding（Oord et al., 2018，InfoNCE 来源） | https://arxiv.org/abs/1807.03748 |
| 4 | BPR: Bayesian Personalized Ranking from Implicit Feedback（Rendle et al., 2009） | https://arxiv.org/abs/1205.2618 |
| 5 | Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations（SBC，RecSys 2019） | https://arxiv.org/abs/1906.07591 |
| 6 | Click-through Prediction for Advertising in Twitter Timeline（Li et al., KDD 2015，Combined-Pair 来源） | https://doi.org/10.1145/2783258.2788582 |
| 7 | Regression Compatible Listwise Objectives for Calibrated Ranking with Binary Relevance（Bai et al., 2023，RCR） | https://arxiv.org/abs/2211.01494 |
| 8 | Joint Optimization of Ranking and Calibration with Contextualized Hybrid Model（Sheng et al., KDD 2023，JRC） | https://arxiv.org/abs/2208.06164 |
| 9 | Learning to Rank: From Pairwise Approach to Listwise Approach（Cao et al., ICML 2007，ListNet，Combined-List 基础） | https://doi.org/10.1145/1273496.1273513 |

> **链接核实说明**：上表中 #1-#9 均为已核实的真实链接（arXiv / DOI / 官方出版物页面）。
>
> **Combined-List（Yan et al. 2022）**：腾讯论文将"BCE + ListNet loss"记为 Combined-List，并引用 Yan et al. 2022；但该确切标题/DOI 未能在公开检索中精确定位（可能是一篇内部或非 arXiv 工作）。本文按腾讯论文口径描述其方法，原始出处建议以腾讯论文参考文献列表为准。其余方法（Combined-Pair / RCR / JRC / ListNet）均已附上述真实链接。

