# 生成式推荐三种样本组织方式对比

> 主要内容：Naive Impression、User-Centric、New Impression Only这三种常见样本组织方式的介绍，同时包含point-wise, query-wise等的简单介绍 
>
>本篇介绍以快手OneRec为例进行说明
>
> 结论先行：**开源版 OpenOneRec pretrain 用的是 User-Centric（一行=一个用户，全序列 loss）**；
> **快手线上 OneRec-V2 用的是 New Impression Only（一条=一次曝光，只在 target 上算 loss，流式训练）**。
> 中间还有一种 **Naive Impression**（一条=一次曝光，全序列 loss）是被两者共同淘汰的方案。
>
> 参考文献：[OneRec-V2 Technical Report (arXiv:2508.20900)](https://arxiv.org/abs/2508.20900) §2.1 Design Principles。

---

## 0. 统一记号

- 用户集合 $\mathcal{U} = \{U_1, U_2, ...\}$，item 集合 $\mathcal{I} = \{A, B, C, X, Y, ...\}$。
- 一条**曝光记录**（impression）是一个四元组 $(u, i, t, \text{label})$：谁在什么时刻看到了什么 item，是否发生了正反馈。
- 每个 item 已通过 residual K-means 编码为 3 个 SID token：$i \to (s^0_i, s^1_i, s^2_i)$。在 token 序列里写成 `<|sid_begin|><s_a><s_b><s_c><|sid_end|>` 共 5 个 token。为了讲清楚 loss，我们下面**把每个 item 的 5 个 token 缩写为一个方框** `[A]`。真实 loss_mask 的粒度是 token，不是 item —— 但除非特别说明，"给 A 算 loss" 意为给 A 对应的 5 个 token 里适合当预测目标的位置（通常是后 4 个）全部算 loss。
- 用一个**极小的模拟日志**贯穿全文，让三种做法可以直接对比：

  | 时刻 | 用户 | 曝光 item |
  | :--: | :--: | :--: |
  | t₁   | U1   | A |
  | t₂   | U2   | X |
  | t₃   | U1   | B |
  | t₄   | U2   | Y |
  | t₅   | U3   | A |
  | t₆   | U1   | C |

  共 6 次曝光，涉及 3 个用户 4 个 item。

---

## 1. 一图看懂三种方式

```
时间轴 →   t₁    t₂    t₃    t₄    t₅    t₆
U1        A ───────── B ─────────────── C
U2              X ─────────  Y
U3                                 A

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
(a) Naive Impression        6 samples, loss = 全序列
      Sample₁: [A]                    loss = {A}
      Sample₂: [X]                    loss = {X}
      Sample₃: [A, B]                 loss = {A, B}       ← A 又被算 loss
      Sample₄: [X, Y]                 loss = {X, Y}
      Sample₅: [A]                    loss = {A}          ← A 再一次
      Sample₆: [A, B, C]              loss = {A, B, C}    ← A, B 再一次

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
(b) User-Centric            3 samples (每用户一条), loss = 全序列
      Sample U1: [A, B, C]            loss = {A, B, C}   ← 但 t₁ 就学到 t₆ 的 C
      Sample U2: [X, Y]               loss = {X, Y}
      Sample U3: [A]                  loss = {A}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
(c) New Impression Only     6 samples, loss = 仅最新曝光的 target
      Sample₁ (t₁,U1,A):  context=[],      target=A     loss = {A}
      Sample₂ (t₂,U2,X):  context=[],      target=X     loss = {X}
      Sample₃ (t₃,U1,B):  context=[A],     target=B     loss = {B}
      Sample₄ (t₄,U2,Y):  context=[X],     target=Y     loss = {Y}
      Sample₅ (t₅,U3,A):  context=[],      target=A     loss = {A}
      Sample₆ (t₆,U1,C):  context=[A,B],   target=C     loss = {C}
```
其中context代表的是曝光前的历史行为流，target代表的是当前曝光的item。

**一句话对比：**

| 维度 | Naive Impression | User-Centric | New Impression Only |
| :--- | :--- | :--- | :--- |
| 样本粒度 | 一次曝光 | 一个用户的整段历史 | 一次曝光 |
| 训练目标 | 全序列 next-token | 全序列 next-token | 只在 target 上 next-token |
| 每 token 算 loss 次数 | O(user 出现次数) 冗余 | 1 次，但整段一起 | 恰好 1 次 |
| 时序泄漏 | ❌ 存在 | ❌❌ 严重 | ✅ 无 |
| 是否需按时间顺序训练 | 无所谓 | 无所谓 | ✅ 宏观必须按序 |
| 是否可流式 / 增量 | 勉强 | ❌ 需等用户历史攒够 | ✅ 天然流式 |
| 冗余算力开销 | 高 | 低 | 低 |
| 训练/推理形态一致性 | 中 | 差 | ✅ 高 |
| 开源 OneRec 采用? | 否 | ✅ pretrain 用 | 否 |
| 线上 OneRec-V2 采用? | 否 | 否 | ✅ |

---

## 2. Naive Impression

### 2.1 定义
每次曝光独立成一条训练样本；样本内容 = 该曝光时刻**之前的所有历史 item + 这次曝光的 item**；loss 落在整条序列的每个 item（自回归 next-token predictor 的默认做法）。

### 2.2 例子（对着 §0 的日志）

```
Sample₃ (t₃, U1, target=B):
    token seq:   <sid_begin><s_a><s_b><s_c><sid_end>       <sid_begin><s_a><s_b><s_c><sid_end>
                  ─────── item A（context+loss） ───────    ─────── item B（loss） ────────
    loss_mask:    0    1    1    1    1                     0    1    1    1    1
                  ↑                                         ↑
                  <sid_begin> 通常不预测自己                 同左

Sample₆ (t₆, U1, target=C):
    token seq:   [A] [B] [C]        # A, B, C 三段各 5 token
    loss_mask:   0 1 1 1 1 | 0 1 1 1 1 | 0 1 1 1 1
```

### 2.3 Batch 形状（假设 batch_size=4，max_seq_len=T）

```python
input_ids  : (B=4, T)   int64      # 每条样本都是一个序列，独立填充到 T
attn_mask  : (B=4, T)   int        # 1=有效，0=padding
loss_mask  : (B=4, T)   int        # 1=算 loss，0=不算（padding + sid_begin 等）
labels     : (B=4, T)   int64      # = input_ids 左移一位 + ignore_index 屏蔽 loss_mask==0
```

Loss:
$$\mathcal{L} = \frac{1}{|\{k:\text{loss\_mask}_k=1\}|} \sum_{k:\text{loss\_mask}_k=1} -\log p_\theta(\text{tok}_{k+1} \mid \text{tok}_{\le k})$$

### 2.4 训练与推理

- **训练**：batch 内样本可任意打乱，一个batch内样本可以来自不用用户，或者包含相同用户的不同请求（不过通常会把一次请求的所有候选item放在一个batch内），总条数等于batch_size即可；因为每条样本自带完整 context，不依赖其他样本。
- **推理**：给用户当前完整历史，让模型自回归生成下一个 SID → item。

### 2.5 应用场景 / 为什么被淘汰

- 直观、实现最简单，早期 GPT-style 序列推荐（TIGER 原始版、部分早期 LC-Rec 尝试）用过。
- **致命问题——重复训练冗余**：例子里 A 被算了 4 次 loss，B 被算了 2 次；同一个 (item, prefix) 组合被反复学，浪费算力。用户越活跃、曝光越密，冗余越严重。真实线上一个用户一天可能有几百条曝光，全序列 loss 意味着历史 item 被算 O(用户日均曝光数) 次。
- **弱泄漏**：Sample₆ 的 loss 里 A→B 的转移和 Sample₃ 里 A→B 的转移**在梯度上是不同时间**的，但它们在**逻辑上是同一个事件**，会互相扰动。

---

## 3. User-Centric（开源 OneRec pretrain 走这条）

### 3.1 定义
每个用户一条样本，把该用户**全部（截断到 HIST_MAX_LEN）**的行为流串成一条长序列；loss 落在整条序列的每个 item（全序列 next-token loss）。

### 3.2 例子

```
Sample U1 (整条时间轴 t₁→t₃→t₆ 的 A, B, C 拼一起):
    token seq:  [A]   [B]   [C]
    loss_mask:  0 1 1 1 1 | 0 1 1 1 1 | 0 1 1 1 1

Sample U2:  [X, Y]     loss on all
Sample U3:  [A]        loss on all
```

### 3.3 Batch 形状（开源版实际值：`batch_size=1`，seq packing）

[参考快手OneRec开源实现](https://github.com/Kuaishou-OneRec/OpenOneRec)（`pretrain/onerec_llm/data/qwen3_dataset.py`）：

以及[RQ-VAE+Transformer开源实现](https://github.com/EdoardoBotta/RQ-VAE-Recommender)
```python
input_ids       : (1, T)  int64,  T ≈ 30016 (max_length=30000 对齐 8 后 +64)
position_ids    : (1, T)
loss_mask       : (1, T)  除末尾 pad 外全 1（pretrain 阶段）
itemic_id_mask  : (1, T)  1=该 token 是 Itemic SID token
cu_seqlens      : (S+2,)  packing 边界。S ≈ 10~15，表示一个 packed 序列里塞了 S 个用户
sample_idx      : (1, T)  每个 token 属于第几个用户
```

**重点**：`batch_size=1` 但一条 packed 序列内部通过 `cu_seqlens` 塞了 ~10 个用户 —— 用 FlashAttention 的变长 attention 隔离，不同用户之间**互不 attention**。真正的有效 batch = `world_size × S`。

Loss：
$$\mathcal{L}_{\text{user-centric}} = \frac{1}{\sum_u L_u - 1} \sum_u \sum_{k=1}^{L_u-1} -\log p_\theta(\text{tok}^u_{k+1} \mid \text{tok}^u_{\le k})$$

其中 $L_u$ 是用户 $u$ 的行为流 token 长度。

### 3.4 训练与推理

- **训练**：数据是静态 parquet dump，一个用户一行；随机打乱、多 epoch 训练都可以。
- **推理**：给用户历史 → 生成下一个 SID。形态和训练一致。

### 3.5 时序数据泄漏（为什么在流式场景里不行）

再看例子：Sample U1 = `[A_{t₁}, B_{t₃}, C_{t₆}]`。
训练时一次前向就同时给 A→B 和 B→C 两个跳转算 loss。**这等价于让 t₁ 时刻的模型"预知" t₆ 时刻的 C**。

举个更具体的例子说明为什么这是问题：
- 假设 t₃ 时刻 U2 也曝光了 B（如实际发生的 Sample₄=Y？换个假设：t₃ 时 U2 曝光 B）。按流式服务视角，t₃ 时应给 U2 推什么？模型应基于 ≤t₃ 的数据回答。
- User-Centric 训练下模型已经通过 U1 的样本学到"看过 A、B 的用户下一个会看 C"（因为 U1 在 t₆ 看了 C）。这**未来信息 (t₆) 泄漏进了 t₃ 时刻的决策**。
- 表现在指标上：offline 评估精度虚高，线上 A/B 掉点。

**为什么开源版仍然选它？** 开源版是**静态数据集 dump**（一份 `onerec_bench_release.parquet`，不带真实时间戳），本来就没有"流式增量"这一说；把用户历史一次性打包进 LLM context 反而是最能榨干长上下文优势的做法。**科研复现友好 ≠ 工业上可行**。

### 3.6 流行度偏差

热门 item（如例子中的 A）在多个用户的历史里都出现。User-Centric 下 A 会被多次算 loss（U1 一次、U3 一次），加剧头部 item 的过训练。

### 3.7 无法增量 / 流式更新（最大问题）

要拼出 U1 的样本，你必须**等到 U1 的所有历史都产生完**（例子里到 t₆ 之后）。这跟"曝光一条来一条训"完全对立。

---

## 4. New Impression Only（OneRec-V2 采用）

### 4.1 定义（[OneRec-V2 §2.1](https://arxiv.org/html/2508.20900)）

> "organizes data chronologically but applies the training loss exclusively to the newest impressed item, ... where items in gray are excluded in next token prediction."

每次曝光独立一条样本，样本内容 = 曝光前的历史 context + 该曝光 item；**loss 只在最新那个 item 上算**，历史部分的每个 item 都是 gray（`loss_mask=0`），只做 attention context。

### 4.2 例子（对着 §0 的日志）

6 次曝光 → 6 条独立样本，严格按时间戳排序：

```
Sample₁  (t₁, U1):  context=∅,        target=A
    token seq:  [A]
    loss_mask:  0 1 1 1 1                          # (target 段有 loss)

Sample₂  (t₂, U2):  context=∅,        target=X
    token seq:  [X]
    loss_mask:  0 1 1 1 1

Sample₃  (t₃, U1):  context=[A],      target=B
    token seq:  [A]         [B]
    loss_mask:  0 0 0 0 0 | 0 1 1 1 1              # A 段全 0（只做 context）

Sample₄  (t₄, U2):  context=[X],      target=Y
    token seq:  [X]         [Y]
    loss_mask:  0 0 0 0 0 | 0 1 1 1 1

Sample₅  (t₅, U3):  context=∅,        target=A
    token seq:  [A]
    loss_mask:  0 1 1 1 1

Sample₆  (t₆, U1):  context=[A, B],   target=C
    token seq:  [A]         [B]         [C]
    loss_mask:  0 0 0 0 0 | 0 0 0 0 0 | 0 1 1 1 1
```

### 4.3 每 item 算 loss 的次数 —— 恰好一次

统计上面 6 条样本里各 item 出现在 loss 段的次数：

| item | 出现在 loss 段的次数 |
| :--: | :--: |
| A (t₁, U1 曝光)  | 1  ← Sample₁ 的 target |
| X (t₂, U2 曝光)  | 1  ← Sample₂ 的 target |
| B (t₃, U1 曝光)  | 1  ← Sample₃ 的 target |
| Y (t₄, U2 曝光)  | 1  ← Sample₄ 的 target |
| A (t₅, U3 曝光)  | 1  ← Sample₅ 的 target（U3 曝光的 A 是新事件，独立 loss） |
| C (t₆, U1 曝光)  | 1  ← Sample₆ 的 target |

**每一次曝光事件贡献恰好一次 loss；无冗余**。作为对照，Naive Impression 里 A 在 loss 段出现了 4 次。

### 4.4 Batch 形状（batch_size = B_global，无 packing）

```python
input_ids : (B, T)   int64
attn_mask : (B, T)   int
loss_mask : (B, T)   int         # 只有每条样本 target 那几个 token 是 1，其余（context+padding）全 0
labels    : (B, T)   int64       # 用 loss_mask 屏蔽 → ignore_index
timestamp : (B,)     int64       # 曝光时间戳（用于按序消费，不进模型）
```

Loss：
$$\mathcal{L}_{\text{NIO}} = \frac{1}{B \cdot L_{\text{target}}} \sum_{b=1}^{B} \sum_{k \in \text{target}(b)} -\log p_\theta(\text{tok}^b_{k+1} \mid \text{tok}^b_{\le k})$$

其中 `target(b)` 是样本 b 里 target item 那 4~5 个 token 位置，$L_{\text{target}}$ 就是这个大小。

### 4.5 训练：宏观按序 + 微观并行

**关键规则**：整个训练过程沿时间轴单调推进：$t_1 \to t_2 \to \dots \to t_6$。

假设 global batch size B=3，按时间顺序切成两个 batch：

```
Batch 1  (处理 t₁, t₂, t₃):
    { Sample₁(t₁,U1,A), Sample₂(t₂,U2,X), Sample₃(t₃,U1,B) }
    → 模型 W₀ → W₁    ← 单步 SGD

Batch 2  (处理 t₄, t₅, t₆):
    { Sample₄(t₄,U2,Y), Sample₅(t₅,U3,A), Sample₆(t₆,U1,C) }
    → 模型 W₁ → W₂
```

- **Batch 内部**（3 条样本）：可以任意打乱、并行 forward；因为 3 条样本互不依赖（不同用户 / 不同 target）。它们的时间戳 t₁<t₂<t₃ 只是消费顺序，不影响梯度语义。
- **Batch 之间**：必须严格按时间偏序。Batch 2 只能用 W₁（已经消化了 t₁..t₃）来处理 t₄..t₆ 的数据。这就是 streaming training 的字面含义。

**线上真实情况**：数据源以 Kafka/流的形式实时推曝光记录，训练器按分钟窗口消费。**"每分钟"不是样本切分粒度，而是"每分钟大约有多少条曝光会作为下一波训练输入"**：
在我们的场景下，这块需要改造训练框架的消费逻辑，保证**宏观上按时间顺序**，微观上 batch 内并行。或者按照One-Epoch的day by day训练策略，其实我认为也可行。
```
Kafka topic: impression_stream
    ├── minute 09:00 -- 例如 5000 万条曝光 → 切成 batch_size=8192 的 N 个 batch → 训练器按序消费 N 步
    ├── minute 09:01 -- 又是 5000 万条 → 又是 N 个 batch → 又是 N 步 SGD
    └── ...
                       ↓
                模型每训完一步立即用于下一步的服务
                （train 和 serve 之间存在 ≤秒级的窗口）
```

### 4.6 推理

- 线上服务：给某用户当前 context（≤t 时刻的所有历史）→ 用**最新 W_t** 模型自回归生成一个（或多个候选）SID → 走后续 ranker / 直接曝光。
- **训练和推理数据形态完全一致**：一次前向都是"context → target"，只是训练时 target 是 ground-truth（用户实际会看的），推理时 target 是模型生成的。因此几乎没有 train-serve skew。

### 4.7 为什么这是目前"主流"的业界做法

综合 [OneRec-V2 §2.1](https://arxiv.org/html/2508.20900)：

1. **无泄漏**：t 时刻的模型只见过 ≤t 的曝光；每条样本的 target 一定晚于其 context。
2. **无冗余**：每条曝光只算一次 loss（对比 Naive Impression 的 O(用户日活跃度) 次冗余）。
3. **天然流式**：曝光一条来一条训，模型持续演进；正好和线上生产系统的日志流对接。
4. **无流行度偏差放大**：热门 item 只在它真正被 target 的那一次算 loss；不像 User-Centric 会因 A 在 U1、U3 历史里都出现而多次训练 A 的 embedding。
5. **训练-推理形态统一**：减少 skew。

### 4.8 代价 / 需要额外做的工程

- 需要一套流式训练基础设施：Kafka + 训练器持续消费 + 快速权重 checkpoint 与 hot-swap。
- 单条样本上下文短（就是这次曝光的历史），要充分利用长上下文得靠**在样本里塞更长的用户历史**（context 段变长），比 User-Centric 那种"整条打包"的写法要精细设计截断策略。
- 反馈延迟：用户点击、长时观看等正反馈可能到曝光后几秒~几十秒才产生。快手用 fast-slow window（5min 快窗 + 1h 慢窗）来处理（参见 [Moment&Cross](https://arxiv.org/html/2508.20900) 相关工作）。

---



## 5. LLM4DLRMs 样本组织变革：从 Pointwise 到 Request-wise / Set-wise

> 上面 §1-§8 讨论的三种样本组织方式（Naive Impression / User-Centric / New Impression Only）都属于 **LLM4GRs（生成式推荐）** 范式。
> 在 **LLM4DLRMs（深度推荐模型）** 范式中，样本组织也在经历一场平行的变革：从传统 **Pointwise** 演进到 **Request-wise（List-wise）** 和 **Set-wise**。
> 本节梳理这条演进线，并讨论它与 LLM4GRs 三种方式的交叉可能性。

### 5.1 传统 Pointwise 样本

**做法**：每个 (用户, 物品) 对独立成一条样本。

```
传统 DLRM 样本:
  Sample₁: user=U1, item=A, label=1 (点击)
  Sample₂: user=U1, item=B, label=0 (未点击)
  Sample₃: user=U2, item=X, label=1
  Sample₄: user=U1, item=C, label=1
```

**特点**：
- 一条样本只包含一个 target item
- 模型对每个 item 独立打分：$\hat{y} = f_\theta(\text{user_feats}, \text{item_feats})$
- 样本数 = 曝光总数 $N$
- **问题**：同一请求中的多个候选 item 之间无交互；无法利用候选间的上下文关系

### 5.2 Request-wise / List-wise 样本（当前主流）
**做法**：基于**请求维度**组织样本，将同一次请求中下发（或曝光）的多个 item 打包为一条样本。

**代表工作**：网易云音乐 [Climber / ASTRO](https://arxiv.org/abs/2502)（2025年2月）

> Climber 将每次请求下发的最多 5 个交互 item 处理为一条样本，将多个 item 的特征和 label 拼成一个 list，共用 user 特征。

```
Request-wise 样本:
  Sample₁ (request r₁, U1):
    user_feats: {U1的特征}
    items: [
      {item=A, feats_A, label=1},
      {item=B, feats_B, label=0},
      {item=C, feats_C, label=1},
    ]
    → 一条样本包含 3 个 item，共用 user 特征

  Sample₂ (request r₂, U2):
    user_feats: {U2的特征}
    items: [
      {item=X, feats_X, label=1},
      {item=Y, feats_Y, label=1},
    ]
```

**特点**：
- 一条样本 = 一次请求，包含 $K$ 个 item（$K$ 通常为 3~10）
- User 特征只存一份，$K$ 个 item 共享 → **样本压缩**
- 同一请求内的 target item 之间**注意力互相 mask**，无交互
- 样本数 = 请求数 $R \ll N$
- **优势**：
  - 存储压缩：user 特征不重复存储
  - 适配增量/流式更新：每个请求独立成样本，无需等用户历史攒够
  - 天然解决时序泄漏：按请求时间排序即可

**Batch 形状**：
```python
user_feats : (B, D_user)           # B个请求的user特征
item_feats : (B, K, D_item)        # 每个请求K个item特征
labels     : (B, K)                # 每个item的label (0/1)
item_mask  : (B, K)                # 有效item mask (实际item数可能 < K)
```

### 5.3 Set-wise 样本（美团 HoMer）

**做法**：将同一请求中粗排给精排的**大量候选 item**（美团是 300 个）打包为一条样本，在单次模型调用中预测所有 item 的 CTR。

**代表工作**：美团 [HoMer](https://arxiv.org/abs/2510.11100)（2025年10月）

> HoMer 将预测范式从 point-wise 转为 set-wise，将所有候选 item 的非序列特征聚合到一条样本中，实现跨 item 交互和并行预测。

```
Set-wise 样本:
  Sample₁ (request r₁, U1):
    user_feats: {U1的特征}
    user_sequence: [A, B, X, ...]     # 用户历史行为序列
    candidates: [                     # 300个候选item
      {item=I₁, feats₁},
      {item=I₂, feats₂},
      ...
      {item=I₃₀₀, feats₃₀₀},
    ]
    labels: [label₁, label₂, ..., label₃₀₀]
    → 一条样本包含 300 个候选 item
```

**与 Request-wise 的关键区别**：

| 维度 | Request-wise | Set-wise |
|------|-------------|----------|
| 候选 item 数 | 少（3~10，已曝光的） | 多（~300，粗排给精排的） |
| 包含未曝光 item | ❌ 仅曝光 item | ✅ 包含未曝光候选 |
| Item 间交互 | ❌ 互相 mask | ✅ 跨 item attention |
| 模型调用次数/请求 | 1 次（但 item 间无交互） | 1 次（且 item 间有交互） |
| 存储成本 | 低 | 较高（但比 pointwise 仍低） |

**Set-wise 的优势**（HoMer 论文验证）：
- **跨 item 交互**：候选 item 之间可以互相 attend，捕获比较信号
- **效率提升**：一次请求只需一次模型调用，300 个 item 并行预测
- **存储压缩**：user 特征和历史序列只存一份，300 个候选共享
- **更丰富训练信号**：包含未曝光 item（负样本），训练分布更完整

### 5.4 三种 DLRM 样本方式对比

| 维度 | Pointwise | Request-wise | Set-wise |
|------|-----------|-------------|----------|
| 一条样本 = | 一个 (user, item) 对 | 一次请求的曝光 item | 一次请求的全部候选 |
| 样本数 | $N$（曝光总数） | $R$（请求数） | $R$（请求数） |
| 单样本 item 数 | 1 | $K$（3~10） | $K$（~300） |
| User 特征存储 | 每条重复 | 共享 | 共享 |
| Item 间交互 | ❌ | ❌（互相 mask） | ✅（跨 item attention） |
| 包含未曝光候选 | 否 | 否 | 是 |
| 适配流式训练 | 可以 | ✅ 天然适配 | ✅ 天然适配 |
| 时序泄漏风险 | 低 | 低 | 低 |
| 存储效率 | 差（user重复） | 好 | 好 |
| 代表系统 | 传统 DLRM | Climber (网易云) | HoMer (美团) |

### 5.5 LLM4DLRMs 与 LLM4GRs 样本组织的结合

LLM4DLRMs 的 Request-wise / Set-wise 思路和 LLM4GRs 的 New Impression Only 等方案**并非互斥**，可以组合使用：

#### 5.5.1 Request-wise 存储 + New Impression Only 训练

```
存储层（Request-wise）:
  每条记录 = 一次请求
  {
    "request_id": "r_001",
    "timestamp": 1712345678,
    "user_id": "U1",
    "user_context": [历史SID序列],
    "impressed_items": [              # 曝光的K个item
      {"sid": [3,7,22], "label": 1},
      {"sid": [15,8,91], "label": 0},
      {"sid": [42,3,67], "label": 1},
    ]
  }

训练层（New Impression Only）:
  从每条请求记录中，对每个impressed item拆出独立样本：
  Sample₁: context=[历史], target=[3,7,22],    loss_mask=仅target
  Sample₂: context=[历史], target=[15,8,91],   loss_mask=仅target
  Sample₃: context=[历史], target=[42,3,67],   loss_mask=仅target
```

**优势**：
- 存储层用 Request-wise 压缩（user 特征不重复）
- 训练层用 NIO 避免冗余 loss 和时序泄漏
- 两者兼得

#### 5.5.2 Set-wise 存储 + User-Centric 训练

```
存储层（Set-wise）:
  每条记录 = 一次请求，含300个候选
  {
    "user_id": "U1",
    "user_sequence": [A, B, X, ...],
    "candidates": [I₁, I₂, ..., I₃₀₀],
    "labels": [1, 0, 0, ..., 1],
  }

训练层（类User-Centric, 但loss在target上）:
  将用户序列 + 所有候选拼成一条长序列
  loss_mask: user_sequence段=0, target候选段=1
```

#### 5.5.3 结合方式对比

| 存储方式 \ 训练方式 | Naive Impression | User-Centric | New Impression Only |
|:--|:--|:--|:--|
| **Pointwise** (传统) | ✅ 经典 | ✅ 开源OneRec | ✅ OneRec-V2 |
| **Request-wise** (Climber式) | ✅ 拆成多条 | ⚠️ 需合并多请求 | ✅ **推荐组合**：存储压缩+NIO无泄漏 |
| **Set-wise** (HoMer式) | ⚠️ 候选太多效率低 | ⚠️ 需特殊mask | ✅ 候选内仅target有loss |

### 5.6 存储效率对比

不同样本组织方式对存储空间的影响（假设：$U$ 个用户，$N$ 次曝光，$R$ 次请求，平均每次请求 $K$ 个曝光，$C$ 个候选）：

| 样本方式 | 样本条数 | User特征存储次数 | 总存储量级 | 相对Pointwise压缩比 |
|----------|---------|----------------|-----------|:--:|
| Pointwise | $N$ | $N$ 次 | $N \times (D_{\text{user}} + D_{\text{item}})$ | 1× |
| Request-wise | $R = N/K$ | $R$ 次 | $R \times D_{\text{user}} + N \times D_{\text{item}}$ | ~$K$×（user特征部分） |
| Set-wise | $R$ | $R$ 次 | $R \times D_{\text{user}} + R \times C \times D_{\text{item}}$ | 视$C$而定 |
| User-Centric | $U$ | $U$ 次 | $U \times D_{\text{user}} + N \times D_{\text{item}}$ | ~$N/U$× |

**具体数值示例**（快手量级）：
- $U = 4 \times 10^8$ 用户，$N = 10^{10}$ 日曝光，$R = 2 \times 10^9$ 日请求，$K \approx 5$
- $D_{\text{user}} = 2$KB，$D_{\text{item}} = 500$B

| 方式 | 日存储估算 |
|------|-----------|
| Pointwise | $10^{10} \times 2.5\text{KB} \approx 25\text{TB}$ |
| Request-wise | $2\times10^9 \times 2\text{KB} + 10^{10} \times 0.5\text{KB} \approx 9\text{TB}$ |
| User-Centric | $4\times10^8 \times 2\text{KB} + 10^{10} \times 0.5\text{KB} \approx 5.8\text{TB}$ |

> **结论**：Request-wise 比 Pointwise 节省约 60% 存储（主要来自 user 特征不重复）；User-Centric 在 GR 场景下最省（但牺牲了流式能力）。
> **New Impression Only + Request-wise 存储** 是工业界 GR 场景下的最优组合：既省存储又支持流式。

### 5.7 小结

```
LLM4GRs 样本演进:
  Naive Impression → User-Centric → New Impression Only
  (全序列loss,冗余)   (用户级,泄漏)    (仅target loss,流式)

LLM4DLRMs 样本演进:
  Pointwise → Request-wise/List-wise → Set-wise
  (单item独立)   (请求级打包,共享user)    (候选级打包,item交互)

两者可以正交组合：
  存储层选 Request-wise/Set-wise (压缩)
  ×
  训练层选 New Impression Only (无泄漏,流式)
  =
  工业最优实践
```

---

## 6. Sources / 参考文献（补充）

- [OneRec-V2 Technical Report (arXiv:2508.20900)](https://arxiv.org/abs/2508.20900) — 三种样本组织方式的定义、Figure 3 图示、streaming training 实验设置的出处。
- [OneRec-V2 HTML 版](https://arxiv.org/html/2508.20900) — §2.1 Design Principles 原文可直接查阅。
- [HoMer: Addressing Heterogeneities by Modeling Sequential and Set-wise Contexts for CTR Prediction (arXiv:2510.11100)](https://arxiv.org/abs/2510.11100) — 美团，Set-wise 样本组织范式的代表工作。
- [Climber / ASTRO: An Efficient Large Recommendation Model (arXiv:2502)](https://arxiv.org/abs/2502) — 网易云音乐，Request-wise 样本组织的代表工作。
- 本仓库开源代码 —— User-Centric 实现的对照参考：
  - `pretrain/onerec_llm/data/qwen3_dataset.py::_process_completion` (pretrain 全序列 loss)
  - `pretrain/onerec_llm/data/qwen3_dataset.py::_get_assistant_mask` + `_process_chat` (SFT target-only loss，最接近 NIO)
  - `data/onerec_data/pretrain/video_rec.py` (User-Centric 样本生成)

---

## 补 · 关于 Loss 的重要澄清（三种方式共通）

三种样本组织方式**用的是同一种 loss —— token 级的多分类 CrossEntropyLoss（Next Token Prediction）**。它们的区别只在于"哪些位置的 token 参与 loss"（即 `loss_mask` 怎么标），而不是 loss 函数本身。

### 唯一的 loss 形式

$$
\mathcal{L} \;=\; -\frac{1}{|\mathcal{S}|}\sum_{k \in \mathcal{S}} \log p_\theta\!\bigl(\text{tok}_{k+1}\bigm|\text{tok}_{\le k}\bigr)
\quad\text{其中}\quad p_\theta(\cdot\mid\text{tok}_{\le k}) = \text{softmax}\bigl(W_\text{lm\_head}\,h_k\bigr) \in \mathbb{R}^{V}
$$

- $\mathcal{S}$ = 参与 loss 的 token 位置集合（由 `loss_mask==1` 决定，三种方式的差异**只在这里**）。
- $V \approx 176\,000$ = 词表大小（Qwen3 原 151 669 词 + 3×8192 个 Itemic Token + 2 个特殊 token，再对齐到 256 的倍数）。
- **不是 BCE，不是 sigmoid，不是 0/1 二分类交叉熵**。始终是 V 维的 softmax + 取正确 token 那一维的 -log 概率。

代码实现在 `pretrain/onerec_llm/losses/ce.py::CrossEntropyLoss.forward`：

```python
logits_flat  = logits.float().reshape(-1, vocab_size)   # (T, V)   V≈176k
labels_flat  = labels.reshape(-1)                        # (T,)     值域 [0, V)，非监督位=-100
per_token_loss = F.cross_entropy(
    logits_flat, labels_flat,
    ignore_index=self.ignore_index,     # -100 → 自动跳过（等价于 loss_mask==0）
    reduction="none",
)                                                        # (T,)
loss = per_token_loss.sum() / (labels_flat != -100).sum()
```

### `labels` 到底是什么

在 `pretrain/recipes/train_qwen3.py::compute_forward_backward` 里构造：

```python
labels = torch.cat([input_ids[:, 1:], pad], dim=-1)                    # (1, T) 自回归右移
labels = labels * loss_mask + loss_fn.ignore_index * (1 - loss_mask)   # 屏蔽 loss_mask=0 的位置
```

- `labels[k]` = 位置 k 需要预测的**下一个 token 的 id**，取值 `[0, V)`（不是 0/1！）。
- `loss_mask[k]==0` 的位置 `labels[k]` 被置为 `-100`，`F.cross_entropy` 会跳过。
- **等价定义**：`loss_mask` 就是"哪些位置需要预测下一 token"的选择器。

三种方式的差异用一张 loss_mask 图就能说清（沿用 §0 的 U1 在 t₆ 的曝光 C 那条日志片段，简化为 item 粒度）：

```
                       item A     item B     item C
                       ─────     ─────     ─────
Naive Impression:  Sample₆ = [ A , B , C ]
                   loss_mask   1   1   1        ← A, B, C 全参与，A/B 又被算了一次

User-Centric:      Sample U1 = [ A , B , C ]
                   loss_mask   1   1   1        ← 与 Naive 相同，但整个用户只出现一次

New Impression Only: Sample₆ = [ A , B , C ]
                     loss_mask   0   0   1      ← A, B 是 context，仅 C 有 loss
```

### 一个具体数值走一遍（同一个位置）

设某位置 k 处，模型 logits 为 $(z_1, ..., z_V) \in \mathbb{R}^V$，真实下一 token id 为 $y$（例如 $y = 155000$，是某个 Itemic Token）。

- Softmax：$p_i = e^{z_i} / \sum_j e^{z_j}$，得到 V 维分布。
- Loss：$-\log p_y = -\log \dfrac{e^{z_y}}{\sum_j e^{z_j}}$。

三种组织方式在**这个位置**上的 loss 值完全一样；它们只是决定"这个位置要不要进 $\mathcal{S}$"。所以：

- **Naive**：一个 item 出现的每次都要 forward 一次这个 loss，可能算 4 次。
- **User-Centric**：一个 item 在其所在用户的样本里出现一次，只算 1 次；但同一次 forward 里同时算了 A→B, B→C 两个跳转（时序泄漏来源）。
- **NIO**：这个 item 只在它被作为 target 的那次曝光里算 1 次；其他所有它作为 context 出现的场合，`loss_mask=0`，$-\log p_y$ 根本不参与反传。

### 分类类任务（如 label_pred 的"是/否"）怎么办？

也是 next-token multi-class CE。做法（见 `data/onerec_data/sft/label_pred.py`）：
- 把二分类标签 `0/1` 翻译成中文字符 `"否"` / `"是"`；
- 拼在 assistant 段作为**一个 token（或极少几个 token）**；
- 训练 loss = 在那几个 token 位置上做 V 维 softmax CE。

真实的 0/1 只在**评测阶段**通过后处理还原（取 `"是"`/`"否"` 两个候选 token 的 logprob 差 → sigmoid → 作为概率 → 算 AUC）。**训练里从头到尾没有出现 BCE / `sigmoid_cross_entropy` / 0/1 label**。

### 结论

**Pretrain / SFT 阶段（三种样本组织方式共通）：**
- Loss 类型 = **token 级 V 维 softmax 多分类交叉熵**，别名 "next-token prediction loss"。
- Loss 的目标 = 下一个 token 的 id（从 176k 个候选里挑一个）。
- 三种方式的差异 = `loss_mask` 里哪些位置是 1（即"在哪些位置上做 next-token 预测"）。
- **不涉及**二元交叉熵 / sigmoid / 0/1 label / 分类头。

**RL 阶段（GRPO，`verl_rl/recipe/onerec/`）**：完全不用 CE，用 PPO/GRPO 的策略梯度 loss + KL 正则，见文末 §8。

---



| 项目 | Naive Impression | User-Centric | New Impression Only |
| :--- | :--- | :--- | :--- |
| 样本 = ? | 一次曝光 | 一个用户 | 一次曝光 |
| 样本数 | 曝光总数 N | 用户总数 U | 曝光总数 N |
| 单样本 token 长度 | 变长，含历史 | 长（整段行为流） | 变长，含历史 |
| Loss 位置 | 全序列 | 全序列 | 仅 target 段 |
| 每 item 计 loss 次数 | 冗余 O(该 item 在样本中的出现次数) | 1 次（在其所属用户样本里） | 1 次（在其被曝光的那条样本里） |
| Batch 组织 | 独立 padding 或 packing | 通常 packing 塞多用户 | 独立 padding 或 packing |
| Batch 内部是否需按时间序 | 无所谓 | 无所谓 | 无所谓（微观） |
| Batch 之间是否需按时间序 | 无所谓 | 无所谓（静态数据集） | **必须**按时间偏序 |
| 时序数据泄漏 | 弱 | 强 | 无 |
| 流行度偏差放大 | 严重 | 中 | 无 |
| 支持流式/增量训练 | 部分 | 否 | 是 |
| 训练-推理形态一致 | 中 | 差（推理时不是整段历史） | 好 |
| 实现复杂度 | 低 | 中 | 中高（需流式基建） |
| 典型采用 | 早期 TIGER-style 尝试 | **本仓库开源 pretrain** | **快手 OneRec-V2 线上** |

---

## 补. 代码层落地对照（相对本仓库现状）

本仓库开源版 pretrain 走 User-Centric。要改成 New Impression Only 需要动的最小代码集合：

1. **数据脚本层**（`data/onerec_data/pretrain/video_rec.py`）：
   - 现在做的：把一个用户的 `hist_pids[-512:] + target_pids[:10]` 全部拼一起当一个 segments 样本。
   - 要改成：把用户完整曝光时序 $[i_1, i_2, ..., i_L]$（带时间戳）**每个位置切一条样本** ——
     ```python
     for k in range(1, L):
         yield {
             'context_sids': [pid2sid[p] for p in hist[max(0,k-CTX):k]],
             'target_sid'  : pid2sid[hist[k]],
             'timestamp'   : timestamp[k],
             'source'      : 'RecIF_VideoRec_NIO',
         }
     ```
2. **样本格式**：从 `segments` 改为 `messages`（chat 格式）：
   ```json
   {
     "messages": [
       {"role": "user",      "content": "<sid1><sid2>..."},   ← context
       {"role": "assistant", "content": "<sid_target>"}       ← target（唯一 loss 段）
     ]
   }
   ```
   —— 这样走 `Qwen3ChatCompletionParquetDataset._process_chat`，其 `_get_assistant_mask` 会天然只在 `<|im_start|>assistant\n...<|im_end|>` 段设 `loss_mask=1`。**开源 SFT 已经是这个结构了**。
3. **Dataloader 层**：
   - 关掉全局 shuffle（或把 `LocalShuffleBuffer` 窗口调到很小），确保文件按时间戳排序消费。
   - 若数据分片按 `timestamp` 分文件（如 `2025-08-10_hour_09.parquet`），worker 按顺序拉取即可。
4. **训练循环层**：无需大改；FSDP 每步的 SGD 更新等价于流式的"批增量"。真正的"分钟级流"需要把静态 parquet 换成 Kafka consumer + checkpoint hot-swap，那是另一层工程。

**观察**：**开源 SFT 阶段的样本形态（chat + `only_assistant_loss=true`）在数学上就是简化版的 New Impression Only**——只不过 target 段可能有多个 item（`target_pids[:10]` 一次性预测），且数据是静态 dump 不是流式。这也侧面印证 OneRec-V2 的做法是 SFT 数据组织向 pretrain 阶段的自然延伸。

---

## 补.8. RL 后训练阶段用不用 0/1 label？—— 不用，用 reward + policy gradient

RL 阶段（`verl_rl/recipe/onerec/`，GRPO 算法）的样本组织和 loss **和前面三种"pretrain-style"完全不同一层**。它不再最小化下一 token 的 CE，而是最大化"奖励期望"。0/1 二元交叉熵仍然**没有出现**。

### 8.1 样本结构

沿用 SFT 样本的 chat 格式（一条 = user prompt + 期望的 assistant 输出）：

```python
row = {
    "prompt":   [{"role": "user", "content": "根据历史 <sid...>，请推荐下一个视频。/think"}],
    "reward_model": {
        "ground_truth": "<sid_begin><s_a_..>...<sid_end> <sid_begin>...",   # ground-truth SID 字符串
        "style": "rule",
    },
    "source": "RecIF_VideoRec",
}
```

**注意**：`ground_truth` 是一段 **SID 文本**（可能包含多个正确 SID），**不是 0/1 标签**。

### 8.2 训练时发生什么

以 `run_grpo.sh` + `main_onerec_ppo.py` 为例，一次 RL 迭代：

1. **Rollout（两阶段生成）**：`onerec_vllm_rollout.py`
   - Stage 1：用 vLLM 让当前 actor 模型生成 `<think>...</think>`（最多 1024 token 的推理内容）。
   - Stage 2：在 `<|sid_begin|>` 之后做 **beam search**，`beam_size=32`，每次只生成 3 个 codebook token（一条 SID）。
   - 一个 prompt 产出 32 条候选 SID，每条候选一个 (prompt, response) pair。
2. **Reward 计算**：`onerec_recipe.py::compute_score`
   ```python
   {
       "score":              first_sid_hit_reward(pred, gt),   # Pass@1：第一条 SID 是否命中 → {0.0, 1.0}
       "format_reward":      think_format_reward(pred),        # <think>...</think> 是否格式对 → {0.0, 1.0}
       "partial_hit_reward": partial_hit_reward(pred, gt),     # 三层 SID 匹配加权 → {0, 1, 10, 100}
       "hit_reward":         hit_reward(pred, gt),             # 命中集 / 预测集 → [0, 1] 连续值
       "pass_rate":          pass_rate(pred, gt),              # 有无交集 → {0.0, 1.0}
       "pass_at_1":          first_sid_hit_reward(pred, gt),
   }
   ```
   看到没有 —— reward 里**确实存在 0/1 值**（比如 `pass_at_1`），但它不是"训练监督信号 label"，而是 RL 的 **scalar reward** $r$，用途完全不同：
   - Label（分类 CE 里的 0/1）：告诉模型"正确答案是 0 类还是 1 类"，直接进 CE 公式。
   - Reward（RL 里的标量）：告诉模型"你刚才那条 rollout 好不好"，用于调 policy 的对数概率梯度。
3. **优势估计（GRPO）**：`algorithm.adv_estimator=grpo`
   32 条候选构成一个 group，reward 减去 group 均值 → 组内相对优势 $A_i$：
   $$A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G) + \varepsilon}$$
4. **Policy Gradient loss**（PPO clip 版，见 verl `verl/trainer/ppo/core_algos.py`）：
   $$
   \mathcal{L}_{\text{policy}} \;=\; -\,\mathbb{E}\!\left[\min\bigl(\rho_t A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,A_t\bigr)\right]
   \;+\; \beta\,\text{KL}\bigl(\pi_\theta \Vert \pi_{\text{ref}}\bigr)
   $$
   其中 $\rho_t = \pi_\theta(\text{tok}_t\mid\text{ctx}) / \pi_{\text{old}}(\text{tok}_t\mid\text{ctx})$ 是新旧 policy 的概率比。`KL_LOSS_COEF=0.001`，`clip_ratio_high=0.28`。

### 8.3 关键对照

| 阶段 | Loss 类型 | 监督信号 | 有无 0/1 label |
| :--- | :--- | :--- | :--- |
| Pretrain / SFT / 蒸馏 | Token 级多分类 CE（softmax over V≈176k） | 下一 token 的 id | ❌ 无（分类任务的 "是/否" 也是 token） |
| **RL (GRPO)** | **Policy Gradient (PPO-clip) + KL** | **标量 reward $r \in \mathbb{R}$**（可能取 {0,1} 也可能连续） | ❌ 无（reward 不是 label） |

### 8.4 一个常见误解

看到 `pass_at_1` 返回 `0.0` 或 `1.0`，容易误以为"这就是 0/1 label 在做二元 CE"。**不是**。区别在于：

- 若是 BCE：`loss = -[y log ŷ + (1-y) log(1-ŷ)]`，`y ∈ {0, 1}` 是硬标签，`ŷ` 是模型直接输出的 sigmoid 概率。
- 而 RL 里 `y=0/1` 是 **rollout 后的 reward**，先做**组内标准化**得优势 $A$，再乘上**已经 rollout 出来的 token 序列**的对数概率梯度。模型没有一个"输出 0 或 1 的 sigmoid 头"，rollout 依然是自回归生成 SID token；reward 只是"权重"。

也就是说 —— **数值上出现 0/1 只是因为这些奖励函数是二值的（命中/未命中）**；如果 reward 函数换成 `partial_hit_reward`（返回 0, 1, 10, 100 之类），公式一字不改，loss 依然是 policy gradient 而不是任何"分类 CE"。

### 8.5 蒸馏阶段（`verl_distillation/recipe/onpolicy_distill/`）呢？

也不是 CE-with-0/1-label：`algorithm.adv_estimator=on_policy_distill`，advantage 由学生 logprob 减去教师 logprob 得到，然后走同样的 PPO-clip loss。等价于"教师模型比学生看好这条 rollout"就给它一个正 advantage。**监督信号是教师的 log 概率密度差，不是硬 0/1**。

### 结论

- 从 pretrain → SFT → 蒸馏 → RL，**整条 pipeline 都没用过 0/1 二元交叉熵**。
- Pretrain/SFT/蒸馏 阶段的 label 是 **token id**（V 维多分类 CE）。
- RL 阶段的"信号"叫 **reward**，是标量（可能取值恰好为 0/1，也可能是连续值），进入 policy gradient 公式，不是分类 loss。
- 二分类任务（`label_pred` 的"是/否"）的 0/1 只在**评测阶段**通过 `"是"/"否"` 两个候选 token 的 logprob 反推概率再算 AUC，**训练时 0/1 也从没直接出现在 loss 里**。

## 参考链接
https://www.huaxiaozhuan.com/applications/recommendation/sequential_recommendation/chapters/2025_OneRecV2_TechReport.html
