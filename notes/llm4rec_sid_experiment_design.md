# SID 辅助酒店序列建模实验设计

> 基于 `notes/llm4rec_sid.md` 的 LLM4DLRMs 调研，结合当前酒店排序模型 `zxhtl_seq_v2_5_nosid_meanpool_M3oE_f_idgate_300` 的实际架构，设计从简单到复杂的 SID 辅助建模实验方案。

## 目录

- [1. 当前状态分析](#1-当前状态分析)
- [2. Slot 规划总览](#2-slot-规划总览)
- [3. 实验设计](#3-实验设计)
  - [实验 1：候选侧 SID 整体 hash 作为离散特征](#实验-1候选侧-sid-整体-hash-作为离散特征)
  - [实验 2：候选侧 SID 逐层拆解 hash](#实验-2候选侧-sid-逐层拆解-hash)
  - [实验 3：序列侧 SID 整体 hash 加入 sideinfo](#实验-3序列侧-sid-整体-hash-加入-sideinfo)
  - [实验 4：候选+序列双侧 SID + Target Attention Query 扩展](#实验-4候选序列双侧-sid--target-attention-query-扩展)
  - [实验 5：SID Prefix 交叉匹配特征](#实验-5sid-prefix-交叉匹配特征)
  - [实验 6：SID 辅助预测任务（Auxiliary Loss）](#实验-6sid-辅助预测任务auxiliary-loss)
  - [实验 7：多 SID 类型加权融合](#实验-7多-sid-类型加权融合)
  - [实验 8：SID 对比学习辅助（Anchor-Image Contrastive）](#实验-8sid-对比学习辅助anchor-image-contrastive)
- [4. 实验优先级与推荐顺序](#4-实验优先级与推荐顺序)
- [5. 各实验与论文方法的对应关系](#5-各实验与论文方法的对应关系)

---

## 1. 当前状态分析

### 1.1 可用的 SID 特征

当前数据中有 **4 种 SID**，每种为 **3 层**语义ID（如 `[3, 4, 6]`）：

| SID 名称 | 含义 | 候选侧 Slot | 序列侧 Slot |
|----------|------|-----------|-----------|
| `detailpage_anchor_sid` | 详情页锚点多模态 SID | 281 | 421 |
| `detailpage_image_sid` | 详情页图片多模态 SID | 282 | 422 |
| `listpage_anchor_sid` | 列表页锚点多模态 SID | 283 | 423 |
| `listpage_image_sid` | 列表页图片多模态 SID | 284 | 424 |

**两个使用位置**：
- **候选侧（candidate item）**：slot 281-284，作为候选酒店的属性特征
- **序列侧（sequence sideinfo）**：slot 421-424，作为用户历史行为序列中每个酒店的 sideinfo

### 1.2 当前 SID 的 hash 方式

当前 SQL 中，SID 字符串**整体做 murmurHash**：

```sql
-- 以 detailpage_anchor_sid 为例
-- SID 值为字符串 "[3, 4, 6]"，整体 hash
concat(dw_traindb.murmurHash(concat(coalesce(IFB.detailpage_anchor_sid, '-1'), ':281')), ':281')
```

即 `murmurHash("[3, 4, 6]:281"):281`，将整个 3 层 SID 作为一个离散特征值。

### 1.3 当前模型架构（nosid 版本）

模型名为 `nosid`，意味着**当前模型未使用 SID 特征**：
- `slotsS` 中不包含 slot 281-284
- `seq_slots_groups` 中不包含 slot 421-424
- `target_attn_cand_slots` 不包含 SID 相关 slot

**模型核心架构**：
- 候选侧 slotsS + slotsM → 拼接 embedding
- 序列侧 slotsL → target-attention pooling（query 由候选特征构建）
- M3oE 多任务多场景 MoE → 3 个任务（click, duration, CVR）
- PPNet ID gate（hotel_id, city_id, uid 做门控）

---

## 2. Slot 规划总览

为支持各实验，规划以下 slot 分配：

### 2.1 候选侧 Slot

| Slot | 特征 | 实验 |
|------|------|------|
| 281 | `detailpage_anchor_sid` 整体 hash | Exp1+ |
| 282 | `detailpage_image_sid` 整体 hash | Exp1+ |
| 283 | `listpage_anchor_sid` 整体 hash | Exp1+ |
| 284 | `listpage_image_sid` 整体 hash | Exp1+ |
| **286** | `detailpage_anchor_sid` 第1层 | Exp2+ |
| **287** | `detailpage_anchor_sid` 第2层 | Exp2+ |
| **288** | `detailpage_anchor_sid` 第3层 | Exp2+ |
| **289** | `detailpage_image_sid` 第1层 | Exp2+ |
| **290** | `detailpage_image_sid` 第2层 | Exp2+ |
| **291** | `detailpage_image_sid` 第3层 | Exp2+ |
| **292** | `listpage_anchor_sid` 第1层 | Exp2+ |
| **293** | `listpage_anchor_sid` 第2层 | Exp2+ |
| **294** | `listpage_anchor_sid` 第3层 | Exp2+ |
| **295** | `listpage_image_sid` 第1层 | Exp2+ |
| **296** | `listpage_image_sid` 第2层 | Exp2+ |
| **297** | `listpage_image_sid` 第3层 | Exp2+ |

### 2.2 序列侧 Slot

| Slot | 特征 | 实验 |
|------|------|------|
| 421 | `detailpage_anchor_sid` 整体 hash | Exp3+ |
| 422 | `detailpage_image_sid` 整体 hash | Exp3+ |
| 423 | `listpage_anchor_sid` 整体 hash | Exp3+ |
| 424 | `listpage_image_sid` 整体 hash | Exp3+ |
| **425** | `detailpage_anchor_sid` 第1层 | Exp4+ |
| **426** | `detailpage_anchor_sid` 第2层 | Exp4+ |
| **427** | `detailpage_anchor_sid` 第3层 | Exp4+ |
| **428** | `detailpage_image_sid` 第1层 | Exp4+ |
| **429** | `detailpage_image_sid` 第2层 | Exp4+ |
| **430** | `detailpage_image_sid` 第3层 | Exp4+ |
| **431** | `listpage_anchor_sid` 第1层 | Exp4+ |
| **432** | `listpage_anchor_sid` 第2层 | Exp4+ |
| **433** | `listpage_anchor_sid` 第3层 | Exp4+ |
| **434** | `listpage_image_sid` 第1层 | Exp4+ |
| **435** | `listpage_image_sid` 第2层 | Exp4+ |
| **436** | `listpage_image_sid` 第3层 | Exp4+ |

---

## 3. 实验设计

### 实验 1：候选侧 SID 整体 hash 作为离散特征

**动机**：最简单的 SID 使用方式——将 4 种 SID 作为候选酒店的离散属性特征直接加入模型。对应 YouTube Semantic IDs 和 QARM 中"SID 作为物品侧特征"的做法。

**对应论文方法**：
- YouTube（§2.1）：SID 替换 Video ID 作为排序模型输入特征
- QARM（§2.4）：VQ/RQ codes 作为物品侧可学习 embedding 特征
- Meta Prefix N-gram（§2.2）：SID 作为 DLRM 顶层稀疏特征

**改动点**：

1. **SQL**：无需修改（slot 281-284 已在数据中）
2. **模型配置**：
   - 在 `slotsS_str` 中添加 `281,282,283,284`
   - 可选：在 `target_attn_cand_slots_str` 中添加 `281,282,283,284`（让 SID 参与 target attention 的 query 构建）
3. **模型代码**：无需修改

**预期效果**：
- 低改动成本，验证 SID 是否对模型有价值
- 整体 hash 的 SID 相当于一个"多模态内容指纹"，相似酒店会 hash 到相近的值（如果 SID 构造合理，部分酒店会共享同一 hash 值 → 参数级泛化）
- 预期 AUC/GAUC 有小幅提升，冷启动酒店提升更明显

**风险**：整体 hash 破坏了 SID 的层次结构（第1层粗粒度、第2层中粒度、第3层细粒度），不同酒店即使只第3层不同也会 hash 到完全不同的值，丢失泛化能力。

---

### 实验 2：候选侧 SID 逐层拆解 hash

**动机**：将每个 3 层 SID 拆成 3 个独立的离散特征，每层单独 hash。这样：
- 第1层（粗粒度）的 embedding 被更多酒店共享 → 泛化能力强
- 第3层（细粒度）的 embedding 更独特 → 记忆能力强
- 模型可以分别学习不同粒度的表示

对应 QARM 中"每层 RQ code 独立 embedding table"的做法。

**对应论文方法**：
- QARM（§2.4.3）：RQ codes 每层有独立 Embedding Table
- YouTube（§2.1.3, 方案一）：逐层独立 Embedding
- Meta Prefix N-gram（§2.2.2）：Prefix N-gram 每层独立 Table

**改动点**：

1. **SQL**：新增 12 个 slot（286-297），对每个 SID 提取各层单独 hash

```sql
-- 以 detailpage_anchor_sid 为例，提取各层
-- 假设 SID 格式为 "3,4,6"（或 "[3, 4, 6]"，需根据实际格式调整 split 逻辑）

-- 在 item_feature_base CTE 中新增:
split(regexp_replace(detailpage_anchor_sid, '[\\[\\]\\s]', ''), ',')[0] AS dp_anchor_sid_l1,
split(regexp_replace(detailpage_anchor_sid, '[\\[\\]\\s]', ''), ',')[1] AS dp_anchor_sid_l2,
split(regexp_replace(detailpage_anchor_sid, '[\\[\\]\\s]', ''), ',')[2] AS dp_anchor_sid_l3,
-- 其他 3 种 SID 同理

-- 在 zxhtl_mapped_final 中新增 hash:
concat(dw_traindb.murmurHash(concat(coalesce(ifb.dp_anchor_sid_l1, '-1'), ':286')), ':286') as f286,
concat(dw_traindb.murmurHash(concat(coalesce(ifb.dp_anchor_sid_l2, '-1'), ':287')), ':287') as f287,
concat(dw_traindb.murmurHash(concat(coalesce(ifb.dp_anchor_sid_l3, '-1'), ':288')), ':288') as f288,
-- 其他 3 种 SID × 3 层 = 9 个 hash 同理
```

2. **模型配置**：
   - `slotsS_str` 添加 `286,287,288,289,290,291,292,293,294,295,296,297`（12 个新 slot）
   - 可以同时保留整体 hash 的 281-284（消融对比），或仅用逐层的 286-297
3. **模型代码**：无需修改

**预期效果**：
- 优于实验 1：逐层拆解保留了 SID 的层次语义结构
- 第1层 embedding 被更多酒店共享，泛化能力更强（参考 YouTube 论文的 "meaningful collisions"）
- 冷启动酒店：即使第3层（细粒度）embedding 未充分学习，第1层（粗粒度）仍能提供有意义的泛化信号
- 额外参数：12 × embedding_size（很小）

**消融建议**：
- Exp2a：仅用第1层（粗粒度）→ 验证泛化能力
- Exp2b：仅用第3层（细粒度）→ 验证记忆能力
- Exp2c：用全部3层 → 预期最优

---

### 实验 3：序列侧 SID 整体 hash 加入 sideinfo

**动机**：将 SID 加入用户行为序列的 sideinfo，让模型能够从用户历史中学习"这个用户偏好什么语义类型的酒店"。对应 YouTube 论文中"用户历史序列用 SID 表示"的做法。

**对应论文方法**：
- YouTube（§2.1.7）：用户历史序列每个视频用 SID 表示 → SPM → target attention
- QARM（§2.4.3, 使用方式二）：用户最近正向交互物品序列的 SID → Attention Pooling
- SIDE（§2.3.5）：用户广告历史序列每个广告用 SID 表示

**改动点**：

1. **SQL**：无需修改（slot 421-424 已在序列数据中）
2. **模型配置**：
   - 在 `seq_slots_groups_str` 中为序列组添加 `421:422:423:424`
   - 例如原来是 `301:305:...:320:400:...:420:314`，改为 `301:305:...:320:400:...:420:421:422:423:424:314`
   - 注意 recency slot（314）保持在最后
3. **模型代码**：无需修改（target attention 自动处理新增的 content slot）

**预期效果**：
- 用户序列中每个酒店多了 4 个 SID embedding → 序列表示更丰富
- target attention 能根据候选酒店特征，从历史中找到"语义相似"的酒店并加权
- 预期 GAUC 提升更显著（GAUC 衡量用户内排序能力，序列建模是关键）

**注意点**：
- 序列 content slot 数增加 → `target_attn_out_dim` 会增大（= content_size × embedding_size）
- 4 个 SID slot 加入后，content_size 从原来的 ~25 增加到 ~29，attention query 维度需相应调整
- 建议同时对比：仅加 SID 到序列 vs. 同时加到候选+序列

---

### 实验 4：候选+序列双侧 SID + Target Attention Query 扩展

**动机**：在实验 2+3 的基础上，将候选侧逐层 SID 也加入 target attention 的 query，使得 attention 能基于 SID 语义进行更精确的序列匹配。

**对应论文方法**：
- YouTube（§2.1.8）：候选视频 SID 参与构建 query
- QARM（§2.4.3, 使用方式三）：候选物品 SID 在用户历史中做匹配

**改动点**：

1. **SQL**：同时做实验 2（候选逐层）和实验 3（序列整体/逐层）的 SQL 修改
2. **模型配置**：
   - 候选侧逐层 SID 加入 `slotsS_str`
   - 序列侧 SID 加入 `seq_slots_groups_str`
   - **关键**：将候选侧 SID slot 加入 `target_attn_cand_slots_str`
   ```
   # 原来: target_attn_cand_slots_str = '2,6,7,10,70,17,19,20,1000'
   # 新增: 281,282,283,284  (整体hash) 或 286-297 (逐层)
   target_attn_cand_slots_str = '2,6,7,10,70,17,19,20,1000,281,282,283,284'
   ```
3. **模型代码**：无需修改（`build_candidate_query` 自动从 plan 中切出新 slot）

**预期效果**：
- Query 中包含 SID 信息 → attention 能"看到"候选酒店的语义类型
- 当候选酒店的 SID 第1层与某个历史酒店的 SID 第1层相同时，对应的 embedding 会共享，attention 自然给予更高权重
- 这是最符合 YouTube/QARM 论文做法的完整方案

**进阶变体**：
- Exp4a：候选侧用整体 hash SID (281-284) 加入 query
- Exp4b：候选侧用逐层 SID (286-297) 加入 query → 预期更优（粒度更细）
- Exp4c：序列侧也用逐层 SID (425-436) → 最完整方案

---

### 实验 5：SID Prefix 交叉匹配特征

**动机**：利用 SID 的层次结构，在候选酒店和用户历史之间构造显式的匹配特征。这是 QARM 论文中效果最好的设计之一——"一类码匹配"和"二类码匹配"交叉特征。

**对应论文方法**：
- QARM（§2.4.3, 使用方式三）：候选物品 RQ code 在用户历史中找"一类码匹配"和"二类码匹配"子序列 → Attention → cross_feature_emb
- TRM（§2.8）：协同感知匹配特征

**核心思想**：

```
候选酒店 detailpage_anchor_sid = (3, 4, 6)
用户历史酒店的 detailpage_anchor_sid:
  酒店A: (3, 4, 8)   ← 前2层匹配 (3,4) = "二类码匹配"
  酒店B: (3, 7, 2)   ← 前1层匹配 (3) = "一类码匹配"
  酒店C: (5, 1, 9)   ← 无匹配

一类码匹配子序列 = [酒店A, 酒店B]  → Attention Pooling → cross_feat_1
二类码匹配子序列 = [酒店A]         → Attention Pooling → cross_feat_2
→ 拼接到模型输入
```

**改动点**：

1. **SQL**：无需额外修改（逐层 SID 已在序列中，实验 4 的基础上）
2. **模型代码**：**需要新增模块**

```python
# 在 build_model 中新增 SID 交叉特征模块
def build_sid_cross_features(self, cand_sid_embs, seq_sid_embs, seq_mask):
    """
    cand_sid_embs: dict, 每种SID每层的候选embedding
      e.g. {'dp_anchor': [cand_l1_emb, cand_l2_emb, cand_l3_emb]}
    seq_sid_embs: dict, 每种SID每层的序列embedding
      e.g. {'dp_anchor': [seq_l1_emb, seq_l2_emb, seq_l3_emb]}  # each [B, L, D]
    seq_mask: [B, L]
    """
    cross_feats = []
    for sid_type in ['dp_anchor', 'dp_image', 'lp_anchor', 'lp_image']:
        for level in range(3):  # 3 layers
            # 候选第 level 层 SID embedding: [B, D]
            cand_l = cand_sid_embs[sid_type][level]
            # 序列第 level 层 SID embedding: [B, L, D]
            seq_l = seq_sid_embs[sid_type][level]

            # 计算相似度（点积或余弦）
            sim = tf.reduce_sum(
                tf.expand_dims(cand_l, 1) * seq_l, axis=-1)  # [B, L]

            # 用相似度作为 attention weight，对序列做加权聚合
            sim_masked = sim * seq_mask + (1 - seq_mask) * (-1e9)
            attn_w = tf.nn.softmax(sim_masked, axis=-1)  # [B, L]
            # 对原始序列 content token 做加权
            weighted = tf.reduce_sum(
                seq_l * tf.expand_dims(attn_w, -1), axis=1)  # [B, D]
            cross_feats.append(weighted)

    return tf.concat(cross_feats, axis=-1)  # [B, 4*3*D]
```

3. **模型配置**：将 cross feature 输出拼接到 `dnn_input`

**预期效果**：
- **预期价值最高的实验之一**——直接利用 SID 层次结构构造交叉特征
- 一类码匹配捕获粗粒度兴趣（"用户看过同类型的酒店"）
- 二类码匹配捕获更精确的兴趣（"用户看过几乎同款酒店"）
- 类似 QARM 的"交叉特征"设计，已被快手验证有效

**风险**：
- 代码改动较大，需要新增模块
- 需要逐层 SID 在序列和候选两侧都可用

---

### 实验 6：SID 辅助预测任务（Auxiliary Loss）

**动机**：将 SID 作为辅助预测目标，添加额外的 loss 项，迫使模型的用户序列表示能"预测"候选酒店的 SID。这可以增强模型对 SID 语义的理解能力。

**对应论文方法**：
- DIG（§2.9）：用判别式 BCE 损失端到端驱动 Tokenizer
- MMQ（§7.5）：推荐损失反向传播微调码本
- 通用 Auxiliary Task 思路

**核心思想**：

```
主任务: P(click | user_seq, candidate) → BCE Loss

辅助任务: 从用户序列表示预测候选酒店的 SID
  user_seq_repr → MLP → P(SID_l1) → CE Loss (第1层)
  user_seq_repr → MLP → P(SID_l2) → CE Loss (第2层)
  user_seq_repr → MLP → P(SID_l3) → CE Loss (第3层)

总损失 = 主任务Loss + α × 辅助任务Loss
```

**改动点**：

1. **模型代码**：新增辅助 loss 模块

```python
# 在 build_model 中，取 target attention 输出作为 user_seq_repr
# 候选侧逐层 SID 作为 label

def build_sid_auxiliary_loss(self, user_seq_repr, cand_sid_layer_ids):
    """
    user_seq_repr: [B, D_seq]  target attention 输出
    cand_sid_layer_ids: list of [B] int tensors, 每层 SID 的原始 id
    """
    aux_losses = []
    for level, sid_ids in enumerate(cand_sid_layer_ids):
        # 预测头: MLP → vocab_size logits
        with tf.variable_scope('sid_pred_level_{}'.format(level), reuse=tf.AUTO_REUSE):
            _, logits_list, _, _ = train_util.multilayer_perceptron(
                [user_seq_repr],
                [user_seq_repr.get_shape()[-1], 256, sid_vocab_size],
                ['none', 'swish', 'none'],
                name_pre='sid_aux_level_{}'.format(level))
            logits = logits_list[0]  # [B, vocab_size]
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=sid_ids, logits=logits))
            aux_losses.append(loss)

    return sum(aux_losses)  # 各层 loss 之和
```

2. **总损失修改**：
```python
self.loss = sum(self.loss_ops) + alpha * sid_aux_loss
# alpha 建议从 0.1 开始调
```

**预期效果**：
- 辅助任务迫使用户序列表征学习"SID 语义空间"的结构
- 相当于一个自监督信号，帮助序列表示更好地理解酒店的多模态内容
- 可能对 GAUC 有帮助（用户内排序需要更好的序列表示）

**风险**：
- 辅助 loss 的权重 α 需要仔细调节（太大会干扰主任务）
- SID vocab size 需要确认（每层的码本大小）
- 训练时间会略微增加

---

#### 6.1 补充：两种辅助任务方案深入对比

上面的核心思想是**方案 A：用原始 SID 作为分类标签（CrossEntropy）**。实际上还有另一种思路——**方案 B：用 SID Embedding 做对齐（InfoNCE/MSE）**。以下深入对比两种方案。

**方案 A（分类，即上面的核心思想）**：

```
user_seq_repr → MLP → logits [vocab_size] → CrossEntropy(label=raw_sid_id)

"从用户序列表征，预测候选酒店的 SID 属于哪个类别"
```

- SID 原始值（如第1层=3, 第2层=4, 第3层=6）作为分类标签
- 每层各自一个分类头，vocab_size = 该层的唯一 SID 值个数
- CrossEntropy 把所有"错误类别"一视同仁

**方案 B（对齐）**：

```
user_seq_repr → MLP → aligned_repr [D]
cand_sid_emb (from embedding table, stop_gradient) → target [D]
Loss = InfoNCE(aligned_repr, cand_sid_emb)  或  MSE / Cosine

"让用户序列表征对齐到候选酒店的 SID embedding 空间"
```

- 候选酒店的 SID 经过 hash + embedding lookup 得到的向量作为 target
- 用 InfoNCE（batch 内其他样本作负例）或 MSE 对齐两者

#### 6.2 方案分析：为什么方案 B 更优

**理由一：语义结构的保留**

方案 A 的 CrossEntropy 把所有错误类别等权惩罚：

```
方案 A: 
  SID 真实值 = 3 (商务酒店)
  预测 SID = 5 (也是商务酒店)  → 和预测 SID = 100 (度假村) 一样被惩罚 ❌

方案 B:
  SID embedding 真实值 ≈ [0.8, 0.3, -0.1] (商务酒店)
  预测 embedding ≈ [0.7, 0.4, -0.2] (也是商务酒店) → 损失小 ✅
  预测 embedding ≈ [-0.5, 0.9, 0.3] (度假村)     → 损失大 ✅
```

SID embedding 已经在主任务中被训练，蕴含了丰富的语义结构（相似酒店的 embedding 相近）。方案 B 自然地保留了这种结构。

**理由二：信息泄漏问题（⚠️ 最关键）**

当前模型中 `user_seq_repr` 是 **target attention 的输出**，而 target attention 的 **query 就是候选物品特征**：

```python
# train.py 中的代码:
cand_query = self.build_candidate_query(s_emb, m_emb, rv_feas_new)
# cand_query 包含候选酒店的各种特征（stop_gradient）
seq_attn_out = self.target_attention_pool_time_aware(query_vec=cand_query, ...)
```

这意味着：**候选物品的信息已经在 `user_seq_repr` 里了！**

- **方案 A（分类）**：用 `user_seq_repr` 预测候选酒店的 SID 类别 → 模型可能走捷径，直接从 query 中"泄漏"的候选信息反推 SID 类别，而不是真正学习用户兴趣结构。辅助任务变得太容易，**信息增益接近零**。
- **方案 B（对齐）**：对齐两个连续空间 → 虽然也有信息泄漏，但 InfoNCE 的目标是学习空间映射关系，仍有一定价值。

**解决方案**：两种方案的辅助任务都应该用**不含 target attention 的序列表征**（如 mean pooling），避免信息泄漏：

```
正确做法（两种方案通用）:
  hist_tokens (序列 content embedding, 不含候选信息)
       ↓ mean pooling
  seq_mean_repr ← 纯用户兴趣，不含候选信息 ✅
       ↓
  方案A: MLP → CE分类    或    方案B: MLP → InfoNCE对齐
```

#### 6.3 方案 B 推荐实现：Mean Pooling + InfoNCE

```python
# ========== 在 build_model 中新增 ==========

# Step 1: 对序列做 mean pooling（不含候选信息，避免信息泄漏）
# hist_tokens: [B, L, F_c*D] 来自序列侧 content slots
# hist_mask:   [B, L]
mask_e = tf.expand_dims(tf.cast(hist_mask, tf.float32), -1)  # [B, L, 1]
valid_cnt = tf.maximum(tf.reduce_sum(mask_e, axis=1), 1.0)   # [B, 1]
seq_mean_pooled = tf.reduce_sum(hist_tokens * mask_e, axis=1) / valid_cnt  # [B, F_c*D]

# Step 2: 投影到 SID embedding 同维度
emb_dim = self.config.parm.embedding_size
with tf.variable_scope('sid_aux_proj', reuse=tf.AUTO_REUSE):
    _, proj_out, _, _ = train_util.multilayer_perceptron(
        [seq_mean_pooled],
        [seq_mean_pooled.get_shape()[-1], 256, emb_dim],
        ['none', 'swish', 'none'],
        name_pre='sid_aux_mlp')
    aligned_repr = proj_out[0]  # [B, emb_dim]

# Step 3: 候选 SID embedding 作为 target（stop_gradient 防止梯度冲突）
# cand_sid_emb: 候选酒店的某个 SID 的 embedding，shape [B, emb_dim]
# 可以从 s_emb 中切出对应 slot 的 embedding
cand_sid_emb = tf.stop_gradient(cand_sid_embedding)  # [B, emb_dim]

# Step 4: InfoNCE loss（batch 内其他样本作为负例）
def infonce_loss(anchor, positive, temperature=0.1):
    """
    anchor:   [B, D]  用户序列投影后的表征
    positive: [B, D]  候选 SID embedding（stop_gradient）
    """
    anchor = tf.nn.l2_normalize(anchor, axis=-1)
    positive = tf.nn.l2_normalize(positive, axis=-1)
    logits = tf.matmul(anchor, positive, transpose_b=True) / temperature  # [B, B]
    labels = tf.range(tf.shape(logits)[0])  # 对角线是正例
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits))

# Step 5: 总损失
alpha = 0.05  # 建议从 0.05~0.1 开始调
sid_aux_loss = infonce_loss(aligned_repr, cand_sid_emb, temperature=0.1)
self.loss = sum(self.loss_ops) + alpha * sid_aux_loss
```

**为什么用 `stop_gradient(cand_sid_emb)`**：
- 主任务的梯度控制 SID embedding 的学习方向（为 CTR/CVR 优化）
- 辅助任务只负责让 `seq_mean_repr` 去"靠近"已学好的 SID embedding 空间
- 避免两个任务的梯度在 embedding 空间上打架

#### 6.4 方案 A 的实现（作为 Baseline 对比）

如果想先快速验证辅助任务是否有效，可以先跑方案 A 作为 baseline：

```python
# 方案 A: 用 mean_pooled（非 target_attn_out）做分类
# 需要获取候选 SID 的原始层 ID（从离散特征中解析）

def build_sid_ce_aux_loss(self, seq_mean_repr, cand_sid_layer_ids, sid_vocab_sizes):
    """
    seq_mean_repr:     [B, D]  序列 mean pooling 输出
    cand_sid_layer_ids: list of [B] int tensors, 每层 SID 的原始 id
    sid_vocab_sizes:    list of int, 每层的 vocab size
    """
    aux_losses = []
    for level, (sid_ids, vocab_size) in enumerate(
            zip(cand_sid_layer_ids, sid_vocab_sizes)):
        with tf.variable_scope('sid_pred_level_{}'.format(level), reuse=tf.AUTO_REUSE):
            _, logits_list, _, _ = train_util.multilayer_perceptron(
                [seq_mean_repr],
                [seq_mean_repr.get_shape()[-1], 256, vocab_size],
                ['none', 'swish', 'none'],
                name_pre='sid_aux_ce_level_{}'.format(level))
            logits = logits_list[0]  # [B, vocab_size]
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=sid_ids, logits=logits))
            aux_losses.append(loss)
    return sum(aux_losses)
```

#### 6.5 两种方案总结对比

| 维度 | 方案 A（CE 分类） | 方案 B（InfoNCE 对齐） |
|------|-----------------|---------------------|
| **语义结构** | ❌ 丢失（所有错类等权惩罚） | ✅ 保留（embedding 距离反映语义） |
| **信息泄漏风险** | ❌ 高（CE 太容易走捷径） | ⚠️ 中（需 stop_gradient + mean pool） |
| **正确做法** | 用 mean_pooled 替代 target_attn_out | 用 mean_pooled + stop_gradient target |
| **调参难度** | 低（CE 很稳定） | 中（温度 τ 和权重 α 需调） |
| **可解释性** | ✅ 高（直接看分类准确率） | 中（看 alignment loss 下降曲线） |
| **实现复杂度** | 需知道 vocab_size 和原始 SID id | 直接用现有 embedding，无需额外信息 |
| **推荐度** | ⭐⭐ 作为 baseline | ⭐⭐⭐⭐ 推荐方案 |

#### 6.6 补充风险与建议

**额外风险**（两种方案通用）：
- InfoNCE 的温度 τ 需要调（建议从 0.1 开始，过大导致梯度消失，过小导致不稳定）
- batch_size 影响 InfoNCE 的负例数量（batch 越大效果越好）
- 如果 SID embedding 质量本身不好（主任务还没学好），辅助任务可能引入噪声 → 建议在主任务预训练若干 step 后再开启辅助 loss（warmup 策略）

**建议的实验顺序**：
1. 先跑方案 A（CE）作为 baseline → 验证辅助任务这个方向是否有价值
2. 如果方案 A 有正向收益 → 切换到方案 B（InfoNCE）→ 预期更优
3. 如果方案 A 无收益 → 检查是否因信息泄漏导致（对比 target_attn_out vs mean_pooled）

---

### 实验 7：多 SID 类型加权融合

**动机**：4 种 SID 有不同的语义含义（详情页 vs 列表页，锚点 vs 图片），在序列建模中它们的重要性可能不同。使用注意力机制或门控机制自适应地加权不同 SID 类型。

**对应论文方法**：
- MMQ（§2.5/§11.3）：MoE 架构对多模态 SID 动态加权
- SIDE（§2.3.4）：VQ-Fusion 多任务融合

**核心思想**：

```
4 种 SID embedding (每种可以是整体或逐层):
  dp_anchor_emb, dp_image_emb, lp_anchor_emb, lp_image_emb

方案 A: 简单 Attention 加权
  gate_input = concat(4种SID_emb)
  gate_weights = softmax(MLP(gate_input))  # [B, 4]
  fused = Σ gate_weights[i] * SID_emb[i]   # [B, D]

方案 B: 页面类型感知的门控
  # 详情页 SID 和列表页 SID 分别处理
  detail_fused = fuse(dp_anchor, dp_image)
  list_fused = fuse(lp_anchor, lp_image)
  # 再融合
  final = concat(detail_fused, list_fused) → MLP → fused

方案 C: MoE 风格（参考 MMQ）
  Router(cand_features) → 4个权重
  Expert_1(dp_anchor), Expert_2(dp_image), Expert_3(lp_anchor), Expert_4(lp_image)
  → 加权聚合
```

**改动点**：

1. **模型代码**：新增融合模块，放在 target attention 之前或之后
2. **配置**：新增超参数（融合方式选择、gate 维度等）

**预期效果**：
- 自适应选择最有信息量的 SID 类型
- 详情页 SID 可能对转化更重要（用户深入了解后的偏好）
- 列表页 SID 可能对点击更重要（第一印象）
- 图片 SID vs 锚点 SID 反映不同的模态信息

**建议**：
- 先跑实验 4 验证 SID 整体价值，再尝试融合
- 融合方式建议从简单（方案 A: Attention 加权）开始

---

### 实验 8：SID 对比学习辅助（Anchor-Image Contrastive）

**动机**：对同一酒店，anchor SID（锚点/关键区域多模态）和 image SID（整体图片多模态）应捕获互补信息。通过对比学习让同一酒店的 anchor/image SID embedding 接近，不同酒店的远离，增强 SID embedding 的区分度。

**对应论文方法**：
- QARM（§2.4.2）：Batch-Contrastive 对齐 trigger-target 对
- DAS（§2.7）：六路对比对齐损失
- PLUM SID-v2（§7.4）：共现对比损失

**核心思想**：

```
对比损失: 同一酒店的 anchor SID embedding 和 image SID embedding 应该接近

正例: (hotel_A_anchor_emb, hotel_A_image_emb) → 距离近
负例: (hotel_A_anchor_emb, hotel_B_image_emb) → 距离远 (batch内其他酒店)

L_contrast = -log(exp(sim(anchor_i, image_i)/τ) / Σ_j exp(sim(anchor_i, image_j)/τ))
```

**改动点**：

1. **模型代码**：新增对比 loss

```python
def build_sid_contrastive_loss(self, anchor_emb, image_emb, temperature=0.1):
    """
    anchor_emb: [B, D] 某个SID类型的anchor embedding (候选侧)
    image_emb:  [B, D] 对应SID类型的image embedding (候选侧)
    """
    # L2 normalize
    anchor_norm = tf.nn.l2_normalize(anchor_emb, axis=-1)
    image_norm = tf.nn.l2_normalize(image_emb, axis=-1)

    # 相似度矩阵: [B, B]
    sim_matrix = tf.matmul(anchor_norm, image_norm, transpose_b=True) / temperature

    # 对角线是正例
    labels = tf.range(tf.shape(sim_matrix)[0])
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=sim_matrix))
    return loss
```

2. **总损失**：`self.loss += beta * contrastive_loss`

**预期效果**：
- 增强 SID embedding 空间的结构性
- anchor 和 image 的 embedding 互相补充
- 可能对冷启动有帮助（即使某一种 SID 信息不足，另一种可以补充）

**风险**：
- 对比学习在排序模型中的效果不稳定，需要调 τ 和 β
- 计算量增加（B×B 相似度矩阵）

---

## 4. 实验优先级与推荐顺序

| 优先级 | 实验 | 改动量 | 预期收益 | 风险 | 建议 |
|--------|------|--------|---------|------|------|
| **P0** | **Exp1: 候选侧整体 hash** | 极小（仅配置） | 低-中 | 极低 | **第一步必做**，验证 SID 基本价值 |
| **P0** | **Exp2: 候选侧逐层拆解** | 小（SQL+配置） | 中 | 低 | **与 Exp1 对比**，验证逐层拆解是否更优 |
| **P1** | **Exp3: 序列侧整体 hash** | 小（仅配置） | 中 | 低 | 验证序列 SID 价值 |
| **P1** | **Exp4: 双侧 + Query 扩展** | 中（SQL+配置） | **中-高** | 低 | **推荐重点实验**，最符合论文做法 |
| **P2** | **Exp5: Prefix 交叉特征** | 大（新增代码） | **高** | 中 | 参考 QARM，预期高价值 |
| **P2** | **Exp6: 辅助预测任务** | 大（新增代码） | 中-高 | 中 | 自监督信号，需调参 |
| **P3** | **Exp7: 多类型加权融合** | 中（新增代码） | 中 | 中 | 在 Exp4 基础上尝试 |
| **P3** | **Exp8: 对比学习** | 大（新增代码） | 中 | 高 | 探索性实验 |

**推荐实验路径**：

```
第一阶段 (1-2天):
  Exp1 → Exp2 → 对比: 整体hash vs 逐层拆解哪个更好
  → 确认 SID 对候选侧的价值

第二阶段 (2-3天):
  Exp3 → Exp4 → 双侧 SID + Query 扩展
  → 确认序列 SID 的价值和完整方案

第三阶段 (3-5天):
  Exp5 (交叉特征) 或 Exp6 (辅助任务)
  → 探索更高阶的 SID 使用方式
```

---

## 5. 各实验与论文方法的对应关系

| 实验 | 对应论文 | 论文中的具体做法 |
|------|---------|----------------|
| **Exp1** | YouTube §2.1, QARM §2.4 | SID 作为候选物品的离散特征，通过 embedding table 查表 |
| **Exp2** | QARM §2.4.3, Meta §2.2 | 每层 SID 独立 embedding table，逐层查表后 concat/sum |
| **Exp3** | YouTube §2.1.7, SIDE §2.3.5 | 用户历史序列中每个酒店用 SID 表示 → attention 聚合 |
| **Exp4** | YouTube §2.1.8, QARM §2.4.3 | 候选 SID 构建 query，序列 SID 作为 KV → target attention |
| **Exp5** | QARM §2.4.3 (使用方式三) | 候选 SID 在历史中做一类码/二类码匹配 → cross feature |
| **Exp6** | DIG §2.9, MMQ §7.5 | 辅助损失驱动模型学习 SID 语义空间 |
| **Exp7** | MMQ §11.3, SIDE §2.3.4 | MoE/注意力机制融合多模态 SID |
| **Exp8** | QARM §2.4.2, DAS §2.7 | 对比学习对齐不同视角的 SID 表示 |
