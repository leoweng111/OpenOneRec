# LLM4DLRMs：语义ID辅助判别式推荐模型训练调研报告

> 本文是 `semantic_id.md` 中"阶段三：量化为离散语义ID"小节的扩展，聚焦于 **LLM4DLRMs 范式**——将语义ID作为特征输入到判别式推荐模型（DLRM/DNN排序/双塔模型）中，辅助其训练和推理，而非用于生成式检索。

## 目录

- [1. 概述](#1-概述)
  - [1.1 LLM4DLRMs 范式的核心思想](#11-llm4dlrms-范式的核心思想)
  - [1.2 为什么用SID替代随机Item ID？](#12-为什么用sid替代随机item-id)
  - [1.3 SID在判别式模型中的三种核心使用方式](#13-sid在判别式模型中的三种核心使用方式)
  - [1.4 SID Token 参数化策略：从码本索引到模型特征](#14-sid-token-参数化策略从码本索引到模型特征)
  - [1.5 两阶段训练流水线](#15-两阶段训练流水线)
- [2. 各论文方法详解](#2-各论文方法详解)
  - [2.1 YouTube Semantic IDs（Google, RecSys 2024）](#21-youtube-semantic-idsgoogle-recsys-2024)
  - [2.2 Meta SID Prefix N-gram（Meta, 2025）](#22-meta-sid-prefix-n-grammeta-2025)
  - [2.3 SIDE：无参数SID转Embedding（Meta, AdKDD 2025）](#23-side无参数sid转embeddingmeta-adkdd-2025)
  - [2.4 QARM：快手多模态对齐+量化（快手, 2024）](#24-qarm快手多模态对齐量化快手-2024)
  - [2.5 QARM V2：混合FSQ量化+推理序列建模（快手, 2026）](#25-qarm-v2混合fsq量化推理序列建模快手-2026)
  - [2.6 SMILE/COINS：冷启动物品表示增强（快手电商, 2025）](#26-smilecoins冷启动物品表示增强快手电商-2025)
  - [2.7 DAS：双对齐语义ID（快手广告, 2025）](#27-das双对齐语义id快手广告-2025)
  - [2.8 TRM：语义Token替代Item ID的大规模排序（字节跳动, 2026）](#28-trm语义token替代item-id的大规模排序字节跳动-2026)
  - [2.9 DIG：判别即生成——从Tokenizer视角统一排序与检索（美团, 2026）](#29-dig判别即生成从tokenizer视角统一排序与检索美团-2026)
  - [2.10 VQ-Rec：可迁移的量化序列推荐（人大, 2022）](#210-vq-rec可迁移的量化序列推荐人大-2022)
- [3. 方法对比总结](#3-方法对比总结)
  - [3.1 各方法核心对比表](#31-各方法核心对比表)
  - [3.2 SID Token 参数化策略对比](#32-sid-token-参数化策略对比)
  - [3.3 SID Embedding 组合方式对比](#33-sid-embedding-组合方式对比)
- [4. 实践指南：如何将SID集成到你现有的推荐模型](#4-实践指南如何将sid集成到你现有的推荐模型)
  - [4.1 最快上手路径](#41-最快上手路径)
  - [4.2 进阶优化路径](#42-进阶优化路径)
  - [4.3 关键超参数与调参建议](#43-关键超参数与调参建议)
  - [4.4 常见陷阱与解决方案](#44-常见陷阱与解决方案)
- [5. 论文链接与参考文献](#5-论文链接与参考文献)

---

## 1. 概述

### 1.1 LLM4DLRMs 范式的核心思想

LLM4DLRMs（LLM-Empowered Deep Learning Recommendation Models）的核心思想是：**将语义ID（Semantic ID）作为离散特征输入到传统判别式推荐模型中，替换或补充原来的随机Item ID embedding，从而在不改变模型主体架构的前提下提升推荐效果**。

```
传统 DLRM/DNN 排序模型:
  随机 Item ID → 巨大 Embedding Table (10亿行) → 特征拼接 → MLP/Attention → CTR预估

LLM4DLRMs 方式:
  语义 ID (c₁, c₂, ..., c_L) → 小 Embedding Table → 特征拼接 → MLP/Attention → CTR预估
                                    ↑
                        码本大小固定 (e.g., 4096 × L层)
                        相似物品共享 embedding 参数
```

**与 LLM4GRs（生成式推荐）的本质区别**：

| 维度 | LLM4DLRMs（本报告聚焦） | LLM4GRs（不在本报告范围） |
|------|------------------------|-------------------------|
| SID 的角色 | **输入特征**（像普通 categorical feature 一样使用） | **生成目标**（像 NLP 中的 token 一样生成） |
| 模型范式 | 判别式（全库打分取 Top-K） | 生成式（自回归解码） |
| 架构改动 | **极小**（仅替换/增加 ID 特征字段） | 极大（需重构为生成模型） |
| 工业部署 | 渐进式迁移，风险低 | 颠覆式变革，风险高 |
| 代表工作 | YouTube, QARM, SIDE, TRM | TIGER, OneRec, PLUM |

### 1.2 为什么用SID替代随机Item ID？

**核心洞察**（YouTube 论文提出）：

> 内容 embedding 泛化好但记忆差，随机 ID 记忆好但泛化差。语义 ID 是两者之间的最佳折衷。

具体来说：

| 问题 | 随机 Item ID | Dense 内容 Embedding | 语义 ID |
|------|-------------|---------------------|---------|
| **冷启动** | ❌ 无交互历史 = 无意义 embedding | ✅ 基于内容有意义的表示 | ✅ 基于内容有意义的离散码 |
| **记忆能力** | ✅ 每个物品独立 embedding，记忆强 | ❌ 共享连续空间，记忆弱 | ✅ 通过深层码本组合实现记忆 |
| **泛化能力** | ❌ ID 之间无关联，无法泛化 | ✅ 相似物品距离近 | ✅ 相似物品共享 token，参数级泛化 |
| **存储效率** | ❌ 10亿行 × d 维 = 巨大 | ❌ 每物品 768~4096 维浮点 | ✅ L × K 码本，小 embedding table |
| **可训练性** | ✅ 端到端梯度更新 | ❌ 通常冻结 | ✅ embedding 可端到端学习 |

### 1.3 SID在判别式模型中的三种核心使用方式

调研发现，各家论文中 SID 在判别式模型中的使用方式可归纳为以下三种：

#### 使用方式一：作为物品侧特征（Item-side Feature）

最常见的用法。将候选物品的 SID 作为 categorical feature 输入模型，替代或补充原来的 Item ID。

```
候选物品的 SID = (c₁, c₂, c₃)
       ↓
  Embedding Lookup: emb₁[c₁], emb₂[c₂], emb₃[c₃]
       ↓
  组合: concat/sum/attention → item_sid_emb ∈ R^{d'}
       ↓
  与其他特征拼接 → MLP → P(click)
```

**代表论文**：YouTube Semantic IDs, QARM, TRM, SMILE

#### 使用方式二：作为用户序列特征（User Sequence Feature）

将用户历史交互过的物品用 SID 表示，形成用户行为序列特征。

```
用户历史: [SID(item₁), SID(item₂), ..., SID(item_n)]
       ↓
  每个 SID → Embedding Lookup → 序列 embedding
       ↓
  Attention/Pooling/DIN → user_interest_emb ∈ R^{d''}
       ↓
  与候选物品特征交叉 → MLP → P(click)
```

**代表论文**：YouTube Semantic IDs, QARM, SIDE, QARM V2

**关键优势**：SID 是紧凑的整数序列（如 3~25 个整数），远比 dense embedding（768 维浮点向量）更适合存储和传输用户历史序列。在实时 serving 中，这大幅降低了网络传输和内存开销。

#### 使用方式三：作为交叉特征（Cross Feature）

利用 SID 的层次结构，在用户历史和候选物品之间构造匹配特征。

```
候选物品 SID = (c₁, c₂, c₃)
       ↓
  用 c₁ 在用户历史中匹配：同 c₁ 的历史物品子序列 → "一类码匹配"特征
  用 (c₁, c₂) 匹配：同 (c₁, c₂) 的历史物品子序列 → "二类码匹配"特征
       ↓
  匹配子序列 → Attention Pooling → cross_feature_emb
       ↓
  与其他特征拼接 → MLP → P(click)
```

**代表论文**：QARM（一类码/二类码匹配）, TRM（协同感知匹配）

### 1.4 SID Token 参数化策略：从码本索引到模型特征

SID 原始形式是一组离散码索引 `(c₁, c₂, ..., c_L)`，如何将这些索引转化为模型可用的特征，是各家论文的核心差异点。调研发现有以下几种参数化策略：

```
策略一：逐层独立 Embedding
  c₁ → Table₁[c₁] → emb₁
  c₂ → Table₂[c₂] → emb₂
  c₃ → Table₃[c₃] → emb₃
  → concat/sum → final_emb

策略二：Prefix N-gram
  (c₁) → hash → Table[hash(c₁)] → emb_prefix1
  (c₁,c₂) → hash → Table[hash(c₁,c₂)] → emb_prefix2
  (c₁,c₂,c₃) → hash → Table[hash(c₁,c₂,c₃)] → emb_prefix3
  → concat/sum → final_emb

策略三：SentencePiece 自适应子词
  SID序列 → SPM分词 → 可变长度子词 → 统一Table查找 → sum → final_emb

策略四：无参数转换（Parameter-free）
  SID码 → 数学分解（DPCA基向量乘积求和） → final_emb（无需查表）
```

各策略的详细对比见 [§3.2](#32-sid-token-参数化策略对比)。

### 1.5 两阶段训练流水线

LLM4DLRMs 范式普遍采用两阶段训练：

```
┌──────────────────────────────────────────────────────────┐
│  Stage 1: 离线生成语义ID（一次性）                         │
│                                                          │
│  物品内容 → 多模态Encoder → 对齐(可选) → 量化器 → SID     │
│                                                          │
│  产出: 每个物品的 SID = (c₁, c₂, ..., c_L)               │
│  码本和量化器训练完成后冻结                                 │
└──────────────────────────────────────────────────────────┘
                          ↓
┌──────────────────────────────────────────────────────────┐
│  Stage 2: 训练判别式推荐模型                               │
│                                                          │
│  输入特征:                                                │
│    ├── 用户特征 (age, gender, history, ...)               │
│    ├── 上下文特征 (time, device, ...)                     │
│    └── 物品特征:                                          │
│        ├── SID → Embedding Lookup (可训练) → 拼接         │
│        └── 其他物品特征 (category, price, ...)            │
│                                                          │
│  模型: DNN/MLP/Attention → P(click) / P(watch)           │
│  损失: BCE / Cross-Entropy                               │
│                                                          │
│  关键: SID embedding 参数随推荐模型端到端梯度更新           │
│        SID 码本本身保持冻结                                │
└──────────────────────────────────────────────────────────┘
```

**核心原则**：
- **码本冻结**：量化器产出的码本（聚类中心）在 Stage 1 训练完成后不再更新。这意味着每个物品的 SID 是固定的静态属性。
- **Embedding 可学习**：SID 对应的 embedding table 参数在 Stage 2 中随推荐模型一起端到端训练。这是 SID 比 dense 内容 embedding 更有效的关键原因——embedding 可以被推荐目标适配。

---

## 2. 各论文方法详解

### 2.1 YouTube Semantic IDs（Google, RecSys 2024）

- **论文**：Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations
- **链接**：[arXiv:2306.08121](https://arxiv.org/abs/2306.08121)
- **发表**：RecSys 2024
- **机构**：Google / YouTube

#### 2.1.1 核心问题

YouTube 的 Watch Next 排序模型需要为数十亿视频打分。传统方法中每个视频有一个随机哈希 ID，对应一个巨大的 embedding table。这导致：
- 新视频（冷启动）没有有意义的 embedding
- 相似视频之间无法共享参数
- 用户历史中如果用 dense 内容 embedding 表示视频，存储和传输开销巨大

#### 2.1.2 SID 生成（Stage 1）

```
视频 → Transformer 视频编码器（冻结）→ 稠密内容 embedding ∈ R^d
                                             ↓
                                    RQ-VAE 量化（离线训练）
                                             ↓
                                   SID = (c₁, c₂, ..., c_K)
                                   K层残差量化，粗→细
```

RQ-VAE 训练完成后冻结，对所有视频批量生成 SID。

#### 2.1.3 SID 在排序模型中的使用（Stage 2）—— 核心创新

**SentencePiece Model (SPM) 自适应子词化**：这是本文最关键的创新。

SID 是一个长度为 K 的离散码序列。问题是如何将这个序列映射为排序模型中的 embedding？

**方案一：逐层独立 Embedding（Baseline）**
```
c₁ → Table₁ → emb₁
c₂ → Table₂ → emb₂
c₃ → Table₃ → emb₃
→ concat → final_emb
```
- 缺点：忽略了码之间的跨层交互

**方案二：N-gram 哈希**
```
1-gram: {c₁}, {c₂}, {c₃}  → 各自 hash → 查表
2-gram: {c₁,c₂}, {c₂,c₃}  → 各自 hash → 查表
3-gram: {c₁,c₂,c₃}         → hash → 查表
→ sum → final_emb
```
- 缺点：固定粒度的划分不够灵活

**方案三：SentencePiece 子词化（本文推荐方法）✅**

受 NLP 中 BPE/SentencePiece 的启发，将 SID 序列类比为一个"句子"，每个离散码类比为"字符"：

```python
# 1. 在已曝光物品的 SID 分布上训练 SPM 模型
# SPM 自动学习哪些码组合频繁出现 → 合并为更长的子词
# 哪些组合罕见 → 保留为短子词

# 2. 对每个视频的 SID 序列做 SPM 分词
sid_sequence = [42, 187, 5]  # 3层 RQ-VAE 码
spm_tokens = spm_model.encode(sid_sequence)
# 可能得到: [token_A, token_B]
# 其中 token_A 对应子序列 [42, 187]（频繁共现 → 合并）
#      token_B 对应 [5]（单独保留）

# 3. 每个 SPM token → 统一 embedding table 查找 → sum
final_emb = sum(embedding_table[t] for t in spm_tokens)
```

**为什么 SPM 优于 N-gram**：
- SPM **自适应学习**子词粒度：高频码组合有专属子词（记忆），低频组合共享短子词（泛化）
- N-gram 是**固定划分**，无法根据数据分布调整
- SPM 在更大的 embedding table 下持续获益，N-gram 在 table 较小时略有优势

#### 2.1.4 SID 在模型中的三处使用

```
YouTube Watch Next 排序模型:

输入特征:
  ├── 用户特征 (age, gender, ...)
  ├── 用户历史序列 ← 每个历史视频用 SID 表示（紧凑！）
  │     而非 dense embedding（太大！）
  ├── 候选视频 ← SID 替换 Video ID
  └── 当前观看视频 ← SID 替换 Video ID
```

**关键**：SID 在三个地方替换了原来的随机 Video ID——候选视频、用户历史中的每个视频、当前上下文视频。

#### 2.1.5 关键实验结论

| 方案 | 整体质量 | 冷启动/长尾 |
|------|---------|-----------|
| 随机哈希 ID (Baseline) | Baseline | 差 |
| Dense 内容 Embedding 直接替换 | **比 Baseline 差** | 较好 |
| SID + N-gram | 优于 Baseline | **显著更好** |
| SID + SentencePiece | **最优** | **最优** |

> **重要发现**：直接用 dense 内容 embedding 替换随机 ID **反而会降低排序质量**。原因是生产排序模型严重依赖 ID embedding 的记忆能力，dense embedding 丢失了这种能力。SID 通过离散化 + embedding table 兼顾了记忆和泛化。

#### 2.1.6 实践要点

- **渐进式迁移**：SID 可以作为额外特征添加到现有模型中，与随机 ID 并存，逐步迁移
- **存储效率**：SID 仅需 K 个整数（如 3~4 个），比 768 维浮点向量小 1000 倍，适合实时传输用户历史
- **周期性更新**：随着视频库更新，RQ-VAE 和 SPM 需要定期重新训练
- **码本大小参考**：C=1024, K=3层, embedding dim=64

#### 2.1.7 用户行为序列的表示与处理

**每个视频用 K 个 SID 码表示**（论文中 K=4），用户的历史观看序列变成：

```
传统方式:
  用户历史 = [video_id₁, video_id₂, ..., video_id_N]   ← N个随机整数
  每个 ID → 查数十亿行的 Embedding Table → 向量

SID 方式:
  用户历史 = [(c₁¹,c₂¹,c₃¹,c₄¹), (c₁²,c₂²,c₃²,c₄²), ..., (c₁ᴺ,c₂ᴺ,c₃ᴺ,c₄ᴺ)]
              ↑ 视频1的4个码          ↑ 视频2的4个码            ↑ 视频N的4个码
  每个视频的 K 个码 → SPM 分词 → 查数千行的小 Table → 向量
```

**SPM 分词是逐视频独立进行的**（不是跨视频的序列级分词）：

```
每个视频的 SID 独立经过 SPM 分词:

  视频1: SID = (42, 157, 892, 3)
       ↓ SPM 分词
    SPM tokens = [T_a, T_b]
      其中 T_a 对应子序列 (42, 157)   ← 频繁共现 → 合并为一个子词
           T_b 对应 (892, 3)         ← 合并为另一个子词
       ↓ 每个 SPM token → 统一 Embedding Table 查找 → 求和
    → 视频1 的 embedding 表示 vec₁ ∈ R^d

  视频2: SID = (88, 12, 445, 9)
       ↓ SPM 分词
    SPM tokens = [T_c, T_d]
       ↓ 查表求和
    → 视频2 的 embedding 表示 vec₂ ∈ R^d

  ...以此类推，每个视频得到一个固定维度的向量
```

**序列聚合：Pooled Multihead Attention (PMA)**

用户历史是一个变长序列 `[vec₁, vec₂, ..., vec_N]`，需要聚合为固定维度的用户兴趣向量。论文使用 **PMA（Pooled Multihead Attention）**，这是一种 **target-aware attention**（目标感知注意力）：

```
┌─────────────────────────────────────────────────────────┐
│  Pooled Multihead Attention (PMA)                        │
│                                                          │
│  Query (Q):  来自候选视频特征 + 上下文特征                │
│  Key (K):    用户历史向量 [vec₁, vec₂, ..., vec_N]      │
│  Value (V):  用户历史向量 [vec₁, vec₂, ..., vec_N]      │
│                                                          │
│  语义: "给定这个候选视频，用户历史中哪些部分最相关？"      │
│                                                          │
│  Attention(Q, K, V) = softmax(Q·K^T / √d) · V           │
│       ↓ Multi-head → concat → 线性投影                   │
│       ↓                                                  │
│  user_interest ∈ R^{d'}   ← 固定维度的用户兴趣向量       │
└─────────────────────────────────────────────────────────┘
```

这与 DIN（Deep Interest Network）的思想类似——候选物品作为 Query 来"查询"用户历史中最相关的部分。

**完整的序列处理流水线**：

```
┌─ 视频1: SID=(42,157,892,3) → SPM → [T_a,T_b] → embed+sum → vec₁ ─┐
│  视频2: SID=(88,12,445,9)  → SPM → [T_c,T_d] → embed+sum → vec₂  │
│  视频3: SID=(42,157,2001,7)→ SPM → [T_a,T_e] → embed+sum → vec₃  │
│  ...                                                               │
└─ 视频N: SID=(...) → SPM → [...] → embed+sum → vec_N ─────────────┘
                                                                    │
     用户历史矩阵: [vec₁, vec₂, ..., vec_N]  shape: [seq_len, d]   │
                                                                    │
     ┌──────────────────────────────────────────────────────┐       │
     │  PMA: Q=候选视频特征, K/V=用户历史 → user_interest   │       │
     └──────────────────────────────────────────────────────┘       │
                                                                    │
     user_interest ∈ R^{d'}  → 与其他特征拼接 → 送入 DNN           │
```

**SPM 的训练**：在所有已曝光物品的 SID 序列上训练 SPM 模型（类似 NLP 中在语料库上训练 BPE/SentencePiece）。SPM 学到的是"哪些码组合经常一起出现在同一个物品中"，从而自适应地确定子词粒度。

#### 2.1.8 整体模型架构

YouTube 论文的排序模型**不是纯 Transformer**，而是基于 YouTube Watch Next 生产排序系统（参考 RecSys 2019 "Recommending what video to watch next: a multitask ranking system"）的 **DLRM 风格多任务模型**：

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入特征层                                 │
│                                                                  │
│  ┌──────────────┐  ┌────────────────┐  ┌──────────────────┐     │
│  │ 候选视频特征   │  │ 用户历史特征     │  │ 上下文特征        │     │
│  │              │  │                │  │                  │     │
│  │ • SID(候选)  │  │ • SID序列      │  │ • 时间/设备      │     │
│  │   →SPM→emb   │  │   →SPM→emb    │  │ • 会话/位置      │     │
│  │ • 标题/类别   │  │   →PMA聚合     │  │                  │     │
│  │ • 其他属性   │  │ • 行为类型     │  │                  │     │
│  │              │  │   (观看/点赞)  │  │                  │     │
│  │              │  │ • 停留时长     │  │                  │     │
│  └──────┬───────┘  └──────┬─────────┘  └────────┬─────────┘     │
│         │                 │                      │               │
│         └────────┬────────┴──────────┬──────────┘               │
│                  ↓                   ↓                           │
│         Embedding Layer → 特征拼接                                │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│              MMoE (Multi-gate Mixture-of-Experts)                 │
│                                                                   │
│  ┌────────┐ ┌────────┐ ┌────────┐                                │
│  │Expert 1│ │Expert 2│ │Expert 3│  ← 共享专家网络                 │
│  └───┬────┘ └───┬────┘ └───┬────┘                                │
│      └──────────┼──────────┘                                      │
│                 ↓                                                 │
│     Gate_click  Gate_watch  Gate_like  ← 每任务独立门控           │
│          ↓          ↓          ↓                                  │
│     click_emb   watch_emb   like_emb                              │
└─────────┬──────────┬──────────┬──────────────────────────────────┘
          ↓          ↓          ↓
┌──────────────────────────────────────────────────────────────────┐
│              Shallow Tower (浅塔)                                  │
│  • 简单交互特征（点积、交叉特征）                                   │
│  • 建模/缓解选择偏差 (selection bias)                              │
└──────────────────────────┬───────────────────────────────────────┘
                           ↓
┌──────────────────────────────────────────────────────────────────┐
│              多任务输出头                                           │
│  P(click) × w₁ + P(watch_time) × w₂ + P(satisfaction) × w₃      │
│  → 最终排序分数                                                    │
└──────────────────────────────────────────────────────────────────┘
```

**关键架构组件说明**：

| 组件 | 作用 | 与 SID 的关系 |
|------|------|--------------|
| **Embedding Layer** | 将稀疏特征（包括 SID token）映射为稠密向量 | SID 的 SPM token 在此查表 |
| **PMA** | 聚合变长用户历史为固定向量 | SID embedding 构成历史的 K/V |
| **MMoE** | 多任务共享专家 + 任务特定门控 | SID 信息通过专家网络被不同任务利用 |
| **Shallow Tower** | 简单交互特征，缓解选择偏差 | SID 可参与交叉特征计算 |
| **多任务头** | 预测多个目标，加权组合 | SID 的泛化能力帮助各任务 |

**SID 的引入不改变模型主体架构**，仅替换了 Embedding Layer 中的 Item ID 查表方式——这正是 LLM4DLRMs 范式的核心特点。

#### 2.1.9 SID 替换前后对比

```
替换前 (Baseline):
  用户历史: [vid_38291, vid_104857, vid_9283, ...]
              ↓ 查巨大 Table（数十亿行）
            [emb₁, emb₂, emb₃, ...]
              ↓ PMA 聚合
            user_interest
  问题: 新视频无有意义 embedding; 相似视频无参数共享

替换后 (SID + SPM):
  用户历史: [(42,157,892,3), (88,12,445,9), (42,157,2001,7), ...]
              ↓ 每个视频 SPM 分词 + 查小 Table（数千行）
            [emb₁, emb₂, emb₃, ...]
              ↓ PMA 聚合（同上）
            user_interest
  优势: 新视频立即有有意义 SID; 相似视频共享 SPM token → 参数级泛化
```

| 维度 | 替换前（随机 Video ID） | 替换后（SID + SPM） |
|------|----------------------|-------------------|
| **每个视频表示** | 1 个随机整数 → 查数十亿行 Table | K=4 个层次码 → SPM → 查数千行 Table |
| **历史序列** | [vid₁, vid₂, ..., vid_N] → N 个 embedding | 每个视频 SPM→embed→sum → N 个 embedding |
| **Table 大小** | 数十亿行（每视频一行） | 数千行（SPM 词汇量） |
| **冷启动** | ❌ 新视频无有意义 embedding | ✅ 新视频从内容获得有意义 SID |
| **泛化能力** | ❌ 随机 ID 无语义结构 | ✅ 相似视频共享 SPM token → 参数级泛化 |
| **存储效率** | ❌ 每视频存储 embedding | ✅ 每视频仅 K 个整数，压缩 100~200 倍 |
| **序列处理** | PMA 聚合（不变） | PMA 聚合（不变） |
| **模型架构** | DLRM + MMoE（不变） | DLRM + MMoE（不变） |

---

### 2.2 Meta SID Prefix N-gram（Meta, 2025）

- **论文**：Enhancing Embedding Representation Stability in Recommendation Systems with Semantic ID
- **链接**：[arXiv:2504.02137](https://arxiv.org/abs/2504.02137)
- **发表**：2025年4月
- **机构**：Meta（广告排序系统）

#### 2.2.1 核心问题

Meta 广告排序系统中，传统的随机 Item ID embedding 存在几个稳定性问题：
- **Embedding 漂移**：随着训练数据增加，同一 ID 的 embedding 含义会不稳定地漂移
- **长尾 ID 过拟合**：交互稀疏的物品容易过拟合
- **分布偏移敏感**：当训练/服务数据分布不一致时表现脆弱

#### 2.2.2 SID Prefix N-gram 方法

```
物品的 SID = (c₁, c₂, c₃, ..., c_K)   ← 来自 RQ-VAE，粗→细层次

构造 Prefix N-gram 特征:
  Prefix-1: hash(c₁)                    → 最粗粒度，最大泛化
  Prefix-2: hash(c₁, c₂)                → 中等粒度
  Prefix-3: hash(c₁, c₂, c₃)            → 较细粒度
  ...
  Prefix-K: hash(c₁, c₂, ..., c_K)      → 最细粒度，最强记忆

每个 Prefix → 独立的 Embedding Table → 查表得到 embedding
所有 Prefix embedding → 拼接/求和 → 最终 SID embedding
```

**与 YouTube SPM 的区别**：
- YouTube SPM 是**自适应**子词化（数据驱动学习分词）
- Meta Prefix N-gram 是**固定**层次前缀（利用 RQ-VAE 的粗→细结构）
- Prefix N-gram 更简单、更可预测，不需要额外训练 SPM 模型

#### 2.2.3 SID 在 Meta 广告排序模型中的使用

```
Meta Ads Ranking DLRM:

输入稀疏特征 (Sparse Features):
  ├── 传统: Ad ID → Embedding Table (巨大, 不稳定)
  └── 新增: SID Prefix N-gram → 多个小 Embedding Table (稳定, 语义共享)

输入稠密特征 (Dense Features):
  ├── 用户特征
  ├── 广告属性特征
  └── 上下文特征

  所有特征 → Feature Interaction Layers (DCN-V2等) → MLP → P(click/conversion)
```

**关键点**：SID Prefix N-gram 特征作为 DLRM 中的**顶层稀疏特征（top sparse features）**使用，与传统的 categorical 特征完全相同的处理方式。

#### 2.2.4 关键实验结论

- **Embedding 稳定性**：SID 创建的 ID 空间更稳定，embedding 随训练持续学习保持一致含义
- **长尾物品提升**：大部分提升来自物品分布的长尾部分
- **正则化效果**：语义碰撞（semantic collisions）起到隐式正则化作用，减少过拟合
- **分布偏移鲁棒性**：比随机哈希对分布偏移更不敏感
- **已在 Meta 广告排序系统生产部署**，取得显著的在线指标提升

#### 2.2.5 实践要点

- **Table 大小控制**：短前缀（Prefix-1, Prefix-2）产生的词汇量小但碰撞多（泛化好），长前缀词汇量大（记忆强）。可以根据可用内存灵活配置各层 table 大小
- **与随机 ID 共存**：SID 特征可以添加到模型中与随机 ID 并存，通过 A/B 测试逐步验证效果
- **不需要端到端训练 tokenizer**：RQ-VAE 离线训练完成后，SID 作为静态特征使用

---

### 2.3 SIDE：无参数SID转Embedding（Meta, AdKDD 2025）

- **论文**：SIDE: Semantic ID Embedding for effective learning from sequences
- **链接**：[arXiv:2506.16698](https://arxiv.org/abs/2506.16698)
- **发表**：AdKDD 2025
- **机构**：Meta Platforms（广告推荐）

#### 2.3.1 核心问题

在广告序列建模中，用户历史可能包含数千到数万个广告交互事件。如果用传统的 SID → Embedding Table 方式，embedding table 会非常大（尤其使用 n-gram 时指数增长）。SIDE 的核心创新是**完全消除 embedding table**。

#### 2.3.2 DPCA（Discrete-PCA）量化方法

DPCA 是一种结构化量化方法，将残差量化与 PCA 联系：

```
输入 embedding x ∈ R^d
     ↓
投影到结构化基方向（类似主成分）
     ↓
每个投影方向用三值码本 {-1, 0, 1} 量化
     ↓
计算残差 → 在下一层重复
     ↓
SID = 多层三值码序列 (e.g., 25个三值码)
```

**为什么用三值 {-1, 0, 1}**：
- 受 BitNet（1.58-bit LLM）启发，三值表示足够通用
- 大幅减少码本存储和计算复杂度
- 使得无参数反量化成为可能

#### 2.3.3 无参数 SID → Embedding 转换（核心创新）

传统方法：SID → 查 Embedding Table → embedding（需要巨大 table）

SIDE 方法：SID → 数学运算 → embedding（无需 table）

```python
# DPCA 的码本是结构化的（基于 PCA 基向量），不是任意学习向量
# 因此已知基向量矩阵 B ∈ R^{L × d}（L层 × d维）

def side_convert(sid_codes, basis_matrix):
    """
    sid_codes: shape [L], 每个值 ∈ {-1, 0, 1}
    basis_matrix: shape [L, d], PCA基向量矩阵（已知常量）
    """
    # 将三值码映射回数值
    ternary_values = sid_codes.astype(float)  # {-1, 0, 1}

    # 矩阵乘法: 用基向量线性组合重构 embedding
    reconstructed_emb = ternary_values @ basis_matrix  # shape: [d]

    return reconstructed_emb

# 对比传统方法:
# 传统: embedding_table[sid_code]  ← 需要存储 K^L × d 参数
# SIDE: sid_codes @ basis_matrix   ← 只需存储 L × d 参数（基向量）
```

**内存对比**：

| 方法 | 存储复杂度 | 实际大小（d=400, L=25） |
|------|-----------|----------------------|
| 传统 Embedding Table | O(K^L × d) | 天文数字 |
| N-gram Table | O(exp(C·b) × d) | 巨大 |
| **SIDE (Parameter-free)** | **O(L × d)** | **25 × 400 = 10000 参数** |

#### 2.3.4 VQ-Fusion 多任务融合

```
多种内容信号:
  text_emb ∈ R^{d1}
  img_emb ∈ R^{d2}
  video_emb ∈ R^{d3}
  category_pred ∈ R^{d4}   ← 广告类别预测等
        ↓ concat
  Encoder Network → 共享隐表示
        ↓
  DPCA 结构化量化
        ↓
  统一 SID（融合多源信息）
```

#### 2.3.5 SID 在广告序列模型中的使用

```
用户广告历史序列: [ad₁, ad₂, ..., ad_n]  (n可达10000+)
                        ↓
每个广告 → SID → SIDE 无参数转换 → d维 embedding
                        ↓
序列编码器 (DIN/Transformer) → 用户兴趣表示
                        ↓
与候选广告特征交叉 → DCN-V2 → P(click/conversion)
```

**关键优势**：
- 每个广告仅需存储 25 个三值整数（vs. 400 维浮点向量）
- 不需要 embedding table 查找，纯矩阵运算，推理高效
- **无碰撞**：每个唯一 SID 映射到唯一的 embedding 空间位置

#### 2.3.6 关键实验结论

| 指标 | 提升 |
|------|------|
| 归一化熵(NE)增益 | **2.4倍**提升（vs. 传统 SID） |
| 数据占用 | **3倍**减少 |
| DPCA 特征 RoI | **5.33~7.40倍**提升 |
| 内存压缩 | **100~200倍** |

#### 2.3.7 实践要点

- **适用场景**：用户历史序列很长（数千到数万事件）、广告库极大（数十亿）的场景
- **实现简单**：无参数转换就是一个矩阵乘法，可以用自定义 CUDA kernel 加速
- **SID 生成为离线工序**：VQ-Fusion + DPCA 离线训练，SID 批量生成后作为静态特征
- **下游模型不需要改动 embedding 基础设施**：用计算替代查表

---

### 2.4 QARM：快手多模态对齐+量化（快手, 2024）

- **论文**：QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou
- **链接**：[arXiv:2411.11739](https://arxiv.org/abs/2411.11739)
- **发表**：2024年11月
- **机构**：快手

#### 2.4.1 解决的两个核心问题

1. **Representation Unmatching**（表征不匹配）：预训练多模态模型（如 CLIP）由通用 NLP/CV 任务监督，推荐模型由用户交互监督，两者表示空间不一致
2. **Representation Unlearning**（表征不可学习）：多模态表征通常冻结缓存输入，无法被推荐梯度更新

#### 2.4.2 Stage 1：多模态对齐 + 量化

```
Step 1: 多模态对齐（Item Alignment）
  ├── 构造对齐数据：
  │   ├── User2Item：用户点击的 trigger→target 物品对
  │   │   (trigger = 用户最近50次点击中与target最相似的物品)
  │   └── Item2Item：已有检索模型(Swing)的稳定相似物品对
  │
  ├── MLLM 微调：
  │   M_trigger = MLLM(text, audio, image of trigger)
  │   M_target  = MLLM(text, audio, image of target)
  │   L_align = Batch-Contrastive(M_trigger, M_target, B)
  │
  └── 产出：对齐后的多模态 embedding

Step 2: 残差 K-Means 量化
  对齐后的 embedding → Res-KMeans(L层, K个聚类) → SID = (c₁, c₂, ..., c_L)

Step 3 (可选): VQ 码
  对齐后的 embedding → TopK 最近邻 → VQ codes（平坦相似度码）
```

#### 2.4.3 Stage 2：SID 在下游模型中的使用

QARM 生成两种 SID（VQ 码和 RQ 码），在下游模型中有**三种使用方式**：

```
┌─────────────────────────────────────────────────────────────┐
│  QARM 下游多任务排序模型                                      │
│                                                              │
│  使用方式一：物品侧特征                                        │
│  ├── 候选物品的 VQ codes → Embedding Table A → emb_vq       │
│  ├── 候选物品的 RQ codes → Embedding Table B → emb_rq       │
│  └── emb_vq + emb_rq + 其他特征 → MLP → P(click/cvr)       │
│                                                              │
│  使用方式二：用户序列特征                                      │
│  ├── 用户最近正向交互物品序列的 SID                           │
│  │   [SID(item₁), SID(item₂), ..., SID(item_n)]            │
│  └── → 序列 Embedding → Attention Pooling → user_interest   │
│                                                              │
│  使用方式三：交叉特征（QARM 独特设计）                         │
│  ├── 候选物品 RQ code = (c₁, c₂, c₃)                      │
│  ├── 在用户历史中找"一类码匹配"物品：共享 c₁ 的历史物品       │
│  ├── 在用户历史中找"二类码匹配"物品：共享 (c₁,c₂) 的历史物品  │
│  └── 匹配子序列 → Attention → cross_feature_emb             │
│                                                              │
│  所有 embedding → Feature Interaction → 多任务输出            │
│  SID Embedding 参数随推荐模型端到端梯度更新                    │
└─────────────────────────────────────────────────────────────┘
```

#### 2.4.4 Embedding Table 结构

| Table | 大小 | 说明 |
|-------|------|------|
| VQ Code Table | 等于物品总数（所有物品的对齐表示作为码本） | TopK 最近邻查找 |
| RQ Code Table (每层) | K（聚类数，如 4096） | 一层一个 table |
| **总 Embedding 数量** | **L × K + N**（远小于 N × d） | N=物品数, L=层数, K=码本大小 |

#### 2.4.5 关键实验结论

| 场景 | 指标 | 提升 |
|------|------|------|
| **广告** | 收入 (Revenue) | **+9.704%** |
| **电商** | GMV | **+2.296%** |

**消融实验关键发现**：
- 去掉对齐（直接用原始 MLLM 表征）→ 效果下降，对齐必不可少
- 用 dense 特征替代 SID → 效果更差，量化步骤是关键
- VQ 码和 RQ 码各自独立贡献，联合使用效果最好
- SID embedding 冻结（不可学习）→ 效果大幅下降，端到端学习是核心

#### 2.4.6 实践要点

- **增量添加**：SID 特征作为额外的 categorical feature 字段添加到现有模型，不需要移除原有特征
- **VQ + RQ 联合使用**：VQ 码捕获平坦相似关系，RQ 码捕获层次语义结构，两者互补
- **交叉特征是高收益设计**：利用 SID 层次结构在用户历史和候选物品之间构造匹配特征
- **冷启动自动解决**：新物品上传后，通过冻结 MLLM + 冻结码本即可立即获得有意义的 SID
- **部署于快手多个推荐场景**，服务 4 亿日活用户

---

### 2.5 QARM V2：混合FSQ量化+推理序列建模（快手, 2026）

- **论文**：QARM V2: Quantitative Alignment Multi-Modal Recommendation for Reasoning User Sequence Modeling
- **链接**：[arXiv:2602.09458](https://arxiv.org/abs/2602.09458)
- **发表**：2026年2月
- **机构**：快手

#### 2.5.1 V1 的问题：码本冲突

QARM V1 使用 Res-KMeans 量化，在长尾分布下容易出现**码本冲突**（code collision）：多个不同物品被量化到相同的 SID，导致下游模型无法区分。

#### 2.5.2 Res-KmeansFSQ 混合量化

```
粗粒度层: Res-KMeans（保留类别语义）
  → 第1层 K-Means 聚类 → c₁ (粗粒度类别码)

细粒度层: FSQ (Finite Scalar Quantization)（捕获属性特征，避免冲突）
  → 对残差做有限标量量化
  → tanh压缩 → round到有限级数
  → (f₁, f₂, ..., f_M) (细粒度属性码)

最终 SID = (c₁, f₁, f₂, ..., f_M)
```

**FSQ 核心思想**：不使用码本查找，而是将连续向量的每个维度量化到有限个离散值：

```python
def fsq_quantize(x, levels=[8, 5, 5, 5, 3, 3, 3]):
    x_compressed = tanh(x)                    # 压缩到(-1, 1)
    quantized = []
    for i, l in enumerate(levels):
        q_i = round(x_compressed[:, i] * (l-1) / 2)
        q_i = clamp(q_i, -l//2, l//2)
        quantized.append(q_i)
    return stack(quantized)
    # 总线数 = 8×5×5×5×3×3×3 = 27000 种组合
    # 远大于传统 K-Means 的 K=4096，冲突率大幅降低
```

#### 2.5.3 SID 在下游模型中的使用

与 QARM V1 类似，SID 作为可学习离散特征输入排序模型。V2 增加了两个模块：

- **GSU（General Search Unit）**：基于推理的 item 对齐机制，生成业务对齐的 LLM embedding，用于语义感知的历史子序列检索
- **ESU（Exact Search Unit）**：Res-KmeansFSQ 生成的 SID 用于精确用户序列匹配

```
候选物品 → GSU (语义检索) → 从用户历史中找到语义相关的子序列
           ESU (精确匹配) → 用 SID 做精确码匹配的子序列
                ↓
        两路子序列 → 融合 → 用户兴趣表示 → 排序打分
```

#### 2.5.4 部署效果

已在快手的**购物、广告、直播**场景部署，服务 4 亿日活用户。

---

### 2.6 SMILE/COINS：冷启动物品表示增强（快手电商, 2025）

- **论文**：SMILE: SeMantic Ids Enhanced CoLd Item Representation for Click-through Rate Prediction in E-commerce Search
  - 同一论文也以 COINS 为名发表
- **链接**：[arXiv:2510.12604](https://arxiv.org/abs/2510.12604)
- **发表**：2025年10月
- **机构**：快手（电商搜索）

#### 2.6.1 核心问题

电商搜索中冷启动物品（新上架商品）缺乏交互历史，传统 Item ID embedding 无意义。SMILE 用 SID 为冷启动物品注入内容语义和协同信号。

#### 2.6.2 RQ-OPQ 双层编码

```
物品内容特征 + 协同信号
         ↓
    ┌────┴────┐
    ↓         ↓
  RQ 编码    OPQ 编码
    ↓         ↓
粗粒度表示   细粒度表示
(共享协同    (物品独特的
 信号)       判别信息)
    ↓         ↓
    └────┬────┘
         ↓
    融合注入 CTR 模型
```

- **RQ 编码**：残差量化 → 层次化的粗粒度表示，不同物品共享部分码字 → 传递协同信号
- **OPQ 编码**：优化乘积量化 → 并行细粒度表示，捕获每个物品的独特判别信息

#### 2.6.3 在 CTR 模型中的使用

```
CTR 预测模型输入:

1. 自适应迁移与对齐模块（粗粒度）:
   RQ codes → Embedding Lookup → 与 Item ID embedding 对齐
   → 即使新物品无 Item ID embedding，也有 RQ embedding 作为替代

2. 物品判别信息学习模块（细粒度）:
   OPQ codes → Embedding Lookup → 注入物品独特信息
   → 补充 RQ 可能丢失的细粒度判别特征

3. 融合:
   RQ_emb + OPQ_emb + 其他特征 → MLP → P(click)
```

#### 2.6.4 关键实验结论（快手商城搜索 A/B 测试）

| 指标 | 整体提升 | 冷启动场景提升 |
|------|---------|-------------|
| 物品 CTR | +1.66% | — |
| 买家数 | +1.57% | **+3.512%** |
| 订单量 | +2.17% | **+9.639%** |

#### 2.6.5 实践要点

- **RQ + OPQ 互补设计**：RQ 负责"共享"（泛化），OPQ 负责"独特"（记忆），两者互补
- **冷启动效果突出**：新物品即使无交互历史，RQ 码与其他相似物品共享 embedding，OPQ 码注入独特信息
- **特别适合电商搜索**：新品频繁上架，冷启动是核心痛点

---

### 2.7 DAS：双对齐语义ID（快手广告, 2025）

- **论文**：DAS: Dual-Aligned Semantic IDs Empowered Industrial Recommender System
- **链接**：[arXiv:2508.10584](https://arxiv.org/abs/2508.10584)
- **发表**：2025年
- **机构**：快手（广告系统）
- **参考讲解**：https://zhuanlan.zhihu.com/p/1943035654511494581

#### 2.7.1 核心创新：一阶段联合训练

不同于 QARM 的两阶段（先对齐 → 再量化），DAS 将**量化、CF建模、对齐**三者在**一个训练阶段**内端到端联合优化。

```
DAS 一阶段联合训练:

输入: 用户和广告的多模态 PLM embedding

┌─────────────────────────────────────────────────────┐
│  联合训练三个模块:                                    │
│                                                      │
│  1. 语义模型 (UISM): PLM → RQ-VAE 量化 → 用户/广告 SID│
│  2. CF去偏模块 (ICDM): 分离无偏兴趣和流行度偏差        │
│  3. 多视角对比对齐 (MDAM): 六路对比损失注入协同信号     │
│                                                      │
│  总损失: L = L_Sem + α·L_CF + β·L_Align             │
│    L_Sem: RQ-VAE 重构+码本+承诺损失                   │
│    L_CF:  有偏+无偏CF损失+去偏正交约束                │
│    L_Align: 六路对比损失之和                          │
│    α=1, β=0.5                                        │
└─────────────────────────────────────────────────────┘

产出: 双对齐的 SID（同时包含内容语义和协同过滤信号）
```

#### 2.7.2 SID 在下游模型中的使用

```
下游广告 CTR 模型:

输入:
  ├── 用户 SID → Embedding Table → user_sid_emb
  ├── 广告 SID → Embedding Table → ad_sid_emb
  ├── 传统 ID 特征 (user_id, ad_id, ...)
  ├── 用户画像特征
  └── 上下文特征
       ↓
  Feature Interaction (DCN/Attention) → MLP → P(click/cvr)

SID Embedding 参数端到端可学习
```

#### 2.7.3 CF 去偏的关键创新

原始 CF 信号包含流行度偏差和从众偏差。DAS 将 CF 表示分解为无偏分量和偏差分量：

```
用户侧:
  c_u = MLP(c_u^int ⊕ c_u^con)
    c_u^int = 无偏用户兴趣（用于对齐）
    c_u^con = 用户从众偏差（丢弃）

广告侧:
  c_i = MLP(c_i^pro ⊕ c_i^pop)
    c_i^pro = 无偏广告内容（用于对齐）
    c_i^pop = 广告流行度偏差（丢弃）

去偏约束: 正交损失（兴趣 ⊥ 从众, 内容 ⊥ 流行度）
```

**对齐时只使用无偏分量**，避免 SID 被偏差信号污染。

#### 2.7.4 六路对比对齐

```
① Dual U2I: z_u ↔ c_i^pro  (用户SID ↔ 广告去偏CF)
② Dual U2I: c_u^int ↔ z_i  (用户去偏CF ↔ 广告SID)
③ Dual U2U: z_u ↔ c_u^int  (batch内用户侧对齐)
④ Dual I2I: z_i ↔ c_i^pro  (batch内广告侧对齐)
⑤ Dual Co-occur U2U: 共现用户SID互对齐
⑥ Dual Co-occur I2I: 共现广告SID互对齐
```

#### 2.7.5 部署效果

已在快手**多个广告场景**部署，服务 4 亿日活用户。

---

### 2.8 TRM：语义Token替代Item ID的大规模排序（字节跳动, 2026）

- **论文**：Farewell to Item IDs: Unlocking the Scaling Potential of Large Ranking Models via Semantic Tokens
- **链接**：[arXiv:2601.22694](https://arxiv.org/abs/2601.22694)
- **发表**：2026年1月
- **机构**：字节跳动

#### 2.8.1 核心问题

大规模排序模型中，Item ID embedding table 占据了绝大部分的稀疏参数（7.52T）。这些参数：
- 无法跨物品泛化
- 冷启动物品无意义
- 随模型扩展时，稀疏参数的 scaling law 不如稠密参数

TRM 的目标是**完全用语义 Token 替代 Item ID**。

#### 2.8.2 协同感知 Token 化

```
Step 1: 多模态内容 → 内容 embedding
Step 2: 用户行为数据 → 协同过滤信号
Step 3: 对比学习对齐内容 embedding 与协同信号
Step 4: 对齐后的 embedding → 量化 → 语义 Token
```

**关键**：纯内容 SID 会丢失协同信号（"这个物品和用户行为的关系"），TRM 在量化前用对比学习将协同信号注入内容 embedding，使得最终的语义 Token 同时包含内容和协同信息。

#### 2.8.3 混合 Token 化：泛化 Token + 记忆 Token

```
Gen-Tokens（泛化Token）:
  来自 RQ 的粗→细层次码
  → 相似物品共享 → 泛化能力强

Mem-Tokens（记忆Token）:
  对高频 Gen-Token 组合做 BPE
  → 为常见物品创建专属 token → 记忆能力强

最终 Token 序列 = Gen-Tokens + Mem-Tokens
```

这个设计与 YouTube SPM 的思路类似但更进一步：**先用 RQ 生成语义 Token，再用 BPE 将高频 Token 组合合并为记忆 Token**。

#### 2.8.4 SID 在排序模型中的使用

```
TRM 排序模型:

输入:
  ├── 用户侧 Token 序列（用户历史物品的语义 Token）
  ├── 候选物品 Token（候选物品的语义 Token）
  ├── 交叉特征
  └── 上下文特征
       ↓
  RankMixer 架构 (Per-Token FFN + Multi-Head Token Mixing)
       ↓
  P(click/watch/...)
```

**存储节省**：稀疏参数从 7.52T 减少到 5.07T（**减少 32.6%**）。

#### 2.8.5 关键实验结论

| 指标 | 提升 |
|------|------|
| CTR AUC | +0.65% |
| Real Play AUC | +0.85% |
| QAUC | +0.54% |
| 稀疏存储 | -33% |

**在线 A/B 测试**（个性化搜索引擎）：
- 用户活跃天数 +0.26%
- 换查率 +0.75%

**Scaling Law 关键发现**：语义 Token 在参数规模和计算量增加时，展现出**优于 Item ID 的 scaling 特性**——这是首个在推荐领域验证语义 Token scaling law 的工作。

#### 2.8.6 实践要点

- **完全替代 Item ID**：TRM 不是添加 SID 作为额外特征，而是完全替代 Item ID
- **CF 感知是关键**：纯内容 Token 效果不如 CF 感知 Token，协同信号不能丢
- **RankMixer 架构**：消费语义 Token 的模型架构也很重要，Per-Token FFN 避免高频特征压制低频特征

---

### 2.9 DIG：判别即生成——从Tokenizer视角统一排序与检索（美团, 2026）

- **论文**：Discrimination Is Generation: Unifying Ranking and Retrieval from a Tokenizer Perspective
- **链接**：[arXiv:2605.14853](https://arxiv.org/abs/2605.14853)
- **发表**：2026年5月
- **机构**：美团

#### 2.9.1 核心洞察

DIG 提出了一个颠覆性观点：**排序和检索本质上是同一优化在不同粒度上的体现**。判别式排序模型本身就蕴含生成式检索能力，只需要一个 Tokenizer 来解锁它。

```
传统视角:
  排序 = argmax_{item ∈ corpus} P(click | user, item)   ← 在物品空间搜索
  检索 = argmax_{item ∈ corpus} P(click | user, item)   ← 同上，只是候选集不同

DIG 视角:
  排序 = argmax_{item} f(user, item)      ← 在物品空间 argmax
  检索 = argmax_{token} g(user, token)     ← 在 token 空间 argmax (beam search)
  两者统一: 排序模型 + tokenizer = 排序和检索一体化
```

#### 2.9.2 Tokenizer 嵌入排序模型

```
DIG 架构:

输入:
  ├── 物品固有静态特征 → Tokenizer → SID tokens
  │     (物品内容、类别等，不含用户特征)
  │
  ├── 用户-物品交叉特征 (u2i) → 指导码本边界
  │     (用户是否点击、历史偏好等)
  │
  └── 其他用户/上下文特征

训练:
  整个系统用判别式 BCE 损失端到端训练
  L = BCE(P(click | user, item), label)
  
  关键: BCE 损失的梯度同时更新:
    1. 排序模型的参数
    2. Tokenizer 的码本参数

推理:
  排序: 正常前向推理，对每个候选物品打分
  检索: 用 beam search 在 token 空间搜索 → 直接生成候选物品
```

#### 2.9.3 MLPu2t 蒸馏模块

推理时无法使用 u2i 交叉特征（因为还不知道目标物品），DIG 通过蒸馏解决：

```
训练时:
  u2i 特征 → 指导码本学习 → SID 码本边界反映推荐决策边界
  
推理时:
  MLPu2t(user_features) → 近似 token-level 的用户偏好
  → 用近似值替代 u2i 特征 → beam search 检索
```

#### 2.9.4 关键实验结论

| 数据集 | 排序 AUC 提升 | 检索 Recall 提升 |
|--------|-------------|----------------|
| Meituan-Large | +0.0013 | 优于 SOTA 生成式检索 |
| Meituan-Small | +0.0205 | 优于 SOTA 生成式检索 |
| Taobao | 有提升 | 优于 SOTA 生成式检索 |
| KuaiRec | 有提升 | 优于 SOTA 生成式检索 |

#### 2.9.5 实践要点

- **最极端的 LLM4DLRMs 集成**：Tokenizer 直接嵌入排序模型，用判别式 BCE 损失端到端训练
- **一套系统同时支持排序和检索**：不需要独立的检索模型
- **u2i 特征是关键**：用户-物品交叉特征隐式引导码本边界向推荐决策边界对齐

---

### 2.10 VQ-Rec：可迁移的量化序列推荐（人大, 2022）

- **论文**：Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders
- **链接**：[arXiv:2210.12316](https://arxiv.org/abs/2210.12316)
- **发表**：2022年10月（早期开创工作）
- **机构**：中国人民大学

#### 2.10.1 核心思想

VQ-Rec 是较早将量化物品表示用于序列推荐的工作，核心是 **"文本 → 码 → 表征"** 的三段式流水线：

```
物品文本 → Text Encoder → z ∈ R^d → PQ量化 → (c₁, c₂, ..., c_M)
                                                    ↓
                                            Code Embedding Table
                                                    ↓
                                             物品表示 → 序列推荐模型 (SASRec等)
```

#### 2.10.2 PQ 量化

使用乘积量化（Product Quantization）将物品 embedding 切分为 M 个子空间，各自独立量化：

```
z = [z^(1); z^(2); ...; z^(M)]  → M个子向量
      ↓        ↓              ↓
   c₁=NN(z¹) c₂=NN(z²) ... c_M=NN(z^M)

物品 Code = (c₁, c₂, ..., c_M)  ← 并行, 无依赖
```

#### 2.10.3 Code Embedding 在序列模型中的使用

```
Code Embedding Table:
  每个子空间 m 有一个 embedding table, 大小 K × d'
  
物品 i 的表示:
  emb_i = concat(Table₁[c₁], Table₂[c₂], ..., Table_M[c_M])
  
序列推荐:
  用户历史 [emb_{i₁}, emb_{i₂}, ..., emb_{i_n}] → SASRec → 预测下一个物品
```

#### 2.10.4 跨域迁移的关键创新

离散码层解耦了文本特征和用户行为之间的紧耦合，使得跨域迁移更容易：

```
源域: 训练 Code Embedding + 序列模型
         ↓
目标域: 新物品的 Code 从文本直接获得
        Code Embedding Table 可迁移
        只需微调序列模型
```

#### 2.10.5 实践要点

- **早期验证了量化表示在判别式模型中的有效性**
- **Code Embedding 可端到端学习**：预训练 + 微调阶段都更新
- **冷启动天然解决**：新物品从文本直接获得码，无需交互历史

---

## 3. 方法对比总结

### 3.1 各方法核心对比表

| 论文 | 机构 | SID 生成方法 | Token 参数化策略 | 在模型中的使用方式 | 端到端可训练? | 部署状态 |
|------|------|-------------|----------------|-------------------|-------------|---------|
| **YouTube** | Google | RQ-VAE | SentencePiece 子词 | 物品+序列+上下文 | ✅ Embedding 可学习 | 生产（YouTube排序） |
| **Meta Prefix** | Meta | RQ-VAE | Prefix N-gram | 物品侧稀疏特征 | ✅ Embedding 可学习 | 生产（Meta广告） |
| **SIDE** | Meta | VQ-Fusion+DPCA | 无参数（基向量重构） | 广告序列 | ❌ 无参数转换 | 生产（Meta广告） |
| **QARM** | 快手 | Res-KMeans | 逐层独立 Table | 物品+序列+交叉 | ✅ Embedding 可学习 | 生产（快手多场景） |
| **QARM V2** | 快手 | Res-KMeans+FSQ | 逐层独立 Table | 物品+序列+精确匹配 | ✅ Embedding 可学习 | 生产（快手多场景） |
| **SMILE** | 快手电商 | RQ-OPQ | 逐层独立 Table | 物品+判别信息注入 | ✅ Embedding 可学习 | 生产（快手电商） |
| **DAS** | 快手广告 | 兼容多种 | 逐层独立 Table | 用户SID+广告SID | ✅ Embedding 可学习 | 生产（快手广告） |
| **TRM** | 字节 | CF感知+RQ+BPE | SentencePiece+BPE混合 | 物品+序列（完全替代ID） | ✅ Embedding 可学习 | 生产（字节搜索） |
| **DIG** | 美团 | 端到端 BCE 驱动 | Tokenizer嵌入排序模型 | 排序+检索统一 | ✅ Tokenizer+模型联合训练 | 实验验证 |
| **VQ-Rec** | 人大 | PQ | 逐子空间 Table | 序列推荐 | ✅ 预训练+微调 | 学术 |

### 3.2 SID Token 参数化策略对比

| 策略 | 代表论文 | 原理 | 优势 | 劣势 |
|------|---------|------|------|------|
| **逐层独立 Table** | QARM, SMILE, DAS | 每层码一个 Table，查表后 concat/sum | 简单，实现方便 | 忽略跨层交互 |
| **Prefix N-gram** | Meta | 按 SID 前缀分层 hash → 独立 Table | 利用层次结构，简单可控 | 固定粒度，不够灵活 |
| **SentencePiece** | YouTube, TRM | 在 SID 分布上训练 SPM → 自适应子词 | 自适应粒度，数据驱动 | 需额外训练 SPM 模型 |
| **无参数转换** | SIDE (DPCA) | SID 码 × 基向量矩阵 → embedding | 无需 Table，内存极小 | 需要结构化码本（DPCA） |
| **Tokenizer嵌入** | DIG | Tokenizer 直接嵌入模型，BCE 端到端 | 最彻底的集成 | 训练复杂度高 |

### 3.3 SID Embedding 组合方式对比

| 组合方式 | 代表论文 | 具体做法 |
|---------|---------|---------|
| **Concat** | YouTube, VQ-Rec | 各子词/子空间 embedding 拼接 → 高维向量 |
| **Sum** | YouTube (SPM), Meta | 各子词 embedding 求和 → 同维度向量 |
| **Attention Pooling** | QARM (序列) | 历史 SID embedding 经 Attention 加权聚合 |
| **特征交叉** | QARM (交叉特征) | 用候选 SID 在历史中匹配子序列 → Attention |
| **Per-Token FFN** | TRM/RankMixer | 每种 Token 类型独立的 FFN → Token Mixing |

---

## 4. 实践指南：如何将SID集成到你现有的推荐模型

### 4.1 最快上手路径

以下是最小改动的上手方案，参考 YouTube 和 Meta Prefix N-gram 的做法：

```python
# ==============================
# Step 1: 离线生成 SID（一次性）
# ==============================

# 1a. 获取物品的内容 embedding（冻结的多模态模型）
content_embeddings = frozen_multimodal_model(items)  # shape: [N, d]

# 1b. 训练 RQ-VAE（或 Res-KMeans）
rq_vae = train_rq_vae(content_embeddings, num_levels=3, codebook_size=4096)

# 1c. 为所有物品生成 SID
all_sids = rq_vae.encode(content_embeddings)  # shape: [N, 3], 每行如 [42, 187, 5]

# 1d. (可选) 训练 SPM 模型
spm_model = train_sentencepiece(all_sids)  # 在 SID 分布上训练

# ==============================
# Step 2: 修改推荐模型（添加特征）
# ==============================

# 原模型中的 Item ID 特征:
# item_id_emb = item_id_table[item_id]  # shape: [B, d_id]

# 新增 SID 特征（添加到模型特征列表中）:
class SIDEmbedding(nn.Module):
    def __init__(self, num_levels, codebook_size, embed_dim):
        super().__init__()
        # 每层一个 embedding table
        self.tables = nn.ModuleList([
            nn.Embedding(codebook_size, embed_dim) 
            for _ in range(num_levels)
        ])
    
    def forward(self, sid_codes):
        """
        sid_codes: shape [B, L], 每行如 [42, 187, 5]
        """
        embs = []
        for level in range(sid_codes.shape[1]):
            emb = self.tables[level](sid_codes[:, level])  # shape: [B, embed_dim]
            embs.append(emb)
        
        # 方式一: concat
        return torch.cat(embs, dim=-1)  # shape: [B, L * embed_dim]
        
        # 方式二: sum
        # return torch.stack(embs, dim=0).sum(dim=0)  # shape: [B, embed_dim]

# 使用:
sid_emb_layer = SIDEmbedding(num_levels=3, codebook_size=4096, embed_dim=64)

# 在模型 forward 中:
def forward(self, features):
    # ... 原有特征处理 ...
    
    # 新增: SID 特征
    sid_codes = features['item_sid']  # shape: [B, 3], 离线预计算
    sid_emb = self.sid_emb_layer(sid_codes)  # shape: [B, 192] (3×64)
    
    # 将 sid_emb 拼接到原有特征中
    all_features = torch.cat([
        original_features,
        sid_emb,
    ], dim=-1)
    
    return self.mlp(all_features)
```

**用户历史序列中的 SID 使用**：

```python
# 用户历史中的每个物品也用 SID 表示
history_sid_codes = get_history_sids(user_history)  # shape: [B, seq_len, L]

# 对序列中每个物品做 SID embedding
history_sid_embs = []
for i in range(seq_len):
    emb = sid_emb_layer(history_sid_codes[:, i, :])  # shape: [B, L*embed_dim]
    history_sid_embs.append(emb)

history_seq = torch.stack(history_sid_embs, dim=1)  # shape: [B, seq_len, L*embed_dim]

# 用 Attention/DIN 处理序列
user_interest = attention_pooling(history_seq, candidate_features)
```

### 4.2 进阶优化路径

在基础方案验证有效后，可依次尝试以下优化：

**优化一：添加 SID 交叉特征（参考 QARM）**

```python
# 候选物品的 SID 第一层码
candidate_c1 = candidate_sid[:, 0]  # shape: [B]

# 在用户历史中找到共享第一层码的物品
match_mask = (history_sid[:, :, 0] == candidate_c1.unsqueeze(1))  # shape: [B, seq_len]

# 提取匹配的子序列
matched_history = history_seq * match_mask.unsqueeze(-1)  # shape: [B, seq_len, d]

# Attention Pooling 得到交叉特征
cross_feature = attention_pooling(matched_history)  # shape: [B, d]
```

**优化二：CF 感知对齐（参考 TRM, DAS）**

在生成 SID 前，先用对比学习将内容 embedding 与协同过滤信号对齐：

```python
# 对比学习对齐
# 正例: 用户实际交互过的 (user_embedding, item_content_embedding) 对
# 负例: batch 内其他物品

contrastive_loss = InfoNCE(user_emb, aligned_item_content_emb)

# 对齐后的 embedding → RQ-VAE → SID（同时包含内容和协同信息）
```

**优化三：RQ + OPQ 双层编码（参考 SMILE）**

```python
# RQ: 粗粒度层次语义（共享/泛化）
rq_codes = rq_quantize(content_emb)     # shape: [B, L_rq]

# OPQ: 细粒度独特信息（判别/记忆）
opq_codes = opq_quantize(content_emb)   # shape: [B, M_opq]

# 两种编码各自有 embedding table，在模型中 concat
final_sid_emb = concat(rq_emb, opq_emb)
```

### 4.3 关键超参数与调参建议

| 超参数 | 建议范围 | 说明 |
|--------|---------|------|
| **码本大小 K** | 1024~8192 | K 越大，区分度越高但 table 越大；4096 是常见选择 |
| **量化层数 L** | 2~5 | L=3 是常见选择；层数越多表达力越强但边际收益递减 |
| **SID Embedding 维度** | 32~128 | 每层的 embedding 维度；64 是常见选择 |
| **组合方式** | sum 或 concat | sum 更省参数，concat 保留更多信息；视模型容量选择 |
| **Token 参数化** | SentencePiece 或 Prefix N-gram | SPM 效果更好但需额外训练；Prefix 更简单 |
| **对齐方式** | 对比学习（推荐）或无对齐 | 有 CF 对齐的 SID 效果显著更好 |

### 4.4 常见陷阱与解决方案

#### 陷阱一：直接用 Dense Embedding 替代 Item ID

> **YouTube 实验证实**：直接用 dense 内容 embedding 替换随机 Item ID **反而会降低排序质量**。

**原因**：生产排序模型严重依赖 ID embedding 的记忆能力（记住"用户 A 喜欢物品 B"这种精确事实）。Dense embedding 是连续共享空间，记忆能力弱。

**解决**：必须将 dense embedding **离散化**为 SID，再通过 embedding table 查找——这样既保留了记忆能力（通过 table），又获得了泛化能力（通过语义共享）。

#### 陷阱二：SID Embedding 冻结不可学习

> **QARM 消融实验证实**：冻结 SID embedding 导致效果大幅下降。

**原因**：SID 码本是基于内容重构优化的，与推荐目标有 gap。SID embedding 必须通过推荐梯度更新来弥合这个 gap。

**解决**：码本冻结，但 SID embedding table 参数必须可学习。

#### 陷阱三：纯内容 SID 丢失协同信号

> **TRM 和 DAS 实验证实**：纯内容 SID 效果不如 CF 感知 SID。

**原因**：两个内容相似的物品可能有完全不同的用户交互模式。纯内容 SID 无法区分。

**解决**：在量化前用对比学习/CF模型注入协同信号（参考 TRM, DAS, QARM 的对齐方法）。

#### 陷阱四：忽略 SID 在用户历史中的使用

> 只在候选物品侧使用 SID，忽略了用户历史序列。

**解决**：同时替换候选物品和用户历史中的 Item ID 为 SID，让模型在用户和物品两侧都利用语义信息。

---

## 5. 论文链接与参考文献

### 核心论文（LLM4DLRMs 范式，本报告重点）

| # | 论文 | 链接 | 机构 | 年份 |
|---|------|------|------|------|
| 1 | Better Generalization with Semantic IDs (YouTube) | [arXiv:2306.08121](https://arxiv.org/abs/2306.08121) | Google | 2023/2024 |
| 2 | Enhancing Embedding Representation Stability with SID (Meta Prefix N-gram) | [arXiv:2504.02137](https://arxiv.org/abs/2504.02137) | Meta | 2025 |
| 3 | SIDE: Semantic ID Embedding for effective learning from sequences | [arXiv:2506.16698](https://arxiv.org/abs/2506.16698) | Meta | 2025 |
| 4 | QARM: Quantitative Alignment Multi-Modal Recommendation | [arXiv:2411.11739](https://arxiv.org/abs/2411.11739) | 快手 | 2024 |
| 5 | QARM V2: Quantitative Alignment for Reasoning User Sequence Modeling | [arXiv:2602.09458](https://arxiv.org/abs/2602.09458) | 快手 | 2026 |
| 6 | SMILE: SeMantic Ids Enhanced CoLd Item Representation | [arXiv:2510.12604](https://arxiv.org/abs/2510.12604) | 快手电商 | 2025 |
| 7 | DAS: Dual-Aligned Semantic IDs Empowered Industrial Recommender System | [arXiv:2508.10584](https://arxiv.org/abs/2508.10584) | 快手广告 | 2025 |
| 8 | TRM: Farewell to Item IDs — Semantic Tokens for Large Ranking Models | [arXiv:2601.22694](https://arxiv.org/abs/2601.22694) | 字节跳动 | 2026 |
| 9 | DIG: Discrimination Is Generation — Unifying Ranking and Retrieval | [arXiv:2605.14853](https://arxiv.org/abs/2605.14853) | 美团 | 2026 |
| 10 | VQ-Rec: Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders | [arXiv:2210.12316](https://arxiv.org/abs/2210.12316) | 人大 | 2022 |

### 相关论文（LLM4GRs 范式或其他相关）

| # | 论文 | 链接 | 机构 | 年份 |
|---|------|------|------|------|
| 11 | TIGER: Recommender Systems with Generative Retrieval | [arXiv:2305.05065](https://arxiv.org/abs/2305.05065) | Google | 2023 |
| 12 | OneRec: Unifying Retrieve and Rank with Generative Recommender | [arXiv:2502.18965](https://arxiv.org/abs/2502.18965) | 快手 | 2025 |
| 13 | RankMixer: Scaling Up Ranking Models in Industrial Recommenders | [arXiv:2507.15551](https://arxiv.org/abs/2507.15551) | 字节跳动 | 2025 |
| 14 | MTGR: Industrial-Scale Generative Recommendation Framework | [arXiv:2505.18654](https://arxiv.org/abs/2505.18654) | 美团 | 2025 |
| 15 | PLUM: Adapting Pre-trained LMs for Industrial-scale Generative Recs | [arXiv:2510.07784](https://arxiv.org/abs/2510.07784) | Google/YouTube | 2025 |
| 16 | DOS: Dual-Flow Orthogonal Semantic IDs for Recommendation | [arXiv:2602.04460](https://arxiv.org/abs/2602.04460) | 美团 | 2026 |
| 17 | RPG: Recommendation with Parallel semantic ID Generation | [arXiv:2506.05781](https://arxiv.org/abs/2506.05781) | Meta | 2025 |
| 18 | DIGER: Differentiable Semantic ID for Generative Recommendation | [arXiv:2601.19711](https://arxiv.org/abs/2601.19711) | — | 2026 |
| 19 | UniSID: End-to-End Semantic ID Generation for Generative Ad Rec | [arXiv:2602.10445](https://arxiv.org/abs/2602.10445) | 腾讯 | 2026 |
| 20 | MMQ: Multimodal Mixture-of-Quantization Tokenization | [arXiv:2508.15281](https://arxiv.org/abs/2508.15281) | 阿里 | 2025 |
| 21 | FORGE: Forming Semantic Identifiers for Generative Retrieval | [arXiv:2509.20904](https://arxiv.org/abs/2509.20904) | 阿里/淘宝 | 2025 |
| 22 | COBRA: Unified GenRec with Cascaded Sparse-Dense Representations | [arXiv:2503.02453](https://arxiv.org/abs/2503.02453) | 百度 | 2025 |
