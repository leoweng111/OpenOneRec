# 生成式推荐中的模型架构与损失函数调研

## 目录

- [1. 概述](#1-概述)
  - [1.1 生成式推荐的架构全景](#11-生成式推荐的架构全景)
  - [1.2 架构分类维度](#12-架构分类维度)
  - [1.3 损失函数概览](#13-损失函数概览)
  - [1.4 架构演进脉络](#14-架构演进脉络)
- [2. Encoder-Decoder 架构](#2-encoder-decoder-架构)
  - [2.1 TIGER：奠基之作](#21-tiger奠基之作)
  - [2.2 OneRec V1：工业级 Enc-Dec](#22-onerec-v1工业级-enc-dec)
  - [2.3 OneSearch：电商搜索 Enc-Dec](#23-onesearch电商搜索-enc-dec)
- [3. Decoder-Only 架构](#3-decoder-only-架构)
  - [3.1 OneRec V2：Lazy Decoder-Only](#31-onerec-v2lazy-decoder-only)
  - [3.2 PLUM：LLM适配 Decoder-Only](#32-plumllm适配-decoder-only)
  - [3.3 HSTU：Meta万亿参数序列转导](#33-hstumeta万亿参数序列转导)
  - [3.4 RPG：并行多token预测](#34-rpg并行多token预测)
- [4. LLM4DLRMs 架构（判别式推荐 + 语义ID）](#4-llm4dlrms-架构判别式推荐--语义id)
  - [4.1 QARM：多模态对齐 + DLRM](#41-qarm多模态对齐--dlrm)
  - [4.2 QARM V2：推理增强 + Res-KmeansFSQ](#42-qarm-v2推理增强--res-kmeansfsq)
  - [4.3 SIDE：Meta广告序列学习](#43-side-meta广告序列学习)
  - [4.4 DAS：双对齐架构](#44-das双对齐架构)
  - [4.5 YouTube Semantic IDs：排序模型中嵌入语义ID](#45-youtube-semantic-ids排序模型中嵌入语义id)
- [5. 混合与级联架构](#5-混合与级联架构)
  - [5.1 COBRA：稀疏-稠密级联](#51-cobra稀疏-稠密级联)
  - [5.2 UniSID：端到端多粒度对比](#52-unisid端到端多粒度对比)
  - [5.3 MMQ：MoE混合量化架构](#53-mmqmoe混合量化架构)
- [6. 架构演进深度分析：从 Enc-Dec 到 Decoder-Only](#6-架构演进深度分析从-enc-dec-到-decoder-only)
- [7. 损失函数详解](#7-损失函数详解)
  - [7.1 Next Token Prediction（NTP）](#71-next-token-predictionntp)
  - [7.2 对比学习损失](#72-对比学习损失)
  - [7.3 强化学习损失（GRPO/PPO）](#73-强化学习损失grpoppo)
  - [7.4 量化损失（RQ-VAE/PQ）](#74-量化损失rq-vaepq)
- [8. 总结与展望](#8-总结与展望)
- [参考文献](#参考文献)

---

## 1. 概述

### 1.1 生成式推荐的架构全景

生成式推荐（Generative Recommendation, GR）将推荐任务重构为序列生成问题，核心是用Transformer等序列模型预测用户下一个会交互的物品。与传统判别式推荐（DLRM、双塔等）不同，GR统一了检索和排序，形成了端到端的生成框架。

目前生成式推荐的模型架构可以分为三大类：

```
                    生成式推荐模型架构
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
    Encoder-Decoder   Decoder-Only   LLM4DLRMs
    (Seq2Seq风格)     (GPT风格)      (判别式+语义ID)
          │              │              │
    TIGER            OneRec V2       QARM, SIDE
    OneRec V1        PLUM            DAS, YouTube
    OneSearch        HSTU (Meta)
                     RPG
```

### 1.2 架构分类维度

| 维度 | 分类 | 说明 |
|------|------|------|
| **模型范式** | 生成式 (GRs) vs 判别式 (DLRMs) | GRs自回归生成SID；DLRMs全库打分 |
| **骨干结构** | Enc-Dec / Decoder-Only / Encoder-Only | 序列建模方式不同 |
| **语义ID角色** | 生成目标 / 输入特征 | GRs生成SID；DLRMs以SID为特征 |
| **训练阶段** | Pretrain → SFT → RL | 多阶段训练流程 |

### 1.3 损失函数概览

生成式推荐中使用的损失函数按训练阶段分类：

| 阶段 | 损失函数 | 数学形式 | 代表工作 |
|------|----------|----------|----------|
| Pretrain / SFT | Next Token Prediction (CE) | `-log p(tok_{k+1} \| tok_{≤k})` | TIGER, OneRec, PLUM |
| 语义ID训练 | 量化损失 (RQ-VAE/PQ) | 重构 + 码本 + 承诺 | TIGER, QARM |
| 对齐训练 | 对比学习损失 | InfoNCE / Triplet | DAS, UniSID |
| RL对齐 | Policy Gradient (PPO/GRPO) | $-\min(\rho_t A_t, \text{clip}(\rho_t)\cdot A_t)$ | OneRec V2 |
| DLRM训练 | 二元交叉熵 / 多任务 | $-[y\log\hat{y} + (1-y)\log(1-\hat{y})]$ | QARM, YouTube |

### 1.4 架构演进脉络

```
2023  TIGER ────────────── Encoder-Decoder (T5-based)
       │                     Seq2Seq生成式检索
       │
2024  YouTube SIDs ──────── DLRM + 语义ID特征
       │                     判别式排序, 语义ID仅替换Item ID
       │
2025  OneRec V1 ─────────── Encoder-Decoder + MoE
       │                     统一检索排序, 但97.6%算力在Encoder
       │
2025  HSTU (Meta) ───────── Decoder-Only (定制序列转导)
       │                     1.5万亿参数, 比FA2快5-15×
       │
2025  OneRec V2 ─────────── Lazy Decoder-Only
       │                     去掉Encoder, 算力100%用于生成, 减少94%计算
       │
2025  PLUM ──────────────── Decoder-Only (复用预训练LLM)
       │                     CPT + 任务微调, YouTube部署
       │
2025  RPG (Meta) ────────── Decoder-Only + 并行MTP
       │                     多token并行预测, 图约束解码
       │
2026  OneRec-Think ──────── Decoder-Only + 推理链
                             Think-Ahead架构, 显式推理
```

**核心演进趋势**：Encoder-Decoder → Decoder-Only，原因详见[§6](#6-架构演进深度分析从-enc-dec-到-decoder-only)。

---

## 2. Encoder-Decoder 架构

**核心思想**：Encoder编码用户行为序列为上下文表示，Decoder自回归生成目标物品的语义ID。类似NLP中的机器翻译（源语言→目标语言）。

### 采用Encoder-Decoder架构的论文

| 论文 | 链接 | 使用范式 | Encoder | Decoder | 核心改进 |
|------|------|----------|---------|---------|----------|
| [TIGER](#21-tiger奠基之作) (Google, 2023) | [arXiv:2305.05065](https://arxiv.org/abs/2305.05065) | LLM4GRs | Transformer Enc | Transformer Dec | 奠基工作 |
| [OneRec V1](#22-onerec-v1工业级-enc-dec) (快手, 2025) | [arXiv:2502.18965](https://arxiv.org/abs/2502.18965) | LLM4GRs | MoE Encoder | MoE Decoder | 统一检索排序 |
| [OneSearch](#23-onesearch电商搜索-enc-dec) (快手, 2025) | [arXiv:2509.03236](https://arxiv.org/abs/2509.03236) | LLM4GRs | 用户行为编码 | 生成式检索 | RQ-OPQ语义ID |

### 2.1 TIGER：奠基之作

- **论文**：[Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) (NeurIPS 2023)
- **架构类型**：**Encoder-Decoder**（基于T5）
- **使用范式**：LLM4GRs

**架构详解**：

```
┌─────────────────────────────────────────────────────────┐
│                    TIGER 架构                            │
│                                                         │
│  ┌───────────────────────┐                              │
│  │     Encoder (T5)       │                              │
│  │                        │                              │
│  │  输入: 用户历史SID序列   │                              │
│  │  [SID₁] [SID₂] [SID₃] │                              │
│  │      ↓                 │                              │
│  │  Bi-directional SA     │                              │
│  │  + FFN × N layers      │                              │
│  │      ↓                 │                              │
│  │  上下文表示 H_enc       │                              │
│  └──────────┬────────────┘                              │
│             ↓ Cross-Attention                           │
│  ┌───────────────────────┐                              │
│  │     Decoder (T5)       │                              │
│  │                        │                              │
│  │  输入: <BOS> + 已生成SID│                              │
│  │      ↓                 │                              │
│  │  Causal SA + Cross-SA  │                              │
│  │  + FFN × N layers      │                              │
│  │      ↓                 │                              │
│  │  LM Head → 预测下一token│                             │
│  │  (语义ID的c₁, c₂, c₃) │                              │
│  └───────────────────────┘                              │
│                                                         │
│  推理: Beam Search逐token生成语义ID                      │
└─────────────────────────────────────────────────────────┘
```

**损失函数**：标准Seq2Seq交叉熵

$$\mathcal{L} = -\sum_{t=1}^{T} \log p_\theta(y_t \mid y_{\lt t}, \mathbf{H}_{\text{enc}})$$

其中 $y_t$ 是目标语义ID的第 $t$ 个token，$\mathbf{H}_{\text{enc}}$ 是Encoder输出。

**推理**：Beam Search自回归生成语义ID的每个token。利用RQ-VAE的层次结构做层次感知beam search：第1层（粗粒度）beam较小，逐层扩展。

### 2.2 OneRec V1：工业级 Enc-Dec

- **论文**：[OneRec: Unifying Retrieve and Rank with Generative Recommender](https://arxiv.org/abs/2502.18965) (2025)
- **架构类型**：**Encoder-Decoder + MoE**
- **使用范式**：LLM4GRs

**架构详解**：

```
┌─────────────────────────────────────────────────────────────┐
│                     OneRec V1 架构                           │
│                                                             │
│  ┌──────────────────────────────┐                           │
│  │       Tokenizer (离线)        │                           │
│  │  视频 → RQ-KMeans → 语义ID    │                           │
│  │  (多级别衡量化)                │                           │
│  └──────────────────────────────┘                           │
│                                                             │
│  ┌──────────────────────────────┐                           │
│  │     Encoder (MoE)            │                           │
│  │                              │                           │
│  │  用户行为序列: [SID₁...SID_n] │                           │
│  │      ↓                       │                           │
│  │  Self-Attention + MoE FFN    │ ← 97.66%算力在这里        │
│  │  (多专家路由,捕获多样兴趣)     │                           │
│  │      ↓                       │                           │
│  │  用户兴趣表示 H_user          │                           │
│  └──────────┬───────────────────┘                           │
│             ↓ Cross-Attention                               │
│  ┌──────────────────────────────┐                           │
│  │     Decoder (MoE)            │                           │
│  │                              │                           │
│  │  自回归生成: <BOS> → SID_{n+1}│ ← 仅2.34%算力            │
│  │      ↓                       │                           │
│  │  Causal SA + Cross-SA + MoE  │                           │
│  │      ↓                       │                           │
│  │  LM Head → 预测SID token     │                           │
│  └──────────────────────────────┘                           │
│                                                             │
│  ┌──────────────────────────────┐                           │
│  │  Reward System + DPO对齐      │                           │
│  │  Iterative Preference Align  │                           │
│  └──────────────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

**关键问题**：Encoder消耗了97.66%的计算资源，Decoder（真正做生成和算loss的部分）仅获得2.34%的算力。这成为V2架构升级的核心动因。

**损失函数**：
- SFT阶段：Next Token Prediction CE，loss仅在新曝光的target item上
- RL阶段：GRPO策略梯度 + KL正则

**部署效果**：快手主场景，观看时长 +1.6%。

### 2.3 OneSearch：电商搜索 Enc-Dec

- **论文**：[OneSearch: End-to-End Generative Framework for E-commerce Search](https://arxiv.org/abs/2509.03236) (2025)
- **架构类型**：**Encoder-Decoder**
- **使用范式**：LLM4GRs

**架构特点**：
- Encoder：编码用户query + 多视角行为序列（短期显式 + 长期隐式）
- Decoder：自回归生成目标物品的RQ-OPQ混合语义ID
- PARS（Preference-Aware Reward System）：多阶段SFT + 自适应奖励加权排序
- Model FLOPs Utilization从3.26%提升至27.32%

---

## 3. Decoder-Only 架构

**核心思想**：去掉Encoder，用户行为序列和生成目标在同一个Decoder中用Causal Attention处理。类似GPT的自回归语言模型。

### 采用Decoder-Only架构的论文

| 论文 | 链接 | 使用范式 | 核心改进 |
|------|------|----------|----------|
| [OneRec V2](#31-onerec-v2lazy-decoder-only) (快手, 2025) | [arXiv:2508.20900](https://arxiv.org/abs/2508.20900) | LLM4GRs | Lazy Decoder-Only, 减少94%计算 |
| [PLUM](#32-plumllm适配-decoder-only) (Google, 2025) | [arXiv:2510.07784](https://arxiv.org/abs/2510.07784) | LLM4GRs | 复用预训练LLM, CPT+微调 |
| [HSTU](#33-hstumeta万亿参数序列转导) (Meta, 2024) | [arXiv:2402.17152](https://arxiv.org/abs/2402.17152) | LLM4GRs | 定制序列转导, 1.5万亿参数 |
| [RPG](#34-rpg并行多token预测) (Meta, 2025) | [arXiv:2506.05781](https://arxiv.org/abs/2506.05781) | LLM4GRs | 多token并行预测, 图约束解码 |
| [OneRec-Think](https://arxiv.org/abs/2510.11639) (快手, 2026) | [arXiv:2510.11639](https://arxiv.org/abs/2510.11639) | LLM4GRs | Think-Ahead推理增强 |

### 3.1 OneRec V2：Lazy Decoder-Only

- **论文**：[OneRec-V2 Technical Report](https://arxiv.org/abs/2508.20900) (2025)
- **架构类型**：**Lazy Decoder-Only**
- **使用范式**：LLM4GRs

**架构详解**：

```
┌──────────────────────────────────────────────────────────────┐
│                    OneRec V2 架构                              │
│                                                              │
│  输入序列（全部在同一个Decoder中）:                              │
│                                                              │
│  [User Profile] [SID₁] [SID₂] ... [SID_n] │ [Target SID]    │
│  ────── context（loss_mask=0） ────────── │ ─ loss_mask=1 ─  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │          Decoder-Only Transformer                       │  │
│  │                                                        │  │
│  │  Causal Self-Attention                                 │  │
│  │  (每个位置只能attend到≤t的位置)                          │  │
│  │                                                        │  │
│  │  + FFN × N layers                                      │  │
│  │                                                        │  │
│  │  100% 算力用于生成（context+target一起处理）              │  │
│  └────────────────────────────────────────────────────────┘  │
│                          ↓                                   │
│                   LM Head → 预测SID token                    │
│                                                              │
│  去除: Encoder, Cross-Attention, K/V Projection             │
│  减少: 94%总计算量, 90%训练资源                                │
│  扩展: 可达80亿参数, 3000 token上下文                          │
│  推理: L20 GPU上36ms延迟                                     │
└──────────────────────────────────────────────────────────────┘
```

**损失函数**：New Impression Only — loss仅在target item上

$$\mathcal{L}_{\text{NIO}} = \frac{1}{B \cdot L_{\text{target}}} \sum_{b=1}^{B} \sum_{k \in \text{target}(b)} -\log p_\theta(\text{tok}^b_{k+1} \mid \text{tok}^b_{\le k})$$

**RL对齐**：
- Duration-Aware Reward Shaping：显式建模观看时长、点赞、评论、关注等多维反馈
- Adaptive Ratio Clipping：自适应裁剪比例，更好对齐真实用户偏好

**部署效果**：App Stay Time +0.467%~0.741%。

### 3.2 PLUM：LLM适配 Decoder-Only

- **论文**：[PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations](https://arxiv.org/abs/2510.07784) (2025)
- **架构类型**：**Decoder-Only**（复用预训练LLM，如Gemma/PaLM）
- **使用范式**：LLM4GRs

**三阶段训练流程**：

```
┌─────────────────────────────────────────────────────────┐
│                    PLUM 架构                              │
│                                                         │
│  Stage 1: Item Tokenization (SID-v2)                    │
│  ┌────────────────────────────────────────┐             │
│  │ 多模态内容 → 融合表示 → 层次量化 → SID   │             │
│  └────────────────────────────────────────┘             │
│                                                         │
│  Stage 2: Continued Pre-training (CPT)                  │
│  ┌────────────────────────────────────────┐             │
│  │ 预训练LLM (e.g., Gemma)               │             │
│  │   ↓ 扩展词汇表 (加入SID tokens)        │             │
│  │   ↓ 混合数据继续预训练:                 │             │
│  │     • 领域物品数据                      │             │
│  │     • 用户行为序列                      │             │
│  │     • 通用文本数据                      │             │
│  └────────────────────────────────────────┘             │
│                                                         │
│  Stage 3: Task-Specific Fine-tuning                     │
│  ┌────────────────────────────────────────┐             │
│  │ 生成式检索微调:                          │             │
│  │   输入: 用户上下文 prompt                │             │
│  │   输出: 自回归生成推荐物品的SID           │             │
│  │   Loss: Next Token Prediction CE        │             │
│  └────────────────────────────────────────┘             │
│                                                         │
│  YouTube Shorts: Panel CTR +4.96%                       │
└─────────────────────────────────────────────────────────┘
```

**关键创新**：
- **SID-v2**：融合多模态内容的改进版语义ID，层次量化增强
- **CPT混合数据**：领域数据+通用文本混合预训练，弥合domain gap
- **高样本效率**：每天仅需数亿样本训练，远少于传统LEM需要的数十亿

### 3.3 HSTU：Meta万亿参数序列转导

- **论文**：[Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/abs/2402.17152) (2024)
- **架构类型**：**定制Decoder-Only**（HSTU = Hierarchical Sequential Transduction Units）
- **使用范式**：LLM4GRs

**HSTU与标准Transformer的区别**：

```
标准Transformer:                    HSTU:
┌──────────────────┐               ┌──────────────────────┐
│ Multi-Head SA    │               │ Pointwise Aggregated │
│ (Q,K,V全投影)     │               │ Attention            │
│ + LayerNorm      │               │ (优化推荐数据特征)     │
│ + FFN            │               │ + 去除部分冗余层       │
│ + LayerNorm      │               │ + 定制Norm            │
└──────────────────┘               └──────────────────────┘
                                    比FA2快 5.3×~15.2×
```

**核心设计**：
- 针对推荐数据的**高基数**（物品ID空间巨大）和**非平稳性**（分布随时间变化）优化
- 去除标准Transformer中的部分LayerNorm和FFN层，简化架构
- Pointwise Aggregated Attention：利用推荐数据集特征（如item流行度分布）优化注意力计算

**推理加速 — M-FALCON算法**：
- 通过micro-batching和caching实现推理900×加速
- 可服务285×更复杂的GR模型

**Scaling Law**：首次观察到推荐领域的power-law scaling（类似GPT-3/LLaMA-2），模型质量随训练计算量幂律增长。

**规模与效果**：
- 最大1.5万亿参数
- 比FlashAttention2 Transformer快5.3×~15.2×
- 在线A/B测试指标提升12.4%
- 部署于Meta多个服务数十亿用户

### 3.4 RPG：并行多token预测

- **论文**：[Recommendation with Parallel semantic ID Generation](https://arxiv.org/abs/2506.05781) (KDD 2025)
- **架构类型**：**Decoder-Only + Multi-Token Prediction**
- **使用范式**：LLM4GRs

**架构特点**：

```
┌─────────────────────────────────────────────────────────┐
│                    RPG 架构                              │
│                                                         │
│  用户行为序列 → Transformer Encoder → h ∈ R^d           │
│                                          ↓              │
│  ┌──────────────────────────────────────────┐           │
│  │     Multi-Token Prediction Head           │           │
│  │                                          │           │
│  │  h → MLP₁ → c₁  ┐                      │           │
│  │  h → MLP₂ → c₂  │                      │           │
│  │  h → MLP₃ → c₃  ├→ 并行预测所有token    │           │
│  │  ...              │                      │           │
│  │  h → MLP_M → c_M┘                      │           │
│  └──────────────────────────────────────────┘           │
│                                          ↓              │
│  语义ID = (c₁, c₂, ..., c_M)  (最多64个token)          │
│                                                         │
│  推理: 图约束解码 (Graph-Constrained Decoding)           │
│  构建合法SID的图, 沿图搜索避免无效组合                    │
└─────────────────────────────────────────────────────────┘
```

**训练目标**：Multi-Token Prediction (MTP)

$$\mathcal{L}_{\text{MTP}} = \sum_{m=1}^{M} \mathcal{L}_{\text{CE}}(c_m, \hat{c}_m)$$

每个token独立预测，互不依赖。

**图约束解码**：构建合法SID组合的图结构，推理时沿图搜索，避免生成无效的SID组合（解决并行生成的组合空间稀疏问题）。

**效果**：NDCG@10平均提升12.6%，同时推理更快、内存更省。

---

## 4. LLM4DLRMs 架构（判别式推荐 + 语义ID）

**核心思想**：语义ID作为输入特征，替换传统推荐模型中的随机Item ID。推荐模型主体仍为判别式架构（双塔、DNN、DLRM等），不做自回归生成。

### 采用LLM4DLRMs架构的论文

| 论文 | 链接 | 推荐模型类型 | 语义ID方法 | 核心改进 |
|------|------|-------------|------------|----------|
| [QARM](#41-qarm多模态对齐--dlrm) (快手, 2024) | [arXiv:2411.11739](https://arxiv.org/abs/2411.11739) | DLRM/DNN | Res-KMeans | 多模态对齐 |
| [QARM V2](#42-qarm-v2推理增强--res-kmeansfsq) (快手, 2026) | [arXiv:2602.09458](https://arxiv.org/abs/2602.09458) | DLRM/DNN | Res-KMeans+FSQ | LLM推理增强 |
| [SIDE](#43-side-meta广告序列学习) (Meta, 2025) | [arXiv:2506.16698](https://arxiv.org/abs/2506.16698) | 广告DLRM | VQ-Fusion | 无参数SID转换 |
| [DAS](#44-das双对齐架构) (快手, 2025) | [arXiv:2508.10584](https://arxiv.org/abs/2508.10584) | 双塔/CF | 兼容多种 | 双对齐协同信号 |
| [YouTube SIDs](#45-youtube-semantic-ids排序模型中嵌入语义id) (Google, 2024) | [arXiv:2306.08121](https://arxiv.org/abs/2306.08121) | 排序DNN | RQ-VAE | 十亿级排序系统 |

### 4.1 QARM：多模态对齐 + DLRM

- **论文**：[QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou](https://arxiv.org/abs/2411.11739) (2024)
- **架构类型**：**DLRM + 语义ID特征**
- **使用范式**：LLM4DLRMs

```
┌─────────────────────────────────────────────────────────┐
│                    QARM 架构                              │
│                                                         │
│  离线部分:                                               │
│  ┌────────────────────────────────────────┐             │
│  │ 多模态预训练 → Item Alignment → Res-KMeans│             │
│  │ → Semantic ID (c₁, c₂, ..., c_L)       │             │
│  └────────────────────────────────────────┘             │
│                                                         │
│  在线推荐模型:                                            │
│  ┌────────────────────────────────────────┐             │
│  │ 输入:                                   │             │
│  │  ├── User Features                     │             │
│  │  ├── Context Features                  │             │
│  │  └── Item Features:                    │             │
│  │      ├── Semantic ID → SID Embedding   │ ← 替换随机ID│
│  │      └── 其他物品特征                    │             │
│  │      ↓ 拼接                             │             │
│  │  DNN Layers → CTR预估                   │             │
│  │      ↓                                  │             │
│  │  σ(Wx + b) → P(click)                  │             │
│  └────────────────────────────────────────┘             │
│                                                         │
│  Loss: 二元交叉熵 + 多任务损失                            │
│  $\mathcal{L} = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$│
│                                                         │
│  广告收入 +9.7%, 电商GMV +2.3%                           │
└─────────────────────────────────────────────────────────┘
```

### 4.2 QARM V2：推理增强 + Res-KmeansFSQ

- **论文**：[QARM V2: Quantitative Alignment Multi-Modal Recommendation for Reasoning User Sequence Modeling](https://arxiv.org/abs/2602.09458) (2026)
- **架构类型**：**DLRM + LLM推理增强语义ID**
- **使用范式**：LLM4DLRMs

**V1→V2的架构升级**：
- Item Alignment → **Reasoning Item Alignment**（利用LLM推理能力增强物品间逻辑关联）
- Res-KMeans → **Res-KmeansFSQ**（混合FSQ避免码本冲突）
- 用户序列建模增强：更强的LLM-based embedding

### 4.3 SIDE：Meta广告序列学习

- **论文**：[SIDE: Semantic ID Embedding for effective learning from sequences](https://arxiv.org/abs/2506.16698) (AdKDD 2025)
- **架构类型**：**广告DLRM + VQ-Fusion**
- **使用范式**：LLM4DLRMs

**三项架构创新**：
1. **VQ-Fusion**：多任务VQ-VAE，融合多种内容embedding + 类别预测为单一SID
2. **Discrete-PCA (DPCA)**：PCA预处理旋转后残差量化，提升量化效率
3. **Parameter-free SID-to-Embedding**：无参数SID→embedding转换，消除大查找表

### 4.4 DAS：双对齐架构

- **论文**：[DAS: Dual-Aligned Semantic IDs Empowered Industrial Recommender System](https://arxiv.org/abs/2508.10584) (CIKM 2025)
- **架构类型**：**双塔 + 协同过滤 + 对比对齐**
- **使用范式**：LLM4DLRMs / LLM4GRs 兼容

```
┌───────────────────────────────────────────────────────┐
│                    DAS 架构                            │
│                                                       │
│  ┌──────────────────┐   ┌──────────────────────┐     │
│  │ User Tower        │   │ Item Tower            │     │
│  │ user_feats → h_u  │   │ item_feats → h_i     │     │
│  └────────┬─────────┘   └──────────┬───────────┘     │
│           ↓                        ↓                  │
│  ┌────────────────┐     ┌──────────────────┐         │
│  │ User Quantizer │     │ Item Quantizer    │         │
│  │ h_u → SID_u   │     │ h_i → SID_i      │         │
│  └────────┬───────┘     └──────────┬────────┘         │
│           ↓                        ↓                  │
│  ┌────────────────────────────────────────────┐       │
│  │        CF Debias Module                     │       │
│  │  (协同过滤去偏, 注入交互信号)                  │       │
│  └────────────────────┬───────────────────────┘       │
│                       ↓                               │
│  ┌────────────────────────────────────────────┐       │
│  │  Multi-View Contrastive Alignment          │       │
│  │  L_align = L_u2i + L_i2i + L_co-occur     │       │
│  └────────────────────────────────────────────┘       │
│                       ↓                               │
│  ┌────────────────────────────────────────────┐       │
│  │  Dual Learning: User量化 ↔ Item量化 对偶训练  │       │
│  └────────────────────────────────────────────┘       │
└───────────────────────────────────────────────────────┘
```

**兼容多种量化方法**：VQ-VAE、RQ-VAE、PQ等。
**兼容多种CF方法**：DSSM、GCN、DCCF等。

### 4.5 YouTube Semantic IDs：排序模型中嵌入语义ID

- **论文**：[Better Generalization with Semantic IDs](https://arxiv.org/abs/2306.08121) (RecSys 2024)
- **架构类型**：**排序DNN + 语义ID特征**
- **使用范式**：LLM4DLRMs

**架构核心**：在YouTube十亿级排序模型中，用RQ-VAE生成的语义ID（经SentencePiece适配）替换随机Item ID embedding。

```
排序模型输入:
  ├── User Features (age, gender, history, ...)
  ├── Context Features (time, device, ...)
  └── Item Features:
      ├── 传统: Random Item ID → Embedding → 拼接
      └── 改进: Semantic ID → SentencePiece → Embedding → 拼接
                                                         ↓
                                                  DNN → P(click)
```

**关键发现**：语义ID在**记忆**（memorization）和**泛化**（generalization）之间取得最优平衡。

---

## 5. 混合与级联架构

### 5.1 COBRA：稀疏-稠密级联

- **论文**：[Unified Generative Recommendations with Cascaded Sparse-Dense Representations](https://arxiv.org/abs/2503.02453) (2025, 百度)
- **架构类型**：**级联生成（稀疏SID → 稠密向量）**

```
┌─────────────────────────────────────────────────────────┐
│                    COBRA 架构                            │
│                                                         │
│  Step 1: 稀疏生成                                       │
│  ┌────────────────────────────────────────┐             │
│  │ 用户历史 → Transformer → 自回归生成       │             │
│  │ → 稀疏语义ID (coarse retrieval)          │             │
│  └────────────────────┬───────────────────┘             │
│                       ↓ 以SID为条件                      │
│  Step 2: 稠密生成                                       │
│  ┌────────────────────────────────────────┐             │
│  │ 条件: 稀疏SID                            │             │
│  │ → 生成稠密向量 (fine-grained ranking)     │             │
│  │ → 最近邻检索精排                           │             │
│  └────────────────────────────────────────┘             │
│                                                         │
│  推理: BeamFusion = Beam Search + 最近邻分数              │
│  优势: 稀疏泛化 + 稠密精度, 解决纯SID的信息损失            │
└─────────────────────────────────────────────────────────┘
```

### 5.2 UniSID：端到端多粒度对比

- **论文**：[End-to-End Semantic ID Generation for Generative Advertisement Recommendation](https://arxiv.org/abs/2602.10445) (2026)
- **架构类型**：**端到端联合优化embedding + SID**

```
┌─────────────────────────────────────────────────────────┐
│                    UniSID 架构                           │
│                                                         │
│  广告增强输入:                                            │
│  [指令] [图片] [文本] [属性] [可学习SID tokens] [emb token]│
│                          ↓                               │
│  ┌──────────────────────────────────────────┐           │
│  │  Transformer Backbone                     │           │
│  │  端到端联合优化:                            │           │
│  │  • SID生成 (离散码)                        │           │
│  │  • Embedding学习 (连续向量)                 │           │
│  │  • 多粒度对比学习 (不同SID层级对齐)           │           │
│  └──────────────────────────────────────────┘           │
│                                                         │
│  摘要式广告重构: 鼓励SID捕获高层语义                       │
│  效果: Hit Rate +4.62%                                  │
└─────────────────────────────────────────────────────────┘
```

### 5.3 MMQ：MoE混合量化架构

- **论文**：[MMQ: Multimodal Mixture-of-Quantization Tokenization](https://arxiv.org/abs/2508.15281) (WSDM 2026)
- **架构类型**：**MoE混合量化**

```
输入embedding → Router → 权重分配
                ↓
    ┌───────────┼───────────┐
    ↓           ↓           ↓
Expert₁      Expert₂     Expert₃
(模态特定    (模态特定     (共享
 码本)        码本)        码本)
    ↓           ↓           ↓
    └───────────┼───────────┘
                ↓
          加权聚合 → 语义ID
```

---

## 6. 架构演进深度分析：从 Enc-Dec 到 Decoder-Only

### 6.1 OneRec V1 → V2 的教训

OneRec的架构演进是Encoder-Decoder → Decoder-Only最有说服力的工业案例。

**V1 (Encoder-Decoder) 的问题**：

| 问题 | 具体数据 |
|------|----------|
| 算力分配不均 | Encoder占97.66%，Decoder仅2.34% |
| 生成能力受限 | 真正做生成和loss计算的部分获得极少算力 |
| 扩展性差 | 增大Encoder不能直接提升生成质量 |
| 训练效率低 | 大量计算花在context编码上，不产生loss |

**V2 (Lazy Decoder-Only) 的改进**：

| 改进 | 效果 |
|------|------|
| 去掉Encoder | 总计算量减少94% |
| 去掉Cross-Attention和K/V投影 | 架构简化，训练资源减少90% |
| 100%算力用于生成 | context和target统一在Decoder处理 |
| 可扩展至80亿参数 | 收敛loss遵循scaling law |
| L20 GPU推理36ms | 满足工业实时性要求 |

### 6.2 为什么 Decoder-Only 更适合生成式推荐

**1. 推荐 ≠ 翻译**

NLP中的机器翻译需要理解源语言（Encoder）再生成目标语言（Decoder），两者语义空间不同。但推荐中"用户行为序列"和"目标物品SID"本质上在**同一个语义空间**（都是物品的SID表示），不需要跨空间翻译。Decoder-Only的Causal Attention天然就能处理这种"同空间序列预测"。

**2. 算力应花在生成上**

生成式推荐的核心任务是"预测下一个SID"。Encoder的大量计算用于理解用户历史（不产生loss），而真正产生梯度的生成部分反而算力不足。Decoder-Only让所有计算都服务于"预测下一个token"。

**3. 与LLM生态对齐**

Decoder-Only是当前LLM的主流架构（GPT、LLaMA等）。使用Decoder-Only可以直接复用：
- 预训练权重（PLUM的做法）
- 推理优化（KV Cache、FlashAttention等）
- 训练框架（FSDP、DeepSpeed等）
- Scaling Law的经验

**4. 流式训练更自然**

New Impression Only等流式样本组织方式天然适合Decoder-Only：每条样本就是"context → target"的一段序列，直接输入Decoder处理。Encoder-Decoder则需要额外的context/target分离逻辑。

### 6.3 演进对比总结

| 维度 | Encoder-Decoder | Decoder-Only |
|------|-----------------|--------------|
| 算力分配 | 大量给Encoder（不产生loss） | 100%给序列预测 |
| 架构复杂度 | 高（双Transformer + Cross-Attn） | 低（单Transformer） |
| 扩展性 | 差（增大Encoder收益递减） | 好（遵循scaling law） |
| LLM生态复用 | 弱（T5-style较少） | 强（GPT-style主流） |
| 流式训练适配 | 中 | 好 |
| 推理延迟 | 较高（两次前向） | 较低（一次前向+KV Cache） |
| 代表 | TIGER, OneRec V1, OneSearch | OneRec V2, PLUM, HSTU |

---

## 7. 损失函数详解

### 7.1 Next Token Prediction（NTP）

**使用场景**：所有LLM4GRs的Pretrain/SFT阶段。

$$\mathcal{L}_{\text{NTP}} = -\frac{1}{|\mathcal{S}|}\sum_{k \in \mathcal{S}} \log p_\theta(\text{tok}_{k+1} \mid \text{tok}_{\le k})$$

其中 $p_\theta(\cdot) = \text{softmax}(W_{\text{lm\_head}} h_k) \in \mathbb{R}^V$，$V$ 是词表大小。

**$\mathcal{S}$ 的选择取决于样本组织方式**：
- Naive Impression：$\mathcal{S}$ = 全序列所有token位置
- User-Centric：$\mathcal{S}$ = 用户序列所有token位置
- New Impression Only：$\mathcal{S}$ = 仅target item的token位置

**代码实现**（`pretrain/onerec_llm/losses/ce.py`）：
```python
logits_flat = logits.float().reshape(-1, vocab_size)     # (T, V)  V≈176k
labels_flat = labels.reshape(-1)                          # (T,)
per_token_loss = F.cross_entropy(
    logits_flat, labels_flat,
    ignore_index=-100,        # loss_mask=0的位置被屏蔽
    reduction="none",
)
loss = per_token_loss.sum() / (labels_flat != -100).sum()
```

### 7.2 对比学习损失

**使用场景**：DAS的多视角对齐、UniSID的多粒度对比学习。

**InfoNCE Loss**：
$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z_i, z_j^+)/\tau)}{\sum_{k=1}^{N}\exp(\text{sim}(z_i, z_k)/\tau)}$$

其中 $\text{sim}$ 是余弦相似度，$\tau$ 是温度参数，$z_j^+$ 是正样本。

**DAS的多视角对比损失**：
$$\mathcal{L}_{\text{align}} = \mathcal{L}_{\text{u2i}} + \mathcal{L}_{\text{i2i}} + \mathcal{L}_{\text{co-occur}}$$

- $\mathcal{L}_{\text{u2i}}$：用户-物品对齐（拉近交互过的user-item对）
- $\mathcal{L}_{\text{i2i}}$：物品-物品对齐（拉近相似物品）
- $\mathcal{L}_{\text{co-occur}}$：共现物品对齐

### 7.3 强化学习损失（GRPO/PPO）

**使用场景**：OneRec V2的偏好对齐阶段。

**GRPO优势估计**：
$$A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G) + \varepsilon}$$

其中 $r_i$ 是第 $i$ 条rollout的reward（如SID命中率），$G$ 是group大小（如32条候选）。

**PPO-clip策略梯度**：
$$\mathcal{L}_{\text{policy}} = -\mathbb{E}\left[\min\bigl(\rho_t A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\bigr)\right] + \beta \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

其中 $\rho_t = \pi_\theta(\text{tok}_t \mid \text{ctx}) / \pi_{\text{old}}(\text{tok}_t \mid \text{ctx})$。

**OneRec V2的Reward设计**：
```python
{
    "pass_at_1":          first_sid_hit_reward,      # {0, 1}
    "format_reward":      think_format_reward,       # {0, 1}
    "partial_hit_reward": partial_hit_reward,         # {0, 1, 10, 100}
    "hit_reward":         hit_reward,                # [0, 1] 连续值
    "duration_reward":    duration_aware_reward,      # V2新增, 观看时长感知
}
```

### 7.4 量化损失（RQ-VAE/PQ）

**使用场景**：语义ID生成阶段的离线训练。

**RQ-VAE损失**：
$$\mathcal{L}_{\text{RQ-VAE}} = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2}_{\text{重构}} + \underbrace{\sum_{l=1}^{L}\|\text{sg}(\mathbf{r}_{l-1}) - \mathbf{e}_{c_l}^{(l)}\|_2^2}_{\text{码本}} + \underbrace{\beta\sum_{l=1}^{L}\|\mathbf{r}_{l-1} - \text{sg}(\mathbf{e}_{c_l}^{(l)})\|_2^2}_{\text{承诺}}$$

**PQ/OPQ损失**：
$$\mathcal{L}_{\text{OPQ}} = \min_{\mathbf{R}, \{C_m\}} \sum_{i=1}^{N} \left\| \mathbf{R}\mathbf{z}_i - \sum_{m=1}^{M}\mathbf{e}_{c_m^{(i)}}^{(m)} \right\|_2^2 \quad \text{s.t.} \quad \mathbf{R}^T\mathbf{R} = \mathbf{I}$$

---

## 8. 总结与展望

### 8.1 架构全景对照

| 论文 | 架构类型 | 使用范式 | 骨干 | 损失函数 | 规模 |
|------|----------|----------|------|----------|------|
| TIGER | Enc-Dec | GRs | T5 | NTP CE | 百万级 |
| OneRec V1 | Enc-Dec+MoE | GRs | 自定义 | NTP + GRPO | 十亿级 |
| OneSearch | Enc-Dec | GRs | 自定义 | NTP + PARS | 十亿级 |
| OneRec V2 | Decoder-Only | GRs | 自定义 | NTP + GRPO | 80亿 |
| PLUM | Decoder-Only | GRs | Gemma/PaLM | CPT + NTP | 十亿级 |
| HSTU | Decoder-Only | GRs | HSTU定制 | NTP | 1.5万亿 |
| RPG | Decoder-Only+MTP | GRs | 自定义 | MTP CE | 十亿级 |
| QARM | DLRM | DLRMs | DNN | BCE + 多任务 | - |
| SIDE | DLRM | DLRMs | DNN | BCE | - |
| DAS | 双塔+CF | DLRMs/GRs | 双塔 | 对比+NTP | - |
| YouTube | 排序DNN | DLRMs | DNN | BCE | 十亿级 |
| COBRA | 级联生成 | GRs | 自定义 | NTP | - |
| UniSID | 端到端 | GRs | Transformer | CE+对比 | - |
| MMQ | MoE量化 | GRs | MoE | NTP | - |

### 8.2 未来方向

1. **Decoder-Only统一化**：更多系统将从Enc-Dec迁移到Decoder-Only（OneRec V2的验证）
2. **LLM原生推荐**：直接复用预训练LLM做推荐（PLUM方向）
3. **推理增强**：在推荐中加入显式推理链（OneRec-Think方向）
4. **Scaling Law**：推荐领域的scaling law探索（HSTU的初步验证）
5. **多模态融合**：MoE等多模态编码方式与生成架构的深度结合（MMQ方向）
6. **端到端优化**：语义ID生成和推荐模型的端到端联合训练（UniSID、DIGER方向）

---

## 参考文献

1. Rajput, S., et al. (2023). Recommender Systems with Generative Retrieval (TIGER). *NeurIPS 2023*. arXiv:2305.05065. https://arxiv.org/abs/2305.05065

2. Singh, A., et al. (2024). Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations. *RecSys 2024*. arXiv:2306.08121. https://arxiv.org/abs/2306.08121

3. OneRec Team, Kuaishou. (2025). OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment. arXiv:2502.18965. https://arxiv.org/abs/2502.18965

4. OneRec V2 Team, Kuaishou. (2025). OneRec-V2 Technical Report. arXiv:2508.20900. https://arxiv.org/abs/2508.20900

5. OneRec-Think Team, Kuaishou & Tsinghua. (2026). OneRec-Think: In-Text Reasoning for Generative Recommendation. *ACL 2026*. arXiv:2510.11639. https://arxiv.org/abs/2510.11639

6. Meta GR Team. (2024). Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (HSTU). arXiv:2402.17152. https://arxiv.org/abs/2402.17152

7. PLUM Team, Google DeepMind & YouTube. (2025). PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations. arXiv:2510.07784. https://arxiv.org/abs/2510.07784

8. RPG Team, Meta. (2025). Recommendation with Parallel semantic ID Generation. *KDD 2025*. arXiv:2506.05781. https://arxiv.org/abs/2506.05781

9. QARM Team, Kuaishou. (2024). QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou. arXiv:2411.11739. https://arxiv.org/abs/2411.11739

10. QARM V2 Team, Kuaishou. (2026). QARM V2: Quantitative Alignment Multi-Modal Recommendation for Reasoning User Sequence Modeling. arXiv:2602.09458. https://arxiv.org/abs/2602.09458

11. Ramasamy, D., et al. (2025). SIDE: Semantic ID Embedding for effective learning from sequences. *AdKDD 2025*. arXiv:2506.16698. https://arxiv.org/abs/2506.16698

12. DAS Team, Kuaishou. (2025). DAS: Dual-Aligned Semantic IDs Empowered Industrial Recommender System. *CIKM 2025*. arXiv:2508.10584. https://arxiv.org/abs/2508.10584

13. OneSearch Team, Kuaishou. (2025). OneSearch: A Preliminary Exploration of the Unified End-to-End Generative Framework for E-commerce Search. arXiv:2509.03236. https://arxiv.org/abs/2509.03236

14. COBRA Team, Baidu. (2025). Unified Generative Recommendations with Cascaded Sparse-Dense Representations. arXiv:2503.02453. https://arxiv.org/abs/2503.02453

15. Jiang, J., et al. (2026). End-to-End Semantic ID Generation for Generative Advertisement Recommendation (UniSID). arXiv:2602.10445. https://arxiv.org/abs/2602.10445

16. MMQ Team. (2025). MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation. *WSDM 2026*. arXiv:2508.15281. https://arxiv.org/abs/2508.15281

17. DIGER Team. (2026). Differentiable Semantic ID for Generative Recommendation. *SIGIR 2026*. arXiv:2601.19711. https://arxiv.org/abs/2601.19711

18. FORGE Team, Alibaba. (2025). FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets. arXiv:2509.20904. https://arxiv.org/abs/2509.20904

19. Hou, Y., et al. (2022). Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders (VQ-Rec). arXiv:2210.12316. https://arxiv.org/abs/2210.12316

20. van den Oord, A., et al. (2017). Neural Discrete Representation Learning (VQ-VAE). *NeurIPS 2017*. arXiv:1711.00937. https://arxiv.org/abs/1711.00937

21. Lee, K., et al. (2022). Autoregressive Image Generation using Residual Quantization. *CVPR 2022*. arXiv:2203.01941. https://arxiv.org/abs/2203.01941
