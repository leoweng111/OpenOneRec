# 生成式推荐：从预训练到后训练的训练管线全景

> 本篇以训练管线为主线，梳理生成式推荐从预训练到后训练的完整方法论。
> 架构选择是预训练的基石，损失函数是每个阶段的核心驱动；后训练则在预训练的基础上做偏好对齐，让模型从"会续写"进化为"懂偏好"。

## 目录

- [1. 开篇概述：生成式推荐的训练管线](#1-开篇概述生成式推荐的训练管线)
  - [1.1 为什么需要多阶段训练](#11-为什么需要多阶段训练)
  - [1.2 训练管线全景图](#12-训练管线全景图)
  - [1.3 各阶段的核心目标与损失函数](#13-各阶段的核心目标与损失函数)
  - [1.4 预训练 vs 后训练的本质区别](#14-预训练-vs-后训练的本质区别)
- [2. 预训练阶段详解](#2-预训练阶段详解)
  - [2.1 架构选择：预训练的基石](#21-架构选择预训练的基石)
  - [2.2 语义ID构造：预训练的前置工序](#22-语义id构造预训练的前置工序)
  - [2.3 预训练数据策略](#23-预训练数据策略)
  - [2.4 预训练损失函数](#24-预训练损失函数)
  - [2.5 预训练分阶段策略：冻结→解冻→SFT](#25-预训练分阶段策略冻结解冻sft)
  - [2.6 预训练样本组织方式](#26-预训练样本组织方式)
- [3. 预训练论文分析](#3-预训练论文分析)
  - [3.1 TIGER (Google, 2023) — Enc-Dec奠基之作](#31-tiger-google-2023--enc-dec奠基之作)
  - [3.2 OneRec V1 (快手, 2025) — Enc-Dec+MoE工业级](#32-onerec-v1-快手-2025--enc-decmoe工业级)
  - [3.3 OneRec V2 (快手, 2025) — Lazy Decoder-Only](#33-onerec-v2-快手-2025--lazy-decoder-only)
  - [3.4 PLUM (Google, 2025) — LLM复用Decoder-Only](#34-plum-google-2025--llm复用decoder-only)
  - [3.5 HSTU (Meta, 2024) — 万亿参数定制Decoder](#35-hstu-meta-2024--万亿参数定制decoder)
  - [3.6 RPG (Meta, 2025) — 并行MTP](#36-rpg-meta-2025--并行mtp)
  - [3.7 OneSearch (快手, 2025) — 电商搜索Enc-Dec](#37-onesearch-快手-2025--电商搜索enc-dec)
  - [3.8 OneRec-Think (快手, 2026) — 推理链增强](#38-onerec-think-快手-2026--推理链增强)
  - [3.9 LLM4DLRMs预训练（QARM, SIDE, DAS等）](#39-llm4dlrms预训练qarm-side-das等)
- [4. 后训练阶段详解](#4-后训练阶段详解)
  - [4.1 后训练的必要性：从"会续写"到"懂偏好"](#41-后训练的必要性从会续写到懂偏好)
  - [4.2 后训练方法分类全景](#42-后训练方法分类全景)
  - [4.3 SFT（监督微调）—— 后训练的桥梁](#43-sft监督微调--后训练的桥梁)
  - [4.4 DPO（直接偏好优化）—— off-policy对齐](#44-dpo直接偏好优化--off-policy对齐)
  - [4.5 PPO/GRPO/GBPO —— on-policy策略梯度](#45-ppogrpgbpo--on-policy策略梯度)
  - [4.6 Reward模型设计](#46-reward模型设计)
  - [4.7 偏好数据构造](#47-偏好数据构造)
  - [4.8 后训练与预训练的衔接](#48-后训练与预训练的衔接)
- [5. 后训练论文分析](#5-后训练论文分析)
  - [5.1 OneRec V1 — Iterative DPO偏好对齐](#51-onerec-v1--iterative-dpo偏好对齐)
  - [5.2 OneRec V2 — GBPO+时长感知Reward](#52-onerec-v2--gbpo时长感知reward)
  - [5.3 OneSearch — PARS自适应奖励](#53-onesearch--pars自适应奖励)
  - [5.4 OneRec-Think — GRPO + Rollout-Beam Reward](#54-onerec-think--grpo--rollout-beam-reward)
  - [5.5 PLUM — 加权SFT（无显式RL）](#55-plum--加权sft无显式rl)
  - [5.6 HSTU — 纯NTP无后训练对齐](#56-hstu--纯ntp无后训练对齐)
  - [5.7 LLM4DLRMs后训练（QARM V2等）](#57-llm4dlrms后训练qarm-v2等)
- [6. 预训练→后训练演进趋势与展望](#6-预训练后训练演进趋势与展望)
- [参考文献](#参考文献)

---

## 1. 开篇概述：生成式推荐的训练管线

### 1.1 为什么需要多阶段训练

生成式推荐（Generative Recommendation, GR）将推荐任务重构为序列生成问题：给定用户历史行为，自回归预测下一个物品的语义ID（Semantic ID）。这一范式与LLM的语言建模天然对齐——"续写下一个token"就是"预测下一个推荐item"。

但**直接把预训练LLM用于推荐，效果远不够好**。原因有三：

```
问题1: 知识断裂
  预训练LLM懂得自然语言，但不知道"SID token代表什么物品"
  → 需要预训练阶段让SID token在语言空间中获得语义锚定

问题2: 目标偏移
  预训练让模型学会"续写序列"，但推荐需要"预测用户真正感兴趣的下一个"
  → "下一个曝光item" ≠ "用户最想看的item"
  → 需要后训练阶段做偏好对齐

问题3: 格式失控
  预训练模型可能生成无效的SID组合、不遵循指令格式
  → 需要SFT阶段让模型学会指令跟随和输出规范化
```

这和LLM的发展历程完全一致：GPT系列经历了 **Pretrain → SFT → RLHF/DPO** 三阶段才变成ChatGPT。生成式推荐走的是同一条路。

### 1.2 训练管线全景图（以快手OneRec V2为例）

```
┌────────────────────────────────────────────────────────────────────────┐
│                 生成式推荐的完整训练管线                                  │
│                                                                        │
│  ┌──────────────────┐                                                  │
│  │  离线前置工序      │                                                  │
│  │  语义ID构造        │  物品→多模态Encoder→量化→SID tokens              │
│  │  (RQ-KMeans/VQ/  │  这是预训练之前的数据准备，不参与LLM训练           │
│  │   OPQ/FSQ等)      │                                                  │
│  └──────────────────┘                                                  │
│           ↓                                                            │
│  ═════════════════════════════════════════════════════════════════════  │
│                     预训练阶段 (Pretrain)                               │
│  ═════════════════════════════════════════════════════════════════════  │
│                                                                        │
│  Stage 1: Itemic-Text Alignment (SID对齐)                              │
│    ├── 冻结LLM骨干，只训SID token embedding                           │
│    ├── 数据: 物品理解 + 用户画像 + 行为序列                             │
│    ├── Loss: NTP CE (全序列)                                            │
│    └── 目标: SID token在语言空间中获得位置                              │
│                                                                        │
│  Stage 2: Full-parameter Co-Pretraining (全参数协同)                   │
│    ├── 解冻全部参数，推荐数据 + 通用文本混合训练                        │
│    ├── Loss: NTP CE (全序列或仅target)                                 │
│    └── 目标: 推荐能力叠加到语言能力上                                   │
│                                                                        │
│  Stage 3: SFT (监督微调)                                               │
│    ├── 全参数，messages格式，指令跟随训练                                │
│    ├── Loss: NTP CE (仅assistant段)                                    │
│    └── 目标: 学会指令跟随 + 输出格式规范化                              │
│                                                                        │
│           ↓                                                            │
│  ═════════════════════════════════════════════════════════════════════  │
│                     后训练阶段 (Post-train / Alignment)                 │
│  ═════════════════════════════════════════════════════════════════════  │
│                                                                        │
│  路径A: Off-policy对齐 (DPO)                                           │
│    ├── 基于预训练模型beam search生成候选                                │
│    ├── Reward模型打分 → 构造偏好对 (preferred vs rejected)             │
│    ├── DPO loss直接优化偏好                                             │
│    └── 代表: OneRec V1, OneSearch                                      │
│                                                                        │
│  路径B: On-policy对齐 (GRPO/PPO/GBPO)                                 │
│    ├── 当前策略模型实时rollout生成候选                                   │
│    ├── Reward评分 → 组内标准化优势估计                                   │
│    ├── PPO-clip / GRPO / GBPO策略梯度优化                              │
│    ├── 可迭代: 多轮on-policy采样持续对齐                                │
│    └── 代表: OneRec V2, OneRec-Think, OpenOneRec开源                   │
│                                                                        │
│           ↓                                                            │
│  部署: Beam Search推理 → SID序列 → 映射回item                          │
└────────────────────────────────────────────────────────────────────────┘
```

### 1.3 各阶段的核心目标与损失函数

| 阶段 | 核心目标 | 主损失函数 | 数据格式          | 代表论文 |
|------|----------|------------|---------------|----------|
| **SID构造** (离线) | 物品→离散token | 量化损失 (RQ-VAE/PQ/FSQ) | 物品（文本或多模态）特征  | TIGER, QARM |
| **Pretrain S1** (对齐) | SID token语义锚定 | NTP CE (全序列) | segments      | OneRec, PLUM |
| **Pretrain S2** (协同) | 推荐能力叠加 | NTP CE (全序列/target) | segments      | OneRec, PLUM |
| **SFT** (指令) | 指令跟随+格式 | NTP CE (仅assistant) | messages      | OneRec, PLUM |
| **DPO** (off-policy) | 偏好对齐 | DPO loss (偏好对 log-ratio) | 偏好对           | OneRec V1, OneSearch |
| **GRPO** (on-policy) | 偏好对齐 | PPO-clip + 组内标准化 | rollout+reward | OneRec V2, OneRec-Think |
| **GBPO** (on-policy) | 偏好对齐 | 动态bound替代clip | rollout+reward | OneRec V2 |

### 1.4 预训练 vs 后训练的本质区别

```
预训练 (Pretrain):
  ├── 学"是什么": SID token A 代表搞笑视频
  ├── 学"怎么续": 看了A之后常看B → 续写B
  ├── 损失驱动: 预测下一个token是否正确 (CE loss)
  ├── 信号来源: 历史数据中的共现统计
  └── 本质: 无监督的语言建模 (self-supervised NTP)

后训练 (Post-train):
  ├── 学"用户真正想要什么": 不是"下一个曝光"，而是"下一个兴趣"
  ├── 学"输出好还是坏": 给偏好排序 (DPO) 或给评分 (GRPO)
  ├── 损失驱动: 偏好信号 (reward) 指导梯度方向
  ├── 信号来源: 用户反馈 (时长、点赞、评论) → Reward模型量化
  └── 本质: 有监督的偏好对齐 (supervised preference alignment)
```

**类比理解**：

```
预训练 ≈ 大量阅读 → 学会"语言怎么写"
           → 每个字都参与学习 (全序列loss)
           → 知道"搞笑"后面常跟"猫咪"

SFT     ≈ 做问答题 → 学会"看到问题怎么回答"
           → 只在"你的回答"上打分 (assistant-only loss)
           → 学会按指令格式输出

后训练   ≈ 考官评分 → 学会"给出更好的回答"
           → 评分来自用户真实反馈 (reward)
           → 不是"随便续写"而是"写出用户真正想看的"
```

---

## 2. 预训练阶段详解

### 2.1 架构选择：预训练的基石

架构决定了预训练的算力分配、训练效率和可扩展性。当前GR领域的架构选择经历了从Encoder-Decoder到Decoder-Only的明确演进。

#### 2.1.1 三大架构类型

```
                    生成式推荐预训练架构
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
    Encoder-Decoder   Decoder-Only   LLM4DLRMs
    (Seq2Seq风格)     (GPT风格)      (判别式+语义ID)
```

#### 2.1.2 Enc-Dec vs Decoder-Only 对预训练的影响

| 维度 | Encoder-Decoder | Decoder-Only |
|------|-----------------|--------------|
| **算力分配** | Encoder占97.66%算力（OneRec V1数据），仅2.34%用于生成 | 100%算力用于序列预测 |
| **训练效率** | 大量计算在Encoder（不产生loss），训练效率低 | 每一步计算都服务于NTP loss |
| **扩展性** | 增大Encoder收益递减，难以遵循scaling law | 遵循power-law scaling (HSTU验证) |
| **LLM复用** | 需基于T5等Enc-Dec LLM，生态较少 | 可直接复用GPT/Llama等Decoder-Only LLM |
| **流式适配** | 需额外context/target分离逻辑 | 天然适配NIO流式样本组织 |
| **推理延迟** | 两次前向（Encoder+Decoder） | 单次前向+KV Cache |

**核心结论**：Decoder-Only是当前GR预训练的主流选择。原因：
- 推荐不是翻译——用户行为和目标SID在**同一语义空间**，不需要Enc-Dec跨空间映射
- 算力应花在NTP loss上，而不是context编码
- 与LLM生态（FlashAttention、KV Cache、FSDP等）天然对齐

#### 2.1.3 架构演进案例

```
OneRec V1 → V2 的教训:
  V1 (Enc-Dec+MoE): Encoder 97.66%算力 → Decoder仅2.34%
                     生成能力受限，扩展性差
  V2 (Lazy Decoder-Only): 去掉Encoder和Cross-Attention
                           总计算量减少94%，100%算力用于生成
                           可扩展至80亿参数
                           L20 GPU推理36ms

HSTU (Meta) 的验证:
  定制Decoder-Only架构 → 1.5万亿参数
  比FlashAttention2 Transformer快5.3×~15.2×
  首次观察到推荐领域的power-law scaling

PLUM (Google) 的路线:
  直接复用预训练Decoder-Only LLM (Gemini家族)
  扩展词表 + CPT → SFT → 部署YouTube Shorts
```

### 2.2 语义ID构造：预训练的前置工序

语义ID（Semantic ID, SID）构造是预训练之前的关键离线步骤。它决定了推荐词表的大小、粒度和语义质量。

#### 2.2.1 主要量化方法

| 方法 | 数学原理 | 代表论文 | 特点 |
|------|----------|----------|------|
| **RQ-VAE** | 残差递进量化 | TIGER, OneRec | 层次化，粗→细 |
| **RQ-KMeans** | 残差KMeans聚类 | OneRec, QARM | 无需VAE，直接聚类 |
| **OPQ** | 优化旋转+产品量化 | RPG, OneSearch | 无序token，可并行 |
| **FSQ** | 有限标量量化 | QARM V2 | 避免码本崩溃 |
| **VQ-Fusion** | 多任务VQ融合 | SIDE | 融合多源embedding |
| **Gumbel-Softmax** | 可微分量化 | DIGER | 端到端可训 |

#### 2.2.2 量化损失

**RQ-VAE损失**：

$$\mathcal{L}_{\mathrm{RQVAE}} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 + \sum_{l=1}^{L}\|\mathrm{sg}(\mathbf{r}_{l-1}) - \mathbf{e}_{c_l}^{(l)}\|_2^2 + \beta \sum_{l=1}^{L}\|\mathbf{r}_{l-1} - \mathrm{sg}(\mathbf{e}_{c_l}^{(l)})\|_2^2$$

- 第一项：**重构损失** $\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2$
- 第二项：**码本损失** $\sum_{l}\|\mathrm{sg}(\mathbf{r}_{l-1}) - \mathbf{e}_{c_l}^{(l)}\|_2^2$
- 第三项：**承诺损失** $\beta\sum_{l}\|\mathbf{r}_{l-1} - \mathrm{sg}(\mathbf{e}_{c_l}^{(l)})\|_2^2$

**OPQ损失**：

$$\min_{\mathbf{R}, \{C_m\}} \sum_{i=1}^{N} \left\| \mathbf{R}\mathbf{z}_i - \sum_{m=1}^{M} \mathbf{e}_{c_m^{(i)}}^{(m)} \right\|_2^2$$

约束条件：$\mathbf{R}^T \mathbf{R} = \mathbf{I}$（正交矩阵）

#### 2.2.3 SID构造与预训练的关系

一般是做双阶段训练的。SID构造是**独立于预训练的离线工序**。量化码本（codebook）不参与LLM预训练——LLM训练的是SID token的embedding，而非码本本身。两者的embedding空间完全独立，因此SID一般不直接
参与后续的预训练，需要先做对齐。（参见notes/onerec_questions.md §3的详细分析）。

但DIGER提出了一种新思路：**端到端可微分SID**，通过Gumbel-Softmax让推荐梯度直接回传到量化层，联合优化SID构造和推荐目标。

### 2.3 预训练数据策略

#### 2.3.1 数据类型

对于推荐大模型，预训练不仅仅是让LLM学会next item prediction，还要让LLM学会**SID token的语义理解**等等内容。因此，一般需要多种数据类型来满足不同的预训练目标。以快手OneRec为例，预训练数据分为四类：

| 数据类型 | 作用 | 格式 | 代表 |
|----------|------|------|------|
| **行为序列数据** | 学会"SID共现规律+序列续写" | segments | OneRec video_rec |
| **物品理解数据** | 学会"SID=物品语义描述" | segments | OneRec item_understand |
| **用户画像数据** | 学会"SID=用户偏好描述" | segments | OneRec user_profile |
| **通用文本数据** | 防止灾难性遗忘 | segments | OneRec general_text |
| **对话指令数据** | 学会指令跟随+格式 | messages | OneRec SFT tasks |

**为什么需要混合通用文本？**

```
如果只用推荐数据:
  → LLM逐渐"忘记"通用语言能力 (catastrophic forgetting)
  → 无法理解用户的自然语言描述
  → 模型变成"只会续写SID"的统计机器

混合通用文本后:
  → LLM同时学习推荐模式和通用语言
  → 保持"通才+专才"的dual capability
  → PLUM的CPT数据: 行为序列+物品元数据(两者50/50) + 通用文本(比例未明确)
```

#### 2.3.2 PLUM的CPT(Continued Pre-training)数据策略（Google YouTube部署经验）

```
PLUM CPT数据构成（基于论文原文）:

  领域数据（50% + 50% 混合）:
  ├── 用户行为序列 (~50%)
  │   格式: <sid_1> <channel_name> <watch_ratio> <watch_time> <hours_since_final_watch> <sid_2> ...
  │   ← 包含SID序列 + 频道名 + 观看比例/时长/时间间隔等行为特征
  │
  └── 物品元数据 (~50%)
      ├── SID + title:  "Video <sid> has title (en): <video_title>"
      ├── SID + topics: "The topics in video <sid> are: <topics>"
      ├── SID + ASR captions / description / channel name
      └── 合成生成数据 (synthetically generated data)
  
  通用文本: 论文提到CPT混合了"general-domain text data"以对齐SID模态与LLM已有知识，
           防止灾难性遗忘，但具体比例未在论文中明确给出。

  训练量: ~260 billion tokens, batch_size=16, 1M steps
  基座LLM: Gemini家族 (Decoder-Only)
  效果: 每天~9亿样本即可训练（远少于传统LEM数十亿）
```

**注意**：通用文本确实被包含在CPT混合数据中（用于保持LLM的通用语言能力），但其具体比例论文未明确给出。

#### 2.3.3 OneRec的混合数据策略（快手开源经验）

```
OneRec Stage 1 数据构成:
  ├── 视频推荐序列 (SID序列续写)
  ├── 物品理解 (SID↔文本语义对齐)
  ├── 用户画像 (SID↔用户偏好)
  └── 通用文本 (数学、代码、推理等)
  
  通过split_data.py合并四类数据，按1000行/shard切分
  训练时WebDataset流式加载 + local shuffle buffer
```

### 2.4 预训练损失函数

#### 2.4.1 Next Token Prediction (NTP) — 预训练的核心损失

所有LLM4GRs的预训练/SFT阶段都使用token级V维softmax多分类交叉熵：

$$\mathcal{L}_{\mathrm{NTP}} = -\frac{1}{|\mathcal{S}|}\sum_{k \in \mathcal{S}} \log p_\theta(\mathrm{tok}_{k+1} \mid \mathrm{tok}_{\leq k})$$

其中 $p_\theta(\cdot) = \mathrm{softmax}(W_{\mathrm{lm\_head}} \, h_k) \in \mathbb{R}^V$，$V$ 是词表大小（OneRec约176k）。

**$\mathcal{S}$ 的选择取决于样本组织方式**：

| 组织方式 | $\mathcal{S}$（算loss的位置）     | loss_mask |
|----------|-----------------------------|-----------|
| Naive Impression | 全序列所有token                  | 全1(除EOS) |
| User-Centric | 用户序列所有token                 | 全1(除EOS) |
| New Impression Only | 仅target item的token          | 仅target段=1 |
| SFT (messages) | 仅assistant段，可参考OpenOneRec做法 | assistant段=1 |

#### 2.4.2 Multi-Token Prediction (MTP) — RPG的并行损失

Meta的RPG不使用逐token自回归，而是所有SID token并行预测：

$$\mathcal{L}_{\mathrm{MTP}} = \sum_{m=1}^{M} \mathcal{L}_{\mathrm{CE}}(c_m, \hat{c}_m)$$

每个token独立预测，互不依赖。推理时通过图约束解码（Graph-Constrained Decoding）避免无效组合。

#### 2.4.3 对比学习损失 — SID对齐阶段的辅助损失

在SID对齐（Stage 1）和DLRM场景中，对比损失用于建立语义关联：

$$\mathcal{L}_{\mathrm{InfoNCE}} = -\log \frac{\exp(\mathrm{sim}(z_i, z_j^+)/\tau)}{\sum_{k=1}^{N}\exp(\mathrm{sim}(z_i, z_k)/\tau)}$$

**DAS的多视角对比损失**：

$$\mathcal{L}_{\mathrm{align}} = \mathcal{L}_{\mathrm{u2i}} + \mathcal{L}_{\mathrm{i2i}} + \mathcal{L}_{\mathrm{co}}$$

**PLUM的SID训练损失**（三部分）：

$$\mathcal{L}_{\mathrm{SID}} = \mathcal{L}_{\mathrm{recon}} + \mathcal{L}_{\mathrm{RQ}} + \mathcal{L}_{\mathrm{contrastive}}$$

### 2.5 预训练分阶段策略：冻结→解冻→SFT

#### 2.5.1 OneRec的三阶段策略

```
Stage 1: Itemic-Text Alignment
  ├── 冻结: Qwen3全部参数 (Transformer层+原始文本embedding)
  ├── 只训: 新增~24,577个SID token embedding
  ├── 数据: 物品理解 + 用户画像 + 行为序列 + 通用文本
  ├── Loss: NTP CE (全序列, loss_mask全1)
  ├── 目标: SID token在语言空间中获得语义锚定
  └── 类比: 先背单词(SID embedding), 语法(LLM)不动

Stage 2: Full-parameter Co-Pretraining  
  ├── 解冻: 全部参数
  ├── 数据: 同Stage 1的混合数据
  ├── Loss: NTP CE (全序列)
  ├── 目标: 推荐能力叠加到语言能力上
  └── 类比: 单词和语法一起练习

Stage 3: SFT (Post-training的桥梁)
  ├── 全参数继续训练
  ├── 数据: messages格式 (8类推荐SFT任务)
  ├── Loss: NTP CE (仅assistant段)
  ├── 目标: 指令跟随 + 输出规范化
  └── 类比: 做问答题, 学会按格式回答
```

#### 2.5.2 PLUM的三阶段策略

```
Stage 1: SID-v2 Item Tokenization (离线工序)
  ├── 多模态内容 → 独立Encoder编码 → 拼接投影 → RQ-VAE量化 → SID tokens
  ├── 融合title/description/ASR/channel等多源信息
  ├── 多分辨率码本: K_l = 2048/2^(l-1), 渐进掩码, 共现对比损失
  └── 独立于LLM的离线工序 — 码本训练完成后冻结

Stage 2: Continued Pre-Training (CPT) ← 关键阶段
  ├── 预训练LLM (Gemini家族) → 扩展词表(加入SID tokens)
  ├── 混合数据: 用户行为(~50%) + 物品元数据(~50%) + 通用文本
  │   (50/50比例是行为与元数据之间; 通用文本确实包含但比例未明确)
  ├── ~260B tokens, 1M steps, batch_size=16
  ├── 目标: 让SID token在LLM语言空间中获得语义锚定
  └── 关键: CPT后模型仍保留in-context few-shot能力

Stage 3: Task-Specific Fine-tuning (SFT)
  ├── Reward-weighted采样训练
  ├── Loss: reward加权NTP CE (详见§4.3.4)
  ├── Beam search推理, hallucination率 < 5%
```

#### 2.5.3 为什么Stage 1要冻结LLM？

```
核心原因: 防止灾难性遗忘 (Catastrophic Forgetting)

如果不冻结:
  SID embedding (随机初始化) → 梯度回传 → 扰动Qwen3已训练好的权重
  → 已学好的语义空间被破坏
  → "猫咪"旁边的概念可能被挤到"汽车"区

冻结后:
  Qwen3定义"语言空间的坐标系"
  SID token只需学会在这个坐标系里找到正确位置
  → 训练稳定，收敛快

先冻结训SID (Stage 1) → 再全参数联合优化 (Stage 2)
→ 先让SID"入门"，再让全部参数协同适配
```

### 2.6 预训练样本组织方式

样本组织方式决定了loss_mask的标记策略和训练效率。详见 `样本组织方式`文档，此处简述：

| 组织方式 | 样本粒度 | loss范围 | 代表 | 适用场景 |
|----------|----------|----------|------|----------|
| **User-Centric** | 一用户一条 | 全序列 | OneRec开源pretrain | 静态数据,科研复现 |
| **New Impression Only** | 一曝光一条 | 仅target | OneRec-V2线上 | 流式训练,工业部署 |
| **Naive Impression** | 一曝光一条 | 全序列 | 早期TIGER | 已淘汰 |

---

## 3. 预训练论文分析

### 3.1 TIGER (Google, 2023) — Enc-Dec奠基之作

- **论文**：[Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065) (NeurIPS 2023)
- **架构**：Encoder-Decoder (基于T5)

**预训练方法论**：

```
TIGER预训练管线:
  1. SID构造: RQ-VAE量化 (离线)
     物品 → 内容embedding → RQ-VAE → 语义ID (c₁, c₂, c₃)
  
  2. 模型训练: Seq2Seq NTP (在线)
     Encoder: 输入用户历史SID序列 [SID₁, SID₂, SID₃]
     → Bi-directional Self-Attention → 上下文表示 H_enc
     Decoder: 自回归生成目标SID
     → Causal SA + Cross-SA → LM Head → 预测c₁,c₂,c₃
  
  3. 推理: 层次感知Beam Search
     第1层(粗粒度)beam较小,逐层扩展
```

**损失函数**：标准Seq2Seq交叉熵

$$\mathcal{L} = -\sum_{t=1}^{T} \log p_\theta(y_t \mid y_{\lt t}, \mathbf{H}_{\mathrm{enc}})$$

**局限**：
- Enc-Dec架构，Encoder消耗大量算力但不产生loss
- Naive Impression样本组织，存在冗余loss
- 规模较小（百万级），未验证scaling law

### 3.2 OneRec V1 (快手, 2025) — Enc-Dec+MoE工业级

- **论文**：[OneRec: Unifying Retrieve and Rank](https://arxiv.org/abs/2502.18965)
- **架构**：Encoder-Decoder + MoE

**预训练方法论**：

```
OneRec V1预训练:
  1. SID构造: RQ-KMeans (多级别衡量化)
  2. Session-wise生成: 不是逐item预测,而是生成5-10个item的完整session
     → 更贴近真实推荐场景(一次推荐多个视频)
  3. MoE扩展: 24个专家,每次激活2个 → 1B总参数,13%激活率
     → 算力效率高
  4. NTP CE loss on all semantic ID positions within target sessions
```

**核心问题**：Encoder占97.66%算力，Decoder仅2.34%。这是V2升级的根本动因。

**部署效果**：快手主场景观看时长+1.6%。

### 3.3 OneRec V2 (快手, 2025) — Lazy Decoder-Only

- **论文**：[OneRec-V2 Technical Report](https://arxiv.org/abs/2508.20900)
- **架构**：Lazy Decoder-Only

**预训练方法论**：

```
OneRec V2预训练:
  1. 去掉Encoder和Cross-Attention
     → 所有计算在单个Decoder中完成
     → 总计算量减少94%, 训练资源减少90%
  
  2. New Impression Only样本组织
     → 每次曝光独立一条样本, loss仅在target item上
     → 消除冗余loss和时序泄漏
  
  3. 流式训练
     → 曝光一条来一条训, 模型持续演进
     → 与线上Kafka日志流对接
```

**损失函数**：

$$\mathcal{L}_{\mathrm{NIO}} = \frac{1}{B \cdot L_{\mathrm{target}}} \sum_{b=1}^{B} \sum_{k \in \mathrm{target}(b)} -\log p_\theta(\mathrm{tok}^b_{k+1} \mid \mathrm{tok}^b_{\leq k})$$

**部署效果**：App Stay Time +0.467%~0.741%。

### 3.4 PLUM (Google, 2025) — LLM复用Decoder-Only

- **论文**：[PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations](https://arxiv.org/abs/2510.07784)
- **架构**：Decoder-Only (复用Gemini家族LLM)

**预训练方法论**：

```
PLUM三阶段预训练:
  
  Stage 1: SID-v2 Item Tokenization (离线工序)
    ├── 多模态内容 → 各模态独立Encoder → 拼接投影 → 统一特征z
    ├── 融合title/description/ASR/channel等多源信息
    ├── RQ-VAE量化, 多分辨率码本 K_l = 2048/2^(l-1)
    ├── 渐进掩码(Progressive Masking)强化层次语义
    ├── 共现对比损失(注入协同过滤信号)
    └── 独立于LLM的离线工序, 码本训练完成后冻结
  
  Stage 2: Continued Pre-Training (CPT) ← 关键阶段
    ├── 预训练LLM (Gemini家族) → 扩展词表(加入SID tokens)
    ├── 混合数据:
    │   ├── 用户行为序列 (~50%): SID序列+频道名+观看特征
    │   ├── 物品元数据 (~50%): SID+title, SID+topics, SID+ASR/desc/channel
    │   └── 通用文本: 包含在CPT中, 用于对齐SID模态与LLM已有知识
    │       (论文原文确认包含通用文本, 但具体比例未明确给出)
    ├── 训练量: ~260B tokens, 1M steps, batch_size=16
    ├── 目标: 让SID token在LLM语言空间中获得语义锚定
    └── 关键: CPT后模型仍保留in-context few-shot能力
  
  Stage 3: Task-Specific SFT
    ├── Reward-weighted采样训练
    ├── Loss: reward加权NTP CE (详见§4.3.4)
    ├── Beam search推理 (beam_size=16), hallucination < 5%
```

**关键创新**：
- **SID-v2**：融合多模态内容的改进版语义ID（多分辨率码本 + 渐进掩码 + 共现对比损失注入协同信号）
- **CPT混合数据**：领域数据（用户行为+物品元数据）+通用文本混合预训练，弥合domain gap
- **高样本效率**：900M MoE模型，每天仅需数亿样本训练（远少于传统LEM数十亿）

**部署**：YouTube Shorts, Panel CTR +4.96%。

### 3.5 HSTU (Meta, 2024) — 万亿参数定制Decoder

- **论文**：[Actions Speak Louder than Words](https://arxiv.org/abs/2402.17152) (HSTU)
- **架构**：定制Decoder-Only (HSTU = Hierarchical Sequential Transduction Units)

**预训练方法论**：

```
HSTU预训练:
  1. 定制架构替代标准Transformer
     → Pointwise Aggregated Attention (替代softmax normalization)
     → 去除冗余LayerNorm和FFN层
     → 比FlashAttention2快5.3×~15.2×
  
  2. Generative Training
     → Session级训练，不是impression级
     → 训练样本在session端点产出，采样比例∝1/nᵢ
     → amortizing encoder costs across multiple targets
  
  3. Stochastic Length (SL)
     → 算法性增加用户历史序列稀疏度
     → 复杂度从O(N²d)降至O(N^αd), α∈(1,2]
  
  4. 无显式SFT/RL后训练阶段
     → 纯NTP pretrain → 直接部署
     → 可能依赖Meta内部的后续优化管线（论文未公开）
```

**Scaling Law**：首次在推荐领域观察到power-law scaling（类似GPT-3/LLaMA-2），模型质量随训练计算量幂律增长。

**规模与效果**：1.5万亿参数，线上A/B测试指标提升12.4%。

### 3.6 RPG (Meta, 2025) — 并行MTP

- **论文**：[Recommendation with Parallel semantic ID Generation](https://arxiv.org/abs/2506.05781) (KDD 2025)
- **架构**：Decoder-Only + Multi-Token Prediction

**预训练方法论**：

```
RPG预训练:
  1. SID构造: OPQ量化 (不是RQ, 产出无序token序列)
     → 每个token独立codebook → 信息均衡分布
  
  2. Multi-Token Prediction (MTP)训练
     → 所有SID token并行预测(不是逐token自回归)
     → Loss: 各token位置独立CE loss之和 (详见§2.4.2)
     → 每个token有独立的projection head
  
  3. 推理: Graph-Constrained Decoding
     → 构建合法SID组合的图结构
     → 沿图搜索,避免无效组合
     → 单步生成 vs 多步自回归 → 更快推理
```

**效果**：NDCG@10平均提升12.6%。

### 3.7 OneSearch (快手, 2025) — 电商搜索Enc-Dec

- **论文**：[OneSearch: End-to-End Generative Framework for E-commerce Search](https://arxiv.org/abs/2509.03236)
- **架构**：Encoder-Decoder

**预训练方法论**：

```
OneSearch三阶段SFT:
  
  Stage 1: Semantic Content Alignment
    → 学习query/item文本 ↔ SID的双向映射
    → 附加category预测任务保持相关性约束
  
  Stage 2: Co-occurrence Synchronization
    → query和item之间的互预测(无用户上下文)
    → 建立内在语义关联
    → 使用大规模交互数据
  
  Stage 3: User Personalization Modeling
    → 整合用户信息: user_id + query + SID + 行为序列
    → 滑动窗口数据增强,适应变化兴趣
  
  SID构造: RQ-OPQ混合语义ID
  Model FLOPs Utilization: 3.26% → 27.32%
```

### 3.8 OneRec-Think (快手, 2026) — 推理链增强

- **论文**：[OneRec-Think: In-Text Reasoning for Generative Recommendation](https://arxiv.org/abs/2510.11639) (ACL 2026)
- **架构**：Decoder-Only + Think-Ahead推理

**预训练方法论**：

```
OneRec-Think预训练:
  
  Stage 1: Itemic Alignment via Multi-Task Pre-training
    ├── 四类任务混合:
    │   • interleaved user persona grounding (用户画像嵌入)
    │   • sequential preference modeling (偏好序列建模)
    │   • itemic dense captioning (物品密集描述)
    │   • general language modeling (通用语言建模)
    ├── 两步策略:
    │   Step a: Token Warm-up — 冻结LLM骨干,只训SID embedding
    │   Step b: Multi-Task Integration — 全参数联合优化,保持语言能力
  
  Stage 2: Reasoning Activation (SFT)
    ├── Bootstrapping: 从精简用户上下文提取推理轨迹
    │   → 用语义相似度检索top-k相关items作为推理依据
    ├── Noisy Sequence Learning: 从原始行为数据生成rationale
    │   → 优化rationale token prediction + target item generation
  
  Stage 3: Reasoning Enhancement (GRPO, 见§5.4)
```

**部署**：Kuaishou平台，APP Stay Time +0.159%。

### 3.9 LLM4DLRMs预训练（QARM, SIDE, DAS等）

LLM4DLRMs（判别式推荐+语义ID）的预训练不涉及LLM自回归训练，而是训练传统DLRM模型，将语义ID作为输入特征替换随机Item ID。

| 论文 | 预训练方式 | 损失函数 | 关键点 |
|------|------------|----------|--------|
| **QARM** (快手) | SID构造(Res-KMeans) → DLRM训练 | BCE + 多任务 | SID embedding替换随机ID |
| **QARM V2** (快手) | LLM推理增强 → Res-KmeansFSQ → DLRM | BCE + 多任务 | 推理能力注入SID |
| **SIDE** (Meta) | VQ-Fusion → DLRM训练 | BCE | 无参数SID→embedding转换 |
| **DAS** (快手) | 双量化+CF去偏+对比对齐 | InfoNCE + CE | 多视角对比学习 |
| **YouTube SIDs** | RQ-VAE → SentencePiece → 排序DNN | BCE | 十亿级排序系统 |

**与LLM4GRs预训练的本质区别**：DLRMs预训练是**判别式训练**（BCE/sigmoid），不涉及NTP或自回归生成。语义ID只是输入特征，不是生成目标。

---

## 4. 后训练阶段详解

### 4.1 后训练的必要性：从"会续写"到"懂偏好"

预训练让模型学会了"续写SID序列"——给定历史，预测下一个token。但这有几个根本问题：

```
问题1: 预测目标偏移
  预训练的NTP目标: 预测"下一个出现的token"
  真实推荐目标:    预测"用户真正感兴趣的下一个item"
  
  "下一个曝光" ≠ "下一个兴趣"
  → 用户被曝光了A但可能只是随便看了50%
  → 模型却在学"看完A之后下一个曝光B"

问题2: 输出质量不可控
  预训练模型可能:
    → 生成无效的SID组合 (beam search的幻觉问题)
    → 不遵循指令格式 (缺少指令跟随能力)
    → 无法根据不同任务切换策略

问题3: 推荐的多目标性
  真实推荐需要同时优化:
    → 观看时长 (duration)
    → 点赞率 (like rate)
    → 评论率 (comment rate)
    → 关注率 (follow rate)
  
  单纯NTP无法建模这些多维偏好
  → 需要reward模型量化用户真实反馈
```

**一句话**：预训练解决"模型懂不懂内容与序列规律"，后训练解决"模型会不会按用户偏好做推荐"。

### 4.2 后训练方法分类全景

后训练方法可以分为三大类，核心区别在于**偏好信号的来源和优化方式**：

```
                    后训练方法分类
                         │
          ┌──────────────┼──────────────┐
          ↓              ↓              ↓
      SFT (监督微调)   Off-policy对齐   On-policy对齐
      (指令格式化)     (DPO系)          (PPO/GRPO/GBPO系)
          
  偏好信号来源:
    SFT:     人工标注/模板 → ground truth assistant回答
    DPO:     预训练模型beam search → Reward模型打分 → 偏好对
    GRPO:    当前策略模型rollout → Reward评分 → 组内标准化优势
    
  优化方式:
    SFT:     NTP CE (仅assistant段)
    DPO:     偏好分类 loss (preferred vs rejected)
    GRPO:    策略梯度 + KL正则
    
  是否需要Reward模型:
    SFT:     不需要
    DPO:     需要(用于构造偏好对), 但loss中不显式使用
    GRPO:    需要(用于rollout评分), loss中显式使用reward值
    
  是否需要on-policy采样:
    SFT:     不需要(静态数据)
    DPO:     不需要(off-policy, 用预训练模型生成候选)
    GRPO:    需要(on-policy, 用当前策略模型生成候选)
```

### 4.3 SFT（监督微调）—— 后训练的桥梁

SFT是预训练到偏好对齐之间的桥梁阶段。它不属于偏好对齐，但为偏好对齐提供了必要的基础。
SFT 只教模型“像训练集一样说话”（极大似然估计 Cross-Entropy），缺乏统筹全局商业目标（点击率、停留时长）和探索未知空间（优中选优）的能力。

后训练（RLHF/RL）的精髓：
  1. 引入全局奖励：打破自回归生成中 token 局部的限制，直接为生成的整个序列（或长列表）赋予业务Reward。
  2. 避免幻觉（Hallucination）：生成式推荐最大的问题是生成库里没有的ID或者用户极度反感的商品，RL可以通过负向惩罚严格控制。
  3. 从隐式反馈（Logs）到对齐：使用 DPO/GBPO 这样的算法，直接基于离线点击/未点击日志，跳过复杂的奖励模型（Reward Model）训练，实现端到端的偏好对齐。

以OneRecV2为例，SFT阶段的训练目标是**让模型学会按指令回答**，而不是单纯续写SID序列：
#### 4.3.1 SFT做什么

```
SFT的核心改变:
  预训练: segments格式, 全序列loss → 模型学会"续写"
  SFT:    messages格式, assistant-only loss → 模型学会"按指令回答"
  
  数据格式变化:
    预训练: {"segments": [{"type": "text", "text": "SID序列..."}]}
    SFT:   {"messages": [
              {"role": "system",    "content": "你是推荐助手..."},
              {"role": "user",      "content": "根据历史推荐..."},
              {"role": "assistant", "content": "<|sid_begin|>...<|sid_end|>"}
            ]}
```

#### 4.3.2 SFT的loss机制

```
apply_chat_template展平:
  messages → 连续文本: "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n...<|im_end|>\n"

_get_assistant_mask:
  扫描token序列 → 找到 <|im_start|>assistant\n 到 <|im_end|>\n 的区间
  → 仅assistant段置1, 其余置0

最终loss_mask:
  [0,...] [0,...] [1,...] [0]    ← system/user不算loss, assistant算loss
```

#### 4.3.3 SFT的训练任务

OneRec开源的SFT包含8类推荐任务：

| 任务 | 指令 | 学会的能力 |
|------|------|------------|
| `video_rec` | 预测下一个视频 | 基础推荐生成 |
| `label_cond_rec` | 推荐可能点赞/长看的内容 | 条件约束推荐 |
| `interactive_rec` | 基于查询词推荐 | 查询理解+匹配 |
| `label_pred` | 判断是否会发生某行为 | 判别式预测 |
| `rec_reason` | 给出推荐理由 | 推理链+解释生成 |
| `ad_rec` | 广告推荐 | 广告场景 |
| `product_rec` | 商品推荐 | 电商场景 |
| `item_understand` | 物品描述理解 | 语义理解 |

#### 4.3.4 PLUM的Reward-Weighted SFT

PLUM的SFT不使用显式RL/DPO，而是用**加权SFT**替代：

$$\mathcal{L}_{\mathrm{SFT}} = -\sum r(u, v_c) \cdot \log P(\mathrm{sid}_t \mid \mathrm{ctx})$$

- $r(u, v_c)$ 是手工设计的reward信号（$u$ = user, $v_c$ = clicked item）
- 训练样本按reward值采样，但loss计算时权重相等
- 效果接近简单RL，但训练更稳定

### 4.4 DPO（直接偏好优化）—— off-policy对齐
详细的强化学习后训练笔记：https://trip.larkenterprise.com/docx/CLHpdWvVKo4QRIxHb2HcLkCVnUc
#### 4.4.1 DPO原理

DPO (Direct Preference Optimization) 由Rafailov等人于2023年提出，核心思想是**绕过显式reward模型，直接从偏好对优化策略**。

$$\mathcal{L}_{\mathrm{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)}\right)\right)\right]$$

其中 $y_w$ 是preferred（高分）样本，$y_l$ 是rejected（低分）样本，$\pi_{\mathrm{ref}}$ 是参考策略（通常是SFT后的模型），$\sigma$ 是sigmoid函数，$\beta$ 是温度参数（控制偏好对齐强度）。

**DPO的本质**：把RL的reward最大化问题，通过Bradley-Terry偏好模型的重参数化，转化为一个**分类loss**——只需判断"preferred样本的log-ratio应该高于rejected样本"。$\sigma$ 将log-ratio差映射到0~1区间，使loss成为标准的二元交叉熵形式。

#### 4.4.2 DPO在推荐中的特殊挑战

**挑战：推荐场景缺乏自然偏好对**

```
LLM场景:
  同一个问题 → 可以让模型生成多个回答
  → 人类标注员选出preferred vs rejected
  → 偏好对天然存在

推荐场景:
  一次推荐请求 → 只有一次展示机会
  → 无法同时获得"用户喜欢"和"用户不喜欢"的样本
  → 偏好对需要额外构造
```

**解决方案**：用Reward模型+Beam Search构造偏好对

```
OneRec V1的DPO偏好数据构造:
  1. 用SFT后的模型对每个用户做beam search (128条候选)
  2. 用Reward模型对128条候选打分
  3. 取最高分的作为preferred (y_w)
  4. 取最低分的作为rejected (y_l)
  5. 组成偏好对 → DPO训练
```

**关于Reward模型与DPO的关系**（基于OneRec V1论文原文）：

OneRec V1确实使用了Reward模型，但**Reward模型仅用于构造偏好对数据，不出现在DPO的loss函数中**。具体关系如下：

```
Reward模型的角色: 数据构造工具 (data construction tool)
  ├── 离线预训练Reward模型: 四任务头 (观看时长/观看概率/关注概率/点赞概率)
  ├── 对128条beam search候选打分 → 选出preferred和rejected
  └── 偏好对构造完成后, Reward模型不再参与DPO训练

DPO loss的角色: 策略优化 (policy optimization)
  ├── 输入: 偏好对 (y_w, y_l) + 参考模型 M_ref (上一轮迭代快照)
  ├── Loss: L_DPO = -log σ(β·log[M_new(y_w)/M_ref(y_w)] - β·log[M_new(y_l)/M_ref(y_l)])
  ├── Loss中只有策略模型和参考模型的log概率比, 无reward项
  └── 这符合DPO原始论文(Rafailov et al., 2023)的设计

为什么需要Reward模型? (推荐 vs LLM的关键差异)
  ├── LLM场景: 人工标注员可直接判断preferred vs rejected → 不需要Reward模型
  ├── 推荐场景: 无法让用户同时评价多组推荐结果 → 偏好对不存在
  └── Reward模型充当"代理标注员": 模拟用户判断, 从128条候选中识别最好/最差
```

**总结**：OneRec V1的DPO本质仍是DPO（loss中无显式reward），Reward模型只是解决推荐场景偏好数据缺乏的工程手段。这与GRPO等方法在loss中显式使用reward值有本质区别。

#### 4.4.3 DPO vs PPO的对比

| 维度 | DPO                                    | PPO/GRPO |
|------|----------------------------------------|----------|
| **采样方式** | off-policy (预训练模型生成)                   | on-policy (当前策略生成) |
| **是否需要Reward模型** | 构造偏好对需要（如果已有明确业务含义的偏好对则不需要）, loss中不显式用 | 需要(评分), loss中显式用 |
| **是否需要Critic** | 不需要                                    | PPO需要, GRPO不需要 |
| **模型数量** | 2个(策略+参考)                              | PPO:4个, GRPO:2个 |
| **训练稳定性** | 较好(分类loss)                             | 可能不稳定(策略梯度) |
| **偏好信号粒度** | 二值(preferred/rejected)                 | 连续(reward值) |
| **迭代能力** | 可迭代DPO (多轮off-policy)                  | 天然支持迭代on-policy |

### 4.5 PPO/GRPO/GBPO —— on-policy策略梯度

#### 4.5.1 PPO (Proximal Policy Optimization)

PPO是经典的on-policy策略梯度方法，通过clip限制策略更新幅度防止训练崩溃：

$$\mathcal{L}_{\mathrm{PPO}} = -\mathbb{E}\left[\min\bigl(\rho_t A_t, \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \, A_t\bigr)\right] + \beta \, \mathrm{KL}(\pi_\theta \| \pi_{\mathrm{ref}})$$

其中 $\rho_t = \pi_\theta(\mathrm{tok}_t \mid \mathrm{ctx}) / \pi_{\mathrm{old}}(\mathrm{tok}_t \mid \mathrm{ctx})$ 是新旧策略概率比。

**PPO的缺点**：
- 需要4个模型（策略、参考、reward、critic）→ 内存开销大
- clip操作丢弃了部分样本的梯度 → 信息利用不充分
- 对推荐场景的负样本处理不够好

#### 4.5.2 GRPO (Group Relative Policy Optimization)

GRPO由DeepSeek于2025年1月提出，核心改进是**去掉critic网络，用组内标准化替代**：

$$A_i = \frac{r_i - \operatorname{mean}(r_1,\dots,r_G)}{\operatorname{std}(r_1,\dots,r_G) + \varepsilon}$$

- $G$ 是一个group中的rollout数量（如32条候选）
- 不需要单独的critic网络来估计baseline
- 仍然使用PPO-clip的策略梯度 + KL正则

**GRPO的优势**：
- 只需2个模型（策略+参考），无需critic → 内存减半
- 组内标准化自然处理了reward尺度问题
- 特别适合推理型模型（DeepSeek-R1的验证）

**快手OpenOneRec开源的GRPO实现**（`verl_rl/recipe/onerec/`）：

```
GRPO rollout流程:
  1. 当前actor模型生成32条候选SID (beam_size=32)
  2. Reward模型评分:
     ├── pass_at_1:          {0, 1}    # 第一条SID是否命中
     ├── format_reward:      {0, 1}    #  thinking格式是否正确
     ├── partial_hit_reward: {0,1,10,100}  # 三层SID匹配加权
     ├── hit_reward:         [0, 1]    # 命中集/预测集连续值
     └── duration_reward:    连续值    # V2新增，观看时长感知
  3. 组内标准化 → 优势估计 A_i (见§4.5.2公式)
  4. PPO-clip策略梯度优化 + KL正则
```

#### 4.5.3 GBPO (Gradient-Bounded Policy Optimization) — OneRec V2的创新

OneRec V2提出了GBPO，对GRPO/PPO做了两项关键改进：

```
改进1: 去除clip操作
  PPO/GRPO: clip(ρ_t, 1-ε, 1+ε) → 丢弃超出范围的梯度
  GBPO:     不clip → 保留所有样本的梯度
  → Full Sample Utilization: 鼓励模型做更多样化的探索

改进2: 对πθold施加动态bound
  传统PPO: 固定clip范围 (1-0.2, 1+0.2)
  GBPO:    动态bound on πθold → 自适应裁剪比例
  → 对负样本用BCE loss的稳定性来bound梯度
  → 解决"负样本概率越小越难抑制"的问题
```

**GBPO的核心理念**：PPO的clip是为了防止策略更新过大，但它同时丢弃了有用梯度。GBPO通过动态bound替代硬clip，既保持稳定性又不浪费梯度信息。

GBPO、ECPO本质上都是针对PPO的改进，主要是提升了训练稳定性。
#### 4.5.4 On-policy vs Off-policy的本质区别

```
Off-policy (DPO):
  ├── 用预训练/SFT模型(off-policy)生成候选
  ├── 偏好对一旦构造就固定,不随训练更新
  ├── 偏好信号是二值: preferred > rejected
  └── 优点: 训练稳定,实现简单
  └── 缺点: 偏好数据可能过时,无法感知策略变化

On-policy (GRPO/GBPO):
  ├── 用当前策略模型(on-policy)生成候选
  ├── 每轮rollout都是最新策略的输出
  ├── reward信号是连续值: 0~1或多维
  ├── 组内标准化: 当前策略的相对好坏
  └── 优点: 偏好信号始终与当前策略匹配
  └── 缺点: 需要频繁rollout,训练成本高
  └── 可以迭代: 多轮on-policy持续对齐 (OneRec V2验证)
```

**OneRec V2的实验结论**：On-policy采样（含25%OneRec自生成流量）比Off-policy效果显著提升，说明**self-improvement through policy optimization**是可行的。

### 4.6 Reward模型设计

Reward模型是后训练的核心组件，它将用户多维反馈量化为标量信号。

#### 4.6.1 OneRec V1的多任务Reward

```
OneRec V1 Reward模型:
  ├── 输入: target-aware item representations (e_i = v_i ⊙ u, 元素级乘积)
  ├── 自attention捕获session内item间依赖
  ├── 四个任务头 (多任务塔):
  │   • session观看时长 (swt)
  │   • 观看概率 (vtr)
  │   • 关注概率 (wtr)
  │   • 点赞概率 (ltr)
  ├── 训练: BCE loss, 每个任务独立预测
  ├── 多任务联合训练 → 综合reward分数
  └── 用于: 仅用于DPO偏好对构造 (128条候选 → 选preferred/rejected)
       └── 注意: Reward模型不出现在DPO loss中
           它是"代理标注员", 解决推荐场景缺乏自然偏好对的问题
           偏好对构造完成后, DPO loss仅使用策略模型的log概率比
```

#### 4.6.2 OneRec V2的时长感知Reward

```
OneRec V2 Reward设计:
  1. Duration-Aware Reward Shaping
     → 按视频时长log分桶 → 组内观看时长percentile排名
     → 长视频和短视频的reward可比
  
  2. 优势值分配:
     +1: 高互动视频 (top 25% quantile观看时长)
     -1: 显式负反馈 (dislike)
      0: 其他
  
  3. 真实用户反馈替代纯Reward模型
     → V1依赖reward模型 → 存在reward hacking风险
     → V2直接用用户真实行为信号 → 更可靠
```

#### 4.6.3 OneSearch的PARS (Preference-Aware Reward System)

```
PARS奖励设计:
  1. 六级分层奖励:
     λ = [2.0, 1.5, 1.0, 0.5, 0.2, 0.0]
     (购买 > 加购 > 点击 > 浏览 > 随机 > 无)
  
  2. 动态权重调整:
     CTR_i = log(Cnt_clk + 10) / Cnt_T
     CVR_i = log(Cnt_order + 10) / log(Cnt_clk + 10)
     → 按品类/场景动态调整λ权重
  
  3. 三塔Reward架构:
     → CTR塔 + CVR塔 + CTCVR塔
     → 相关性评分放大10×权重 (确保query-item约束)
  
  4. List-wise DPO:
     → Reward模型reranking → 识别偏好shift
     → 生成正/负样本对 → DPO继续对齐
```

#### 4.6.4 OneRec-Think的Rollout-Beam Reward

```
OneRec-Think Reward:
  → 在constrained beam search内评估
  → 解决传统verification reward的稀疏性问题
  → Beam内多候选同时评估 → 更丰富的偏好信号
```

### 4.7 偏好数据构造

#### 4.7.1 DPO偏好对的构造方式

```
OneRec V1:
  1. SFT模型 → beam search (128条候选/session)
  2. Reward模型打分 → 排序
  3. 取最高分=preferred, 最低分=rejected
  4. 1%训练数据做DPO, 99%仍用NTP
     → hybrid approach: 4%性能提升, GPU需求减5×

OneSearch (PARS):
  1. Reward模型reranking → 识别偏好shift
  2. 高分=preferred, 低分=rejected
  3. List-wise DPO训练
```

#### 4.7.2 GRPO rollout采样的构造方式

```
OneRec V2 / OpenOneRec:
  1. 当前actor模型 → beam search (32条候选)
  2. Reward评分 (多维: pass_at_1, hit_reward, partial_hit等)
  3. 组内标准化 → 优势估计 A_i
  4. PPO-clip / GBPO策略梯度优化
  
  关键: rollout来自当前策略(on-policy),不是预训练模型(off-policy)
```

#### 4.7.3 一个关键对比

| 偏好数据来源 | DPO (off-policy) | GRPO (on-policy) |
|-------------|-------------------|-------------------|
| 生成模型 | SFT后的固定模型 | 当前策略模型(持续更新) |
| 候选数量 | 128条/session | 32条/query |
| 偏好信号 | 二值(preferred/rejected) | 连续(reward值+组内标准化) |
| 数据新鲜度 | 固定,可能过时 | 每轮rollout都是最新策略 |
| 是否可迭代 | 可(Iterative DPO) | 天然可迭代 |

### 4.8 后训练与预训练的衔接

```
预训练 → 后训练的衔接点:

  预训练产出: 一个"会续写SID"的语言模型
    → 模型懂序列规律,但不一定懂用户偏好
    → 可能生成无效SID组合
    → 不遵循指令格式
  
  SFT衔接: 从"续写器"到"指令跟随助手"
    → messages格式训练 → 学会按指令回答
    → assistant-only loss → 聚焦回答质量
    → 多任务SFT → 学会不同推荐场景的策略
  
  DPO/GRPO衔接: 从"指令跟随助手"到"偏好对齐助手"
    → 用户反馈 → reward量化 → 偏好信号
    → DPO: preferred vs rejected → 直接优化偏好排序
    → GRPO: reward + 组内标准化 → 策略梯度优化
    → 多轮迭代 → 模型持续逼近真实用户偏好
```

---

## 5. 后训练论文分析

### 5.1 OneRec V1 — Iterative DPO偏好对齐

- **论文**：[OneRec](https://arxiv.org/abs/2502.18965)

**后训练方法论**：

```
OneRec V1后训练 (Iterative Preference Alignment + DPO):

  1. Reward模型训练
     → 四任务头: 观看时长 + 观看概率 + 关注概率 + 点赞概率
     → target-aware representation + self-attention
  
  2. 偏好数据构造
     → SFT模型 beam search (128条候选)
     → Reward模型打分排序
     → preferred = 最高分, rejected = 最低分
  
  3. Constrained DPO应用
     → 仅1%训练数据做DPO per epoch
     → 99%仍用标准NTP loss
     → Hybrid approach: 4%性能提升, GPU需求减5×
     → 为什么不全用DPO? 计算成本高,且1%已足够
  
  4. Iterative DPO
     → 多轮迭代: 每轮用更新后的模型重新生成候选
     → 偏好数据随策略改善而更新
```

**关于DPO与Reward模型的关系澄清**（基于论文原文）：
- OneRec V1使用了Reward模型，但**Reward模型仅用于构造偏好对**（即上述步骤1-2），不出现在DPO loss函数中
- DPO loss = `-log σ(β·log[M_new(y_w)/M_ref(y_w)] - β·log[M_new(y_l)/M_ref(y_l)])`，只包含策略模型和参考模型的log概率比
- Reward模型充当"代理标注员"：因为推荐场景无法像LLM场景那样由人类标注偏好对，所以用Reward模型模拟用户判断来从128条候选中选出最好和最差的
- 这与GRPO等方法在loss中显式使用reward值有本质区别

**核心创新**：
- 解决推荐场景缺乏自然偏好对的问题（通过Reward模型+Beam Search构造）
- Constrained DPO（1%混合策略）在效果和成本之间取得最优平衡
- Iterative DPO实现策略自改善

### 5.2 OneRec V2 — GBPO+时长感知Reward

- **论文**：[OneRec-V2](https://arxiv.org/abs/2508.20900)

**后训练方法论**：

```
OneRec V2后训练 (GBPO + Duration-Aware Reward):

  1. GBPO (Gradient-Bounded Policy Optimization) ← 核心创新
     → 去除PPO/GRPO的clip操作
     → 对πθold施加动态bound
     → Full Sample Utilization: 保留所有梯度
     → 对负样本用BCE稳定性bound梯度
  
  2. Duration-Aware Reward Shaping
     → 视频时长log分桶 → 组内percentile排名
     → 优势值: +1(高互动) / -1(负反馈) / 0(其他)
  
  3. 真实用户反馈替代纯Reward模型
     → V1依赖reward模型 → reward hacking风险
     → V2直接用用户真实行为 → 更可靠
  
  4. On-policy采样验证
     → 含25%OneRec自生成流量 → 效果显著提升
     → 验证self-improvement through policy optimization
```

**GBPO vs GRPO vs PPO对比**：

| 方法 | clip操作 | Critic | 样本利用 | 动态bound | 模型数量 |
|------|----------|--------|----------|-----------|----------|
| PPO | 有(硬clip) | 需要 | 部分(clip丢弃) | 无 | 4 |
| GRPO | 有(硬clip) | 不需要 | 部分(clip丢弃) | 无 | 2 |
| GBPO | **无** | 不需要 | **全部** | **有** | 2 |

### 5.3 OneSearch — PARS自适应奖励

- **论文**：[OneSearch](https://arxiv.org/abs/2509.03236)

**后训练方法论**：

```
OneSearch后训练 (PARS + List-wise DPO):

  1. PARS (Preference-Aware Reward System)
     → 六级分层奖励 λ=[2.0, 1.5, 1.0, 0.5, 0.2, 0.0]
     → 动态权重调整: CTR/CVR按品类自适应
     → 三塔架构: CTR + CVR + CTCVR
     → 相关性评分10×权重 (query-item约束)
  
  2. List-wise DPO
     → Reward模型reranking → 识别偏好shift
     → 生成正/负样本对 → DPO继续对齐
     → 结合NTP loss混合训练
```

### 5.4 OneRec-Think — GRPO + Rollout-Beam Reward

- **论文**：[OneRec-Think](https://arxiv.org/abs/2510.11639) (ACL 2026)
- **强化学习方法**：**GRPO** (Group Relative Policy Optimization, DeepSeek 2024)

**后训练方法论**：

```
OneRec-Think后训练 (GRPO + Rollout-Beam Reward):

  Stage 3: Reasoning Enhancement (RL)
  
  RL算法: GRPO (论文原文: "we optimize the model using GRPO based on ℛ_Rollout-Beam")
  ├── 使用VERL分布式RL基础设施
  ├── 超参数: 16条采样CoT路径, beam宽度K=32, 2个训练epoch
  ├── 学习率: 1e-5, KL系数β=0.001, clip ratio ε=0.2
  └── 目标: 同时优化推理连贯性和推荐准确性

  1. Rollout-Beam Reward Mechanism ← 核心创新
     → 问题: 传统verification reward (pass/fail二值) 极度稀疏
       → 大部分推理rollout未命中目标item → 全部得到0 reward → 无法学习
     → 解决: 在constrained beam search (K=32) 内评估
     → 公式: ℛ_Rollout-Beam = max_{ŝ∈ℬ} Σ_{l=1}^L 𝕀(ŝ^l = s^l)
       → 对beam内最优候选计算SID token级匹配数 → 连续值reward
     → 利用了推荐中"多有效性" (multi-validity): 多个item可能都是合理推荐
     → 提供比二值pass/fail更密集的学习信号
  
  2. Think-Ahead推理架构 (部署)
     → Stage1(离线): 生成T条推理路径 + beam search生成prefix (前2个SID token)
     → Stage2(在线): 用prefix约束快速生成第3个token
     → 推理与生成解耦 → 满足延迟约束
```

### 5.5 PLUM — 加权SFT（无显式RL）

- **论文**：[PLUM](https://arxiv.org/abs/2510.07784)

**后训练方法论**：

```
PLUM后训练 (Reward-Weighted SFT, 无显式RL/DPO):

  SFT阶段:
    → reward加权NTP CE (详见§4.3.4公式)
    → r是手工设计的reward信号
    → 训练样本按reward值采样
    → loss计算时权重相等
  
  无显式RL/DPO:
    → PLUM认为简单的加权SFT已足够
    → 可能是因为YouTube场景的反馈信号较强(CTR等)
    → 不需要额外的偏好对齐层
  
  Beam search推理:
    → beam_size=16, hallucination率 < 5%
```

### 5.6 HSTU — 纯NTP无后训练对齐

- **论文**：[HSTU](https://arxiv.org/abs/2402.17152)

**后训练方法论**：

```
HSTU后训练: 无公开的后训练对齐方法

  → 论文只描述了预训练(NTP), 未公开SFT/RL/DPO细节
  → 可能Meta内部有后续优化管线, 但未在论文中披露
  → 1.5万亿参数的纯NTP pretrain → 直接部署
  → 线上A/B测试指标提升12.4%
  
  猜测: Meta可能在内部做了类似RLHF的对齐, 但未公开
```

### 5.7 LLM4DLRMs后训练（QARM V2等）

LLM4DLRMs的后训练（如果统一称为后训练的话）一般就是**传统DLRM的训练**（BCE + 多任务loss），不涉及DPO/GRPO等偏好对齐方法。

```
QARM V2的后训练:
  → LLM推理增强 → Res-KmeansFSQ → DLRM微调
  → 推理能力注入语义ID → DLRM直接用SID embedding做判别式推荐
  → Loss: BCE + 多任务 (与传统DLRM训练相同)
  → 无DPO/GRPO偏好对齐
```

---

## 6. 预训练→后训练演进趋势与展望

### 6.1 各论文训练管线对照

| 论文 | 架构 | 预训练 | SFT | 后训练对齐 | 部署效果 |
|------|------|--------|-----|------------|----------|
| **TIGER** | Enc-Dec(T5) | NTP CE (全序列) | 无 | 无 | 离线验证 |
| **OneRec V1** | Enc-Dec+MoE | NTP CE (session-wise) | NTP CE | Iterative DPO (RM仅构造偏好对) | 观看时长+1.6% |
| **OneRec V2** | Decoder-Only | NTP CE (NIO) | NTP CE | GBPO + Duration Reward | Stay Time+0.5~0.7% |
| **PLUM** | Decoder-Only(Gemini) | CPT+NTP (行为+元数据+通用文本) | Reward-weighted SFT | 无显式RL | CTR+4.96% |
| **HSTU** | Decoder-Only(HSTU) | NTP (Generative) | 未公开 | 未公开 | A/B+12.4% |
| **RPG** | Decoder-Only+MTP | MTP CE | 无 | 无 | NDCG+12.6% |
| **OneSearch** | Enc-Dec | 3-stage SFT | SFT | PARS+List-wise DPO | 电商搜索 |
| **OneRec-Think** | Decoder-Only | Multi-task CPT | Reasoning SFT | **GRPO** + Rollout-Beam Reward | Stay Time+0.16% |
| **QARM** | DLRM | SID+DLRM训练 | 无 | 无 | 广告收入+9.7% |

### 6.2 演进趋势

```
趋势1: 架构统一化 → Decoder-Only
  OneRec V1(Enc-Dec) → V2(Decoder-Only): 减少94%计算
  PLUM直接复用Decoder-Only LLM
  HSTU定制Decoder-Only → 1.5万亿参数
  → Decoder-Only是预训练架构的主流选择

趋势2: 后训练从off-policy到on-policy
  OneRec V1: DPO (off-policy, reward模型构造偏好对)
  OneRec V2: GBPO (on-policy, 真实用户反馈+动态bound)
  → On-policy能感知策略变化,效果更好
  → GBPO去除clip → 更充分的梯度利用

趋势3: Reward设计从模型评分到真实反馈
  OneRec V1: Reward模型评分 → reward hacking风险
  OneRec V2: 用户真实行为反馈 → 更可靠
  → Duration-Aware Reward Shaping解决时长偏差

趋势4: 推理链增强 → Think-before-Recommend
  OneRec-Think: 生成推理链再做推荐决策
  → 类似LLM的CoT (Chain-of-Thought)
  → 推理与生成解耦 → 满足延迟约束

趋势5: 多目标对齐
  单纯NTP → 单维reward → 多维reward
  → 时长、点赞、关注、评论等多维反馈
  → Reward Shaping / Adaptive Weighting
  → 推荐不是单目标优化

趋势6: 端到端优化
  当前: SID构造(离线) → LLM预训练(在线) → 分离优化
  未来: DIGER式端到端 → 推荐梯度回传到量化层
  → SID构造和推荐目标联合优化
```

### 6.3 未来方向

1. **Scaling Law深化**：HSTU初步验证了推荐领域的power-law scaling，但更细粒度的scaling study（数据量、模型大小、训练步数的关系）仍需探索
2. **On-policy迭代对齐**：GRPO/GBPO的多轮on-policy迭代能否持续逼近用户偏好？理论保证和收敛条件是什么？
3. **DPO vs GRPO的选择标准**：什么场景该用DPO，什么场景该用GRPO？计算成本、数据可得性、反馈质量如何影响选择？
4. **多模态融合与预训练**：MoE多模态编码（MMQ）+ 生成架构的深度结合
5. **端到端SID优化**：DIGER方向——可微分SID让推荐梯度直接影响量化层
6. **推理增强推荐**：OneRec-Think方向——显式推理链提升推荐质量
7. **流式后训练**：如何让DPO/GRPO也适配流式训练（Kafka + 持续rollout + 持续对齐）

---

## 参考文献

1. Rajput, S., et al. (2023). Recommender Systems with Generative Retrieval (TIGER). *NeurIPS 2023*. arXiv:2305.05065.

2. Singh, A., et al. (2024). Better Generalization with Semantic IDs. *RecSys 2024*. arXiv:2306.08121.

3. OneRec Team, Kuaishou. (2025). OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment. arXiv:2502.18965.

4. OneRec V2 Team, Kuaishou. (2025). OneRec-V2 Technical Report. arXiv:2508.20900.

5. OneRec-Think Team, Kuaishou & Tsinghua. (2026). OneRec-Think: In-Text Reasoning for Generative Recommendation. *ACL 2026*. arXiv:2510.11639.

6. Meta GR Team. (2024). Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (HSTU). arXiv:2402.17152.

7. PLUM Team, Google DeepMind & YouTube. (2025). PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations. arXiv:2510.07784.

8. RPG Team, Meta. (2025). Recommendation with Parallel semantic ID Generation. *KDD 2025*. arXiv:2506.05781.

9. QARM Team, Kuaishou. (2024). QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou. arXiv:2411.11739.

10. QARM V2 Team, Kuaishou. (2026). QARM V2: Quantitative Alignment Multi-Modal Recommendation for Reasoning User Sequence Modeling. arXiv:2602.09458.

11. Ramasamy, D., et al. (2025). SIDE: Semantic ID Embedding for effective learning from sequences. *AdKDD 2025*. arXiv:2506.16698.

12. DAS Team, Kuaishou. (2025). DAS: Dual-Aligned Semantic IDs Empowered Industrial Recommender System. *CIKM 2025*. arXiv:2508.10584.

13. OneSearch Team, Kuaishou. (2025). OneSearch: A Preliminary Exploration of the Unified End-to-End Generative Framework for E-commerce Search. arXiv:2509.03236.

14. COBRA Team, Baidu. (2025). Unified Generative Recommendations with Cascaded Sparse-Dense Representations. arXiv:2503.02453.

15. Jiang, J., et al. (2026). End-to-End Semantic ID Generation for Generative Advertisement Recommendation (UniSID). arXiv:2602.10445.

16. MMQ Team. (2025). MMQ: Multimodal Mixture-of-Quantization Tokenization. *WSDM 2026*. arXiv:2502.16077.

17. DIGER Team. (2026). Differentiable Semantic ID for Generative Recommendation. *SIGIR 2026*. arXiv:2601.19711.

18. FORGE Team, Alibaba. (2025). FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets. arXiv:2509.20904.

19. Rafailov, R., et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. *NeurIPS 2023*. arXiv:2305.18284.

20. DeepSeek Team. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. (GRPO方法出处).

21. SSRLive Team, ByteDance. (2025). SSRLive: Live Streaming Recommendation with Dynamic Semantic ID. arXiv:2606.06970.

22. QuaSID Team, Alibaba. (2026). QuaSID: Qualification-Aware Semantic ID Learning. arXiv:2603.00632.

23. Survey: A Survey on Generative Recommendation: Data, Model, and Tasks. arXiv:2510.27157.

24. van den Oord, A., et al. (2017). Neural Discrete Representation Learning (VQ-VAE). *NeurIPS 2017*. arXiv:1711.00937.

25. Lee, K., et al. (2022). Autoregressive Image Generation using Residual Quantization. *CVPR 2022*. arXiv:2203.01941.
