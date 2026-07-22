# 生成式推荐中的语义ID（Semantic ID）调研

## 目录

- [1. 概述](#1-概述)
  - [1.1 什么是语义ID](#11-什么是语义id)
  - [1.2 语义ID的核心原理：从连续到离散](#12-语义id的核心原理从连续到离散)
  - [1.3 经典方法：RQ-VAE详解](#13-经典方法rq-vae详解)
  - [1.4 语义ID的两种使用范式：LLM4DLRMs vs LLM4GRs](#14-语义id的两种使用范式llm4dlrms-vs-llm4grs)
  - [1.5 语义ID的构造范式分类：串行 vs 并行](#15-语义id的构造范式分类串行-vs-并行)
  - [1.6 语义ID方法演进总览](#16-语义id方法演进总览)
- [2. RQ-VAE方法（残差量化变分自编码器）](#2-rq-vae方法残差量化变分自编码器)
  - [2.1 TIGER：奠基工作](#21-tiger奠基工作)
  - [2.2 YouTube Semantic IDs：工业排序落地](#22-youtube-semantic-ids工业排序落地)
  - [2.3 DIGER：可微分语义ID](#23-diger可微分语义id)
- [3. RQ-KMeans方法（残差K-Means量化）](#3-rq-kmeans方法残差k-means量化)
  - [3.1 RQ-KMeans原理与公式](#31-rq-kmeans原理与公式)
  - [3.2 QARM：快手多模态对齐](#32-qarm快手多模态对齐)
  - [3.3 QARM V2：混合FSQ量化](#33-qarm-v2混合fsq量化)
  - [3.4 OneRec：快手统一生成式推荐](#34-onerec快手统一生成式推荐)
- [4. VQ/PQ方法（向量量化/乘积量化）](#4-vqpq方法向量量化乘积量化)
  - [4.1 VQ原理与公式](#41-vq原理与公式)
  - [4.2 PQ原理与公式](#42-pq原理与公式)
  - [4.3 VQ-Rec：可迁移序列推荐](#43-vq-rec可迁移序列推荐)
  - [4.4 SIDE：Meta广告序列学习](#44-side-meta广告序列学习)
  - [4.5 MMQ：多模态混合量化](#45-mmq多模态混合量化)
- [5. OPQ方法（优化乘积量化 · 并行）](#5-opq方法优化乘积量化--并行)
  - [5.1 OPQ原理与公式](#51-opq原理与公式)
  - [5.2 RPG：Meta并行语义ID生成](#52-rpgmeta并行语义id生成)
- [6. RQ-OPQ混合方法（串行+并行）](#6-rq-opq混合方法串行并行)
  - [6.1 设计动机](#61-设计动机)
  - [6.2 OneSearch：快手电商搜索](#62-onesearch快手电商搜索)
- [7. 语义ID对齐方法](#7-语义id对齐方法)
  - [7.1 为什么需要对齐？](#71-为什么需要对齐)
  - [7.2 QARM：快手前置对齐](#72-qarm快手前置对齐)
  - [7.3 DAS：快手一阶段联合对齐](#73-das快手一阶段联合对齐)
  - [7.4 PLUM SID-v2：共现对比对齐](#74-plum-sid-v2共现对比对齐)
  - [7.5 MMQ：后置行为感知微调](#75-mmq后置行为感知微调)
  - [7.6 UniSID：端到端联合优化](#76-unisid端到端联合优化)
  - [7.7 DIGER：可微分对齐](#77-diger可微分对齐)
  - [7.8 无显式对齐的方法](#78-无显式对齐的方法)
  - [7.9 对齐方法对比总结](#79-对齐方法对比总结)
- [8. 稀疏+稠密级联方法](#8-稀疏稠密级联方法)
  - [8.1 COBRA：百度级联表示](#81-cobra百度级联表示)
- [9. 端到端语义ID生成与LLM适配](#9-端到端语义id生成与llm适配)
  - [9.1 UniSID：广告推荐端到端生成](#91-腾讯unisid广告推荐端到端生成)
  - [9.2 PLUM：LLM适配生成式推荐](#92-plumllm适配生成式推荐)
  - [9.3 DIGER：可微分语义ID端到端联合优化](#93-diger可微分语义id端到端联合优化)
- [10. 工业基准](#10-工业基准)
  - [10.1 FORGE：淘宝大规模基准](#101-forge淘宝大规模基准)
- [11. 多模态语义ID构造专题](#11-多模态语义id构造专题)
  - [11.1 问题背景](#111-问题背景)
  - [11.2 多模态语义ID构造方法分类](#112-多模态语义id构造方法分类)
  - [11.3 MMQ：阿里MoE混合量化（WWW 2025）](#113-mmq阿里moe混合量化www-2025)
  - [11.4 与其他多模态方法的对比](#114-与其他多模态方法的对比)
  - [11.5 何时选择何种方式？](#115-何时选择何种方式)
- [12. 总结与展望](#12-总结与展望)
- [参考文献](#参考文献)

---

## 1. 概述

### 1.1 什么是语义ID

在传统推荐系统中，每个物品（item）通常被赋予一个**随机哈希ID**（如从1到N的整数），然后通过一个巨大的embedding table将其映射为稠密向量。这种方式存在几个关键问题：

- **语义缺失**：随机ID不包含任何物品内容信息，相似物品的ID之间没有关联
- **泛化困难**：新物品（冷启动）没有交互历史，无法学到有意义的embedding
- **规模瓶颈**：数十亿物品的embedding table会占用巨大的存储和计算资源

**语义ID（Semantic ID）** 的核心思想是：用物品的**内容特征**（如文本、图像、视频等多模态信息）来生成一个紧凑的、离散的标识符，使得语义相似的物品拥有相同或相近的ID。

> **定义**：语义ID是一组由量化（quantization）方法从物品的内容embedding中得到的离散token序列 $s = (c_1, c_2, \ldots, c_L)$，其中每个 $c_i \in \{0, 1, \ldots, K-1\}$ 是一个离散的码本索引，$L$ 是序列长度，$K$ 是码本大小。

语义ID的关键性质：
- **层次性**：在串行生成中，前面的token表示粗粒度语义，后面的token表示细粒度语义
- **共享性**：相似物品共享部分token，实现语义泛化
- **紧凑性**：用 $L \times \log K$ bit即可表示一个物品，远少于原始embedding

### 1.2 语义ID的核心原理：从连续到离散

语义ID的生成涉及两个核心环节：**量化（Quantization）** 和**对齐（Alignment）**。核心流程如下：

```
物品多模态特征 → [对齐: 融合协同信号] → Encoder → 连续向量z → 量化器 → 离散码 (c₁, c₂, ..., c_L)
                                                                          ↓
                                                                     语义ID (Semantic ID)
```

- **量化**：将连续向量离散化为码本索引序列（详见§1.3 RQ-VAE）
- **对齐**：弥合内容语义与协同过滤信号之间的gap（详见§7 对齐方法），可在量化前、量化中、量化后或端到端进行

**数学框架**：给定物品 $i$，其特征经过编码器得到连续向量 $\mathbf{z}_i \in \mathbb{R}^d$。量化的目标是找到一组离散码 $\mathbf{c}_i = (c_1, c_2, \ldots, c_L)$，使得从离散码重构的向量 $\hat{\mathbf{z}}_i$ 尽可能接近 $\mathbf{z}_i$。

### 1.3 经典方法：RQ-VAE详解

**RQ-VAE（Residual Quantized Variational AutoEncoder）** 是TIGER等工作中使用的最经典的语义ID生成方法，也是理解后续所有方法的基础。

#### 1.3.1 架构概览

```
┌─────────┐     ┌──────────────┐     ┌──────────────────┐     ┌─────────┐
│  输入    │     │   Encoder    │     │ 残差量化模块(RQ)   │     │ Decoder │
│  x ∈ R^D│ ──→ │  f_enc(x)=z  │ ──→ │  z → (c₁,...,cL) │ ──→ │  重构 x̂  │
└─────────┘     └──────────────┘     └──────────────────┘     └─────────┘
                                              ↑
                                    L层码本 {C₁, C₂, ..., C_L}
                                    每层K个码字, 每个码字维度d
```

#### 1.3.2 残差量化过程（逐步详解）

**输入**：编码器输出 $\mathbf{z} \in \mathbb{R}^d$

**第1层量化**：
$$c_1 = \arg\min_{k \in \{0,...,K-1\}} \| \mathbf{z} - \mathbf{e}_k^{(1)} \|_2$$
$$\mathbf{r}_1 = \mathbf{z} - \mathbf{e}_{c_1}^{(1)}$$

**第 $l$ 层量化**（$l = 2, \ldots, L$）：对上一层的残差 $\mathbf{r}_{l-1}$ 进行量化：
$$c_l = \arg\min_{k \in \{0,...,K-1\}} \| \mathbf{r}_{l-1} - \mathbf{e}_k^{(l)} \|_2$$
$$\mathbf{r}_l = \mathbf{r}_{l-1} - \mathbf{e}_{c_l}^{(l)}$$

**重构向量**：
$$\hat{\mathbf{z}} = \sum_{l=1}^{L} \mathbf{e}_{c_l}^{(l)}$$

> **直觉理解**：第1层码本捕获粗粒度/全局语义，第2层捕获中等粒度残差，第3层捕获更细粒度残差...类似JPEG的逐层细化编码。

#### 1.3.3 训练损失函数

RQ-VAE的总损失由三部分组成：

$$\mathcal{L} = \underbrace{\| \mathbf{x} - \hat{\mathbf{x}} \|_2^2}_{\text{重构损失}} + \underbrace{\sum_{l=1}^{L} \| \text{sg}(\mathbf{r}_{l-1}) - \mathbf{e}_{c_l}^{(l)} \|_2^2}_{\text{码本损失 (Codebook Loss)}} + \underbrace{\beta \sum_{l=1}^{L} \| \mathbf{r}_{l-1} - \text{sg}(\mathbf{e}_{c_l}^{(l)}) \|_2^2}_{\text{承诺损失 (Commitment Loss)}}$$

其中 $\text{sg}(\cdot)$ 表示 stop-gradient（停止梯度），$\beta$ 是超参数（通常取0.25）。

| 损失项 | 作用 | 梯度流向 |
|--------|------|----------|
| 重构损失 | 使decoder能准确重构输入 | 更新encoder和decoder |
| 码本损失 | 使码字靠近encoder的输出 | 仅更新码本 |
| 承诺损失 | 使encoder的输出靠近选中的码字 | 仅更新encoder |

#### 1.3.4 Straight-Through Estimator (STE)

量化操作中的 $\arg\min$ 不可微，无法直接反向传播。VQ-VAE/RQ-VAE使用**直通估计器（STE）**：
- **前向传播**：$\hat{\mathbf{z}} = \mathbf{e}_{c} $ （离散码字）
- **反向传播**：$\frac{\partial \mathcal{L}}{\partial \mathbf{z}} \approx \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{z}}}$（直接将梯度复制到encoder输出）

值得注意的是， 这里除了STE的做法，还有一些改进方法，如Gumbel-Softmax、Rotation Trick等，但STE是最常用且简单有效的。
#### 1.3.5 训练与推理流程示例

**训练阶段（以batch_size=2, d=64, L=3, K=256为例）**：

```python
# 1. 编码
z = encoder(items)               # shape: [2, 64] → 2个物品, 每个64维

# 2. 第1层残差量化
dist_1 = cdist(z, codebook_1)    # shape: [2, 256] → 每个item到256个码字的距离
c1 = dist_1.argmin(dim=-1)       # shape: [2] → e.g., [42, 187]
r1 = z - codebook_1[c1]          # shape: [2, 64] → 残差

# 3. 第2层残差量化
dist_2 = cdist(r1, codebook_2)   # shape: [2, 256]
c2 = dist_2.argmin(dim=-1)       # shape: [2] → e.g., [15, 203]
r2 = r1 - codebook_2[c2]         # shape: [2, 64]

# 4. 第3层残差量化
dist_3 = cdist(r2, codebook_3)   # shape: [2, 256]
c3 = dist_3.argmin(dim=-1)       # shape: [2] → e.g., [89, 12]

# 5. 重构
z_hat = codebook_1[c1] + codebook_2[c2] + codebook_3[c3]  # shape: [2, 64]

# 6. 解码
x_hat = decoder(z_hat)           # shape: [2, D] → 重构的物品特征

# 7. 计算损失
loss = reconstruction_loss(x, x_hat) + codebook_loss + commitment_loss
```

**生成的语义ID**：
- 物品A的语义ID = (42, 15, 89)
- 物品B的语义ID = (187, 203, 12)

**推理阶段**（生成式检索）：

```
用户历史: [物品X的ID=(3,7,22), 物品Y的ID=(3,15,89)]
                                    ↓
            Transformer自回归生成: P(c1|历史) → P(c2|历史,c1) → P(c3|历史,c1,c2)
                                    ↓
                        生成目标物品ID = (3, 15, 42)
                                    ↓
                        在码本中查找对应的物品 → 推荐结果
```

### 1.4 语义ID的两种使用范式：LLM4DLRMs vs LLM4GRs

语义ID在推荐系统中有两种主要的使用范式，对应于两种不同的架构思想：

#### 1.4.1 LLM4DLRMs（LLM赋能的深度推荐模型）

**核心思想**：将语义ID作为**特征**输入到传统推荐模型（如双塔模型、DNN排序模型）中，替换或补充原来的随机Item ID embedding。

```
传统方式: 随机Item ID → Embedding Table → 特征拼接 → DNN → CTR预估
                                         ↑
LLM4DLRMs方式: 语义ID → 小Embedding Table → 特征拼接 → DNN → CTR预估
```

**特点**：
- 推荐模型的主体架构不变（仍是DLRM、双塔等）
- 语义ID仅替换Item ID特征，不改变模型范式
- 训练流程：先离线生成语义ID → 再训练推荐模型
- **优势**：工业部署改动小，可渐进式迁移

**发展历程**：

> LLM4DLRMs中使用多模态/LLM表征经历了三个阶段的演进：

**阶段一：冻结Dense特征直接拼接（早期）**

将LLM encoder（或多模态encoder）的输出作为额外的dense特征，对齐后直接拼接到下游DLRM中。

```
LLM Encoder (frozen) → dense_emb ∈ R^d → 缓存 → 拼接到DLRM输入
```

- **代表工作**：
  - 华为 [Towards Open-World Recommendation with Knowledge Augmentation from Large Language Models](https://arxiv.org/abs/2306.10933) (arXiv:2306.10933)
  - 蚂蚁 [Enhancing Sequential Recommenders with Augmented Knowledge from Aligned Large Language Models](https://dl.acm.org/doi/10.1145/3626772.3657782) (SIGIR 2024)
- **问题**：
  - LLM embedding存储在缓存中，作为**固定输入**，无法通过推荐模型梯度更新
  - 需要先过一个参数可学习的对齐网络，再用用户交互信号对齐，才能在下游使用
  - 对下游训练不友好，表征与推荐目标存在gap（一般只能做双阶段训练）

**阶段二：多模态Embedding作为ID初始化（中期尝试）**

直接将多模态dense embedding当成普通ID embedding去梯度更新，相当于用多模态信息做embedding table的初始化。

```
多模态embedding ∈ R^d → 作为Embedding Table初始值 → 端到端梯度更新
```

- **问题**：
  - 维度太大（$d=768$~$4096$），训练难以有效更新
  - 增量训练几个月后，LLM的世界知识遗忘殆尽，效果退化为与随机初始化相当
  - 存储和计算开销大

**阶段三：量化为离散语义ID（当前主流）** ✅

**为什么阶段三比阶段一更好？**

阶段一（冻结Dense特征直接拼接）存在几个核心问题，而阶段三（量化为离散语义ID）通过**信息压缩+离散化**巧妙解决了这些问题：

| 维度 | 阶段一：冻结Dense特征 | 阶段三：量化为离散语义ID | 优势来源 |
|------|---------------------|------------------------|----------|
| **存储开销** | 每个物品存储 768~4096 维浮点向量 | 每个物品仅需 $L$ 个整数（如 3 个 token） | 存储减少 1000×+ |
| **计算效率** | 拼接高维向量，下游模型输入维度大 | 查小 Embedding Table，维度可控 | 训练/推理更快 |
| **可训练性** | Dense 特征冻结，无法通过推荐梯度更新 | SID Embedding 可端到端学习 | 表征与目标对齐 |
| **泛化能力** | 向量空间连续，相似物品距离近但无共享参数 | 相似物品共享 token，实现参数级泛化 | 语义层次结构 |
| **冷启动** | 新物品需先过 Encoder，表征与推荐目标有 gap | 新物品基于内容获得有意义 SID，立即可用 | 内容驱动 ID |
| **LLM 兼容** | Dense 向量难以融入 LLM 词表 | 离散 token 自然融入 LLM 词表扩展 | 统一建模 |

**核心洞察**：

1. **信息压缩的本质**：高维 Dense Embedding 包含大量冗余信息（维度间高度相关），量化通过**残差逐层编码**将信息压缩到最关键的语义维度，实现"去冗余、保语义"。

2. **离散化的价值**：
   - **参数共享**：相似物品共享部分 token（如前缀相同），对应的 Embedding 参数被多个物品共用，实现**隐式正则化**和**语义泛化**
   - **组合爆炸**：$L$ 层码本每层 $K$ 个码字，可表达 $K^L$ 种组合（如 $256^3 = 1677$ 万），远大于实际物品数，保证唯一性的同时实现层次语义

3. **与 LLM 的天然契合**：LLM 本身就是处理离散 token 序列的模型，将物品表示为离散 SID token 后，可以：
   - 直接复用 LLM 的 Transformer 架构和训练范式（预训练 + 微调）
   - 统一处理文本和物品的多模态序列（如 "用户画像 + 历史 SID 序列 → 生成目标 SID"）
   - 利用 LLM 的世界知识和推理能力增强推荐

**代表工作**：[QARM](https://arxiv.org/abs/2411.11739)（快手）、[YouTube Semantic IDs](https://arxiv.org/abs/2306.08121)（Google）、[SIDE](https://arxiv.org/abs/2506.16698)（Meta）
- **两阶段训练**：
  1. **第一阶段**：训练量化器（RQ-VAE、RQ-KMeans等），得到物品的语义ID，一般也会加入对比学习对齐语义和协同空间
  2. **第二阶段**：训练下游推荐模型，使用语义ID作为离散特征输入
- **关键特性**：
  - 语义ID对应的**码本是固定的**，不在下游训练中更新 → 避免大规模embedding table的存储/计算开销
  - 语义ID对应的**embedding是可学习的** → 可在下游DLRM训练中微调，提升推荐效果
- **本质**：将高维多模态信息进行**降维离散化**，得到紧凑的语义ID，兼顾信息压缩和可训练性
**代表论文概览**：

| 论文 | 链接 | 量化方法 | 生成方式 | 核心贡献 |
|------|------|----------|----------|----------|
| [Better Generalization with Semantic IDs](#22-youtube-semantic-ids工业排序落地) (Google, RecSys 2024) | [arXiv:2306.08121](https://arxiv.org/abs/2306.08121) | RQ-VAE | 串行 | 十亿级YouTube排序中替换随机ID |
| [QARM](#32-qarm快手多模态对齐) (快手, 2024) | [arXiv:2411.11739](https://arxiv.org/abs/2411.11739) | RQ-KMeans | 串行 | 多模态对齐+残差K-Means语义ID |
| [QARM V2](#33-qarm-v2混合fsq量化) (快手, 2026) | [arXiv:2602.09458](https://arxiv.org/abs/2602.09458) | RQ-KMeans+FSQ | 串行 | 混合量化解决码本冲突 |
| [VQ-Rec](#42-vq-rec可迁移序列推荐) (人大, 2022) | [arXiv:2210.12316](https://arxiv.org/abs/2210.12316) | PQ | 并行 | 可迁移序列推荐 |
| [SIDE](#43-side-meta广告序列学习) (Meta, 2025) | [AdKDD 2025](https://arxiv.org/abs/2506.16698) | VQ-Fusion | 并行 | 无参数SID转换，广告序列学习 |
| [DAS](#71-das快手双对齐语义id) (快手, 2025) | [arXiv:2508.10584](https://arxiv.org/abs/2508.10584) | 兼容多种 | 串行 | 双对齐注入协同信号 |

#### 1.4.2 LLM4GRs（LLM赋能的生成式推荐）

**核心思想**：将推荐任务重构为**序列生成任务**，利用LLM/Transformer的自回归生成能力，直接生成目标物品的语义ID。

```
用户行为序列: [SID₁, SID₂, ..., SID_n] → Transformer → 生成 SID_{n+1}
                                                     ↓
                                              自回归逐token生成:
                                              P(c₁) → P(c₂|c₁) → P(c₃|c₁,c₂)
```

**特点**：
- 推荐 = 生成，检索/排序统一为自回归解码
- 语义ID是生成的"词汇"，类似NLP中的token
- 可以直接复用LLM的架构和训练范式（预训练+微调）
- **优势**：统一检索和排序，强泛化能力，适合冷启动

**代表论文概览**：

| 论文                                            | 链接 | 量化方法 | 生成方式 | 核心贡献 |
|-----------------------------------------------|------|----------|----------|----------|
| [TIGER](#21-tiger奠基工作) (Google, NeurIPS 2023) | [arXiv:2305.05065](https://arxiv.org/abs/2305.05065) | RQ-VAE | 串行 | 奠基工作，首次提出语义ID+生成式检索 |
| [OneRec](#34-onerec快手统一生成式推荐) (快手, 2025)      | [arXiv:2502.18965](https://arxiv.org/abs/2502.18965) | RQ-KMeans | 串行 | 统一检索排序+多级别衡量化 |
| [OneRec V2](#34-onerec快手统一生成式推荐) (快手, 2025)   | [arXiv:2508.20900](https://arxiv.org/abs/2508.20900) | RQ-KMeans | 串行 | Lazy Decoder-Only, 80亿参数 |
| [RPG](#52-rpgmeta并行语义id生成) (Meta, KDD 2025)   | [arXiv:2506.05781](https://arxiv.org/abs/2506.05781) | OPQ | 并行 | 一步并行生成全部语义ID |
| [OneSearch](#62-onesearch快手电商搜索) (快手, 2025)   | [arXiv:2509.03236](https://arxiv.org/abs/2509.03236) | RQ-OPQ | 混合 | RQ串行+OPQ并行，电商搜索 |
| [PLUM](#92-plumllm适配生成式推荐) (Google, 2025)     | [arXiv:2510.07784](https://arxiv.org/abs/2510.07784) | SID-v2 (RQ-VAE+对比损失) | 串行 | 预训练LLM适配生成式推荐，SID离线构造 |
| [UniSID](#91-unisid广告推荐端到端生成) (2026)          | [arXiv:2602.10445](https://arxiv.org/abs/2602.10445) | 端到端 | 串行 | 端到端联合优化embedding和SID |
| [DIGER](#23-diger可微分语义id) (SIGIR 2026)        | [arXiv:2601.19711](https://arxiv.org/abs/2601.19711) | 可微分RQ | 串行 | Gumbel-Softmax可微分量化 |
| [COBRA](#81-cobra百度级联表示) (百度, 2025)           | [arXiv:2503.02453](https://arxiv.org/abs/2503.02453) | 级联 | 级联 | 稀疏语义ID+稠密向量级联 |
| [MMQ](#44-mmq多模态混合量化) (阿里, 2025)               | [arXiv:2508.15281](https://arxiv.org/abs/2508.15281) | MoE | 并行 | 多模态专家混合量化 |

#### 1.4.3 两种范式对比

| 维度 | LLM4DLRMs | LLM4GRs |
|------|-----------|---------|
| 模型范式 | 判别式（Discriminative） | 生成式（Generative） |
| 语义ID角色 | 输入特征 | 生成目标 |
| 推荐方式 | 全库打分取Top-K | 自回归生成+Beam Search |
| 架构改动 | 小（仅换ID特征） | 大（重构为生成模型） |
| 推理复杂度 | O(N)（N为候选数） | O(L×K)（L为ID长度，K为码本大小） |
| 代表系统 | [QARM](https://arxiv.org/abs/2411.11739)、[SIDE](https://arxiv.org/abs/2506.16698)、[YouTube](https://arxiv.org/abs/2306.08121)、[DAS](https://arxiv.org/abs/2508.10584) | [TIGER](https://arxiv.org/abs/2305.05065)、[PLUM](https://arxiv.org/abs/2510.07784)、[OneRec](https://arxiv.org/abs/2502.18965)、[OneSearch](https://arxiv.org/abs/2509.03236) |

### 1.5 语义ID的构造范式分类：串行 vs 并行

#### 1.5.1 串行语义ID（Serial Semantic ID）

**原理**：逐层/逐个token地生成语义ID，后续token的生成依赖于前面的token。

```
Serial: c₁ → c₂ → c₃ → ... → c_L     （自回归，逐层生成）
```

**代表方法**：RQ-VAE（TIGER）、Res-KMeans（QARM、OneRec）

**优势**：
- 天然形成层次语义结构（粗→细）
- 码本利用率高：$L$ 层码本 $K$ 可表达 $K^L$ 种组合
- 适合Beam Search，可在生成过程中剪枝

**劣势**：
- 推理延迟随层数 $L$ 线性增长
- 深层码本可能利用不充分

#### 1.5.2 并行语义ID（Parallel Semantic ID）

**原理**：所有token同时生成，互不依赖。

```
Parallel: c₁, c₂, c₃, ..., c_M        （一步并行生成）
```

**代表方法**：OPQ/PQ（RPG、VQ-Rec）

**优势**：
- 推理速度快，一步生成全部token
- 推理延迟与ID长度无关

**劣势**：
- 缺乏层次语义结构
- 码本组合空间稀疏，"合法"组合占比极小
- 需要特殊策略处理无效组合

#### 1.5.3 串行+并行混合（Hybrid）

**原理**：结合串行的层次语义和并行的推理效率。

```
Hybrid: [c₁ → c₂ → c₃] (串行, 粗粒度层次) + [p₁, p₂, ..., p_M] (并行, 细粒度特征)
```

**代表方法**：RQ-OPQ（OneSearch）

### 1.6 语义ID方法部分时间线

```
2022  VQ-Rec ─────────────────────── PQ (乘积量化, 并行)
       │
2023  TIGER ──────────────────────── RQ-VAE (残差量化, 串行) ← 奠基之作
       │
2023  YouTube Semantic IDs ──────── RQ-VAE + SentencePiece适配 (串行)
       │
2024  QARM (快手) ───────────────── RQ-KMeans + 多模态对齐 (串行)
       │
2025  OneRec (快手) ─────────────── RQ-KMeans + 多级别平衡 (串行)
       │
2025  RPG (Meta) ────────────────── OPQ (并行)
       │
2025  OneSearch (快手) ──────────── RQ-OPQ (串行+并行混合)
       │
2025  DAS (快手) ────────────────── 双对齐语义ID
       │
2025  SIDE (Meta) ───────────────── VQ-Fusion + DPCA (并行)
       │
2025  COBRA (百度) ──────────────── 级联稀疏-稠密表示
       │
2026  QARM V2 (快手) ───────────── RQ-KMeans + FSQ (串行)
       │
2026  UniSID ────────────────────── 端到端多粒度对比学习
       │
2026  MMQ ───────────────────────── MoE混合量化
```

---

## 2. RQ-VAE方法（残差量化变分自编码器）

**方法分类**：串行语义ID
**核心原理**：通过Encoder-Decoder架构 + 残差量化模块，端到端学习离散语义表示。码本通过梯度更新优化。

### 方法原理总览

RQ-VAE = Encoder（神经网络） + 残差量化（最近邻查找） + Decoder（神经网络），通过重构损失 + 码本损失 + 承诺损失联合优化。详细公式见[1.3节](#13-经典方法rq-vae详解)。

### 采用RQ-VAE方法的论文

| 论文 | 使用范式 | 生成方式 | 核心改进 |
|------|----------|----------|----------|
| [TIGER](https://arxiv.org/abs/2305.05065) (Google, 2023) | LLM4GRs | 串行 | 奠基工作，首次提出语义ID概念 |
| [YouTube Semantic IDs](https://arxiv.org/abs/2306.08121) (Google, 2024) | LLM4DLRMs | 串行 | SentencePiece适配，十亿级排序系统 |
| [DIGER](https://arxiv.org/abs/2601.19711) (2026) | LLM4GRs | 串行 | 可微分语义ID，联合优化量化与推荐 |

### 2.1 TIGER：奠基工作

- **标题**：[Recommender Systems with Generative Retrieval](https://arxiv.org/abs/2305.05065)
- **机构**：Google / Google DeepMind
- **发表**：NeurIPS 2023
- **arXiv**：https://arxiv.org/abs/2305.05065
- **使用范式**：**LLM4GRs**（生成式检索，自回归解码语义ID）
- **构造范式**：**串行**（RQ-VAE逐层生成）

**语义ID生成**（离线训练）：

```
物品文本特征 → SBERT (frozen, d=768) → RQ-VAE Encoder (可训练) → 残差量化(3层, K=4096) → (c₁, c₂, c₃)
                  ↑                         ↑
             外层Encoder               内层Encoder
             (冻结,产出内容embedding)   (可训练,映射到量化空间)
```

> **关于Encoder**：TIGER有**两层Encoder**。外层SBERT（冻结）负责将物品文本编码为内容embedding；内层RQ-VAE自带的Encoder（可训练）将内容embedding映射到量化空间。RQ-VAE的Encoder、Decoder和码本通过重构损失+码本损失+承诺损失端到端联合训练。这与RQ-KMeans方案（如OneRec）有本质区别——RQ-KMeans没有内层Encoder，直接在frozen embedding上做聚类（详见§3.1对比表）。

**生成式检索**（在线推理）：
```
用户历史序列 → Transformer Seq2Seq → 自回归生成目标物品的语义ID
                                         ↓
                              Beam Search: P(c₁) → P(c₂|c₁) → P(c₃|c₁,c₂)
```

**层次感知Beam Search**：利用RQ-VAE的层次结构，第1层beam_size较小（粗粒度），逐层扩展（细粒度），实现高效剪枝。

**关键结果**：冷启动物品Recall@10提升超过50%。

### 2.2 YouTube Semantic IDs：工业排序落地

- **标题**：[Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations](https://arxiv.org/abs/2306.08121)
- **机构**：Google / Google DeepMind
- **发表**：RecSys 2024
- **arXiv**：https://arxiv.org/abs/2306.08121
- **使用范式**：**LLM4DLRMs**（语义ID作为排序模型的输入特征）
- **构造范式**：**串行**（RQ-VAE）

**核心创新**：
- 使用**SentencePiece模型（SPM）** 将语义ID适配到推荐模型词汇表
- 在YouTube十亿级排序系统中替换随机Item ID

**语义ID在排序模型中的使用**：
```
输入特征:
  ├── 用户特征 (age, gender, history, ...)
  ├── 上下文特征 (time, device, ...)
  └── 物品特征:
      ├── 传统: 随机Item ID → Embedding → 拼接
      └── 改进: 语义ID → SentencePiece → Embedding → 拼接
                                                        ↓
                                                 DNN → P(click)
```

**核心洞察**：
> 内容embedding泛化好但记忆差，随机ID记忆好但泛化差。语义ID是两者之间的最佳折衷。

### 2.3 DIGER：可微分语义ID

- **标题**：[Differentiable Semantic ID for Generative Recommendation](https://arxiv.org/abs/2601.19711)
- **发表**：SIGIR 2026
- **使用范式**：**LLM4GRs**
- **构造范式**：**串行**（可微分RQ）

**核心问题**：传统方法中语义ID是冻结的，推荐模型的梯度无法回传到量化器，导致**目标不一致**。

**解决方案**：使用Gumbel-Softmax实现可微分量化：

$$c_l = \text{Gumbel-Softmax}(\text{logits}_l, \tau)$$

其中温度 $\tau$ 随训练逐步降低，从探索过渡到利用。推荐损失可以直接优化语义ID的生成。

DIGER实现了基于语义ID的生成式推荐从双阶段训练到端到端联合优化的过渡。

---

## 3. RQ-KMeans方法（残差K-Means量化）

**方法分类**：串行语义ID
**核心原理**：用K-Means聚类替代神经网络量化器，在残差上迭代聚类生成层次码本。

### 3.1 RQ-KMeans原理与公式

RQ-KMeans与RQ-VAE的核心区别在于：**码本学习不使用梯度更新，而是直接对残差做K-Means聚类**。

```
┌─────────────┐     ┌──────────────────────┐
│ 物品embedding │     │ 残差K-Means量化模块    │
│ z ∈ R^d      │ ──→ │ L层K-Means → 语义ID   │
│ (可frozen)    │     └──────────────────────┘
└─────────────┘
```

**算法步骤**：

```python
def residual_kmeans(embeddings, num_levels=L, num_clusters=K):
    """
    embeddings: shape [N, d], N个物品的embedding
    返回: semantic_ids [N, L], codebooks [L, K, d]
    """
    semantic_ids = []
    residuals = embeddings.clone()      # shape: [N, d]
    codebooks = []

    for level in range(L):
        # 对当前残差做K-Means聚类
        centroids, labels = kmeans(residuals, k=K)
        # centroids shape: [K, d]
        # labels shape: [N] → 第level层的语义ID token

        codebooks.append(centroids)
        semantic_ids.append(labels)

        # 计算新的残差
        quantized = centroids[labels]    # shape: [N, d]
        residuals = residuals - quantized # shape: [N, d]

    return stack(semantic_ids), codebooks
    # semantic_ids shape: [N, L] → 每个物品L个token
    # codebooks: L个码本, 每个[K, d]
```

**与RQ-VAE的关键区别**：

| 维度 | RQ-VAE (TIGER) | RQ-KMeans (QARM/OneRec) |
|------|----------------|-------------------------|
| 量化方法 | 神经网络 + 最近邻查找 | K-Means聚类 |
| 码本学习 | 端到端梯度更新 | 迭代聚类中心更新 |
| **量化器内部Encoder** | **✅ 有，可训练（映射到量化空间）** | **❌ 无（直接在frozen embedding上聚类）** |
| **Decoder** | **✅ 有，可训练（重构输入）** | **❌ 无** |
| 初始内容Encoder | 冻结（SBERT等） | 冻结（多模态模型） |
| 输入来源 | 单一frozen embedding | 对齐后的多模态embedding |
| 训练复杂度 | 高（需训练VAE） | 低（仅K-Means） |
| 工业部署 | 较重 | 轻量 |
| 码本平衡性 | 可能出现不均匀 | K-Means天然更均衡 |

> **关于Encoder的关键区别**：两种方法的初始内容Encoder（SBERT/CLIP/多模态模型）都是**冻结**的。但RQ-VAE自带一个**可训练的Encoder/Decoder**——外层frozen embedding先经过RQ-VAE的Encoder映射到量化空间，再查码本、再经Decoder重构。而RQ-KMeans**没有**这个内层Encoder/Decoder，直接在frozen embedding上做K-Means聚类。这也是RQ-KMeans更轻量的原因之一。

**RQ-KMeans的优势**（被OneRec等工作验证）：
- **重构质量**：K-Means直接最小化量化误差
- **码本利用率**：聚类中心分布更均匀，减少空码字
- **平衡性**：K-Means的聚类平衡性优于最近邻查找

### 采用RQ-KMeans方法的论文

| 论文 | 使用范式 | 生成方式 | 核心改进 |
|------|----------|----------|----------|
| [QARM](https://arxiv.org/abs/2411.11739) (快手, 2024) | LLM4DLRMs | 串行 | 多模态对齐 + Res-KMeans |
| [QARM V2](https://arxiv.org/abs/2602.09458) (快手, 2026) | LLM4DLRMs | 串行 | Res-KMeans + FSQ混合 |
| [OneRec](https://arxiv.org/abs/2502.18965) (快手, 2025) | LLM4GRs | 串行 | 统一检索排序 + 多级别衡量化 |

### 3.2 QARM：快手多模态对齐

- **标题**：[QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou](https://arxiv.org/abs/2411.11739)
- **机构**：快手
- **发表**：arXiv 2024年11月
- **arXiv**：https://arxiv.org/abs/2411.11739
- **使用范式**：**LLM4DLRMs**（语义ID作为特征输入推荐模型）
- **构造范式**：**串行**（Res-KMeans）

**解决的两个核心问题**：
1. **Representation Unmatching**：预训练多模态embedding与推荐业务分布不一致
2. **Representation Unlearning**：冻结的多模态embedding无法被推荐任务适配

**整体流程**：

```
┌──────────────────────────────────────────────────────┐
│                QARM Pipeline                          │
│                                                      │
│  Step 1: 多模态对齐 (Item Alignment)                  │
│  ┌────────────────────────────────────────────┐      │
│  │ CLIP Image → img_emb ∈ R^512               │      │
│  │ Text Enc   → text_emb ∈ R^512              │      │
│  │   ↓ concat with business features          │      │
│  │ MLP_align → aligned_emb ∈ R^d              │      │
│  │ (对比学习损失 + 业务标签损失)                 │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
│  Step 2: 残差K-Means量化 (Quantitative Code)          │
│  ┌────────────────────────────────────────────┐      │
│  │ aligned_emb → Res-KMeans(L层, K个聚类)      │      │
│  │ → Semantic ID = (c₁, c₂, ..., c_L)         │      │
│  │ (码本训练完成后冻结)                          │      │
│  └────────────────────────────────────────────┘      │
│                                                      │
│  Step 3: 端到端推荐训练                                │
│  ┌────────────────────────────────────────────┐      │
│  │ Semantic ID → Embedding Lookup             │      │
│  │ sid_emb = [emb₁[c₁], emb₂[c₂], ...]       │      │
│  │   → concat → DNN → CTR预估                 │      │
│  │ (embedding参数可端到端更新)                   │      │
│  └────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────┘
```

**部署效果**：2024年3月起部署于快手多个推荐场景，服务4亿日活用户。

### 3.3 QARM V2：混合FSQ量化

- **标题**：[QARM V2: Quantitative Alignment Multi-Modal Recommendation for Reasoning User Sequence Modeling](https://arxiv.org/abs/2602.09458)
- **机构**：快手
- **发表**：arXiv 2026年2月
- **arXiv**：https://arxiv.org/abs/2602.09458
- **使用范式**：**LLM4DLRMs**
- **构造范式**：**串行**（Res-KMeans + FSQ混合）

**核心改进**：将Res-KMeans与**Finite Scalar Quantization (FSQ)** 结合。

**FSQ原理**：不使用码本查找，而是将连续向量的每个维度量化到有限个离散值：

```python
def fsq_quantize(x, levels):
    """
    x: shape [B, d]
    levels: 每个维度的量化级数, e.g., [8, 5, 5, 5, 3, 3, 3]
    """
    x_compressed = tanh(x)                    # 压缩到(-1, 1)
    quantized = []
    for i, l in enumerate(levels):
        q_i = round(x_compressed[:, i] * (l-1) / 2)
        q_i = clamp(q_i, -l//2, l//2)
        quantized.append(q_i)
    return stack(quantized)                   # shape: [B, len(levels)]
```

**混合策略**：
- 粗粒度层用Res-KMeans（捕获类别语义）
- 细粒度层用FSQ（捕获属性特征，避免码本冲突）

### 3.4 OneRec：快手统一生成式推荐

- **标题**：[OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment](https://arxiv.org/abs/2502.18965)
- **机构**：快手
- **发表**：arXiv 2025年2月
- **arXiv**：https://arxiv.org/abs/2502.18965
- **使用范式**：**LLM4GRs**（统一检索和排序的生成式推荐）
- **构造范式**：**串行**（RQ-KMeans + 多级别衡）

**整体架构**：

```
┌─────────────────────────────────────────────────────────┐
│                      OneRec 架构                         │
│                                                         │
│  ┌──────────────┐                                       │
│  │ Tokenizer     │  视频 → RQ-KMeans → 语义ID           │
│  │ (离线训练)     │  多级平衡量化机制                      │
│  └──────┬───────┘                                       │
│         ↓ 语义ID                                        │
│  ┌──────────────────────────────────────────┐           │
│  │ Encoder-Decoder (V1) / Lazy Decoder (V2) │           │
│  │                                          │           │
│  │  用户行为序列 → Encoder → 上下文表示       │           │
│  │  上下文表示 → Decoder → 生成语义ID序列     │           │
│  │  (Session-wise生成, 非逐点预测)            │           │
│  └──────────────────────────────────────────┘           │
│         ↓                                               │
│  ┌──────────────────────────────────────────┐           │
│  │ Reward System + DPO对齐                   │           │
│  │  • 模拟用户生成                            │           │
│  │  • 迭代偏好对齐 (Iterative Preference      │           │
│  │    Alignment)                             │           │
│  │  • Direct Preference Optimization (DPO)   │           │
│  └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

**RQ-KMeans在OneRec中的特殊改进 — 多级别衡量化**：

标准RQ-KMeans的一个问题是码本分布可能不均衡。OneRec引入**多级平衡量化机制（Multi-level Balanced Quantitative Mechanism）**：

```python
def balanced_residual_kmeans(embeddings, L, K, balance_weight=α):
    """改进的RQ-KMeans, 增加平衡约束"""
    residuals = embeddings.clone()
    codebooks = []
    for level in range(L):
        # 带平衡约束的K-Means
        # 在聚类损失中加入对聚类大小的正则化
        centroids, labels = balanced_kmeans(
            residuals, k=K,
            balance_weight=α  # 控制聚类大小均衡性
        )
        codebooks.append(centroids)
        quantized = centroids[labels]
        residuals = residuals - quantized
    return codebooks
```

**推理阶段**：
```
1. 语义ID生成 → Beam Search自回归解码
2. 映射回视频 → 可选的Reward-based筛选
```

**OneRec V1 部署效果**：快手主场景，观看时长 +1.6%

**OneRec V2**（arXiv:2508.20900, 2025年8月）：
- 架构升级为**Lazy Decoder-Only**，训练资源减少90%
- 引入Duration-Aware Reward Shaping
- 可扩展到80亿参数
- App Stay Time +0.467%~0.741%

**OneRec-Think**（arXiv:2510.11639, ACL 2026）：
- 统一对话、文本推理和个性化推荐
- 引入Think-Ahead架构
- 后训练使用 **GRPO** + Rollout-Beam Reward（详见pre_and_post_training.md §5.4）
- App Stay Time +0.159%

---

## 4. VQ/PQ方法（向量量化/乘积量化）

**方法分类**：并行语义ID
**核心原理**：VQ将整个向量映射到最近码字；PQ将向量切分为多个子空间，各子空间独立量化，一步并行生成所有token。

### 4.1 VQ原理与公式

**VQ（Vector Quantization，向量量化）** 是最基础的量化方法：将整个 $d$ 维向量直接映射到码本中距离最近的码字。

```
z ∈ R^d → 码本 C = {e₀, e₁, ..., e_{K-1}}  (K个码字, 每个d维)
              ↓
       c = argmin_k ||z - e_k||₂
              ↓
       语义ID = c   ← 单个token (或配合RQ得到多个token)
```

**VQ-VAE的损失函数**（参考[原始论文](https://arxiv.org/abs/1711.00937)）：

$$\mathcal{L}_{\text{VQ}} = \underbrace{\|\mathbf{x} - \hat{\mathbf{x}}\|_2^2}_{\text{重构损失}} + \underbrace{\|\text{sg}(\mathbf{z}) - \mathbf{e}\|_2^2}_{\text{码本损失}} + \underbrace{\beta \|\mathbf{z} - \text{sg}(\mathbf{e})\|_2^2}_{\text{承诺损失}}$$

**代码示例**（参考QARM中的VQ实现）：

```python
# 输入: z shape [B, d], 码本 codebook shape [K, d]

# 1. 计算距离
dist = cdist(z, codebook)             # shape: [B, K]

# 2. 找最近码字
codes = dist.argmin(dim=-1)            # shape: [B]
quantized = codebook[codes]            # shape: [B, d]

# 3. 损失计算
recon_loss = MSE(decoder(quantized), x)                        # 重构损失
codebook_loss = MSE(sg(z), quantized)                          # 码本损失: 码字靠近encoder输出
commitment_loss = MSE(z, sg(quantized))                        # 承诺损失: encoder输出靠近码字

loss = recon_loss + codebook_loss + β * commitment_loss        # β通常=0.25

# 4. 前向传播用STE (Straight-Through Estimator)
# forward:  output = quantized (离散码字)
# backward: gradient直接复制到z (跳过argmin)
output = z + sg(quantized - z)        # 前向=quantized, 反向梯度=z
```

**VQ的局限**：单个VQ只能产生1个token（$K$ 种可能），表达能力有限。因此在推荐中通常配合以下策略使用：
- **RQ（残差量化）**：多层VQ逐层量化残差 → RQ-VAE（TIGER等）
- **PQ（乘积量化）**：多个子空间各自VQ → 并行多token（VQ-Rec等）
- **VQ-Fusion**：多任务VQ融合多源信号 → 单一语义ID（SIDE等）

### 4.2 PQ原理与公式

**PQ（Product Quantization，乘积量化）** 将 $d$ 维向量**切分**为 $M$ 个子空间，每个子空间独立量化：

```
z ∈ R^d = [z^(1); z^(2); ...; z^(M)]   (M个子向量, 每个d/M维)
              ↓         ↓                ↓
          c₁=NN(z¹)  c₂=NN(z²)  ...  c_M=NN(z^M)

语义ID = (c₁, c₂, ..., c_M)   ← 并行, 无依赖
```

```python
# z shape: [B, d=256], M=8, K=256
sub_vectors = z.reshape(B, M, d//M)     # shape: [B, 8, 32]

codes = []
for m in range(M):
    sub_v = sub_vectors[:, m, :]         # shape: [B, 32]
    dist = cdist(sub_v, codebook_m)      # shape: [B, K]
    c_m = dist.argmin(dim=-1)            # shape: [B]
    codes.append(c_m)

# 最终: codes shape [B, M=8], 每个值在[0, K-1]
```

**PQ vs RQ 的核心区别**：

| 维度 | PQ | RQ |
|------|-----|-----|
| 量化方式 | 子空间并行量化 | 残差逐层量化 |
| 生成方式 | 并行 | 串行 |
| 语义层次 | 无（各子空间独立） | 有（粗→细） |
| 表达能力 | $K^M$（各子空间独立） | $K^L$（逐层递进） |
| 推理速度 | 快（一步生成） | 慢（L步生成） |

### 采用VQ/PQ方法的论文

| 论文 | 使用范式 | 生成方式 | 核心改进 |
|------|----------|----------|----------|
| [VQ-Rec](https://arxiv.org/abs/2210.12316) (2022) | LLM4DLRMs/预训练 | 并行(PQ) | 可迁移序列推荐 |
| [SIDE](https://arxiv.org/abs/2506.16698) (Meta, 2025) | LLM4DLRMs | 并行(VQ) | 无参数SID转换，广告序列 |
| [MMQ](https://arxiv.org/abs/2508.15281) (2025) | LLM4GRs | 并行(MoE) | 多模态专家混合量化 |

### 4.3 VQ-Rec：可迁移序列推荐

- **标题**：[Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders](https://arxiv.org/abs/2210.12316)
- **作者**：Yupeng Hou, Shanlei Mu, Wayne Xin Zhao 等
- **机构**：中国人民大学
- **发表**：arXiv 2022年10月
- **arXiv**：https://arxiv.org/abs/2210.12316
- **使用范式**：**LLM4DLRMs**（预训练+迁移微调）
- **构造范式**：**并行**（PQ乘积量化）

**核心方法**：
```
物品文本 → Text Encoder → z ∈ R^d → PQ量化 → (c₁, c₂, ..., c_M) → 物品Code
                                                              ↓
                                                     Code Embedding Table
                                                              ↓
                                                      物品表示 → 序列推荐模型
```

**关键创新**：
- 通过PQ生成的物品Code可以跨域迁移
- 增强对比预训练 + 跨域微调
- 解耦文本特征和物品表示

### 4.4 SIDE：Meta广告序列学习

- **标题**：[SIDE: Semantic ID Embedding for effective learning from sequences](https://arxiv.org/abs/2506.16698)
- **机构**：Meta Platforms
- **发表**：KDD Workshop / AdKDD 2025
- **使用范式**：**LLM4DLRMs**
- **构造范式**：**并行**（VQ-Fusion + DPCA）

**三项创新**：

**1. VQ-Fusion（多任务VQ-VAE融合）**：
```
多种内容embedding:
  img_emb ∈ R^{d1}
  text_emb ∈ R^{d2}
  category_pred ∈ R^{d3}
        ↓ concat
  fused ∈ R^{d1+d2+d3}
        ↓ VQ-VAE
  SID = (c₁, c₂, ..., c_M)  # 单一语义ID融合多源信息
```

**2. Discrete-PCA (DPCA)**：通过PCA预处理旋转数据分布，使量化更高效。

**3. Parameter-free SID-to-Embedding**：直接从SID恢复embedding，无需大参数量的查找表。

**效果**：归一化熵增益提升2.4倍，数据占用减少3倍。

### 4.5 MMQ：多模态混合量化

- **标题**：[MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation](https://arxiv.org/abs/2508.15281)
- **发表**：WSDM 2026
- **arXiv**：https://arxiv.org/abs/2508.15281
- **使用范式**：**LLM4GRs**
- **构造范式**：**并行**（MoE混合量化）

**核心创新**：使用**Mixture-of-Experts (MoE)** 架构对多个模态表征动态加权，进行量化：

```
输入embedding → Router → 分配权重
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

每个Expert使用不同的码本，Router学习根据输入分配不同专家的权重，从多个语义视角编码物品。

---

## 5. OPQ方法（优化乘积量化 · 并行）

**方法分类**：并行语义ID
**核心原理**：在PQ之前学习正交旋转矩阵，优化子空间分解，然后并行量化。

### 5.1 OPQ原理与公式

**PQ的问题**：直接切分子空间可能不够好，因为原始特征空间的维度之间可能高度相关。

**OPQ的解决方案**：在PQ之前学习一个**正交旋转矩阵** $\mathbf{R}$，使旋转后的数据更适合子空间切分。

```
原始PQ:    z → 直接切分为M个子空间 → 各子空间独立量化
OPQ:       z → R·z (旋转) → 切分为M个子空间 → 各子空间独立量化
```

**目标函数**：

$$\min_{\mathbf{R}, \{C_m\}} \sum_{i=1}^{N} \left\| \mathbf{R}\mathbf{z}_i - \sum_{m=1}^{M} \mathbf{e}_{c_m^{(i)}}^{(m)} \right\|_2^2$$

约束条件：$\mathbf{R}^T \mathbf{R} = \mathbf{I}$（正交矩阵）

**交替优化算法**：

```python
# 初始化: R = I (单位矩阵)
for iteration in range(num_iters):
    # Step 1: 固定R, 优化码本 (标准PQ训练)
    rotated_data = R @ data              # shape: [N, d]
    for m in range(M):
        sub_data = rotated_data[:, sub_range(m)]  # shape: [N, d/M]
        codebook_m = kmeans(sub_data, k=K)

    # Step 2: 固定码本, 优化旋转矩阵R
    quantized = reconstruct(all_codes, all_codebooks)
    R = procrustes(data, quantized)      # R = U @ V^T (SVD分解)
```

**旋转矩阵的直觉**：$\mathbf{R}$ 将数据旋转到最优方向，使得切分为 $M$ 个子空间后，每个子空间捕获的方差相当，码本利用率最大化。

### 采用OPQ方法的论文

| 论文 | 使用范式 | 生成方式 | 核心改进 |
|------|----------|----------|----------|
| [RPG](https://arxiv.org/abs/2506.05781) (Meta, 2025) | LLM4GRs | 并行 | 一步并行解码，子物品语义约束 |

### 5.2 RPG：Meta并行语义ID生成

- **标题**：[Recommendation with Parallel semantic ID Generation](https://arxiv.org/abs/2506.05781)
- **机构**：Meta
- **发表**：KDD 2025
- **arXiv**：https://arxiv.org/abs/2506.05781
- **代码**：https://github.com/facebookresearch/RPG_KDD2025
- **使用范式**：**LLM4GRs**
- **构造范式**：**并行**（OPQ）

**核心动机**：串行语义ID的推理延迟与层数 $L$ 成正比，工业推荐不可接受。RPG目标：**一步并行生成全部语义ID token**。

**并行生成架构**：

```
用户行为序列 → Transformer Encoder → h ∈ R^d
                                          ↓
  ┌──────────────────────────────────────────┐
  │         并行解码器 (Parallel Decoder)      │
  │                                          │
  │  h → MLP₁ → c₁  ┐                      │
  │  h → MLP₂ → c₂  │                      │
  │  h → MLP₃ → c₃  ├→ 一步生成全部M个token  │
  │  ...              │                      │
  │  h → MLP_M → c_M┘                      │
  └──────────────────────────────────────────┘
                                          ↓
  语义ID = (c₁, c₂, ..., c_M)  (并行, 无依赖)
```

**训练损失**：

$$\mathcal{L}_{\text{RPG}} = \mathcal{L}_{\text{OPQ}} + \lambda \mathcal{L}_{\text{sub-item}}$$

其中 $\mathcal{L}_{\text{sub-item}}$ 是子物品级别的语义损失。

**推理效率对比**：

| 方法 | 生成方式 | 推理步数 | 推理复杂度 |
|------|----------|----------|------------|
| TIGER (RQ-VAE) | 串行 | $L$ 步 | $O(L \times K)$ |
| RPG (OPQ) | 并行 | 1 步 | $O(M \times K)$ |

**挑战**：并行生成的组合空间 $K^M$ 中，"合法"的物品组合占比极小。RPG通过子物品级语义约束和训练时注入组合约束来解决。

---

## 6. RQ-OPQ混合方法（串行+并行）

**方法分类**：串行+并行混合语义ID
**核心原理**：用RQ（串行）捕获层次语义 + 用OPQ（并行）捕获独特细粒度特征。

### 6.1 设计动机

| 方法 | 优势 | 劣势 |
|------|------|------|
| RQ（串行） | 层次语义，粗→细 | 推理慢 |
| OPQ（并行） | 推理快 | 无层次结构 |
| **RQ-OPQ** | **层次语义 + 快速推理** | 设计复杂度更高 |

```
物品特征 → 属性提取 → ┬→ RQ-KMeans: c₁ → c₂ → c₃ (串行, 粗粒度层次)
                      └→ OPQ:       p₁, p₂, ..., p_M (并行, 细粒度特征)

语义ID = [(c₁, c₂, c₃), (p₁, p₂, ..., p_M)]
          串行部分(层次)    并行部分(细粒度)
```

### 采用RQ-OPQ方法的论文

| 论文 | 使用范式 | 生成方式 | 核心改进 |
|------|----------|----------|----------|
| [OneSearch](https://arxiv.org/abs/2509.03236) (快手, 2025) | LLM4GRs | 串行+并行混合 | KHQE模块，电商搜索 |

### 6.2 OneSearch：快手电商搜索

- **标题**：[OneSearch: A Preliminary Exploration of the Unified End-to-End Generative Framework for E-commerce Search](https://arxiv.org/abs/2509.03236)
- **机构**：快手
- **发表**：arXiv 2025年
- **arXiv**：https://arxiv.org/abs/2509.03236
- **使用范式**：**LLM4GRs**（端到端电商搜索生成式框架）
- **构造范式**：**串行+并行混合**（RQ-OPQ）

**KHQE模块架构**：

```
┌─────────────────────────────────────────────────────────┐
│                   KHQE 模块                              │
│                                                         │
│  输入: 物品多模态特征 + 关键词特征                         │
│                                                         │
│  ┌─────────────────────────┐                            │
│  │ 关键词增强语义协同编码器   │                            │
│  │ 提取物品核心属性特征       │                            │
│  └────────────┬────────────┘                            │
│               ↓                                         │
│  ┌────────────┴────────────────────────┐                │
│  │                                      │                │
│  │  ┌───────────────────┐  ┌──────────┐ │                │
│  │  │ RQ-KMeans          │  │ OPQ      │ │                │
│  │  │ (层次特征编码)       │  │ (独特特征│ │                │
│  │  │                    │  │  量化)   │ │                │
│  │  │ c₁ → c₂ → c₃      │  │ p₁,p₂,p₃│ │                │
│  │  │ (串行, 粗→细)       │  │ (并行)   │ │                │
│  │  └───────────────────┘  └──────────┘ │                │
│  └──────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
```

**推理时的混合生成**：
```
用户query + 行为序列 → Transformer → 生成:
  Step 1-3: 自回归生成RQ部分 (c₁ → c₂ → c₃)   ← 层次剪枝
  Step 4:   一步并行生成OPQ部分 (p₁, p₂, ..., p_M) ← 快速解码
```

**实验效果**（快手商城搜索A/B测试）：
- 物品CTR +1.45%，PV CTR +1.40%
- 运营支出降低75.40%
- OneSearch V2（2026年）：物品CTR +3.98%，GMV +3.45%

---

## 7. 语义ID对齐方法

**核心问题**：语义ID从内容特征量化而来，只包含**内容语义**（物品是什么），不包含**协同过滤信号**（用户-物品交互模式）。这导致语义ID与推荐目标之间存在**语义-行为gap（Semantic-Behavioral Gap）**——两个内容相似的物品可能有截然不同的用户交互模式，反之亦然。

对齐（Alignment）的目标是弥合这一gap，使语义ID不仅反映物品内容相似性，还能反映用户行为相似性。

### 采用对齐方法的论文总览

| 论文 | 对齐时机 | 对齐方法 | 核心改进 |
|------|----------|---------|----------|
| [QARM](#72-qarm快手前置对齐) (快手, 2024) | Pre-alignment | MLLM微调+Batch对比损失 | 多模态表示与业务交互对齐 |
| [DAS](#73-das快手一阶段联合对齐) (快手, 2025) | Joint (一阶段) | 多视角对比+CF去偏 | 去偏CF信号+六路对比对齐 |
| [PLUM SID-v2](#74-plum-sid-v2共现对比对齐) (Google, 2025) | Joint (量化中) | 共现对比损失 | 轻量级单损失注入协同信号 |
| [MMQ](#75-mmq后置行为感知微调) (阿里, 2025) | Post-alignment | 动态码本调整 | 推荐梯度微调码本聚类 |
| [UniSID](#76-unisid端到端联合优化) (2026) | End-to-end | 多粒度对比学习 | embedding和SID联合优化 |
| [DIGER](#77-diger可微分对齐) (SIGIR 2026) | End-to-end | Gumbel-Softmax可微量化 | 推荐梯度直接优化码本 |
| TIGER, OneRec | 无显式对齐 | — | 纯内容量化，CF信号在后续预训练中学 |

#### 对齐时机分类

```
语义ID对齐的四种时机:

  Pre-alignment (量化前对齐):
    先对齐多模态Encoder → 再量化对齐后的embedding
    代表: QARM (MLLM微调)
    
  Joint alignment (量化中联合对齐):
    对齐损失作为量化训练的一部分，联合优化
    代表: DAS (一阶段), PLUM SID-v2 (共现对比损失)
    
  Post-alignment (量化后对齐):
    先完成内容量化 → 再用推荐信号微调码本
    代表: MMQ Stage 2 (行为感知微调)
    
  End-to-end (端到端):
    不分离量化和对齐，推荐目标直接驱动SID生成
    代表: UniSID (联合优化), DIGER (可微量化)
```

### 7.1 为什么需要对齐？

```
问题: 语义-行为gap

  内容空间:                行为空间:
  猫视频A ←相似→ 猫视频B   用户常看A后看C(搞笑狗)
  猫视频A ←相似→ 猫视频D   用户从不看A后看B
  
  纯内容SID:              理想SID:
  A, B, D 共享前缀        A, C 应共享前缀
  A, C 前缀不同           B, D 可以有不同前缀
  
  → 对齐就是让SID同时反映"内容相似"和"行为相似"
```

### 7.2 QARM：快手前置对齐

- **论文**：[QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou](https://arxiv.org/abs/2411.11739)
- **对齐时机**：**Pre-alignment**（量化前对齐多模态Encoder）

**解决的两个核心问题**：
1. **Representation Unmatching**：预训练多模态模型（如CLIP）由通用NLP/CV任务监督，而推荐模型由用户-物品交互监督，两者的表示空间不一致
2. **Representation Unlearning**：多模态表示通常作为冻结缓存输入，无法被推荐模型的梯度更新

**对齐方法**：用业务交互数据微调多模态大语言模型（MLLM）

```
QARM 前置对齐流程:

  Step 1: 构造对齐训练数据
    ├── User2Item Retrieval: 用户点击的trigger→target物品对
    │   (trigger = 用户最近50次点击中与target最相似的物品)
    └── Item2Item Retrieval: 已有检索模型(Swing)的稳定相似物品对
    
  Step 2: MLLM微调 (对齐)
    ├── M_trigger = MLLM(text, audio, image of trigger)
    ├── M_target  = MLLM(text, audio, image of target)
    └── L_align = Batch-Contrastive(M_trigger, M_target, B)
        → 鼓励行为相似的trigger-target对具有相近的多模态表示
        → 推开batch内不相关的物品
        
  Step 3: 量化 (使用对齐后的表示)
    └── 对齐后的embedding → RQ-KMeans → SID
```

**关键特点**：利用已有检索模型（如Swing）的知识来监督对齐，将协同过滤信号注入多模态Encoder，然后对对齐后的表示做量化。

### 7.3 DAS：快手一阶段联合对齐

- **论文**：[DAS: Dual-Aligned Semantic IDs Empowered Industrial Recommender System](https://arxiv.org/abs/2508.10584)
- **对齐时机**：**Joint alignment**（量化、CF建模、对齐三者一阶段联合训练）
- **参考讲解**：https://zhuanlan.zhihu.com/p/1943035654511494581

**DAS架构**：

```
┌──────────────────────────────────────────────────────────┐
│                    DAS 一阶段联合对齐                      │
│                                                          │
│  ┌─────────────────┐  ┌─────────────────────┐           │
│  │ User Semantic    │  │ Item Semantic        │           │
│  │ Model (UISM)     │  │ Model (UISM)         │           │
│  │ PLM→RQ-VAE量化   │  │ PLM→RQ-VAE量化       │           │
│  │ z_u (用户SID)    │  │ z_i (物品SID)        │           │
│  └────────┬────────┘  └──────────┬───────────┘           │
│           ↓                      ↓                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │    ICDM: ID-based CF Debias Module                │   │
│  │    协同过滤去偏模块                                 │   │
│  │                                                   │   │
│  │    用户侧: c_u = MLP(c_u^int ⊕ c_u^con)          │   │
│  │      c_u^int = 无偏用户兴趣                        │   │
│  │      c_u^con = 用户从众偏差                        │   │
│  │                                                   │   │
│  │    物品侧: c_i = MLP(c_i^pro ⊕ c_i^pop)          │   │
│  │      c_i^pro = 无偏物品内容                        │   │
│  │      c_i^pop = 物品流行度偏差                      │   │
│  │                                                   │   │
│  │    去偏约束: 正交损失 (兴趣⊥从众, 内容⊥流行度)     │   │
│  └──────────────────────┬───────────────────────────┘   │
│                         ↓                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │    MDAM: Multi-View Contrastive Alignment         │   │
│  │    多视角对比对齐 (六路对比损失)                     │   │
│  │                                                   │   │
│  │    ① Dual U2I: z_u ↔ c_i^pro  (用户SID↔物品去偏CF) │   │
│  │    ② Dual U2I: c_u^int ↔ z_i  (用户去偏CF↔物品SID) │   │
│  │    ③ Dual U2U: z_u ↔ c_u^int  (batch内用户侧)     │   │
│  │    ④ Dual I2I: z_i ↔ c_i^pro  (batch内物品侧)     │   │
│  │    ⑤ Dual Co-occur U2U: 共现用户SID互对齐          │   │
│  │    ⑥ Dual Co-occur I2I: 共现物品SID互对齐          │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
│  总损失: L_All = L_Sem + α·L_CF + β·L_Align             │
│    L_Sem: 重构+RQ-VAE码本+承诺损失                       │
│    L_CF:  有偏+无偏CF损失+去偏正交约束                    │
│    L_Align: 六路对比对齐损失之和                          │
│    α=1, β=0.5                                            │
└──────────────────────────────────────────────────────────┘
```

**CF去偏模块（ICDM）的关键创新**：

原始CF信号包含流行度偏差（热门物品获得更多交互）和从众偏差（用户倾向跟随大众行为）。如果直接用原始CF信号对齐，SID会被这些偏差污染。ICDM将CF表示分解为**无偏分量**和**偏差分量**，对齐时只使用无偏分量（$c_u^{int}$ 和 $c_i^{pro}$）。

**关键特点**：
- **一阶段训练**：量化、CF建模、对齐三者端到端联合优化，避免两阶段信息损失
- **去偏对齐**：唯一显式对CF信号去偏的SID对齐方法
- **灵活性**：兼容各种量化方法（RQ-VAE、Res-KMeans等）
- 已在快手广告系统部署，服务4亿日活用户

### 7.4 PLUM SID-v2：共现对比对齐

- **论文**：[PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations](https://arxiv.org/abs/2510.07784)
- **对齐时机**：**Joint alignment**（共现对比损失作为RQ-VAE训练的一部分）

**对齐方法**：在RQ-VAE训练损失中添加共现对比损失（Co-occurrence Contrastive Loss）

$$\mathcal{L}_{\text{SID-v2}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{RQ}} + \mathcal{L}_{\text{con}}$$

其中共现对比损失：

$$\mathcal{L}_{\text{con}} = -\sum_{i=1}^{2N_b} \log \frac{\exp(\text{sim}(\mathbf{p}_i, \mathbf{p}_{i^+}))}{\sum_{j=1}^{2N_b} \exp(\text{sim}(\mathbf{p}_i, \mathbf{p}_j))}$$

$i^+$ 表示在用户观看历史中与物品 $i$ 共现的物品。

**与DAS对齐的对比**：
- PLUM只在物品级别做共现对比，DAS有更丰富的多视角对齐（六路）
- PLUM不对CF信号做去偏，DAS显式分离流行度/从众偏差
- PLUM更轻量（单损失项），DAS需要独立的CF模型+去偏模块

**效果**：SID唯一性从94.0%（SID-v1无对齐）提升到96.7%，Recall@10从12.3%提升到14.4%。

### 7.5 MMQ：后置行为感知微调

- **论文**：[MMQ: Multimodal Mixture-of-Quantization Tokenization](https://arxiv.org/abs/2508.15281)
- **对齐时机**：**Post-alignment**（先训练内容量化器，再用推荐信号微调码本）

```
MMQ 两阶段对齐:

  Stage 1: 多模态Tokenizer训练 (内容量化)
    ├── MoE多专家量化 (模态特定Expert+共享Expert)
    └── 输出: 基于内容的SID
    
  Stage 2: 行为感知微调 (对齐)
    ├── 不冻结码本 → 推荐损失(NTP/BCE)梯度回传
    ├── 动态调整码本聚类中心
    │   → 行为相似的物品(即使内容不同)被映射到相近的聚类
    └── 同时保持多模态重构损失 → 防止模态信息在对齐中丢失
```

### 7.6 UniSID：端到端联合优化

- **论文**：[End-to-End Semantic ID Generation](https://arxiv.org/abs/2602.10445)
- **对齐时机**：**End-to-end**（embedding和SID联合优化，无独立量化阶段）

**对齐方法**：多粒度对比学习——在每个SID层级 $l$，构造不同粒度的正例集 $P_l$：粗粒度层用宽泛的相似性标准（同顶级类目），细粒度层用严格标准（同子类目）。

### 7.7 DIGER：可微分对齐

- **论文**：[Differentiable Semantic ID for Generative Recommendation](https://arxiv.org/abs/2601.19711)
- **对齐时机**：**End-to-end**（推荐损失通过可微量化直接优化码本）

**对齐方法**：Gumbel-Softmax使量化操作可微，推荐损失的梯度通过软概率传播到码本。总损失 $L = L_{gen} + L_{vq} + L_{recon}$，其中 $L_{gen}$（自回归next-SID预测）是主要驱动力。

**码本坍塌缓解**：不确定性衰减策略（SDUD和FrqUD），随训练进行自动降低Gumbel噪声。

**关键特点**：最彻底的对齐形式——推荐目标直接塑造码本，无需显式对齐步骤。

### 7.8 无显式对齐的方法

- **TIGER**：纯内容RQ-VAE量化，无对齐
- **OneRec**：RQ-KMeans纯内容量化+多级别衡，无CF信号注入。OneRec的"Itemic-Text Alignment"（Stage 1预训练）是SID-to-LLM语言空间的对齐（让SID token在LLM词表中找到正确位置），**不是**内容-to-CF的对齐

这些方法依赖后续预训练阶段（行为序列的NTP训练）隐式学习协同信号。

### 7.9 对齐方法对比总结

| 维度 | QARM | DAS | PLUM SID-v2 | MMQ | UniSID | DIGER |
|------|------|-----|-------------|-----|--------|-------|
| **对齐时机** | Pre | Joint | Joint | Post | E2E | E2E |
| **对齐复杂度** | 中 | 高 | 低 | 中 | 高 | 高 |
| **CF去偏** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **对齐信号来源** | 业务物品对 | CF模型+交互 | 用户共现序列 | 推荐loss梯度 | 多粒度对比 | 推荐loss梯度 |
| **是否需独立CF模型** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **码本是否动态调整** | ❌ | ❌ | ❌ | ✅ | N/A | ✅ |

**演进趋势**：从两阶段分离（无对齐/Pre-alignment）走向端到端联合优化（DIGER/UniSID），使SID天然包含协同过滤信号。

---

## 8. 稀疏+稠密级联方法

**核心问题**：纯语义ID（稀疏）存在信息损失，纯稠密向量检索无法利用语义层次结构。

### 采用稀疏+稠密方法的论文

| 论文 | 使用范式 | 核心改进 |
|------|----------|----------|
| [COBRA](https://arxiv.org/abs/2503.02453) (百度, 2025) | LLM4GRs | 级联稀疏-稠密表示，BeamFusion |

### 8.1 COBRA：百度级联表示

- **标题**：[Unified Generative Recommendations with Cascaded Sparse-Dense Representations](https://arxiv.org/abs/2503.02453)
- **机构**：百度
- **发表**：arXiv 2025年3月
- **使用范式**：**LLM4GRs**
- **构造范式**：**级联**（稀疏语义ID → 稠密向量）

**核心流程**：

```
生成过程:
  Step 1: 生成稀疏语义ID → 粗粒度检索 (自回归beam search)
  Step 2: 以语义ID为条件 → 生成稠密向量 → 精粒度排序

推理: BeamFusion = Beam Search + 最近邻分数
```

**端到端训练**：允许稠密表示在训练中被动态精化，同时捕获语义信息和协同信号。

**在线A/B测试**（2亿日活广告平台）：关键指标显著提升。

---

## 9. 端到端语义ID生成与LLM适配

**核心问题**：传统两阶段方法（先量化生成语义ID，再训练推荐模型）存在**目标不一致**问题。本节包含两类方法：
- **端到端语义ID生成**（UniSID）：SID构造与推荐模型联合优化
- **LLM适配生成式推荐**（PLUM）：离线SID构造 + 预训练LLM适配（SID生成本身是离线的，但整体框架将SID纳入LLM词表进行端到端预训练）

### 9.1 腾讯UniSID：广告推荐端到端生成

- **标题**：[End-to-End Semantic ID Generation for Generative Advertisement Recommendation](https://arxiv.org/abs/2602.10445)
- **机构**：腾讯 + 武汉大学
- **发表**：arXiv 2026年2月
- **使用范式**：**LLM4GRs**
- **构造范式**：**端到端**（联合优化embedding和语义ID）

**核心创新**：
- **端到端优化**：embedding和语义ID联合学习（取代传统两阶段：先训RQ-VAE tokenizer，再训推荐模型）
- **多粒度对比学习**：在不同SID层级对齐不同粒度的语义
- **广告增强输入模式**：将异构广告信号线性化为统一token序列
- **摘要式广告重构**：鼓励SID捕获高层语义

**关于"端到端"的重要澄清**：

UniSID的"端到端"指的是**embedding生成和SID生成的端到端联合优化**（从原始广告数据同时产出embedding+SID），**并非**从tokenizer一直到最终推荐的全链路端到端。具体来说：

```
UniSID的"端到端"范围:
  原始广告数据 (指令/图片/文本/属性)
    ↓ MLLM (Qwen2.5-VL-3B)
    ↓ 双投影头
  ┌─────────────────────────┐
  │ SID生成 + Embedding生成  │  ← 这部分是联合优化的
  └─────────────────────────┘
    ↓                           ↓
  SID tokens              Embeddings
    ↓                           ↓
  ┌──────────────────────────────────────────┐
  │    下游推荐模型 (如TIGER框架)              │  ← 仍需独立的下游模型
  │    用SID+embedding作为输入做生成式推荐     │
  └──────────────────────────────────────────┘
```

UniSID本身是一个**tokenizer/embedding生成框架**，不是完整的推荐系统。论文中明确提到"integrate UniSID into the TIGER framework by replacing its original RQ-VAE SID generation module"。

与DIGER的对比：DIGER通过可微量化让推荐梯度直接回传到码本，实现从tokenizer到推荐的真正端到端；UniSID则只是消除了embedding和SID之间的两阶段分离，但SID产出后仍需下游推荐模型使用。

**效果**：Hit Rate比最强baseline提升4.62%。

### 9.2 PLUM：LLM适配生成式推荐

- **标题**：[PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations](https://arxiv.org/abs/2510.07784)
- **发表**：arXiv 2025年10月
- **使用范式**：**LLM4GRs**
- **构造范式**：**串行**（SID-v2，离线RQ-VAE量化）

> **注意**：PLUM的SID-v2是独立于LLM的离线工序（先量化得到SID，再扩展LLM词表进行CPT），**不属于端到端语义ID生成**。此处归类仅因其LLM适配的整体框架涉及SID构造。

**核心流程**：
1. 物品Tokenization：使用SID-v2将物品离散化（离线）
2. 持续预训练（CPT）：扩展LLM词汇表以包含SID token
3. 任务微调：Reward-weighted SFT

#### 9.2.1 SID-v2：增强型语义ID

SID-v2是PLUM对传统语义ID（SID-v1）的改进版，核心创新在于**多分辨率码本**和**渐进掩码**机制。

**多模态融合**：

```
┌───────────────────────────────────────────────────────────┐
│                   SID-v2 多模态融合                        │
│                                                           │
│  物品 i 的 M 个模态 embedding: {x_m}_{m=1}^M              │
│                                                           │
│  x₁ (title emb)    → Encoder ℰ₁ → z₁ ∈ R^d             │
│  x₂ (desc emb)     → Encoder ℰ₂ → z₂ ∈ R^d             │
│  x₃ (ASR emb)      → Encoder ℰ₃ → z₃ ∈ R^d             │
│  x₄ (channel emb)  → Encoder ℰ₄ → z₄ ∈ R^d             │
│  ...                                                      │
│           ↓ concat + project                              │
│  z = Proj([z₁; z₂; ...; z_M]) ∈ R^d  ← 统一特征向量     │
│           ↓                                               │
│  RQ-VAE 残差量化 → SID = (c₁, c₂, ..., c_L)             │
└───────────────────────────────────────────────────────────┘
```

每个模态有独立的 Encoder $\mathcal{E}_m$，编码后拼接并投影为统一特征向量 $\mathbf{z}$。

**创新1：多分辨率码本（Multi-Resolution Codebooks）**

传统RQ-VAE使用统一的码本大小（每层K相同），SID-v2改为**逐层递减的码本分辨率**：

$$K_l = \frac{2048}{2^{l-1}}$$

即第1层码本大小2048，第2层1024，第3层512，以此类推。

```
传统 RQ-VAE (SID-v1):
  Level 1: K=2048  → c₁ ∈ {0,...,2047}
  Level 2: K=2048  → c₂ ∈ {0,...,2047}
  Level 3: K=2048  → c₃ ∈ {0,...,2047}

SID-v2 (Multi-Resolution):
  Level 1: K=2048  → c₁ ∈ {0,...,2047}  ← 高分辨率，最大区分度
  Level 2: K=1024  → c₂ ∈ {0,...,1023}  ← 中分辨率，编码中等残差
  Level 3: K=512   → c₃ ∈ {0,...,511}   ← 低分辨率，编码低熵残差
```

**设计动机**：第1层负责最粗粒度的语义区分（需要大码本），后续层编码的是逐层递减的残差（熵更低，小码本即可）。这与信息论中的逐层细化编码思想一致。

**创新2：渐进掩码（Progressive Masking）**

在RQ-VAE训练过程中，引入随机掩码机制以强化层次结构的语义一致性：

$$m_l = \mathbb{1}_{l < r}, \quad r \sim \text{Uniform}(1, L)$$

量化向量变为：$\hat{\mathbf{z}} = \sum_{l=1}^{L} m_l \cdot \mathbf{e}_{c_l}^{(l)}$

```
训练时随机选择截断层 r:
  r=1: ẑ = e₁*           (只用第1层码字)
  r=2: ẑ = e₁* + e₂*     (用前2层码字)
  r=3: ẑ = e₁* + e₂* + e₃* (用全部3层码字)
  
效果: 迫使每层独立编码有意义的语义，而非依赖后续层补偿
     → 更严格的层次语义层次结构
```

**训练损失（三部分）**：

$$\mathcal{L}_{\text{SID-v2}} = \mathcal{L}_{\text{recon}} + \mathcal{L}_{\text{RQ}} + \mathcal{L}_{\text{con}}$$

- **重构损失**：$\mathcal{L}_{\text{recon}} = \sum_{m=1}^{M} \| \mathbf{x}_m - \hat{\mathbf{x}}_m \|^2$（多模态重构）
- **RQ损失**：$\mathcal{L}_{\text{RQ}} = \sum_{l=1}^{L} \left[ \beta \| \mathbf{r}_l - \text{sg}(\mathbf{e}_{c_l}^{(l)}) \|^2 + \| \text{sg}(\mathbf{r}_l) - \mathbf{e}_{c_l}^{(l)} \|^2 \right]$（码本+承诺损失）
- **共现对比损失**（Co-occurrence Contrastive Loss）：

$$\mathcal{L}_{\text{con}} = -\sum_{i=1}^{2N_b} \log \frac{\exp(\text{sim}(\mathbf{p}_i, \mathbf{p}_{i^+}))}{\sum_{j=1}^{2N_b} \exp(\text{sim}(\mathbf{p}_i, \mathbf{p}_j))}$$

其中 $i^+$ 是与 $i$ 在用户行为序列中共现的物品。该损失鼓励频繁共现的物品获得相似的SID表示，不共现的物品被推远。这是SID-v2相对于SID-v1的重要改进——**将协同过滤信号注入语义ID**。

**SID-v2 vs SID-v1 总结**：

| 维度 | SID-v1 | SID-v2 |
|------|--------|--------|
| 码本分辨率 | 统一（每层K相同） | 多分辨率（$K_l = 2048/2^{l-1}$） |
| 层次约束 | 无显式约束 | 渐进掩码（Progressive Masking） |
| 协同信号 | 无 | 共现对比损失注入 |
| 多模态融合 | 简单拼接 | 独立Encoder + 拼接投影 |

#### 9.2.2 PLUM 的LLM适配流程

```
PLUM 整体流程:

  Stage 1: SID-v2 Item Tokenization (离线)
    ├── 多模态内容 → Encoder融合 → RQ-VAE量化 → SID tokens
    ├── 融合title/description/ASR/channel等多源信息
    └── 独立于LLM的离线工序（码本固定后不再更新）
  
  Stage 2: Continued Pre-Training (CPT)
    ├── 预训练LLM (Gemini家族) → 扩展词表(加入SID tokens)
    ├── 混合数据CPT (详见pre_and_post_training.md §3.4)
    └── 目标: SID token在LLM语言空间中获得语义锚定
  
  Stage 3: Task-Specific SFT
    ├── Reward-weighted采样训练
    └── Beam search推理, hallucination率 < 5%
```

**YouTube Shorts实验**：Panel CTR +4.96%。

### 9.3 DIGER：可微分语义ID端到端联合优化

- **标题**：[Differentiable Semantic ID for Generative Recommendation](https://arxiv.org/abs/2601.19711)
- **发表**：SIGIR 2026
- **使用范式**：**LLM4GRs**
- **构造范式**：**端到端**（可微分量化，推荐梯度直接优化码本）

> **与UniSID的区别**：UniSID的"端到端"是embedding和SID的联合生成，SID产出后仍需下游推荐模型；DIGER通过可微量化让推荐梯度直接回传到码本，实现从量化器到推荐模型的真正端到端优化。

**核心问题**：传统两阶段方法中，SID是冻结的，推荐模型的梯度无法回传到量化器，导致**目标不一致**——SID为内容重构优化，推荐模型为next-item prediction优化，两者从未协调。

**解决方案**：Gumbel-Softmax可微分量化

```
DIGER 端到端架构:

  前向传播 (硬分配, 产出离散SID):
    logits_l = sim(r_{l-1}, codebook_l)      ← 残差与码本的相似度
    g ~ Gumbel(0, 1)                          ← Gumbel噪声
    c_l = argmax(logits_l + g)                ← 离散SID token
    
  反向传播 (软更新, 梯度流到码本):
    y_soft = softmax((logits + g) / τ)        ← 软概率
    ē = Σ y_soft_i · e_i                      ← 加权码本embedding
    → 推荐损失梯度通过y_soft传播到码本参数

  总损失: L = L_gen + L_vq + L_recon
    L_gen: 自回归next-SID预测 (主要驱动力, 推荐目标)
    L_vq:  RQ-VAE量化损失 (稳定码本)
    L_recon: 重构损失 (保持内容语义)
```

**码本坍塌缓解**——不确定性衰减策略：
- **SDUD**（标准差衰减）：随 $L_{gen}$ 降低，Gumbel噪声尺度自动缩小，从探索过渡到利用
- **FrqUD**（频率衰减）：仅对高频码字施加噪声，低频码字做确定性分配

**关键特点**：
- 最彻底的对齐形式——推荐目标直接塑造码本，无需显式对齐步骤
- 固定 $\tau=2.0$（不做温度退火），因为SID在推理时必须离散
- Recall@10比两阶段baseline提升13.4%（Amazon Beauty数据集）

## 10. 工业基准

### 10.1 FORGE：淘宝大规模基准

- **标题**：[FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets](https://arxiv.org/abs/2509.20904)
- **机构**：阿里巴巴（淘宝）
- **发表**：arXiv 2025年9月
- **使用范式**：**LLM4GRs**（基准评测）

**核心贡献**：
- 首个大规模工业级语义ID基准数据集
- 数据规模：140亿用户交互，2.5亿物品
- 比现有数据集大：100倍交互、26倍用户、3倍物品
- 优化的SID构造策略使淘宝交易量提升0.35%

---

## 11. 多模态语义ID构造

### 11.1 问题背景

当物品的语义信息涉及**多种模态**（如文本、图片、音频、视频等）时，如何将这些模态的embedding融合后量化为语义ID，是一个关键问题。

**朴素方法：直接拼接（Concat）**

最简单的做法是将各模态embedding直接拼接后做量化：

```python
# 朴素拼接
text_emb: shape [B, d_text]        # e.g., d_text = 768
img_emb:  shape [B, d_img]         # e.g., d_img = 512

concat_emb = torch.cat([text_emb, img_emb], dim=-1)  # shape: [B, 1280]

# 直接对拼接向量做RQ-VAE量化
semantic_id = rq_vae(concat_emb)   # shape: [B, L]
```

**朴素方法的问题**：
- **模态主导**：某一模态（如文本）的embedding方差可能远大于另一模态（如图片），导致量化过程被单一模态主导
- **模态冲突**：不同模态的语义空间可能不一致（如图片描述外观，文本描述功能），直接拼接导致语义模糊
- **码本冲突加剧**：语义空间混乱使得不同物品可能被量化到相同的码字
- **忽略模态独特性**：每个模态的独特信息在拼接后被稀释

### 11.2 多模态语义ID构造方法分类

| 方法 | 代表论文 | 核心思路 | 优势 |
|------|----------|----------|------|
| **直接拼接** | TIGER, 早期工作 | concat所有模态 → 统一量化 | 简单，实现方便 |
| **跨模态量化** | MACRec | 模态间交叉注意力 → 联合量化 | 捕获模态交互 |
| **MoE混合量化** | MMQ (阿里) | 多专家路由 → 模态分离/共享 | 兼顾模态独特性与协同性 |
| **多模态融合VQ** | SIDE (Meta) | VQ-Fusion多任务 | 融合多种内容信号 |
| **行为感知微调** | MMQ, DAS | 量化后用行为数据微调 | 弥合语义-行为gap |

### 11.3 MMQ：阿里MoE混合量化（WWW 2025）

- **标题**：[MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation](https://arxiv.org/abs/2508.15281)
- **机构**：阿里巴巴
- **发表**：WSDM 2026 / arXiv 2025年2月
- **使用范式**：**LLM4GRs**
- **构造范式**：**并行**（MoE多专家混合量化）

#### 11.3.1 核心思想

MMQ的核心创新在于：**不是简单拼接多模态embedding，而是用Mixture-of-Experts（MoE）架构同时捕获模态间的协同信息和各模态的独特信息**。

```
┌──────────────────────────────────────────────────────────────┐
│                     MMQ 两阶段架构                            │
│                                                              │
│  Stage 1: 多模态共享-特定 Tokenizer 训练                      │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                                                      │    │
│  │  text_emb ──→ Router ──┬→ Expert_text (文本特定码本)  │    │
│  │                        │                              │    │
│  │  img_emb  ──→ Router ──┼→ Expert_img  (图片特定码本)  │    │
│  │                        │                              │    │
│  │  concat   ──→ Router ──┼→ Expert_shared (共享码本)    │    │
│  │                        │                              │    │
│  │            正交正则化 → └→ 各专家输出加权聚合           │    │
│  │                                  ↓                    │    │
│  │                          多模态语义ID                   │    │
│  └──────────────────────────────────────────────────────┘    │
│                                                              │
│  Stage 2: 行为感知微调                                        │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  语义ID + 推荐目标 → 动态调整码本聚类                   │    │
│  │  + 多模态重构损失（保持模态信息不丢失）                  │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

#### 11.3.2 多专家架构详解

**Expert设计**：
- **Modality-Specific Experts**（模态特定专家）：每个模态有专属的量化专家和码本，捕获该模态的独特语义
- **Modality-Shared Experts**（模态共享专家）：跨模态的共享专家，捕获模态间的协同语义

**Router机制**：

```python
# 输入: text_emb [B, d_text], img_emb [B, d_img]

# 1. 各模态经过各自的专家
text_out = expert_text(text_emb)     # shape: [B, d_hidden]
img_out  = expert_img(img_emb)       # shape: [B, d_hidden]

# 2. 拼接后经过共享专家
shared_input = cat([text_emb, img_emb])  # shape: [B, d_text + d_img]
shared_out = expert_shared(shared_input)  # shape: [B, d_hidden]

# 3. Router分配权重
router_logits = router(cat([text_emb, img_emb]))  # shape: [B, 3]
router_weights = softmax(router_logits)            # shape: [B, 3]

# 4. 加权聚合
output = (router_weights[:, 0:1] * text_out +
          router_weights[:, 1:2] * img_out +
          router_weights[:, 2:3] * shared_out)      # shape: [B, d_hidden]

# 5. 对聚合输出做量化
semantic_id = quantize(output)                      # shape: [B, L]
```

#### 11.3.3 正交正则化

为了防止所有Expert退化为相同的行为（expert collapse），MMQ引入**正交正则化**：

$$\mathcal{L}_{\text{orth}} = \| \mathbf{W}_i^T \mathbf{W}_j \|_F^2 \quad (i \neq j)$$

其中 $\mathbf{W}_i$, $\mathbf{W}_j$ 是不同Expert的权重矩阵。正交约束迫使各Expert学习不同的表示子空间，确保模态特定专家和共享专家各司其职。

**总损失**：

$$\mathcal{L}_{\text{MMQ}} = \mathcal{L}_{\text{quant}} + \lambda_1 \mathcal{L}_{\text{recon}} + \lambda_2 \mathcal{L}_{\text{orth}}$$

其中 $\mathcal{L}_{\text{quant}}$ 是量化损失（码本+承诺），$\mathcal{L}_{\text{recon}}$ 是重构损失，$\mathcal{L}_{\text{orth}}$ 是正交正则化。

#### 11.3.4 行为感知微调（Stage 2）

第一阶段训练的语义ID仅基于内容语义，与推荐目标存在gap。MMQ在第二阶段用下游推荐任务微调：

```
语义ID (冻结码本) + 用户行为数据
    ↓
推荐损失 (NTP / BCE) 反向传播
    ↓
动态调整: 码本中的聚类中心
    ↓
同时保持多模态重构损失 (防止模态信息丢失)
```

**关键创新**：不是简单冻结码本，而是**动态调整语义ID的聚类**，使得行为相似的物品即使内容不同也能获得相近的语义ID。

### 11.4 与其他多模态方法的对比

| 维度 | 直接拼接 | MMQ (MoE) | MACRec (跨模态) | SIDE (VQ-Fusion) |
|------|----------|-----------|----------------|------------------|
| 模态融合方式 | concat | 多专家路由 | 交叉注意力 | 多任务VQ |
| 模态独特性 | ❌ 被稀释 | ✅ 特定Expert保留 | ✅ 模态级对齐 | ✅ 多任务分离 |
| 模态协同性 | ⚠️ 隐式 | ✅ 共享Expert | ✅ 显式交叉 | ⚠️ 隐式 |
| 行为适配 | ❌ | ✅ Stage 2 | ✅ 多维对齐 | ❌ |
| 码本冲突缓解 | ❌ | ✅ 正交正则化 | ✅ 跨模态量化 | ✅ DPCA |
| 计算开销 | 低 | 中 | 中 | 低 |

### 11.5 何时选择何种方式？

```
模态数量少（2~3个）且语义一致 → 直接拼接（简单有效）
模态差异大、需要保留各模态独特性 → MMQ (MoE混合量化)
需要模态间显式交互/对齐 → MACRec (跨模态量化)
需要融合非模态信号（如类别预测） → SIDE (VQ-Fusion)
语义ID需要适配推荐行为 → MMQ Stage 2 / DAS 行为对齐
```

---

## 12. 总结与展望

### 12.1 方法全景分类

#### 按量化方法分类

| 量化方法 | 代表论文 | 使用范式 | 生成方式 | 核心优势 |
|----------|----------|----------|----------|----------|
| **RQ-VAE** | TIGER, YouTube, DIGER | GRs/DLRMs | 串行 | 端到端优化，层次语义 |
| **RQ-KMeans** | QARM, QARM V2, OneRec | DLRMs/GRs | 串行 | 轻量，码本均衡 |
| **VQ/PQ** | VQ-Rec, SIDE | DLRMs | 并行 | 推理快，可迁移 |
| **MoE混合量化** | MMQ (阿里) | GRs | 并行 | 多模态分离/协同，行为适配 |
| **OPQ** | RPG | GRs | 并行 | 优化子空间，一步解码 |
| **RQ-OPQ** | OneSearch | GRs | 混合 | 层次+效率 |
| **双对齐** | DAS | DLRMs/GRs | 兼容多种 | 注入协同信号 |
| **级联** | COBRA | GRs | 级联 | 稀疏+稠密互补 |
| **端到端** | UniSID | GRs | 串行 | 目标一致 |
| **LLM适配** | PLUM | GRs | 串行 | 预训练LLM复用+SID-v2离线量化 |

#### 按解决的问题分类

| 问题 | 解决方法 | 代表论文 |
|------|----------|----------|
| 冷启动泛化 | 语义ID替代随机ID | TIGER, YouTube |
| 多模态对齐 | Item Alignment + 量化 | QARM, QARM V2 |
| **多模态融合** | **MoE混合量化/跨模态量化** | **MMQ, MACRec** |
| 推理延迟 | 并行/混合生成 | RPG, OneSearch |
| 目标不一致 | 可微分/端到端量化 | DIGER, UniSID |
| 码本冲突 | FSQ混合量化 | QARM V2 |
| 存储效率 | 无参数SID转换 | SIDE |
| 信息损失 | 稀疏+稠密级联 | COBRA |
| 缺乏协同信号 | 双对齐/CF注入/共现对比损失 | DAS, PLUM (SID-v2) |
| 码本不均衡 | 多级别衡量化 | OneRec |
| **语义-行为gap** | **行为感知微调** | **MMQ, DAS** |
| LLM适配 | 离线SID+预训练LLM词表扩展+CPT | PLUM |

### 12.2 未来方向

1. **端到端统一**：语义ID生成与推荐模型的端到端联合优化（DIGER、UniSID的方向）
2. **多模态深度融合**：更深度的多模态信息编码（MMQ的MoE方式）
3. **推理效率**：并行生成 + 层次剪枝的混合方案（OneSearch的方向）
4. **协同信号整合**：将协同过滤信号融入语义ID生成（DAS的方向）
5. **大规模基准**：建立统一的评测标准和数据集（FORGE的方向）
6. **LLM原生推荐**：直接在预训练LLM上适配语义ID（PLUM的方向）
7. **推理增强**：在推荐中加入显式推理能力（OneRec-Think的方向）

---

## 参考文献

1. Rajput, S., Mehta, N., Singh, A., Sathiamoorthy, M., Heldt, L., Hong, L., Chi, E., & Sculley, D. (2023). Recommender Systems with Generative Retrieval. *NeurIPS 2023*. arXiv:2305.05065. https://arxiv.org/abs/2305.05065

2. Singh, A., Vu, T., Mehta, N., Keshavan, R., Sathiamoorthy, M., Zheng, Y., Hong, L., Heldt, L., Wei, L., Chi, E., & Yi, X. (2024). Better Generalization with Semantic IDs: A Case Study in Ranking for Recommendations. *RecSys 2024*. arXiv:2306.08121. https://arxiv.org/abs/2306.08121

3. QARM Team, Kuaishou. (2024). QARM: Quantitative Alignment Multi-Modal Recommendation at Kuaishou. arXiv:2411.11739. https://arxiv.org/abs/2411.11739

4. QARM V2 Team, Kuaishou. (2026). QARM V2: Quantitative Alignment Multi-Modal Recommendation for Reasoning User Sequence Modeling. arXiv:2602.09458. https://arxiv.org/abs/2602.09458

5. OneRec Team, Kuaishou. (2025). OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment. arXiv:2502.18965. https://arxiv.org/abs/2502.18965

6. OneRec V2 Team, Kuaishou. (2025). OneRec-V2 Technical Report. arXiv:2508.20900. https://arxiv.org/abs/2508.20900

7. OneRec-Think Team, Kuaishou & Tsinghua University. (2026). OneRec-Think: In-Text Reasoning for Generative Recommendation. *ACL 2026*. arXiv:2510.11639. https://arxiv.org/abs/2510.11639

8. DAS Team, Kuaishou. (2025). DAS: Dual-Aligned Semantic IDs Empowered Industrial Recommender System. arXiv:2508.10584. https://arxiv.org/abs/2508.10584

9. Hou, Y., Mu, S., Zhao, W.X., et al. (2022). Learning Vector-Quantized Item Representation for Transferable Sequential Recommenders. arXiv:2210.12316. https://arxiv.org/abs/2210.12316

10. Ramasamy, D., Kumar, S., Cadonic, C., Yang, J., Roychowdhury, S., Abdel Rhman, E., & Reddy, S. (2025). SIDE: Semantic ID Embedding for effective learning from sequences. *KDD Workshop / AdKDD 2025*. arXiv:2506.16698. https://arxiv.org/abs/2506.16698

11. RPG Team, Meta. (2025). Recommendation with Parallel semantic ID Generation. *KDD 2025*. arXiv:2506.05781. https://arxiv.org/abs/2506.05781

12. OneSearch Team, Kuaishou. (2025). OneSearch: A Preliminary Exploration of the Unified End-to-End Generative Framework for E-commerce Search. arXiv:2509.03236. https://arxiv.org/abs/2509.03236

13. DIGER Team. (2026). Differentiable Semantic ID for Generative Recommendation. *SIGIR 2026*. arXiv:2601.19711. https://arxiv.org/abs/2601.19711

14. PLUM Team. (2025). PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations. arXiv:2510.07784. https://arxiv.org/abs/2510.07784

15. Jiang, J., Zhang, X., Zhang, E., Xiong, Y., Zhang, J., Wang, J., Yu, H., Wang, Y., Wang, H., Yan, X., & Jiang, J. (2026). End-to-End Semantic ID Generation for Generative Advertisement Recommendation (UniSID). arXiv:2602.10445. https://arxiv.org/abs/2602.10445

16. COBRA Team, Baidu. (2025). Unified Generative Recommendations with Cascaded Sparse-Dense Representations. arXiv:2503.02453. https://arxiv.org/abs/2503.02453

17. Xu, et al. (2025). MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation. *WSDM 2026*. arXiv:2508.15281. https://arxiv.org/abs/2508.15281

18. MACRec Team. (2025). Multi-Aspect Cross-modal Quantization for Generative Recommendation. arXiv:2511.15122. https://arxiv.org/abs/2511.15122

19. FORGE Team, Alibaba. (2025). FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets. arXiv:2509.20904. https://arxiv.org/abs/2509.20904

20. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning (VQ-VAE). *NeurIPS 2017*. arXiv:1711.00937. https://arxiv.org/abs/1711.00937

20. Lee, K., Barnes, C., & Kim, J. (2022). Autoregressive Image Generation using Residual Quantization. *CVPR 2022*. arXiv:2203.01941. https://arxiv.org/abs/2203.01941

21. Ge, T., He, K., Long, J., & Jian, S. (2013). Optimized Product Quantization. *IEEE TPAMI*.
