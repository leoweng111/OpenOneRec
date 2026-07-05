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
  - [4.1 PQ原理与公式](#41-pq原理与公式)
  - [4.2 VQ-Rec：可迁移序列推荐](#42-vq-rec可迁移序列推荐)
  - [4.3 SIDE：Meta广告序列学习](#43-side-meta广告序列学习)
  - [4.4 MMQ：多模态混合量化](#44-mmq多模态混合量化)
- [5. OPQ方法（优化乘积量化 · 并行）](#5-opq方法优化乘积量化--并行)
  - [5.1 OPQ原理与公式](#51-opq原理与公式)
  - [5.2 RPG：Meta并行语义ID生成](#52-rpgmeta并行语义id生成)
- [6. RQ-OPQ混合方法（串行+并行）](#6-rq-opq混合方法串行并行)
  - [6.1 设计动机](#61-设计动机)
  - [6.2 OneSearch：快手电商搜索](#62-onesearch快手电商搜索)
- [7. 协同对齐方法](#7-协同对齐方法)
  - [7.1 DAS：快手双对齐语义ID](#71-das快手双对齐语义id)
- [8. 稀疏+稠密级联方法](#8-稀疏稠密级联方法)
  - [8.1 COBRA：百度级联表示](#81-cobra百度级联表示)
- [9. 端到端语义ID生成](#9-端到端语义id生成)
  - [9.1 UniSID：广告推荐端到端生成](#91-unisid广告推荐端到端生成)
  - [9.2 PLUM：LLM适配生成式推荐](#92-plumllm适配生成式推荐)
- [10. 工业基准](#10-工业基准)
  - [10.1 FORGE：淘宝大规模基准](#101-forge淘宝大规模基准)
- [11. 总结与展望](#11-总结与展望)
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

语义ID的生成本质上是一个**向量量化（Vector Quantization）** 过程。核心流程如下：

```
物品多模态特征 → Encoder → 连续向量 z → 量化器 → 离散码 (c₁, c₂, ..., c_L)
                                                          ↓
                                                     语义ID (Semantic ID)
```

**数学框架**：给定物品 $i$，其特征经过编码器得到连续向量 $\mathbf{z}_i \in \mathbb{R}^d$。量化的目标是找到一组离散码 $\mathbf{c}_i = (c_1, c_2, \ldots, c_L)$，使得从离散码重构的向量 $\hat{\mathbf{z}}_i$ 尽可能接近 $\mathbf{z}_i$。

### 1.3 经典方法：RQ-VAE详解

**RQ-VAE（Residual Quantized Variational AutoEncoder）** 是TIGER等工作中使用的最经典的语义ID生成方法，也是理解后续所有方法的基础。

#### 1.3.1 架构概览

```
┌─────────┐     ┌──────────────┐     ┌──────────────────┐     ┌─────────┐
│  输入    │     │   Encoder    │     │ 残差量化模块(RQ)  │     │ Decoder │
│  x ∈ R^D │ ──→ │  f_enc(x)=z  │ ──→ │  z → (c₁,...,cL) │ ──→ │  重构 x̂  │
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

| 论文 | 链接 | 量化方法 | 生成方式 | 核心贡献 |
|------|------|----------|----------|----------|
| [TIGER](#21-tiger奠基工作) (Google, NeurIPS 2023) | [arXiv:2305.05065](https://arxiv.org/abs/2305.05065) | RQ-VAE | 串行 | 奠基工作，首次提出语义ID+生成式检索 |
| [OneRec](#34-onerec快手统一生成式推荐) (快手, 2025) | [arXiv:2502.18965](https://arxiv.org/abs/2502.18965) | RQ-KMeans | 串行 | 统一检索排序+多级别衡量化 |
| [OneRec V2](#34-onerec快手统一生成式推荐) (快手, 2025) | [arXiv:2508.20900](https://arxiv.org/abs/2508.20900) | RQ-KMeans | 串行 | Lazy Decoder-Only, 80亿参数 |
| [RPG](#52-rpgmeta并行语义id生成) (Meta, KDD 2025) | [arXiv:2506.05781](https://arxiv.org/abs/2506.05781) | OPQ | 并行 | 一步并行生成全部语义ID |
| [OneSearch](#62-onesearch快手电商搜索) (快手, 2025) | [arXiv:2509.03236](https://arxiv.org/abs/2509.03236) | RQ-OPQ | 混合 | RQ串行+OPQ并行，电商搜索 |
| [PLUM](#92-plumllm适配生成式推荐) (Google, 2025) | [arXiv:2510.07784](https://arxiv.org/abs/2510.07784) | SID-v2 | 串行 | 预训练LLM适配生成式推荐 |
| [UniSID](#91-unisid广告推荐端到端生成) (2026) | [arXiv:2602.10445](https://arxiv.org/abs/2602.10445) | 端到端 | 串行 | 端到端联合优化embedding和SID |
| [DIGER](#23-diger可微分语义id) (SIGIR 2026) | [arXiv:2601.19711](https://arxiv.org/abs/2601.19711) | 可微分RQ | 串行 | Gumbel-Softmax可微分量化 |
| [COBRA](#81-cobra百度级联表示) (百度, 2025) | [arXiv:2503.02453](https://arxiv.org/abs/2503.02453) | 级联 | 级联 | 稀疏语义ID+稠密向量级联 |
| [MMQ](#44-mmq多模态混合量化) (WSDM 2026) | [arXiv:2508.15281](https://arxiv.org/abs/2508.15281) | MoE | 并行 | 多模态专家混合量化 |

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

### 1.6 语义ID方法演进总览

```
2022  VQ-Rec ─────────────────────── PQ (乘积量化, 并行)
       │
2023  TIGER ──────────────────────── RQ-VAE (残差量化, 串行) ← 奠基之作
       │
2023  YouTube Semantic IDs ──────── RQ-VAE + SentencePiece适配 (串行)
       │
2024  QARM (快手) ───────────────── RQ-KMeans + 多模态对齐 (串行)
       │
2025  OneRec (快手) ─────────────── RQ-KMeans + 多级别衡 (串行)
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
物品文本特征 → SBERT (frozen, d=768) → RQ-VAE Encoder → 残差量化(3层, K=4096) → (c₁, c₂, c₃)
```

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
| 输入来源 | 单一frozen embedding | 对齐后的多模态embedding |
| 训练复杂度 | 高（需训练VAE） | 低（仅K-Means） |
| 工业部署 | 较重 | 轻量 |
| 码本平衡性 | 可能出现不均匀 | K-Means天然更均衡 |

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
- App Stay Time +0.159%

---

## 4. VQ/PQ方法（向量量化/乘积量化）

**方法分类**：并行语义ID
**核心原理**：将向量切分为多个子空间，各子空间独立量化，一步并行生成所有token。

### 4.1 PQ原理与公式

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

### 4.2 VQ-Rec：可迁移序列推荐

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

### 4.3 SIDE：Meta广告序列学习

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

### 4.4 MMQ：多模态混合量化

- **标题**：[MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation](https://arxiv.org/abs/2508.15281)
- **发表**：WSDM 2026
- **arXiv**：https://arxiv.org/abs/2508.15281
- **使用范式**：**LLM4GRs**
- **构造范式**：**并行**（MoE混合量化）

**核心创新**：使用**Mixture-of-Experts (MoE)** 架构进行量化：

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

## 7. 协同对齐方法

**核心问题**：语义ID从内容特征量化而来，只包含内容语义，不包含协同过滤信号（用户-物品交互模式）。这导致语义ID与推荐目标不一致。

### 采用协同对齐方法的论文

| 论文 | 使用范式 | 核心改进 |
|------|----------|----------|
| [DAS](https://arxiv.org/abs/2508.10584) (快手, 2025) | LLM4DLRMs / LLM4GRs | 一阶段双对齐，融入协同信号 |

### 7.1 DAS：快手双对齐语义ID

- **标题**：[DAS: Dual-Aligned Semantic IDs Empowered Industrial Recommender System](https://arxiv.org/abs/2508.10584)
- **机构**：快手
- **发表**：arXiv 2025年8月
- **arXiv**：https://arxiv.org/abs/2508.10584
- **使用范式**：**LLM4DLRMs / LLM4GRs**（兼容判别式和生成式推荐）
- **构造范式**：兼容多种量化方法

**DAS架构**：

```
┌─────────────────────────────────────────────────┐
│                   DAS 架构                       │
│                                                 │
│  ┌─────────────────┐  ┌─────────────────────┐  │
│  │ User Semantic    │  │ Item Semantic        │  │
│  │ Model            │  │ Model                │  │
│  │ (量化用户特征)    │  │ (量化物品特征)        │  │
│  └────────┬────────┘  └──────────┬───────────┘  │
│           ↓                      ↓              │
│  ┌────────────────────────────────────────┐     │
│  │     ID-based CF Debias Module           │     │
│  │  (协同过滤去偏模块 — 注入交互信号)         │     │
│  └────────┬───────────────────────────────┘     │
│           ↓                                      │
│  ┌────────────────────────────────────────┐     │
│  │     Multi-View Contrastive Alignment    │     │
│  │  (多视角对比对齐)                        │     │
│  │                                         │     │
│  │  • Dual u2i: 用户→物品对齐               │     │
│  │  • Dual i2i/u2u: 物品→物品/用户→用户     │     │
│  │  • Dual co-occur: 共现物品/用户对齐       │     │
│  └────────────────────────────────────────┘     │
│           ↓                                      │
│  ┌────────────────────────────────────────┐     │
│  │     Dual Learning (双向学习)             │     │
│  │  用户量化 ↔ 物品量化 对偶训练             │     │
│  └────────────────────────────────────────┘     │
└─────────────────────────────────────────────────┘
```

**对齐损失**：

$$\mathcal{L}_{\text{align}} = \mathcal{L}_{\text{u2i}} + \mathcal{L}_{\text{i2i}} + \mathcal{L}_{\text{co-occur}}$$

**关键创新**：
- **一阶段训练**：量化和对齐同时优化，避免两阶段方法的信息损失
- **灵活性**：兼容各种量化方法（RQ-VAE、Res-KMeans等）和CF方法
- 已在快手广告系统部署，服务4亿日活用户

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

## 9. 端到端语义ID生成

**核心问题**：传统两阶段方法（先量化生成语义ID，再训练推荐模型）存在**目标不一致**问题。

### 9.1 UniSID：广告推荐端到端生成

- **标题**：[End-to-End Semantic ID Generation for Generative Advertisement Recommendation](https://arxiv.org/abs/2602.10445)
- **发表**：arXiv 2026年2月
- **使用范式**：**LLM4GRs**
- **构造范式**：**端到端**（联合优化embedding和语义ID）

**核心创新**：
- **端到端优化**：embedding和语义ID联合学习
- **多粒度对比学习**：在不同SID层级对齐不同粒度的语义
- **广告增强输入模式**：将异构广告信号线性化为统一token序列
- **摘要式广告重构**：鼓励SID捕获高层语义

**效果**：Hit Rate比最强baseline提升4.62%。

### 9.2 PLUM：LLM适配生成式推荐

- **标题**：[PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations](https://arxiv.org/abs/2510.07784)
- **发表**：arXiv 2025年10月
- **使用范式**：**LLM4GRs**
- **构造范式**：**串行**（复用现有语义ID方法）

**核心流程**：
1. 物品Tokenization：使用语义ID（SID-v2）将物品离散化
2. 持续预训练（CPT）：扩展LLM词汇表以包含SID token
3. 任务微调：针对生成式检索任务微调

**YouTube Shorts实验**：Panel CTR +4.96%。

---

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

## 11. 总结与展望

### 11.1 方法全景分类

#### 按量化方法分类

| 量化方法 | 代表论文 | 使用范式 | 生成方式 | 核心优势 |
|----------|----------|----------|----------|----------|
| **RQ-VAE** | TIGER, YouTube, DIGER | GRs/DLRMs | 串行 | 端到端优化，层次语义 |
| **RQ-KMeans** | QARM, QARM V2, OneRec | DLRMs/GRs | 串行 | 轻量，码本均衡 |
| **VQ/PQ** | VQ-Rec, SIDE, MMQ | DLRMs/GRs | 并行 | 推理快，可迁移 |
| **OPQ** | RPG | GRs | 并行 | 优化子空间，一步解码 |
| **RQ-OPQ** | OneSearch | GRs | 混合 | 层次+效率 |
| **双对齐** | DAS | DLRMs/GRs | 兼容多种 | 注入协同信号 |
| **级联** | COBRA | GRs | 级联 | 稀疏+稠密互补 |
| **端到端** | UniSID, PLUM | GRs | 串行 | 目标一致 |

#### 按解决的问题分类

| 问题 | 解决方法 | 代表论文 |
|------|----------|----------|
| 冷启动泛化 | 语义ID替代随机ID | TIGER, YouTube |
| 多模态对齐 | Item Alignment + 量化 | QARM, QARM V2 |
| 推理延迟 | 并行/混合生成 | RPG, OneSearch |
| 目标不一致 | 可微分/端到端量化 | DIGER, UniSID |
| 码本冲突 | FSQ混合量化 | QARM V2 |
| 存储效率 | 无参数SID转换 | SIDE |
| 信息损失 | 稀疏+稠密级联 | COBRA |
| 缺乏协同信号 | 双对齐/CF注入 | DAS |
| 码本不均衡 | 多级别衡量化 | OneRec |

### 11.2 未来方向

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

17. MMQ Team. (2025). MMQ: Multimodal Mixture-of-Quantization Tokenization for Semantic ID Generation and User Behavioral Adaptation. *WSDM 2026*. arXiv:2508.15281. https://arxiv.org/abs/2508.15281

18. FORGE Team, Alibaba. (2025). FORGE: Forming Semantic Identifiers for Generative Retrieval in Industrial Datasets. arXiv:2509.20904. https://arxiv.org/abs/2509.20904

19. van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning (VQ-VAE). *NeurIPS 2017*. arXiv:1711.00937. https://arxiv.org/abs/1711.00937

20. Lee, K., Barnes, C., & Kim, J. (2022). Autoregressive Image Generation using Residual Quantization. *CVPR 2022*. arXiv:2203.01941. https://arxiv.org/abs/2203.01941

21. Ge, T., He, K., Long, J., & Jian, S. (2013). Optimized Product Quantization. *IEEE TPAMI*.
