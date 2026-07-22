# 生成式推荐（Generative Recommendation）调研

> 本文是生成式推荐领域的调研总结，涵盖概述、样本组织、语义ID、预训练、后训练五大主题。
> 详细论文解读和技术细节请参见各专题文档：
> - 样本组织：`notes/sample_schema.md`
> - 语义ID：`notes/semantic_id.md`
> - 预训练与后训练：`notes/pre_and_post_training.md`

---

## 目录

- [1. 生成式推荐概述](#1-生成式推荐概述)
  - [1.1 什么是生成式推荐](#11-什么是生成式推荐)
  - [1.2 两大技术范式](#12-两大技术范式)
  - [1.3 完整训练管线](#13-完整训练管线)
- [2. 样本组织方式](#2-样本组织方式)
  - [2.1 LLM4GRs的三种样本组织](#21-llm4grs的三种样本组织)
  - [2.2 LLM4DLRMs的样本组织演进](#22-llm4dlrms的样本组织演进)
  - [2.3 反馈信号选择](#23-反馈信号选择)
- [3. 语义ID](#3-语义id)
  - [3.1 核心思想与原理](#31-核心思想与原理)
  - [3.2 量化方法分类](#32-量化方法分类)
  - [3.3 串行vs并行vs混合](#33-串行vs并行vs混合)
  - [3.4 多模态语义ID](#34-多模态语义id)
  - [3.5 语义ID对齐（Alignment）](#35-语义id对齐alignment)
  - [3.6 端到端优化与LLM适配](#36-端到端优化与llm适配)
- [4. 预训练](#4-预训练)
  - [4.1 架构选择](#41-架构选择)
  - [4.2 预训练数据策略](#42-预训练数据策略)
  - [4.3 预训练损失函数](#43-预训练损失函数)
  - [4.4 分阶段策略](#44-分阶段策略)
- [5. 后训练](#5-后训练)
  - [5.1 后训练的必要性](#51-后训练的必要性)
  - [5.2 SFT：后训练的桥梁](#52-sft后训练的桥梁)
  - [5.3 Off-policy对齐：DPO](#53-off-policy对齐dpo)
  - [5.4 On-policy对齐：GRPO/GBPO](#54-on-policy对齐grpogbpo)
  - [5.5 Reward模型设计](#55-reward模型设计)
- [6. 总结与展望](#6-总结与展望)
  - [6.1 方法全景](#61-方法全景)
  - [6.2 演进趋势](#62-演进趋势)
  - [6.3 未来方向](#63-未来方向)

---

## 1. 生成式推荐概述

### 1.1 什么是生成式推荐

生成式推荐（Generative Recommendation, GR）将推荐任务重构为**序列生成问题**：给定用户历史行为序列，自回归预测下一个物品的语义ID（Semantic ID）。这一范式与大语言模型（LLM）的语言建模天然对齐——"续写下一个token"就是"预测下一个推荐item"。

与传统推荐系统的关键区别：

| 维度 | 传统推荐系统 | 生成式推荐 |
|------|-------------|-----------|
| 任务定义 | 判别式：对每个候选打分排序 | 生成式：自回归生成目标物品ID |
| 物品表示 | 随机ID → Embedding Table | 语义ID（内容驱动的离散token） |
| 检索方式 | 全库遍历打分 O(N) | Beam Search自回归解码 O(L×K) |
| 模型架构 | 双塔/DNN/DLRM | Transformer（Enc-Dec或Decoder-Only） |
| 冷启动 | 困难（无交互历史 → 无embedding） | 较好（语义ID基于内容，天然泛化） |

### 1.2 两大技术范式

生成式推荐有两种主要的技术路线：

**LLM4GRs（LLM赋能的生成式推荐）**：将推荐重构为序列生成任务，利用Transformer的自回归生成能力直接生成目标物品的语义ID。代表工作包括TIGER、OneRec、PLUM、RPG等。

**LLM4DLRMs（LLM赋能的深度推荐模型）**：保持判别式推荐的任务定义（对候选item打分排序），但在模型架构或特征表示上借鉴LLM的技术。这一范式内部又可分为两个发展阶段：

- **阶段一：语义ID作为输入特征**——在传统DLRM架构中用语义ID替换随机Item ID，模型主体仍为DNN/DLRM。代表工作包括快手QARM、快手DAS、YouTube Semantic IDs、SIDE等。
- **阶段二：Transformer/Decoder-like架构用于判别式排序**——直接采用Transformer（特别是Decoder-like）架构作为排序模型，但输出仍为CTR/CVR预测分数而非生成的物品ID序列。这一方向的代表工作包括：
  - **RankMixer**（字节跳动, arXiv:2507.15551, 2025.7）：用Token Mixing替代标准self-attention，配合per-token FFN和Sparse MoE，扩展至10亿参数规模，部署于抖音Feed排序
  - **HyFormer**（字节跳动, arXiv:2601.12681, 2026.1）：用Query Decoding（类decoder cross-attention）统一长序列建模与特征交互，全球Token跨attend行为序列的KV表示，部署于抖音
  - **LONGER**（字节跳动, RecSys 2025）：面向长序列的GPU高效Transformer，处理万级用户行为序列
  - **TokenMixer-Large**（字节跳动, 2026.2）：RankMixer的进化版，在线7B/离线15B参数，验证了判别式推荐模型的scaling law
  - **HiFormer**（Google, 2023）：异构交互Transformer，用于Google Play应用排序
  - **LiGR**（LinkedIn, 2025）：改造Transformer做set-wise排序打分，验证排序模型的scaling law

两者的核心区别在于：LLM4GRs中语义ID是**生成目标**（输出为token序列），LLM4DLRMs中语义ID是**输入特征**（输出为标量分数）。但值得注意的是，LLM4DLRMs阶段二的模型已经在**架构层面**与LLM4GRs趋同（都使用Transformer/Decoder-like架构），区别仅在于最终任务是**生成**（自回归next-token prediction）还是**判别**（CTR/CVR打分）。GPSD（KDD 2025）进一步验证了两条路线可以衔接：先用生成式预训练（NTP）初始化模型，再迁移到判别式微调，判别式模型同样遵循power-law scaling。

### 1.3 完整训练管线

生成式推荐的训练管线与LLM发展历程一致（Pretrain → SFT → RLHF/PPO/DPO），分为以下阶段：

```
离线前置: 语义ID构造 (物品→多模态Encoder→量化→离散token)
    ↓
预训练阶段:
  Stage 1: SID对齐 (冻结LLM, 只训SID embedding)
  Stage 2: 全参数协同预训练 (推荐数据+通用文本)
  Stage 3: SFT (指令跟随+格式规范化)
    ↓
后训练阶段:
  路径A: Off-policy对齐 (DPO)
  路径B: On-policy对齐 (GRPO/GBPO)
    ↓
部署: Beam Search推理 → SID序列 → 映射回物品
```

---

## 2. 样本组织方式

样本组织方式决定了训练时loss的计算位置和数据的消费模式，是影响训练效率和模型质量的基础性设计选择。

### 2.1 LLM4GRs的三种样本组织

| 维度 | Naive Impression | User-Centric | New Impression Only |
|------|-----------------|-------------|-------------------|
| **样本粒度** | 一次曝光 | 一个用户 | 一次曝光 |
| **loss位置** | 全序列 | 全序列 | 仅target段 |
| **每item计loss次数** | 冗余（O(出现次数)） | 1次 | 恰好1次 |
| **时序泄漏** | 存在 | 严重 | 无 |
| **支持流式训练** | 勉强 | 否 | 是 |
| **代表** | 早期TIGER | OneRec开源pretrain | OneRec-V2线上 |

**Naive Impression**：每次曝光独立成样本，loss落在整条序列。实现最简单，但同一item被反复计算loss，浪费算力。已被淘汰。

**User-Centric**：每个用户一条样本，全序列loss。适合静态数据集的科研复现（OneRec开源版采用），但存在时序泄漏（t₁时刻的模型"预知"t₆时刻的结果）且无法流式更新。

**New Impression Only (NIO)**：每次曝光独立成样本，**loss只在最新曝光的target item上**，历史部分仅作为attention context。无泄漏、无冗余、天然支持流式训练，是当前工业界主流（OneRec-V2线上采用）。

三种方式使用的是**同一种loss**——token级V维softmax多分类交叉熵（Next Token Prediction），差异仅在于loss_mask的标记策略。

### 2.2 LLM4DLRMs的样本组织演进

DLRM范式下，样本组织也在经历平行变革：

- **Pointwise**（传统）：每个(user, item)对独立成样本，模型独立打分
- **Request-wise / List-wise**（当前主流）：同一次请求中的多个item打包为一条样本，共享user特征（如网易云音乐Climber）
- **Set-wise**（美团HoMer）：将粗排给精排的大量候选item（如300个）打包为一条样本，支持跨item交互（如美团HoMer）

LLM4GRs和LLM4DLRMs的样本组织方式可以正交组合：存储层选Request-wise压缩，训练层选New Impression Only避免泄漏。不过这块开源GR很多都是使用User-centric的范式预训练。

### 2.3 反馈信号选择

用户行为序列的构造还涉及一个上游选择：

- **弱反馈正向行为**（如曝光/有效播放）：数据量大但兴趣信号噪，需要强化学习"洗"出真偏好。OneRec即预测"Next Impression Item"。
- **强反馈正向行为**（如点赞/收藏/完播）：信号纯但量少，且可能存在时序乱序问题。

两者不是互斥的——弱反馈做预训练学共现规律，强反馈做SFT/RL学偏好对齐，这也是多阶段训练隐含的策略。当然，可以只用强信号构造行为序列（比如Meta的HSTU），
或者做多阶段训练（比如快手OneRec）

---

## 3. 语义ID

### 3.1 核心思想与原理

语义ID（Semantic ID）用物品的**内容特征**（文本、图像、视频等）生成紧凑的离散标识符，使语义相似的物品拥有相近的ID。

核心流程：`物品多模态特征 → [对齐] → Encoder → 连续向量z → 量化器 → 离散码(c₁, c₂, ..., c_L)`

语义ID构造涉及两个核心环节：**量化**（将连续向量离散化为码本索引）和**对齐**（弥合内容语义与协同过滤信号的gap，详见§3.5）。

关键性质：
- **层次性**：前面的token表示粗粒度语义，后面的token表示细粒度语义（串行生成时）
- **共享性**：相似物品共享部分token，实现语义泛化
- **紧凑性**：用L×logK bit表示一个物品，远少于原始embedding

### 3.2 量化方法分类

| 量化方法 | 代表论文 | 生成方式 | 核心特点 |
|----------|---------|---------|---------|
| **RQ-VAE** | TIGER, YouTube | 串行 | 端到端梯度优化，层次语义 |
| **RQ-KMeans** | QARM, OneRec | 串行 | 轻量，码本均衡，无需VAE |
| **VQ/PQ** | VQ-Rec, SIDE | 并行 | 推理快，子空间独立 |
| **OPQ** | RPG | 并行 | 优化旋转子空间，一步解码 |
| **RQ-OPQ** | OneSearch | 混合 | 层次语义+快速推理 |
| **MoE混合** | MMQ | 并行 | 多专家路由，模态分离/共享 |
| **FSQ** | QARM V2 | 串行 | 有限标量量化，避免码本崩溃 |

**RQ-VAE**是最经典的方法：通过Encoder-Decoder+残差量化模块端到端学习，码本通过梯度更新。训练损失由重构损失、码本损失和承诺损失三部分组成，使用Straight-Through Estimator解决argmin不可微问题。

**RQ-KMeans**是工业界更偏好的选择：直接对残差做K-Means聚类，无需训练神经网络，码本利用率更均匀，部署更轻量。

### 3.3 串行vs并行vs混合

**串行语义ID**（如RQ-VAE、RQ-KMeans）：逐层生成，后续token依赖前面的token。天然形成层次语义结构，码本利用率高（K^L种组合），适合Beam Search剪枝。但推理延迟随层数线性增长。

**并行语义ID**（如PQ、OPQ）：所有token同时生成。推理快（一步生成），但缺乏层次语义，码本组合空间稀疏，需要特殊策略处理无效组合。

**混合语义ID**（如RQ-OPQ）：结合串行的层次语义和并行的推理效率。先自回归生成RQ部分（粗粒度），再一步并行生成OPQ部分（细粒度）。

### 3.4 多模态语义ID

当物品包含多种模态信息时，如何融合后量化是关键问题：

- **直接拼接**（TIGER等）：简单但存在模态主导、语义模糊问题
- **MoE混合量化**（MMQ）：用Mixture-of-Experts架构，模态特定Expert+共享Expert+正交正则化，兼顾模态独特性与协同性
- **VQ-Fusion**（SIDE）：多任务VQ融合多源信号
- **跨模态量化**（MACRec）：模态间交叉注意力后联合量化

MMQ还引入行为感知微调（Stage 2），用下游推荐信号动态调整码本聚类，弥合语义-行为gap。

### 3.5 语义ID对齐（Alignment）

语义ID从内容特征量化而来，只包含**内容语义**，不包含**协同过滤信号**（用户-物品交互模式），存在**语义-行为gap**。对齐的目标是弥合这一gap，使SID同时反映内容相似性和行为相似性。

对齐可在SID构造的不同时机进行，形成四种范式：

| 对齐时机 | 代表论文 | 核心做法 |
|----------|---------|---------|
| **Pre-alignment**（量化前） | QARM | 用业务交互数据微调多模态Encoder，再量化对齐后的embedding |
| **Joint alignment**（量化中） | DAS, PLUM SID-v2 | 对齐损失作为量化训练的一部分，联合优化 |
| **Post-alignment**（量化后） | MMQ Stage 2 | 先完成内容量化，再用推荐梯度动态调整码本聚类 |
| **End-to-end**（端到端） | UniSID, DIGER | 不分离量化和对齐，推荐目标直接驱动SID生成 |

**DAS**（快手）是最完整的对齐方案：一阶段联合训练量化、CF建模和对齐三者。核心创新是CF去偏模块（ICDM），将CF信号分解为无偏兴趣/内容和偏差分量，对齐时只用无偏分量，防止流行度偏差污染SID。采用六路多视角对比对齐（用户-物品、物品-物品、用户-用户、共现等）。

**PLUM SID-v2** 采用最轻量的对齐：在RQ-VAE训练中添加共现对比损失（InfoNCE on 用户观看历史中的共现物品对），单损失项即可将SID唯一性从94.0%提升到96.7%。

**DIGER** 代表最彻底的对齐形式：通过Gumbel-Softmax使量化操作可微，推荐损失梯度直接传播到码本，无需显式对齐步骤。

**OneRec、TIGER**等方法不做显式对齐，SID纯基于内容量化，协同信号在后续预训练阶段（行为序列NTP训练）隐式学习。

**演进趋势**：从无对齐 → Pre-alignment → Joint alignment → End-to-end，逐步走向联合优化。

### 3.6 端到端优化与LLM适配

传统两阶段方法（先量化SID，再训推荐模型）存在目标不一致问题。两个解决方向：

- **端到端语义ID生成**（UniSID）：embedding和SID联合学习，推荐梯度直接优化SID构造
- **LLM适配**（PLUM）：离线构造SID-v2后，扩展预训练LLM词表，通过CPT让SID token在语言空间中获得语义锚定。SID本身是离线工序，但整体框架将SID纳入LLM进行端到端预训练

---

## 4. 预训练

### 4.1 架构选择

架构决定了预训练的算力分配和可扩展性，经历了从Encoder-Decoder到Decoder-Only的演进：

| 维度 | Encoder-Decoder | Decoder-Only |
|------|----------------|-------------|
| 算力分配 | Encoder占大量算力（如OneRec V1的97.66%） | 100%算力用于序列预测 |
| 训练效率 | 大量计算在Encoder（不产生loss） | 每步计算都服务于NTP loss |
| 扩展性 | 增大Encoder收益递减 | 遵循power-law scaling |
| LLM复用 | 需T5等Enc-Dec LLM | 可复用GPT/Llama/Gemini等 |
| 推理延迟 | 两次前向 | 单次前向+KV Cache |

**核心结论**：Decoder-Only是当前主流。OneRec从V1（Enc-Dec）到V2（Decoder-Only）的升级减少了94%计算量；HSTU首次验证了推荐领域的power-law scaling；PLUM直接复用Gemini家族Decoder-Only LLM。

### 4.2 预训练数据策略

预训练需要多种数据类型满足不同目标：

| 数据类型 | 作用 | 代表 |
|----------|------|------|
| **行为序列数据** | 学会SID共现规律+序列续写 | OneRec video_rec |
| **物品理解数据** | 学会SID=物品语义描述 | OneRec item_understand |
| **用户画像数据** | 学会SID=用户偏好描述 | OneRec user_profile |
| **通用文本数据** | 防止灾难性遗忘，保持语言能力 | OneRec general_text, PLUM CPT |

**混合通用文本的必要性**：如果只用推荐数据，LLM会逐渐"忘记"通用语言能力，变成"只会续写SID"的统计机器。混合通用文本后，LLM同时学习推荐模式和通用语言，保持"通才+专才"的双重能力。

PLUM的CPT数据由用户行为序列（约50%）、物品元数据（约50%，包含SID+title、SID+topics、ASR、channel等）以及通用文本混合组成，总计约260 billion tokens。

### 4.3 预训练损失函数

所有LLM4GRs的预训练/SFT阶段都使用**token级V维softmax多分类交叉熵**（Next Token Prediction）：

$$\mathcal{L}_{\text{NTP}} = -\frac{1}{|\mathcal{S}|}\sum_{k \in \mathcal{S}} \log p_\theta(\text{tok}_{k+1} \mid \text{tok}_{\le k})$$

其中 $\mathcal{S}$ 是参与loss的token位置集合，由样本组织方式（loss_mask）决定。V是词表大小（OneRec约176k，包含原始文本token和SID token）。

特殊变体包括：RPG的Multi-Token Prediction（所有SID token并行预测，独立CE loss之和），以及PLUM SID训练中的共现对比损失。

### 4.4 分阶段策略

预训练通常分为多个阶段，核心思路是"冻结→解冻→SFT"：

**Stage 1: SID对齐（冻结LLM）**
- 冻结LLM全部参数，只训练新增的SID token embedding
- 目标：SID token在LLM语言空间中获得语义锚定
- 类比：先背单词（SID embedding），语法（LLM）不动
- 冻结原因：防止随机初始化的SID embedding梯度扰动LLM已训练好的权重

**Stage 2: 全参数协同预训练**
- 解冻全部参数，推荐数据+通用文本混合训练
- 目标：推荐能力叠加到语言能力上

**Stage 3: SFT（监督微调）**
- 全参数，messages格式（system/user/assistant），指令跟随训练
- loss仅在assistant段（`loss_mask`只标记回答部分）
- 目标：学会指令跟随+输出格式规范化

---

## 5. 后训练

### 5.1 后训练的必要性

预训练让模型学会"续写SID序列"，但存在三个根本问题：
1. **预测目标偏移**："下一个曝光item" ≠ "用户真正感兴趣的item"
2. **输出质量不可控**：可能生成无效SID组合或不遵循指令格式
3. **推荐多目标性**：需要同时优化观看时长、点赞率、评论率等多维偏好，单纯NTP无法建模

后训练的本质是从"会续写"进化为"懂偏好"。

### 5.2 SFT：后训练的桥梁

SFT是预训练到偏好对齐之间的桥梁。核心改变是将segments格式改为messages格式，loss从全序列变为仅assistant段。

SFT教会模型"按指令回答"，但缺乏统筹全局商业目标和探索未知空间的能力。典型SFT任务包括：视频推荐、条件约束推荐、查询推荐、行为预测、推荐理由生成等。

PLUM使用Reward-Weighted SFT替代显式RL，训练样本按reward值采样，效果接近简单RL但训练更稳定。

### 5.3 Off-policy对齐：DPO

DPO（Direct Preference Optimization）绕过显式reward模型，直接从偏好对优化策略：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log \sigma\left(\beta \left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right)\right]$$

**推荐场景的特殊挑战**：缺乏自然偏好对（不像LLM场景可由人类标注员选择）。解决方案是用Reward模型+Beam Search构造偏好对（如OneRec V1：128条候选→Reward打分→最高分preferred/最低分rejected）。

**重要澄清**：Reward模型仅用于偏好对**数据构造**，不出现在DPO loss函数中。DPO loss只使用策略模型和参考模型的log概率比，这符合DPO原始设计。

OneRec V1采用Constrained DPO（仅1%数据做DPO，99%用NTP）和Iterative DPO（多轮迭代更新偏好数据），在效果和成本之间取得最优平衡。

### 5.4 On-policy对齐：GRPO/GBPO

**GRPO**（Group Relative Policy Optimization）：去掉PPO的critic网络，用组内标准化替代基线估计。一个group中的多条rollout候选互相比较，优势值为：

$$A_i = \frac{r_i - \text{mean}(r_1,...,r_G)}{\text{std}(r_1,...,r_G) + \varepsilon}$$

只需2个模型（策略+参考），内存减半。OneRec V2和OneRec-Think均采用GRPO。

**GBPO**（Gradient-Bounded Policy Optimization，OneRec V2提出）：去除PPO/GRPO的clip操作，对π_old施加动态bound，保留所有样本的梯度，解决负样本抑制问题。

**On-policy vs Off-policy**：On-policy用当前策略模型生成候选（偏好信号始终与当前策略匹配），Off-policy用固定模型生成候选（偏好数据可能过时）。OneRec V2实验验证了on-policy采样的显著优势。

**OneRec-Think**的RL使用GRPO + Rollout-Beam Reward：在beam search（K=32）内评估推理质量，计算SID token级匹配数作为连续值reward，解决传统二值pass/fail reward的稀疏性问题。

### 5.5 Reward模型设计

Reward模型将用户多维反馈量化为标量信号，是后训练的核心组件：

| 系统 | Reward设计 | 关键特点 |
|------|-----------|---------|
| OneRec V1 | 四任务头（观看时长/概率/关注/点赞） | RM仅用于DPO偏好对构造 |
| OneRec V2 | Duration-Aware Reward Shaping | 按时长分桶消除偏差，用真实用户反馈替代RM |
| OneSearch | PARS六级分层奖励 | 动态权重调整，三塔架构 |
| OneRec-Think | Rollout-Beam Reward | beam内多候选评估，密集学习信号 |
| OpenOneRec GRPO | 多维reward（pass@1, partial_hit, hit, duration） | 组内标准化，连续值 |

---

## 6. 总结与展望

### 6.1 方法全景

**按架构分类**：

| 架构 | 代表论文 | 核心优势 |
|------|---------|---------|
| Encoder-Decoder | TIGER, OneRec V1, OneSearch | 上下文编码能力强 |
| Decoder-Only | OneRec V2, PLUM, HSTU, RPG | 算力效率高，遵循scaling law |
| LLM复用 | PLUM (Gemini) | 复用预训练知识，开发效率高 |
| **Transformer判别式** | RankMixer, HyFormer, HiFormer | 借用LLM架构做CTR打分，scaling law验证 |

**按训练管线分类**：

| 论文 | 预训练 | SFT | 后训练对齐 |
|------|--------|-----|-----------|
| TIGER | NTP (全序列) | 无 | 无 |
| OneRec V1 | NTP (session-wise) | NTP | Iterative DPO |
| OneRec V2 | NTP (NIO, 流式) | NTP | GBPO + Duration Reward |
| PLUM | CPT (混合数据) | Reward-weighted SFT | 无显式RL |
| HSTU | NTP (Generative) | 未公开 | 未公开 |
| RPG | MTP (并行) | 无 | 无 |
| OneSearch | 3-stage SFT | SFT | PARS + List-wise DPO |
| OneRec-Think | Multi-task CPT | Reasoning SFT | GRPO + Rollout-Beam |

### 6.2 演进趋势

1. **架构统一化 → Decoder-Only**：OneRec V1→V2减少94%计算，PLUM复用Decoder-Only LLM，HSTU验证power-law scaling

2. **后训练从off-policy到on-policy**：DPO（off-policy, 偏好对固定）→ GRPO/GBPO（on-policy, 持续更新），偏好信号始终与当前策略匹配

3. **Reward设计从模型评分到真实反馈**：Reward模型打分→用户真实行为反馈，Duration-Aware Reward Shaping解决时长偏差

4. **样本组织从静态到流式**：User-Centric（静态dump）→ New Impression Only（流式训练），曝光一条训一条，模型持续演进

5. **语义ID从纯内容到融合协同信号**：纯内容量化→注入CF信号（DAS双对齐、PLUM SID-v2共现对比损失）

6. **推理增强**：OneRec-Think引入Think-before-Recommend，生成推理链再做推荐决策，类似LLM的Chain-of-Thought

7. **多目标对齐**：单维reward→多维reward（时长、点赞、关注、评论等），Reward Shaping/Adaptive Weighting

8. **生成式与判别式架构趋同**：LLM4GRs和LLM4DLRMs在架构层面逐步趋同——两者都在使用Transformer/Decoder-like架构，区别仅在最终任务是生成（NTP）还是判别（CTR打分）。GPSD验证了生成式预训练可以初始化判别式模型，暗示两条路线可能最终收敛

### 6.3 未来方向

1. **端到端统一**：语义ID生成与推荐模型的端到端联合优化（DIGER可微分SID、UniSID端到端生成）

2. **Scaling Law深化**：推荐领域的power-law scaling仍需更细粒度的探索（数据量、模型大小、训练步数的关系）

3. **多模态深度融合**：MoE多模态编码（MMQ）+生成架构的深度结合

4. **LLM原生推荐**：直接在预训练LLM上适配语义ID（PLUM的方向），弥合domain gap

5. **推理增强推荐**：显式推理链提升推荐质量（OneRec-Think方向）

6. **流式后训练**：DPO/GRPO适配流式训练（持续rollout+持续对齐）

7. **大规模基准**：建立统一的评测标准和数据集（FORGE方向）

---

## 参考文献

1. TIGER — Rajput et al. (2023). Recommender Systems with Generative Retrieval. *NeurIPS 2023*. arXiv:2305.05065
2. YouTube Semantic IDs — Singh et al. (2024). Better Generalization with Semantic IDs. *RecSys 2024*. arXiv:2306.08121
3. OneRec V1 — OneRec Team, Kuaishou. (2025). OneRec: Unifying Retrieve and Rank. arXiv:2502.18965
4. OneRec V2 — OneRec V2 Team, Kuaishou. (2025). OneRec-V2 Technical Report. arXiv:2508.20900
5. OneRec-Think — OneRec-Think Team. (2026). OneRec-Think: In-Text Reasoning for Generative Recommendation. *ACL 2026*. arXiv:2510.11639
6. PLUM — PLUM Team, Google. (2025). PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations. arXiv:2510.07784
7. HSTU — Meta. (2024). Actions Speak Louder than Words. arXiv:2402.17152
8. RPG — Meta. (2025). Recommendation with Parallel semantic ID Generation. *KDD 2025*. arXiv:2506.05781
9. QARM — QARM Team, Kuaishou. (2024). QARM: Quantitative Alignment Multi-Modal Recommendation. arXiv:2411.11739
10. QARM V2 — QARM V2 Team, Kuaishou. (2026). QARM V2. arXiv:2602.09458
11. DAS — DAS Team, Kuaishou. (2025). DAS: Dual-Aligned Semantic IDs. arXiv:2508.10584
12. SIDE — Ramasamy et al. (2025). SIDE: Semantic ID Embedding. *AdKDD 2025*. arXiv:2506.16698
13. OneSearch — OneSearch Team, Kuaishou. (2025). OneSearch. arXiv:2509.03236
14. COBRA — COBRA Team, Baidu. (2025). Unified Generative Recommendations with Cascaded Sparse-Dense Representations. arXiv:2503.02453
15. UniSID — Jiang et al. (2026). End-to-End Semantic ID Generation. arXiv:2602.10445
16. MMQ — MMQ Team. (2025). MMQ: Multimodal Mixture-of-Quantization Tokenization. *WSDM 2026*. arXiv:2508.15281
17. DIGER — DIGER Team. (2026). Differentiable Semantic ID for Generative Recommendation. *SIGIR 2026*. arXiv:2601.19711
18. FORGE — FORGE Team, Alibaba. (2025). FORGE: Forming Semantic Identifiers for Generative Retrieval. arXiv:2509.20904
19. DPO — Rafailov et al. (2023). Direct Preference Optimization. *NeurIPS 2023*. arXiv:2305.18284
20. GRPO — DeepSeek Team. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.
21. HoMer — Meituan. (2025). Addressing Heterogeneities by Modeling Sequential and Set-wise Contexts. arXiv:2510.11100
22. RankMixer — Zhu et al., ByteDance. (2025). RankMixer: Scaling Up Ranking Models in Industrial Recommenders. arXiv:2507.15551
23. HyFormer — Huang et al., ByteDance. (2026). HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction. arXiv:2601.12681
24. LONGER — ByteDance. (2025). Scaling Up Long Sequence Modeling in Industrial Recommenders. *RecSys 2025*.
25. HiFormer — Google. (2023). Heterogeneous Interaction Transformer for App Ranking. arXiv:2311.xxxxx
26. LiGR — LinkedIn. (2025). From Features to Transformers: Redefining Ranking for Scalable Impact. arXiv:2502.xxxxx
27. GPSD — (2025). Generative Pretraining for Scalable Discriminative Recommendation. *KDD 2025*.
