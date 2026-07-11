# OneRec 架构理解笔记

> 记录学习OneRec过程中遇到的核心问题和解答，聚焦于架构设计背后的"为什么"。

---

## 1. Qwen3在OneRec中的角色与词表机制

### 1.1 OneRec为什么需要Qwen3？

OneRec**没有从零训练一个全新的Transformer**，而是以Qwen3（预训练LLM）作为骨干模型，在其基础上扩展而来。

```
Qwen3 (预训练LLM)
  ├── 原始词表: ~151,669 tokens (中文/英文/符号等)
  ├── Transformer层: 预训练好的权重
  └── LM Head: 预测下一个token的投影层

      ↓ 扩展

OneRec = Qwen3 + 扩展词表 + 微调
  ├── 扩展词表: ~151,669 + 3×8,192 = ~176,000 tokens
  │              原始文本token    Itemic SID tokens
  ├── Transformer层: 在Qwen3权重基础上继续训练
  └── LM Head: 扩展到176k维
```

**为什么不能从零训练？**
- 从零训练数十亿参数的Transformer成本极高
- Qwen3已具备强大的序列建模能力和世界知识
- 做法：加载Qwen3权重 → 扩展词表（加入SID tokens）→ 继续预训练（CPT）→ 任务微调（SFT）→ RL对齐

这与PLUM的思路类似：**复用预训练LLM做推荐**，而不是从头造一个。

### 1.2 OneRec V2的"Lazy Decoder-Only"和Qwen3的关系

OneRec V2的"Lazy Decoder-Only"本质上就是一个**GPT-style的Decoder-Only Transformer** — 这和Qwen3的架构完全一致。所以OneRec V2不需要自己设计新架构，直接复用Qwen3的Decoder结构即可。

```
OneRec V1: 自定义Encoder-Decoder + MoE（从头设计架构）
OneRec V2: 直接用Qwen3的Decoder-Only（复用已有架构）
           → 去掉Encoder，去掉Cross-Attention
           → 计算量减少94%，可扩展到80亿参数
```

### 1.3 LLM的词表是什么？

**所有LLM都有词表。词表本质上就是一个双向映射关系：**

```
词表 (Vocabulary) = 一个双向映射

  "你好"    ↔  token_id = 15167
  "hello"   ↔  token_id = 8342
  "的"      ↔  token_id = 1916
  "<sid_begin>" ↔ token_id = 151669   ← OneRec新增的特殊token
  "<s_0_42>"    ↔ token_id = 151670   ← OneRec新增的SID token
  ...

  词表大小 V = 176,000
```

### 1.4 为什么LLM需要词表？

因为LLM的核心机制是 **Next Token Prediction**：在每一步，模型输出一个 $V$ 维的logits向量，表示"下一个token是词表中每个token的概率"。

```
Transformer最后一层输出 h ∈ R^d
        ↓
LM Head: W ∈ R^{V × d}
        ↓
logits = W · h ∈ R^V          # V=176,000维
        ↓
probs = softmax(logits)       # 每个token的概率
        ↓
loss = -log(probs[true_token_id])   # 交叉熵
```

**模型的输入和输出都是token_id（整数），不是原始向量。** 这就是LLM的工作方式。

### 1.5 为什么要把SID code映射到词表token？

这是最核心的问题。对比两种做法：

#### 做法A：传统量化用法（QARM等DLRM的做法）

```
物品 → RQ-KMeans → 语义ID = (c₁=42, c₂=15, c₃=89)
                          ↓
              Embedding Table (可学习, 独立于推荐模型)
              emb_table_1[42] → e₁ ∈ R^d
              emb_table_2[15] → e₂ ∈ R^d
              emb_table_3[89] → e₃ ∈ R^d
                          ↓
              concat(e₁, e₂, e₃) → DNN → CTR预估
```

这里语义ID就是普通的离散特征，通过embedding table查表后喂给DNN。完全可以用RQ-KMeans的聚类中心作为初始embedding。

#### 做法B：LLM生成式用法（OneRec/PLUM/TIGER的做法）

```
物品 → RQ-KMeans → 语义ID = (c₁=42, c₂=15, c₃=89)
                          ↓
              映射为词表token_id:
              c₁=42 → token_id = 151670 + 42 = 151712
              c₂=15 → token_id = 151670 + 8192 + 15 = 159877
              c₃=89 → token_id = 151670 + 2×8192 + 89 = 168151
                          ↓
              输入LLM: [user_tokens..., 151712, 159877, 168151]
                          ↓
              LLM自回归预测: P(next_token | history)
                          ↓
              输出: V维logits → 预测下一个SID token
```

#### 为什么OneRec选择做法B？

| 维度 | 做法A (DLRM) | 做法B (LLM/OneRec) |
|------|-------------|-------------------|
| 模型能力 | 只能做判别（打分） | 能生成（自回归预测） |
| 预训练知识 | 无 | 继承LLM的世界知识 |
| 统一检索+排序 | ❌ 需要多阶段 | ✅ 生成即推荐 |
| 冷启动 | 依赖ID embedding | 语义泛化 |
| Scaling Law | 不明显 | 遵循power-law |
| 推理方式 | 全库打分 O(N) | Beam Search O(L×K) |

#### 直觉类比

把OneRec想象成一个"会说推荐语言的ChatGPT"：

```
普通LLM:   "今天天气" → 预测下一个词 → "很好"
OneRec:    "用户历史[A][B]" → 预测下一个"词" → [C] (一个SID token序列)
```

这里的[A]、[B]、[C]都是**词表中的token**，就像"天气"、"很好"是词表中的token一样。LLM不区分"自然语言token"和"物品SID token"——对它来说都是词表里的整数，都需要预测下一个。

### 1.6 推理时只需要关注SID token子集吗？

**是的**，但要区分训练和推理：

**训练时**：在全部 $V≈176k$ 个token上算loss。即使当前位置的正确答案是SID token，loss仍然在全部176k个token上算softmax。这让模型学会"生成物品时不要生成文本"。

**推理时**：当模型已生成 `<|sid_begin|>`，接下来**自然倾向于生成SID token**（但代码中没有显式logits masking）：

```
实际代码逻辑（benchmarks/benchmark/base_generator.py）：

1. 在prompt末尾追加 prompt_token = "<|sid_begin|>"
2. 直接调用vLLM的beam search，不修改logits
3. 模型通过训练学会：<|sid_begin|>后总是跟SID token
   → beam search自然选择高概率的SID token
   → 文本token概率极低，不会被选中

推理配置（config.py:26）：
  "prompt_token": "<|sid_begin|>",  # 追加到prompt末尾
  "num_beams": 16,                   # beam search宽度
  "max_new_tokens": 3,               # 生成3个SID code
```

**为什么不需要显式masking？**

模型在训练时看到数百万条样本，每条都是 `<|sid_begin|>` 后跟SID token。训练结果：
```
P(token | "<|sid_begin|>") 的分布：
  <s_a_0> ~ <s_a_8191>:  总概率 ≈ 99.9%
  其他176k个文本token:   总概率 ≈ 0.1%
```

beam search在 `<|sid_begin|>` 后自然选择高概率的SID token，无需显式mask。

**训练时学"全语言"，推理时说"方言"（模型自然倾向SID子集）。**

### 1.7 如果不用Qwen3，从头训练还需要扩词表吗？

**不需要。** 从头训练意味着词表从一开始就由你定义，SID token可以直接包含在内：

```python
# 从头训练：词表设计阶段就把SID token包含进去
vocab = {
    "the": 0, "a": 1, "你好": 2, ...           # 文本token
    "<s_a_0>": 10000, "<s_a_1>": 10001, ...    # SID codebook token
    "<|sid_begin|>": 15000, "<|sid_end|>": 15001,  # 特殊token
}
# 词表大小 = 文本token数 + SID token数，一步到位
```

**但从头训练的代价极大：**

| 方面 | 基于Qwen3 (OneRec做法) | 从头训练 |
|------|----------------------|---------|
| 词表操作 | 原有151,669 + 新增~24,577 | 一开始就设计好完整词表 |
| Embedding初始化 | 前151,669行已训练好，新增行随机 | 全部随机初始化 |
| Stage 1对齐 | **需要** — 冻结LLM，只训SID embedding | **不需要** — 所有参数一起从头学 |
| 语言理解能力 | 继承Qwen3的预训练知识 | 从零学起，需要海量文本数据 |
| 训练成本 | 低（复用预训练权重） | 极高（相当于训一个小LLM） |
| 数据需求 | 少量对齐数据即可 | 需要海量文本+推荐数据联合训练 |

**核心trade-off：**

- 基于Qwen3：通过Stage 1把SID语义"嫁接"到已有的语言空间中，成本低但需要分阶段训练
- 从头训练：所有参数一起学，不需要Stage 1对齐，但失去了预训练LLM的语言理解能力，训练资源需求翻倍

OneRec选择基于Qwen3的理由：**用较低成本复用LLM的语言理解能力**。如果推荐场景非常垂直（如纯视频推荐）且有海量领域数据，从头训练也是一种选择。

---

## 2. Stage 1 训练详解

### 2.1 Stage 1 在做什么？

Stage 1 是 **Itemic-Text Alignment**（物品token与文本的对齐）。核心目标：让新加入的SID token embedding在Qwen3的语言空间中获得合理的位置。

### 2.2 三类训练数据

#### 数据1：视频推荐序列（核心数据）

```
来源: data/onerec_data/pretrain/video_rec.py

输入: metadata parquet (uid, hist_video_pid, target_video_pid)
     + pid2sid parquet (pid → [c₀, c₁, c₂])

输出格式（每个用户一条样本）:
  用户最近512个历史视频的SID + 未来10个目标视频的SID 拼成一条长序列

例如:
  <|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>    ← 历史视频1
  <|sid_begin|><s_a_120><s_b_203><s_c_891><|sid_end|>      ← 历史视频2
  ...（最多512个历史SID）...
  <|sid_begin|><s_a_42><s_b_15><s_c_89><|sid_end|>         ← 目标视频1
  <|sid_begin|><s_a_7><s_b_33><s_c_201><|sid_end|>         ← 目标视频2
```

**训练目标**：让模型学会"给定历史SID序列 → 预测下一个SID序列"，即把推荐当作序列续写。

#### 数据2：物品理解

```
来源: data/onerec_data/pretrain/item_understand.py

让LLM建立 "视频SID ↔ 视频文字描述" 的语义对齐

格式（随机选模板）:
  模板1: {"视频ID": "<|sid_begin|>...", "视频内容": "这是一个搞笑猫咪视频..."}
  模板2: "视频<|sid_begin|>... 展示了以下内容：搞笑猫咪视频..."
  模板3: "视频<|sid_begin|>... 的内容完整描述如下：..."
```

**训练目标**：让SID token与文本语义对齐，使LLM理解"SID代表的物品内容是什么"。

#### 数据3：用户画像

```
来源: data/onerec_data/pretrain/user_profile.py

预先生成的用户兴趣画像文本（已嵌入SID）

例如: "用户U1的兴趣画像：喜欢观看<|sid_begin|><s_a_42>...类型的视频，
      偏好<|sid_begin|><s_a_7>...风格的..."
```

**训练目标**：让LLM学会用画像+SID描述用户。

### 2.3 训练配置

| 参数 | 值 | 含义 |
|------|-----|------|
| `--freeze_llm` | ✅ | **冻结Qwen3原始参数**，只训练SID Token embedding |
| `--start_optimize_embedding_index` | 151669 | SID token的起始token_id |
| `--model_dir` | `Qwen3-1.7B_itemic` | 已扩词表的Qwen3-1.7B |
| `--max_length` | 32768 | 最大序列长度 |
| `--learning_rate` | 2e-4 | 学习率 |
| `--num_training_steps` | 2000 | 训练步数 |
| `--num_epochs` | 4 | 训练轮数 |

### 2.4 Stage 1 的参数结构

```
┌─────────────────────────────────────┐
│      Qwen3-1.7B (frozen)            │
│  原始151,669个文本token参数          │
│  ← 全部冻结，不更新 ←               │
└──────────────┬──────────────────────┘
               │
┌──────────────┴──────────────────────┐
│   新增 ~24,577 个Itemic Token        │
│   (3层 × 8192 + 2个特殊token)       │
│   ← 仅这些embedding可训练 →         │
└──────────────┬──────────────────────┘
               ↓
  三类数据混合训练:
  1. 视频序列续写 (学会"推荐=续写SID")
  2. 物品理解 (学会"SID=物品描述")
  3. 用户画像 (学会"SID=用户画像")
```

### 2.5 为什么Stage 1要冻结LLM？

这不是"必须"，而是"最优策略"。对比三种做法：

#### 方案A：全部参数一起训练（不冻结）

```
SID embedding（随机初始化）→ 梯度回传 → 扰动Qwen3已训练好的权重
```

**问题：灾难性遗忘（Catastrophic Forgetting）**

Qwen3的1.7B参数是在万亿token上训出来的，已经形成了稳定的语义空间。新SID token的embedding是随机初始化的，产生的梯度噪声会破坏已学好的权重。就像在一栋建好的大楼里强行加新房间，可能导致整体结构变形。

#### 方案B：冻结LLM，只训SID embedding（OneRec Stage 1的做法）

```
Qwen3已训练好 → 它的hidden states形成稳定的语义空间
SID embedding → 学习如何"嵌入"这个已有空间
```

**优势：** LLM定义了"语言空间的坐标系"，SID token只需学会在这个坐标系里找到正确位置。训练稳定，收敛快。

#### 方案C：先冻结训SID（Stage 1），再全部解冻联合微调（Stage 2/3）

这正是OneRec的三阶段设计——先让SID token"入门"（学会在语义空间的位置），再让全部参数一起适配推荐任务。

**类比理解：**

```
想象Qwen3是一个已经建好的图书馆，每本书（文本token）都有固定位置。

现在要加入新书（SID token）：

方案A（不冻结）：
  → 强行把新书塞进书架，导致其他书的位置都乱了
  → 原来"猫咪"旁边的书可能被挤到"汽车"区

方案B（冻结）：
  → 在图书馆旁边建一个新书架，新书学习如何与旧书架对应
  → "搞笑视频SID" 学会放在靠近 "搞笑"、"幽默" 类书籍的位置

方案C（OneRec实际做法）：
  → 先建新书架（Stage 1冻结训练）
  → 再整体优化图书馆布局（Stage 2解冻微调）
```

### 2.6 为什么Stage 1不能只用视频序列数据？

假设**只用视频推荐数据做NTP**：

```
输入：SID_1 → SID_2 → SID_3 → ... → 预测 SID_4
```

SID token能学到什么？**只会学到"SID_A后面经常跟SID_B"这种共现统计规律。**

#### 问题：SID token不知道"自己代表什么"

```
具体例子：
  <s_a_340><s_b_6566><s_c_5603> = 某搞笑视频的SID
  <s_a_120><s_b_2300><s_c_4100> = 某科技评测视频的SID

只用视频序列数据时：
  ✓ 模型知道"搞笑视频后面常跟搞笑视频"（共现规律）
  ✗ 模型不理解这些SID代表的是"搞笑"还是"科技"（语义含义）

原因：SID embedding只是随机初始化的向量，没有语义锚定
```

#### 物品理解数据的作用：给SID注入语义

```
输入：请描述这个物品 <|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>
输出：这是一段关于宠物猫做出滑稽动作的搞笑视频...
```

这迫使SID token的embedding**与对应文本描述在语义空间中对齐**。训完之后：

```
"搞笑视频"的SID embedding → 靠近"搞笑"、"幽默"、"宠物"等文本token的embedding
"科技评测"的SID embedding → 靠近"科技"、"数码"、"评测"等文本token的embedding
```

**本质：让SID token借用LLM已有的语言语义来理解自己代表什么内容。**

#### 用户画像数据的作用：建立SID与用户兴趣语言的桥梁

```
输入：该用户喜欢 <|sid_begin|>SID_A<|sid_end|> 和 <|sid_begin|>SID_B<|sid_end|> 相关的内容
输出：推荐 <|sid_begin|>SID_C<|sid_end|>
```

这让模型学会：当用户画像文本中出现"喜欢搞笑"时，应该推荐对应搞笑类SID。

#### 三种数据缺一不可

| 数据类型 | 类比 | 作用 |
|---------|------|------|
| 视频序列数据 | 看一堆不认识的文字序列 | 学会"符号A后面常跟符号B"（序列转移模式） |
| 物品理解数据 | 给你看图识字卡片 | 学会"这个符号代表猫"（语义含义） |
| 用户画像数据 | 告诉你"张三喜欢猫" | 学会把用户兴趣和符号关联（个性化推荐） |

**只用视频序列 = 只会统计，不懂语义。** 就像一个人背下了所有答案的排列规律，但完全不知道题目在问什么。三种数据共同让SID token既有语义含义、又懂用户偏好、还会序列预测。

### 2.7 三类训练数据如何处理？

**核心设计：统一格式 + 随机混合 + 统一处理**

#### 预训练实际使用的数据

**重要发现**：预训练不仅使用三类推荐数据，还混合了**通用文本数据**：

```bash
# data/prepare_pretrain.sh
GENERAL_TEXT_PATH="../raw_data/general_text/pretrain"  # ← 通用文本数据
REC_DATA_PATH="../raw_data/onerec_data"                # ← 三类推荐数据
```

**通用文本数据**（`general_text_df`）：
- 来源：`raw_data/general_text/pretrain/`
- 内容：数学、代码、推理、通用知识等领域的文本数据
- 作用：**防止灾难性遗忘**，保持LLM的通用语言理解能力
- 数据集：[OpenOneRec-General-Pretrain](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-General-Pretrain)

**为什么需要混合通用文本？**

```
如果只用推荐数据训练：
  → LLM会逐渐"忘记"通用语言能力（catastrophic forgetting）
  → 模型变成"推荐专家"，但失去理解通用文本的能力
  → 无法处理用户输入的通用文本（如"推荐一个适合下雨天看的电影"）

混合通用文本后：
  → LLM同时学习推荐模式和通用语言
  → 保持"通才+专才"的能力
  → 既能推荐，又能理解用户的自然语言描述
```

#### 数据准备阶段（离线）

**四类数据**各自独立处理，但输出格式完全统一：

```python
# 1. video_rec.py (视频推荐序列)
{
  "source": "RecIF_VideoRec_Pretrain",
  "uuid": "...",
  "segments": [{"type": "text", "text": "<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|><|sid_begin|><s_a_120>..."}]
}

# 2. item_understand.py (物品理解)
{
  "source": "RecIF_ItemUnderstand_Pretrain",
  "uuid": "...",
  "segments": [{"type": "text", "text": "视频<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|> 展示了以下内容：搞笑猫咪视频..."}]
}

# 3. user_profile.py (用户画像)
{
  "source": "RecIF_UserProfile_Pretrain",
  "uuid": "...",
  "segments": [{"type": "text", "text": "用户U1的兴趣画像：喜欢观看<|sid_begin|><s_a_42>...<|sid_end|>类型的视频..."}]
}

# 4. general_text (通用文本：数学、代码、推理等)
{
  "source": "GeneralText_Math",  # 或其他通用文本source
  "uuid": "...",
  "segments": [{"type": "text", "text": "求解方程 x^2 + 2x + 1 = 0..."}]
}
```

**关键点**：四类数据都使用 `segments: [{"type": "text", "text": "..."}]` 格式，只是文本内容不同。

#### Tokenizer如何切分SID序列？

对于包含SID的文本，如 `<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>`，tokenizer会将其切分为**恰好5个独立token**：

```
输入: "<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>"
      ↓ tokenizer()
输出: [176245, 152009, 158235, 174372, 176246]
       ↑        ↑        ↑        ↑        ↑
       <|sid_begin|> <s_a_340> <s_b_6566> <s_c_5603> <|sid_end|>
       (1个token)   (1个token) (1个token)  (1个token) (1个token)
```

**为什么每个SID标记是一个独立token？**

因为这些标记在**词表扩展阶段**就被注册为独立token了（`expand_qwen3_vocab.py:402-458`）：

```python
def generate_itemic_tokens(itemic_layer_n=3, vocab_size_per_layer=8192):
    """生成所有SID特殊token"""
    new_tokens = []
    
    # 第1层: <s_a_0>, <s_a_1>, ..., <s_a_8191>  → 8192个token
    for i in range(vocab_size_per_layer):
        new_tokens.append(f"<s_a_{i}>")
    
    # 第2层: <s_b_0>, <s_b_1>, ..., <s_b_8191>  → 8192个token
    for i in range(vocab_size_per_layer):
        new_tokens.append(f"<s_b_{i}>")
    
    # 第3层: <s_c_0>, <s_c_1>, ..., <s_c_8191>  → 8192个token
    for i in range(vocab_size_per_layer):
        new_tokens.append(f"<s_c_{i}>")
    
    # 特殊标记
    new_tokens.append('<|sid_begin|>')  # 1个token
    new_tokens.append('<|sid_end|>')    # 1个token
    
    # 总计: 3 × 8192 + 2 = 24,578 个新token
    return new_tokens

# 将这些token添加到tokenizer的词表中
num_added = tokenizer.add_tokens(new_tokens)
# add_tokens() 会把每个字符串注册为一个不可分割的token
```

**Tokenizer的匹配机制**：

Qwen3使用BPE（Byte Pair Encoding）tokenizer，BPE在编码时会**优先匹配最长的已知token**：

```
输入文本: "<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>"

BPE匹配过程（贪心最长匹配）:
  位置0:  "<|sid_begin|>" ← 在词表中！→ token_id=176245，前进13个字符
  位置13: "<s_a_340>"     ← 在词表中！→ token_id=152009，前进8个字符
  位置21: "<s_b_6566>"    ← 在词表中！→ token_id=158235，前进9个字符
  位置30: "<s_c_5603>"    ← 在词表中！→ token_id=174372，前进9个字符
  位置39: "<|sid_end|>"   ← 在词表中！→ token_id=176246，前进11个字符

结果: [176245, 152009, 158235, 174372, 176246]  ← 5个token
```

**如果SID token没有被注册到词表会怎样？**

tokenizer会把 `<s_a_340>` 拆分成多个子词（如 `<`, `s`, `_`, `a`, `_`, `3`, `4`, `0`, `>` → 9个token），这不仅浪费token数量，还会导致模型无法将SID作为一个语义整体来学习。

**完整示例：一条视频推荐序列的tokenize**

```
输入文本（来自 video_rec.py 的一条样本）:
  "<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>"   ← 历史视频1
  "<|sid_begin|><s_a_120><s_b_203><s_c_891><|sid_end|>"     ← 历史视频2
  ... (512个历史SID) ...
  "<|sid_begin|><s_a_42><s_b_15><s_c_89><|sid_end|>"        ← 目标视频1
  "<|sid_begin|><s_a_7><s_b_33><s_c_201><|sid_end|>"        ← 目标视频2
  ... (10个目标SID) ...

每个SID = 5个token (<|sid_begin|> + 3个code + <|sid_end|>)
522个SID × 5个token = 2610个token

tokenize后的 input_ids: shape [1, 2610]
```

#### 数据合并（`data/scripts/split_data.py:242`）

```python
# 合并通用文本数据和推荐数据（包含三类）
combined_df = pd.concat([general_text_df, rec_data_df], ignore_index=True)

# 切分成多个shard文件（每个1000条样本）
output_files = split_dataframe(combined_df, max_rows=1000, ...)
```

三类数据在文件级别混合，存储为多个parquet shards。

#### 训练时加载（`pretrain/onerec_llm/data/qwen3_dataset.py`）

```python
def _process(self, sample, source_name=None):
    # 判断数据格式：segments 或 messages
    if "segments" in sample["json"] and sample["json"]["segments"] is not None:
        inputs = self._process_completion(sample)  # ← 三类数据都走这里
    else:
        inputs = self._process_chat(sample)
    return inputs
```

**`data/qwen3_dataset.py/_process_completion` 处理逻辑**（三类数据完全相同）：

```python
def _process_completion(self, sample):
    segments = sample["json"]["segments"]
    
    # 1. 拼接所有segment的text
    segments_text = ""
    for segment in segments:
        if segment["type"] == "text":
            segments_text += segment["text"]
    
    # 2. 添加EOS token
    segments_text += self.tokenizer.pad_token
    
    # 3. Tokenize
    inputs = self.tokenizer(segments_text, return_tensors="pt", ...)
    
    # 4. 设置loss_mask（除EOS外全部为1）
    inputs["loss_mask"] = torch.ones_like(input_ids)
    inputs["loss_mask"][..., -1] = 0  # EOS不计算loss
    
    # 5. 标记SID token（用于监控）
    itemic_id_mask = torch.zeros_like(input_ids)
    if self.itemic_id_range is not None:
        itemic_id_mask[(input_ids >= 151669) & (input_ids <= 176246)] = 1
    
    return inputs
```

**三类数据的处理逻辑完全相同**，只是输入的文本内容不同。

#### Sample Packing（样本拼接）
data/qwen3_dataset.py中
```python
def __iter__(self):
    buffer = []
    cur_length = 0
    
    for sample in dataset:
        inputs = self._process(sample)  # 统一处理三类数据
        sample_length = inputs["input_ids"].shape[-1]
        
        # 如果当前buffer + 新样本超过max_length，执行packing
        if cur_length + sample_length >= self.max_length:
            packed_inputs = self._packing(buffer)  # 把buffer中多个样本拼接
            yield packed_inputs
            buffer = [inputs]
            cur_length = sample_length
        else:
            buffer.append(inputs)  # 加入buffer等待packing
            cur_length += sample_length
```

**Packing过程**：

```python
def _packing(self, buffer):
    """把多个短样本拼接成一条长序列（~32768 tokens）"""
    for inputs in buffer:
        # 沿sequence维度拼接不同样本
        packed_input_ids.append(inputs["input_ids"].flatten())
        packed_loss_mask.append(inputs["loss_mask"].flatten())
        packed_position_ids.append(inputs["position_ids"])
        packed_itemic_id_mask.append(inputs["itemic_id_mask"].flatten())
        cu_seqlens.append(...)  # 记录每个样本的边界
    
    # 拼接成一条长序列
    packed_input_ids = torch.cat(packed_input_ids, dim=0).unsqueeze(0)
    # shape: [1, ~32768]
```

**Packing效果**：一个batch中包含多条样本（可能是视频序列 + 物品理解 + 用户画像混合），通过 `cu_seqlens` 记录边界供FlashAttention使用。

#### 完整数据流

```
┌─────────────────────────────────────────────────────────┐
│  数据准备阶段（离线处理）                                 │
├─────────────────────────────────────────────────────────┤
│  推荐数据（三类）:                                       │
│    video_rec.py         → video_rec/train.parquet       │
│    item_understand.py   → item_understand/train.parquet │
│    user_profile.py      → user_profile/train.parquet    │
│                                                         │
│  通用文本数据:                                           │
│    general_text/pretrain → 数学/代码/推理等文本          │
│                              ↓                          │
│  split_data.py:                                         │
│    pd.concat([general_text_df, rec_data_df])            │
│    → 合并四类数据（通用文本 + 三类推荐数据）             │
│                              ↓                          │
│  切分成 shards: part-00000.parquet, part-00001.parquet  │
│                              ↓                          │
│  file_list.json: ["part-00000.parquet", ...]            │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  训练阶段（在线加载）                                     │
├─────────────────────────────────────────────────────────┤
│  WebDataset(file_list.json)                             │
│         ↓ shardshuffle (文件级shuffle)                  │
│         ↓ local shuffle buffer (样本级shuffle)          │
│         ↓                                               │
│  _process_completion(sample):                           │
│    - segments_text = 拼接所有segment                    │
│    - tokenize → input_ids                               │
│    - loss_mask = 全1（除EOS）                           │
│    - itemic_id_mask = 标记SID token                     │
│         ↓                                               │
│  __iter__(): sample packing                             │
│    - buffer累积多个样本（四类数据随机混合）               │
│    - _packing() 拼接成 ~32768 tokens 的长序列           │
│    - cu_seqlens 记录样本边界（供FlashAttention使用）     │
│         ↓                                               │
│  yield batch: {                                         │
│    input_ids: [1, 32768],                               │
│    loss_mask: [1, 32768],                               │
│    itemic_id_mask: [1, 32768],                          │
│    cu_seqlens: [0, 500, 1200, 2100, ..., 32768]        │
│  }                                                      │
└─────────────────────────────────────────────────────────┘
```

### 2.8 Stage 1 vs Stage 2 训练对比

**关键发现：Stage 1和Stage 2使用完全相同的数据源，但训练策略不同。**

#### 训练脚本对比

| 配置项 | Stage 1 (`pretrain_stg1.sh`) | Stage 2 (`pretrain_stg2.sh`) |
|--------|---------|---------|
| **模型起点** | `Qwen3-1.7B_itemic` (原始扩词表) | Stage 1的输出 `step2000/converted` |
| **数据配置** | `examples/dataset_config/pretrain.json` | `examples/dataset_config/pretrain.json` ✅ **相同** |
| **`--freeze_llm`** | ✅ 有（冻结LLM） | ❌ 没有（解冻全部参数） |
| **`--start_optimize_embedding_index`** | 151669（只训SID embedding） | 没有（全部参数可训） |
| **训练步数** | 2000步 | 5000步 |
| **warmup步数** | 200步 | 500步 |
| **学习率** | 2e-4 | 2e-4 |

#### Stage 1 训练策略

```bash
# pretrain_stg1.sh 关键参数
python3 recipes/train_qwen3.py \
    --model_dir $MODEL_DIR \                    # Qwen3-1.7B_itemic
    --dataset_config examples/dataset_config/pretrain.json \
    --freeze_llm \                              # ← 冻结LLM
    --start_optimize_embedding_index 151669 \   # ← 只训SID embedding
    --num_training_steps 2000 \
    --num_warmup_steps 200
```

**Stage 1 做什么**：
- 冻结Qwen3全部参数（Transformer层 + 原始文本embedding）
- 只训练新增的~24,577个SID token的embedding
- 使用三类数据混合训练：视频序列 + 物品理解 + 用户画像
- 目标：让SID token在Qwen3的语言空间中获得合理位置

#### Stage 2 训练策略

```bash
# pretrain_stg2.sh 关键参数
python3 recipes/train_qwen3.py \
    --model_dir ${STAGE1_OUTPUT_DIR}/step2000/converted \  # ← Stage 1的输出
    --dataset_config examples/dataset_config/pretrain.json \  # ← 相同数据
    --use_tie_weights \                         # 权重绑定
    --num_training_steps 5000 \                 # ← 更长训练
    --num_warmup_steps 500
```

**Stage 2 做什么**：
- 解冻全部参数（Qwen3 + SID embedding）
- 使用相同的三类数据继续训练
- 目标：让Qwen3的Transformer层也适配推荐任务，SID embedding和语言参数协同优化

#### 为什么用相同数据？

```
Stage 1:
  ├── 数据: 通用文本 + 视频序列 + 物品理解 + 用户画像
  ├── 冻结: Qwen3全部参数
  ├── 只训练: SID token embedding (~24,577个)
  └── 目标: SID token在语言空间中获得合理位置

Stage 2:
  ├── 数据: 通用文本 + 视频序列 + 物品理解 + 用户画像（相同）
  ├── 解冻: 全部参数
  ├── 训练: Qwen3 + SID embedding 联合优化
  └── 目标: 推荐能力叠加到语言能力上，同时保持通用能力
```

**通用文本数据的关键作用**：
- Stage 1：防止SID embedding训练时"污染"Qwen3的通用语言能力
- Stage 2：防止全参数微调时模型"遗忘"通用语言理解能力
- 最终效果：模型既能推荐，又能理解用户的自然语言描述

**类比理解**：
- Stage 1：只背单词（SID embedding），语法（Qwen3）不动，同时复习通用课本
- Stage 2：单词和语法一起练习，整体提升 fluency，同时继续学通用课本

#### 训练参数实现（`pretrain/recipes/train_qwen3.py`）

```python
# Stage 1: 冻结LLM，只训SID embedding
if args.freeze_llm:
    assert args.start_optimize_embedding_index > 0
    for name, param in model.named_parameters():
        if "embed_tokens" in name or "lm_head" in name:
            param.requires_grad = True  # 只允许embedding和lm_head训练
        else:
            param.requires_grad = False  # 冻结其他参数

# 使用EmbeddingGradientMasker进一步控制
embedding_masker = EmbeddingGradientMasker(
    model, model.config, args.start_optimize_embedding_index
)
# 每次optimizer.step()后，恢复冻结的embedding行
if args.start_optimize_embedding_index > 0:
    embedding_masker.restore_frozen_params()
```

**Stage 2**：没有 `--freeze_llm`，所有参数 `requires_grad = True`，正常训练。

---

## 3. 为什么RQ-KMeans生成的SID不能直接用？

### 3.1 核心误解

> "RQ-KMeans用的embedding就是LLM/多模态模型生成的，SID不是已经在语言空间中蕴含了语义信息吗？为什么还需要Stage 1对齐？"

### 3.2 关键事实：RQ-KMeans码本 ≠ LLM的SID Embedding

这两者**完全独立**，处于不同的空间：

```
RQ-KMeans 的世界:                    LLM (Qwen3) 的世界:

物品 → 某个Encoder → emb ∈ R^d      Qwen3词表中的每个token
              ↓                        都有自己的embedding
       RQ-KMeans码本                    ∈ R^d_model (Qwen3的隐层维度)
       codebook[0] = [0.3, -0.1, ...]
       codebook[1] = [-0.2, 0.5, ...]        这是两套完全独立的
       ...                                    embedding空间！
              ↓
       物品A → SID = (42, 15, 89)
```

**RQ-KMeans输出的 `(42, 15, 89)` 只是三个整数**。这三个整数在Qwen3的词表中**没有任何对应的embedding**。

### 3.3 RQ-KMeans码本不能直接用的三个原因

**原因1：空间维度不一致**

```
RQ-KMeans的embedding空间:
  - 可能来自CLIP (维度512/768)
  - 可能来自BERT (维度768)
  - 可能来自多模态融合后的向量

Qwen3的embedding空间:
  - 维度是Qwen3的hidden_size (1.7B对应1536维)
  - 这是Qwen3预训练得到的独特空间

→ 两个空间维度不同，语义坐标轴也不同
→ 不能直接把RQ-KMeans的码字向量塞进Qwen3的embedding table
```

**原因2：即使维度相同，语义也不对齐**

```
假设两者都是1536维:

RQ-KMeans码本中:
  codebook[42] = [0.3, -0.1, 0.7, ...]    ← "某种视觉/多模态语义"

Qwen3词表中:
  "猫咪" embedding = [0.1, 0.5, -0.3, ...]  ← 语言语义空间
  "汽车" embedding = [-0.2, 0.1, 0.8, ...]

即使把codebook[42]强行放进Qwen3的SID token embedding:
  → Qwen3的Transformer层不知道这个向量意味着什么
  → 它和"猫咪"、"搞笑"等文本token的关系是断裂的
  → 模型无法利用预训练的语言知识
```

**原因3：RQ-KMeans码本是冻结的，不参与LLM训练**

```python
# Stage 1 的实际参数结构:

# 冻结的参数:
qwen3_transformer_layers    # Qwen3的所有Transformer层，不更新
qwen3_text_embeddings       # 原有的151,669个文本token embedding，不更新
rq_kmeans_codebooks         # RQ-KMeans的码本，不更新（甚至不在LLM中）

# 可训练的参数:
sid_token_embeddings        # 新增的~24,577个SID token的embedding
                            # shape: [24577, hidden_size]
                            # 这些是Stage 1唯一要训练的参数!
```

### 3.4 Stage 1的本质：教LLM一门"新语言"

```
想象Qwen3是一个只会说中文的人。

RQ-KMeans给每个视频分配了编号:
  视频A = 编号(42, 15, 89)
  视频B = 编号(7, 33, 201)

但这些编号对Qwen3来说就是一串数字，
它不知道"42号"和"搞笑猫咪"有什么关系。

Stage 1做的事情:
  1. 告诉它: "编号42的视频是一个搞笑猫咪视频"
     (物品理解数据: SID ↔ caption对齐)

  2. 让它学会: "看了编号42和15的用户，下一个会看编号7"
     (视频序列数据: SID序列续写)

  3. 让它学会: "喜欢编号42的用户，画像描述是..."
     (用户画像数据: SID ↔ 用户描述)

→ 训练过程中，Qwen3的Transformer层是冻结的
→ 只有"编号42"这个token的embedding在更新
→ 它学会了在Qwen3的语言空间中，"编号42"应该靠近"搞笑猫咪"的概念
```

### 3.5 为什么不能跳过Stage 1？

```
如果直接跳过Stage 1，进入Stage 2全参数训练:

  SID token embedding: 随机初始化
  Qwen3参数: 预训练好的

  → SID embedding和语言空间完全断裂
  → Stage 2需要同时:
     1. 从头学习SID embedding
     2. 调整Qwen3参数适配推荐任务
  → 两个目标冲突，容易训崩
  → Qwen3的语言能力可能被破坏（catastrophic forgetting）

Stage 1 的作用:
  → 先让SID embedding在冻结的语言空间中找到合理位置
  → Stage 2 再解冻全部参数协同优化
  → 避免灾难性遗忘，训练更稳定
```

---

## 4. 三阶段训练全景

```
Stage 1: Itemic-Text Alignment
  ├── 冻结: Qwen3全部参数
  ├── 训练: SID token embedding
  ├── 数据: 视频序列 + 物品理解 + 用户画像
  └── 目标: SID token在语言空间中获得合理位置

Stage 2: Full-parameter Co-Pretraining
  ├── 解冻: 全部参数
  ├── 训练: Qwen3 + SID embedding 联合优化
  ├── 数据: 推荐数据 + 通用文本混合
  └── 目标: 推荐能力叠加到语言能力上

SFT: Post-training
  ├── 全参数继续训练
  ├── 数据: 指令数据（chat模板）
  ├── only_assistant_loss=True（loss只在assistant段计算）
  └── 目标: 学会指令跟随，输出格式规范化
```

---

## 5. token_id、词表、Embedding Table的关系

### 5.1 词表 ≠ Embedding Table

这是两个不同的东西，经常被混淆：

```
词表 (Vocabulary / vocab.json):
  只是一个映射字典，记录 "文本 → 整数编号"
  存储在磁盘上的JSON文件中
  例如:
    "猫咪" → 15167
    "汽车" → 23456
    "的"   → 1916

  它的作用仅仅是: 把人类可读的文字转换成模型能处理的整数

Embedding Table (嵌入表):
  是一个巨大的可训练参数矩阵
  shape: [V, d] = [176000, 1536]
  存储在模型的权重文件中 (如 model.safetensors)

  第 15167 行 = "猫咪"的embedding向量 ∈ R^1536
  第 23456 行 = "汽车"的embedding向量 ∈ R^1536
  第 1916 行  = "的"的embedding向量 ∈ R^1536
```

### 5.2 token_id是Embedding Table的"行号"

```
token_id 就是 Embedding Table 的行号（索引）

"猫咪" → token_id = 15167 → Embedding Table 的第15167行 → [0.1, 0.5, -0.3, ...]
"汽车" → token_id = 23456 → Embedding Table 的第23456行 → [-0.2, 0.1, 0.8, ...]
```

用代码表示：

```python
# Embedding Table 是一个大矩阵
embedding_table = nn.Embedding(num_embeddings=176000, embedding_dim=1536)
# 内部就是一个 shape=[176000, 1536] 的矩阵 ，代表总共176000个token的embedding

# 查表操作就是取对应行
token_id = 15167                        # "猫咪"的编号
embedding = embedding_table[token_id]   # shape: [1536] → 取出第15167行
# embedding = [0.1, 0.5, -0.3, ...]    # 这就是"猫咪"在语言空间中的语义向量
```

### 5.3 Embedding从哪来？

**不是人工设定的，是预训练过程中学出来的。**

```
预训练开始时:
  embedding_table 的每一行都是随机初始化的
  "猫咪" 的embedding = [随机, 随机, 随机, ...]
  "汽车" 的embedding = [随机, 随机, 随机, ...]

预训练过程中:
  模型读了大量文本，通过 Next Token Prediction 不断调整参数
  "猫咪" 经常出现在 "可爱的___在睡觉"、"___在追毛线球" 等上下文中
  "汽车" 经常出现在 "开___去上班"、"___的发动机" 等上下文中

  → Transformer 层学会区分不同上下文
  → embedding_table 中的向量被梯度不断更新
  → "猫咪"的embedding逐渐靠近"可爱"、"睡觉"等概念
  → "汽车"的embedding逐渐靠近"驾驶"、"发动机"等概念
  → 语义相近的词，embedding也相近

预训练结束后:
  "猫咪" embedding ≈ [0.1, 0.5, -0.3, ...]  ← 这些数字是训练的结果
  "汽车" embedding ≈ [-0.2, 0.1, 0.8, ...]  ← 和猫咪的向量方向不同
```

### 5.4 完整的输入处理流程

```
输入文本: "猫咪很可爱"

Step 1: Tokenizer分词 (词表映射)
  "猫咪很可爱" → ["猫咪", "很", "可爱"]
                → [15167,  3842,  9521]    ← token_id序列

Step 2: Embedding查表
  embedding_table[15167] → [0.1, 0.5, -0.3, ...]   # "猫咪"的语义向量
  embedding_table[3842]  → [0.3, -0.1, 0.2, ...]   # "很"的语义向量
  embedding_table[9521]  → [0.4, 0.6, -0.1, ...]   # "可爱"的语义向量

  得到输入: shape [3, 1536]  (3个token, 每个1536维)

Step 3: 送入Transformer
  Transformer([3, 1536]) → [3, 1536]  # 经过多层attention+FFN

Step 4: LM Head预测下一个token
  最后一个位置的输出 h ∈ R^1536
  logits = LM_Head @ h ∈ R^176000  # 对词表中每个token打分
  probs = softmax(logits)          # 概率分布
  → 预测下一个token可能是 "，" 或 "的" 或 "啊" 等
```

### 5.5 类比理解

```
token_id       ≈  身份证号（只是编号）
词表           ≈  户籍系统（记录"名字 ↔ 身份证号"的对应关系）
Embedding Table ≈  这个人的所有特征（身高、体重、性格...）

身份证号 15167 → 查户籍系统 → 这个人叫"猫咪"
身份证号 15167 → 查特征表 → [0.1, 0.5, -0.3, ...]

身份证号本身不包含特征信息
特征表中的信息是通过"社会历练"（预训练）逐渐形成的
```

### 5.6 在OneRec的Stage 1中

```
Qwen3预训练完成后:
  embedding_table 的前151,669行已经训练好了
  → "猫咪"、"汽车"、"搞笑"等文本token都有合理的语义向量

OneRec扩词表后:
  embedding_table 新增 ~24,577 行（SID tokens）
  → 第151,670行: <s_0_0> 的embedding = [随机, 随机, ...]  ← 未训练!
  → 第151,671行: <s_0_1> 的embedding = [随机, 随机, ...]  ← 未训练!
  → ...

Stage 1 做的事情:
  → 冻结前151,669行（文本token已训练好的embedding）
  → 只训练后24,577行（SID token的embedding）
  → 让这些随机向量学会在Qwen3的语言空间中占据合理的位置
  → 例如: <s_0_42> 的embedding 应该靠近 "搞笑猫咪" 的概念
```

---

## 6. Post-train SFT（Stage 3）详细解析

### 6.1 SFT 在训练流程中的位置

SFT（Supervised Fine-Tuning）位于 Stage 3，接在 Stage 2 全参数协同预训练之后：

```text
Stage 1: Itemic-Text Alignment（冻结LLM，仅对齐新SID embedding）
   ↓
Stage 2: Full-parameter Co-Pretraining（全参数；推荐+通用文本）
   ↓
Stage 3: Post-train SFT（全参数；指令/对话格式数据）
```

SFT 的核心目标不是再学基础共现统计，而是让模型学会「指令跟随 + 推荐任务输出格式对齐」（尤其是 assistant 侧输出 SID 序列与推理文本）。

### 6.2 代码入口与启动方式

#### 训练启动脚本
- `pretrain/examples/posttrain_sft.sh`
  - 起始模型：`MODEL_DIR=${STAGE2_OUTPUT_DIR}/step5000/global_step5000/converted`
  - 数据配置：`--dataset_config examples/dataset_config/sft.json`
  - 关键超参：`--learning_rate 2e-4`、`--num_warmup_steps 500`、`--num_training_steps 5000`
  - 训练方式：多机多卡 MPI + FSDP

#### 主训练脚本
- `pretrain/recipes/train_qwen3.py`
  - 读取 `dataset_config` 并调用 `get_dataloader(...)`
  - 训练主流程：`forward -> shift labels -> masked CE loss -> backward -> optimizer.step`
  - SFT/Pretrain 的统一入口在同一脚本，通过数据格式和配置差异区分

### 6.3 SFT 数据来源与准备流程

#### 数据准备入口
- `data/prepare_sft.sh`
  - `GENERAL_TEXT_PATH=../raw_data/general_text/sft`
  - `REC_DATA_PATH=../raw_data/onerec_data`
  - 调用 `data/scripts/split_data.py` 合并后按 shard 切分

#### SFT 任务源（8类推荐任务）
- `data/onerec_data/sft/video_rec.py`
- `data/onerec_data/sft/interactive_rec.py`
- `data/onerec_data/sft/label_cond_rec.py`
- `data/onerec_data/sft/label_pred.py`
- `data/onerec_data/sft/ad_rec.py`
- `data/onerec_data/sft/product_rec.py`
- `data/onerec_data/sft/item_understand.py`
- `data/onerec_data/sft/rec_reason.py`

总调度脚本：`data/onerec_data/run.sh`（会依次生成上述 SFT parquet）。

### 6.4 SFT 数据格式（messages）

SFT 训练样本采用 `messages` 字段（而不是 pretrain 的 `segments`）。

典型结构（见 `data/onerec_data/sft/video_rec.py`）：

```json
[
  {"role": "system", "content": [{"type": "text", "text": "你是一个智能推荐助手..."}]},
  {"role": "user", "content": [{"type": "text", "text": "根据历史预测下一个...<|sid_begin|>..."}]},
  {"role": "assistant", "content": [{"type": "text", "text": "<|sid_begin|><s_a_xxx><s_b_xxx><s_c_xxx><|sid_end|>"}]}
]
```

并以 parquet 行存储，核心列通常是：
- `source`: 数据来源标识（任务名）
- `uuid`: 样本 id
- `messages`: 对话内容（json 字符串/列表）
- `metadata`: 任务辅助信息

### 6.5 数据加载与 mask 构造（SFT 与 Pretrain 的关键分野）

核心在 `pretrain/onerec_llm/data/qwen3_dataset.py`：

1. `_process(...)` 分流
   - 有 `segments` -> `_process_completion(...)`（pretrain 主路径）
   - 否则走 `_process_chat(...)`（SFT 主路径）

2. `_process_chat(...)`（SFT）
   - `tokenizer.apply_chat_template(...)` 把 messages 展平（见下方 6.5.0 详解）
   - 文本末尾追加 `pad_token`
   - 调用 `_get_assistant_mask(...)` 构造 `loss_mask`

#### 6.5.0 `apply_chat_template` 展平详解：前后数据形状对比

代码位置：`pretrain/onerec_llm/data/qwen3_dataset.py:477-481`

```python
text = self.tokenizer.apply_chat_template(
    msg_converted,
    tokenize=False,          # ← 返回文本字符串而非 token ID
    add_generation_prompt=False  # ← 不追加 assistant 空提示
)
```

**展平前**（`msg_converted`）——嵌套对话列表结构：

```python
# 数据类型: List[Dict[str, str]]
# 形状概念: 3条对话 × 每条含 role + content → 嵌套结构
msg_converted = [
    {"role": "system",    "content": "你是一个智能推荐助手..."},
    {"role": "user",      "content": "根据历史预测下一个...<|sid_begin|><s_a_340>..."},
    {"role": "assistant", "content": "<|sid_begin|><s_a_42><s_b_15><s_c_89><|sid_end|>"},
]
# ↑ 这是一个 Python 列表，每个元素是独立的对话轮次（role + content）
# ↑ LLM 无法直接消费这种结构——它只接受连续文本
```

**展平后**（`text`）——单一连续文本字符串：

```text
# 数据类型: str（一维连续字符串）
# 形状概念: 1 条连续文本，所有轮次按 Qwen3 的 ChatML 模板拼接
"<|im_start|>system\n你是一个智能推荐助手...<|im_end|>\n<|im_start|>user\n根据历史预测下一个...<|sid_begin|><s_a_340>...<|im_end|>\n<|im_start|>assistant\n<|sid_begin|><s_a_42><s_b_15><s_c_89><|sid_end|><|im_end|>\n"
# ↑ 这是一个扁平的字符串，所有对话轮次用 <|im_start|>role\n 和 <|im_end|>\n 标记分隔
# ↑ LLM 的 tokenizer 可以直接处理这种格式
```

**展平的本质：从"结构化对话列表"到"带角色标记的连续文本"**

```
展平前（嵌套结构）:                     展平后（扁平文本）:

[                                      "<|im_start|>system\n
  {role: "system",                       你是一个智能推荐助手...
   content: "你是..."}                    <|im_end|>\n
  {role: "user",                        <|im_start|>user\n
   content: "根据历史..."}                根据历史预测下一个...
  {role: "assistant",                    <|im_end|>\n
   content: "<|sid_begin|>..."}         <|im_start|>assistant\n
]                                        <|sid_begin|><s_a_42>...
                                         <|im_end|>\n"

↑ 3个独立的 dict 对象                   ↑ 1个连续的字符串
↑ role/content 分离存储                 ↑ role 被嵌入文本中（<|im_start|>role\n）
↑ 无法直接 tokenize                    ↑ 可以直接 tokenizer(text) → input_ids
```

**展平后追加 pad_token → tokenize → 最终张量形状：**

```python
text += self.tokenizer.pad_token              # 末尾追加 pad
inputs = self.tokenizer(text, return_tensors="pt", ...)
# inputs["input_ids"] shape: (1, L)
#   L = system token数 + user token数 + assistant token数 + pad token数
#   例如: 系统prompt约30 token + 用户输入约200 token + assistant约20 token + 1 pad ≈ 251
```

**为什么叫"展平"？**

messages 是一个**分层结构**（列表中的每个 dict 代表一个对话轮次），而 `apply_chat_template` 把它"压扁"成**一维连续文本**——不再有 role/content 的键值对区分，而是用 `<|im_start|>role\n` 和 `<|im_end|>\n` 这些特殊 token 在连续文本中标记每段的角色边界。这样 tokenizer 就能将其当作普通文本一维化处理为 token ID 序列。

**对比 Pretrain 的 segments 格式**：

```
Pretrain (segments):     拼接 segment.text → 连续文本 → tokenize → (1, L)
SFT (messages):          apply_chat_template → 连续文本 → tokenize → (1, L)

两者最终都产出 (1, L) 的 input_ids，但 messages 多了一步"展平"（注入角色标记），
且展平后的角色标记会被 _get_assistant_mask 用来定位 assistant 段。
```

3. `_get_assistant_mask(...)`
   - 通过匹配 `<|im_start|>assistant\n` 到 `<|im_end|>\n` 的 token 区间
   - 仅 assistant 内容位置置 1，其余（system/user）置 0
   - 末尾 token（追加的 pad）再强制置 0

因此，SFT 的「只在 assistant 上算 loss」是由 `messages + assistant mask` 实现的。

#### 6.5.1 为什么 SFT 必须做 `get_assistant_mask`？——从机制到原理

##### 问题的起点：SFT 数据长什么样？

SFT 采用 `messages` 格式，一条样本包含 system、user、assistant 三轮对话。经过 `apply_chat_template` 展平后，token 序列形如：

```
<|im_start|>system\n你是一个推荐助手...<|im_end|>\n
<|im_start|>user\n根据用户历史预测下一个视频<|sid_begin|>...<|im_end|>\n
<|im_start|>assistant\n<|sid_begin|><s_a_42><s_b_15><s_c_89><|sid_end|><|im_end|>\n
```

这里有三种角色的 token 混在一起。如果不加区分地算 loss，模型会同时学习"复述 system prompt"和"复述 user 问题"——这不是我们想要的。

##### `get_assistant_mask` 做了什么？

核心代码在 `pretrain/onerec_llm/data/qwen3_dataset.py:302-359`：

```python
def _get_assistant_mask(self, batch_input_ids, start_pattern, end_pattern):
    """扫描 token 序列，找到 assistant 段，生成 loss_mask"""
    # start_pattern = tokenize("<|im_start|>assistant\n")
    # end_pattern   = tokenize("<|im_end|>\n")
    
    mask = [0] * len(ids)          # 初始全0：所有token都不算loss
    i = 0
    while i < len(ids):
        # 1. 扫描找到 "<|im_start|>assistant\n" 的位置
        if ids[i:i+start_len] == start_pattern:
            content_start = i + start_len   # assistant内容的起始位置
            # 2. 从content_start继续扫描，找到 "<|im_end|>\n" 的位置
            j = content_start
            while j < len(ids):
                if ids[j:j+end_len] == end_pattern:
                    break
                j += 1
            # 3. 把 content_start 到 j 之间的位置标为1
            for k in range(content_start, j):
                mask[k] = 1
        i += 1
    return mask
```

**直观效果：**

```
Token 序列:
  [system tokens] [user tokens] [assistant tokens] [pad]
      ↓               ↓              ↓              ↓
loss_mask:
  [0, 0, 0, ...] [0, 0, 0, ...] [1, 1, 1, ...]     [0]
   ↑ 不算loss      ↑ 不算loss     ↑ 算loss           ↑ 不算loss
```

##### 为什么不能对所有 token 都算 loss？

这是 SFT 与 Pretrain 最核心的区别。分三个层面理解：

**原因 1：避免模型"复述问题"**

```
不对：
  输入: [system: 你是推荐助手] [user: 推荐篮球视频] [assistant: <SID>]
  loss 在所有 token 上计算
  → 模型会学到：给定 system prompt → 复述 system prompt
                  给定 user 问题  → 复述 user 问题
  → 这是 "鹦鹉学舌"，不是"回答问题"

正确：
  loss 只在 assistant 段计算
  → 模型学到的是：给定 system + user → 生成正确的 assistant 回答
  → system 和 user 作为条件（context），assistant 回答作为目标（target）
```

**原因 2：Pretrain 和 SFT 的学习目标本质不同**

```
Pretrain（Stage 1 & 2）：
  数据: 一段连续的文本序列（segments格式）
  目标: 学会续写——给定前文，预测下一个token
  loss: 几乎在所有token上计算（每个token都要学会预测下一个）
  本质: 无监督的语言建模（self-supervised）

SFT（Stage 3）：
  数据: 多轮对话（messages格式：system + user + assistant）
  目标: 学会按指令回答——给定问题，生成正确回答
  loss: 只在assistant回答上计算
  本质: 有监督的条件生成（supervised conditional generation）
```

两者的区别可以用一个类比理解：

```
Pretrain ≈ 大量阅读文章，学会"文章怎么写"
           → 每个字都参与学习（loss全开）

SFT     ≈ 做问答题，学会"看到问题怎么回答"  
           → 只在"你的回答"上打分（loss只开assistant段）
           → 题目本身（system/user）不参与打分
```

**原因 3：SFT 的 system/user 部分是"模板"，不是"知识"**

```
SFT 中 system/user 的内容往往是固定的任务模板：
  system: "你是一个智能推荐助手，根据用户历史行为推荐下一个视频"
  user:   "用户最近看了<|sid_begin|>...<|sid_end|>，请推荐下一个"

这些模板在训练集中重复出现成千上万次。
如果对这些模板也算 loss：
  → 模型会把大量梯度浪费在"学会复述模板"上
  → 模板已经很简单了（几乎可以背诵），不需要再学
  → 真正需要学习的是 assistant 如何根据输入生成正确的 SID 序列

只在 assistant 段算 loss：
  → 梯度100%聚焦在"回答质量"上
  → 训练效率更高，模型更快收敛到正确的推荐行为
```

##### `get_assistant_mask` 如何影响最终的 loss 计算？

在 `pretrain/recipes/train_qwen3.py:829`，loss_mask 被用来屏蔽非 assistant 位置：

```python
# Step 1: labels = input_ids 向右移动一位（Next Token Prediction的标准做法）
labels = torch.cat([input_ids[:, 1:], pad], dim=-1)

# Step 2: 用 loss_mask 把非 assistant 位置的 label 替换为 ignore_index (-100)
labels = labels * loss_mask + loss_fn.ignore_index * (1 - loss_mask)
#         ↑ loss_mask=1的位置保留原始label    ↑ loss_mask=0的位置设为-100

# Step 3: 交叉熵损失会自动忽略 label=-100 的位置
loss = CrossEntropyLoss(logits, labels)
# 只有 assistant 段的 token 参与 loss 计算和梯度回传
```

**完整流程图：**

```
messages (system + user + assistant)
    ↓ apply_chat_template
token序列: [sys...] [user...] [asst...] [pad]
    ↓ _get_assistant_mask
loss_mask: [0,...]  [0,...]   [1,...]   [0]
    ↓ train_qwen3.py: labels = input_ids shifted
labels:    [tok_1, tok_2, ..., tok_n, pad]
    ↓ labels = labels * loss_mask + (-100) * (1-loss_mask)
labels:    [-100, -100, ..., SID_1, SID_2, ..., -100]
               ↑ 忽略            ↑ 算loss           ↑ 忽略
    ↓ CrossEntropyLoss
loss 只在 SID_1, SID_2, ... 等 assistant token 上计算
```

##### 如果去掉 `get_assistant_mask`，会发生什么？

| 场景 | 有 mask | 无 mask（全 token 算 loss） |
|------|---------|--------------------------|
| system prompt | 不参与训练 | 模型学会"背诵"模板 |
| user 问题 | 作为条件，不打分 | 模型学会"复述"问题 |
| assistant 回答 | 100%梯度聚焦 | 梯度被模板和问题稀释 |
| 推荐准确性 | 高（聚焦回答） | 低（梯度噪声大） |
| 训练效率 | 高 | 低（浪费在简单模板上） |

**一句话总结**：`get_assistant_mask` 的本质是把 SFT 从"全文续写"转变为"条件生成"——system/user 是条件输入（不打分），assistant 回答才是模型需要学习的目标输出（打分）。这与 Pretrain 的"所有 token 都要学会预测下一个"有着本质区别。

### 6.6 `add_think_pattern` 的作用

在 `pretrain/examples/dataset_config/sft.json` 中：
- `"add_think_pattern": true`

在 `qwen3_dataset.py` 的 `_convert_messages(...)` 中：
- 若 assistant 内容含 `<think>...</think>`，为对应 user 追加 `/think`
- 若不含或为空，追加 `/no_think`，并可补空 `<think>\n</think>\n`

这使 SFT 能显式学习「是否需要推理链」及其输出风格，和推荐 reasoning 任务（如 `rec_reason.py`）配套。

### 6.7 SFT 批数据张量格式（训练时看到的真实输入）

在 sample packing 后，送入 `train_qwen3.py` 的 batch 为：
- `input_ids`: `(1, T)`
- `position_ids`: `(1, T)`
- `loss_mask`: `(1, T)`（SFT 下 assistant 段为 1）
- `itemic_id_mask`: `(1, T)`（SID token 区间监控）
- `cu_seqlens`: `(S+1,)`
- `sample_idx`: `(1, T)`

说明：第一维为 1 是 packed 容器设计，不代表只有 1 条原始样本；真实样本数由 `cu_seqlens` 给出。

### 6.8 与 Stage 2 Pretrain 的实质差异（代码视角）

| 维度 | Stage 2 Pretrain | Stage 3 SFT |
|---|---|---|
| 样本主格式 | `segments` | `messages` |
| 文本组织 | 直接拼接 | `apply_chat_template` 展平 |
| loss 区域 | 基本全 token（除末尾） | assistant token only |
| 训练目标偏向 | 语义与序列建模 | 指令跟随与输出规范 |
| 参数更新 | 全参数 | 全参数 |

### 6.9 一个容易混淆的代码事实

在 `pretrain/examples/dataset_config/sft.json` 里有：
- `"only_assistant_loss": false`

但当前代码中，assistant-only loss 主要是由 `messages` 分支的 `_get_assistant_mask(...)` 决定；`only_assistant_loss` 这个字段在现有 `qwen3_dataset.py` 中没有直接分支使用。也就是说，现版本更依赖数据格式与 mask 逻辑本身来实现 assistant-only 监督。

### 6.10 SFT 训练步骤总结（端到端）

```text
(1) 生成 SFT 任务数据（8类） + 通用文本SFT数据
(2) split_data.py 合并并切分为 parquet shards
(3) posttrain_sft.sh 读取 stage2 checkpoint 作为初始化
(4) dataloader 读取 messages 样本，apply_chat_template
(5) 构造 assistant-only loss_mask + sample packing
(6) 全参数继续训练 5000 steps
(7) 保存 checkpoint，并可转 HF 格式用于评测/蒸馏/RL
```
### 6.11 SFT 的核心作用：从“会续写”到“会按指令回答”

SFT 的关键价值是把模型从通用续写器，转成可执行推荐指令的助手。

Stage 2（Pretrain）主要让模型学到：
- SID token 的语义（某些 SID 对应某类内容）
- SID 序列共现规律（哪些内容容易连续出现）
- SID 与文本语义关联

但仅靠 pretrain，模型更擅长“继续写下一个 token”，不一定擅长“理解用户任务并按格式回答”。

### 6.12 为什么 Pretrain 不够

如果输入只是历史 SID 串，模型可以继续生成类似 SID 序列；
但当输入变成自然语言任务（例如“请推荐用户可能感兴趣的篮球内容”），就需要：
- 识别“推荐/预测/解释”等指令意图
- 结合条件（用户画像、查询词、目标行为）进行受控生成
- 输出符合任务要求的结构（如 SID 序列或推理文本）

这正是 SFT 要额外学习的能力。

### 6.13 SFT 如何补齐能力

SFT 通过 `messages` 监督样本（system/user/assistant）教会模型：

1. **指令理解**
   - 从 user 指令中解析任务目标与约束条件。

2. **多任务切换**
   - 在不同任务模板下切换生成策略（推荐、条件推荐、标签预测、推荐理由等）。

3. **回答对齐**
   - 仅在 assistant 段计算 loss，强化“回答质量”，而不是“复述问题”。

### 6.14 任务能力映射（SFT）

| 任务 | 指令示例 | 训练后希望学会的能力 |
|---|---|---|
| `video_rec` | 预测用户下一个可能观看内容 | 基础推荐生成 |
| `label_cond_rec` | 推荐用户可能点赞/长看内容 | 条件约束推荐 |
| `interactive_rec` | 基于查询词推荐 | 查询理解与匹配 |
| `label_pred` | 判断是否会发生某行为 | 判别式预测 |
| `rec_reason` | 给出推荐理由 | 推理链与解释生成 |

### 6.15 为什么“只算 assistant loss”很重要

SFT 的典型监督方式：

```python
# Pretrain（常见）: 大部分 token 都参与监督
loss_mask = [1, 1, 1, 1, 1, ...]

# SFT（chat）: 主要监督 assistant 回复
loss_mask = [0, 0, 0, 1, 1, 1, 0]
#             system/user    assistant
```

这样做的收益：
- 减少模型对 prompt 文本的机械拟合
- 更聚焦“给出正确答案”
- 提升指令跟随与输出稳定性

### 6.16 一句话总结

Pretrain 解决“模型懂不懂内容与序列规律”，SFT 解决“模型会不会按用户指令完成任务”。

没有 SFT，模型更像“会写 SID 的语言模型”；
有了 SFT，模型才更像“可交互的推荐助手”。

---

*最后更新: 2026-07-08*
