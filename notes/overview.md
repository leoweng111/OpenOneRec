# OpenOneRec 代码全流程导读

本文档面向想快速搞懂 OpenOneRec 端到端 pipeline 的读者。按"数据 → Tokenizer → Pretrain → Post-training → Inference/评测"的顺序讲清楚每一段代码在做什么、输入输出的张量/数据形状是什么、以及关键脚本入口在哪。

> 目标：读完本文档 + 各文件顶部注释，就能不看论文直接读通仓库代码。

---

## 0. 一图看懂 OneRec

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  原始交互数据 (parquet)                                                       │
│     users × items × behaviors                                                 │
└──────────┬─────────────────────────────────────────────────────┬─────────────┘
           │                                                     │
           │(A) item embedding                                   │(B) 用户/交互
           ▼   (Qwen3-Embedding-8B)                              ▼
   ┌─────────────────┐                          ┌─────────────────────────────┐
   │ pid → 4096-d 向量│──残差K-Means──▶ pid→codes │ history + target + labels    │
   └─────────────────┘   tokenizer/               │ + user_profile + captions    │
                                                  └──────────┬──────────────────┘
                                                             │
       codes  = [c_a, c_b, c_c]  (每个 ∈ [0, 8192))           │
       token  = <s_a_{c_a}><s_b_{c_b}><s_c_{c_c}>             │
                夹在 <|sid_begin|> ... <|sid_end|> 中间          │
                                                             │
   ┌─────────────────────────────────────────────────────────▼─────────────────┐
   │  data/onerec_data/{pretrain,sft}/*.py                                      │
   │  → 每个任务把上面的原料拼成 messages / segments (chat/plain) parquet      │
   │  data/scripts/split_data.py → 按 1000 样本/文件分片                        │
   └─────────────────────────────────────────────────────────┬─────────────────┘
                                                             │
   ┌─────────────────────────────────────────────────────────▼─────────────────┐
   │  pretrain/tools/model_converter/expand_qwen3_vocab.py                     │
   │  Qwen3 base + n_layers*8192+2 个 Itemic Token → 扩词表 base model         │
   └─────────────────────────────────────────────────────────┬─────────────────┘
                                                             │
        ┌───── Stage 1: Itemic-Text Alignment ───┐           │
        │ 只解冻新加的 embedding 行               │◀──────────┤
        │ freeze_llm + start_optimize_...=151669 │           │
        └────────────────────┬───────────────────┘           │
                             ▼                               │
        ┌───── Stage 2: Full-parameter Co-Pretrain ───┐      │
        │ 混合"推荐 + 通用文本"全参训                  │◀─────┤
        └────────────────────┬────────────────────────┘      │
                             ▼                               │
        ┌───── Post Stage 1: SFT ────────────────┐           │
        │ only_assistant_loss + chat 模板         │◀──────────┤
        └────────────────────┬───────────────────┘           │
                             ▼                               │
        ┌───── Post Stage 2: On-policy Distillation ─┐       │
        │ verl_distillation/recipe/onpolicy_distill  │◀──────┤
        │  Teacher: Qwen3-1.7B(未扩词表)              │       │
        │  Student: 上一步 SFT 模型                   │       │
        └────────────────────┬────────────────────────┘      │
                             ▼                               │
        ┌───── Post Stage 3: GRPO RL ─────────────────┐      │
        │ verl_rl/recipe/onerec/                       │◀────┤
        │  Two-Stage rollout：先思考 → 再 beam SID     │      │
        └────────────────────┬─────────────────────────┘      │
                             ▼                               │
                       ┌───────────┐                          │
                       │ 评测 / 上线 │◀────────────────────────┘
                       │ benchmarks │
                       └───────────┘
```

---

## 1. 数据（`data/`）

### 1.1 通用文本
`data/general_text/{pretrain,sft}.csv` 列出所有 HuggingFace 数据集名 + 样本数。真实数据下载后放到 `raw_data/general_text/`。用途：Stage 2 co-pretrain 和 SFT 阶段拌进去，防止模型退化到只会 SID。

### 1.2 OneRec 业务数据（`data/onerec_data/`）

上游是 [OpenOneRec/OpenOneRec-RecIF](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF)，主要文件（在 `raw_data/onerec_data/` 下）：

| 文件 | 关键列 | 用途 |
| :--- | :--- | :--- |
| `onerec_bench_release.parquet` | uid, split, hist_video_pid, hist_video_{longview,like,follow,forward,not_interested}（bit mask）, target_video_pid, target_video_{...}, inter_user_profile_with_sid, inter_keyword_to_items | 主 metadata |
| `video_ad_pid2sid.parquet` | pid, sid (list[int]，长度 n_layers=3) | 视频/广告域的 tokenizer 产物 |
| `product_pid2sid.parquet` | pid, sid | 商品域 |
| `pid2caption.parquet` | pid, dense_caption | Item Understanding / rec_reason 用 |

#### 用户交互行为的样本组织形式（重点）

**结论：user-centric，不是 impression-centric。**

`onerec_bench_release.parquet` 里**一行 = 一个用户**（`uid` 唯一），把该用户"过去-未来"的一段行为流全部装在这一行里。所有历史 & 目标交互都以**并列的、等长的多路数组 + bit-mask** 方式组织，而不是"一次曝光/一次点击一行"。

一行大致长这样（下面 `L_h` 是历史长度，`L_t` 是目标长度，二者对同一用户内固定；不同用户可以不同）：

```
uid:                    int64                       # 用户 id
split:                  int8                        # 0=train, !=0=eval
inter_user_profile_with_sid: str                    # 预先生成的画像文本，SID 已嵌入
inter_keyword_to_items: dict[str, list[int]]        # {搜索词: [该词下用户交互的 item pid, ...]}

# ---- 历史段（用户过去交互过的 item 序列，按时间从旧到新排好）----
hist_video_pid:            list[int]   长度 L_h     # item pid 的时间序列（曝光位）
hist_video_longview:       list[0/1]   长度 L_h     # 与 hist_video_pid 逐位对齐的 bit mask
hist_video_like:           list[0/1]   长度 L_h     #   1 = 该位置的 pid 上发生了该行为
hist_video_follow:         list[0/1]   长度 L_h     #   0 = 未发生
hist_video_forward:        list[0/1]   长度 L_h
hist_video_not_interested: list[0/1]   长度 L_h

# ---- 目标段（用户接下来会交互的 item 序列，即预测目标）----
target_video_pid:            list[int]  长度 L_t
target_video_longview:       list[0/1]  长度 L_t
target_video_like:           list[0/1]  长度 L_t
target_video_follow:         list[0/1]  长度 L_t
target_video_forward:        list[0/1]  长度 L_t
target_video_not_interested: list[0/1]  长度 L_t
```

关键约定（在所有数据处理脚本里被反复利用，见 `data/onerec_data/sft/label_cond_rec.py:79-146` 和 `sft/label_pred.py:100-178`）：

* **主键是 pid 时间序列**（`hist_video_pid` / `target_video_pid`），它就是"用户被曝光过/将被曝光的视频序列"，一条曝光记录对应一个 pid 位置——所以一个 pid 位置**同时可能有多个行为**（一个视频既被 `longview=1` 又被 `like=1`），这正是 `hist_video_<action>` 用多个并列 mask 而不是单一 action 列的原因。
* **每一列 `hist_video_<action>` 是一个 0/1 数组，长度必须 `== len(hist_video_pid)`**；`mask[i] == 1` 表示 `hist_video_pid[i]` 上发生了 `<action>`。这套结构等价于稀疏的 (user, item, action) 交互矩阵，只不过按用户压成了一行。
* **曝光信息本身**：曝光对应"pid 出现在 hist/target 序列里"这个事实——mask 全 0 也算被曝光过；点击/长时观看/点赞/关注/转发/不感兴趣等更强的行为通过各 `_<action>` mask 表达。因此**没有单独的 "impression" 表**，所有曝光都嵌在 pid 序列里，"订购"这类交易行为在这份开源数据里没有单独列，被抽象为对应领域的 pid 序列（商品域走 `product_pid2sid`，广告域走 `video_ad_pid2sid`）。
* `inter_keyword_to_items` 用来做交互式推荐：把用户的搜索关键词映射到该词下用户交互过的 item pid 列表；每个 keyword 会展开成独立的 SFT 样本（见 `sft/interactive_rec.py`）。
* `inter_user_profile_with_sid` 是文本形式的用户画像（人写模板 + SID 嵌入），直接当 pretrain / SFT 的输入文本用。

数据脚本消费这种结构的通用模式（伪代码）：

```python
for row in parquet:                                   # 遍历用户
    hist_pids = row['hist_video_pid']                 # list[int] 长度 L_h
    for action in ['longview','like','follow','forward','not_interested']:
        mask = row[f'hist_video_{action}']            # list[0/1] 长度 L_h
        acted_pids = [p for p, m in zip(hist_pids, mask) if m == 1]
        acted_sids = [pid2sid[p] for p in acted_pids[-K:]]     # 取最近 K 个
        # → 拼成 "用户点赞过以下内容：<SID><SID>..." 之类的 user prompt 片段
```

也就是说，**每条训练样本对应"一个用户 + 一次目标预测"，而不是"一条曝光"**——这是 user-centric 组织的直接体现，天然适合 LLM 的长上下文序列建模（一整条用户行为流塞进一个 prompt）。相比 impression-centric（每次曝光一行）：

* 优点：user-centric 保留了完整时间序列，便于 pretrain 阶段做"给一段历史 → 续写下一段"的自回归目标；也便于用不同行为 mask 派生出多种 SFT 任务（`label_cond_rec` 就选一种行为作为目标输出，`label_pred` 就把 target 展平成 point-wise 分类）。
* 代价：单条样本更大更稀疏，需要 `HIST_MAX_LEN=512`、`TARGET_MAX_LEN=10` 这些常量做截断；要在数据脚本里手动展开成 (user, keyword) / (user, candidate) 这种更细粒度样本时，才会"分裂"成多条 SFT 样本。

#### ⚠ 开源版 vs. 技术报告描述的"New Impression Only"——一个重要差异

Kuaishou 的 OneRec 技术报告 / 后续论文里把样本组织形式分成三档：

| 方式 | 描述 | 问题 |
| :--- | :--- | :--- |
| User-Centric | 每个用户一条样本，整条历史 + target 全部拿来做自回归 loss | 时序泄漏、流行度偏差、**不适配增量/流式更新** |
| Naive Impression | 每次曝光独立一条样本，含完整历史 → target | 历史被重复展开进无数样本，计算冗余大 |
| **New Impression Only** | 每次曝光一条样本，但**只对新 target item 计算 loss**（历史只做 context） | 无泄漏、无冗余、可增量/流式，**报告声称是主流方案** |

**但本仓库的开源 pretrain 代码走的是 User-Centric，不是 New Impression Only**。证据在 `pretrain/onerec_llm/data/qwen3_dataset.py::_process_completion`：

```python
segments_text = f"{hist_sids}{target_sids}"      # video_rec.py 里把 hist + target 拼一起
inputs = self.tokenizer(segments_text, ...)
inputs["loss_mask"] = torch.ones_like(input_ids)
inputs["loss_mask"][..., -1] = 0                 # 只 mask 末尾 EOS
```

`loss_mask` 全 1 意味着 `hist_sids` 段里的每个 item token 也被当作预测目标算 loss。这就等价于对整条用户行为流做完整的自回归 next-token 预测——正是"User-Centric"的定义。

可能的解释（我们无法从代码本身完全证实，只是合理推测）：

1. **开源版是"研究友好"的静态 dump，不是线上生产管道**。真正线上的 OneRec 系统是流式训练：曝光一条、训一条、模型立刻更新。这种流式语义在把日志按用户压成静态 `onerec_bench_release.parquet` 之后就不复存在，只能按 user-centric 训。仓库 `README.md` 的 Roadmap 里也隐约暗示"One-click reproduction"、"Unified VeRL integration" 等工业化能力还在 TODO。
2. **Pro 版 vs. Standard 版**。技术报告里的四款模型（1.7B/8B/1.7B-Pro/8B-Pro）中，Pro 版明确写了 "enhanced with a hundred-billion-token industrial corpus"，用的是内部流式 impression 数据管道；Standard 版训在开源静态数据上，正是这个仓库能看到的东西。
3. **"New Impression Only" 可能特指 RL 阶段**。`verl_rl/recipe/onerec/onerec_vllm_rollout.py` 的 two-stage rollout 确实只在 stage2 生成的 target SID 上产 reward、计算 GRPO 优势——从这个视角看 RL 阶段是接近 New Impression Only 的。但 pretrain / SFT 阶段的静态样本组织在开源代码里仍是 user-centric。

要把开源 pretrain 改成 New Impression Only 其实很轻量：

* 把 `data/onerec_data/pretrain/video_rec.py` 的 segments 格式改成 messages（`user=hist_sids`, `assistant=target_sids`），
* dataset 走 `_process_chat` 分支即可——`_get_assistant_mask` 会自动只在 `<|im_start|>assistant\n...<|im_end|>` 段设 `loss_mask=1`。

顺带一提：**开源 SFT (`data/onerec_data/sft/video_rec.py`) 已经是这种 messages + `only_assistant_loss=true` 结构，效果上更接近简化版 New Impression Only**（只不过把 target 段的 10 个新 item 一次性作为 assistant 输出，而不是每次曝光单独训一步）。所以更精确的定位是：

* 开源版 pretrain = **User-Centric**（full-sequence loss）
* 开源版 SFT     ≈ **简化版 New Impression Only**（target 段 loss，静态 batch 而非流式）
* 内部工业版     = **New Impression Only + 真·流式增量**（本仓库未开源）

各数据脚本从这一行 user-centric 记录派生出的样本粒度：

| 脚本 | 一行用户 → 生成几条训练样本 | 派生规则 |
| :--- | :--- | :--- |
| `pretrain/video_rec.py` | 1 条 | 一整段 `hist_pids[-512:] + target_pids[:10]` 拼一起 |
| `pretrain/user_profile.py` | 1 条 | 直接用 `inter_user_profile_with_sid` |
| `pretrain/item_understand.py` | 1 条（按 pid 遍历） | 每个 (pid, caption) 一条 |
| `sft/video_rec.py` | 1 条 | history → target 的 chat 三段 |
| `sft/label_cond_rec.py` | 1 条 | 从可用的 target 行为里随机选一个 |
| `sft/label_pred.py` | **N 条**（N = 有效候选数） | `target_pids[:10]` 每个候选一条 point-wise 分类 |
| `sft/interactive_rec.py` | **N 条**（N = keyword 数） | `inter_keyword_to_items` 每个 keyword 一条 |
| `sft/ad_rec.py` / `product_rec.py` | 1 条 | 混合视频历史 + 广告/商品历史，多域拼接 |


`data/onerec_data/{pretrain,sft}/*.py` 每个脚本对应一个训练任务：

| 任务 | 阶段 | 数据格式 | Assistant 输出 |
| :--- | :--- | :--- | :--- |
| `pretrain/video_rec.py` | Pretrain | segments (纯文本) | `HIST_SIDS + TARGET_SIDS` |
| `pretrain/user_profile.py` | Pretrain | segments | 直接把 `inter_user_profile_with_sid` 放进去 |
| `pretrain/item_understand.py` | Pretrain | segments | SID + caption 模板拼接 |
| `sft/video_rec.py` | SFT | messages (chat) | TARGET_SIDS |
| `sft/ad_rec.py` | SFT | messages | 目标广告 SIDs |
| `sft/product_rec.py` | SFT | messages（含跨域） | 目标商品 SIDs |
| `sft/interactive_rec.py` | SFT | messages | keyword → items |
| `sft/label_cond_rec.py` | SFT | messages | 指定行为下的 target SIDs |
| `sft/label_pred.py` | SFT | messages | "是" / "否"（长时观看二分类） |
| `sft/item_understand.py` | SFT | messages | 给 SID 生成 caption |
| `sft/rec_reason.py` | SFT | messages | 带 `<think>...</think>` 的推理链 |

**SID 编码约定**（所有脚本共享）：
```python
SID_FORMAT = '<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>'
```
一条 item 会占 **5 个 token**：2 个特殊边界 + 3 个 codebook token（对应 n_layers=3）。

**总入口**：`data/onerec_data/run.sh` 依次调用所有任务脚本，生成 `output/{pretrain,sft}_{task}.parquet`。

### 1.3 分片与混合（`data/scripts/`）

* `prepare_pretrain.sh` / `prepare_sft.sh` → 调用 `split_data.py`：把上一步产出的 parquet + 通用文本 parquet 混合、shuffle，按 `MAX_ROWS=1000` 分片为 `part-00000-of-XXXXX.parquet`。同时生成 `shardlist.json`（供 WebDataset 读）。
* `prepare_distillation.sh` → 从通用文本里抽 `NUM_SAMPLES` 样本作蒸馏输入。
* `prepare_rl.sh` → 合并 5 个 SFT 任务的 parquet，按 `TEST_SIZE=1000` 切 train/test，供 GRPO 用。

---

## 2. Itemic Tokenizer（`tokenizer/`）

**目的**：把每个 item 的连续 embedding（Qwen3-Embedding-8B 输出的 4096-d 向量）压缩成 3 个整数 code。code 加上词表偏移就是 LLM 词表中的 Itemic Token id。

### 2.1 模型：`res_kmeans.py::ResKmeans`
残差量化（Residual Vector Quantization）：
1. 第 0 层：对原始 x 做 K-Means，得到最近质心 → 记 code $c_0$，残差 $r_0 = x - centroid_{c_0}$。
2. 第 1 层：对 $r_0$ 做 K-Means → $c_1$，$r_1 = r_0 - centroid_{c_1}$。
3. 依此类推 n_layers 层。

关键结构：
* `self.centroids[l]`: `nn.Parameter`, shape=`(codebook_size, dim)` = `(8192, 4096)`。共 3 层。
* `encode(x)`: `(B, dim) → (B, n_layers)`，见 `res_kmeans.py:40-54`。
* `decode(code)`: `(B, n_layers) → (B, dim)`，各层质心相加。

### 2.2 训练：`train_res_kmeans.py`
* 读入若干 parquet 里的 `embedding` 列 → `(N, 4096)`。
* 用 `faiss.Kmeans` 逐层聚类；每层结束把残差 x 更新为 `x - 本层选中的质心`。
* 保存 `state_dict` 到 `{model_path}/model.pt`。

### 2.3 推理：`infer_res_kmeans.py`
* 读 `(pid, embedding)` parquet，`model.encode(batch)` → `(N, 3)`。
* 写 `(pid, codes)` parquet；用于后续把 codes 灌进词表扩展脚本。
* 抽样估算重建 MSE 作为 sanity check。

### 2.4 扩词表：`pretrain/tools/model_converter/expand_qwen3_vocab.py`
把 3 × 8192 = 24576 个 Itemic Token 加进 Qwen3 tokenizer：
```
<s_a_0>, <s_a_1>, ..., <s_a_8191>,
<s_b_0>, ..., <s_b_8191>,
<s_c_0>, ..., <s_c_8191>,
<|sid_begin|>, <|sid_end|>
```
顺序**必须严格**（否则和数据脚本对不上）。同时把词表大小对齐到 256 的倍数，`resize_token_embeddings` 补零。新的 base model 保存到 `output_model_dir`，作为 Stage 1 训练的起点。

---

## 3. Pretrain / SFT 训练（`pretrain/`）

### 3.1 总入口：`recipes/train_qwen3.py`
一份脚本用同一个 `train()` 函数覆盖 4 种训练场景，靠命令行开关切换：

| 场景 | 关键参数 |
| :--- | :--- |
| Stage 1 (Itemic-Text Alignment) | `--freeze_llm --start_optimize_embedding_index 151669` |
| Stage 2 (Full Co-Pretrain) | 不加 --freeze_llm |
| SFT | 数据配置改成 `sft.json`，其中 `only_assistant_loss=true` |
| Continual pretrain | 加 `--resume_from ... --resume_training_state` |

启动方式：MPI 起多进程，各进程从 `OMPI_COMM_WORLD_*` 拿 rank/world_size，用 FSDP 全参数分片。

**主循环（简化版，见 `train_qwen3.py:1174-1275`）：**
```python
while True:
    batch = next(data_iter)                          # 详见 §3.2
    to_cuda(batch)
    loss, per_token_loss = compute_forward_backward( # 详见 §3.3
        model, batch, ...)
    grad_norm = compute_fsdp_zero2_grad_norm(model)
    optimizer.step()
    if start_optimize_embedding_index > 0:
        embedding_masker.restore_frozen_params()     # Stage 1 关键：把冻结的
                                                      # embedding 行拷回去
    lr_scheduler.step(); optimizer.zero_grad()
    if global_step % logging_per_step == 0: 打日志
    if 到点: save_checkpoint()                        # DCP + optimizer + dataloader state
```

### 3.2 数据 pipeline：`onerec_llm/data/`

三层结构：

```
sources (JSON 文件列表 / 目录)
    ↓
Qwen3NaiveParquetDataset  ── 按 rank×worker 切文件，读一行
    ↓                       (使用 LocalShuffleBuffer 做 hash-based 本地 shuffle)
Qwen3ChatCompletionParquetDataset._process
    ↓  segments 模式  → _process_completion → tokenize，loss_mask 全 1
    ↓  messages 模式  → _process_chat        → apply_chat_template + _get_assistant_mask
_packing (sample packing)
    ↓
collate → 1 个 batch 就是 1 条长度 T ≈ max_length 的打包序列
```

#### Pretrain 阶段一个 batch 到底长什么样？

先说一句最简短的：**每个 GPU 每步只处理一个 batch，而这个 batch 只包含"1 条"长度 T ≈ `max_length`（默认 30000）的序列——这条序列是由若干条短样本头尾相接拼接（sample packing）而成的**。也就是说 `batch_size=1`，真正的"多样本"在序列内部通过 `cu_seqlens` 边界标出。

以 pretrain 阶段视频推荐任务（`data/onerec_data/pretrain/video_rec.py`）为例，一条源样本就是**一整个用户**的行为流的 token 化，形如：

```text
<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|>   ← hist item 1（5 tokens/SID）
<|sid_begin|><s_a_...>...<|sid_end|>                     ← hist item 2
...  (最近 HIST_MAX_LEN=512 个历史 item → 512 × 5 = 2560 个 token)
<|sid_begin|><s_a_...>...<|sid_end|>                     ← target item 1
...  (前 TARGET_MAX_LEN=10 个 target item → 50 个 token)
<pad_token>                                              ← 收尾 EOS
```

一条源样本的长度 `L_i ≈ 2610` 个 token（不同用户有波动）。dataloader 会一直向 `buffer` 追加这样的源样本，直到再加一条就要越过 `max_length=30000`——这时触发一次 `_packing`：

* 把 buffer 里 ~11 条源样本沿 seq 维 `torch.cat` 成一条 `(1, ~28600)` 的长序列；
* 尾部 pad 到 `T = ceil(max_length/8)*8 + 64`（≈ 30 016）；
* 在 `cu_seqlens` 里记下每条源样本的边界 `[0, L_0, L_0+L_1, ..., sum_L, T]`；
* 在 `sample_idx` 里给每个 token 打上"属于 packing 内第几条源样本"的标签；
* `yield` 这一整块 → 就是一个 batch。

所以**一个 batch 内部对应 S 个用户**（S 由 `cu_seqlens` 长度决定，通常 10~15），每个用户各自有独立的 attention mask（借 FlashAttention 的 `cu_seqlens` 变长机制实现，各用户之间**互不 attention**）。这样 pretrain 的自回归目标"给一段历史 → 续写下一段"就落在了单个用户序列内部；跨用户的 loss 通过 packing 只是并行加速，语义上等价于一次训练一个用户一次。

**Batch 里每个 tensor 的形状（`T = ceil(max_length/8)*8 + 64`，`S` = packing 内的源样本数）：**

| 字段 | 形状 | 含义 |
| :--- | :--- | :--- |
| `input_ids` | `(1, T)` int64 | packing 后的 token 序列，末尾用 `pad_token_id` 填 |
| `position_ids` | `(1, T)` | 每条子样本内部 0..L_i-1；padding 段=0 |
| `loss_mask` | `(1, T)` int | 1 = 计入 loss，0 = 忽略（padding / 非 assistant / EOS）。Pretrain 阶段除末尾 pad 之外全 1；SFT 阶段只有 assistant 段是 1 |
| `itemic_id_mask` | `(1, T)` int | 1 = 该 token 落在 `itemic_id_range=[151669, 176246]` 内，用于分开统计 itemic/text loss |
| `cu_seqlens` | `(S+2,)` int32 | packing 边界：`[0, L_0, L_0+L_1, ..., sum_L, T]`。喂给 FlashAttention 变长 attention |
| `sample_idx` | `(1, T)` int32 | 每个 token 属于打包块里的第几条源样本；padding 段=-1，用于按 data_source 拆 loss 统计 |
| `epoch_idx` | `(1,)` float32 | 平均 epoch 编号 |
| `data_source` | `list[str]`, 长 S | 每条源样本的 source 字段（如 `RecIF_VideoRec_Pretrain`） |

关于两个"batch 大小"的澄清（很容易混淆）：

* **`--minibatch_size 4096`**：不是一个数据 batch 的大小，而是 `ChunkedLossComputer` 把 seq 维切 chunk 时每 chunk 的长度——用来省 lm_head 的显存，与样本组织无关。
* **`--max_length` (=30000)**：单卡单步真正处理的 token 数上限，即上面那条打包序列的长度 T。
* **有效梯度累积**：因为 `batch_size=1`（每卡每步 1 条 packed 序列 ≈ 10+ 个用户），全局有效批 = `world_size × 平均 S`。没有额外的 grad accumulation 步——DP 通过 FSDP 完成。

#### SFT 阶段一个 batch 长什么样？

结构和 pretrain **完全一致**（同一份 `Qwen3ChatCompletionParquetDataset` 类），差别仅在：

1. 源样本走 `_process_chat` 而不是 `_process_completion` → 输入文本是 `apply_chat_template(messages)` 的产物，形如：
   ```text
   <|im_start|>system\n{系统提示词}<|im_end|>\n
   <|im_start|>user\n{用户提示 + 历史 SID}<|im_end|>\n
   <|im_start|>assistant\n{目标 SID / 分类答案 / caption}<|im_end|>\n
   ```
2. `loss_mask` 由 `_get_assistant_mask` 生成——只在 `<|im_start|>assistant\n...<|im_end|>\n` 内部标 1，system/user 部分标 0；这就是配置里 `only_assistant_loss=true` 的实现方式。
3. SFT 数据本身粒度更细（见 §1.2 尾部的派生表：一个用户可以展开为 N 条候选/关键词样本），所以 packing 里一个 batch 通常包含**更多、更短**的源样本（几十条量级）。

**关键子函数：**
* `_get_assistant_mask` (`qwen3_dataset.py:203-255`)：扫 token 序列找 `<|im_start|>assistant\n...<|im_end|>\n` 之间的位置标 1；SFT 阶段 loss_mask 就是它。
* `_packing` (`qwen3_dataset.py:436-488`)：多条短样本 concat 成一条 ≈ max_length 的长序列，配合 cu_seqlens 用 FlashAttn 变长 attention，避免 padding 浪费。
* `LocalShuffleBuffer` (`local_shuffle_buffer.py`)：SortedDict + MD5 hash 索引实现的固定容量 buffer；边填边随机弹出，做样本级 shuffle。

### 3.3 Forward / Backward：`compute_forward_backward`（`train_qwen3.py:683-757`）

```
input_ids: (1, T) → 把 ≤0 位置清零（防止 embedding 查表出错）
     ↓
Qwen3ForCausalLM(input_ids, cu_seqlens=..., position_ids=...)
     ↓
logits: (1, T, V)             V ≈ 176 000（对齐 256 后的扩词表大小）
     ↓
labels = concat(input_ids[:, 1:], pad)  →  (1, T)   # 自回归右移
labels = labels * loss_mask + (-100) * (1 - loss_mask)  # 屏蔽非监督位
     ↓
CrossEntropyLoss(reduction='mean', ignore_index=-100)
     ↓
loss (scalar), per_token_loss (T,)   →  loss.backward()
     ↓
clip_grad_norm(model, max_grad_norm)
```

**显存优化**：如果 `--use_chunked_loss_computer`，则用 `ChunkedLossComputer`（`losses/ce.py:72-199`），把 seq 维切成 `minibatch_size=4096` 的 chunk，逐个走 lm_head 前向 + `torch.autograd.grad` 反传，梯度手动累加到 lm_head 权重和输入。避免一次性物化 (T, V) 大张量。

### 3.4 Stage 1 的核心机制：`EmbeddingGradientMasker`（`training/gradients.py`）

**问题**：Stage 1 只想优化 id ≥ 151669 的 Itemic Token embedding，其他 id 的原 Qwen3 embedding 必须完全不变。单纯"梯度置零"不够（AdamW 的动量、weight decay 还会污染）。

**做法**：
1. 构造时把每 rank 本地要冻结的 embedding 行 `clone()` 保存到 `self.saved_weights`（FSDP 下 `param.to_local()` 拿本地分片）。
2. 每次 `optimizer.step()` 之后调 `restore_frozen_params()`，把保存的行 `copy_()` 回去，等于把 optimizer 的更新在这段行上撤销。
3. lm_head 与 embed_tokens tie weight 时，两处都要恢复。

### 3.5 Checkpoint：`training/checkpoint.py`
* 使用 `torch.distributed.checkpoint`（DCP）保存分片模型。
* 目录结构：`output_dir/step{N}/global_step{N}/{app/, optimizer_ckpt/, dataloader_ckpt/}`。
* Resume：`--resume_from ... --resume_training_state` 恢复模型 + optimizer + scheduler + dataloader 位置（`Qwen3NaiveParquetDataset` 用一个 `finish_dict` 位图记录每个文件里已消费的样本 index）。

### 3.6 Checkpoint → HuggingFace：`tools/model_converter/convert_checkpoint_to_hf.py`
* 读 DCP checkpoint 到 CPU；提取 `app.model` 子字典。
* 分片保存为 `model-XXXXX-of-YYYYY.safetensors`（每 shard ≤ 5 GB），写 `model.safetensors.index.json`。
* 从原 HF 目录复制 tokenizer / config / chat_template。
* 转出的模型可直接用 `AutoModelForCausalLM.from_pretrained` 加载，供 SFT / 蒸馏 / RL / 评测。

---

## 4. Post-training（`verl_distillation/`, `verl_rl/`）

这两个目录是 [verl](https://github.com/volcengine/verl) 框架的 fork，绝大多数代码是 verl 通用实现，OpenOneRec 只是**在 `recipe/` 下新增了 OneRec 定制版**：

### 4.1 On-policy Distillation：`verl_distillation/recipe/onpolicy_distill/`

* `run_qwen3_distill.sh`：启动脚本。关键环境变量：
  * `BASE_MODEL` = OneRec Stage 2 出来的扩词表模型（student）
  * `TEACHER_MODEL` = 原版 Qwen3-1.7B（未扩词表）
  * `EXTEND_VOCAB_START_TOKEN=151669`：教师模型没有 Itemic Token 词表，因此对 id ≥ 151669 的 rollout token 特殊处理（`mask_response_if_have_extend_token` 决定是否整条 mask）。
* `main_onpolicy_distill.py`：Hydra 入口，调 verl 的 PPO 训练器但把 advantage estimator 换成 `on_policy_distill`。
* **动机**：Stage 2 后模型在推荐任务上很强，但通用能力（数学/推理/编码）掉了。用原版 Qwen3 当教师做 on-policy 蒸馏可以"补回"通用能力，同时不动 Itemic Token 相关的分布。

### 4.2 GRPO RL：`verl_rl/recipe/onerec/`

* `run_grpo.sh`：算法 `adv_estimator=grpo`，rollout backend `two_stage`（下详），reward 用 `onerec_recipe.py::compute_score`（基于 SID 是否命中 ground truth）。
* `onerec_recipe.py::OneRecDataset`：读取 `output/rl_data/train.parquet`，每条样本给 verl 一个 prompt + source（作为 reward key）。
* `onerec_recipe.py::compute_score`：把 rollout 出的 SID 和 ground_truth SID 做 Pass@k / Recall@k 计算奖励（同 §5.2 的评测口径）。
* **两阶段 rollout**（`onerec_vllm_rollout.py`）：
  1. Stage 1：用 vLLM 让模型生成 <think>...</think> 结束（最多 `STAGE1_MAX_TOKENS=1024` token）。
  2. Stage 2：在 `<|sid_begin|>` 后走 **beam search**，beam=`STAGE2_BEAM_SIZE=32`，每次只生成 `STAGE2_NUM_TOKENS=3`（正好一条 SID 的 3 个 codebook token）。产出 32 条候选 SID。
  3. 32 条 SID 一起过 reward function 得 32 个分数，喂给 GRPO 做 group 相对优势估计。
* 训练策略：actor & ref 都用 FSDP2；`use_kl_loss=True, kl_loss_coef=0.001` 防漂移。

---

## 5. 评测（`benchmarks/`）

评测是**两阶段**的："生成"和"打分"完全解耦。

### 5.1 生成阶段：`benchmarks/scripts/ray-vllm/evaluate.py`

被 `eval_script.sh` 调起。基础设施：
* Ray 起集群（可选多节点，`init_ray_cluster.sh`）。
* `RayVllmGenerator`：内部起若干 vLLM worker，每个 worker 拿 `tensor_parallel_size` 张卡切模型；生成时把 prompts 分片喂给各 worker 并行推理。

调用流程：
```python
Benchmark(model_path, task_types=[...], splits=["test"], enable_thinking=...)
   .run(generator=RayVllmGenerator(...), output_dir=...)
```
`Benchmark.run` 遍历 (task, split)，调用 `GenerationRunner`：

1. `data_loader.load_data(task_name, split, sample_size)` → 返回 `{sample_id: {"prompt": str, "ground_truth": str, "metadata": {...}}}`。
2. `generator.generate(prompts, num_return_sequences=..., ...)` → `generations: {sid: list[str]}`, `logprobs: {sid: list[float]}`（每个 prompt 生成 K 条候选，附带各自的 cumulative logprob）。
3. 收集 `input_tokens/output_tokens/times` 用于算 MFU。
4. 写入 `{output_dir}/{model_name}/{task}/{split}_generated.json`。

**关键点**：推荐类任务通常 `num_return_sequences=128`（beam × diverse），一次生成给出 Top-128 候选，后续算指标时按需截 k。

### 5.2 打分阶段：`benchmarks/scripts/eval_dev_results.py`（或 `Benchmark.evaluate_dev`）

按 registry 查找每个任务对应的 `evaluator`，加载 `*_generated.json`，算指标写到 `eval_results.json`。

| 任务 | Evaluator | 主指标 |
| :--- | :--- | :--- |
| video / product / ad / label_cond | `RecommendationEvaluator` | Recall@32 / Pass@K / Position1_Pass@K，SID 空间 + PID 空间双路 |
| interactive | `RecommendationEvaluator` | 同上，只是 prompt 里有 keyword |
| label_pred | `LabelPredEvaluator` | AUC（用 `<是>`/`<否>` 的 logprob 差算 P 再算 AUC） |
| item_understand | `ItemUnderstandEvaluator` | macro_wip_double_weighted_f1（外部 LLM Judge，`api/gemini.py`） |
| rec_reason | `RecoReasonEvaluator` | llm_score 0-5（外部 LLM Judge，取 `</think>` 之后的正文打分） |

**推荐类核心逻辑**（`RecommendationEvaluator._evaluate_single_mode`, `recommendation/evaluator.py:105-284`）：
1. 按 `select_k` 策略排序 K 条 generation：`first_k` 保留原序；`top_k_by_logprobs` 按 cumulative logprob 降序去重。
2. 从每条 generation 里 regex 抽出 SID → `predicted_ids`。
3. 对每个 k：
   * `Pass@k`   ：`predicted_ids[:k]` 命中任一 ground truth SID
   * `Position1_Pass@k`：`predicted_ids[:k]` 命中 ground truth 的第一个 SID
   * `Recall@k`：交集 / |ground_truth|
4. `evaluation_mode='both'` 时会再把 SID 映射回真实 pid（`sid2pid.json`，一 SID 通常映射多个 pid，`most_popular` 挑最热）在 pid 空间上再算一遍指标。

---

## 6. 快速索引：关键脚本入口

> 简表版本；每一步的具体参数、上下游依赖、路径填法见 §9 端到端复现手册。

| 目的 | 命令 |
| :--- | :--- |
| 训练 Tokenizer | `python tokenizer/train_res_kmeans.py --data_path ... --model_path ./ckpt` |
| Tokenizer 推理 | `python tokenizer/infer_res_kmeans.py --model_path ... --emb_path ...` |
| 扩 Qwen3 词表 | `bash pretrain/scripts/expand_qwen3_vocab.sh` |
| 生成 RecIF 训练数据 | `cd data/onerec_data && bash run.sh` |
| Pretrain 数据分片 | `bash data/scripts/prepare_pretrain.sh` |
| SFT 数据分片 | `bash data/scripts/prepare_sft.sh` |
| Stage 1 训练 | `bash pretrain/examples/pretrain_stg1.sh` |
| Stage 2 训练 | `bash pretrain/examples/pretrain_stg2.sh` |
| SFT | `bash pretrain/examples/posttrain_sft.sh` |
| Checkpoint → HF | `bash pretrain/scripts/convert_checkpoint_to_hf.sh <base> <out> <step>` |
| 蒸馏 | `cd verl_distillation && bash recipe/onpolicy_distill/run_qwen3_distill.sh` |
| RL (GRPO) | `cd verl_rl && bash recipe/onerec/run_grpo.sh` |
| 评测 | `cd benchmarks && bash eval_script.sh <model_path> <name> <enable_thinking>` |

---

## 7. 关键常量速查

| 名称 | 值 | 出处 |
| :--- | :--- | :--- |
| Itemic Token 起始 id | 151669 | Qwen3 原词表大小 |
| Itemic Token 数量 | 3 × 8192 + 2 = 24578 | n_layers × codebook_size + `<\|sid_begin\|>` + `<\|sid_end\|>` |
| Itemic id range | `[151669, 176246]` | dataset_config，用于统计 itemic vs text loss |
| 单条 SID 的 token 数 | 5 | `<\|sid_begin\|> + s_a + s_b + s_c + <\|sid_end\|>` |
| Codebook 维度 | 4096 | 与 Qwen3-Embedding-8B 保持一致 |
| Codebook 层数 | 3 | s_a, s_b, s_c |
| 每层 codebook 大小 | 8192 | 见 `expand_qwen3_vocab.sh` |
| 最大历史长度（视频） | 512 | `HIST_MAX_LEN` in `data/onerec_data/**/*.py` |
| 最大目标长度 | 10 | `TARGET_MAX_LEN` |
| Pretrain packing 长度 | ≈ max_length | `dataset_config/*.json::max_length`，一般 30000 |

---

## 8. 二次深入的推荐路径

不同角色关注的重点不同：

* **想复现训练** → 依次读 `data/onerec_data/run.sh` → `data/scripts/prepare_*.sh` → `pretrain/examples/pretrain_stg{1,2}.sh` → `pretrain/examples/posttrain_sft.sh`。
* **想改模型架构** → `pretrain/onerec_llm/models/qwen3/modeling_qwen3.py`（HF Qwen3 的一份 fork，加了 flash-attn + chunked loss 钩子）。
* **想改数据构造** → 挑一个 `data/onerec_data/{pretrain,sft}/*.py` 参照改；主要就是选原始列 + 拼 prompt 模板。
* **想加新评测任务** → 在 `benchmarks/benchmark/tasks/v1_0/` 下建目录，实现 `loader.py`（继承 `BaseLoader`）+ `evaluator.py`（继承 `BaseEval`）+ `config.py`（放 prompt/generation config），最后在 `registry.py` 注册。
* **想理解 RL rollout** → `verl_rl/recipe/onerec/onerec_vllm_rollout.py` 是两阶段 rollout 的核心；`onerec_recipe.py::compute_score` 是 reward。

---

## 9. 从零跑通一次完整 OneRec 流程（端到端复现手册）

> 这一节是"手把手把流程跑起来"的操作指南。前面 §1–§8 讲的是"每块代码在干嘛"，这里讲"实操依次跑什么命令"。
>
> **前提说明（重要）**：完整 pretrain / SFT / RL 是**多机多卡**任务（原脚本假设 8 节点 × 8 GPU × A100/H100 级别，MPI 编排），单机个人机器只能跑通"最小可复现"版本 —— 也就是数据准备 + Tokenizer + 走一遍 pretrain 少量步数 + 评测。生产级完整训练的资源需求参考 [OneRec 技术报告](https://arxiv.org/abs/2512.24762)。
>
> 也可以完全跳过训练环节，直接从 HuggingFace 拉预训练好的 [OneRec-1.7B / 8B](https://huggingface.co/OpenOneRec) 只跑评测，这是最快的路径。

### 9.0 三条最短通路（先决定你要哪一条）

| 目标 | 步骤 | 大致耗时 | 硬件门槛 |
| :--- | :--- | :--- | :--- |
| A. 只想看效果 | 拉官方模型 → 直接跑 `benchmarks/eval_script.sh` | 几小时 | 1–8 张 A100/H100 |
| B. 想跑通训练 pipeline（学习目的） | 数据准备 → 扩词表 → Stage 1 少量步数（可关掉 MPI 走 torchrun） | 半天到一天 | 至少 1 台 8×A100 |
| C. 完整复现 1.7B / 8B 训练 | 全流程 + 蒸馏 + RL | 数周 | 数十节点 8×H100 |

下面按 **B**（最典型）展开，同时说明 **A / C** 在哪一步分岔。

### 9.1 环境准备（step 0）

**基础依赖**（所有阶段共用）：

```bash
# 建议为不同阶段用独立 conda 环境，因为 pretrain / verl / benchmark 的依赖版本不一样
# 这里以 pretrain 环境为例：
conda create -n onerec_pretrain python=3.10 -y
conda activate onerec_pretrain
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.51.0 accelerate flash-attn webdataset \
            torchdata sortedcontainers easydict pyarrow pandas tqdm safetensors \
            faiss-gpu tensorboard
```

**MPI（Pretrain 分布式训练必需）**：
- `pretrain/examples/*.sh` 通过 OpenMPI 跨节点起进程。单机复现如果不想装 MPI，可以把 `mpirun ...` 那段替换成 `torchrun --nnodes=1 --nproc_per_node=8 recipes/train_qwen3.py ...`，并给脚本手动设置 `OMPI_COMM_WORLD_RANK / SIZE / LOCAL_RANK` 三个环境变量（`recipes/train_qwen3.py::initialize_distributed` 读的就是这三个）。
- 完整安装参考 [`pretrain/README.md`](../pretrain/README.md)。

**基础模型**：需要一份原版 Qwen3 HF checkpoint 作为起点，从 [HuggingFace](https://huggingface.co/Qwen) 拉：

```bash
mkdir -p ./raw_data/hf_models && cd ./raw_data/hf_models
# 小模型验证 pipeline 用 0.6B，正式训练用 1.7B / 8B
hf download Qwen/Qwen3-1.7B --local-dir ./Qwen3-1.7B
```

**数据**（3 份 HuggingFace dataset）：

```bash
export HF_TOKEN=<你的 token>

# 从项目根目录执行（下同）
hf download OpenOneRec/OpenOneRec-General-Pretrain --repo-type dataset \
    --token $HF_TOKEN --local-dir ./raw_data/general_text/pretrain
hf download OpenOneRec/OpenOneRec-General-SFT --repo-type dataset \
    --token $HF_TOKEN --local-dir ./raw_data/general_text/sft
hf download OpenOneRec/OpenOneRec-RecIF --repo-type dataset \
    --token $HF_TOKEN --local-dir ./raw_data/onerec_data
```

下载完你的目录结构应是：

```
OpenOneRec/
├── raw_data/
│   ├── hf_models/Qwen3-1.7B/                     # 基础模型
│   ├── general_text/{pretrain,sft}/*.parquet     # 通用文本
│   └── onerec_data/                              # RecIF 业务数据
│       ├── onerec_bench_release.parquet
│       ├── video_ad_pid2sid.parquet
│       ├── product_pid2sid.parquet
│       ├── pid2caption.parquet
│       └── benchmark_data/                        # 评测集
├── tokenizer/ pretrain/ data/ benchmarks/ verl_rl/ verl_distillation/
└── notes/
```

### 9.2 Step 1 — 训练 Itemic Tokenizer

**如果直接用官方模型的 tokenizer（推荐 & 最省事）：**
从 [OpenOneRec/OneRec-tokenizer](https://huggingface.co/OpenOneRec/OneRec-tokenizer) 拉 tokenizer 权重和 `pid2sid.parquet`，跳过 §9.2 剩余步骤。

**如果要自己训练 tokenizer**（例如换了 item 集合）：

```bash
# 前提：先用 Qwen3-Embedding-8B 对所有 item 打出 embedding，
# 存成 parquet（列: pid, embedding），假设放在 ./raw_data/item_embeddings/
cd tokenizer

# (1) 训练残差 K-Means（faiss GPU，1 卡即可，~1 小时 for 百万级 item）
python train_res_kmeans.py \
    --data_path ../raw_data/item_embeddings/ \
    --model_path ../output/tokenizer_ckpt \
    --n_layers 3 \
    --codebook_size 8192 \
    --dim 4096 \
    --niter 20 \
    --seed 42

# (2) 用训好的 codebook 把每个 pid 编码成 3 元组 SID
python infer_res_kmeans.py \
    --model_path ../output/tokenizer_ckpt/model.pt \
    --emb_path ../raw_data/item_embeddings/all.parquet \
    --output_path ../raw_data/onerec_data/video_ad_pid2sid.parquet \
    --batch_size 10000 \
    --device cuda
```

产出：`pid2sid.parquet`（列 `pid`, `codes`），下游数据脚本会读它。**注意开源的业务数据集里已经带了 `video_ad_pid2sid.parquet` 和 `product_pid2sid.parquet`，用它们就够了，不必自己训**。

### 9.3 Step 2 — 扩 Qwen3 词表

给原版 Qwen3 加 `3 × 8192 + 2 = 24578` 个 Itemic Token（含 `<|sid_begin|>` / `<|sid_end|>`）：

```bash
cd pretrain

# 编辑 scripts/expand_qwen3_vocab.sh，把这三个改成你实际的路径：
#   HF_MODEL_DIR=<项目路径>/raw_data/hf_models/Qwen3-1.7B
#   OUTPUT_MODEL_DIR=<项目路径>/raw_data/hf_models/Qwen3-1.7B_itemic
#   ITEMIC_LAYER_N=3
#   VOCAB_SIZE_PER_LAYER=8192

bash scripts/expand_qwen3_vocab.sh
```

产出：`Qwen3-1.7B_itemic/` —— 一份新的 HF checkpoint，词表 = 原 151669 + 24578，向上对齐到 256 后 ≈ 176 384；新加的 embedding 行按 HF 的 `resize_token_embeddings` 默认初始化方式随机初始化。**这份 checkpoint 是后面 Stage 1 训练的起点**。

### 9.4 Step 3 — 生成训练数据

分为三步：先把原始 metadata 变成任务级 parquet，再和通用文本混合分片，最后为 RL 准备 train/test 切分。

```bash
cd data

# (1) 从 raw_data/onerec_data/*.parquet 派生所有 pretrain + SFT 任务的样本
#     产出：output/pretrain_video_rec.parquet, output/sft_video_rec.parquet, ... 共 11 个
#     耗时：几分钟到几十分钟
cd onerec_data && bash run.sh && cd ..

# (2) 把 (1) 的 pretrain 数据 + general_text/pretrain 混合，按 1000 行一分片
#     产出：output/split_data_pretrain/*.parquet + file_list.json
bash prepare_pretrain.sh

# (3) 同理，SFT 数据
#     产出：output/split_data_sft/*.parquet + file_list.json
bash prepare_sft.sh

# (4) RL 用（可选，做完 SFT 才需要）：从 5 个 SFT 任务里切出 train/test
#     产出：output/rl_data/{train,test}.parquet
bash prepare_rl.sh

# (5) 蒸馏用（可选）：从通用文本里抽 20 万样本
bash prepare_distillation.sh
```

**下游需要检查**：`pretrain/examples/dataset_config/pretrain.json::sources` 指向的路径 = `../output/split_data_pretrain/file_list.json`（相对 `pretrain/` 目录），需要匹配 §9.4-(2) 的输出。同理 `sft.json`。同时把这两个 JSON 里的 `base_model_dir` 改成 §9.3 产出的 `Qwen3-1.7B_itemic` 绝对路径。

### 9.5 Step 4 — Pretrain Stage 1（Itemic-Text Alignment）

**只解冻新加的 Itemic Token embedding**，其余参数完全冻结。让新词表和原 LLM 表征对齐。

```bash
cd pretrain

# 编辑 examples/pretrain_stg1.sh，把这几行改成你的实际路径：
#   MODEL_DIR=<项目路径>/raw_data/hf_models/Qwen3-1.7B_itemic
#   OUTPUT_DIR=<项目路径>/output/model_output/stg1
#   --num_training_steps 2000    ← 完整训练用，快速跑通改成 50~200

# 关键 flag（脚本里已写好）：
#   --freeze_llm                             ← 冻结 LLM
#   --start_optimize_embedding_index 151669  ← 只优化 id ≥ 151669 的 embedding
#   --use_tie_weights                        ← embed_tokens 和 lm_head tie
#   --dataset_config examples/dataset_config/pretrain.json

bash examples/pretrain_stg1.sh
# 日志：$OUTPUT_DIR/{stdout,stderr}.log
# 权重：$OUTPUT_DIR/step{N}/global_step{N}/
# TensorBoard：tensorboard --logdir $OUTPUT_DIR
```

训完后把 DCP checkpoint 转成 HF 格式（供 Stage 2 加载）：

```bash
bash scripts/convert_checkpoint_to_hf.sh \
    <项目路径>/raw_data/hf_models/Qwen3-1.7B_itemic \
    <项目路径>/output/model_output/stg1 \
    2000
# 产出：$OUTPUT_DIR/step2000/global_step2000/converted/  ← Stage 2 的 --model_dir
```

**单机快速验证的做法**：把 `--num_training_steps` 改成 `50`，`--save_checkpoint_per_step` 改成 `20`，能在几十分钟内跑出一个能用的 checkpoint 走通后续所有步骤。

### 9.6 Step 5 — Pretrain Stage 2（Full-parameter Co-Pretrain）

**解冻所有参数**，用同一份混合数据继续训，让全部 Transformer 权重也吸收 SID 分布。

```bash
# 编辑 examples/pretrain_stg2.sh：
#   MODEL_DIR=<Stage1 converted 路径>
#   OUTPUT_DIR=<项目路径>/output/model_output/stg2
#   --num_training_steps 5000

# 关键 flag：
#   （没有 --freeze_llm，即全参数训练）
#   --use_tie_weights
#   --dataset_config examples/dataset_config/pretrain.json  ← 与 Stage 1 相同

bash examples/pretrain_stg2.sh

# 再次转 HF 格式
bash scripts/convert_checkpoint_to_hf.sh \
    <项目路径>/raw_data/hf_models/Qwen3-1.7B_itemic \
    <项目路径>/output/model_output/stg2 \
    5000
```

Stage 2 产出的 `converted/` 就是 **OneRec-Foundation Base 模型** —— 已经具备生成 SID 的能力，可直接跑评测（作为 baseline）。

### 9.7 Step 6 — SFT（Post-training Stage 1，指令跟随）

用 chat 格式数据 + `only_assistant_loss=true`，教模型理解各种推荐场景的自然语言指令。

```bash
# 编辑 examples/posttrain_sft.sh：
#   MODEL_DIR=<Stage2 converted 路径>
#   OUTPUT_DIR=<项目路径>/output/model_output/sft

# 关键 flag：
#   --dataset_config examples/dataset_config/sft.json  ← 换到 SFT 配置
#   （sft.json 里 only_assistant_loss=true, add_think_pattern=true）

bash examples/posttrain_sft.sh

# 转 HF 格式
bash scripts/convert_checkpoint_to_hf.sh \
    <项目路径>/raw_data/hf_models/Qwen3-1.7B_itemic \
    <项目路径>/output/model_output/sft \
    5000
```

**这一步产出的模型足以做所有 RecIF-Bench 评测**（相当于官方 `OneRec-1.7B` 的 non-Pro 版本）。想跳过蒸馏和 RL 的话可以直接跳到 §9.10 评测。

### 9.8 Step 7 — On-policy Distillation（Post-training Stage 2，可选）

用未扩词表的原版 Qwen3 当 teacher，把丢失的通用能力（数学/推理）拉回来。

```bash
cd verl_distillation

# 先装 verl 环境（依赖较重，建议独立 conda env）
pip install -r requirements.txt

# 编辑 recipe/onpolicy_distill/run_qwen3_distill.sh：
export BASE_MODEL=<SFT 产出的 converted 路径>     # student
export TEACHER_MODEL=<原版 Qwen3-1.7B 路径>       # teacher（未扩词表）
export DATASET_PARQUET=<项目路径>/output/onpolicy_distillation.parquet

# 起 Ray 集群（多节点见 init_ray_cluster.sh；单机可跳过）
bash init_ray.sh

bash recipe/onpolicy_distill/run_qwen3_distill.sh /etc/mpi/hostfile
# 产出：outputs/ckpts/<project>/<exp>/global_step*/actor/
```

### 9.9 Step 8 — GRPO RL（Post-training Stage 3，可选）

强化学习提升推荐命中率；两阶段 rollout（先思考 → 再 beam SID）。

```bash
cd verl_rl
pip install -r requirements.txt

# 编辑 recipe/onerec/run_grpo.sh：
export BASE_MODEL=<Distillation 或 SFT 产出的 HF 路径>
export DATA_DIR=<项目路径>/output/rl_data          # §9.4-(4) 产出

bash init_ray.sh
bash recipe/onerec/run_grpo.sh
# 产出：./output/ckpt/global_step*/
```

### 9.10 Step 9 — 评测（RecIF-Bench）

无论你在 §9.7 / 9.8 / 9.9 哪一步停下来，最终检验都是这一步。

```bash
cd benchmarks

# 独立环境：vllm 依赖比较敏感
conda create -n benchmark python=3.10 -y
conda activate benchmark
uv pip install torch==2.5.1 transformers==4.52.0 vllm==0.7.3
pip install -r requirements.txt
pip install -e . --no-deps --no-build-isolation

# 单机多卡直接跑；多节点先 bash scripts/init_ray_cluster.sh
export BENCHMARK_BASE_DIR="."
export BENCHMARK_DATA_DIR="../raw_data/onerec_data/benchmark_data"
export DATA_VERSION="v1.0"

# 第一个参数：待评测模型路径（HF 格式）
# 第二个参数：结果输出目录后缀
# 第三个参数：是否开启 <think> 模式（推荐/label_pred 用 false，rec_reason 用 true）
bash eval_script.sh <SFT 产出 converted 绝对路径> my_sft_nothink false

# 只想跑单个任务快速验证时，把 eval_script.sh 里其他 python 命令注释掉，只留一个
# 且给 python 命令加 --sample_size 10 就能 10 个样本跑通全流程
```

**结果目录**：`benchmarks/results/v1.0/results_my_sft_nothink/<model_name>/<task>/test_generated.json`

**跑完生成后单独算指标**：

```bash
# 会扫描 results/ 目录下所有 *_generated.json，为每个任务调对应的 evaluator
python scripts/eval_dev_results.py \
    --generation_results_dir ./results/v1.0/results_my_sft_nothink \
    --output_path ./eval_results.json \
    --data_dir ../raw_data/onerec_data/benchmark_data
```

Item Understanding 和 Rec Reason 需要外部 LLM 打分，配置 `api/config/llm_config.json`（Gemini API key）后自动调用。

### 9.11 常见坑与排查

| 现象 | 原因 | 处理 |
| :--- | :--- | :--- |
| `pretrain_stg1.sh` 卡在启动 | MPI hostfile / SSH 无密码 / NCCL_IB 配错 | 单机建议直接改成 `torchrun`；参考 `set_env.sh` 和 `pretrain/README.md` 的 MPI 段 |
| 训练 loss 长期 NaN | Stage 1 时 lm_head 没跟 embed_tokens tie | 确认 `--use_tie_weights` 加了（0.6B/1.7B/4B 必需） |
| Stage 2 加载 checkpoint 时 shape mismatch | 忘了先做 `convert_checkpoint_to_hf.sh` | Stage N 加载的必须是 `.../global_stepX/converted/` |
| 推理时生成不出 SID | Base model 没扩词表 / tokenizer 版本对不上 | 用 §9.3 输出的 tokenizer；`transformers>=4.51.0` |
| 评测阶段 vLLM OOM | `gpu_memory_utilization` 太高 | 调低到 0.7；或减 `worker_batch_size`；或把 `num_return_sequences` 从 32 降到 16 |
| 推荐类任务 Recall@32 极低 | `enable_thinking=true` 用在了 `label_pred` / 推荐类任务上 | 只有 `rec_reason` 用 thinking；其他都用 false |
| Data script 报 `pid not in pid2sid` | pid2sid mapping 不覆盖当前数据 | 换回官方 `video_ad_pid2sid.parquet` / `product_pid2sid.parquet` |

### 9.12 最短路径（推理 + 评测，跳过所有训练）

如果你只想验证 pipeline 或对比自己训的模型：

```bash
# 1. 从 HuggingFace 拉官方模型
hf download OpenOneRec/OneRec-1.7B --local-dir ./raw_data/hf_models/OneRec-1.7B

# 2. 数据只需要 benchmark_data
hf download OpenOneRec/OpenOneRec-RecIF --repo-type dataset \
    --token $HF_TOKEN --local-dir ./raw_data/onerec_data

# 3. 直接评测
cd benchmarks
export BENCHMARK_DATA_DIR="../raw_data/onerec_data/benchmark_data"
bash eval_script.sh ../raw_data/hf_models/OneRec-1.7B official_baseline false
```

单机 8×A100 全任务大约几小时。

### 9.13 目录依赖关系速览

```
raw_data/hf_models/Qwen3-1.7B                     ← 起点
    ↓ expand_qwen3_vocab.sh
raw_data/hf_models/Qwen3-1.7B_itemic              ← §9.3 产出，Stage 1 起点

raw_data/onerec_data/*.parquet                    ← HF 下载
    ↓ data/onerec_data/run.sh
output/{pretrain,sft}_{task}.parquet              ← 任务级样本
    ↓ data/prepare_{pretrain,sft,rl}.sh
output/split_data_{pretrain,sft}/*.parquet        ← 分片后可直接给 dataloader
output/rl_data/{train,test}.parquet               ← RL 用

Qwen3-1.7B_itemic + split_data_pretrain
    ↓ pretrain_stg1.sh (freeze_llm)
output/model_output/stg1/step2000/.../converted/
    ↓ pretrain_stg2.sh (full param)
output/model_output/stg2/step5000/.../converted/  ← 官方 OneRec-Foundation Base
    ↓ posttrain_sft.sh
output/model_output/sft/step5000/.../converted/   ← 官方 OneRec-1.7B/8B
    ↓ (可选) verl_distillation/recipe/onpolicy_distill/run_qwen3_distill.sh
distill_ckpt/actor/
    ↓ (可选) verl_rl/recipe/onerec/run_grpo.sh
rl_ckpt/
    ↓ benchmarks/eval_script.sh
results/v1.0/results_*/ + eval_results.json       ← 最终指标
```

