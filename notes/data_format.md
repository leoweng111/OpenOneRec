# OpenOneRec 数据文件格式说明

本文档说明 `raw_data/onerec_data/` 下所有数据文件的**真实字段结构、类型、含义、样例**，让你在不打开 parquet 的情况下也能建立正确心智模型。

**数据来源**：[OpenOneRec/OpenOneRec-RecIF (HuggingFace)](https://huggingface.co/datasets/OpenOneRec/OpenOneRec-RecIF)

**编写方法说明**：
- ✅ 打勾字段 = 已本地下载并用 `pandas.read_parquet` / `json.load` 实测 schema 得到
- 📖 未打勾字段 = 从代码消费方式（`data/onerec_data/**/*.py` 里怎么用这些列）+ 官方 `README.md` 推断
- 涉及 SID 编码约定：`[c0, c1, c2]` 长度=3 的 int list → 渲染为 `<|sid_begin|><s_a_{c0}><s_b_{c1}><s_c_{c2}><|sid_end|>` 共 5 个 token

---

## 1. 文件总览

| 文件 | 大小 | 主键 | 用途 | 已下载 |
| :--- | :--- | :--- | :--- | :--- |
| `onerec_bench_release.parquet` | 1.6 GB | uid | **主 metadata** —— 每行一个用户，包含全部行为流 & 用户画像 | ❌ 未下载 |
| `video_ad_pid2sid.parquet` | 153 MB | pid | 视频 + 广告域的 pid → SID 映射 | ✅ |
| `product_pid2sid.parquet` | 20 MB | pid | 商品域的 pid → SID 映射 | ✅ |
| `pid2caption.parquet` | 5.6 GB | pid | 视频 pid → 中文文字描述 | ❌ 未下载 |
| `benchmark_data/sid2pid.json` | 165 MB | sid | SID → 视频/广告 pid 反向映射（多对多） | ✅ |
| `benchmark_data/sid2iid.json` | 200 MB | sid | SID → 商品 iid 反向映射 | ✅ |
| `benchmark_data/{task}/{task}_test.parquet` | 各任务 | uuid / row | 8 个评测任务的 test 集 | ✅ 全部 |

---

## 2. 主数据文件

### 2.1 `onerec_bench_release.parquet` ❌ (1.6 GB, 未下载)

> **一行 = 一个用户**（user-centric 组织）。所有历史 & 目标行为并列组织成等长的 bit-mask + pid 序列。参见 `notes/sample_schema.md` §3。

**基本字段**：

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `uid` | `int64` | 用户 ID（已脱敏） |
| `split` | `int` | 数据集分割标记；`0=train`，其他值 = eval |

**短视频历史 & 目标**（`L_h = len(hist_video_pid)`, `L_t = len(target_video_pid)`）：

| 字段 | 类型 | 长度 | 说明 |
| :--- | :--- | :--- | :--- |
| `hist_video_pid` | `list[int]` | $L_h$ | 历史视频 pid 时间序列（**旧→新**） |
| `hist_video_longview` | `list[0/1]` | $L_h$ | 该位置视频是否被长时观看，逐位对齐 `hist_video_pid` |
| `hist_video_like` | `list[0/1]` | $L_h$ | 是否点赞 |
| `hist_video_follow` | `list[0/1]` | $L_h$ | 是否关注了该视频作者 |
| `hist_video_forward` | `list[0/1]` | $L_h$ | 是否转发/分享 |
| `hist_video_not_interested` | `list[0/1]` | $L_h$ | 是否标记为"不感兴趣" |
| `target_video_pid` | `list[int]` | $L_t$ | 待预测的目标视频 pid 序列 |
| `target_video_longview` | `list[0/1]` | $L_t$ | Ground-truth：这些目标是否被长时观看 |
| `target_video_like` | `list[0/1]` | $L_t$ | Ground-truth：点赞 |
| `target_video_follow` | `list[0/1]` | $L_t$ | Ground-truth：关注 |
| `target_video_forward` | `list[0/1]` | $L_t$ | Ground-truth：转发 |
| `target_video_not_interested` | `list[0/1]` | $L_t$ | Ground-truth：不感兴趣 |

**广告域**：

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `hist_ad_pid` | `list[int]` | 历史点击的广告视频 pid |
| `target_ad_pid` | `list[int]` | 目标广告 pid（用户实际会点击的） |
| `hist_longview_video_list` | `list[int]` | **超长版本**的长时观看视频历史，长度远大于 `hist_video_pid`；作为广告/商品推荐任务的额外用户兴趣信号 |

> ⚠ 广告 pid 与视频 pid **共享同一个 id 空间**，都用 `video_ad_pid2sid.parquet` 映射。

**商品域**：

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `hist_goods_pid` | `list[int]` | 历史浏览的商品 id |
| `target_goods_pid` | `list[int]` | 目标商品 id |

> ⚠ 商品 pid 用**独立 id 空间**，走 `product_pid2sid.parquet` 映射。做商品推荐时通常还配合 `hist_longview_video_list` 一起当输入（跨域）。

**交互式推荐（关键词 → items）**：

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `inter_keyword_to_items` | `str (JSON)` | 序列化 dict：`{"keyword1": [pid1, pid2, ...], ...}`。每个 keyword 会展开成一条独立 SFT 样本（`data/onerec_data/sft/interactive_rec.py`） |
| `inter_user_profile_with_pid` | `str` | 用户画像文本，内嵌 `<photoid\|xxx>` 或 `<itemid\|xxx>` 引用 |
| `inter_user_profile_with_sid` | `str` | 上一列的另一版本：把 `<photoid\|xxx>` 换成实际 SID `<\|sid_begin\|>...<\|sid_end\|>`。**pretrain / SFT 阶段直接用这一列作为文本** |

**推荐理由（rec_reason）**：

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `reco_gsu_caption` | `list[str]` | GSU（粗筛召回）阶段候选视频的文字描述 |
| `reco_target_caption` | `str` | 目标视频的文字描述 |
| `reco_cot` | `str` | 思维链推理文本，解释为什么会推这个视频。SFT 阶段 `add_think_pattern=true` 时会把它塞进 `<think>...</think>` |

**样例行大小估计**：一行大致 20 个字段 × 平均几百个 int + 几段几百字文本 → 单行 20-50 KB；全表 1.6 GB / 50 KB ≈ 3 万用户量级。

---

## 3. Pid ↔ Sid 映射

### 3.1 `video_ad_pid2sid.parquet` ✅ (153 MB)

**实测 shape**: `(15 885 203, 2)` — 约 1590 万个视频/广告 item。

| 字段 | dtype | 样例 | 说明 |
| :--- | :--- | :--- | :--- |
| `pid` | `int64` | `13508074` | 视频/广告的原始 item id |
| `sid` | `object` (`np.ndarray[int64]`) | `array([1636, 1470, 676])` | 长度 3 的 SID 三元组 |

**用途**：
- `data/onerec_data/**/*.py` 里最常见的操作 `pid2sid[pid]` → `[c0, c1, c2]`；
- 拼接后 → `<|sid_begin|><s_a_1636><s_b_1470><s_c_676><|sid_end|>` 作为 prompt/label 的 5 个 token。

### 3.2 `product_pid2sid.parquet` ✅ (20 MB)

**实测 shape**: `(2 066 115, 2)` — 约 207 万商品。同 schema。

| 字段 | dtype | 样例 |
| :--- | :--- | :--- |
| `pid` | `int64` | `1685583` |
| `sid` | `object` | `array([4032, 201, 6726])` |

### 3.3 `pid2caption.parquet` ❌ (5.6 GB, 未下载)

从 README 与 `data/onerec_data/{pretrain,sft}/item_understand.py` 的用法确认：

| 字段 | 类型 | 说明 |
| :--- | :--- | :--- |
| `pid` | `int` | 视频 item id（与 `video_ad_pid2sid.parquet::pid` 同空间） |
| `dense_caption` | `str` | 该视频的中文长描述（生成 caption 或 rec_reason 训练时用） |

> ⚠ 数据集顶层的 `pid2caption.parquet` 官方 README 里字段名叫 `caption`，但 `data/onerec_data/pretrain/item_understand.py:52` 里读的列名是 `dense_caption` —— 二者应为同一字段的不同命名，具体以下载后为准。

---

## 4. SID 反向映射（供评测用）

### 4.1 `benchmark_data/sid2pid.json` ✅ (165 MB)

**实测**：dict, 顶层 key 数 = **1 519 078**。

| Key | Value |
| :--- | :--- |
| `str`：拼接后的 SID token，形如 `"349707037775"` （由三层 code 拼成的字符串） | `list[dict]`，每 dict 有 `{"pid": int, "count": int, "count_after_downsample": int}` |

**样例**：
```json
"349707037775": [
    {"pid": 7098944, "count": 8883, "count_after_downsample": 1000}
]
"某另一 sid": [
    {"pid": p1, "count": 500, "count_after_downsample": 500},
    {"pid": p2, "count": 30,  "count_after_downsample": 30},
    ...
]
```

**含义**：
- 一个 SID 通常对应**多个真实视频 pid**（因为 residual K-means 量化会把相似 item 聚到同一 SID）。
- `count` = 该 pid 在训练集里被该 SID 命中的原始次数（可看作"流行度"）。
- `count_after_downsample` = 做过分布均衡后的次数（上限一般是 1000）。
- 每个 SID 的候选 pid list 按 count 降序。

**评测用途**（见 `benchmarks/benchmark/tasks/v1_0/recommendation/evaluator.py:130-141`）：
- `evaluation_mode='pid'` 或 `'both'` 时加载此文件；
- 模型生成一个 SID → 从这里查回它对应的 pid list → 用 `sid_to_pid_strategy='most_popular'`（取 count 最高的那个）或 `'random'` 挑一个 pid；
- 然后在 pid 空间上算 Recall@K / Pass@K，让指标更接近线上真实商品命中。

### 4.2 `benchmark_data/sid2iid.json` ✅ (200 MB)

**实测**：dict，顶层 key 数 = **1 293 437**；结构与 `sid2pid.json` **完全相同**，只是 value 里字段名是 `iid` 而不是 `pid`。

**样例**：
```json
"378529993943": [
    {"iid": 541971,  "count": 86841, "count_after_downsample": 1000},
    {"iid": 1917095, "count": 344,   "count_after_downsample": 344},
    ...
]
```

**用途**：商品域的 SID → iid（商品 id）反向映射。`product` 任务的 evaluator 在 `evaluation_mode='both'` 时用它。

---

## 5. 评测集 `benchmark_data/*/` ✅ (全部已下载)

所有评测 parquet **共享两个必备字段**：
- **`metadata`** (`object` / `str`): JSON 字符串，需 `json.loads` 后使用；内含 `answer`（ground truth）+ 任务特定字段
- **`messages`** (`object` / `str`): 已经过 chat template 组织好的对话结构 `list[dict]`，供模型直接输入

各任务独有字段如下：

### 5.1 `video/video_test.parquet` ✅

**实测 shape**: `(38 781, 3)` — 3.9 万条测试样本。

| 字段 | dtype | 样例（缩略） |
| :--- | :--- | :--- |
| `hist_pid` | `object` (`ndarray[int64]`) | `array([12928241, 2812586, 9509763, ...])` |
| `metadata` | `object` (`str`) | `'{"answer": "<\|sid_begin\|><s_a_2398>...", "uid": ..., "answer_pid": [...]}'` |
| `messages` | `object` (`str`) | `'[{"role": "system", "content": [...]}, {"role": "user", "content": [...]}]'` |

`metadata` 反序列化后关键字段：
- `answer`: 拼接了多个 SID 的字符串（1 个或多个 target 视频）
- `answer_pid`: `list[int]` — ground-truth 视频 pid 列表
- `uid`, `uuid` 等辅助字段

### 5.2 `ad/ad_test.parquet` ✅

**实测 shape**: `(27 677, 4)`

| 字段 | dtype | 说明 |
| :--- | :--- | :--- |
| `hist_longview` | `object` (`ndarray[int64]`) | 长时观看的视频 pid 序列 |
| `hist_ad` | `object` (`ndarray[int64]`) | 历史点击的广告 pid 序列 |
| `metadata` | `object` (`str`) | 含 `answer` (多个 SID)、`answer_pid` (list[int])、`uid` 等 |
| `messages` | `object` (`str`) | chat 格式 |

### 5.3 `product/product_test.parquet` ✅

**实测 shape**: `(27 910, 4)`

| 字段 | dtype | 说明 |
| :--- | :--- | :--- |
| `hist_longview` | `object` (`ndarray[int64]`) | 视频观看历史（跨域信号） |
| `hist_goods` | `object` (`ndarray[int64]`) | 商品交互历史（**独立 id 空间**） |
| `metadata` | `object` (`str`) | 含 `answer`、**`answer_iid`** (list[int])、`uid` |
| `messages` | `object` (`str`) | chat 格式 |

> ⚠ 注意 metadata 里 ground-truth 键名是 **`answer_iid`** 而不是 `answer_pid`（`benchmarks/benchmark/tasks/v1_0/recommendation/evaluator.py:186` 会先 fallback 到 `answer_iid`）。

### 5.4 `label_cond/label_cond_test.parquet` ✅

**实测 shape**: `(34 891, 7)`

| 字段 | dtype | 说明 |
| :--- | :--- | :--- |
| `hist_longview` | `ndarray[int64]` | 长时观看历史 |
| `hist_like` | `ndarray[int64]` | 点赞过的视频 pid（**已 filter，不是 mask**） |
| `hist_follow` | `ndarray[int64]` | 关注过其作者的视频 pid |
| `hist_forward` | `ndarray[int64]` | 转发过的视频 pid |
| `hist_not_interested` | `ndarray[int64]` | 不感兴趣的视频 pid |
| `metadata` | `str` | 含 `answer` (SID)、`answer_pid` (list[int])、`uid`、`uuid` |
| `messages` | `str` | chat 格式（system + user + assistant） |

> ⚠ 与主表 `onerec_bench_release.parquet` 里 `hist_video_<action>` 是 **0/1 bit mask** 不同，评测集这里已经把 mask==1 的位置**筛出来变成纯 pid list**了。这是"训练侧数据"和"评测侧数据"的一个 schema 差异。

### 5.5 `label_pred/label_pred_test.parquet` ✅

**实测 shape**: `(346 190, 8)` — 34.6 万条 point-wise 分类样本。

| 字段 | dtype | 说明 |
| :--- | :--- | :--- |
| `hist_longview` | `ndarray[int64]` | 同 label_cond |
| `hist_like` | `ndarray[int64]` | 同 |
| `hist_follow` | `ndarray[int64]` | 同 |
| `hist_forward` | `ndarray[int64]` | 同 |
| `hist_not_interested` | `ndarray[int64]` | 同 |
| `candidate_pid` | `ndarray[object]` | 候选视频 pid（可以为空数组） |
| `metadata` | `str` | 含 `answer`（**"是" 或 "否"**）、`uid`、`uuid` |
| `messages` | `str` | chat 格式；user 段包含候选视频 SID + 问句 |

### 5.6 `interactive/interactive_test.parquet` ✅

**实测 shape**: `(1 000, 3)` — 1000 条交互式推荐样本。

| 字段 | dtype | 说明 |
| :--- | :--- | :--- |
| `metadata` | `str` | 含 `uid`、`keyword`（搜索关键词）、`answer` (SID)、`answer_pid` (list[int]) |
| `messages` | `str` | chat 格式，SID 版本 |
| `messages_with_pid` | `str` | chat 格式的**另一版本**：把 SID 换成 `<photoid\|xxx>` 引用；仅用于分析/对比实验 |

### 5.7 `rec_reason/rec_reason_test.parquet` ✅

**实测 shape**: `(470, 3)`

| 字段 | dtype | 说明 |
| :--- | :--- | :--- |
| `metadata` | `str` | 含 `uid`、`answer`（生成推理文本，很长的中文段落） |
| `messages` | `str` | chat 格式，user prompt 包含目标视频 SID 与历史 |
| `messages_with_pid` | `str` | 同上，SID 换成 `<photoid\|xxx>` |

### 5.8 `item_understand/item_understand_test.parquet` ✅

**实测 shape**: `(500, 3)` — 500 条 SID → caption 生成样本。

| 字段 | dtype | 说明 |
| :--- | :--- | :--- |
| `metadata` | `str` | 含 `pid`、`sid`（渲染后的 SID token 字符串）、`answer`（gold caption） |
| `messages` | `str` | chat 格式，user prompt 是 SID + "介绍下这个视频"之类 |
| `index` | `int64` | 序号 |

---

## 6. 上下游依赖速查

```
                                    ┌── data/onerec_data/pretrain/*.py ── 训练时消费
onerec_bench_release.parquet        │      调用 pid2sid[pid] 转 SID → prompt 拼接
   (user-centric 主表) ─────────────┼── data/onerec_data/sft/*.py
                                    │
video_ad_pid2sid / product_pid2sid.parquet
   (pid → [c_a, c_b, c_c])
                                    ┌── benchmarks/**/recommendation/evaluator.py
sid2pid.json / sid2iid.json         │      code_to_pid[sid] → pid list（most_popular 挑一个）
   (SID → pid list w/ 流行度) ──────┴── 评测阶段计算 pid 空间的 Recall

pid2caption.parquet ────────────────── data/onerec_data/{pretrain,sft}/item_understand.py
   (pid → dense_caption)               生成 SID ↔ caption 对齐样本

benchmark_data/{task}/{task}_test.parquet
   ├─ messages    ← 直接输入模型
   └─ metadata.answer / answer_pid / answer_iid ← 评测比对的 ground truth
```

---

## 7. 常见坑

1. **`sid` 列的 dtype 是 `object`**：pandas 把 `array([c0, c1, c2])` 包成 object，不能直接向量化取值。取值用 `df['sid'].iloc[i][0]` 或 `np.stack(df['sid'].values)`。
2. **`hist_video_<action>` 在训练主表里是 0/1 mask，在评测集 label_cond/label_pred 里是筛过的 pid list**——两种 schema 不同，处理逻辑不能直接搬。
3. **广告 pid 与视频 pid 共享 id 空间；商品 pid 独立**：错用映射表会导致 KeyError。
4. **`metadata` 是 JSON 字符串**，不是 dict，需 `json.loads` 后再用。
5. **`answer_pid` vs `answer_iid`**：product 任务用 `answer_iid`，其他推荐类任务用 `answer_pid`。evaluator 里的 fallback 逻辑（`answer_pid or answer_iid`）就是为了兼容这一点。
6. **`inter_keyword_to_items` 是 JSON 字符串**（不是 dict），也需 `json.loads`。
7. **视频 caption 列名不确定**：README 说 `caption`，但 `data/onerec_data/pretrain/item_understand.py` 读 `dense_caption`——以文件到手后 `df.columns` 为准。
