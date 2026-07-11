"""Qwen3 的训练数据集与 sample packing。

本模块实现了 OneRec 预训练的数据加载核心逻辑，支持两种数据格式和高效的 sample packing。

类的继承关系：
    IterableDataset (PyTorch 基类)
        ├── Qwen3ChatCompletionDataset    — 核心处理逻辑（tokenize、loss_mask、sample packing）
        └── Qwen3ChatCompletionParquetDataset — 实际使用的入口类，管理 parquet 文件列表和 epoch

数据流概览：
    parquet 文件（含 `messages` 或 `segments` 字段，格式见 `data/README.md`）
      ↓ Qwen3NaiveParquetDataset：分片到各 rank/worker，读一行、做 local shuffle
      ↓ Qwen3ChatCompletionParquetDataset._process：
            - chat 模式（messages）：apply_chat_template → tokenize；loss_mask 只在 assistant 段=1
            - segments 模式：拼接 text → tokenize；loss_mask 全 1（除末尾 EOS）
      ↓ __iter__：把多个样本拼接（sample packing）填满 max_length
      ↓ collate → 每个 batch 是一条长度 ≈ max_length 的"打包序列"

两种数据格式：
    1. segments 格式（预训练阶段的主要格式）：
       - 用于推荐数据（视频序列、物品理解、用户画像）和通用文本数据
       - 直接拼接所有 segment 的 text，loss_mask 全 1（除 EOS）
       - 示例：视频推荐序列 "<|sid_begin|><s_a_340>...<|sid_end|>"

    2. messages 格式（SFT 阶段的主要格式）：
       - 用于对话/指令数据，包含 user/assistant 多轮对话
       - 通过 apply_chat_template 转换，loss_mask 只在 assistant 回复段 = 1
       - 示例：[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

Sample Packing（样本拼接）：
    核心加速手段：把多个短样本沿 sequence 维度拼接成一条长序列（~max_length），
    配合 FlashAttention 的 cu_seqlens 变长 attention，避免 padding 浪费。

    例如：3 条长度分别为 500、800、700 的样本 → 拼接成一条 2000 长度的序列
    cu_seqlens = [0, 500, 1300, 2000] 记录每个样本的边界

产出 batch（供 recipes/train_qwen3.py 使用）：
    input_ids       (1, T)  T = ceil(max_length/8)*8 + 64，token ID 序列
    position_ids    (1, T)  位置编码，每条子样本内部从 0 开始（除非 full_attention=True）
    loss_mask       (1, T)  1=计 loss，0=忽略（chat 模式下只在 assistant 段 = 1）
    itemic_id_mask  (1, T)  1=该 token 是 Itemic Token（id 落在 itemic_id_range 里），用于监控
    cu_seqlens      (S+1,)  packing 边界，供 FlashAttention 使用
    sample_idx      (1, T)  每个 token 属于打包块中的第几条样本
    epoch_idx       (1,)    浮点，平均 epoch 索引
"""

import logging

import os
import json
import time
import traceback
import random
import re

import multiprocessing
import numpy as np

import webdataset as wds
from easydict import EasyDict as edict 
from typing import Union, Iterable, Optional, List, Dict, Tuple, Any


import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import IterableDataset

from transformers import AutoTokenizer, AutoConfig

from onerec_llm.data.local_shuffle_buffer import LocalShuffleBuffer

from onerec_llm.utils.common import print_rank_0
from onerec_llm.utils.worker_utils import pytorch_worker_info
from onerec_llm.utils.data_utils import shell_hdfs_ls, load_parquet_file

from onerec_llm.models.qwen3.configuration_qwen3 import Qwen3Config


logger = logging.getLogger(__name__)

def set_kwargs(self, kwargs, **_kwargs):
    """将 kwargs 字典中的键值对设置为对象的属性。

    这是一个工具函数，用于把配置参数（从 JSON 配置文件读取）直接映射为对象属性，
    这样后续代码可以直接通过 self.xxx 访问配置项，而不需要 self.kwargs["xxx"]。

    Args:
        self: 目标对象
        kwargs: 配置参数字典（通常来自 dataset_config JSON 文件）
        **_kwargs: 额外的覆盖参数（优先级高于 kwargs）
    """
    kwargs.update(_kwargs)
    self.kwargs = edict(kwargs)  # EasyDict 支持属性访问，如 kwargs.max_length
    for k, v in kwargs.items():
        setattr(self, k, v)      # 把每个配置项设置为对象属性

class Qwen3ChatCompletionDataset(IterableDataset):
    """Qwen3 训练数据集的核心处理逻辑。

    本类实现了两种数据格式的处理：
    1. segments 格式：预训练阶段使用，直接拼接文本（视频序列、物品理解、用户画像、通用文本）
    2. messages 格式：SFT 阶段使用，处理多轮对话数据

    关键功能：
    - tokenize：将文本转换为 token ID 序列
    - loss_mask：标记哪些 token 需要计算 loss（chat 模式只在 assistant 段）
    - itemic_id_mask：标记 SID token（用于监控，不参与训练）
    - sample packing：将多个短样本拼接成一条长序列，提高训练效率

    继承自 PyTorch 的 IterableDataset，支持流式数据加载和多 worker 并行。
    """

    def __init__(self, **kwargs):
        """初始化数据集。

        Args:
            **kwargs: 配置参数，主要包括：
                - base_model_dir: Qwen3 模型路径（用于加载 tokenizer 和 config）
                - sources: 数据文件列表（JSON 文件路径或 parquet 文件列表）
                - max_length: 每个 batch 的最大序列长度（用于 sample packing）
                - itemic_id_range: SID token 的 ID 范围，如 [151669, 176246]
                - add_think_pattern: 是否添加  标签（用于推理任务）
                - cut_to_pad: 是否在 packing 时截断样本以填满 max_length
        """
        # 将配置参数设置为对象属性
        set_kwargs(self, kwargs)
        print_rank_0(f"ChatCompletionDataset init with kwargs={kwargs}")

        # 加载模型配置（获取 pad_token_id 等信息）
        try:
            model_config = AutoConfig.from_pretrained(self.kwargs.base_model_dir)
        except Exception:
            model_config = Qwen3Config.from_pretrained(self.kwargs.base_model_dir)

        self.pad_token_id = model_config.pad_token_id

        # 构建数据源（加载文件列表，初始化 WebDataset 或 NaiveParquetDataset）
        self.dataset, self.total_samples = self._build_source_dataset(self.sources)

        # 数据源监控：统计每个 source 的样本数和错误数
        self.source_sample_cnt = {}
        self.source_error_cnt = {}

        # 加载 tokenizer（用于将文本转换为 token ID）
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_dir, trust_remote_code=True)

        # max_sample_length: 单个样本的最大长度（不能超过 max_length）
        self.max_sample_length = min(self.max_length, self.kwargs.get("max_sample_length", 9999999))
        assert self.max_length > 0

        # ========== Chat 模板相关的特殊 token ==========
        # Qwen3 的 chat 模板使用这些 token 标记对话的边界：
        #   <|im_start|>user\n用户输入<|im_end|>\n
        #   <|im_start|>assistant\n模型回复<|im_end|>\n
        self.im_start_token = "<|im_start|>"
        self.im_end_token = "<|im_end|>"
        self.im_start_token_id = self.tokenizer.encode(self.im_start_token)[0]
        self.im_end_token_id = self.tokenizer.encode(self.im_end_token)[0]

        # 预计算 chat 模板的 pattern（用于定位 assistant 段，生成 loss_mask）
        # assistant_start_pattern: tokenize("<|im_start|>assistant\n") 的 token ID 序列
        # im_end_pattern: tokenize("<|im_end|>\n") 的 token ID 序列
        self.assistant_start_pattern = self.tokenizer.encode(
            f"{self.im_start_token}assistant\n",
            add_special_tokens=False,
        )
        self.im_end_pattern = self.tokenizer.encode(
            f"{self.im_end_token}\n",
            add_special_tokens=False,
        )
        # 兜底：如果 <|im_end|>\n 无法 tokenize，退化为只用 <|im_end|>
        if not self.im_end_pattern:
            self.im_end_pattern = self.tokenizer.encode(
                self.im_end_token,
                add_special_tokens=False,
            )

        # 是否添加  标签（用于推理任务的 thinking pattern）
        self.add_think_pattern = self.kwargs.get("add_think_pattern", False)
        if self.add_think_pattern:
            logger.info(f"Thinking pattern enabled: add_think_pattern={self.add_think_pattern}")

        # SID token 的 ID 范围（用于生成 itemic_id_mask，监控 SID token 的 loss）
        # 例如：itemic_id_range = [151669, 176246] 表示 SID token 的 ID 在这个范围内
        self.itemic_id_range = self.kwargs.get("itemic_id_range", None)
        if self.itemic_id_range is not None:
            assert len(self.itemic_id_range) == 2, "itemic_id_range must be a list of two elements"
            assert self.itemic_id_range[0] < self.itemic_id_range[1], "itemic_id_range[0] must be less than itemic_id_range[1]"

    def _build_source_dataset(self, sources):
        """Build WebDataset from source configuration files.
        
        Args:
            sources: String (comma-separated) or list of JSON config file paths
            
        Returns:
            tuple: (dataset, total_samples)
        """
        if isinstance(sources, str):
            sources = sources.split(",")
        
        # Read URLs from configuration files
        urls = []
        total_samples = 0
        for source in sources:
            with open(source, encoding="utf-8") as f:
                index = json.loads(f.read())["shardlist"]
                source_dir = os.path.dirname(source)
                for item in index:
                    urls.append(os.path.join(source_dir, item["url"]))
                    total_samples += item["nsamples"]

        # Sort, shuffle and broadcast URLs across all ranks
        urls.sort()
        random.shuffle(urls)
        url_list = [urls]
        dist.broadcast_object_list(url_list, src=0)
        urls = url_list[0]
        logger.info(f"[RANK{dist.get_rank()}] Loaded {len(urls)} URLs, total_samples={total_samples}")

        # Build WebDataset
        dataset = wds.WebDataset(
            urls,
            handler=wds.warn_and_continue,
            resampled=True,
            shardshuffle=True,
            cache_dir="/tmp/_wids_cache",
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker
        )
        
        dataset = dataset.shuffle(
            self.shuffle_size, 
            initial=self.shuffle_initial_size
        ).decode("pil", handler=wds.warn_and_continue)

        return dataset, total_samples
    
    def _convert_messages(self, messages):
        msg_list = []
        for msg in messages:
            content = msg['content']
            if isinstance(content, str):
                msg_list.append({
                    'role': msg['role'],
                    'content': content
                })
            elif isinstance(content, dict) and 'type' in content and content['type'] == 'text':
                msg_list.append({
                    'role': msg['role'],
                    'content': content['text']
                })
            elif isinstance(content, list) and len(content) > 0:
                content_text = ""
                for c in content:
                    if isinstance(c, dict) and 'type' in c and c['type'] == 'text':
                        content_text += c['text']
                    elif isinstance(c, str):
                        content_text += c
                    else:
                        continue
                msg_list.append({
                    'role': msg['role'],
                    'content': content_text
                })
            else:
                raise ValueError(f"Unsupported content type: {type(content)}")
        
        if self.add_think_pattern:
            # Process thinking pattern: add /think or /no_think suffix to user messages
            # based on whether assistant message contains reasoning content
            for i in range(len(msg_list)):
                if msg_list[i]['role'] == 'assistant':
                    assistant_content = msg_list[i]['content']
                    
                    # Find corresponding user message (typically the previous one)
                    user_idx = i - 1
                    if user_idx < 0 or msg_list[user_idx]['role'] != 'user':
                        continue
                    
                    # Check if assistant content contains <think> tags
                    pattern = r'<think>(.*?)</think>'
                    match = re.search(pattern, assistant_content, re.DOTALL)
                    
                    if match is None:
                        # No reasoning tags found: add empty tags and mark as /no_think
                        msg_list[user_idx]['content'] += "/no_think"
                        msg_list[i]['content'] = "<think>\n</think>\n" + assistant_content
                    else:
                        # Reasoning tags found: check if they contain actual content
                        reasoning_content = match.group(1)
                        if reasoning_content.strip():
                            # Has reasoning content: mark as /think
                            msg_list[user_idx]['content'] += "/think"
                        else:
                            # Empty reasoning tags: mark as /no_think
                            msg_list[user_idx]['content'] += "/no_think"
            
        return msg_list

    def _get_assistant_mask(self, batch_input_ids: torch.Tensor,
                       start_pattern: Optional[List[int]],
                       end_pattern: Optional[List[int]]):
        """扫描 token 序列，标出 assistant 段 → loss_mask。

        Chat 模板产出的 token 大致形如：
            <|im_start|>user\n ... <|im_end|>\n
            <|im_start|>assistant\n <ASSISTANT_TEXT> <|im_end|>\n
        只有 <ASSISTANT_TEXT> 那段被计入 loss。函数用两个 pattern：
            start_pattern = tokenize("<|im_start|>assistant\n")
            end_pattern   = tokenize("<|im_end|>\n")
        以子序列匹配的方式找出 start_pattern 之后到 end_pattern 之前的位置置 1。
        若某个 assistant 段没有匹配到结束 pattern（截断了），则从起点一直标到末尾。

        Args:
            batch_input_ids: (B, L) int，一般 B=1（本 dataset 每次处理一条样本）
        Returns:
            mask: (B, L) int64，1=assistant token，0=其他
        """
        if not start_pattern:
            start_pattern = self.assistant_start_pattern
        if not end_pattern:
            end_pattern = self.im_end_pattern

        masks = []
        for input_ids in batch_input_ids:
            ids = input_ids.tolist()
            mask = [0] * len(ids)
            start_len = len(start_pattern)
            end_len = len(end_pattern)
            i = 0

            while i <= len(ids) - start_len:
                if ids[i:i + start_len] != start_pattern:
                    i += 1
                    continue

                content_start = i + start_len
                j = content_start
                found_end = False
                while j <= len(ids) - end_len:
                    if ids[j:j + end_len] == end_pattern:
                        found_end = True
                        break
                    j += 1

                if not found_end:
                    for k in range(content_start, len(ids)):
                        mask[k] = 1
                    break

                for k in range(content_start, j):
                    mask[k] = 1

                i = j + end_len

            masks.append(mask)
        return torch.tensor(masks, dtype=torch.long)
    
    def _get_rope_index_qwen3(
                                self,
                                input_ids: torch.LongTensor,
                            ) -> torch.Tensor:
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(input_ids.shape[0], -1)
        return position_ids
    
    def _process_completion(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process segments format data into model inputs.
        
        Args:
            sample: Sample containing segments with pre-tokenized tokens
            
        Returns:
            Dictionary containing input_ids, attention_mask, labels, etc.

        segments 是一个列表，通常只有1个元素

        # 视频推荐数据（video_rec.py:55-59）
        segments = [{"type": "text", "text": "<|sid_begin|><s_a_340>...<|sid_end|><|sid_begin|><s_a_120>..."}]
        # ↑ 只有1个segment，text里包含了所有历史SID和目标SID的拼接

        # 物品理解数据（item_understand.py:40-45）
        segments = [{"type": "text", "text": "视频<|sid_begin|><s_a_340><s_b_6566><s_c_5603><|sid_end|> 展示了以下内容：搞笑猫咪视频..."}]
        # ↑ 只有1个segment

        # 用户画像数据（user_profile.py:25-30）
        segments = [{"type": "text", "text": "用户U1的兴趣画像：喜欢观看<|sid_begin|><s_a_42>...类型的视频..."}]
        # ↑ 只有1个segment

        为什么只有1个 segment？

        因为 OneRec 的预训练数据设计就是"一条样本 = 一个完整的文本序列"。

        segments 是一个列表，设计初衷是为了支持多段文本拼接（例如 [{"type": "text", "text": "段落1"}, {"type": "text", "text": "段落2"}]），但在 OneRec 的实际使用中：

        - 每个 segment 就是一个文本段落
        - segments 列表通常只有1个元素（因为一条训练样本就是一条完整的序列）
        - 循环 for segment in segments 只是为了兼容多段的情况

        """
        segments = sample["json"]["segments"]

        segments_text = ""

        for segment in segments:
            if segment["type"] == "text":
                segments_text += segment["text"]
            else:
                logger.error(f"segment type is not text, skip: {segment}")
                continue
        
        # Note: do not use self.tokenizer.eos_token as it's always set to <im_end>
        # References: 
        # 1. https://huggingface.co/Qwen/Qwen3-8B/blob/main/tokenizer_config.json#L232
        # 2. https://qwen.readthedocs.io/zh-cn/latest/getting_started/concepts.html#control-tokens
        # 添加EOS token
        # 在每条 segments 样本末尾手动追加一个“结束标记 token”（这里项目选择用 pad_token 充当末尾特殊符号）。
        segments_text += self.tokenizer.pad_token
        
        # Tokenize：文本转换为 token ID 序列
        # 每个SID标记都会被tokenizer转换为一个独立的token，因为这些标记在词表扩展阶段就被注册为独立token了
        #
        inputs = self.tokenizer(
            segments_text,
            return_tensors="pt",
            padding=False,
            truncation=False
        )

        input_ids = inputs["input_ids"]
        
        # Check length
        if input_ids.shape[-1] > self.max_length:
            raise ValueError(f"Sample too long: {input_ids.shape[-1]} > {self.max_length}")
        
        # Mask EOS token
        inputs["loss_mask"] = torch.ones_like(input_ids)
        inputs["loss_mask"][..., -1] = 0

        # itemic id index mask
        itemic_id_mask = torch.zeros_like(input_ids)
        if self.itemic_id_range is not None:
            itemic_id_mask[(input_ids >= self.itemic_id_range[0]) & (input_ids <= self.itemic_id_range[1])] = 1
        inputs["itemic_id_mask"] = itemic_id_mask
        
        # Generate position IDs
        inputs["position_ids"] = self._get_rope_index_qwen3(input_ids)
        
        return inputs

    def _process_chat(self, sample: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process messages format data into model inputs.
        
        Args:
            sample: Sample containing messages in the new format
            
        Returns:
            Dictionary containing input_ids, attention_mask, labels, etc.
        """

        # messages格式：
        # messages = [
        #         {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        #         {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        #         {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        #     ]
        msg_key = "message" if "message" in sample["json"] else "messages"
        messages = sample["json"][msg_key]

        msg_converted = self._convert_messages(messages)
        
        # Convert messages to text using chat template
        text = self.tokenizer.apply_chat_template(
            msg_converted, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        # Add EOS token
        text += self.tokenizer.pad_token

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=False
        )

        input_ids = inputs["input_ids"]        
        # Check length
        if input_ids.shape[-1] > self.max_length:
            raise ValueError(f"Sample too long: {input_ids.shape[-1]} > {self.max_length}")
        
        inputs["loss_mask"] = self._get_assistant_mask(
            input_ids,
            start_pattern=self.assistant_start_pattern,
            end_pattern=self.im_end_pattern,
        )
        
        # Mask EOS token
        inputs["loss_mask"][..., -1] = 0

        # itemic id index mask
        itemic_id_mask = torch.zeros_like(input_ids)
        if self.itemic_id_range is not None:
            itemic_id_mask[(input_ids >= self.itemic_id_range[0]) & (input_ids <= self.itemic_id_range[1])] = 1
        inputs["itemic_id_mask"] = itemic_id_mask
        
        # Generate position IDs
        inputs["position_ids"] = self._get_rope_index_qwen3(input_ids)
        
        return inputs

    def _process(self, sample, source_name=None):
        if "segments" in sample["json"] and sample["json"]["segments"] is not None:
            inputs = self._process_completion(sample)
        else:  # SFT 任务脚本里并没有产出 segments，走这个分支
            inputs = self._process_chat(sample)

        inputs['epoch_idx'] = sample['epoch_idx']
        if not inputs:
            raise ValueError("Empty inputs, skip")
        
        # Check if sample exceeds max_sample_length (always <= max_length)
        if inputs["input_ids"].shape[-1] > self.max_sample_length:
            logger.warning(f"Sample exceeds max_sample_length={self.max_sample_length}, length={inputs['input_ids'].shape[-1]}")
            raise ValueError(
                f"Unable to generate sample within max_sample_length={self.max_sample_length}"
            )
        
        return inputs

    def _cut_sample(self, inputs, packable_length):
        inputs["input_ids"] = inputs["input_ids"][:, :packable_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :packable_length]
        inputs["loss_mask"] = inputs["loss_mask"][:, :packable_length]
        inputs["position_ids"] = inputs["position_ids"][..., :packable_length]
        inputs["itemic_id_mask"] = inputs["itemic_id_mask"][:, :packable_length]
        return inputs

    def _append_sample_packing(self,
                                inputs: Dict[str, torch.Tensor],
                                packed_input_ids: List[torch.Tensor],
                                packed_position_ids: List[torch.Tensor],
                                packed_loss_mask: List[torch.Tensor],
                                packed_itemic_id_mask: List[torch.Tensor],
                                packed_sample_idx: List[torch.Tensor],
                                cu_seqlens: List[int],
                                sample_idx: Optional[int] = None,
                                ):
        packable_length = self.max_length - cu_seqlens[-1]
        if packable_length == 0: return

        if self.cut_to_pad and inputs['input_ids'].shape[1] > packable_length:
            inputs = self._cut_sample(inputs, packable_length)

        packed_input_ids.append(inputs["input_ids"].flatten())
        packed_loss_mask.append(inputs["loss_mask"].flatten())
        packed_position_ids.append(inputs["position_ids"])
        packed_itemic_id_mask.append(inputs["itemic_id_mask"].flatten())

        if sample_idx is None:
            sample_idx = len(cu_seqlens) - 1

        packed_sample_idx.append(
            torch.full_like(packed_input_ids[-1], sample_idx))

        cu_seqlens.append(cu_seqlens[-1] + len(inputs["input_ids"][0]))
        return len(inputs["input_ids"][0])

    def _packing(self, buffer: List[Dict[str, torch.Tensor]]):
        """把 buffer 里多个短样本 concat 成一条长度 ≈ max_length 的"packed 序列"。

        Sample packing 是把 N 条短样本沿 seq 维拼起来放进一个 batch，配合 FlashAttention 的
        cu_seqlens 变长 attention，可以避免 padding 浪费，是 pretrain 阶段的核心加速手段。

        输入 buffer：List[{"input_ids": (1, L_i), "loss_mask": (1, L_i), ...}]，共 N 条。
        产出（形状说明中 T = ceil(max_length/8)*8 + 64）：
            input_ids       (1, T)
            position_ids    (1, T)   每条样本内 0..L_i-1；padding 段用 0
            loss_mask       (1, T)   末尾 padding 段=0
            itemic_id_mask  (1, T)
            cu_seqlens      (N+2,)   [0, L_0, L_0+L_1, ..., sum_L, T]，最后一段是 padding
            sample_idx      (1, T)   token 属于哪条源样本；padding 段=-1
            epoch_idx       (1,)     float32
        """
        packed_position_ids: List[torch.Tensor] = []
        packed_loss_mask: List[torch.Tensor] = []
        packed_itemic_id_mask: List[torch.Tensor] = []
        packed_sample_idx: List[torch.Tensor] = []
        cu_seqlens: List[int] = [0]
        epochs = []
        valid_seq_len = 0
        for _, inputs in enumerate(buffer):
            epochs.append(inputs.get("epoch_idx", None))
            valid_seq_len += self._append_sample_packing(inputs,
                                            packed_input_ids,
                                            packed_position_ids,
                                            packed_loss_mask,
                                            packed_itemic_id_mask,
                                            packed_sample_idx,
                                            cu_seqlens,
                                            )

        packed_input_ids = torch.cat(packed_input_ids, dim=0).unsqueeze(0)
        packed_loss_mask = torch.cat(packed_loss_mask, dim=0).unsqueeze(0)
        packed_itemic_id_mask = torch.cat(packed_itemic_id_mask, dim=0).unsqueeze(0)
        packed_position_ids = torch.cat(packed_position_ids, dim=-1)
        packed_sample_idx = torch.cat(packed_sample_idx, dim=0).unsqueeze(0)

        max_length = max(self.max_length, packed_input_ids.numel())
        padding_len = (max_length + 7) // 8 * 8 + 64 - packed_input_ids.numel()
        assert padding_len > 0, f"padding_len should be greater than 0, got {padding_len}"
        packed_input_ids = F.pad(
            packed_input_ids, (0, padding_len),
            value=self.tokenizer.pad_token_id)
        packed_sample_idx = F.pad(packed_sample_idx, (0, padding_len), value=-1)
        packed_position_ids = F.pad(packed_position_ids, (0, padding_len), value=0)
        packed_loss_mask = F.pad(packed_loss_mask, (0, padding_len), value=0)
        packed_itemic_id_mask = F.pad(packed_itemic_id_mask, (0, padding_len), value=False)
        cu_seqlens.append(cu_seqlens[-1] + padding_len)

        if self.kwargs.get("full_attention", False):
            packed_position_ids = self._get_rope_index_qwen3(packed_input_ids)
            cu_seqlens = [0, cu_seqlens[-1]]

        epochs = [x for x in epochs if x is not None]
        inputs = {
            "input_ids": packed_input_ids,
            "position_ids": packed_position_ids,
            "loss_mask": packed_loss_mask,
            "itemic_id_mask": packed_itemic_id_mask,
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "sample_idx": packed_sample_idx.to(torch.int32),
            "epoch_idx": torch.tensor([sum(epochs) / len(epochs)], dtype=torch.float32),
        }
        return inputs

    def __iter__(self):
        """
        从底层 parquet 迭代器拿原始 sample（Qwen3NaiveParquetDataset）
        _process(...)：分 segments 或 messages 路径做 tokenize/loss_mask
        做 sample packing（_packing）
        yield packed_inputs（就是训练脚本拿到的 batch）
        """
        if self.dataset is None:
            self.dataset, self.total_samples = self._build_source_dataset(self.sources)

        buffer = []
        source_list = []
        cur_length = 0
        ds_iter = iter(self.dataset)
        while True:
            try:
                # sample 是一个字典，由 Qwen3NaiveParquetDataset._parser() 方法构造
                # 每个 sample 对应 parquet 文件中的一行数据（即一个训练样本）。
                sample = next(ds_iter)
                sample_key = sample["__key__"] if "__key__" in sample else ""
                sample_url = sample["__url__"] if "__url__" in sample else ""

                try:
                    source_name = sample["json"]["source"]
                except Exception:
                    source_name = "None"

                self.source_sample_cnt.setdefault(source_name, 0)
                self.source_sample_cnt[source_name] += 1
            
                inputs = self._process(sample, source_name)
            except Exception:
                self.source_error_cnt.setdefault(source_name, 0)
                self.source_error_cnt[source_name] += 1
                error_ratio = self.source_error_cnt[source_name] * 1.0 / \
                    self.source_sample_cnt[source_name]
                
                rank, world_size, worker, num_workers = pytorch_worker_info()
                logger.error(
                    f"Qwen3ChatCompletionDataset process sample error. worker=r{rank}_w{worker}"
                    f"{source_name=}, {error_ratio=}, {sample_key=}, {sample_url=}, sample=\n{str(sample)[:50]}"
                    f"errmsg={traceback.format_exc()}")
                continue

            sample_length = inputs["input_ids"].shape[-1]
            # # 如果当前buffer + 新样本超过max_length，执行packing
            if cur_length + sample_length >= self.max_length:
                if self.cut_to_pad:
                    buffer.append(inputs)
                    source_list.append(source_name)
                    # packed的样本之间间隔了EOS token
                    packed_inputs = self._packing(buffer)

                    packed_inputs["data_source"] = source_list
                    buffer = []
                    source_list = []
                    cur_length = 0
                    if packed_inputs["loss_mask"].sum().item() == 0:
                        logger.warning(f"Packed sample has no valid loss tokens, cur_length={cur_length}, skipping. "
                                    f"This usually happens when a single sample has no valid tokens after processing.")
                        continue
                else:
                    packed_inputs = self._packing(buffer)
                    packed_inputs["data_source"] = source_list
                    buffer = [inputs]
                    source_list = [source_name]
                    cur_length = sample_length

                if packed_inputs["loss_mask"].sum() == 0:
                    logger.warning("Skipping sample with no valid loss tokens.")
                    continue

                # 每次调用dataloader，生成一条packing后的长样本（B = 1），形状为(1, T)
                yield packed_inputs

            else:
                buffer.append(inputs)
                source_list.append(source_name)
                cur_length += sample_length  # packing序列中每条样本的实际长度

class Qwen3NaiveParquetDataset(IterableDataset):
    """Naive parquet dataset for Qwen3 that handles file reading and parsing."""
    
    def __init__(self, data_files, num_workers, **kwargs):
        set_kwargs(self, kwargs, data_files=data_files, num_workers=num_workers)
        self.local_shuffle_buffer = LocalShuffleBuffer(buffer_size=self.kwargs.get("local_shuffle_buffer_size", 81920), 
                                                        random_fetch=self.kwargs.get("local_shuffle_random_fetch", 0.00001))
    
        manager = multiprocessing.Manager()
        def make_dict(): return manager.dict()

        self.finish_dict_all = make_dict()
        for i in range(self.num_workers):
            self.finish_dict_all[i] = make_dict()
    
    def _parser(self, raw_row_data, file_url):
        """Parse a single row from parquet file."""
        try:
            messages = None
            segments = None
            
            if "messages" in raw_row_data:
                messages = raw_row_data["messages"]
                if isinstance(messages, str):
                    messages = json.loads(messages)

            if "segments" in raw_row_data:
                segments = raw_row_data["segments"]
                if isinstance(segments, str):
                    segments = json.loads(segments)

            data_source = raw_row_data["source"]
            key = raw_row_data["uuid"]
            
            samples = {
                "__key__": key,
                "__url__": file_url,
            }

            sample_data = {
                "source": data_source,
            }

            if messages is not None and isinstance(messages, list) and len(messages) > 0:
                sample_data["messages"] = messages
            elif segments is not None and isinstance(segments, list) and len(segments) > 0:
                sample_data["segments"] = segments
            elif messages is not None and isinstance(messages, np.ndarray):
                sample_data["messages"] = messages.tolist()
            else:
                raise NotImplementedError(f"Unsupported sample, message type is {type(messages)}, message={messages}, segments type is {type(segments)}, segments={segments}")

            samples["json"] = sample_data
            
            return samples
        except Exception as e:
            logger.error(f"Qwen3NaiveParquetDataset parse sample error: {str(e)}")
            return None

    def __iter__local_shuffle(self):
        rank, world_size, worker, num_workers = pytorch_worker_info()
        finish_dict = self.finish_dict_all[worker]
        assert num_workers == self.num_workers

        total_num_workers = num_workers * world_size
        local_worker_idx = rank * num_workers + worker
        fn_list = [fn for idx, fn in enumerate(self.data_files) if idx % total_num_workers == local_worker_idx]
        logger.warning(
            f"ParquetDataset Info: {rank=}, {world_size=}, {worker=}, {num_workers=}, {len(fn_list)=}"
        )   
        
        def get_sample():
            for fn_index, (fn, epoch_idx) in enumerate(fn_list):
                try:
                    df = load_parquet_file(fn).read_row_group(0).to_pandas()
                except Exception as e:
                    logger.warning(
                        f"ParquetDataset Info: {rank=}, {world_size=}, {worker=}, {num_workers=}, {fn} failed" + \
                        f"traceback=\n{traceback.format_exc()}"
                    )
                    continue
                df['epoch_idx'] = epoch_idx
                df['fn_idx'] = fn_index
                df['__fn__'] = fn
                df['sample_index'] = range(len(df))
                for i, (_, row) in enumerate(df.iterrows()):
                    sample_bit = 1 << row['sample_index']
                    if sample_bit & finish_dict.get((row['__fn__'], row['epoch_idx']), 0) != 0:
                        logger.debug(f"[Rank{rank}-Worker{worker}] Skipping already processed sample: "
                                    f"{row['__fn__']}-epoch{row['epoch_idx']}-sample{row['sample_index']}")
                        continue
                    if self.local_shuffle_buffer.add(row, fn, epoch_idx): continue
                    row = self.local_shuffle_buffer.get()
                    yield row

            while len(self.local_shuffle_buffer) > 0:
                row = self.local_shuffle_buffer.get()
                yield row

        for row in get_sample():
            sample_bit = 1 << row['sample_index']

            key = (row['__fn__'], row['epoch_idx'])
            if key not in finish_dict:
                finish_dict[key] = 0
            finish_dict[key] |= sample_bit

            sample = self._parser(row, row['__fn__'])
            sample['epoch_idx'] = torch.tensor(row['epoch_idx'])
            yield sample

    def __iter__(self,):
        for sample in self.__iter__local_shuffle():
            if sample is None: continue
            yield sample
    
    def state_dict(self):
        """Get state dict for checkpointing."""
        rank, world_size, worker, num_workers = pytorch_worker_info()

        state_dict = {
            "finish_dict": dict(self.finish_dict_all[worker]),
        }
        return state_dict
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        rank, world_size, worker, num_workers = pytorch_worker_info()
        
        finish_dict = state_dict["finish_dict"]
        
        # Convert to regular dict to support old checkpoint format
        tmp_finish_dict = dict(finish_dict)
        
        # Clear current state and update
        self.finish_dict_all[worker].clear()
        self.finish_dict_all[worker].update(tmp_finish_dict)
        logger.info(f"[rank{rank}-worker{worker}] Loaded checkpoint successfully. finish_dict_size={len(tmp_finish_dict)}")

class Qwen3ChatCompletionParquetDataset(Qwen3ChatCompletionDataset):
    def __init__(self, sources, num_workers, shuffle_seed=1024, num_epochs=1, **kwargs):
        self.rng = random.Random(shuffle_seed)
        self.num_workers = num_workers
        self.num_epochs = num_epochs
        self.cut_to_pad = kwargs.get("cut_to_pad", True)
        self.kwargs = kwargs
        self.num_readers = kwargs.get("num_readers", 1)
        self.shuffle_window = kwargs.get("shuffle_window", 0)
        super().__init__(sources=sources, **kwargs)

    def _build_source_dataset(self, sources):
        data_file_list = []
        if dist.get_rank() == 0:
            data_files = []
            if isinstance(sources, str) and sources.endswith(".json"):
                with open(sources, "r") as fp:
                    data_files = json.loads(fp.read())
                    data_files = [fn for fn in data_files if fn.endswith(".parquet")]
            elif isinstance(sources, list):
                for source in sources:
                    hdfs_files = shell_hdfs_ls(source)
                    data_files += [fn for fn in hdfs_files if fn.endswith(".parquet")]
            # repeat
            for i in range(self.num_epochs):
                data_files.sort()
                self.rng.shuffle(data_files)
                data_file_list += [(fn, i) for fn in data_files]
            logger.info(f"ParquetDataset rank{dist.get_rank()}: original_file_num={len(data_files)}, total_file_num={len(data_file_list)}")

        t = [data_file_list]
        dist.broadcast_object_list(t, src=0)
        data_file_list = t[0]

        logger.info(f"ParquetDataset rank{dist.get_rank()}: file_num={len(data_file_list)}")
        if len(data_file_list) == 0:
            raise ValueError(f"no datafile found!")

        dataset = Qwen3NaiveParquetDataset(data_file_list, self.num_workers, **self.kwargs)
        return dataset, -1

    def state_dict(self):
        if self.dataset is None:
            return {}
        return self.dataset.state_dict()
    
    def load_state_dict(self, state_dict):
        if self.dataset is None:
            return
        self.dataset.load_state_dict(state_dict)