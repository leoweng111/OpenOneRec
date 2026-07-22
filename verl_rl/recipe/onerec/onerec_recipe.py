"""
OneRec RL 数据加载与 Reward 函数定义
=====================================

本文件包含三个核心组件:
1. collate_fn:        batch 拼接函数，将多条样本合并为一个 batch
2. OneRecDataset:     RL 训练数据集类，负责加载 parquet 数据并构造 prompt
3. compute_score:     Reward 函数，评判模型生成的推荐结果好坏

Reward 函数体系:
  - think_format_reward:   检查 CoT 格式是否正确 (<think>...</think>)
  - first_sid_hit_reward:  Pass@1 — 第一个推荐物品是否命中 (主 reward)
  - hit_reward:            整体命中率 — 预测 SID 集合与 ground truth 的交集比例
  - partial_hit_reward:    层次化部分匹配 — 按码本层级给分 (100/10/1/0)
  - pass_rate:             二值 — 是否有任何 SID 命中
"""
from __future__ import annotations

import ast
import copy
import logging
import os
import re
from collections import defaultdict
from typing import Any, Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

__all__ = ["collate_fn", "OneRecDataset", "compute_score"]


def collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """将多条样本拼接为一个 batch。

    Tensor 类型的字段 (input_ids, attention_mask 等) 用 torch.stack 堆叠;
    非 Tensor 字段 (ground_truth, data_source 等) 用 numpy object 数组保存。
    """
    tensors: dict[str, list[torch.Tensor]] = defaultdict(list)
    non_tensors: dict[str, list[Any]] = defaultdict(list)

    for sample in samples:
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    batch: dict[str, Any] = {}
    for key, value in tensors.items():
        batch[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        batch[key] = np.array(value, dtype=object)

    return batch


class OneRecDataset(Dataset):
    """OneRec RL 训练数据集。

    数据格式 (parquet 每行):
      messages: [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]

    数据加载流程:
      1. 读取 parquet 文件
      2. _extract_prompt_fields(): 分离 prompt (system+user) 和 ground_truth (assistant)
      3. __getitem__(): tokenize prompt, 构造训练样本

    与 SFT 数据的关键区别:
      - SFT: 模型看到 system+user 后学习模仿 assistant 的回答
      - RL:  模型看到 system+user 后自己生成回答，然后由 reward 函数评判好坏
    """
    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ) -> None:
        if not isinstance(data_files, (list, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(list(data_files))
        self.original_data_files = copy.deepcopy(list(data_files))
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.enable_think = config.get("enable_think", True)
        self.enable_nonthink = config.get("enable_nonthink", False)

        self.use_force_prefix = config.get("use_force_prefix", False)
        self._FORCE_PREFIX_CONTENT = "<think>\n</think><|sid_begin|>"

        if self.enable_think and self.enable_nonthink:
            raise ValueError("enable_think and enable_nonthink cannot be both True") 

        self.num_workers = os.cpu_count()
        self.use_shm = config.get("use_shm", False)
        self.serialize_dataset = False

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet: bool = False) -> None:
        from verl.utils.fs import copy_to_local

        target_files = self.original_data_files if use_origin_parquet else self.data_files
        for idx, parquet_file in enumerate(target_files):
            local_path = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)
            target_files[idx] = local_path

        if use_origin_parquet:
            self.data_files = target_files

    def _read_files_and_tokenize(self) -> None:
        dataframes: list[datasets.Dataset] = []
        for parquet_file in self.data_files:
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)

        self.dataframe = datasets.concatenate_datasets(dataframes)  # type: ignore[attr-defined]
        logger.info("dataset len: %s", len(self.dataframe))

        if self.max_samples > 0 and self.max_samples < len(self.dataframe):
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(len(self.dataframe), size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {len(self.dataframe)}")

        self.dataframe = self.dataframe.map(
            self._extract_prompt_fields,
            num_proc=self.num_workers,
            desc="Extract prompts and reward annotations",
        )

        logger.info("processed dataset len: %s", len(self.dataframe))
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def _extract_prompt_fields(self, row: dict[str, Any]) -> dict[str, Any]:
        """从 messages 中分离 prompt 和 ground_truth。

        处理逻辑:
          - prompt_messages = messages[:-1]   (system + user，作为模型输入)
          - ground_truth = messages[-1]       (assistant 的内容，作为 reward 评判标准)
          - 根据配置在 user 消息末尾追加 /think 或 /no_think 后缀
        """
        raw_messages = row.get("messages")
        if isinstance(raw_messages, str):
            messages = ast.literal_eval(raw_messages)
        else:
            messages = raw_messages or []

        clean_chats = [
            {
                "role": message.get("role"),
                "content": "".join(segment.get("text", "") for segment in message.get("content", []) if segment.get("type") == "text"),
            }
            for message in messages
        ]

        if not clean_chats:
            raise ValueError("Sample has empty messages; please check data integrity.")

        prompt_messages = clean_chats[:-1]

        # Append /think or /no_think suffix to user messages based on config
        if self.enable_think:
            for message in prompt_messages:
                if message["role"] == "user":
                    message["content"] = message["content"] + "/think"
        if self.enable_nonthink:
            for message in prompt_messages:
                if message["role"] == "user":
                    message["content"] = message["content"] + "/no_think"


        ground_truth_message = clean_chats[-1]["content"]

        reward_payload = {
            "ground_truth": ground_truth_message,
            "style": "rule",
        }

        row[self.prompt_key] = prompt_messages
        row["reward_model"] = reward_payload
        return row

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset) -> datasets.Dataset:
        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer = self.tokenizer
        processor = self.processor
        prompt_key = self.prompt_key
        image_key = self.image_key
        video_key = self.video_key

        if processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            def doc_length(doc: dict[str, Any]) -> int:
                messages = self._build_messages(dict(doc))
                raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                images = [process_image(image) for image in doc.get(image_key, [])]
                videos = [process_video(video) for video in doc.get(video_key, [])]
                encoded = processor(text=[raw_prompt], images=images or None, videos=videos or None, return_tensors="pt")
                return int(encoded["input_ids"].shape[-1])

        else:

            def doc_length(doc: dict[str, Any]) -> int:
                messages = doc[prompt_key]
                return len(tokenizer.apply_chat_template(messages, add_generation_prompt=True))

        filtered = dataframe.filter(
            lambda doc: doc_length(doc) <= self.max_prompt_length - 10,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length - 10} tokens",
        )

        logger.info("filtered dataset len: %s", len(filtered))
        return filtered

    def resume_dataset_state(self) -> None:
        self.serialize_dataset = not hasattr(self, "original_data_files")
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)
            self._read_files_and_tokenize()
        else:
            logger.warning("resume with serialized dataloader, consider restarting from scratch for better perf")

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.dataframe)

    def _build_messages(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                segments = [segment for segment in re.split(r"(<image>|<video>)", content) if segment]
                parsed_segments = []
                for segment in segments:
                    if segment == "<image>":
                        parsed_segments.append({"type": "image"})
                    elif segment == "<video>":
                        parsed_segments.append({"type": "video"})
                    else:
                        parsed_segments.append({"type": "text", "text": segment})
                message["content"] = parsed_segments

        return messages

    def __getitem__(self, index: int) -> dict[str, Any]:  # type: ignore[override]
        row: dict[str, Any] = dict(self.dataframe[index])
        messages = self._build_messages(dict(row))
        model_inputs: dict[str, Any] = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            if self.use_force_prefix:
                raw_prompt = raw_prompt + self._FORCE_PREFIX_CONTENT

            multi_modal_data: dict[str, Any] = {}

            images = None
            if self.image_key in row and row.get(self.image_key):
                images = [process_image(image) for image in row.pop(self.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row and row.get(self.video_key):
                videos = [process_video(video) for video in row.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(
                text=[raw_prompt],
                images=images,
                videos=videos,
                return_tensors="pt",
            )

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            row["multi_modal_data"] = multi_modal_data
            if self.return_multi_modal_inputs:
                mm_inputs = dict(model_inputs)
                mm_inputs.pop("second_per_grid_ts", None)
                row["multi_modal_inputs"] = mm_inputs
        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            if self.use_force_prefix:
                raw_prompt = raw_prompt + self._FORCE_PREFIX_CONTENT

            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if (
            self.processor is not None
            and hasattr(self.processor, "image_processor")
            and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
        ):
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row["input_ids"] = input_ids[0]
        row["attention_mask"] = attention_mask[0]
        row["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            raw_prompt_ids = self._truncate_ids(raw_prompt_ids)

        row["raw_prompt_ids"] = raw_prompt_ids
        if self.return_raw_chat:
            row["raw_prompt"] = messages
        if self.return_full_prompt:
            row["full_prompts"] = raw_prompt

        extra_info = row.get("extra_info", {}) or {}
        row["index"] = extra_info.get("index", index)
        row["tools_kwargs"] = extra_info.get("tools_kwargs", {})
        row["interaction_kwargs"] = extra_info.get("interaction_kwargs", {})


        if "source" in row or "data_source" in row:
            pass
        else:
            row["data_source"] = "unknown"
            logger.warning("No source/data_source field found for index %s, set to 'unknown'", row["index"])

        if self.need_tools_kwargs and not row["tools_kwargs"]:
            logger.warning("tools_kwargs is empty for index %s, data source: %s", row["index"], row.get("data_source", row.get("source", "unknown")))

        return row

    def _truncate_ids(self, token_ids: list[int]) -> list[int]:
        if self.truncation == "left":
            return token_ids[-self.max_prompt_length :]
        if self.truncation == "right":
            return token_ids[: self.max_prompt_length]
        if self.truncation == "middle":
            left = self.max_prompt_length // 2
            right = self.max_prompt_length - left
            return token_ids[:left] + token_ids[-right:]
        if self.truncation == "error":
            raise RuntimeError(
                f"Prompt length {len(token_ids)} exceeds max_prompt_length={self.max_prompt_length}. "
                "Consider increasingmax_prompt_length or enabling truncation."
            )
        raise ValueError(f"Unsupported truncation mode: {self.truncation}")

    def __getstate__(self) -> dict[str, Any]:
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            state.pop("dataframe", None)
            return state
        return self.__dict__.copy()


# ============================================================================
# Reward 函数体系
# ============================================================================
# 以下函数评判模型生成的推荐结果 (prediction) 与正确答案 (ground_truth) 的匹配程度。
# 每个函数返回一个标量 reward，用于 GRPO 优势估计。
#
# SID 格式: <s_a_X><s_b_Y><s_c_Z>  (三层码本的离散编码)
# 例如: <s_a_42><s_b_15><s_c_89> 表示一个物品的语义ID

# 正则表达式: 匹配 SID 三元组 (s_a, s_b, s_c)
SLOT_PATTERN = re.compile(r"<s_a_(\d+)><s_b_(\d+)><s_c_(\d+)>")


def _extract_all_tuples(text: Any) -> list[tuple[str, str, str]]:
    """从文本中提取所有 SID 三元组。

    例如: "...<s_a_42><s_b_15><s_c_89>...<s_a_7><s_b_33><s_c_201>..."
      → [("42", "15", "89"), ("7", "33", "201")]
    """
    if not isinstance(text, str):
        logger.warning("_extract_all_tuples received non-string input: %s", type(text))
        return []

    matches = SLOT_PATTERN.findall(text)
    return [tuple(match) for match in matches] if matches else []


def think_format_reward(prediction: str) -> float:
    """格式奖励: 检查模型是否遵循了 <think>...</think> 的推理格式。

    评判标准:
      - 必须包含 <think> 和 </think> 标签
      - 标签内的内容 (去除空白后) 长度必须 > 10 个字符
      → 防止模型生成空的 </think> 来"骗"奖励

    返回: 1.0 (格式正确) 或 0.0 (格式错误)
    """
    if "<think>" not in prediction or "</think>" not in prediction:
        return 0.0

    start_idx = prediction.find("<think>") + len("<think>")
    end_idx = prediction.find("</think>")

    if end_idx < start_idx:
        return 0.0

    content = prediction[start_idx:end_idx]
    content_stripped = content.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")

    return 1.0 if len(content_stripped) > 10 else 0.0


def partial_hit_reward(prediction: str, ground_truth: str) -> float:
    """层次化部分匹配奖励 — GRPO 中提供"方向正确"的梯度信号。

    即使模型没有完全命中目标物品，只要码本的部分层级匹配了，也能获得正向奖励:
      - s_a + s_b + s_c 全部匹配 → 100 分 (完全命中)
      - s_a + s_b 匹配           → 10 分  (粗粒度+中粒度对了)
      - 仅 s_a 匹配              → 1 分   (粗粒度对了)
      - 都不匹配                 → 0 分

    设计动机:
      pass@1 只返回 0 或 1，信号极度稀疏。32 条 beam 中大部分都没命中 → 全得 0。
      partial_hit_reward 给出"部分匹配"的梯度信号，让模型知道"方向是对的"，
      即使没完全命中也能学到有用的信息。

    返回: 所有预测 SID 的平均匹配分数。
    """
    pred_tuples = _extract_all_tuples(prediction)
    gt_tuples = _extract_all_tuples(ground_truth)

    if not pred_tuples or not gt_tuples:
        return 0.0

    total_reward = 0.0

    # Find best match for each predicted sid and calculate score
    for pred_tuple in pred_tuples:
        max_score = 0.0
        
        for gt_tuple in gt_tuples:
            # Full match (s_a, s_b, s_c)
            if pred_tuple == gt_tuple:
                max_score = max(max_score, 100.0)
            # s_a and s_b match
            elif pred_tuple[:2] == gt_tuple[:2]:
                max_score = max(max_score, 10.0)
            # Only s_a match
            elif pred_tuple[0] == gt_tuple[0]:
                max_score = max(max_score, 1.0)
        
        total_reward += max_score
    
    # Return average score to avoid inflated scores with multiple predictions
    return total_reward / len(pred_tuples)

def hit_reward(prediction: str, ground_truth: str) -> float:
    """整体命中率奖励: 预测 SID 集合与 ground truth 的交集比例。

    计算方式: |预测 ∩ 正确| / |预测|
    只统计 </think> 之后的 SID (推理结果部分)。

    返回: [0, 1] 连续值，1.0 表示所有预测的 SID 都命中。
    """
    if "</think>" in prediction and "<think>" in prediction:
        think_end_idx = prediction.find("</think>") + len("</think>")
        prediction = prediction[think_end_idx:]
    else:
        return 0.0


    pred_tuples = _extract_all_tuples(prediction)
    gt_tuples = _extract_all_tuples(ground_truth)
    if not pred_tuples or not gt_tuples:
        return 0.0

    pred_set = set(pred_tuples)
    gt_set = set(gt_tuples)
    return len(pred_set & gt_set) / len(pred_tuples)

def first_sid_hit_reward(prediction: str, ground_truth: str) -> float:
    """Pass@1 奖励 (主 reward): 模型推荐的第一个物品是否命中 ground truth。

    这是 compute_score 返回的 "score" 字段，是 GRPO 优势估计的主要信号。
    只检查 </think> 之后第一个 SID 三元组。

    返回: 1.0 (命中) 或 0.0 (未命中)
    """
    # Extract content after </think>
    if "</think>" in prediction and "<think>" in prediction:
        think_end_idx = prediction.find("</think>") + len("</think>")
        prediction = prediction[think_end_idx:]
    else:
        return 0.0

    pred_tuples = _extract_all_tuples(prediction)
    if not pred_tuples:
        return 0.0

    # Get the first predicted sid tuple
    first_pred_tuple = pred_tuples[0]

    gt_tuples = _extract_all_tuples(ground_truth)
    if not gt_tuples:
        return 0.0

    gt_set = set(gt_tuples)
    
    return float(first_pred_tuple in gt_set)

def pass_rate(prediction: str, ground_truth: str) -> float:
    """通过率: 预测集合与 ground truth 是否有交集 (二值)。

    返回: 1.0 (至少一个 SID 命中) 或 0.0 (全部未命中)
    """
    pred_tuples = _extract_all_tuples(prediction)
    gt_tuples = _extract_all_tuples(ground_truth)
    if not pred_tuples or not gt_tuples:
        return 0.0

    # Convert to set for intersection calculation
    pred_set = set(pred_tuples)
    gt_set = set(gt_tuples)
    intersection_count = len(pred_set & gt_set)
    
    return float(intersection_count > 0)



def compute_score(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any],  # noqa: ARG001
) -> dict[str, float]:
    """计算推荐结果的 reward 分数 — RL 训练的核心评判函数。

    本函数由 run_grpo.sh 中 custom_reward_function.path/name 指定，
    在训练循环的 "Step 3: 计算Reward" 阶段被调用。

    参数:
        data_source: 数据来源标识 (如 "RecIF_VideoRec")，当前未使用
        solution_str: 模型生成的完整输出 (CoT推理 + SID序列)
        ground_truth: 正确的 SID 序列字符串
        extra_info:   额外信息，当前未使用

    返回:
        字典，包含多个 reward 指标:
        - "score" (主 reward): pass_at_1，用于 GRPO 优势估计
        - 其余为监控指标，记录到 wandb/tensorboard
    """
    prediction = solution_str
    format_reward_value = think_format_reward(prediction)
    partial_hit_reward_value = partial_hit_reward(prediction, ground_truth)
    hit_reward_value = hit_reward(prediction, ground_truth)
    pass_rate_value = pass_rate(prediction, ground_truth)
    pass_at_1_value = first_sid_hit_reward(prediction, ground_truth)

    return {
        "score": pass_at_1_value,
        "format_reward": format_reward_value,
        "partial_hit_reward": partial_hit_reward_value,
        "hit_reward": hit_reward_value,
        "pass_rate": pass_rate_value,
        "pass_at_1": pass_at_1_value,
    }