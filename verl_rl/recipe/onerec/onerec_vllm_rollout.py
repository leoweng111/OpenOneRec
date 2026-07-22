"""
OneRec 两阶段 Rollout 生成
==========================

本文件实现 OneRec 特有的两阶段生成策略 (Two-Stage Generation):

  Stage 1 — CoT 推理采样:
    模型以 sampling 方式生成推理链 (Chain-of-Thought)
    输入: prompt tokens (system + user + /think)
    停止: 遇到 </think> 或达到 max_tokens (默认1024)
    特点: 鼓励多样性推理，温度 T=1.0

  Stage 2 — 物品 Beam Search:
    模型以 beam search 方式搜索最优物品 SID
    输入: [prompt + CoT + </think>] + [<|sid_begin|>]
    参数: beam_width=32 (默认), max_tokens=16
    特点: 精确搜索最优物品组合

  为什么分两阶段?
    CoT 是自由文本 → 适合 sampling (鼓励多样性)
    SID 是离散码   → 适合 beam search (精确搜索)
    两种解码策略不兼容，所以分开做。

  扩展机制:
    每个 prompt 生成 beam_width=32 条候选
    batch 从 [B] 扩展到 [B×32]
    同一 prompt 的 32 条候选共享 UID，用于 GRPO 组内比较
"""
import numpy as np
import torch
from tensordict import TensorDict
from vllm import SamplingParams
from vllm.lora.request import LoRARequest
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.vllm_rollout.vllm_rollout_spmd import vLLMRollout, _pre_process_inputs

try:
    from vllm.sampling_params import BeamSearchParams
except ImportError:
    BeamSearchParams = None


class OneRecvLLMRollout(vLLMRollout):
    """OneRec 自定义 vLLM Rollout，实现两阶段生成。

    继承自 vLLMRollout，重写 generate_sequences() 方法:
      标准 vLLMRollout: 一步生成 (sampling 或 beam search)
      OneRecvLLMRollout: 两步生成 (Stage1 sampling → Stage2 beam search)
    """

    @torch.no_grad()
    def _two_stage_generation(self, prompts: DataProto, **kwargs) -> DataProto:
        """两阶段生成的核心实现。

        数据流:
          [B 条 prompt] → Stage 1 (CoT采样) → [B 条 prompt+CoT]
                       → Stage 2 (Beam搜索) → [B×beam_width 条候选序列]
        """
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)
        
        # Prepare vllm inputs (same as standard)
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )
            
        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data"), strict=True
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]
            
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()

        # ====================================================================
        # Stage 1: CoT 推理采样
        # ====================================================================
        # 用 sampling 方式生成推理链 (Chain-of-Thought)
        # 遇到 </think> 自动停止，最多生成 stage1_max_tokens 个 token
        # n=1: 每个 prompt 只生成 1 条 CoT (beam search 在 Stage 2 做)
        stage1_max_tokens = kwargs.get("stage1_max_tokens",
                            getattr(self.config, "stage1_max_tokens",
                            kwargs.get("max_tokens", 1024)))  # fallback to max_tokens or 1024

        cot_sampling_params = SamplingParams(
            n=1, # We generate 1 CoT per prompt (prompts are already repeated by Trainer if needed)
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=stage1_max_tokens,  # Use stage1_max_tokens for CoT length
            stop=["</think>"],
            include_stop_str_in_output=True,
        )

        print(f"[TwoStage] Stage 1 params: max_tokens={stage1_max_tokens}, temperature={kwargs.get('temperature', 1.0)}")
        
        lora_requests = None # Assuming LoRA logic is same as standard, omitted for brevity or copy if needed
        if self.lora_kwargs:
             # Copy lora logic
             lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
             if len(lora_int_ids) > 0:
                 lora_int_id = lora_int_ids[0]
                 lora_requests = [
                     LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                 ] * batch_size

        # print(f"[TwoStage] Starting Stage 1: CoT Generation...")
        cot_outputs = self.inference_engine.generate(
            prompts=vllm_inputs,
            sampling_params=cot_sampling_params,
            lora_request=lora_requests,
            use_tqdm=False,
        )
        
        # ====================================================================
        # 处理 Stage 1 输出，构造 Stage 2 输入
        # ====================================================================
        # Stage 2 的输入 = 原始 prompt + CoT 输出 + 前缀 "\n<|sid_begin|>"
        # 前缀告诉模型: "推理结束，现在开始推荐物品"
        stage2_inputs = []
        cot_responses = []

        # We need tokenizer to encode the prefix
        tokenizer = self.inference_engine.get_tokenizer()
        # Prefix: \n<|sid_begin|> (assuming </think> is already in output)
        # Note: if model stopped because of length, </think> might be missing.
        # We should handle this.
        prefix_ids = tokenizer.encode("\n<|sid_begin|>", add_special_tokens=False)

        # Get vocab size for OOV filtering
        vocab_size = len(tokenizer)

        for i, output in enumerate(cot_outputs):
            # Get CoT tokens
            cot_token_ids = list(output.outputs[0].token_ids)

            # Filter OOV tokens from CoT output
            cot_token_ids_filtered = [tid for tid in cot_token_ids if tid < vocab_size]
            if len(cot_token_ids_filtered) < len(cot_token_ids):
                print(f"[TwoStage] Filtered {len(cot_token_ids) - len(cot_token_ids_filtered)} OOV tokens from CoT output {i}")

            cot_responses.append(cot_token_ids_filtered)

            # Construct Stage 2 Prompt: Original Prompt + CoT + Prefix
            original_prompt_ids = vllm_inputs[i]["prompt_token_ids"]
            new_prompt_ids = original_prompt_ids + cot_token_ids_filtered + prefix_ids

            stage2_input = {"prompt_token_ids": new_prompt_ids}
            if "multi_modal_data" in vllm_inputs[i]:
                stage2_input["multi_modal_data"] = vllm_inputs[i]["multi_modal_data"]
            stage2_inputs.append(stage2_input)

        # ====================================================================
        # Stage 2: 物品 Beam Search
        # ====================================================================
        # 在 CoT 推理完成后，用 beam search 精确搜索最优物品 SID 组合
        # beam_width=32: 同时保留 32 条最优候选 (用于 GRPO 组内比较)
        # max_tokens=16: 最多生成 16 个 token (足够放多个 SID 三元组)
        beam_width = kwargs.get("stage2_beam_size", getattr(self.config, "stage2_beam_size", 32))
        # Support both stage2_max_tokens and stage2_num_tokens (for backward compatibility)
        max_tokens_item = kwargs.get("stage2_max_tokens",
                          kwargs.get("stage2_num_tokens",
                          getattr(self.config, "stage2_max_tokens",
                          getattr(self.config, "stage2_num_tokens", 16))))

        print(f"[TwoStage] Stage 2 params: beam_width={beam_width}, max_tokens={max_tokens_item}, batch_size={batch_size}")

        if BeamSearchParams is None:
             raise ImportError("BeamSearchParams not available, cannot run Stage 2")

        beam_params = BeamSearchParams(
            beam_width=beam_width,
            max_tokens=max_tokens_item,
        )

        # Call beam search (aligned with standard implementation)
        item_outputs = self.inference_engine.beam_search(
            prompts=stage2_inputs,
            params=beam_params,
        )

        # ====================================================================
        # 后处理: 展开 beam search 结果
        # ====================================================================
        # return_all_beams=True 时，将每个 prompt 的 32 条 beam 全部返回
        # 输出从 [B] 扩展到 [B×beam_width]
        # 同一 prompt 的 32 条候选共享 UID → GRPO 组内标准化时互相比较
        return_all_beams = kwargs.get("return_all_beams", True)
        # For two-stage rollout, n_beams_to_return should be beam_width, not kwargs["n"] (which is rollout_n)
        n_beams_to_return = beam_width

        print(f"[TwoStage] Post-process: return_all_beams={return_all_beams}, n_beams_to_return={n_beams_to_return}")

        response = []

        if return_all_beams:
            # Return all beams, expand output
            # Output will be exactly batch_size * n_beams_to_return (pad if needed)
            expanded_idx = []
            beam_indices = []  # Track which beam index within each prompt

            for i, output in enumerate(item_outputs):
                # Prompt length including CoT + Prefix
                stage2_prompt_len = len(stage2_inputs[i]["prompt_token_ids"])
                original_prompt_len = len(vllm_inputs[i]["prompt_token_ids"])

                # Get top n beams for this prompt, pad if not enough
                num_seqs = len(output.sequences)
                for seq_idx in range(n_beams_to_return):
                    if seq_idx < num_seqs:
                        best_seq = output.sequences[seq_idx]
                        full_seq = best_seq.tokens
                        # Response = full_seq - original_prompt (not stage2_prompt!)
                        response_ids = full_seq[original_prompt_len:]
                    else:
                        # Pad with first beam's result if not enough beams
                        best_seq = output.sequences[0]
                        full_seq = best_seq.tokens
                        response_ids = full_seq[original_prompt_len:]
                    response.append(response_ids)
                    expanded_idx.append(i)
                    beam_indices.append(seq_idx)

            # Expand idx, attention_mask, position_ids to match expanded output
            idx = idx[expanded_idx]  # (batch_size * n, prompt_length)
            attention_mask = attention_mask[expanded_idx]
            position_ids = position_ids[expanded_idx]

            # Expand non_tensor_batch to match expanded output
            expanded_non_tensor_batch = {}
            for key, val in non_tensor_batch.items():
                if isinstance(val, np.ndarray):
                    expanded_non_tensor_batch[key] = val[expanded_idx]
                elif isinstance(val, list):
                    expanded_non_tensor_batch[key] = [val[i] for i in expanded_idx]
                else:
                    expanded_non_tensor_batch[key] = val
            non_tensor_batch = expanded_non_tensor_batch

            # Store beam indices for reference
            non_tensor_batch["_beam_indices"] = np.array(beam_indices, dtype=np.int64)

            batch_size = len(response)  # Update batch_size

            print(f"[TwoStage] Expanded output: original_bs={len(item_outputs)}, expanded_bs={batch_size}, n_beams={n_beams_to_return}")
        else:
            # Original path: use beam_idx to select specific beam
            beam_idxs = non_tensor_batch.get("beam_idx", None)

            for i, output in enumerate(item_outputs):
                original_prompt_len = len(vllm_inputs[i]["prompt_token_ids"])

                seq_idx = 0
                if beam_idxs is not None:
                    seq_idx = beam_idxs[i]

                if seq_idx >= len(output.sequences):
                    seq_idx = 0

                best_seq = output.sequences[seq_idx]
                full_seq = best_seq.tokens
                response_ids = full_seq[original_prompt_len:]
                response.append(response_ids)

        # Pad responses
        response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(idx.device)

        if self.config.calculate_log_probs:
            rollout_log_probs = torch.zeros_like(response, dtype=torch.float32)

        seq = torch.cat([idx, response], dim=-1)

        # Position IDs & Attention Mask Update (standard logic)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)

        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)


    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """
        Generate sequences using two-stage generation.
        """
        # Extract params from meta_info and merge into kwargs
        for key in ["max_tokens", "temperature", "n", "top_p", "top_k",
                    "stage2_beam_size", "stage2_max_tokens", "return_all_beams"]:
            if key in prompts.meta_info:
                kwargs[key] = prompts.meta_info[key]

        return self._two_stage_generation(prompts, **kwargs)
