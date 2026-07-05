"""RecIF-Bench 评测的正式入口脚本（由 `eval_script.sh` 调用）。

流程：
    1) HfArgumentParser 解析 6 组 dataclass 参数（模型/基础设施/推理/生成/prompt/benchmark）。
    2) 构造 Benchmark 对象 —— 传入 model_path 用于 tokenizer + 待评测任务集合。
    3) 构造 RayVllmGenerator —— 用 Ray + vLLM 起分布式推理集群；tensor_parallel_size 决定 TP 分片，
        num_gpus 决定总卡数；支持多节点。
    4) benchmark.run(generator, ...) → 每个 task 走一次 generation_runner，
        num_return_sequences 决定每个 prompt 生成多少条候选（推荐类任务默认 128）。
    5) 生成完释放 vLLM 显存，脚本外面再调 `eval_dev_results.py` 计算指标。

关键并行方式：Ray 起若干 vllm worker，每个 worker 拿到一批 prompt 独立生成；
    多 GPU 用 tensor_parallel_size 切模型，多节点用 --ray_address 连集群。
"""

from transformers import HfArgumentParser
import torch

from benchmark import Benchmark
from benchmark.console import *
from utils.generator import RayVllmGenerator
from utils.arguments import (
    ModelConfig,
    InfrastructureConfig,
    InferenceConfig,
    GenerationConfig,
    PromptConfig,
    BenchmarkConfig
)


def main():
    parser = HfArgumentParser([
        ModelConfig,
        InfrastructureConfig,
        InferenceConfig,
        GenerationConfig,
        PromptConfig,
        BenchmarkConfig
    ])
    model_config, infra_config, inference_config, generation_config, prompt_config, benchmark_config = \
        parser.parse_args_into_dataclasses()

    # 1. Initialize Benchmark
    benchmark = Benchmark(
        model_path=model_config.model_path,
        task_types=benchmark_config.task_types,
        splits=benchmark_config.splits,
        data_dir=benchmark_config.data_dir,
        enable_thinking=prompt_config.enable_thinking,
    )
    # Benchmark.print_benchmark_table()

    # 2. Initialize Ray + vLLM generator (Multi-Node Support)
    generator = RayVllmGenerator(
        model_name_or_path=model_config.model_path,
        checkpoint_path=model_config.checkpoint_path,
        trust_remote_code=model_config.trust_remote_code,
        dtype=model_config.dtype,
        max_model_len=model_config.max_model_len,
        max_logprobs=model_config.max_logprobs,
        gpu_memory_utilization=infra_config.gpu_memory_utilization,
        tensor_parallel_size=infra_config.tensor_parallel_size,
        ray_address=infra_config.ray_address,  # Ray cluster address
        allow_cross_node_tensor_parallel=infra_config.allow_cross_node_tensor_parallel,  # Cross-node TP
        num_gpus=infra_config.num_gpus,
        gpu_ids=infra_config.gpu_ids,
        force_enable_optimizations=inference_config.force_enable_optimizations,
        force_disable_optimizations=inference_config.force_disable_optimizations,
        worker_batch_size=inference_config.worker_batch_size,
        task_types=benchmark_config.task_types
    )

    # 3. Generate text
    benchmark.run(
        generator=generator,
        output_dir=benchmark_config.output_dir,
        overwrite=benchmark_config.overwrite,
        # Generation parameters
        enable_thinking=prompt_config.enable_thinking,
        num_beams=generation_config.num_beams,
        num_return_sequences=generation_config.num_return_sequences,
        temperature=generation_config.temperature,
        top_p=generation_config.top_p,
        top_k=generation_config.top_k,
        presence_penalty=generation_config.presence_penalty,
        num_return_thinking_sequences=generation_config.num_return_thinking_sequences,
        sample_size=benchmark_config.sample_size,
    )

    # 4. Release GPU memory occupied by vLLM
    console.print("\nReleasing vLLM GPU memory...", style=warning_style)
    generator.cleanup()
    del generator
    import gc
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    console.print("✓ GPU memory release completed\n", style=success_style)

    # 5. Calculate evaluation metrics
    eval_results_path = f"{benchmark_config.output_dir}/eval_results.json"
    Benchmark.evaluate_dev(
        generation_results_dir=benchmark_config.output_dir,
        output_path=eval_results_path,
        data_dir=benchmark_config.data_dir,
        overwrite=benchmark_config.overwrite,
        task_types=benchmark_config.task_types
    )


if __name__ == "__main__":
    main()

