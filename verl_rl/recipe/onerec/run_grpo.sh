#!/bin/bash
# ============================================================================
# OneRec GRPO 训练启动脚本
# ============================================================================
# 使用两阶段 Rollout (CoT采样 + Beam Search) 进行 GRPO 强化学习训练。
#
# 训练流程:
#   1. 加载 SFT 后的模型作为初始策略
#   2. 每个 prompt 生成 32 条候选 (beam search)
#   3. compute_score() 计算每条候选的 reward
#   4. GRPO 组内标准化 → 优势估计
#   5. PPO-clip 策略梯度更新模型
#   6. 循环直到收敛
#
# 用法:
#   bash run_grpo.sh                           # 使用默认参数
#   BASE_MODEL=/path/to/model bash run_grpo.sh # 指定模型路径
#   STAGE2_BEAM_SIZE=64 bash run_grpo.sh       # 自定义 beam 宽度
# ============================================================================

set -e

# ============================================================================
# Cluster Configuration (auto-detect from Ray)
# ============================================================================
RAY_INFO=$(python -c "import ray; ray.init(address='auto', ignore_reinit_error=True); nodes = [n for n in ray.nodes() if n['Alive']]; gpus=next((int(n.get('Resources',{}).get('GPU',0)) for n in nodes if n.get('Resources',{}).get('GPU',0)>0), 0); print(f'{len(nodes)} {gpus}')" 2>/dev/null)

export N_NODES=$(echo $RAY_INFO | awk '{print $1}')
export N_GPUS=$(echo $RAY_INFO | awk '{print $2}')

if [ -z "$N_NODES" ] || [ -z "$N_GPUS" ] || [ "$N_NODES" -eq 0 ]; then
    echo "Could not detect Ray cluster. Using defaults: N_NODES=1, N_GPUS=8"
    export N_NODES=1
    export N_GPUS=8
else
    echo "Detected Ray cluster: $N_NODES nodes, $N_GPUS GPUs per node"
fi

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ============================================================================
# Model Configuration
# ============================================================================
export BASE_MODEL=${BASE_MODEL:-"/path/to/your/model"}
export ROLLOUT_TP_SIZE=${ROLLOUT_TP_SIZE:-1}
export VLLM_ATTENTION_BACKEND=XFORMERS

# ============================================================================
# Training Hyperparameters
# ============================================================================
export LEARNING_RATE=${LEARNING_RATE:-2e-6}
export KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
export TEMPERATURE=${TEMPERATURE:-1}

# ============================================================================
# Batch Size Configuration
# ============================================================================
export USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
export MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-40960}
export TRAIN_BATCH_SIZE=$((N_GPUS * N_NODES))

# ============================================================================
# Rollout Configuration
# ============================================================================
export ROLLOUT_N=${ROLLOUT_N:-1}
export STAGE2_BEAM_SIZE=${STAGE2_BEAM_SIZE:-32}
export RESPONSE_LENGTH=${RESPONSE_LENGTH:-2048}
export STAGE1_MAX_TOKENS=${STAGE1_MAX_TOKENS:-1024}
export STAGE2_NUM_TOKENS=${STAGE2_NUM_TOKENS:-3}

# Think mode configuration
export ENABLE_THINK=${ENABLE_THINK:-False}
export ENABLE_NONTHINK=${ENABLE_NONTHINK:-False}
export USE_FORCE_PREFIX=${USE_FORCE_PREFIX:-False}

# ============================================================================
# Data Configuration
# ============================================================================
export DATA_DIR=${DATA_DIR:-"$(realpath ../output/rl_data)"}
export TRAIN_FILES=${TRAIN_FILES:-"[$DATA_DIR/train.parquet]"}
export VAL_FILES=${VAL_FILES:-"[$DATA_DIR/test.parquet]"}

# ============================================================================
# Output Configuration
# ============================================================================
export PROJECT_NAME=${PROJECT_NAME:-"OneRec_RL"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo_two_stage"}
export OUTPUT_DIR=${OUTPUT_DIR:-"./output"}
export WANDB_MODE=${WANDB_MODE:-offline}

# ============================================================================
# Network Configuration (for distributed training)
# ============================================================================
export TCP_NIC=$(ifconfig 2>/dev/null | grep -B1 " "$(hostname -i 2>/dev/null)" " | grep -o "^\w*" || echo "eth0")
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}

# ============================================================================
# Print Configuration
# ============================================================================
echo "==================================="
echo "GRPO Training with Two-Stage Rollout"
echo "==================================="
echo "Model: $BASE_MODEL"
echo "Cluster: $N_NODES nodes x $N_GPUS GPUs"
echo "Batch Size: $TRAIN_BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Rollout N: $ROLLOUT_N"
echo "Stage2 Beam Size: $STAGE2_BEAM_SIZE"
echo "Enable Think: $ENABLE_THINK"
echo "Enable NonThink: $ENABLE_NONTHINK"
echo "==================================="

# ============================================================================
# Launch Training
# ============================================================================
mkdir -p logs

conda activate verl

python3 -u -m recipe.onerec.main_onerec_ppo \
    algorithm.adv_estimator=grpo \              # 使用 GRPO 优势估计 (不需要 Critic 网络)
    data.train_files=$TRAIN_FILES \             # 训练数据 (parquet)
    data.val_files=$VAL_FILES \                 # 验证数据
    data.max_prompt_length=10240 \              # prompt 最大长度 (token 数)
    ++data.enable_think=$ENABLE_THINK \         # 是否启用 <think> 推理模式
    ++data.enable_nonthink=$ENABLE_NONTHINK \   # 是否启用非推理模式
    ++data.use_force_prefix=$USE_FORCE_PREFIX \ # 是否强制 </think><|sid_begin|> 前缀
    data.prompt_key='prompt' \                  # 数据中 prompt 字段的 key
    data.shuffle=True \                         # 数据 shuffle
    data.max_response_length=$RESPONSE_LENGTH \ # 生成序列最大长度 (CoT + SID)
    data.train_batch_size=$TRAIN_BATCH_SIZE \   # batch size = GPU数 × 节点数
    data.filter_overlong_prompts=True \         # 过滤超长 prompt
    data.truncation='error' \                   # 超长时报错 (不截断)
    data.custom_cls.path=$SCRIPT_DIR/onerec_recipe.py \   # 自定义数据集类路径
    data.custom_cls.name=OneRecDataset \                  # 自定义数据集类名
    data.reward_fn_key='source' \                         # reward 函数路由 key
    ++data.data_source_key='source' \                     # 数据来源标识 key
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \  # 参考策略熵计算优化
    actor_rollout_ref.actor.entropy_checkpointing=True \             # 梯度检查点节省显存
    actor_rollout_ref.rollout.enable_chunked_prefill=True \          # vLLM 分块 prefill
    actor_rollout_ref.rollout.calculate_log_probs=False \            # rollout 时不计算 log_prob
    actor_rollout_ref.actor.clip_ratio_high=0.28 \                  # PPO clip 上限 (ratio ≤ 1.28)
    actor_rollout_ref.model.enable_activation_offload=True \         # 激活值 offload 省显存
    actor_rollout_ref.model.use_remove_padding=True \                # 去除 padding 优化计算
    custom_reward_function.path=$SCRIPT_DIR/onerec_recipe.py \      # Reward 函数文件路径
    custom_reward_function.name=compute_score \                      # Reward 函数名
    actor_rollout_ref.actor.use_dynamic_bsz=$USE_DYNAMIC_BSZ \      # 动态 batch size
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \  # 每 GPU 最大 token 数
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \          # PPO mini batch
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \ # 参考策略 log_prob 限制
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$MAX_TOKENS_PER_GPU \ # rollout log_prob 限制
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_TOKENS_PER_GPU \ # vLLM 最大 batch token
    actor_rollout_ref.rollout.max_num_seqs=2048 \                   # vLLM 最大序列数
    actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \               # Actor 学习率
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \              # 学习率 warmup 步数
    actor_rollout_ref.actor.optim.weight_decay=0.1 \                # 权重衰减
    actor_rollout_ref.model.path=$BASE_MODEL \                      # 模型路径 (SFT checkpoint)
    actor_rollout_ref.model.enable_gradient_checkpointing=True \    # 梯度检查点
    actor_rollout_ref.rollout.n=$ROLLOUT_N \                        # 每个 prompt 的 CoT 采样数
    actor_rollout_ref.rollout.dtype=bfloat16 \                      # 推理精度
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \ # 张量并行度
    actor_rollout_ref.rollout.name=two_stage \                      # 使用两阶段 Rollout
    ++actor_rollout_ref.rollout.backend=vllm \                      # 推理后端: vLLM
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \          # vLLM GPU 显存利用率
    ++actor_rollout_ref.rollout.max_length=$RESPONSE_LENGTH \       # 最大生成长度
    ++actor_rollout_ref.rollout.stage1_max_tokens=$STAGE1_MAX_TOKENS \  # Stage 1 CoT 最大长度
    ++actor_rollout_ref.rollout.stage2_num_tokens=$STAGE2_NUM_TOKENS \  # Stage 2 SID 最大 token 数
    ++actor_rollout_ref.rollout.stage2_beam_size=$STAGE2_BEAM_SIZE \    # Stage 2 beam 宽度 (默认32)
    ++actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs=320 \   # vLLM 最大 logprobs
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \            # CoT 采样温度
    actor_rollout_ref.rollout.top_p=1.0 \                           # CoT 采样 top_p
    actor_rollout_ref.rollout.do_sample=True \                      # CoT 启用采样 (非贪心)
    actor_rollout_ref.actor.use_kl_loss=True \                      # 启用 KL 正则 (防止策略偏移过大)
    actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \           # KL 正则系数 β (默认0.001)
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \               # KL 散度计算方式
    algorithm.norm_adv_by_std_in_grpo=True \                        # GRPO 是否用标准差标准化优势
    algorithm.use_kl_in_reward=False \                              # 不在 reward 中加入 KL 惩罚
    trainer.default_hdfs_dir=null \                                 # 不使用 HDFS
    trainer.n_gpus_per_node=$N_GPUS \                               # 每节点 GPU 数
    trainer.nnodes=$N_NODES \                                       # 节点数
    trainer.save_freq=50 \                                          # 每 50 步保存 checkpoint
    trainer.test_freq=50 \                                          # 每 50 步验证
    trainer.project_name=$PROJECT_NAME \                            # wandb 项目名
    trainer.experiment_name=$EXPERIMENT_NAME \                      # wandb 实验名
    trainer.default_local_dir=$OUTPUT_DIR/ckpt \                    # checkpoint 保存路径
    trainer.total_epochs=20 \                                       # 总训练 epoch 数
    trainer.val_before_train=True \                                 # 训练前先验证一次
    actor_rollout_ref.ref.strategy=fsdp2 \                          # 参考策略并行方式: FSDP2
    actor_rollout_ref.actor.strategy=fsdp2 \                        # Actor 并行方式: FSDP2
    ++critic.enable=False \                                         # 禁用 Critic (GRPO 不需要)
    ++actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \    # Actor 模型精度
    ++actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \      # 参考策略模型精度
    "$@"
