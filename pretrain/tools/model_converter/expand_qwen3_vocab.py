"""Qwen3 Vocabulary Expansion Tool

Expand the standard Qwen3 HuggingFace checkpoint vocabulary to support post-training.
Add new tokens and adjust model vocabulary size (aligned to multiples of 256).

几乎所有主流 LLM 都有词表（vocabulary）。
可以把词表理解成两层关系：
字符串 token <-> token id（整数）
例如 "hello" -> 12345，"<s_a_340>" -> 151900
token id <-> embedding 向量
E[id] 是输入嵌入（以及常见的输出头权重共享/关联）
所以词表本质是“离散符号系统 + 索引入口”。
LLM 的训练与推理都是围绕“预测下一个 token id”展开的。



OneRec并没有从零训练一个全新的Transformer，而是以Qwen3（预训练LLM）作为骨干模型，在其基础上扩展而来。

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

为什么需要Qwen3的参数？

- OneRec的"Lazy Decoder-Only"本质上就是一个GPT-style的Decoder-Only Transformer — 这和Qwen3的架构完全一致
- 从零训练一个数十亿参数的Transformer成本极高，而Qwen3已经预训练好了，具备强大的序列建模能力和世界知识
- 做法是：加载Qwen3权重 → 扩展词表（加入SID tokens）→ 继续预训练（CPT）→ 任务微调（SFT）→ RL对齐

这和PLUM的思路类似：复用预训练LLM做推荐，而不是从头造一个。

词表（Vocabulary）是什么？

是的，所有LLM都有词表。词表本质上就是一个映射关系：

词表 (Vocabulary) = 一个双向映射

  "你好"    ↔  token_id = 15167
  "hello"   ↔  token_id = 8342
  "的"      ↔  token_id = 1916
  "<sid_begin>" ↔ token_id = 151669   ← OneRec新增的特殊token
  "<s_0_42>"    ↔ token_id = 151670   ← OneRec新增的SID token
  ...

  词表大小 V = 176,000

为什么LLM需要词表？

因为LLM的核心机制是 Next Token Prediction：在每一步，模型输出一个 $V$ 维的logits向量，表示"下一个token是词表中每个token的概率"。

Transformer最后一层输出 h ∈ R^d
        ↓
LM Head: W ∈ R^{V × d}
        ↓
logits = W · h ∈ R^V          # V=176,000维
        ↓
probs = softmax(logits)       # 每个token的概率
        ↓
loss = -log(probs[true_token_id])   # 交叉熵

模型的输入和输出都是token_id（整数），不是原始向量。 这就是LLM的工作方式。

为什么要把SID code映射到词表token？

这是你最核心的疑问。让我对比两种做法：

做法A：传统量化用法（QARM等DLRM的做法）

物品 → RQ-KMeans → 语义ID = (c₁=42, c₂=15, c₃=89)
                          ↓
              Embedding Table (可学习, 独立于推荐模型)
              emb_table_1[42] → e₁ ∈ R^d
              emb_table_2[15] → e₂ ∈ R^d
              emb_table_3[89] → e₃ ∈ R^d
                          ↓
              concat(e₁, e₂, e₃) → DNN → CTR预估

这里语义ID就是普通的离散特征，通过embedding table查表后喂给DNN。完全可以用RQ-KMeans的聚类中心作为初始embedding。

做法B：LLM生成式用法（OneRec/PLUM/TIGER的做法）

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

为什么必须映射到词表？

因为OneRec是一个LLM，它的：
- 输入：必须是词表中的token_id
- 输出：是词表大小的概率分布（$V$维softmax）
- 训练目标：Next Token Prediction，即从$V$个候选中选出正确的下一个token

如果不映射到词表，LLM根本无法"看到"也无法"生成"这些语义ID。

为什么OneRec选择做法B而不是做法A？

┌───────────────┬────────────────────┬──────────────────────┐
│     维度      │    做法A (DLRM)    │  做法B (LLM/OneRec)  │
├───────────────┼────────────────────┼──────────────────────┤
│ 模型能力      │ 只能做判别（打分） │ 能生成（自回归预测） │
├───────────────┼────────────────────┼──────────────────────┤
│ 预训练知识    │ 无                 │ 继承LLM的世界知识    │
├───────────────┼────────────────────┼──────────────────────┤
│ 统一检索+排序 │ ❌ 需要多阶段      │ ✅ 生成即推荐        │
├───────────────┼────────────────────┼──────────────────────┤
│ 冷启动        │ 依赖ID embedding   │ 语义泛化             │
├───────────────┼────────────────────┼──────────────────────┤
│ Scaling Law   │ 不明显             │ 遵循power-law        │
├───────────────┼────────────────────┼──────────────────────┤
│ 推理方式      │ 全库打分 O(N)      │ Beam Search O(L×K)   │
└───────────────┴────────────────────┴──────────────────────┘

核心区别：

- 做法A中，语义ID是输入特征，推荐模型是判别式的（给每个候选打分）
- 做法B中，语义ID是LLM词汇的一部分，推荐模型是生成式的（自回归生成目标物品的SID）

一个直觉类比

把OneRec想象成一个"会说推荐语言的ChatGPT"：

普通LLM:   "今天天气" → 预测下一个词 → "很好"
OneRec:    "用户历史[A][B]" → 预测下一个"词" → [C] (一个SID token序列)

这里的[A]、[B]、[C]都是词表中的token，就像"天气"、"很好"是词表中的token一样。LLM不区分"自然语言token"和"物品SID token"——对它来说都是词表里的整数，都需要预测下一个。

这就是为什么OneRec需要：
1. Qwen3的Transformer架构和预训练权重（作为"大脑"）
2. 扩展词表以包含SID tokens（让"大脑"能理解和生成"物品语言"）
3. LM Head扩展到$V≈176k$维（让输出层能预测SID tokens）

核心是为了利用Qwen3原本的参数（而非冷启训练一个LLM），所以才必须扩展Qwen3原本的词表。


#=========================================================#
训练和推理时词表的作用？
#=========================================================#


训练时：在全部 $V≈176k$ 个token上算loss

logits = LM_Head(h) ∈ R^{176000}    # 全词表的logits

# 即使当前位置的正确答案是SID token (比如 token_id=151712)
# loss仍然在全部176k个token上算softmax:
loss = -log( softmax(logits)[151712] )

# 这意味着模型需要学会:
# 1. 给正确的SID token高概率
# 2. 给所有文本token低概率  ← 这也很重要！
# 3. 给其他SID token低概率

为什么训练时不只看SID部分？ 因为全词表softmax起到一种正则化作用 — 模型需要学会"当我要生成物品时，不要生成文本"。这帮助模型区分"什么时候该生成文本"和"什么时候该生成SID"。

推理时：只关注SID token子集

当模型已经生成了 <|sid_begin|>，接下来要生成SID的3个code token时：

logits = LM_Head(h) ∈ R^{176000}

# 只看SID token范围的logits:
sid_logits = logits[151670 : 151670+8192]   # 只看第1层codebook的8192个token

# 在这些候选中做beam search或argmax:
next_code = argmax(sid_logits)              # 从8192个候选中选

# 而不是从176000个token中选（那会把文本token也选进来）

这实际上就是代码里的做法。在OneRec的推理代码（vllm_rollout）中，beam search时会限制生成的token范围只在SID token索引内。

一个完整例子

推理过程:

Step 1: 模型看到 "<|sid_begin|>"，知道接下来要生成物品的SID
        → 限制输出范围为 SID Layer 0 tokens [151670, 151670+8192)
        → 从8192个候选中 beam search → 选出 c₁=42 (token_id=151712)

Step 2: 已知c₁=42，预测第2层code
        → 限制输出范围为 SID Layer 1 tokens [159862, 159862+8192)
        → 从8192个候选中选 → 选出 c₂=15 (token_id=159877)

Step 3: 已知c₁=42, c₂=15，预测第3层code
        → 限制输出范围为 SID Layer 2 tokens [168054, 168054+8192)
        → 选出 c₃=89 (token_id=168143)

Step 4: 生成 "<|sid_end|>"，SID生成完毕
        → 映射回 (42, 15, 89) → 查找对应的物品 → 推荐给用户


"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def _align_vocab_size(vocab_size: int, alignment: int = 256) -> int:
    """Align vocabulary size to the nearest multiple of alignment.
    
    Args:
        vocab_size: Current vocabulary size
        alignment: Alignment value (default: 256)
        
    Returns:
        Aligned vocabulary size
    """
    return ((vocab_size + alignment - 1) // alignment) * alignment


def _fix_chat_template(reco_model_dir: str, hf_model_dir: str) -> None:
    """Fix chat template in tokenizer config by copying from original model.
    
    Args:
        reco_model_dir: Output model directory
        hf_model_dir: Original HuggingFace model directory
    """
    reco_tokenizer_config_path = os.path.join(reco_model_dir, "tokenizer_config.json")
    hf_tokenizer_config_path = os.path.join(hf_model_dir, "tokenizer_config.json")
    
    if not os.path.exists(hf_tokenizer_config_path):
        logger.warning(f"Original tokenizer_config.json not found: {hf_tokenizer_config_path}")
        return
    
    if not os.path.exists(reco_tokenizer_config_path):
        logger.warning(f"Output tokenizer_config.json not found: {reco_tokenizer_config_path}")
        return
    
    # Load configs
    with open(reco_tokenizer_config_path, "r", encoding="utf-8") as f:
        reco_config = json.load(f)
    
    with open(hf_tokenizer_config_path, "r", encoding="utf-8") as f:
        hf_config = json.load(f)
    
    # Copy chat template from original
    if "chat_template" in hf_config:
        reco_config["chat_template"] = hf_config["chat_template"]
        
        with open(reco_tokenizer_config_path, "w", encoding="utf-8") as f:
            json.dump(reco_config, f, indent=2, ensure_ascii=False)
        
        logger.info("Chat template copied from original model")


def _test_expanded_vocab(model, tokenizer, new_tokens: List[str]) -> None:
    """Test the expanded vocabulary with sample tokens.
    
    Args:
        model: Expanded model
        tokenizer: Expanded tokenizer
        new_tokens: List of newly added tokens
    """
    if not new_tokens:
        logger.info("No new tokens to test")
        return
    
    # Sample 3-5 tokens from new_tokens
    num_samples = min(random.randint(3, 5), len(new_tokens))
    sampled_tokens = random.sample(new_tokens, num_samples)
    input_text = " ".join(sampled_tokens) + " Hello world"
    
    try:
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        
        # Test generation (use eval mode to avoid training-specific behavior)
        model.eval()
        with torch.no_grad():
            output = model.generate(input_ids, max_new_tokens=10, do_sample=False)
        
        logger.info("Vocabulary expansion test:")
        logger.info(f"  Input text: {input_text}")
        logger.info(f"  Decoded input: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
        logger.info(f"  Input IDs shape: {input_ids.shape}")
        logger.info(f"  Generated: {tokenizer.decode(output[0], skip_special_tokens=True)}")
        
    except Exception as e:
        logger.warning(f"Vocabulary test failed: {e}")


def expand_qwen3_vocab_for_pretraining(
    hf_model_dir: str,
    output_model_dir: str,
    new_tokens: List[str]
) -> None:
    """Expand Qwen3 vocabulary for pretraining by adding new tokens.
    
    This function:
    1. Loads the original Qwen3 model and tokenizer
    2. Adds new tokens to the tokenizer
    3. Resizes model embeddings to aligned vocabulary size (multiple of 256)
    4. Updates model configuration
    5. Saves the expanded model, tokenizer, and config
    6. Fixes chat template from original model
    7. Tests the expanded vocabulary
    
    Args:
        hf_model_dir: Path to original HuggingFace model directory
        output_model_dir: Path to save expanded model
        new_tokens: List of new tokens to add
        
    Raises:
        FileNotFoundError: If model directory doesn't exist
        ValueError: If new_tokens is empty
    """
    if not new_tokens:
        raise ValueError("new_tokens list cannot be empty")
    
    if not os.path.exists(hf_model_dir):
        raise FileNotFoundError(f"Model directory does not exist: {hf_model_dir}")
    
    # Create output directory
    os.makedirs(output_model_dir, exist_ok=True)
    logger.info(f"Expanding vocabulary for pretraining")
    logger.info(f"  Input model: {hf_model_dir}")
    logger.info(f"  Output model: {output_model_dir}")
    logger.info(f"  New tokens: {len(new_tokens)}")
    
    # Step 1: Load original model components
    logger.info("Loading original model components...")
    config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_model_dir,
        torch_dtype=torch.float32,  # Use float32 for compatibility
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, trust_remote_code=True)
    
    original_vocab_size = len(tokenizer)
    logger.info(f"Original vocabulary size: {original_vocab_size}")
    
    # Step 2: Add new tokens
    logger.info(f"Adding {len(new_tokens)} new tokens...")
    num_added = tokenizer.add_tokens(new_tokens)
    logger.info(f"Successfully added {num_added} tokens")
    
    # Step 3: Calculate aligned vocabulary size
    new_vocab_size = len(tokenizer)
    target_vocab_size = _align_vocab_size(new_vocab_size, alignment=256)
    logger.info(f"New vocabulary size: {new_vocab_size}")
    logger.info(f"Target vocabulary size (aligned to 256): {target_vocab_size}")
    
    # Step 4: Resize model embeddings
    logger.info("Resizing model token embeddings...")
    model.resize_token_embeddings(target_vocab_size)
    
    # Step 5: Update configuration
    config.vocab_size = target_vocab_size
    logger.info(f"Updated config vocab_size to {target_vocab_size}")
    
    # Step 6: Save expanded components
    logger.info("Saving expanded model components...")
    tokenizer.save_pretrained(output_model_dir)
    model.save_pretrained(output_model_dir)
    config.save_pretrained(output_model_dir)
    logger.info("Model components saved successfully")
    
    # Step 7: Fix chat template
    logger.info("Fixing chat template...")
    _fix_chat_template(output_model_dir, hf_model_dir)
    
    # Step 8: Test expanded vocabulary
    logger.info("Testing expanded vocabulary...")
    _test_expanded_vocab(model, tokenizer, new_tokens)
    
    logger.info(f"✓ Vocabulary expansion completed! Final vocab size: {target_vocab_size}")


def generate_itemic_tokens(itemic_layer_n: int, vocab_size_per_layer: int) -> List[str]:
    """Generate itemic special tokens dynamically.
    
    IMPORTANT: Token order must strictly match gen_itemic_sp_tokens.py:
    1. All <s_a_{i}> tokens (i from 0 to vocab_size_per_layer-1)
    2. All <s_b_{i}> tokens (i from 0 to vocab_size_per_layer-1)
    3. All <s_c_{i}> tokens (i from 0 to vocab_size_per_layer-1)
    4. ... (for itemic_layer_n layers, in alphabetical order)
    5. <|sid_begin|>
    6. <|sid_end|>
    
    Args:
        itemic_layer_n: Number of itemic layers (determines s_a, s_b, s_c, ...)，残差量化层数，对应 token 前缀层数。
        vocab_size_per_layer: Vocabulary size per layer (determines range of i)，每层 codebook 的大小（每层有多少码向量）。
        
    Returns:
        List of generated tokens in strict order
        内容是新增 token 字符串，顺序严格如下：
        全部 <s_a_0> ... <s_a_{K-1}>
        全部 <s_b_0> ... <s_b_{K-1}>
        全部 <s_c_0> ...
        ...
        最后追加 <|sid_begin|>
        最后追加 <|sid_end|>
        其中 K = vocab_size_per_layer。
        总数量是：itemic_layer_n * vocab_size_per_layer + 2。
        
    Raises:
        ValueError: If itemic_layer_n or vocab_size_per_layer is invalid
    """
    if itemic_layer_n <= 0:
        raise ValueError(f"itemic_layer_n must be positive, got {itemic_layer_n}")
    if vocab_size_per_layer <= 0:
        raise ValueError(f"vocab_size_per_layer must be positive, got {vocab_size_per_layer}")
    
    # Generate layer names in alphabetical order: a, b, c, d, ...
    # This ensures the same order as gen_itemic_sp_tokens.py
    layer_names = [chr(ord('a') + i) for i in range(itemic_layer_n)]
    
    new_tokens = []
    
    # Generate tokens in strict order:
    # For each layer (a, b, c, ...), generate all tokens with i from 0 to vocab_size_per_layer-1
    # This matches the order: [*s_a_0..8191, *s_b_0..8191, *s_c_0..8191, ...]
    for layer_name in layer_names:
        for i in range(vocab_size_per_layer):
            new_tokens.append(f"<s_{layer_name}_{i}>")
    
    # Add special tokens at the end (must be in this exact order)
    new_tokens.append('<|sid_begin|>')
    new_tokens.append('<|sid_end|>')
    
    total_tokens = itemic_layer_n * vocab_size_per_layer + 2
    logger.info(f"Generated {total_tokens} itemic tokens in strict order:")
    logger.info(f"  Layers: {itemic_layer_n} ({', '.join([f's_{name}' for name in layer_names])})")
    logger.info(f"  Vocab size per layer: {vocab_size_per_layer}")
    logger.info(f"  Special tokens: <|sid_begin|>, <|sid_end|>")
    
    return new_tokens


def load_tokens_from_file(tokens_file: str) -> List[str]:
    """Load tokens from a text file (one token per line).
    
    Args:
        tokens_file: Path to text file containing tokens (one per line)
        
    Returns:
        List of tokens (empty lines are skipped)
        
    Raises:
        FileNotFoundError: If tokens file doesn't exist
    """
    if not os.path.exists(tokens_file):
        raise FileNotFoundError(f"Tokens file does not exist: {tokens_file}")
    
    new_tokens = []
    line_count = 0
    
    with open(tokens_file, "r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            token = line.strip()
            if token:  # Skip empty lines
                new_tokens.append(token)
    
    logger.info(f"Loaded {len(new_tokens)} tokens from {line_count} lines in {tokens_file}")
    return new_tokens


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Expand Qwen3 vocabulary for pretraining by adding new tokens. '
                    'Supports two modes: loading from file or generating itemic tokens dynamically.'
    )
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        required=True,
        help="Path to original HuggingFace Qwen3 model directory"
    )
    parser.add_argument(
        "--output_model_dir",
        type=str,
        required=True,
        help="Path to save expanded model directory"
    )
    
    # Itemic token generation parameters
    parser.add_argument(
        "--itemic_layer_n",
        type=int,
        required=True,
        help="Number of itemic layers (e.g., 3 for s_a, s_b, s_c)"
    )
    parser.add_argument(
        "--vocab_size_per_layer",
        type=int,
        required=True,
        help="Vocabulary size per layer (e.g., 8192 for tokens from 0 to 8191)"
    )
    
    args = parser.parse_args()
    
    try:
        # Generate itemic tokens dynamically
        logger.info("Generating itemic tokens dynamically...")
        new_tokens = generate_itemic_tokens(
            itemic_layer_n=args.itemic_layer_n,
            vocab_size_per_layer=args.vocab_size_per_layer
        )
        
        if not new_tokens:
            logger.error("No tokens to add")
            sys.exit(1)
        
        # Expand vocabulary
        expand_qwen3_vocab_for_pretraining(
            hf_model_dir=args.hf_model_dir,
            output_model_dir=args.output_model_dir,
            new_tokens=new_tokens
        )
        
        logger.info("All operations completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Program execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
