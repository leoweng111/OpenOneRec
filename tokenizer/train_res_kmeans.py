"""训练残差 K-Means（Residual K-Means）分词器。

输入：一个/多个 parquet 文件，每行含 `embedding` 列（长度 >= dim 的浮点数组），
    通常是 Qwen3-Embedding-8B 对所有 item 打出的语义 embedding（dim=4096）。
输出：`{model_path}/model.pt`，可用 `infer_res_kmeans.py` 加载并把 item embedding
    映射成一串 Itemic Token（形如 `<s_a_340><s_b_6566><s_c_5603>`）。
上游依赖：faiss（GPU K-Means），需另行安装。
下游用途：产物 codes 会被灌进 `pretrain/onerec_llm` 的词表扩展脚本
    (`tools/model_converter/expand_qwen3_vocab.py`)，为 LLM 增加 `n_layers * codebook_size`
    个 Itemic Token。
"""

import os
import argparse
import random
import numpy as np
import torch
import pyarrow.parquet as pq
from tqdm import tqdm
from res_kmeans import ResKmeans


def read_train_data(path, emb_dim):
    """从本地 parquet（可以是文件或目录）读取 embedding 矩阵。

    Returns:
        result: numpy.ndarray, shape=(N, emb_dim), dtype=float32
            N = 所有 fragment 的行数之和。fragments 顺序被打乱，避免同一类 item 集中。
    """
    dataset = pq.ParquetDataset(path)

    fragments = list(dataset.fragments)
    random.shuffle(fragments)
    print(f"Total files: {len(fragments)}")

    embeddings = []
    current_size = 0

    for fragment in tqdm(fragments, desc="Reading files"):
        table = fragment.to_table(columns=['embedding'])
        if table.num_rows == 0:
            continue

        # parquet 里 embedding 可能被存成 list<float>（object 数组）或原生 numpy，需要统一
        emb_chunk = table['embedding'].to_numpy(zero_copy_only=False)
        if emb_chunk.dtype == 'object':
            emb_chunk = np.vstack(emb_chunk)  # (rows, D)

        emb_chunk = emb_chunk[:, :emb_dim].astype(np.float32)  # 只取前 emb_dim 维
        embeddings.append(emb_chunk)
        current_size += len(emb_chunk)

    result = np.concatenate(embeddings, axis=0)  # (N, emb_dim)
    print(f"Final shape: {result.shape}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Train ResKmeans')
    parser.add_argument('--data_path', type=str, required=True, help='training data path')
    parser.add_argument('--model_path', type=str, required=True, help='model save path')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--codebook_size', type=int, default=8192, help='codebook size')
    parser.add_argument('--dim', type=int, default=4096, help='embedding dimension')
    parser.add_argument('--niter', type=int, default=20, help='kmeans iterations')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1) 读入全部 item embedding：(N, dim)
    embeddings = read_train_data(args.data_path, args.dim)

    # 2) 构建 & 训练残差 K-Means（逐层 faiss.Kmeans）
    model = ResKmeans(
        n_layers=args.n_layers,
        codebook_size=args.codebook_size,
        dim=args.dim,
    )
    model.train_kmeans(torch.tensor(embeddings))

    # 3) 保存 state_dict：包含 `centroids.0 / centroids.1 / ...`，每个 (codebook_size, dim)
    os.makedirs(args.model_path, exist_ok=True)
    save_path = os.path.join(args.model_path, "model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    main()
