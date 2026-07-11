"""残差 K-Means 分词器推理脚本。

功能：加载 `train_res_kmeans.py` 训练出的 codebook，对一个 item embedding 表做量化，
    输出每个 pid 对应的一串离散 code（长度 = n_layers）。这些 code 与词表偏移相加后
    就是 LLM 词表里的 Itemic Token id。
输入 parquet：需含列 `pid`（item id）和 `embedding`（长度 dim 的向量）。
输出 parquet：列 `pid` + `codes`（长度 n_layers 的 int 列表）。
"""

import argparse
import torch
import numpy as np
import pandas as pd
from res_kmeans import ResKmeans


def load_embeddings(emb_path):
    """读取 (pid, embedding) 表。

    Returns:
        pids: list[任意]，长度 N
        emb:  torch.FloatTensor，shape=(N, dim)
    """
    df = pd.read_parquet(emb_path)
    pids = df['pid'].tolist()
    emb = torch.tensor(np.stack(df['embedding'].values), dtype=torch.float32)
    return pids, emb


def main():
    parser = argparse.ArgumentParser(description='ResKmeans Inference')
    parser.add_argument('--model_path', type=str, required=True, help='model checkpoint path')
    parser.add_argument('--emb_path', type=str, required=True, help='embedding file path')
    parser.add_argument('--output_path', type=str, default=None, help='output path (default: emb_path + _codes.parquet)')
    parser.add_argument('--batch_size', type=int, default=10000, help='inference batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--n_layers', type=int, default=None, help='number of layers to use (default: all layers)')
    args = parser.parse_args()

    # -------- 1) 加载模型 --------
    # 兼容两种 checkpoint：整个 ResKmeans 对象 / 只有 state_dict（train 脚本保存的形式）
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cpu')  # 这个读取进来的就是train_res_kmeans.py训练得到的ResKmeans对象

    if isinstance(checkpoint, ResKmeans):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 从 state_dict 的 key 反推超参：centroids.0, centroids.1, ...
        # 对应模型的self.centroids属性
        n_layers = sum(1 for k in state_dict.keys() if k.startswith('centroids.'))
        first_centroid = state_dict['centroids.0']       # 第0层的聚类中心，形状是(codebook_size, dim)，表示有codebook_size个聚类中心，每个都是dim维度的向量
        codebook_size, dim = first_centroid.shape

        model = ResKmeans(n_layers=n_layers, codebook_size=codebook_size, dim=dim)
        model.load_state_dict(state_dict)
    else:
        raise ValueError("Unknown checkpoint format")

    model = model.to(args.device)
    model.eval()
    print(f"Model loaded: n_layers={model.n_layers}, codebook_size={model.codebook_size}, dim={model.dim}")

    # -------- 2) 加载 item embedding：(N, dim) --------
    print(f"Loading embeddings from {args.emb_path}")
    pids, emb = load_embeddings(args.emb_path)
    print(f"Embeddings shape: {emb.shape}, num pids: {len(pids)}")

    # -------- 3) 分 batch 编码 --------
    print("Encoding...")
    all_codes = []
    with torch.no_grad():
        for i in range(0, len(emb), args.batch_size):
            batch = emb[i:i + args.batch_size].to(args.device)      # (b, dim)
            codes = model.encode(batch, n_layers=args.n_layers)      # (b, n_layers)
            all_codes.append(codes.cpu())
            if (i // args.batch_size) % 10 == 0:
                print(f"  Processed {min(i + args.batch_size, len(emb))}/{len(emb)}")

    all_codes = torch.cat(all_codes, dim=0)   # (N, n_layers)
    print(f"Output codes shape: {all_codes.shape}")

    # -------- 4) 保存为 parquet（每行一个 pid + 对应 codes 列表） --------
    output_path = args.output_path or args.emb_path.rsplit('.', 1)[0] + '_codes.parquet'
    df_out = pd.DataFrame({
        'pid': pids,
        'codes': all_codes.numpy().tolist()
    })
    df_out.to_parquet(output_path, index=False)
    print(f"Codes saved to {output_path}")

    # -------- 5) 抽样估计重建误差（sanity check） --------
    print("\nComputing reconstruction loss...")
    with torch.no_grad():
        sample_size = min(10000, len(emb))
        sample_emb = emb[:sample_size].to(args.device)                  # (S, dim)
        sample_codes = all_codes[:sample_size].to(args.device)          # (S, n_layers)
        reconstructed = model.decode(sample_codes)                       # (S, dim)
        loss_info = model.calc_loss(sample_emb, reconstructed)
        print(f"Reconstruction loss (MSE): {loss_info['loss']:.6f}")
        print(f"Relative loss: {loss_info['rel_loss']:.6f}")


if __name__ == '__main__':
    main()
