import torch
from torch import nn


class ResKmeans(nn.Module):
    """残差 K-Means 分层量化器（Residual K-Means Tokenizer）。

    作用：把每个 item 的连续 embedding（形状 (dim,)，一般是 Qwen3-Embedding-8B 的 4096 维输出）
    编码成一串离散 code：`(c_0, c_1, ..., c_{n_layers-1})`，每个 c_l ∈ [0, codebook_size)。
    这里codebook_size就是每个codebook中的码向量个数。
    这串 code 就是 OneRec 中的 "Itemic Token"（视频/商品/广告等 item 的语义 ID），会被拼接进
    LLM 的 vocabulary，从而让 LLM 用生成 token 的方式来表征 & 生成 item。

    核心思想（RQ / RVQ 思路）：
        - 第 0 层：对原始向量 x 做 K-Means，得到最近质心 c_0，残差 r_0 = x - centroid[c_0]。
        - 第 1 层：对 r_0 再做 K-Means，得到 c_1，残差 r_1 = r_0 - centroid[c_1]。
        - ... 依次类推共 n_layers 层。
        - 解码时把 n_layers 个质心相加就能重建 x。
    与普通 K-Means 相比，残差量化可以用 n_layers 个大小为 codebook_size 的 codebook 表达出
    `codebook_size ^ n_layers` 数量级的组合，从而以很小的显存代价覆盖大 item 集合。

    典型配置（README 默认）：n_layers=3, codebook_size=8192, dim=4096 →
        - 每个 item 的表示：3 个 int（形状 (3,)），可能组合 8192^3 ≈ 5.5e11。
        - 模型参数：3 × 8192 × 4096 float32 ≈ 400 MB。
    """

    def __init__(self, n_layers, codebook_size, dim, extra_kmeans_config=None, **kwargs):
        super().__init__()
        self.n_layers = n_layers          # 残差层数（一般 3）
        self.codebook_size = codebook_size  # 每层 codebook 大小（一般 8192）
        self.dim = dim                    # embedding 维度（一般 4096）
        self.extra_kmeans_config = extra_kmeans_config  # 透传给 faiss.Kmeans 的额外参数
        # 每层一个 codebook，形状 (codebook_size, dim)，不参与反向传播（K-Means 训练）
        self.centroids = nn.ParameterList([
            nn.Parameter(torch.zeros((codebook_size, dim), requires_grad=False))
            for i in range(n_layers)
        ])

    def calc_loss(self, x, out, epsilon=1e-4):
        """计算重建误差（供训练时打印进度用）。

        Args:
            x:   原始向量，形状 (N, dim)
            out: 重建向量（各层质心之和），形状 (N, dim)
        Returns:
            {'loss': MSE, 'rel_loss': 逐元素相对误差均值}
        """
        loss = ((out - x) ** 2).mean()
        rel_loss = (torch.abs(x - out) / (torch.maximum(torch.abs(x), torch.abs(out)) + epsilon)).mean()
        return {'loss': loss.item(), 'rel_loss': rel_loss.item()}

    def train_kmeans(self, inputs, verbose=True):
        """逐层训练残差 K-Means（使用 faiss.Kmeans）。

        Args:
            inputs: 训练数据，形状 (N, dim)，N 为 item 数（通常几十万到千万级）。
        流程（第 l 层）：
            1. faiss.Kmeans 在残差 x（初始为 inputs）上聚类 → 得到 codebook_size 个质心。
            2. 用质心把 x 中每个样本映射到最近质心 I（形状 (N,)）。
            3. 累加当前层的质心到 out，用于估计整体重建误差。
            4. 更新残差 x ← x - centroids[I]，作为下一层输入。
        训练完成后 `self.centroids[l]` 保存每一层的 codebook。
        """
        import faiss
        kmeans = faiss.Kmeans(self.dim, self.codebook_size, spherical=False, **self.extra_kmeans_config)
        x = inputs.clone()                # (N, dim)：本层要聚类的残差
        out = torch.zeros_like(x)         # (N, dim)：累积重建结果，仅用于打印 loss
        for l in range(self.n_layers):
            kmeans.train(x)                          # faiss 内部迭代 niter 次
            _, I = kmeans.index.search(x, 1)         # I: (N, 1)  最近质心索引。对于N个item，每个item都有一个最近质心的索引，所以总共有N个。
            I = I.reshape([-1])                      # (N,)
            # kmeans.centroids 形状是 (codebook_size, dim)，代表了codebook_size个质心。
            o = torch.tensor(kmeans.centroids[I])    # (N, dim)  本层选中的质心向量
            out += o                                 # 累计 → 越来越接近 inputs
            if verbose:
                losses = self.calc_loss(inputs, out)  # RQ-VAE训练的时候其实完全不需要做反向传播梯度下降，losses仅供打印查看
                print(l, losses)
            x = x - o                                # (N, dim)  更新残差 → 下一层输入
            # self.centroids[l]保存的是每层的聚类中心，总共有codebook_size个。
            # 注意：在RQ-Kmeans中，每层的centroids是固定的，不参与反向传播，所以requires_grad=False。
            # 另外，每层的聚类中心，就是每层的码向量，每层的聚类中心个数=码向量个数
            self.centroids[l] = nn.Parameter(
                torch.tensor(kmeans.centroids.copy()), requires_grad=False
            )  # (codebook_size, dim)
            print(f"layer {l} finished")

    def encode(self, x, n_layers=None):
        """把连续向量编码成离散 code（推理阶段用）。

        Args:
            x:        输入向量，形状 (B, dim)，B 为 batch 大小。
            n_layers: 使用前 n_layers 层做编码；None 表示用全部层。
        Returns:
            out: 形状 (B, n_layers)，dtype int64，每列取值 ∈ [0, codebook_size)。代表每层与该层残差距离最近质心的索引。
        算法：与 train_kmeans 中"最近质心 + 残差"完全一致，但是用矩阵运算做距离计算，避免依赖 faiss。
            distances[i, k] = ||x_i - centroids[k]||^2
                            = ||x_i||^2 + ||centroids[k]||^2 - 2 · x_i · centroids[k]
        """
        if n_layers is None:
            n_layers = self.n_layers
        else:
            assert n_layers <= self.n_layers
        out = []
        for l in range(n_layers):
            # x: (B, dim)；centroids[l]: (K, dim)，K=codebook_size
            x_norm_sq = x.pow(2.).sum(dim=1, keepdim=True)                      # (B, 1)
            codebook_t_norm_sq = self.centroids[l].T.pow(2.).sum(dim=0, keepdim=True)  # (1, K)
            # distances = x_norm_sq + codebook_norm_sq - 2 * x @ centroids^T → (B, K)，表示B个样本，每个样本距离K个质心的距离
            distances = torch.addmm(x_norm_sq + codebook_t_norm_sq,
                                    x, self.centroids[l].T, alpha=-2.0)
            code = distances.argmin(dim=-1)          # (B,)  本层选中的 code，即对于每个样本而言，距离其最近的聚类中心（质心）的索引
            x = x - self.centroids[l][code]          # (B, dim)  更新残差
            out.append(code)
        out = torch.stack(out, dim=1)                # (B, n_layers)
        return out

    def decode(self, code):
        """把离散 code 反解回连续向量（用于评估重建误差）。

        Args:
            code: 形状 (B, n_layers)，每列取值 ∈ [0, codebook_size)。
        Returns:
            out:  形状 (B, dim)，各层选中质心相加。
        """
        out = torch.zeros((code.shape[0], self.dim), dtype=torch.float32, device=code.device)
        n_layers = code.shape[1]
        assert n_layers <= self.n_layers
        for l in range(n_layers):
            c = code[:, l]                           # (B,)  本层 code
            out += self.centroids[l][c]              # (B, dim)  累加质心向量
        return out
