import torch
from torch import nn


class DEMA(nn.Module):
    """
    Double Exponential Moving Average (DEMA) block to highlight the trend of time series
    (双指数移动平均(DEMA)模块，用于突出时间序列的趋势)

    DEMA, 也称为Holt线性趋势法, 是一种比EMA更复杂的平滑技术，因为它同时考虑了序列的水平（level）和趋势（trend）。
    TODO: 这不是论文中使用的主要分解方法，而是作为一种可选项提供的。
    """

    def __init__(self, alpha, beta):
        """
        构造函数。

        Args:
            alpha (float): 水平分量的平滑因子。
            beta (float): 趋势分量的平滑因子。
        """
        super(DEMA, self).__init__()
        # self.alpha = nn.Parameter(alpha)    # Learnable alpha (可学习的alpha，当前被注释)
        # self.beta = nn.Parameter(beta)      # Learnable beta (可学习的beta，当前被注释)
        # 将固定的alpha和beta因子移动到CUDA设备上
        self.alpha = alpha.to(device='cuda')
        self.beta = beta.to(device='cuda')

    def forward(self, x):
        """
        前向传播函数。

        这是一个迭代实现，时间复杂度为 O(L)，其中L是序列长度。
        它遵循Holt线性趋势法的更新规则:
        Level: s_t = alpha * x_t + (1 - alpha) * (s_{t-1} + b_{t-1})
        Trend: b_t = beta * (s_t - s_{t-1}) + (1 - beta) * b_{t-1}

        Args:
            x (torch.Tensor): 输入的时间序列张量。
                               维度: (B, L, D)
                               B: Batch size, L: Sequence length, D: Feature dimension

        Returns:
            torch.Tensor: DEMA平滑后的时间序列，代表趋势。
                           维度: (B, L, D)
        """
        # self.alpha.data.clamp_(0, 1)        # Clamp learnable alpha to [0, 1] (若alpha可学习，则限制范围)
        # self.beta.data.clamp_(0, 1)         # Clamp learnable beta to [0, 1] (若beta可学习，则限制范围)

        # --- 初始化 ---
        # 初始化第一个时间步的水平分量 s_0 = x_0
        # # x[:, 0, :] -> (B, D)
        s_prev = x[:, 0, :]
        # 初始化第一个时间步的趋势分量 b_0 = x_1 - x_0
        # # b -> (B, D)
        b = x[:, 1, :] - s_prev
        # 将第一个时间步的结果存入列表
        # # s_prev.unsqueeze(1) -> (B, 1, D)
        res = [s_prev.unsqueeze(1)]

        # --- 迭代计算 ---
        # 从第二个时间步开始循环 (t=1 to L-1)
        for t in range(1, x.shape[1]):
            # 获取当前时间步的数据 xt
            # # xt -> (B, D)
            xt = x[:, t, :]
            # 更新水平分量 s_t
            # # s -> (B, D)
            s = self.alpha * xt + (1 - self.alpha) * (s_prev + b)
            # 更新趋势分量 b_t
            # # b -> (B, D)
            b = self.beta * (s - s_prev) + (1 - self.beta) * b
            # 更新 s_prev 以便下一次迭代使用
            s_prev = s
            # 将当前步的结果存入列表
            # # s.unsqueeze(1) -> (B, 1, D)
            res.append(s.unsqueeze(1))

        # --- 结果拼接 ---
        # 将列表中所有时间步的结果在时间维度(dim=1)上拼接起来
        # # list of L tensors with shape (B, 1, D) -> (B, L, D)
        return torch.cat(res, dim=1)
