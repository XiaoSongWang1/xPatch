import torch
from torch import nn


class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    (指数移动平均(EMA)模块，用于突出时间序列的趋势)

    该模块对应论文3.1节中提出的基于EMA的分解方法。
    """

    def __init__(self, alpha):
        """
        构造函数。

        Args:
            alpha (float): 指数移动平均的平滑因子。
        """
        super(EMA, self).__init__()
        # self.alpha = nn.Parameter(alpha)    # Learnable alpha (可学习的alpha，当前版本被注释，未使用)
        # 使用固定的alpha值
        self.alpha = alpha

    # Optimized implementation with O(1) time complexity (时间复杂度为O(1)的优化实现)
    def forward(self, x):
        """
        前向传播函数。

        这是一个并行化的EMA实现，利用累积和（cumsum）来避免循环，从而获得高效的计算性能。
        它在数学上等价于标准的递归EMA公式: s_t = alpha * x_t + (1 - alpha) * s_{t-1}，其中 s_0 = x_0。
        该优化方法对应论文附录D (Appendix D) 中提到的将EMA优化到O(1)时间复杂度的实现。

        Args:
            x (torch.Tensor): 输入的时间序列张量。
                               维度: [Batch, Input, Channel] -> (B, L, D)
                               B: Batch size, L: Sequence length, D: Feature dimension

        Returns:
            torch.Tensor: EMA平滑后的时间序列，代表趋势部分。
                           维度: (B, L, D)
        """
        # x: [Batch, Input, Channel]
        # self.alpha.data.clamp_(0, 1)        # Clamp learnable alpha to [0, 1] (如果alpha是可学习的，需要将其限制在[0, 1]范围内)

        # 获取输入的维度
        _, t, _ = x.shape  # L

        # 1. 计算权重和除数
        # powers: 生成一个从 t-1 到 0 的倒序序列，如 [L-1, L-2, ..., 0]
        powers = torch.flip(torch.arange(t, dtype=torch.double), dims=(0,))
        # weights: 计算 (1-alpha) 的相应次幂，得到 [(1-alpha)^(L-1), ..., (1-alpha)^0]
        weights = torch.pow((1 - self.alpha), powers).to('cuda')
        # divisor: 除数，其初始值与weights相同，用于后续的并行计算
        divisor = weights.clone()

        # 2. 构造用于累积和的最终权重
        # 根据EMA展开式 s_t = a*x_t + a(1-a)x_{t-1} + ... + (1-a)^t*x_0，
        # 这里的权重是为并行扫描算法（parallel scan）设计的，并非直接的公式系数。
        # 第一个权重保持为(1-alpha)^(L-1)，其余权重乘以alpha
        weights[1:] = weights[1:] * self.alpha

        # 3. 调整权重和除数的形状以进行广播
        # # 维度变化: (L,) -> (1, L, 1)
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)

        # 4. 并行计算EMA
        # 将权重应用到输入x上
        # # x(B, L, D) * weights(1, L, 1) -> (B, L, D)
        x = x * weights
        # 沿时间维度计算累积和
        # # 维度变化: (B, L, D) -> (B, L, D)
        x = torch.cumsum(x, dim=1)
        # 将累积和除以之前计算的除数，得到每个时间步的EMA值
        # # 维度变化: (B, L, D) -> (B, L, D)
        x = torch.div(x, divisor)

        # 将结果转回 float32 类型并返回
        return x.to(torch.float32)

    # # Naive implementation with O(n) time complexity (时间复杂度为O(n)的朴素实现)
    # def forward(self, x):
    #     """
    #     这是EMA的标准递归实现，使用for循环，时间复杂度为O(L)。
    #     这个版本更容易理解，但效率较低，因此在最终模型中被优化版本替代。
    #     """
    #     # self.alpha.data.clamp_(0, 1)        # Clamp learnable alpha to [0, 1]
    #
    #     # s_0 = x_0
    #     s = x[:, 0, :]
    #     res = [s.unsqueeze(1)] # 存储每个时间步的结果
    #
    #     # 循环计算 s_t = alpha * x_t + (1-alpha) * s_{t-1}
    #     for t in range(1, x.shape[1]):
    #         xt = x[:, t, :]
    #         s = self.alpha * xt + (1 - self.alpha) * s
    #         res.append(s.unsqueeze(1))
    #
    #     # 将所有时间步的结果拼接起来
    #     return torch.cat(res, dim=1)
