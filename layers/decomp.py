import torch
from torch import nn

# 从自定义层中导入指数移动平均(EMA)和双指数移动平均(DEMA)的实现
from layers.ema import EMA
from layers.dema import DEMA


class DECOMP(nn.Module):
    """
    Series decomposition block (序列分解模块)
    这个模块实现了论文中3.1节描述的季节性趋势分解。
    它将输入的时间序列分解为季节性部分（残差）和趋势性部分（移动平均）。
    """

    def __init__(self, ma_type, alpha, beta):
        """
        构造函数。

        Args:
            ma_type (str): 移动平均的类型，'ema' 或 'dema'。
            alpha (float): EMA或DEMA中的alpha平滑因子。
            beta (float): DEMA中的beta平滑因子。
        """
        super(DECOMP, self).__init__()
        if ma_type == 'ema':
            # 如果选择'ema'，则实例化EMA模块，这对应于论文中提出的主要分解方法
            self.ma = EMA(alpha)
        elif ma_type == 'dema':
            # 如果选择'dema'，则实例化DEMA模块，这是一种可选的、更复杂的移动平均方法
            self.ma = DEMA(alpha, beta)

    def forward(self, x):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入的时间序列张量。
                               维度: (B, L, D)
                               B: Batch size (批次大小)
                               L: Sequence length (序列长度)
                               D: Number of features/channels (特征维度)

        Returns:
            tuple: 包含季节性部分和趋势性部分的元组 (res, moving_average)。
        """
        # 使用实例化的移动平均模块计算趋势部分 (X_T)
        # # x(B, L, D) -> moving_average(B, L, D)
        moving_average = self.ma(x)

        # 通过从原始序列中减去趋势部分来计算季节性（或残差）部分 (X_S = X - X_T)
        # # res(B, L, D) = x(B, L, D) - moving_average(B, L, D)
        res = x - moving_average

        # 返回季节性部分和趋势性部分
        return res, moving_average
