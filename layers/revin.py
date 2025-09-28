import torch
from torch import nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) a.k.a. Reversible Instance Normalization.

    This module normalizes each time series instance (a single sample in a batch) independently.
    The key idea is to store the statistics (mean and std. dev.) used for normalization and
    later use them to perform a denormalization ('reversing') operation on the model's output.
    This helps the model handle distribution shifts between different time series instances.
    """

    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels (特征或通道的数量, D)
        :param eps: a value added for numerical stability (为保证数值稳定性而添加的一个小值)
        :param affine: if True, RevIN has learnable affine parameters (如果为True, RevIN将包含可学习的仿射参数)
        """
        super(RevIN, self).__init__()
        self.num_features = num_features  # 特征数量 D
        self.eps = eps  # 防止除以零的小值
        self.affine = affine  # 是否使用可学习的仿射变换参数
        self.subtract_last = subtract_last  # 一种变体，使用最后一个时间步的值代替均值进行中心化
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，维度通常为 (B, L, D)。
            mode (str): 操作模式，'norm' 表示归一化，'denorm' 表示反归一化。

        Returns:
            torch.Tensor: 处理后的张量，维度与输入相同 (B, L, D)。
        """
        if mode == 'norm':
            # 归一化模式：先计算并存储统计量，然后进行归一化
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            # 反归一化模式：使用存储的统计量进行反归一化
            x = self._denormalize(x)
        else:
            raise NotImplementedError  # 如果模式不是 'norm' 或 'denorm'，则抛出异常
        return x

    def _init_params(self):
        # initialize RevIN params: (C,) (初始化RevIN的仿射参数)
        # 可学习的缩放参数，初始化为1
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        # 可学习的平移参数，初始化为0
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """
        计算并存储每个实例的统计量（均值和标准差）。

        Args:
            x (torch.Tensor): 输入张量。维度: (B, L, D)
        """
        # dim2reduce: 指定需要计算统计量的维度。对于(B, L, D)，range(1, 2)即为(1,)，表示在时间步(L)维度上计算。
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            # 如果使用'subtract_last'模式，则将最后一个时间步的值作为中心化的基准
            # # x[:,-1,:] -> (B, D), .unsqueeze(1) -> (B, 1, D)
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            # 否则，计算在时间维度上的均值
            # # mean on dim=1 -> (B, 1, D)
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        # 计算在时间维度上的标准差
        # # var on dim=1 -> (B, 1, D), sqrt -> (B, 1, D)
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        # .detach()用于阻止梯度通过这些统计量计算过程回传

    def _normalize(self, x):
        """
        执行归一化操作。

        Args:
            x (torch.Tensor): 输入张量。维度: (B, L, D)

        Returns:
            torch.Tensor: 归一化后的张量。维度: (B, L, D)
        """
        if self.subtract_last:
            # # x(B, L, D) - self.last(B, 1, D) -> (B, L, D) # 广播机制
            x = x - self.last
        else:
            # # x(B, L, D) - self.mean(B, 1, D) -> (B, L, D) # 广播机制
            x = x - self.mean
        # # x(B, L, D) / self.stdev(B, 1, D) -> (B, L, D) # 广播机制
        x = x / self.stdev
        if self.affine:
            # 应用可学习的仿射变换
            # # x(B, L, D) * self.affine_weight(D) -> (B, L, D) # 广播机制
            x = x * self.affine_weight
            # # x(B, L, D) + self.affine_bias(D) -> (B, L, D) # 广播机制
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        """
        执行反归一化操作，是_normalize的逆过程。

        Args:
            x (torch.Tensor): 经过模型处理后的张量。维度: (B, T, D) or (B, L, D)

        Returns:
            torch.Tensor: 反归一化后的张量。维度与输入相同。
        """
        if self.affine:
            # 逆转仿射变换
            # # x(B, T, D) - self.affine_bias(D) -> (B, T, D) # 广播机制
            x = x - self.affine_bias
            # # x(B, T, D) / self.affine_weight(D) -> (B, T, D) # 广播机制
            x = x / (self.affine_weight + self.eps * self.eps)  # 加上eps防止除以零
        # # x(B, T, D) * self.stdev(B, 1, D) -> (B, T, D) # 广播机制
        x = x * self.stdev
        if self.subtract_last:
            # # x(B, T, D) + self.last(B, 1, D) -> (B, T, D) # 广播机制
            x = x + self.last
        else:
            # # x(B, T, D) + self.mean(B, 1, D) -> (B, T, D) # 广播机制
            x = x + self.mean
        return x
