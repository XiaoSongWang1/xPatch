import torch
import torch.nn as nn
import math

# 从自定义的层中导入分解模块
from layers.decomp import DECOMP
# 从自定义的层中导入核心网络结构
from layers.network import Network
# from layers.network_mlp import NetworkMLP # For ablation study with MLP-only stream # 用于MLP-only流的消融研究
# from layers.network_cnn import NetworkCNN # For ablation study with CNN-only stream # 用于CNN-only流的消融研究
# 从自定义的层中导入Reversible Instance Normalization模块
from layers.revin import RevIN


class Model(nn.Module):
    """
    xPatch模型的主类。
    这个类整合了所有的组件，包括实例归一化（RevIN）、季节性趋势分解（DECOMP）以及核心的双流网络（Network）。
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # --- 获取参数 ---
        seq_len = configs.seq_len  # lookback window L, 输入序列长度
        pred_len = configs.pred_len  # prediction length T, 预测序列长度
        c_in = configs.enc_in  # input channels, 输入通道数/特征维度

        # --- Patching 相关参数 ---
        patch_len = configs.patch_len  # 每个patch的长度
        stride = configs.stride  # patch的步长
        padding_patch = configs.padding_patch  # patch的填充方式

        # --- 归一化 ---
        self.revin = configs.revin  # 是否启用Reversible Instance Normalization
        # 实例化RevIN层，affine=True表示有可学习的缩放和平移参数
        self.revin_layer = RevIN(c_in, affine=True, subtract_last=False)

        # --- 移动平均 (用于分解) ---
        self.ma_type = configs.ma_type  # 移动平均的类型, 'ema' 对应论文中的方法
        alpha = configs.alpha  # EMA的平滑因子
        beta = configs.beta  # DEMA的平滑因子

        # 实例化分解层，对应论文3.1节的 Seasonal-Trend Decomposition
        self.decomp = DECOMP(self.ma_type, alpha, beta)
        # 实例化核心的双流网络，对应论文3.2节的 Dual Flow Net
        self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch)
        # self.net_mlp = NetworkMLP(seq_len, pred_len) # For ablation study with MLP-only stream # 为MLP-only消融实验准备的网络
        # self.net_cnn = NetworkCNN(seq_len, pred_len, patch_len, stride, padding_patch) # For ablation study with CNN-only stream # 为CNN-only消融实验准备的网络

    def forward(self, x):
        # x: [Batch, Input, Channel] -> (B, L, D)
        # B: Batch size, L: seq_len (输入长度), D: c_in (特征维度)

        # --- 实例归一化 ---
        if self.revin:
            # # (B, L, D) -> (B, L, D)
            x = self.revin_layer(x, 'norm')

        if self.ma_type == 'reg':  # If no decomposition, directly pass the input to the network
            # 如果不进行分解，则直接将原始输入x同时作为季节项和趋势项传入网络
            # TODO: 论文中没有描述不使用分解的情况，这应是一种实验设置或基线对比。
            x = self.net(x, x)
            # # (B, L, D), (B, L, D) -> (B, T, D)  T是pred_len

            # x = self.net_mlp(x) # For ablation study with MLP-only stream
            # x = self.net_cnn(x) # For ablation study with CNN-only stream
        else:
            # --- 季节性趋势分解 ---
            # # (B, L, D) -> seasonal_init(B, L, D), trend_init(B, L, D)
            seasonal_init, trend_init = self.decomp(x)
            # 将分解后的季节项和趋势项分别送入双流网络
            # # seasonal_init(B, L, D), trend_init(B, L, D) -> x(B, T, D)
            x = self.net(seasonal_init, trend_init)

        # --- 反归一化 ---
        if self.revin:
            # # (B, T, D) -> (B, T, D)
            x = self.revin_layer(x, 'denorm')

        return x  # # 返回最终预测结果, 维度 (B, T, D)
