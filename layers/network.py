import torch
from torch import nn


class Network(nn.Module):
    """
    xPatch的核心网络结构，实现了论文3.2节中描述的Dual Flow Net (双流网络)。
    它包含一个处理季节性分量的非线性流 (Non-linear Stream) 和一个处理趋势分量的线性流 (Linear Stream)。
    """

    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch):
        super(Network, self).__init__()

        # --- 通用参数 ---
        self.pred_len = pred_len  # 预测长度 T

        # ==================== Non-linear Stream (非线性流) ====================
        # 该流处理季节性分量，主要由Patching、CNN和MLP组成

        # --- Patching 参数 ---
        self.patch_len = patch_len  # 每个patch的长度 P
        self.stride = stride  # 步长 S
        self.padding_patch = padding_patch  # 填充策略
        self.dim = patch_len * patch_len  # Patch嵌入后的维度，论文中未明确指定为P*P，这是一个实现选择
        # 计算Patch的数量 N = floor((L - P) / S) + 1
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':  # 如果在末尾填充
            # 使用复制填充，在序列末尾填充stride个元素，以确保序列能被完整切分
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1  # 填充后会多一个patch

        # --- Patch Embedding (对应论文公式5) ---
        self.fc1 = nn.Linear(patch_len, self.dim)  # 嵌入层，将每个patch从长度P映射到维度dim
        self.gelu1 = nn.GELU()  # GELU激活函数
        self.bn1 = nn.BatchNorm1d(self.patch_num)  # 对patch维度进行批归一化

        # --- CNN Depthwise (深度可分离卷积的第一部分：深度卷积) ---
        # 对应论文公式6,7。使用分组卷积实现，每个patch(组)使用独立的卷积核。
        self.conv1 = nn.Conv1d(self.patch_num, self.patch_num,
                               kernel_size=patch_len, stride=patch_len, groups=self.patch_num)
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        # --- Residual Stream (残差连接) ---
        # 对应论文公式8中的残差项
        self.fc2 = nn.Linear(self.dim, patch_len)  # 将残差从dim映射回patch_len，以便相加

        # --- CNN Pointwise (深度可分离卷积的第二部分：逐点卷积) ---
        # 对应论文公式9,10。使用1x1卷积实现。
        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, kernel_size=1, stride=1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        # --- Flatten Head (展平头部，对应论文公式11) ---
        self.flatten1 = nn.Flatten(start_dim=-2)  # 展平patch_num和patch_len维度
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)  # MLP第一层，放大维度
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)  # MLP第二层，映射到预测长度

        # ======================= Linear Stream (线性流) =======================
        # 该流处理趋势分量，主要由MLP、平均池化和层归一化组成
        # 对应论文公式3,4

        # --- MLP Block 1 ---
        self.fc5 = nn.Linear(seq_len, pred_len * 4)  # 线性层
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)  # 平均池化层，用于平滑和降维
        self.ln1 = nn.LayerNorm(pred_len * 2)  # 层归一化

        # --- MLP Block 2 ---
        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        # --- Final Expansion Layer ---
        self.fc7 = nn.Linear(pred_len // 2, pred_len)  # 最终扩展到预测长度

        # ================= Streams Concatenation (双流拼接) =================
        # 对应论文公式12
        self.fc8 = nn.Linear(pred_len * 2, pred_len)  # 将拼接后的特征融合并映射到最终预测长度

    def forward(self, s, t):
        # s: seasonality (季节性分量) -> 维度: (B, L, D)
        # t: trend (趋势分量) -> 维度: (B, L, D)

        # 调整维度顺序以匹配Conv1d的输入要求 (Batch, Channel, Length)
        # # (B, L, D) -> (B, D, L)
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        # --- Channel Independence (通道独立性) ---
        # 为了让每个通道（特征）独立处理，将Batch和Channel维度合并
        B = s.shape[0]  # Batch size
        C = s.shape[1]  # Channel size (D)
        I = s.shape[2]  # Input size (L)
        # # (B, D, L) -> (B*D, L)
        s = torch.reshape(s, (B * C, I))
        t = torch.reshape(t, (B * C, I))

        # ==================== Non-linear Stream Forward ====================
        # --- Patching ---
        if self.padding_patch == 'end':
            # # (B*D, L) -> (B*D, L+S)
            s = self.padding_patch_layer(s)
        # 使用unfold进行滑动窗口切片，实现Patching
        # # (B*D, L+S) -> (B*D, N, P) N:patch_num, P:patch_len
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # s: [Batch and Channel, Patch_num, Patch_len]

        # --- Patch Embedding ---
        # # (B*D, N, P) -> (B*D, N, P*P)
        s = self.fc1(s)
        s = self.gelu1(s)
        # # (B*D, N, P*P) -> (B*D, N, P*P) # BatchNorm1d作用于N维度
        s = self.bn1(s)

        # # (B*D, N, P*P)
        res = s  # 保存残差

        # --- CNN Depthwise ---
        # # (B*D, N, P*P) -> (B*D, N, 1)  # Conv1d在最后一个维度上操作
        s = self.conv1(s)
        s = self.gelu2(s)
        # # s维度 (B*D, N, P)，因为卷积核大小和步长都是P
        # # TODO: 论文中深度卷积后维度应为(B*D, N, P)，这里代码实现后的维度是(B*D, N, P)。
        # # 经过卷积后，每个patch_len的输入变成了一个点。
        s = self.bn2(s)

        # --- Residual Stream ---
        # # res(B*D, N, P*P) -> res(B*D, N, P)
        res = self.fc2(res)
        # # s(B*D, N, P) + res(B*D, N, P) -> (B*D, N, P)
        s = s + res

        # --- CNN Pointwise ---
        # # (B*D, N, P) -> (B*D, N, P) # 1x1卷积不改变维度
        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        # --- Flatten Head ---
        # # (B*D, N, P) -> (B*D, N*P)
        s = self.flatten1(s)
        # # (B*D, N*P) -> (B*D, T*2)
        s = self.fc3(s)
        s = self.gelu4(s)
        # # (B*D, T*2) -> (B*D, T)
        s = self.fc4(s)  # 非线性流输出 s_out

        # ======================= Linear Stream Forward =======================
        # --- MLP ---
        # # t(B*D, L) -> (B*D, T*4)
        t = self.fc5(t)
        # # (B*D, T*4) -> (B*D, T*2)
        t = self.avgpool1(t)
        t = self.ln1(t)

        # # (B*D, T*2) -> (B*D, T)
        t = self.fc6(t)
        # # (B*D, T) -> (B*D, T/2)
        t = self.avgpool2(t)
        t = self.ln2(t)

        # # (B*D, T/2) -> (B*D, T)
        t = self.fc7(t)  # 线性流输出 t_out

        # ================= Streams Concatination Forward =================
        # # s_out(B*D, T), t_out(B*D, T) -> x(B*D, T*2)
        x = torch.cat((s, t), dim=1)
        # # x(B*D, T*2) -> x(B*D, T)
        x = self.fc8(x)

        # --- Channel Concatination (恢复通道维度) ---
        # # (B*D, T) -> (B, D, T)
        x = torch.reshape(x, (B, C, self.pred_len))

        # 恢复维度顺序为 (Batch, Length, Channel)
        # # (B, D, T) -> (B, T, D)
        x = x.permute(0, 2, 1)

        return x  # 返回最终预测结果```
