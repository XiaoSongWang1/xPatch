import numpy as np
import torch
import matplotlib.pyplot as plt  # 导入绘图库
import time

# 设置matplotlib的后端为'agg'。
# 'agg'是一个非交互式的后端，它只将图像渲染到文件（如PNG, PDF）。
# 这在服务器或没有图形用户界面（GUI）的环境中运行代码时非常重要，可以防止程序报错。
plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    """
    根据给定的策略，在每个epoch调整优化器的学习率。

    Args:
        optimizer: PyTorch的优化器 (e.g., AdamW).
        epoch (int): 当前的epoch数。
        args: 包含所有配置参数的对象，特别是args.lradj和args.learning_rate。
    """
    # lr = args.learning_rate * (0.2 ** (epoch // 2)) # 一种指数衰减策略，当前被注释

    # --- 学习率调整策略 ---
    if args.lradj == 'type1':
        # 每1个epoch，学习率乘以0.5
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        # 在指定的epoch手动设置学习率
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        # 前3个epoch保持初始学习率，之后每个epoch乘以0.9
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}

    # Sigmoid learning rate decay (Sigmoid学习率衰减)
    # 对应论文3.4节和公式(23)中提出的Sigmoid学习率调整方案
    elif args.lradj == 'sigmoid':
        k = 0.5  # logistic growth rate (逻辑斯蒂增长率)
        s = 10  # decreasing curve smoothing rate (下降曲线平滑率)
        w = 10  # warm-up coefficient (预热系数)
        # TODO: 这里的实现是两个Sigmoid函数相减，与论文最终的公式(36)形式相同，但具体参数组合可能需要参照附录H。
        lr_adjust = {epoch: args.learning_rate / (1 + np.exp(-k * (epoch - w))) - args.learning_rate / (
                1 + np.exp(-k / s * (epoch - w * s)))}

    elif args.lradj == 'constant':
        # 保持学习率不变
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate * 0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate * 0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate * 0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate * 0.1}

        # 如果当前epoch在预设的调整点上
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        # 遍历优化器中所有的参数组，并更新它们的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    """
    早停法，用于防止模型过拟合。
    当验证集上的损失在一定epoch数量（patience）内不再改善时，就停止训练。
    """

    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): 在停止训练前，允许验证集损失不改善的epoch数。
            verbose (bool): 如果为True，则在验证集损失改善时打印信息。
            delta (float): 损失改善的最小变化量，小于此值被视为没有改善。
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0  # 计数器，记录损失未改善的epoch数
        self.best_score = None  # 存储最佳分数（-val_loss）
        self.early_stop = False  # 早停标志
        self.val_loss_min = np.Inf  # 存储最小的验证集损失
        self.delta = delta

    def __call__(self, val_loss, model, path):
        """
        使得类的实例可以像函数一样被调用。

        Args:
            val_loss (float): 当前的验证集损失。
            model (nn.Module): 当前的模型。
            path (str): 模型检查点的保存路径。
        """
        score = -val_loss  # 我们希望损失越小越好，所以分数越大越好
        if self.best_score is None:
            # 第一个epoch
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            # 当前分数没有比最佳分数好（考虑到delta）
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                # 如果计数器超过了耐心值，则触发早停
                self.early_stop = True
        else:
            # 当前分数比最佳分数好
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)  # 保存模型
            self.counter = 0  # 重置计数器

    def save_checkpoint(self, val_loss, model, path):
        """
        当验证集损失减小时，保存模型。
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # 保存模型的状态字典
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """
    一个字典的子类，允许使用点表示法（dot notation）来访问字典的键值。
    例如，可以用 `d.key` 代替 `d['key']`。
    这是一个方便的工具类，常用于配置参数管理。
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    """
    标准缩放器，用于数据的标准化（z-score normalization）。
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        """
        标准化数据：(data - mean) / std
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """
        反标准化，将数据恢复到原始尺度：(data * std) + mean
        """
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization (结果可视化函数)

    Args:
        true (np.array): 真实值序列。
        preds (np.array, optional): 预测值序列。默认为None。
        name (str, optional): 保存图像的文件路径和名称。
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')  # 保存图像，bbox_inches='tight'会裁剪掉图像周围的空白
