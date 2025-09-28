# 导入os库，用于与操作系统交互，如此处的环境变量设置
import os
# 导入PyTorch库
import torch
# 导入Numpy库
import numpy as np


class Exp_Basic(object):
    """
    基础实验类 (Exp_Basic)。
    这个类定义了所有实验流程的骨架和通用功能，例如设备（CPU/GPU）的获取、模型的初始化。
    具体的实验类（如Exp_Main）将继承自这个基类，并实现其中的抽象方法。
    """

    def __init__(self, args):
        """
        构造函数，在创建实验对象时被调用。

        Args:
            args: 包含所有配置参数的对象。
        """
        # 保存传入的参数配置
        self.args = args
        # 调用_acquire_device方法获取并设置计算设备（GPU或CPU）
        self.device = self._acquire_device()
        # 调用_build_model方法构建模型，并使用.to(self.device)将其移动到已设置的设备上
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        """
        构建模型的占位符方法。
        这个方法应该在子类中被重写（override），以返回一个具体的模型实例。
        如果子类没有实现这个方法，调用时会引发NotImplementedError异常。
        """
        raise NotImplementedError
        return None

    def _acquire_device(self):
        """
        获取并配置计算设备。
        """
        # 检查参数中是否设置了使用GPU
        if self.args.use_gpu:
            # 设置CUDA_VISIBLE_DEVICES环境变量，这会告诉PyTorch哪些GPU是可见的。
            # 如果不使用多GPU，就设置为指定的单个GPU ID；如果使用多GPU，就设置为设备ID列表字符串。
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            # 创建一个PyTorch设备对象，指向配置的GPU
            device = torch.device('cuda:{}'.format(self.args.gpu))
            # 打印正在使用的GPU信息
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            # 如果不使用GPU，则创建指向CPU的设备对象
            device = torch.device('cpu')
            # 打印正在使用CPU的信息
            print('Use CPU')
        # 返回创建的设备对象
        return device

    def _get_data(self):
        """
        获取数据的占位符方法。
        子类应重写此方法以实现数据的加载逻辑。
        """
        pass

    def vali(self):
        """
        验证过程的占位符方法。
        子类应重写此方法以实现验证逻辑。
        """
        pass

    def train(self):
        """
        训练过程的占位符方法。
        子类应重写此方法以实现训练逻辑。
        """
        pass

    def test(self):
        """
        测试过程的占位符方法。
        子类应重写此方法以实现测试逻辑。
        """
        pass
