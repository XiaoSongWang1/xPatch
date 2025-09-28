# 导入所需的库
import argparse  # 用于解析命令行参数
import os  # 用于与操作系统交互，例如文件路径操作
import torch  # PyTorch深度学习框架

# 从自定义的实验管理模块中导入主实验类
from exp.exp_main import Exp_Main
import random  # 用于生成随机数
import numpy as np  # 用于数值计算，特别是数组操作

# 设置固定的随机种子以确保实验的可复现性
# 不同的种子可能会导致不同的初始化权重和数据划分，从而影响最终结果
fix_seed = 2021
# fix_seed = 2022
# fix_seed = 2023
# fix_seed = 2024
# fix_seed = 2025
random.seed(fix_seed)  # 设置Python内置random模块的种子
torch.manual_seed(fix_seed)  # 为CPU设置PyTorch的随机种子
np.random.seed(fix_seed)  # 设置NumPy的随机种子

# 创建一个ArgumentParser对象，用于添加和解析命令行参数
parser = argparse.ArgumentParser(description='xPatch')

# --- 基本配置 ---
# is_training: 标志位，1表示训练模式，0表示测试模式
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
# train_only: 是否只在完整输入数据集上进行训练，而不进行验证和测试
parser.add_argument('--train_only', type=bool, required=False, default=False,
                    help='perform training on full input dataset without validation and testing')
# model_id: 模型的唯一标识符，用于保存模型文件和日志
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
# model: 使用的模型名称，这里默认为xPatch
parser.add_argument('--model', type=str, required=True, default='xPatch',
                    help='model name, options: [xPatch]')

# --- 数据加载器配置 ---
# data: 数据集类型，例如ETTh1, ETTm2等
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
# root_path: 数据文件所在的根目录
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
# data_path: 具体的数据文件名
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
# features: 预测任务类型。M: 多变量预测多变量, S: 单变量预测单变量, MS: 多变量预测单变量
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
# target: 在S或MS任务中，需要预测的目标特征列名
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
# freq: 时间特征编码的频率。例如'h'表示小时级。这用于生成更精细的时间特征
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
# checkpoints: 保存模型检查点的路径
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
# embed: 时间特征编码的方式。timeF: 使用TimeFeature-Embedding, fixed/learned: 其他编码方式
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')

# --- 预测任务配置 ---
# seq_len: 输入序列的长度（lookback window）
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
# label_len: 标签序列长度，在解码器中用作起始token
parser.add_argument('--label_len', type=int, default=48, help='start token length')
# pred_len: 预测序列的长度（prediction horizon）
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
# enc_in: 编码器输入的大小，即数据集的特征维度（变量数量）
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')

# --- Patching 配置 --- (对应论文中的Patching技术)
# patch_len: 每个Patch的长度
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
# stride: 创建Patch时的步长
parser.add_argument('--stride', type=int, default=8, help='stride')
# padding_patch: Patching时的填充策略。'end'表示在序列末尾填充
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')

# --- 移动平均配置 --- (对应论文中的Seasonal-Trend Decomposition)
# ma_type: 移动平均的类型。ema对应论文中的Exponential Moving Average
parser.add_argument('--ma_type', type=str, default='ema', help='reg, ema, dema')
# alpha: EMA中的平滑因子alpha
parser.add_argument('--alpha', type=float, default=0.3, help='alpha')
# beta: Double EMA中的beta因子
parser.add_argument('--beta', type=float, default=0.3, help='beta')

# --- 优化配置 ---
# num_workers: 数据加载器使用的工作进程数
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
# itr: 重复实验的次数
parser.add_argument('--itr', type=int, default=1, help='experiments times')
# train_epochs: 训练的总轮数
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
# batch_size: 训练输入的批次大小
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
# patience: 早停法的耐心值，即连续多少个epoch验证集损失没有改善就停止训练
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
# learning_rate: 优化器的学习率
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
# des: 实验的描述信息，会附加到setting字符串中
parser.add_argument('--des', type=str, default='test', help='exp description')
# loss: 损失函数类型，例如mse, mae
parser.add_argument('--loss', type=str, default='mse', help='loss function')
# lradj: 学习率调整策略
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
# use_amp: 是否使用自动混合精度训练，可以加速训练并减少显存占用
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
# revin: 是否使用RevIN (Reversible Instance Normalization)，一种归一化方法
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
# parser.add_argument('--warmup_epochs',type=int,default = 0) # 预热轮数，当前被注释掉了

# --- GPU 配置 ---
# use_gpu: 是否使用GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
# gpu: 使用的GPU设备ID
parser.add_argument('--gpu', type=int, default=0, help='gpu')
# use_multi_gpu: 是否使用多GPU
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
# devices: 使用的多GPU设备ID列表，以逗号分隔
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
# test_flop: 是否测试模型的浮点运算次数(FLOPs)
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# 解析命令行传入的参数
args = parser.parse_args()

# 根据PyTorch的可用性以及用户设置来决定是否最终使用GPU
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

# 如果使用多GPU，则进行相应的配置
if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')  # 去除设备ID字符串中的空格
    device_ids = args.devices.split(',')  # 按逗号分割成列表
    args.device_ids = [int(id_) for id_ in device_ids]  # 转换为整数列表
    args.gpu = args.device_ids[0]  # 将主GPU设备设置为列表中的第一个

# 打印本次实验的所有参数配置
print('Args in experiment:')
print(args)

# 将主实验类赋值给一个简短的变量名
Exp = Exp_Main

# 如果是训练模式
if args.is_training:
    # 根据itr参数设定的次数，循环进行多次实验
    for ii in range(args.itr):
        # 构建一个详细的、唯一的实验设置字符串，用于命名和记录
        # 格式: {模型ID}_{模型名}_{数据名}_ft{特征类型}_sl{输入长度}_ll{标签长度}_pl{预测长度}_{描述}_{实验次数}
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.des, ii)

        # 实例化实验对象
        exp = Exp(args)  # set experiments
        # 开始训练
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        # 训练结束后进行测试
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        # 清空CUDA缓存，释放未被引用的显存
        torch.cuda.empty_cache()
else:
    # 如果是测试模式
    ii = 0
    # 构建实验设置字符串
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(args.model_id,
                                                          args.model,
                                                          args.data,
                                                          args.features,
                                                          args.seq_len,
                                                          args.label_len,
                                                          args.pred_len,
                                                          args.des, ii)

    # 实例化实验对象
    exp = Exp(args)  # set experiments
    # 只进行测试
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)  # test=1可能表示加载已保存的模型进行测试
    # 清空CUDA缓存
    torch.cuda.empty_cache()
