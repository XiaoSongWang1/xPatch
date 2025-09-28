# 从自定义的数据加载器模块中导入各种数据集类
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Solar, Dataset_Pred
# 从PyTorch中导入DataLoader，用于创建数据批次
from torch.utils.data import DataLoader

# 创建一个字典，将字符串名称映射到对应的数据集类
# 这样做的好处是可以通过命令行参数（如 'ETTh1'）方便地选择要使用的数据集类
data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'Solar': Dataset_Solar,  # TODO: 论文中没有提到Solar数据集，这是一个额外的数据集
    'custom': Dataset_Custom,
}


def data_provider(args, flag):
    """
    数据提供函数，根据参数args和模式flag（train, val, test）来创建并返回数据集和数据加载器。

    Args:
        args: 包含所有配置的参数对象。
        flag: 'train', 'val', or 'test'，指示当前是哪个阶段。

    Returns:
        data_set: 实例化的数据集对象。
        data_loader: 实例化的数据加载器对象。
    """
    # 根据命令行参数 args.data 从字典中获取对应的数据集类
    Data = data_dict[args.data]

    # timeenc: 时间编码标志。如果 args.embed 不是 'timeF'，则为0，表示不使用TimeFeature编码；否则为1。
    timeenc = 0 if args.embed != 'timeF' else 1
    # train_only: 是否只在完整数据集上训练，不划分验证集和测试集
    train_only = args.train_only

    if flag == 'test':
        # 如果是测试模式
        shuffle_flag = False  # 测试时不需要打乱数据顺序
        drop_last = True  # 保持与训练时一致，丢弃最后一个不完整的批次
        # drop_last = False # without the "drop-last" trick (不丢弃最后一个批次，这是另一种实验设置)
        batch_size = args.batch_size  # 测试时批次大小与训练时相同
        freq = args.freq  # 时间频率
    elif flag == 'pred':
        # 如果是预测模式（通常用于对新数据进行推理）
        shuffle_flag = False  # 不需要打乱
        drop_last = False  # 需要预测所有数据，不丢弃
        batch_size = 1  # 一次只预测一个样本
        freq = args.freq
        Data = Dataset_Pred  # 使用专门为预测任务设计的数据集类
    else:
        # 如果是训练模式 ('train') 或验证模式 ('val')
        shuffle_flag = True  # 训练时需要打乱数据顺序
        drop_last = True  # 丢弃最后一个不完整的批次，以确保所有批次大小一致
        batch_size = args.batch_size
        freq = args.freq
    # if flag == 'train': # 这段被注释掉的代码原本可能用于在训练时不丢弃最后一个批次
    #     drop_last = False

    # 实例化数据集对象
    data_set = Data(
        root_path=args.root_path,  # 数据文件根目录
        data_path=args.data_path,  # 数据文件名
        flag=flag,  # 'train', 'val', or 'test'
        size=[args.seq_len, args.label_len, args.pred_len],  # [输入序列长度, 标签长度, 预测序列长度]
        features=args.features,  # 预测任务类型 'M', 'S', 'MS'
        target=args.target,  # 目标变量名
        timeenc=timeenc,  # 时间编码标志
        freq=freq,  # 时间频率
        train_only=train_only  # 是否只训练
    )
    # 打印当前模式和数据集的样本数量
    print(flag, len(data_set))

    # 实例化数据加载器对象
    data_loader = DataLoader(
        data_set,  # 数据集
        batch_size=batch_size,  # 批次大小
        shuffle=shuffle_flag,  # 是否打乱
        num_workers=args.num_workers,  # 使用多少个子进程来加载数据
        drop_last=drop_last)  # 是否丢弃最后一个不完整的批次

    # 返回数据集和数据加载器
    return data_set, data_loader
