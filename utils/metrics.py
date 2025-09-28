import numpy as np


def RSE(pred, true):
    """
    Root Relative Squared Error (RSE) - 根相对平方误差。
    它通过将模型的均方根误差(RMSE)与一个简单基线模型（预测所有值为真实值的均值）的RMSE进行比较来衡量性能。
    值越低越好。
    """
    # np.sum((true - pred) ** 2): 计算预测值和真实值之间差值的平方和 (分子)。
    # np.sum((true - true.mean()) ** 2): 计算真实值和其均值之间差值的平方和 (分母)，代表了基线模型的误差。
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    """
    Correlation Coefficient (CORR) - 相关系数。
    衡量预测值和真实值之间的线性相关程度。
    值在-1到1之间，越接近1表示正相关性越强，模型性能越好。
    """
    # u: 计算去中心化后的pred和true的协方差。
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    # d: 计算去中心化后的pred和true的标准差的乘积。
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    # 为防止分母为零，加上一个极小值。
    d += 1e-12
    # TODO: 乘以0.01的原因不明确，这可能是一个特定于某个基准测试的缩放因子，并非标准CORR计算的一部分。
    return 0.01 * (u / d).mean(-1)


def MAE(pred, true):
    """
    Mean Absolute Error (MAE) - 平均绝对误差。
    计算预测值和真实值之间绝对差的平均值。
    对异常值不那么敏感。值越低越好。
    """
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    """
    Mean Squared Error (MSE) - 均方误差。
    计算预测值和真实值之间差的平方的平均值。
    对较大的误差给予更高的权重。值越低越好。
    """
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    """
    Root Mean Squared Error (RMSE) - 均方根误差。
    是MSE的平方根，使其量纲与原始数据相同。
    """
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    """
    Mean Absolute Percentage Error (MAPE) - 平均绝对百分比误差。
    计算绝对误差占真实值的百分比的平均值。
    当真实值接近于零时，这个指标可能会变得非常大或无定义。
    """
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    """
    Mean Squared Percentage Error (MSPE) - 均方百分比误差。
    计算误差百分比的平方的平均值。
    """
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    """
    计算并返回一组评估指标。

    Args:
        pred (np.array): 模型的预测值。
        true (np.array): 对应的真实值。

    Returns:
        tuple: 包含计算出的指标。
    """
    # 对应论文第4节 "Evaluation Metrics" 部分，该研究主要使用 MAE 和 MSE。
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    # rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    # rse = RSE(pred, true)
    # corr = CORR(pred, true)

    # 当前代码只返回mae和mse，其他指标被注释掉了。
    return mae, mse
    # return mae, mse, rmse, mape, mspe, rse, corr
