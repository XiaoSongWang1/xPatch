from typing import List

import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    """
    时间特征提取器的基类。
    定义了所有具体时间特征类需要遵循的接口。
    """

    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        """
        使得类的实例可以像函数一样被调用。
        子类需要实现这个方法来从pandas的DatetimeIndex中提取具体的特征。

        Args:
            index: 一个包含日期时间信息的时间索引。

        Returns:
            一个numpy数组，包含计算出的时间特征。
        """
        pass

    def __repr__(self):
        # 定义对象的字符串表示形式，方便调试。
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    # 将“分钟内的秒数”编码为[-0.5, 0.5]之间的值。

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # index.second: 获取秒数 (0-59)。
        # / 59.0: 归一化到 [0, 1] 范围。
        # - 0.5: 平移到 [-0.5, 0.5] 范围。
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]"""

    # 将“小时内的分钟数”编码为[-0.5, 0.5]之间的值。

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    # 将“一天中的小时数”编码为[-0.5, 0.5]之间的值。

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]"""

    # 将“一周中的天数”（星期几）编码为[-0.5, 0.5]之间的值。

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # index.dayofweek: 获取星期几 (Monday=0, Sunday=6)。
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]"""

    # 将“一月中的天数”编码为[-0.5, 0.5]之间的值。

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # index.day: 获取日期 (1-31)。
        # - 1: 平移到 0-30。
        # / 30.0: 近似归一化到 [0, 1]。
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]"""

    # 将“一年中的天数”编码为[-0.5, 0.5]之间的值。

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # index.dayofyear: 获取一年中的第几天 (1-366)。
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]"""

    # 将“一年中的月份”编码为[-0.5, 0.5]之间的值。

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # index.month: 获取月份 (1-12)。
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]"""

    # 将“一年中的周数”编码为[-0.5, 0.5]之间的值。

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # index.isocalendar().week: 获取ISO标准的周数 (1-53)。
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """
    根据给定的频率字符串，返回一个适合该频率的时间特征列表。
    例如，对于小时级数据('H')，我们关心当前是几点、星期几、几月几号等。
    对于分钟级数据('T')，我们除了关心小时级信息外，还关心当前是第几分钟。

    Parameters
    ----------
    freq_str
        频率字符串，例如 "12H", "5min", "1D" 等。
    """

    # 一个字典，将不同的时间偏移量类型映射到相应的时间特征类列表。
    # 粒度越细的时间频率，包含的特征越多。
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.QuarterEnd: [MonthOfYear],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [DayOfMonth, WeekOfYear],
        offsets.Day: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.BusinessDay: [DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Hour: [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear],
        offsets.Minute: [
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
        offsets.Second: [
            SecondOfMinute,
            MinuteOfHour,
            HourOfDay,
            DayOfWeek,
            DayOfMonth,
            DayOfYear,
        ],
    }

    # 将频率字符串转换为pandas的偏移量对象。
    offset = to_offset(freq_str)

    # 遍历字典，找到与当前频率匹配的偏移量类型。
    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            # 如果找到匹配的类型，则实例化该类型对应的所有特征类，并返回列表。
            return [cls() for cls in feature_classes]

    # 如果遍历完所有支持的类型都没有找到匹配的，则抛出错误。
    supported_freq_msg = f"""
    Unsupported frequency {freq_str}
    The following frequencies are supported:
        Y   - yearly
            alias: A
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
            alias: min
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, freq='h'):
    """
    主函数，用于从一个日期时间索引中提取所有相关的时间特征。

    Args:
        dates (pd.DatetimeIndex): 包含日期时间信息的时间索引。
        freq (str, optional): 数据的频率。默认为'h' (hourly)。

    Returns:
        np.ndarray: 一个2D numpy数组，每一行代表一个时间特征，每一列代表一个时间点。
                     维度: (num_features, num_dates)
    """
    # np.vstack: 垂直堆叠数组。
    # 列表推导式 [feat(dates) for feat in ...]:
    #   1. time_features_from_frequency_str(freq) 获取特征对象列表。
    #   2. 遍历列表，对每个特征对象feat调用其__call__方法 (即 feat(dates))，生成一个特征数组。
    #   3. vstack将所有这些特征数组垂直堆叠起来。
    return np.vstack([feat(dates) for feat in time_features_from_frequency_str(freq)])
