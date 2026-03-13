# 导入PyTorch深度学习框架
import torch
# 导入random模块，用于随机打乱数据
import random
# 导入numpy数值计算库
import numpy as np
# 从itertools导入islice函数，用于切片迭代器
from itertools import islice


# 定义基于长度的批次采样器类
# 继承自PyTorch的BatchSampler，用于按序列长度组织批次
class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    # 初始化函数
    def __init__(self, data_source, batch_size: int, drop_last: bool, shuffle: bool = True, seed: int = 0) -> None:
        # 检查数据源的第一个元素是否为字典类型
        if isinstance(next(iter(data_source)), dict):
            # 如果是字典，获取第一个键
            first_key = next(iter(next(iter(data_source)).keys()))
            # 提取每个样本中该键对应值的长度
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            # 如果不是字典，直接获取每个样本的长度
            self.lengths = [len(d) for d in data_source]
        # 保存批次大小
        self.batch_size = batch_size
        # 是否丢弃最后一个不完整的批次
        self.drop_last = drop_last
        # 是否打乱数据顺序
        self.shuffle = shuffle
        # 随机种子，用于保证可重复性
        self.seed = seed

    # 定义迭代器方法
    def __iter__(self):
        # 创建索引列表，范围为数据集的长度
        ids = list(range(len(self.lengths)))
        # 如果需要丢弃最后一个不完整批次，截断索引列表
        if self.drop_last: ids = ids[:len(ids) // self.batch_size * self.batch_size]

        # 如果需要打乱，使用指定种子打乱索引顺序
        if self.shuffle: random.Random(self.seed).shuffle(ids)
        # 将索引列表分割成多个批次
        batches = [ids[i:i+self.batch_size] for i in range(0, len(ids), self.batch_size)]

        # 逐个返回批次
        for b in batches: yield b

    # 定义长度方法，返回批次数量
    def __len__(self):
        # 如果丢弃最后一个不完整批次
        if self.drop_last:
            # 返回完整批次的数量
            return len(self.lengths) // self.batch_size
        else:
            # 返回所有批次的数量（包括可能不完整的最后一个批次）
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)


# 定义分布式基于长度的批次采样器类
# 用于在多GPU分布式训练中按长度组织批次
class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    # 初始化函数
    def __init__(self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0) -> None:
        # 创建基础的长度批次采样器
        # drop_last=True确保每个GPU获得相同数量的批次
        self.batch_sampler = LengthBasedBatchSampler(
            data_source, batch_size=batch_size, drop_last=True, shuffle=shuffle, seed=seed
        )
        # 保存副本数量（GPU数量）
        self.num_replicas = num_replicas
        # 保存当前进程的排名（GPU编号）
        self.rank = rank

    # 定义迭代器方法
    def __iter__(self):
        # 计算最大长度，确保能被副本数整除
        max_length = len(
            self.batch_sampler) // self.num_replicas * self.num_replicas
        # 使用islice从批次采样器中提取属于当前GPU的批次
        # 从rank位置开始，每隔num_replicas取一个批次
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    # 定义长度方法，返回当前GPU的批次数量
    def __len__(self):
        # 总批次数除以GPU数量
        return len(self.batch_sampler) // self.num_replicas
