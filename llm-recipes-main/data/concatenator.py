# 从tqdm库导入进度条功能
# tqdm用于显示数据处理的进度
from tqdm import tqdm
# 从PyTorch导入Dataset基类
# Dataset是所有自定义数据集的基类
from torch.utils.data import Dataset

# 定义拼接数据集类
# 该类将多个短序列拼接成固定长度的块，提高训练效率
class ConcatDataset(Dataset):
    # 初始化函数
    def __init__(self, dataset, chunk_size):
        # 保存原始数据集
        self.dataset = dataset
        # 保存块大小（每个训练样本的序列长度）
        self.chunk_size = chunk_size

        # 初始化样本列表，用于存储处理后的固定长度样本
        self.samples = []

        # 初始化缓冲区，用于临时存储拼接中的数据
        buffer = {
            "input_ids": [],        # 输入token ID列表
            "attention_mask": [],   # 注意力掩码列表
            "labels": [],           # 标签列表
        }

        # 遍历原始数据集的每个样本，显示进度条
        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            # 将当前样本的数据追加到缓冲区
            # 使用字典推导式将sample中的每个字段追加到buffer对应字段
            buffer = {k: v + sample[k] for k, v in buffer.items()}

            # 当缓冲区中的数据长度超过chunk_size时，切分出固定长度的块
            while len(next(iter(buffer.values()))) > self.chunk_size:
                # 从缓冲区前面切出chunk_size长度的数据作为一个样本
                self.samples.append({k: v[:self.chunk_size]
                                    for k, v in buffer.items()})
                # 将缓冲区中已使用的数据移除，保留剩余部分
                buffer = {k: v[self.chunk_size:] for k, v in buffer.items()}

    # 定义索引访问方法
    # 返回指定索引的样本
    def __getitem__(self, idx):
        return self.samples[idx]

    # 定义长度方法
    # 返回数据集中样本的总数
    def __len__(self):
        return len(self.samples)
