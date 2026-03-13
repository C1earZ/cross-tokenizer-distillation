# 导入dataclasses模块中的dataclass装饰器
# dataclass用于自动生成__init__、__repr__等特殊方法，简化类的定义
from dataclasses import dataclass

# 使用dataclass装饰器定义数据集配置类
@dataclass
class dataset:
    # 数据集文件路径，默认为None
    # 指定训练数据集的位置（可以是本地路径或Hugging Face数据集ID）
    file: str = None

    # 训练集使用比例，默认为1（使用全部数据）
    # 可以设置为0-1之间的值来使用部分数据集，用于快速实验
    training_size: float = 1

    # 是否为编码器-解码器模型，默认为False
    # True表示使用T5、BART等seq2seq模型，False表示使用GPT、Llama等因果语言模型
    encoder_decoder: bool = False

    # 蒸馏相关配置：指定数据是由哪个模型生成的
    # 在知识蒸馏中，可以使用教师模型预先生成的输出作为训练数据
    # 例如："gpt-4"表示数据由GPT-4生成
    generated_by: str = None
