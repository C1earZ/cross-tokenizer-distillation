# 导入dataclasses模块中的dataclass装饰器
# 用于简化配置类的定义
from dataclasses import dataclass

# 使用dataclass装饰器定义训练配置类
# 包含模型训练的所有超参数和设置
@dataclass
class train_config:
    # 项目名称，默认为None
    # 用于wandb等实验追踪工具的项目标识
    project_name: str=None

    # 模型名称，默认为Llama-2-7b
    # 指定要训练或微调的预训练模型
    model_name: str="meta-llama/Llama-2-7b-hf"

    # 是否启用FSDP分布式训练，默认为False
    enable_fsdp: bool=False

    # 是否使用低CPU内存的FSDP模式，默认为False
    low_cpu_fsdp: bool=False

    # 是否运行验证，默认为True
    # True表示在训练过程中定期在验证集上评估模型
    run_validation: bool=True

    # 训练批次大小，默认为8
    # 每次前向传播处理的样本数量
    batch_size_training: int=8

    # 批处理策略，默认为"padding"
    # "padding"表示将序列填充到相同长度，"packing"表示将多个短序列打包
    batching_strategy: str="padding"

    # 上下文长度限制，默认为None
    # 指定输入序列的最大长度，None表示不限制
    context_length: int=None

    # 梯度累积步数，默认为1
    # 累积多个批次的梯度后再更新参数，可以模拟更大的批次
    gradient_accumulation_steps: int=1

    # 训练轮数，默认为1
    # 完整遍历训练集的次数
    num_epochs: int=1

    # 数据加载器的工作线程数，默认为2
    # 用于并行加载数据
    num_workers_dataloader: int=2

    # 学习率，默认为1e-6
    # 控制参数更新的步长
    lr: float=1e-6

    # 权重衰减系数，默认为0.1
    # L2正则化参数，防止过拟合
    weight_decay: float=0.1

    # OneCycle学习率调度器的峰值位置，默认为0.1
    # 表示在训练的前10%达到最大学习率
    pct_start=0.1

    # OneCycle学习率调度器的初始除数，默认为2
    # 初始学习率 = max_lr / div_factor
    div_factor=2

    # OneCycle学习率调度器的最终除数，默认为5
    # 最终学习率 = max_lr / (div_factor * final_div_factor)
    final_div_factor=5

    # 随机种子，默认为42
    # 用于保证实验的可重复性
    seed: int=42

    # 是否使用FP16精度，默认为False
    use_fp16: bool=False

    # 是否启用混合精度训练，默认为True
    mixed_precision: bool=True

    # 验证批次大小，默认为1
    # 验证时每次处理的样本数量
    val_batch_size: int=1

    # PEFT方法类型，默认为"lora"
    # 可选值：lora、prefix、llama_adapter等
    peft_method: str = "lora"

    # 是否使用PEFT，默认为False
    use_peft: bool=False

    # 输出目录，默认为空字符串
    # 保存模型检查点和日志的路径
    output_dir: str = ""

    # 是否冻结部分层，默认为False
    freeze_layers: bool = False

    # 冻结的层数，默认为1
    num_freeze_layers: int = 1

    # 是否启用量化，默认为False
    quantization: bool = False

    # 是否保存模型，默认为True
    save_model: bool = True

    # 保存检查点的步数间隔，默认为1000
    # 每训练1000步保存一次模型
    save_step: int = 1000

    # 是否保存优化器状态，默认为False
    save_optimizer: bool=False

    # 是否使用快速内核，默认为False
    use_fast_kernels: bool = False

    # 是否启用知识蒸馏，默认为False
    distillation: bool = False

    # 是否保存所有检查点，默认为False
    # True表示保存所有中间检查点，False表示只保存最新的
    save_all: bool = False

    # 训练集使用比例，默认为1
    # 可以设置为小于1的值来使用部分训练数据
    training_size: int = 1

    # 是否为编码器-解码器模型，默认为False
    encoder_decoder: bool = False
