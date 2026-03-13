# 导入dataclasses模块中的dataclass装饰器
# 用于简化配置类的定义
from dataclasses import dataclass
# 从PyTorch分布式FSDP模块导入分片策略枚举
from torch.distributed.fsdp import ShardingStrategy
# 从PyTorch FSDP模块导入状态字典类型枚举
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

# 使用dataclass装饰器定义知识蒸馏配置类
# 包含蒸馏训练的所有超参数和设置
@dataclass
class distillation_config:
    # 学生模型名称，默认为Llama-2-7b
    # 指定要训练的学生模型（较小的模型）
    model_name: str = "meta-llama/Llama-2-7b-hf"

    # 是否启用FSDP分布式训练，默认为False
    # True表示使用多GPU分布式训练
    enable_fsdp: bool = False

    # 是否使用低CPU内存的FSDP模式，默认为False
    # True可以减少CPU内存使用，适合内存受限的环境
    low_cpu_fsdp: bool = False

    # 是否启用模型量化，默认为False
    # 量化可以减少模型大小和推理时间
    quantization: bool = False

    # 是否使用快速内核（优化的CUDA内核），默认为False
    # True可以加速训练，但可能需要特定的硬件支持
    use_fast_kernels: bool = False

    # 是否使用PEFT（参数高效微调），默认为False
    # True表示使用LoRA等方法只训练部分参数
    use_peft: bool = False

    # 是否冻结部分层，默认为False
    # True表示冻结模型的前几层，只训练后面的层
    freeze_layers: bool = False

    # 冻结的层数，默认为0
    # 指定从模型开始冻结多少层
    num_freeze_layers: int = 0

    # 交叉熵损失的权重因子，默认为1
    # 控制标准语言模型损失在总损失中的比重
    cross_entropy_factor: float = 1

    # 蒸馏损失的权重因子，默认为1.5
    # 控制知识蒸馏损失在总损失中的比重，通常大于1以强调蒸馏
    distil_factor: float = 1.5

    # 学生模型的温度参数，默认为1
    # 温度用于软化学生模型的输出概率分布，温度越高分布越平滑
    student_temperature: float = 1

    # 教师模型的温度参数，默认为1
    # 温度用于软化教师模型的输出概率分布
    teacher_temperature: float = 1

    # 是否为编码器-解码器模型，默认为False
    # True表示使用T5、BART等seq2seq模型
    encoder_decoder: bool = False

    # ===== FSDP相关配置 =====
    # 以下参数用于配置FSDP分布式训练

    # 是否启用混合精度训练，默认为False
    mixed_precision: bool = False

    # 是否使用FP16精度，默认为False
    use_fp16: bool = False

    # 分片策略，默认为完全分片
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD

    # 检查点保存类型，默认为分片状态字典
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT

    # 是否启用FSDP激活检查点，默认为True
    fsdp_activation_checkpointing: bool = True

    # 是否将参数卸载到CPU，默认为False
    fsdp_cpu_offload: bool = False

    # 是否使用纯BF16精度，默认为False
    pure_bf16: bool = False

    # 优化器类型，默认为AdamW
    optimizer: str = "AdamW"
