# 导入dataclasses模块中的dataclass装饰器
# 用于简化配置类的定义
from dataclasses import dataclass
# 从PyTorch分布式FSDP模块导入分片策略枚举
# ShardingStrategy定义了模型参数在多GPU间的分片方式
from torch.distributed.fsdp import ShardingStrategy
# 从PyTorch FSDP模块导入状态字典类型枚举
# StateDictType定义了模型检查点的保存格式
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

# 使用dataclass装饰器定义FSDP（全分片数据并行）配置类
# FSDP是一种分布式训练策略，可以在多GPU上训练超大模型
@dataclass
class fsdp_config:
    # 是否启用混合精度训练，默认为True
    # 混合精度可以加速训练并减少内存使用
    mixed_precision: bool = True

    # 是否使用FP16精度，默认为False
    # False表示使用BF16（如果mixed_precision为True）
    use_fp16: bool = False

    # 分片策略，默认为FULL_SHARD（完全分片）
    # FULL_SHARD：分片所有参数、梯度和优化器状态，最节省内存
    # 其他选项：SHARD_GRAD_OP（只分片梯度和优化器）、NO_SHARD（不分片）
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD

    # 检查点保存类型，默认为分片状态字典
    # SHARDED_STATE_DICT：每个GPU保存自己的分片，适合大模型
    # FULL_STATE_DICT：保存完整模型，占用更多磁盘空间
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT

    # 是否启用FSDP激活检查点，默认为True
    # 激活检查点通过重新计算来节省内存，适合训练超大模型
    fsdp_activation_checkpointing: bool = True

    # 是否将参数卸载到CPU，默认为False
    # True可以进一步节省GPU内存，但会降低训练速度
    fsdp_cpu_offload: bool = False

    # 是否使用纯BF16精度，默认为False
    # True表示所有计算都使用BF16，False表示混合精度（部分FP32）
    pure_bf16: bool = False

    # 优化器类型，默认为AdamW
    # AdamW是带权重衰减的Adam优化器，适合训练Transformer模型
    optimizer: str = "AdamW"
