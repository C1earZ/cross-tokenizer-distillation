# 从混合精度策略模块导入所有内容
# 混合精度训练可以使用FP16或BF16来加速训练并减少内存使用
from policies.mixed_precision import *

# 从包装策略模块导入所有内容
# 包装策略用于配置FSDP（全分片数据并行）如何包装模型层
from policies.wrapping import *

# 从激活检查点函数模块导入FSDP检查点应用函数
# 激活检查点通过重新计算来节省内存，适用于训练大型模型
from policies.activation_checkpointing_functions import apply_fsdp_checkpointing

# 从任意精度优化器模块导入AnyPrecisionAdamW优化器
# 这是一个支持混合精度训练的AdamW优化器变体
from policies.anyprecision_optimizer import AnyPrecisionAdamW
