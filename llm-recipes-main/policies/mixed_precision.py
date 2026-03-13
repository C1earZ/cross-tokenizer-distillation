# 导入PyTorch深度学习框架
import torch

# 从PyTorch FSDP模块导入混合精度配置类
from torch.distributed.fsdp import (
    MixedPrecision,
)

# 定义FP16混合精度策略
# 需要在主训练循环中使用梯度缩放器（grad scaler）
fpSixteen = MixedPrecision(
    # 参数数据类型：使用float16存储模型参数
    param_dtype=torch.float16,
    # 梯度通信精度：使用float16进行梯度的跨GPU通信
    reduce_dtype=torch.float16,
    # 缓冲区精度：使用float16存储缓冲区数据
    buffer_dtype=torch.float16,
)

# 定义BF16混合精度策略
# BF16（Brain Float 16）比FP16更稳定，不需要梯度缩放器
bfSixteen = MixedPrecision(
    # 参数数据类型：使用bfloat16存储模型参数
    param_dtype=torch.bfloat16,
    # 梯度通信精度：使用bfloat16进行梯度的跨GPU通信
    reduce_dtype=torch.bfloat16,
    # 缓冲区精度：使用bfloat16存储缓冲区数据
    buffer_dtype=torch.bfloat16,
    # 是否将前向传播的输入转换为指定精度
    cast_forward_inputs=True,
)

# 定义BF16混合精度策略（混合版本）
# 参数保持FP32精度，但梯度通信和缓冲区使用BF16
bfSixteen_mixed = MixedPrecision(
    # 参数数据类型：使用float32存储模型参数（更高精度）
    param_dtype=torch.float32,
    # 梯度通信精度：使用bfloat16进行梯度通信（节省带宽）
    reduce_dtype=torch.bfloat16,
    # 缓冲区精度：使用bfloat16存储缓冲区数据
    buffer_dtype=torch.bfloat16,
)

# 定义FP32全精度策略
# 所有计算都使用float32，精度最高但速度最慢、内存占用最大
fp32_policy = MixedPrecision(
    # 参数数据类型：使用float32存储模型参数
    param_dtype=torch.float32,
    # 梯度通信精度：使用float32进行梯度通信
    reduce_dtype=torch.float32,
    # 缓冲区精度：使用float32存储缓冲区数据
    buffer_dtype=torch.float32,
)
