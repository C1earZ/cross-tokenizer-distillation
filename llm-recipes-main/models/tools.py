# 导入PyTorch深度学习框架
import torch
# 导入PyTorch CUDA NCCL通信库
import torch.cuda.nccl as nccl
# 导入PyTorch分布式训练模块
import torch.distributed as dist

# 从pkg_resources导入packaging模块，用于版本比较
from pkg_resources import packaging
# 从policies模块导入混合精度策略和包装策略
from policies import fpSixteen, bfSixteen, get_wrapper

# 定义函数：获取模型参数的数据类型
def get_parameter_dtypes(model):
    """获取模型中所有参数的数据类型"""
    # 初始化参数数据类型字典
    parameter_dtypes = {}
    # 遍历模型的所有命名参数
    for name, parameter in model.named_parameters():
        # 记录每个参数的数据类型
        parameter_dtypes[name] = parameter.dtype
    # 返回参数数据类型字典
    return parameter_dtypes

# 定义函数：冻结Transformer模型的前几层
def freeze_transformer_layers(model, num_layer):
    """冻结模型的前num_layer层，使其不参与训练"""
    # 遍历模型的所有层（带索引）
    for i, layer in enumerate(model.model.layers):
        # 如果层索引小于指定的冻结层数
        if i < num_layer:
            # 遍历该层的所有参数
            for param in layer.parameters():
                # 设置参数不需要梯度（冻结）
                param.requires_grad = False

# 定义函数：检查PEFT模型中冻结的层
def check_frozen_layers_peft_model(model):
    """打印PEFT模型中每层每个参数的requires_grad状态"""
    # 遍历PEFT模型基础模型的所有层
    for i, layer in enumerate(model.base_model.model.model.layers):
        # 遍历该层的所有命名参数
        for name, param in layer.named_parameters():
            # 打印层索引、参数名和是否需要梯度
            print(
                f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")

# 定义函数：获取混合精度和包装策略
def get_policies(cfg, rank):
    """根据配置和硬件支持情况选择合适的混合精度策略"""
    # 验证是否支持BFloat16
    # 需要满足：CUDA可用、硬件支持BF16、CUDA版本>=11.0、NCCL可用且版本>=2.10
    verify_bfloat_support = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and packaging.version.parse(torch.version.cuda).release >= (11, 0)
        and dist.is_nccl_available()
        and nccl.version() >= (2, 10)
    )

    # 初始化混合精度策略为None
    mixed_precision_policy = None
    # 初始化包装策略为None
    wrapping_policy = None

    # 如果配置启用了混合精度
    if cfg.mixed_precision:
        # 检查BFloat16是否就绪
        bf16_ready = verify_bfloat_support
        # 如果BF16就绪且未强制使用FP16
        if bf16_ready and not cfg.use_fp16:
            # 使用BFloat16混合精度策略
            mixed_precision_policy = bfSixteen
            # 主进程打印信息
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        # 如果配置强制使用FP16
        elif cfg.use_fp16:
            # 使用FP16混合精度策略
            mixed_precision_policy = fpSixteen
            # 主进程打印信息
            if rank == 0:
                print(f"FP16 enabled")
        else:
            # 如果不支持BF16，使用FP32全精度
            print(f"bFloat16 support not present. Using FP32, and not mixed precision")
    # 获取FSDP包装策略
    wrapping_policy = get_wrapper()
    # 返回混合精度策略和包装策略
    return mixed_precision_policy, wrapping_policy

# 定义函数：打印模型大小信息
def print_model_size(model, config, rank: int = 0) -> None:
    """打印模型的参数数量（仅在主进程打印）"""
    # 只在主进程（rank 0）打印
    if rank == 0:
        # 打印模型名称
        print(f"--> Model {config.model_name}")
        # 计算可训练参数总数
        total_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
        # 打印参数数量（以百万为单位）
        print(
            f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")
