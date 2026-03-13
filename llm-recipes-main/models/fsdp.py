# 导入functools模块，用于创建偏函数
import functools
# 从PEFT库导入三种编码器类型
# PrefixEncoder: 前缀微调的编码器
# PromptEmbedding: 提示嵌入层
# PromptEncoder: 提示编码器
from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder
# 从PyTorch FSDP包装模块导入策略组合和自动包装策略函数
# _or_policy: 用于组合多个策略
# lambda_auto_wrap_policy: 基于lambda函数的自动包装策略
# transformer_auto_wrap_policy: 针对Transformer层的自动包装策略
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy

# TODO: 待完善的功能标记
# 定义函数：创建FSDP自动包装策略
# 该策略决定模型的哪些部分应该被FSDP包装（分片）
def fsdp_auto_wrap_policy(model, transformer_layer_name: list):
    # 定义lambda策略函数：判断模块是否应该被包装
    def lambda_policy_fn(module):
        # 检查三个条件：
        # 1. 模块没有子模块（叶子节点）
        # 2. 模块有weight属性
        # 3. weight需要梯度（可训练）
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            # 满足条件则包装该模块
            return True
        # 否则不包装
        return False

    # 创建lambda策略的偏函数
    # 使用上面定义的lambda_policy_fn作为判断函数
    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)

    # 创建Transformer包装策略的偏函数
    # 指定哪些类型的层应该被包装
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        # transformer_layer_cls指定要包装的层类型
        # 包括PEFT相关的编码器和用户指定的Transformer层
        transformer_layer_cls=(
            PrefixEncoder,      # 前缀编码器
            PromptEncoder,      # 提示编码器
            PromptEmbedding,    # 提示嵌入
            *transformer_layer_name,  # 展开用户指定的Transformer层列表
        ),
    )

    # 使用_or_policy组合两个策略
    # 只要满足任一策略，模块就会被包装
    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    # 返回组合后的自动包装策略
    return auto_wrap_policy
