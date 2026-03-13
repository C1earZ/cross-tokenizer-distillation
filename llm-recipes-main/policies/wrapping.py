# 导入functools模块，用于创建偏函数
import functools

# 从transformers库导入Llama模型的解码器层
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
# 从transformers库导入GPT-NeoX模型的解码器层
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
# 从transformers库导入Mistral模型的解码器层
from  transformers.models.mistral.modeling_mistral import MistralDecoderLayer
# 从transformers库导入Falcon模型的解码器层
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
# 从PyTorch FSDP包装模块导入自动包装策略函数
# transformer_auto_wrap_policy: 基于Transformer层的自动包装策略
# size_based_auto_wrap_policy: 基于参数数量的自动包装策略
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)


# 定义函数：获取基于大小的包装策略
def get_size_policy(min_params=1e8):
    """创建基于参数数量的FSDP包装策略
    参数:
        min_params: 最小参数数量阈值，默认为1亿（1e8）
    返回:
        包装策略函数"""
    # 创建基于大小的包装策略偏函数
    # 只有参数数量超过min_params的模块才会被FSDP包装
    num_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=min_params
    )
    return num_wrap_policy


# 定义函数：获取Transformer层包装策略
def get_wrapper():
    """创建针对Transformer层的FSDP包装策略
    返回:
        包装策略函数"""
    # 创建Transformer自动包装策略偏函数
    # 指定哪些类型的层应该被FSDP包装
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        # transformer_layer_cls: 指定要包装的Transformer层类型集合
        # 包括Llama、GPT-NeoX、Mistral和Falcon的解码器层
        transformer_layer_cls={
            LlamaDecoderLayer,      # Llama解码器层
            GPTNeoXLayer,           # GPT-NeoX解码器层
            MistralDecoderLayer,    # Mistral解码器层
            FalconDecoderLayer,     # Falcon解码器层
        },
    )

    return auto_wrap_policy
