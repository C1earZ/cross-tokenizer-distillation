# 从functools导入partial函数，用于创建偏函数
from functools import partial
# 从transformers库导入Llama模型的解码器层
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
# 从transformers库导入GPT-NeoX模型的解码器层
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
# 从transformers库导入Mistral模型的解码器层
from  transformers.models.mistral.modeling_mistral import MistralDecoderLayer
# 从transformers库导入Falcon模型的解码器层
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
# 从PyTorch分布式算法模块导入检查点包装相关函数
# checkpoint_wrapper: 用于包装模块以启用激活检查点
# CheckpointImpl: 检查点实现方式的枚举
# apply_activation_checkpointing: 应用激活检查点到模型
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

# 创建非重入式检查点包装器的偏函数
# 非重入式（NO_REENTRANT）检查点更节省内存，但不支持某些高级功能
non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

# 定义检查函数：判断子模块是否应该应用激活检查点
# 只对Llama、GPT-NeoX、Mistral和Falcon的解码器层应用检查点
check_fn = lambda submodule: isinstance(submodule, (LlamaDecoderLayer, GPTNeoXLayer, MistralDecoderLayer, FalconDecoderLayer))


# 定义函数：对模型应用FSDP激活检查点
def apply_fsdp_checkpointing(model):
    """对模型应用激活检查点
    激活检查点通过在反向传播时重新计算激活值来节省内存
    返回None，因为模型是直接更新的"""
    # 打印提示信息
    print(f"--> applying fsdp activation checkpointing...")

    # 应用激活检查点到模型
    # checkpoint_wrapper_fn: 指定使用非重入式包装器
    # check_fn: 指定哪些子模块应该被包装
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )
