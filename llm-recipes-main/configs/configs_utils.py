# 导入PyTorch分布式训练模块
import torch.distributed as dist

# 从dataclasses导入asdict函数，用于将dataclass转换为字典
from dataclasses import asdict
# 从transformers导入默认数据整理器
from transformers import default_data_collator
# 从PyTorch导入分布式采样器
from torch.utils.data import DistributedSampler
# 从transformers导入序列到序列任务的数据整理器
from transformers.data import DataCollatorForSeq2Seq
# 从configs模块导入三种PEFT配置类
from configs import lora_config, llama_adapter_config, prefix_config
# 从peft库导入三种PEFT配置类
from peft import LoraConfig, AdaptionPromptConfig, PrefixTuningConfig
# 从data.sampler导入两种批次采样器
from data.sampler import LengthBasedBatchSampler, DistributedLengthBasedBatchSampler

# 定义函数：更新配置对象的属性
def update_config(config, isSubmodule=False, **kwargs):
    # 如果config是元组或列表（多个配置对象）
    if isinstance(config, (tuple, list)):
        # 递归更新每个配置对象
        for c in config:
            update_config(c, isSubmodule, **kwargs)
    else:
        # 遍历所有要更新的键值对
        for k, v in kwargs.items():
            # 如果键包含"."，表示是子模块的参数（如"lora_config.r"）
            if "." in k:
                # 分割配置名和参数名
                config_name, param_name = k.split(".")
                # 检查当前配置的类名是否匹配
                if type(config).__name__ == config_name:
                    # 如果配置有该参数属性
                    if hasattr(config, param_name):
                        # 设置参数值
                        setattr(config, param_name, v)
            # 如果不是子模块且配置有该属性
            elif not isSubmodule and hasattr(config, k):
                # 直接设置属性值
                setattr(config, k, v)

# 定义函数：生成PEFT配置对象
def generate_peft_config(train_config, kwargs):
    # 定义三种PEFT配置类的元组
    configs = (lora_config, llama_adapter_config, prefix_config)
    # 定义对应的peft库配置类的元组
    peft_configs = (LoraConfig, AdaptionPromptConfig, PrefixTuningConfig)
    # 提取配置类的名称（去掉"_config"后缀）
    names = tuple(c.__name__.rstrip("_config") for c in configs)

    # 断言训练配置中的PEFT方法在支持的方法列表中
    assert train_config.peft_method in names, f"Peft config not found: {train_config.peft_method}"

    # 根据PEFT方法名称获取对应的配置类并实例化
    config = configs[names.index(train_config.peft_method)]()

    # 使用kwargs更新配置
    update_config(config, **kwargs)
    # 将配置对象转换为字典
    params = asdict(config)
    # 使用字典参数创建peft库的配置对象
    peft_config = peft_configs[names.index(train_config.peft_method)](**params)
    # 返回PEFT配置对象
    return peft_config

# 定义函数：获取数据加载器的关键字参数
def get_dataloader_kwargs(train_config, dataset, tokenizer, mode, distil_config=None):
    # 判断是否启用FSDP（优先使用蒸馏配置，否则使用训练配置）
    fsdp = train_config.enable_fsdp or distil_config.enable_fsdp if distil_config else train_config.enable_fsdp
    # 初始化关键字参数字典
    kwargs = {}
    # 根据模式（训练/验证）选择批次大小
    batch_size = train_config.batch_size_training if mode == "train" else train_config.val_batch_size
    # 如果批处理策略是padding（填充）
    if train_config.batching_strategy == "padding":
        # 如果启用了FSDP分布式训练
        if fsdp:
            # 使用分布式长度批次采样器
            kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                dataset,
                batch_size=batch_size,
                rank=dist.get_rank(),  # 当前进程的排名
                num_replicas=dist.get_world_size(),  # 总进程数
                shuffle=mode == "train",  # 训练时打乱，验证时不打乱
                seed=train_config.seed  # 随机种子
            )
        else:
            # 使用普通长度批次采样器
            kwargs["batch_sampler"] = LengthBasedBatchSampler(
                dataset, batch_size, drop_last=True, shuffle=mode == "train", seed=train_config.seed)
        # 使用序列到序列数据整理器（自动填充）
        kwargs["collate_fn"] = DataCollatorForSeq2Seq(tokenizer)
    # 如果批处理策略是packing（打包）
    elif train_config.batching_strategy == "packing":
        # 如果启用了FSDP分布式训练
        if fsdp:
            # 使用分布式采样器
            kwargs["sampler"] = DistributedSampler(
                dataset,
                rank=dist.get_rank(),  # 当前进程的排名
                num_replicas=dist.get_world_size(),  # 总进程数
                shuffle=mode == "train",  # 训练时打乱
                seed=train_config.seed  # 随机种子
            )
        # 设置批次大小
        kwargs["batch_size"] = batch_size
        # 丢弃最后一个不完整的批次
        kwargs["drop_last"] = True
        # 使用默认数据整理器
        kwargs["collate_fn"] = default_data_collator
    else:
        # 如果批处理策略未知，抛出错误
        raise ValueError(
            f"Unknown batching strategy: {train_config.batching_strategy}")
    # 返回数据加载器的关键字参数
    return kwargs
