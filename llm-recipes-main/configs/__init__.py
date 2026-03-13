# 从PEFT（参数高效微调）配置模块导入三种配置
# lora_config: LoRA（低秩适应）配置，通过低秩矩阵分解来减少可训练参数
# llama_adapter_config: Llama适配器配置，为Llama模型设计的轻量级适配器
# prefix_config: 前缀微调配置，通过添加可训练的前缀token来适应新任务
from configs.peft import lora_config, llama_adapter_config, prefix_config

# 从FSDP配置模块导入FSDP配置
# FSDP（全分片数据并行）用于在多GPU上分布式训练大型模型
from configs.fsdp import fsdp_config

# 从训练配置模块导入训练配置
# 包含学习率、批次大小、训练轮数等超参数
from configs.training import train_config

# 从蒸馏配置模块导入蒸馏配置
# 包含知识蒸馏的特定参数，如温度、损失权重等
from configs.distillation import distillation_config

# 从数据集配置模块导入数据集配置
# 定义使用哪些数据集以及数据集的加载参数
from configs.datasets import dataset
