# 从模型检查点处理模块导入多个检查点管理函数
# 这些函数用于保存和加载模型及优化器的状态

# 导入模型检查点加载函数
# 用于从磁盘加载已保存的模型权重
from models.checkpoint_handler import (
    load_model_checkpoint,
    # 导入模型检查点保存函数
    # 用于将模型权重保存到磁盘
    save_model_checkpoint,
    # 导入优化器检查点加载函数
    # 用于恢复优化器的状态（如动量、学习率等）
    load_optimizer_checkpoint,
    # 导入优化器检查点保存函数
    # 用于保存优化器的状态以便后续恢复训练
    save_optimizer_checkpoint,
    # 导入分片模型和优化器保存函数
    # 用于在分布式训练中保存FSDP分片的模型和优化器
    save_model_and_optimizer_sharded,
    # 导入分片模型加载函数
    # 用于加载FSDP分片的模型权重
    load_model_sharded,
    # 导入单GPU分片模型加载函数
    # 用于在单个GPU上加载原本分片保存的模型
    load_sharded_model_single_gpu
)
