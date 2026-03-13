# 导入操作系统模块
import os
# 导入fire库，用于自动生成命令行接口
import fire
# 导入random模块，用于设置随机种子
import random
# 导入PyTorch深度学习框架
import torch

# 从configs模块导入数据集配置类
from configs import dataset as DATA_CONFIG
# 从configs模块导入FSDP配置类
from configs import fsdp_config as FSDP_CONFIG
# 从configs模块导入训练配置类
from configs import train_config as TRAIN_CONFIG
# 从configs模块导入蒸馏配置类
from configs import distillation_config as DISTIL_CONFIG

# 从train.train_utils导入训练函数
from train.train_utils import train
# 从configs.configs_utils导入配置更新函数
from configs.configs_utils import update_config
# 从data.data_utils导入数据加载器获取函数
from data.data_utils import (get_dataloader, get_distillation_dataloader)
# 从train.tools导入分布式训练工具函数
from train.tools import (setup, setup_environ_flags, clear_gpu_cache)
# 从models.models_utils导入模型和优化器获取函数
from models.models_utils import (get_model, get_distillation_models, get_optimizer)

# 设置环境变量：禁用transformers库的警告信息
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = "true"
# 设置环境变量：启用分词器并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# 定义主函数：训练或微调模型的入口点
def main(**kwargs):
    """主函数：初始化配置、加载模型和数据、执行训练"""
    # 实例化四个配置对象：训练配置、FSDP配置、蒸馏配置、数据配置
    train_config, fsdp_config, distil_config, data_config = TRAIN_CONFIG(), FSDP_CONFIG(), DISTIL_CONFIG(), DATA_CONFIG()
    # 使用命令行参数更新训练、FSDP和数据配置
    update_config((train_config, fsdp_config, data_config), **kwargs)
    # 使用命令行参数更新蒸馏配置（作为子模块）
    update_config((distil_config), isSubmodule=True, **kwargs)

    # 设置CUDA随机种子，确保GPU计算的可重复性
    torch.cuda.manual_seed(train_config.seed)
    # 设置PyTorch随机种子，确保CPU计算的可重复性
    torch.manual_seed(train_config.seed)
    # 设置Python随机种子，确保随机操作的可重复性
    random.seed(train_config.seed)

    # 如果启用了FSDP分布式训练（训练配置或蒸馏配置中任一启用）
    if train_config.enable_fsdp or distil_config.enable_fsdp:
        # 初始化分布式进程组
        setup()
        # 获取本地进程排名（当前机器上的GPU编号）
        local_rank = int(os.environ["LOCAL_RANK"])
        # 获取全局进程排名（所有机器上的进程编号）
        rank = int(os.environ["RANK"])
    # 如果不使用分布式训练，设置rank为0（单GPU）
    else: rank = 0

    # 如果分布式训练已初始化
    if torch.distributed.is_initialized():
        # 设置当前进程使用的CUDA设备
        torch.cuda.set_device(local_rank)
        # 清空GPU缓存
        clear_gpu_cache(local_rank)
        # 设置环境标志用于调试
        setup_environ_flags(rank)

    # 加载模型和分词器
    # 如果启用了知识蒸馏
    if train_config.distillation:
        # 加载学生分词器、教师分词器和学生模型
        student_tokenizer, teacher_tokenizer, model = get_distillation_models(train_config, distil_config, fsdp_config, rank, kwargs)
    # 如果不使用蒸馏（常规训练）
    else:
        # 加载分词器和模型
        tokenizer, model = get_model(train_config, fsdp_config, rank, kwargs)
    # 主进程打印模型结构
    if rank == 0: print(model)

    # 加载数据
    # 设置数据配置的编码器-解码器标志
    data_config.encoder_decoder = train_config.encoder_decoder
    # 如果启用了知识蒸馏
    if train_config.distillation:
        # 获取学生和教师的训练/验证数据加载器
        train_dataloader, teacher_train_dataloader, eval_dataloader, teacher_eval_dataloader = get_distillation_dataloader(data_config, train_config, distil_config, student_tokenizer, teacher_tokenizer, rank)
    # 如果不使用蒸馏
    else:
        # 获取训练和验证数据加载器
        train_dataloader, eval_dataloader = get_dataloader(data_config, train_config, tokenizer, rank)

    # 获取优化器
    optimizer = get_optimizer(model, train_config, fsdp_config)
    # 创建OneCycle学习率调度器
    # OneCycle策略：学习率先上升后下降，有助于快速收敛
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=train_config.lr, epochs=train_config.num_epochs, steps_per_epoch=len(train_dataloader),
                                                    pct_start=train_config.pct_start, div_factor=train_config.div_factor, final_div_factor=train_config.final_div_factor)

    # 执行训练
    results = train(
        model,  # 学生模型
        train_dataloader,  # 训练数据加载器
        eval_dataloader,  # 验证数据加载器
        optimizer,  # 优化器
        scheduler,  # 学习率调度器
        train_config.gradient_accumulation_steps,  # 梯度累积步数
        train_config,  # 训练配置
        distil_config,  # 蒸馏配置
        data_config,  # 数据配置
        teacher_train_dataloader if train_config.distillation else None,  # 教师训练数据加载器（仅蒸馏时使用）
        teacher_eval_dataloader if train_config.distillation else None,  # 教师验证数据加载器（仅蒸馏时使用）
        fsdp_config if train_config.enable_fsdp else None,  # FSDP配置（仅分布式时使用）
        local_rank if train_config.enable_fsdp or distil_config.enable_fsdp else None,  # 本地排名（仅分布式时使用）
        rank,  # 全局排名
    )
    # 主进程打印训练结果
    if rank == 0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]


# 脚本入口点
if __name__ == "__main__":
    # 使用fire库自动将main函数转换为命令行接口
    # 可以通过命令行参数覆盖配置，例如：python finetuning.py --batch_size_training=16
    fire.Fire(main)
