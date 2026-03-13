# 导入操作系统模块
import os
# 导入YAML配置文件处理模块
import yaml
# 导入PyTorch分布式训练模块
import torch.distributed as dist

# 从pathlib导入Path类，用于路径操作
from pathlib import Path
# 从PyTorch FSDP导入状态字典类型枚举
from torch.distributed.fsdp import StateDictType
# 从模型检查点处理模块导入保存函数
from models.checkpoint_handler import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint


# 定义函数：保存模型检查点
def save_model(model, optimizer, step, train_config, distil_config, fsdp_config, rank):
    """保存模型和优化器的检查点"""
    # 如果启用了FSDP分布式训练，等待所有进程同步
    if train_config.enable_fsdp or distil_config.enable_fsdp:
        dist.barrier()
    # 构建保存路径，使用步数+1作为目录名
    path = fr"{train_config.output_dir}/{step+1}"
    # 尝试创建目录，如果已存在则忽略错误
    try: os.mkdir(path)
    except: pass

    # 如果使用PEFT（参数高效微调）
    if train_config.use_peft:
        # 主进程打印提示信息
        if rank == 0: print(f"We are about to save the PEFT modules")
        # 保存PEFT模块（只保存可训练的适配器参数）
        model.save_pretrained(path)
        # 主进程打印保存完成信息
        if rank == 0: print(f"PEFT modules are saved in {path} directory")

    # 如果启用了FSDP分布式训练
    elif train_config.enable_fsdp:
        # 如果检查点类型是完整状态字典
        if fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
            print("Saving the FSDP model checkpoints using FULL_STATE_DICT")
            # 保存完整的模型检查点（所有参数合并到一个文件）
            save_model_checkpoint(model, optimizer, rank, path)

        # 如果检查点类型是分片状态字典
        elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
            # 如果需要保存优化器状态
            if train_config.save_optimizer:
                print("Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                # 保存分片的模型和优化器检查点
                save_model_and_optimizer_sharded(model, rank, path, optim=optimizer)
            else:
                print("Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                # 只保存分片的模型检查点
                save_model_and_optimizer_sharded(model, rank, path)

    # 如果不使用FSDP（单GPU训练）
    else:
        # 主进程保存模型
        if rank == 0:
            print(f"We are about to save the model")
            # 使用Hugging Face的save_pretrained方法保存模型
            model.save_pretrained(path)
            print(f"Model are saved in {path} directory")

    # 如果启用了FSDP，等待所有进程完成保存
    if train_config.enable_fsdp or distil_config.enable_fsdp:
        dist.barrier()

# 定义函数：保存训练参数配置
def save_train_params(train_config, fsdp_config, rank):
    """
    将train_config和FSDP配置保存到train_params.yaml文件
    这将被推理文件夹中的转换脚本用于获取HF模型名称或路径
    也可以作为日志供将来参考
    """
    # 将train_config对象转换为字典
    # 将所有值转换为字符串以确保可以序列化为YAML文件
    train_config_dict = {k: str(v) for k, v in vars(
        train_config).items() if not k.startswith('__')}
    # 将fsdp_config对象转换为字典
    fsdp_config_dict = {k: str(v) for k, v in vars(
        fsdp_config).items() if not k.startswith('__')}
    # 合并两个字典为一个
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # 构建文件夹名称（遵循FSDP检查点样式）
    # 使用train_config对象的属性
    folder_name = (
        train_config.dist_checkpoint_root_folder
        + "/"
        + train_config.dist_checkpoint_folder
        + "-"
        + train_config.model_name
    )

    # 构建保存目录路径
    save_dir = Path.cwd() / folder_name
    # 如果目录不存在，创建它
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 将字典转换为YAML字符串
    config_yaml = yaml.dump(train_params_dict, indent=4)
    # 构建文件完整路径
    file_name = os.path.join(save_dir, 'train_params.yaml')

    # 检查是否存在同名目录（而不是文件）
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # 将YAML字符串写入文件
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        # 主进程打印保存完成信息
        if rank == 0:
            print(f"training params are saved in {file_name}")
