# 导入时间模块
import time
# 导入PyTorch深度学习框架
import torch

# 从pathlib导入Path类，用于路径操作
from pathlib import Path
# 从datetime导入日期时间类
from datetime import datetime
# 导入PyTorch分布式训练模块
import torch.distributed as dist
# 导入PyTorch分布式检查点模块
import torch.distributed._shard.checkpoint as dist_cp
# 从分布式检查点导入文件系统读取器
from torch.distributed._shard.checkpoint import FileSystemReader
# 从分布式检查点导入默认保存规划器
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
# 从FSDP导入状态字典类型
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

# 从FSDP导入完全分片数据并行、状态字典类型和完整状态字典配置
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)


# 定义函数：获取当前运行的日期和时间
def get_date_of_run():
    """创建日期和时间以确保文件保存的唯一性
    示例：'2022-05-07-08:31:12_PM'
    返回:
        date_of_run: 格式化的日期时间字符串"""
    # 获取当前日期时间并格式化
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"--> current date and time of run = {date_of_run}")
    return date_of_run


# 定义完整状态保存策略：卸载到CPU，只在rank0保存
fullstate_save_policy = FullStateDictConfig(
    offload_to_cpu=True, rank0_only=True)


# 定义函数：加载分片模型
def load_model_sharded(model, rank, cfg):
    """从分片检查点加载模型
    参数:
        model: 要加载的模型
        rank: 进程排名
        cfg: 配置对象
    """
    # 构建检查点文件夹名称
    folder_name = (
        cfg.dist_checkpoint_root_folder
        + "/"
        + cfg.dist_checkpoint_folder
        + "-"
        + cfg.model_name
    )

    # 构建加载目录路径
    load_dir = Path.cwd() / folder_name

    # 如果目录不存在
    if not load_dir.exists():
        if rank == 0:
            print(f"No sharded_state_dict checkpoint directory found...skipping")
        return
    # 只在主进程打印加载信息
    if rank == 0:
        print(f"loading model from model path: {load_dir} ")
    # 创建文件系统读取器
    reader = FileSystemReader(load_dir)

    # 使用分片状态字典类型
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        # 创建检查点字典
        checkpoint = {"model": model.state_dict()}
        if rank == 0:
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")

        # 加载状态字典
        dist_cp.load_state_dict(
            state_dict=checkpoint,
            storage_reader=reader,
        )
        if rank == 0:
            print(f"checkpoint after load_state_dict()")
            ck = checkpoint.keys()
            print(f" checkpoint key len = {len(ck)} and \n keys =  {ck}")
        # 将检查点加载到模型
        model.load_state_dict(checkpoint["model"])
    if rank == 0:
        print(f"Sharded state checkpoint loaded from {load_dir}")


# 定义函数：保存分片模型和优化器
def save_model_and_optimizer_sharded(model, rank, path, optim=None):
    """通过分片状态字典保存模型和优化器到save_dir
    参数:
        model: 要保存的模型
        rank: 进程排名
        path: 保存路径
        optim: 优化器（可选）
    """
    if rank == 0:
        print(f"Saving model to {path}")

    # 创建分布式文件系统写入器
    distributed_writer = dist_cp.FileSystemWriter(path)
    # 记录开始时间
    t0 = time.perf_counter()

    # 使用分片状态字典类型
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
        # 创建状态字典
        state_dict = {"model": model.state_dict()}
        # 如果提供了优化器，也保存优化器状态
        if optim is not None:
            state_dict["optim"] = FSDP.optim_state_dict(model, optim)

        # 保存状态字典
        dist_cp.save_state_dict(
            state_dict=state_dict,
            storage_writer=distributed_writer,
            planner=DefaultSavePlanner(),

        )
    # 等待所有进程完成
    dist.barrier()
    # 记录结束时间
    t1 = time.perf_counter()
    if rank == 0:
        print(f"Sharded state checkpoint saved to {path}")
        print(
            f"Checkpoint Time = {t1-t0:.4f}\n"
        )


# 定义函数：保存模型检查点
def save_model_checkpoint(model, rank, path):
    """通过rank0 CPU流式传输和完整状态字典保存模型
    参数:
        model: 要保存的模型
        rank: 进程排名
        path: 保存路径
    """
    # 使用完整状态字典类型和完整状态保存策略
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, fullstate_save_policy):
        # 获取CPU上的状态字典
        cpu_state = model.state_dict()
        print(f"saving process: rank {rank} with model state_dict\n")

    # 只在主进程保存
    if rank == 0:
        print(f"--> saving model ...")
        # 保存到磁盘
        torch.save(cpu_state, path+".pt")
        print(f"model checkpoint saved at {path}\n")


# 定义函数：加载模型检查点
def load_model_checkpoint(model, rank, cfg):
    """将本地检查点加载到rank0 CPU
    必须在传递给FSDP之前调用
    参数:
        model: 要加载的模型
        rank: 进程排名
        cfg: 配置对象
    """

    # 只在主进程加载
    if rank != 0:
        return

    # 检查点在哪里...
    full_state_dict_model_path = (
        Path.cwd() / cfg.checkpoint_folder / cfg.checkpoint_model_filename
    )
    # 检查点是否存在...
    if not full_state_dict_model_path.is_file():
        print(
            f"model checkpoint {full_state_dict_model_path} not present. Returning..."
        )
        return

    # 加载模型检查点
    model_checkpoint = torch.load(full_state_dict_model_path)
    # 集成到加载的模型中
    model.load_state_dict(model_checkpoint)

    print(f"model checkpoint loaded to rank0 cpu")


# 定义函数：保存优化器检查点
def save_optimizer_checkpoint(model, optimizer, rank, cfg, epoch=1):
    """通过完整状态字典保存优化器状态
    参数:
        model: 模型
        optimizer: 优化器
        rank: 进程排名
        cfg: 配置对象
        epoch: 当前轮数（默认为1）
    """

    print(f"--> optim state call on rank {rank}\n")

    # 将所有分片的优化器状态拉取到rank0 CPU...
    optim_state = FSDP.full_optim_state_dict(model, optimizer)

    print(f"optim state dict ready on {rank} and len of {len(optim_state)}\n")

    # 只在主进程保存
    if rank == 0:
        # 构建文件夹名称
        folder_name = (
            cfg.dist_checkpoint_root_folder
            + "/"
            + cfg.dist_checkpoint_folder
            + "-"
            + cfg.model_name
        )
        # 创建保存目录
        save_dir = Path.cwd() / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # 构建优化器保存文件名
        opt_save_name = (
            "optimizer" + "-" + cfg.model_name + "-" + str(epoch) + ".pt"
        )
        opt_save_full_path = save_dir / opt_save_name

        print(f"--> saving optimizer state...")

        # 保存优化器状态到磁盘
        torch.save(optim_state, opt_save_full_path)

        print(f"--> saved {opt_save_full_path} to disk")


# 定义函数：加载优化器检查点
def load_optimizer_checkpoint(model, optimizer_checkpoint_path, rank):
    """使用scatter方法加载FSDP优化器完整状态检查点
    这确保只有rank0加载优化器状态字典并分散到其他rank
    参数:
        model: 模型
        optimizer_checkpoint_path: 优化器检查点路径
        rank: 进程排名
    """

    # 检查优化器检查点是否存在
    if not optimizer_checkpoint_path.is_file():
        print(
            f"warning - optimizer checkpoint not present {optimizer_checkpoint_path}. Returning. "
        )
        return

    # 初始化完整优化器状态字典为None
    full_osd = None

    # 只在主进程加载
    if rank == 0:
        full_osd = torch.load(optimizer_checkpoint_path)

    # 从所有rank调用，尽管只有rank0有有效的full_osd参数
    # 将完整优化器状态字典分散到各个rank
    sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)

    print(f"optimizer shard loaded on rank {rank}")


# 定义函数：在单个GPU上加载分片模型
def load_sharded_model_single_gpu(model, model_path):
    """在单个GPU上加载分片模型检查点
    参数:
        model: 要加载的模型
        model_path: 模型检查点路径
    返回:
        model: 加载了权重的模型
    """

    # 创建文件系统读取器
    reader = FileSystemReader(model_path)

    # 创建状态字典
    state_dict = {
        "model": model.state_dict()
    }

    # 加载状态字典（非分布式模式）
    dist_cp.load_state_dict(
        state_dict=state_dict,
        storage_reader=FileSystemReader(model_path),
        no_dist=True,  # 非分布式模式
    )

    # 将状态字典加载到模型
    model.load_state_dict(state_dict["model"])

    print(f"Sharded state checkpoint loaded from {model_path}")
    return model
