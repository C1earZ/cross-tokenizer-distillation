# 导入操作系统模块，用于设置环境变量
import os
# 导入PyTorch深度学习框架
import torch
# 导入PyTorch分布式训练模块
import torch.distributed as dist

# 定义函数：初始化分布式训练的进程组
def setup():
    """初始化分布式训练的进程组
    使用NCCL后端进行GPU间通信"""
    # 初始化进程组，使用NCCL（NVIDIA Collective Communications Library）作为后端
    # NCCL是专为NVIDIA GPU优化的通信库，提供高效的多GPU通信
    dist.init_process_group("nccl")

# 定义函数：设置用于调试的环境变量标志
def setup_environ_flags(rank):
    """设置用于调试目的的环境变量标志
    参数:
        rank: 当前进程的排名（在分布式训练中的编号）"""
    # 设置环境变量以显示C++堆栈跟踪信息，便于调试
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    # 设置NCCL异步错误处理，使错误能够及时被捕获
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # 只在主进程（rank 0）打印调试信息
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")

# 定义函数：清理分布式训练的进程组
def cleanup():
    """训练结束后清理进程组
    释放分布式训练占用的资源"""
    # 销毁进程组，释放通信资源
    dist.destroy_process_group()

# 定义函数：清空所有GPU的缓存
def clear_gpu_cache(rank=None):
    """清空所有进程的GPU缓存
    参数:
        rank: 当前进程的排名，默认为None"""
    # 只在主进程打印清理信息
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    # 清空CUDA缓存，释放未使用的GPU内存
    torch.cuda.empty_cache()
