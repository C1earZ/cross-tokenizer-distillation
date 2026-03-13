# 导入垃圾回收模块，用于手动触发内存清理
import gc
# 导入PyTorch深度学习框架
import torch
# 导入psutil系统监控库，用于获取CPU内存使用情况
import psutil
# 导入threading线程模块，用于创建后台监控线程
import threading

# 定义函数：将字节转换为MB
def byte2mb(x):
    # 除以2^20（1048576）得到MB
    return int(x / 2**20)

# 定义函数：将字节转换为GB
def byte2gb(x):
    # 除以2^30（1073741824）得到GB
    return int(x / 2**30)

# 定义内存追踪上下文管理器类
# 用于追踪训练过程中的峰值内存使用情况
class MemoryTrace:
    # 进入上下文时执行
    def __enter__(self):
        # 执行垃圾回收，清理未使用的Python对象
        gc.collect()
        # 清空CUDA缓存，释放未使用的GPU内存
        torch.cuda.empty_cache()
        # 重置CUDA最大内存分配记录
        torch.cuda.reset_max_memory_allocated()
        # 记录开始时的GPU内存使用量（GB）
        self.begin = byte2gb(torch.cuda.memory_allocated())
        # 获取当前进程对象
        self.process = psutil.Process()
        # 记录开始时的CPU内存使用量（GB）
        self.cpu_begin = byte2gb(self.cpu_mem_used())
        # 设置峰值监控标志为True
        self.peak_monitoring = True
        # 创建峰值监控线程
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        # 设置为守护线程，主线程结束时自动退出
        peak_monitor_thread.daemon = True
        # 启动监控线程
        peak_monitor_thread.start()
        # 返回self以支持with语句
        return self

    # 定义方法：获取当前进程的CPU内存使用量
    def cpu_mem_used(self):
        """获取当前进程的常驻集大小（RSS）内存"""
        # 返回进程的RSS内存（实际占用的物理内存）
        return self.process.memory_info().rss

    # 定义方法：峰值监控函数（在后台线程中运行）
    def peak_monitor_func(self):
        # 初始化CPU峰值内存为-1
        self.cpu_peak = -1
        # 持续监控循环
        while True:
            # 更新CPU峰值内存（取当前值和历史峰值的最大值）
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)
            # 如果监控标志为False，退出循环
            if not self.peak_monitoring:
                break

    # 定义字符串表示方法，用于打印内存统计信息
    def __str__(self):
        # 返回格式化的内存使用统计信息
        return f"""
        Max CUDA memory allocated was {self.peak} GB
        Max CUDA memory reserved was {self.max_reserved} GB
        Peak active CUDA memory was {self.peak_active_gb} GB
        Cuda Malloc retires : {self.cuda_malloc_retries}
        CPU Total Peak Memory consumed during the train (max): {self.cpu_peaked + self.cpu_begin} GB
        """

    # 退出上下文时执行
    def __exit__(self, *exc):
        # 停止峰值监控
        self.peak_monitoring = False

        # 执行垃圾回收
        gc.collect()
        # 清空CUDA缓存
        torch.cuda.empty_cache()
        # 记录结束时的GPU内存使用量（GB）
        self.end = byte2gb(torch.cuda.memory_allocated())
        # 记录峰值GPU内存使用量（GB）
        self.peak = byte2gb(torch.cuda.max_memory_allocated())
        # 获取CUDA内存统计信息
        cuda_info = torch.cuda.memory_stats()
        # 记录活跃内存的峰值（GB）
        self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
        # 记录CUDA内存分配重试次数
        self.cuda_malloc_retries = cuda_info.get("num_alloc_retries", 0)
        # 再次记录活跃内存峰值（重复赋值）
        self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
        # 记录CUDA内存溢出次数
        self.m_cuda_ooms = cuda_info.get("num_ooms", 0)
        # 计算使用的GPU内存（结束-开始）
        self.used = byte2gb(self.end - self.begin)
        # 计算峰值GPU内存增量
        self.peaked = byte2gb(self.peak - self.begin)
        # 记录最大保留的GPU内存
        self.max_reserved = byte2gb(torch.cuda.max_memory_reserved())

        # 记录结束时的CPU内存使用量
        self.cpu_end = self.cpu_mem_used()
        # 计算使用的CPU内存（GB）
        self.cpu_used = byte2gb(self.cpu_end - self.cpu_begin)
        # 计算CPU峰值内存增量（GB）
        self.cpu_peaked = byte2gb(self.cpu_peak - self.cpu_begin)
