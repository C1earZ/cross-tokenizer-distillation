# 导入PyTorch深度学习框架
import torch
# 从PyTorch优化器模块导入基础优化器类
from torch.optim.optimizer import Optimizer

# 定义类：任意精度AdamW优化器
class AnyPrecisionAdamW(Optimizer):
    """支持任意精度的AdamW优化器
    允许使用不同的数据类型存储动量、方差和补偿缓冲区
    支持Kahan求和以提高数值精度"""
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        use_kahan_summation=False,
        momentum_dtype=torch.bfloat16,
        variance_dtype=torch.bfloat16,
        compensation_buffer_dtype=torch.bfloat16,
    ):
        """初始化任意精度AdamW优化器
        参数:
            params (iterable): 要优化的参数或定义参数组的字典
            lr (float, optional): 学习率（默认：1e-3）
            betas (Tuple[float, float], optional): 用于计算梯度及其平方的运行平均值的系数（默认：(0.9, 0.999)）
            eps (float, optional): 添加到分母以提高数值稳定性的项（默认：1e-8）
            weight_decay (float, optional): 权重衰减系数（默认：0.0）

            # 任意精度特定参数
            use_kahan_summation: 创建辅助缓冲区以确保高精度模型参数更新（默认：False）
            momentum_dtype: 动量的数据类型（默认：BFloat16）
            variance_dtype: 非中心方差的数据类型（默认：BFloat16）
            compensation_buffer_dtype: Kahan求和缓冲区的数据类型（默认：BFloat16）

            # 使用说明
            此优化器实现了优化器状态和Kahan求和以进行高精度更新，
            所有这些都使用用户控制的数据类型。
            默认设置是方差使用BF16，动量使用FP32。
            这可以在FSDP混合精度、AMP或全精度下运行，
            取决于您希望使用的训练管道。

            设置use_kahan_summation = False，并将动量和方差数据类型更改为FP32，
            将使其恢复为标准AdamW优化器。
        """
        # 设置默认参数
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            use_kahan_summation=use_kahan_summation,
            momentum_dtype=momentum_dtype,
            variance_dtype=variance_dtype,
            compensation_buffer_dtype=compensation_buffer_dtype,
        )

        # 调用父类初始化
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行单个优化步骤
        参数:
            closure (callable, optional): 重新评估模型并返回损失的闭包函数
        """

        # 如果提供了闭包函数
        if closure is not None:
            # 启用梯度计算
            with torch.enable_grad():
                # 执行闭包（目前不保留返回的损失）
                closure()

        # 遍历所有参数组
        for group in self.param_groups:

            # 获取beta参数
            beta1, beta2 = group["betas"]
            # 获取学习率
            lr = group["lr"]
            # 获取权重衰减
            weight_decay = group["weight_decay"]
            # 获取epsilon
            eps = group["eps"]
            # 获取是否使用Kahan求和
            use_kahan_summation = group["use_kahan_summation"]

            # 获取数据类型设置
            momentum_dtype = group["momentum_dtype"]
            variance_dtype = group["variance_dtype"]
            compensation_buffer_dtype = group["compensation_buffer_dtype"]

            # 遍历参数组中的所有参数
            for p in group["params"]:
                # 如果参数没有梯度，跳过
                if p.grad is None:
                    continue

                # 如果梯度是稀疏的，抛出错误
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "AnyPrecisionAdamW does not support sparse gradients"
                    )

                # 获取参数的状态
                state = self.state[p]

                # 状态初始化（第一次更新时）
                if len(state) == 0:

                    # 初始化步数为0
                    state["step"] = torch.tensor(0.0)

                    # 动量 - 梯度值的指数移动平均（EMA）
                    state["exp_avg"] = torch.zeros_like(
                        p,
                        dtype=momentum_dtype,
                    )

                    # 非中心方差 - 梯度平方值的指数移动平均（EMA）
                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        dtype=variance_dtype,
                    )

                    # 可选的Kahan求和 - 累积误差跟踪器
                    if use_kahan_summation:
                        state["compensation"] = torch.zeros_like(
                            p,
                            dtype=compensation_buffer_dtype,
                        )

                # 主要处理 -------------------------

                # 更新每个参数组的步数
                state["step"] += 1
                step = state["step"]

                # 获取动量和方差
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # 获取梯度
                grad = p.grad

                # 权重衰减，AdamW风格（解耦权重衰减）
                if weight_decay:
                    p.data.mul_(1 - lr * weight_decay)

                # 更新动量：exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 更新非中心方差：exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 使用bias1进行偏差校正
                bias_correction1 = 1 - beta1**step

                # 计算步长
                step_size = lr / bias_correction1

                # 使用bias2进行偏差校正
                denom_correction = (1 - beta2**step) ** 0.5  # 避免导入math模块

                # 计算中心化方差（加上epsilon以提高数值稳定性）
                centered_variance = (exp_avg_sq.sqrt() / denom_correction).add_(
                    eps, alpha=1
                )

                # 学习率更新到补偿缓冲区
                if use_kahan_summation:
                    # 获取补偿缓冲区
                    compensation = state["compensation"]

                    # 计算更新量并添加到补偿缓冲区
                    # compensation -= step_size * exp_avg / centered_variance
                    compensation.addcdiv_(exp_avg, centered_variance, value=-step_size)

                    # 使用补偿更新权重（Kahan求和）
                    # 将误差保存回补偿缓冲区以供下次迭代使用
                    temp_buffer = p.detach().clone()
                    p.data.add_(compensation)
                    compensation.add_(temp_buffer.sub_(p.data))

                else:
                    # 常规AdamW更新
                    # p = p - step_size * exp_avg / centered_variance
                    p.data.addcdiv_(exp_avg, centered_variance, value=-step_size)
