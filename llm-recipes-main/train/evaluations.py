# 导入操作系统模块
import os
# 导入PyTorch深度学习框架
import torch
# 导入PyTorch分布式训练模块
import torch.distributed as dist

# 从tqdm导入进度条功能
from tqdm import tqdm
# 从蒸馏模型模块导入蒸馏损失和批次预处理函数
from models.distillation_model import DistillationLoss, preprocess_distillation_batch

# 定义函数：在验证集上评估模型性能
def evaluation(model, train_config, distil_config, eval_dataloader, steps_per_eval, local_rank):
    """在验证集上评估模型，计算损失和困惑度
    参数:
        model: 要评估的模型
        train_config: 训练配置
        distil_config: 蒸馏配置
        eval_dataloader: 验证数据加载器
        steps_per_eval: 评估步数
        local_rank: 本地GPU排名
    返回:
        eval_ppl: 困惑度
        eval_loss: 总损失
        eval_cross_loss: 交叉熵损失
        eval_dist_loss: 蒸馏损失"""
    # 如果启用了FSDP分布式训练，获取总进程数
    if train_config.enable_fsdp or distil_config.enable_fsdp: world_size = int(os.environ["WORLD_SIZE"])
    # 如果启用了蒸馏，创建蒸馏损失计算器（跳过学生模型的EOS token）
    if train_config.distillation: distillation_loss = DistillationLoss(skip_student_eos=True)
    # 初始化评估损失为0
    eval_loss = 0.0
    # 初始化交叉熵损失为0
    eval_cross_loss = 0.0
    # 初始化蒸馏损失为0
    eval_dist_loss = 0.0
    # 创建绿色进度条，显示评估进度
    pbar = tqdm(colour="green", desc="Evaluating", total=steps_per_eval, dynamic_ncols=True)
    # 使用torch.no_grad()禁用梯度计算，节省内存
    with torch.no_grad():
        # 遍历验证数据加载器中的每个批次
        for _, batch in enumerate(eval_dataloader):
            # 如果启用了蒸馏，预处理批次数据
            if train_config.distillation:
                batch = preprocess_distillation_batch(batch)
            # 将批次中的所有张量移动到指定设备
            for key in batch.keys():
                # 如果启用了FSDP，移动到本地GPU
                if train_config.enable_fsdp or distil_config.enable_fsdp:
                    batch[key] = batch[key].to(local_rank)
                # 否则移动到第一个GPU
                else:
                    batch[key] = batch[key].to('cuda:0')

            # 如果启用了蒸馏
            if train_config.distillation:
                # 前向传播，获取学生和教师模型的输出
                outputs, teacher_output = model(**batch)
                # 计算蒸馏损失（总损失、交叉熵损失、蒸馏损失）
                loss, cross_loss, dist_loss = distillation_loss(outputs, teacher_output, batch['student_labels'], batch['teacher_labels'])
                # 累加交叉熵损失（分离梯度并转换为float）
                eval_cross_loss += cross_loss.detach().float()
                # 累加蒸馏损失
                eval_dist_loss += dist_loss.detach().float()
            # 如果不使用蒸馏（常规训练）
            else:
                # 前向传播
                outputs = model(**batch)
                # 获取损失
                loss = outputs.loss
            # 累加总损失
            eval_loss += loss.detach().float()
            # 更新进度条
            pbar.update()

    # 如果有多个GPU且启用了FSDP，同步所有进程的损失
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp or distil_config.enable_fsdp:
        # 对总损失进行all_reduce求和操作
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        # 如果启用了蒸馏，也同步蒸馏相关的损失
        if train_config.distillation:
            dist.all_reduce(eval_cross_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(eval_dist_loss, op=dist.ReduceOp.SUM)

    # 计算平均损失（除以评估步数）
    eval_loss /= steps_per_eval
    eval_cross_loss /= steps_per_eval
    eval_dist_loss /= steps_per_eval
    # 如果启用了FSDP，再除以进程数得到真正的平均值
    if train_config.enable_fsdp or distil_config.enable_fsdp:
        eval_loss /= world_size
        eval_cross_loss /= world_size
        eval_dist_loss /= world_size
    # 计算困惑度（perplexity）= e^loss
    # 如果使用蒸馏，基于交叉熵损失计算；否则基于总损失
    eval_ppl = torch.exp(eval_cross_loss if train_config.distillation else eval_loss)

    # 返回困惑度和各项损失
    return eval_ppl, eval_loss, eval_cross_loss, eval_dist_loss
