# 导入操作系统模块
import os
# 导入时间模块，用于计时
import time
# 导入PyTorch深度学习框架
import torch
# 导入Weights & Biases实验跟踪工具
import wandb
# 导入PyTorch分布式训练模块
import torch.distributed as dist

# 从tqdm导入进度条功能
from tqdm import tqdm
# 从contextlib导入nullcontext，用于条件性上下文管理
from contextlib import nullcontext
# 从models.memory导入内存跟踪工具
from models.memory import MemoryTrace
# 从train.tools导入GPU缓存清理函数
from train.tools import clear_gpu_cache
# 从train.evaluations导入评估函数
from train.evaluations import evaluation
# 从train.save导入保存训练参数和模型的函数
from train.save import save_train_params, save_model
# 从FSDP导入分片梯度缩放器（用于混合精度训练）
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
# 从蒸馏模型模块导入蒸馏损失和批次预处理函数
from models.distillation_model import DistillationLoss, preprocess_distillation_batch

# 定义函数：执行模型训练的主循环
def train(model, train_dataloader, eval_dataloader, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, distil_config, dataset_config, teacher_train_dataloader=None, teacher_eval_dataloader=None, fsdp_config=None, local_rank=None, rank=None):
    """执行模型训练的主循环，支持常规训练和知识蒸馏
    参数:
        model: 要训练的模型（可能是单个模型或包含学生和教师的蒸馏模型）
        train_dataloader: 训练数据加载器
        eval_dataloader: 验证数据加载器
        optimizer: 优化器
        lr_scheduler: 学习率调度器
        gradient_accumulation_steps: 梯度累积步数
        train_config: 训练配置
        distil_config: 蒸馏配置
        dataset_config: 数据集配置
        teacher_train_dataloader: 教师模型的训练数据加载器（蒸馏时使用）
        teacher_eval_dataloader: 教师模型的验证数据加载器（蒸馏时使用）
        fsdp_config: FSDP配置
        local_rank: 本地GPU排名
        rank: 全局进程排名
    返回:
        results: 包含训练统计信息的字典"""
    # 初始化Weights & Biases实验跟踪系统
    # 设置服务等待时间为300秒
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    # 只在主进程（rank 0）初始化wandb
    if rank == 0:
        wandb.init(
            # 项目名称：llm_distillation_{数据集名称}
            project=f"llm_distillation_{dataset_config.file.split('/')[-1][:-3]}",
            # 运行名称：如果是蒸馏则包含学生-教师-蒸馏参数，否则只包含模型名称
            name=f"{train_config.model_name.split('/')[-1]}-{model.teacher.name_or_path.split('/')[-1]}-d{distil_config.distil_factor}-t{distil_config.teacher_temperature}{distil_config.student_temperature}" if train_config.distillation else f"{train_config.model_name.split('/')[-1]}",
            # 配置字典：记录所有训练超参数
            config={
                "model_name": train_config.model_name.split('/')[-1],  # 模型名称
                "dataset": dataset_config.file.split('/')[-1],  # 数据集名称
                "batch_size_training": train_config.batch_size_training,  # 训练批次大小
                "val_batch_size": train_config.val_batch_size,  # 验证批次大小
                "gradient_accumulation_steps": train_config.gradient_accumulation_steps,  # 梯度累积步数
                "num_epochs": train_config.num_epochs,  # 训练轮数
                "lr": train_config.lr,  # 学习率
                "weight_decay": train_config.weight_decay,  # 权重衰减
                "pct_start": train_config.pct_start,  # OneCycle学习率调度器的上升阶段比例
                "div_factor": train_config.div_factor,  # 初始学习率除数
                "final_div_factor": train_config.final_div_factor,  # 最终学习率除数
                "seed": train_config.seed,  # 随机种子
                "use_fp16": train_config.use_fp16,  # 是否使用FP16混合精度
                "mixed_precision": train_config.mixed_precision,  # 混合精度策略
                "peft_method": train_config.peft_method,  # PEFT方法（LoRA等）
                "use_peft": train_config.use_peft,  # 是否使用PEFT
                "freeze_layers": train_config.freeze_layers,  # 是否冻结层
                "num_freeze_layers": train_config.num_freeze_layers,  # 冻结层数
                "quantization": train_config.quantization,  # 量化策略
                # 蒸馏相关参数（如果启用蒸馏）
                "cross_entropy_factor": distil_config.cross_entropy_factor if train_config.distillation else -1,  # 交叉熵损失权重
                "distil_factor": distil_config.distil_factor if train_config.distillation else -1,  # 蒸馏损失权重
                "student_temperature": distil_config.student_temperature if train_config.distillation else -1,  # 学生温度
                "teacher_temperature": distil_config.teacher_temperature if train_config.distillation else -1  # 教师温度
            }
        )

    # 如果启用了蒸馏，初始化蒸馏损失计算器
    if train_config.distillation:
        distillation_loss = DistillationLoss(distillation_weight=distil_config.distil_factor, student_temperature=distil_config.student_temperature, teacher_temperature=distil_config.teacher_temperature, skip_student_eos=True, debug=True, debug_rank=0, tokenizer_student=model.student.name_or_path, tokenizer_teacher=model.teacher.name_or_path)

    # 为FP16混合精度训练创建梯度缩放器
    if train_config.use_fp16 and train_config.enable_fsdp:
        # FSDP模式下使用分片梯度缩放器
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        # 非FSDP模式下使用标准梯度缩放器
        scaler = torch.cuda.amp.GradScaler()
    # 如果启用了FSDP，获取总进程数
    if train_config.enable_fsdp or distil_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    # 设置自动混合精度上下文：如果使用FP16则使用autocast，否则使用nullcontext（无操作）
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext

    # 初始化训练统计列表
    train_prep = []  # 训练困惑度列表
    train_loss = []  # 训练损失列表
    val_ppl = []  # 验证困惑度列表
    val_loss = []  # 验证损失列表
    epoch_times = []  # 每轮训练时间列表
    checkpoint_times = []  # 检查点保存时间列表
    results = {}  # 最终结果字典
    steps_per_eval = len(eval_dataloader)  # 每次评估的步数
    steps_per_epoch = len(train_dataloader)  # 每轮训练的步数
    best_val_loss = float("inf")  # 最佳验证损失（初始化为无穷大）
    # 开始训练循环，遍历每个epoch
    for epoch in range(train_config.num_epochs):
        # 记录epoch开始时间
        epoch_start_time = time.perf_counter()
        # 计算总的更新步数（考虑梯度累积）
        total_length = steps_per_epoch//gradient_accumulation_steps
        # 将模型设置为训练模式（如果是蒸馏，只训练学生模型）
        model.student.train() if train_config.distillation else model.train()
        # 使用内存跟踪器监控内存使用
        with MemoryTrace() as memtrace:
            # 初始化总损失
            total_loss = 0.0
            # 创建蓝色进度条
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            # 遍历训练数据
            # 如果是蒸馏，同时遍历学生和教师的数据加载器
            for step, batch in enumerate(train_dataloader if not train_config.distillation else zip(train_dataloader, teacher_train_dataloader)):
                # 如果是蒸馏，预处理批次数据（合并学生和教师的批次）
                if train_config.distillation: batch = preprocess_distillation_batch(batch)
                # 将批次中的所有张量移动到指定设备
                for key in batch.keys():
                    if train_config.enable_fsdp or distil_config.enable_fsdp:
                        # FSDP模式：移动到本地GPU
                        batch[key] = batch[key].to(local_rank)
                    else:
                        # 非FSDP模式：移动到第一个GPU
                        batch[key] = batch[key].to('cuda:0')

                # 在自动混合精度上下文中执行前向传播
                with autocast():
                    if train_config.distillation:
                        # 蒸馏模式：获取学生和教师的输出
                        student_output, teacher_output = model(**batch)
                        # 计算蒸馏损失（总损失、交叉熵损失、蒸馏损失）
                        loss, cross_loss, dist_loss = distillation_loss(student_output, teacher_output, batch['student_labels'], batch['teacher_labels'], rank=rank)
                    else:
                        # 常规训练：只计算模型损失
                        loss = model(**batch).loss

                # 将损失除以梯度累积步数（实现梯度累积）
                loss = loss / gradient_accumulation_steps
                # 累加损失（分离梯度并转换为float）
                total_loss += loss.detach().float()
                # 如果使用FP16混合精度
                if train_config.use_fp16:
                    # 使用梯度缩放器进行反向传播
                    scaler.scale(loss).backward()
                    # 如果达到梯度累积步数或是最后一步，执行优化器更新
                    if (step + 1) % gradient_accumulation_steps == 0 or step == steps_per_epoch - 1:
                        scaler.step(optimizer)  # 执行优化步骤
                        scaler.update()  # 更新缩放因子
                        optimizer.zero_grad()  # 清零梯度
                        pbar.update()  # 更新进度条
                # 如果不使用FP16
                else:
                    # 标准反向传播
                    loss.backward()
                    # 如果达到梯度累积步数或是最后一步，执行优化器更新
                    if (step + 1) % gradient_accumulation_steps == 0 or step == steps_per_epoch - 1:
                        optimizer.step()  # 执行优化步骤
                        optimizer.zero_grad()  # 清零梯度
                        pbar.update()  # 更新进度条

                # 只在主进程记录到wandb
                if rank == 0:
                    if train_config.distillation:
                        # 蒸馏模式：记录多个损失和学习率
                        wandb.log({
                            "train_loss": loss.detach().float(),  # 总训练损失
                            "cross_loss": cross_loss.detach().float(),  # 交叉熵损失
                            "distil_loss": dist_loss.detach().float(),  # 蒸馏损失
                            "teacher_loss": teacher_output.loss.detach().float(),  # 教师损失
                            "lr": optimizer.param_groups[0]['lr']  # 当前学习率
                        })
                    else:
                        # 常规训练：只记录训练损失和学习率
                        wandb.log({
                            "train_loss": loss.detach().float(),
                            "lr": optimizer.param_groups[0]['lr']
                        })

                # 更新学习率（OneCycle调度器）
                lr_scheduler.step()
                # 更新进度条描述，显示当前损失
                pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{steps_per_epoch} completed (loss: {loss.detach().float()})")

                # 如果启用了验证，并且达到保存步数或是最后一步
                if train_config.run_validation and ((step+1) % train_config.save_step == 0 or step+1 == steps_per_epoch):
                    if rank == 0: print("Running evaluation...")
                    # 将模型设置为评估模式
                    model.eval()
                    # 执行评估
                    eval_ppl, eval_epoch_loss, eval_cross_loss, eval_dist_loss = evaluation(
                        model, train_config, distil_config,
                        # 如果是蒸馏，同时使用学生和教师的验证数据
                        eval_dataloader if not train_config.distillation else zip(eval_dataloader, teacher_eval_dataloader),
                        steps_per_eval, local_rank)
                    # 将模型设置回训练模式
                    model.student.train() if train_config.distillation else model.train()
                    # 记录验证损失和困惑度
                    val_loss.append(eval_epoch_loss)
                    val_ppl.append(eval_ppl)

                    # 只在主进程打印和记录
                    if rank == 0:
                        print(f"Perplexity {eval_ppl}, loss {eval_epoch_loss}")
                        if train_config.distillation:
                            # 蒸馏模式：记录多个评估指标
                            wandb.log({
                                "eval_ppl": eval_ppl,
                                "eval_epoch_loss": eval_epoch_loss,
                                "eval_cross_loss": eval_cross_loss,
                                "eval_dist_loss": eval_dist_loss
                            })
                        else:
                            # 常规训练：只记录困惑度和损失
                            wandb.log({
                                "eval_ppl": eval_ppl,
                                "eval_epoch_loss": eval_epoch_loss,
                            })

                    # 如果验证损失改善或配置为保存所有检查点
                    if eval_epoch_loss < best_val_loss or train_config.save_all:
                        # 如果验证损失改善，更新最佳验证损失
                        if eval_epoch_loss < best_val_loss:
                            best_val_loss = eval_epoch_loss
                            if rank == 0:
                                print(f"best eval loss is {best_val_loss}")
                        # 如果配置为保存模型
                        if train_config.save_model:
                            # 记录检查点保存开始时间
                            checkpoint_start_time = time.perf_counter()
                            # 保存模型（如果是蒸馏，只保存学生模型）
                            save_model(
                                model if not train_config.distillation else model.student,
                                optimizer, ((steps_per_epoch*epoch)+step), train_config, distil_config, fsdp_config, rank
                            )
                            # 计算检查点保存时间
                            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                            checkpoint_times.append(checkpoint_end_time)
                    # 清理GPU缓存
                    clear_gpu_cache(rank)
            # 关闭进度条
            pbar.close()

        # 只在主进程打印内存跟踪信息
        if rank == 0: print(memtrace)
        # 计算epoch时间
        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)

        # 如果有多个GPU且启用了FSDP，同步所有进程的总损失
        if torch.cuda.device_count() > 1 and train_config.enable_fsdp or distil_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        # 计算平均训练损失
        train_epoch_loss = total_loss / steps_per_epoch
        # 如果启用了FSDP，再除以进程数
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        # 计算训练困惑度（perplexity = e^loss）
        train_perplexity = torch.exp(train_epoch_loss)

        # 记录训练困惑度和损失
        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)

        # 只在主进程打印和记录epoch统计信息
        if rank == 0:
            print(
                f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
            wandb.log({
                "train_perplexity": train_perplexity,
                "train_epoch_loss": train_epoch_loss,
                "train_epoch_time": epoch_end_time
            })

    # 计算平均统计信息
    avg_epoch_time = sum(epoch_times) / len(epoch_times)  # 平均epoch时间
    avg_checkpoint_time = sum(
        checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0  # 平均检查点保存时间
    avg_train_prep = sum(train_prep)/len(train_prep)  # 平均训练困惑度
    avg_train_loss = sum(train_loss)/len(train_loss)  # 平均训练损失
    # 如果运行了验证，计算平均验证统计信息
    if train_config.run_validation:
        avg_eval_prep = sum(val_ppl)/len(val_ppl)  # 平均验证困惑度
        avg_eval_loss = sum(val_loss)/len(val_loss)  # 平均验证损失

    # 将统计信息存入结果字典
    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    # 如果启用了FSDP且不使用PEFT，保存训练参数
    if train_config.enable_fsdp and not train_config.use_peft:
        save_train_params(train_config, fsdp_config, rank)

    # 返回训练结果
    return results
