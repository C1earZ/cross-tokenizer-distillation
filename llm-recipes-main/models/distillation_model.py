# 导入PyTorch深度学习框架
import torch
# 导入PyTorch神经网络模块
import torch.nn as nn
# 导入PyTorch函数式API
import torch.nn.functional as F

# 从transformers导入自动分词器
from transformers import AutoTokenizer

# 定义函数：预处理蒸馏批次数据
def preprocess_distillation_batch(batch):
    """将学生和教师的批次数据合并到一个字典中
    参数:
        batch: 包含学生和教师数据的批次元组 [student_batch, teacher_batch]
    返回:
        batch_dict: 合并后的字典，学生数据键名前缀为'student_'，教师数据键名前缀为'teacher_'"""
    # 为学生批次的所有键添加'student_'前缀
    batch_dict = {"student_" + key: value for key, value in batch[0].items()}
    # 为教师批次的所有键添加'teacher_'前缀并更新到字典中
    batch_dict.update({"teacher_" + key: value for key,
                      value in batch[1].items()})
    return batch_dict


# 定义类：知识蒸馏模型
class DistillationModel(nn.Module):
    """包装学生模型和教师模型的蒸馏模型类
    用于同时运行学生和教师模型的前向传播"""
    def __init__(self, student, teacher):
        """初始化蒸馏模型
        参数:
            student: 学生模型（需要训练的小模型）
            teacher: 教师模型（提供知识的大模型）"""
        super().__init__()
        # 保存学生模型
        self.student = student
        # 保存教师模型
        self.teacher = teacher
        # 将教师模型设置为评估模式（不更新参数）
        self.teacher.eval()
        # 冻结教师模型所有参数，不计算梯度也不占用优化器内存
        # 如果不冻结，优化器会为7B参数创建momentum和variance状态（额外~56GB显存）
        for param in self.teacher.parameters():
            param.requires_grad = False

    def forward(self, student_input_ids, student_attention_mask, student_labels, teacher_input_ids, teacher_attention_mask, teacher_labels):
        """前向传播：同时运行学生和教师模型
        参数:
            student_input_ids: 学生模型的输入token ID
            student_attention_mask: 学生模型的注意力掩码
            student_labels: 学生模型的标签
            teacher_input_ids: 教师模型的输入token ID
            teacher_attention_mask: 教师模型的注意力掩码
            teacher_labels: 教师模型的标签
        返回:
            (student_output, teacher_output): 学生和教师模型的输出元组"""
        # 禁用梯度计算（教师模型不需要反向传播）
        with torch.no_grad():
            # 运行教师模型前向传播
            teacher_output = self.teacher(
                input_ids=teacher_input_ids,
                attention_mask=teacher_attention_mask,
                labels=teacher_labels
            )

        # 运行学生模型前向传播（需要梯度）
        student_output = self.student(
            input_ids=student_input_ids,
            attention_mask=student_attention_mask,
            labels=student_labels
        )
        # 返回学生和教师的输出
        return student_output, teacher_output


# 定义类：知识蒸馏损失函数
class DistillationLoss(nn.Module):
    """实现跨分词器知识蒸馏的损失函数
    结合交叉熵损失和蒸馏损失，支持不同分词器的学生和教师模型"""
    def __init__(self, crossentropy_weight=1, distillation_weight=1, student_temperature=1, teacher_temperature=1, skip_student_eos=False, skip_teacher_eos=False, ignore_index=-100, debug=False, debug_rank=0, tokenizer_student=None, tokenizer_teacher=None):
        """初始化蒸馏损失函数
        参数:
            crossentropy_weight: 交叉熵损失的权重（默认：1）
            distillation_weight: 蒸馏损失的权重（默认：1）
            student_temperature: 学生模型的温度参数，用于软化概率分布（默认：1）
            teacher_temperature: 教师模型的温度参数，用于软化概率分布（默认：1）
            skip_student_eos: 是否跳过学生模型的EOS token（默认：False）
            skip_teacher_eos: 是否跳过教师模型的EOS token（默认：False）
            ignore_index: 计算损失时忽略的标签索引（默认：-100）
            debug: 是否启用调试模式，打印详细信息（默认：False）
            debug_rank: 在哪个进程rank上打印调试信息（默认：0）
            tokenizer_student: 学生模型的分词器名称（用于调试）
            tokenizer_teacher: 教师模型的分词器名称（用于调试）"""
        super().__init__()
        # 保存交叉熵损失权重
        self.crossentropy_weight = crossentropy_weight
        # 保存蒸馏损失权重
        self.distillation_weight = distillation_weight
        # 保存学生温度参数
        self.student_temperature = student_temperature
        # 保存教师温度参数
        self.teacher_temperature = teacher_temperature
        # 保存是否跳过学生EOS标志
        self.skip_student_eos = skip_student_eos
        # 保存是否跳过教师EOS标志
        self.skip_teacher_eos = skip_teacher_eos
        # 保存忽略索引
        self.ignore_index = ignore_index
        # 保存调试rank
        self.debug_rank = debug_rank
        # 保存调试标志
        self.debug = debug

        # 如果启用调试模式
        if self.debug:
            # 打印所有蒸馏损失参数
            print("Distillation loss parameters:")
            print(f"Crossentropy weight: {crossentropy_weight}")
            print(f"Distillation weight: {distillation_weight}")
            print(f"Student temperature: {student_temperature}")
            print(f"Teacher temperature: {teacher_temperature}")
            print(f"Skip student eos: {skip_student_eos}")
            print(f"Skip teacher eos: {skip_teacher_eos}")
            print(f"Ignore index: {ignore_index}")
            print(f"Debug: {debug}")
            print(f"Debug rank: {debug_rank}")

            # 加载学生和教师的分词器（用于调试时解码token）
            self.student_tokenizer = AutoTokenizer.from_pretrained(tokenizer_student)
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(tokenizer_teacher)

    def forward(self, student_predictions, teacher_predictions, student_targets, teacher_targets, rank=0):
        """计算蒸馏损失
        参数:
            student_predictions: 学生模型的预测输出（包含logits）
            teacher_predictions: 教师模型的预测输出（包含logits）
            student_targets: 学生模型的目标标签
            teacher_targets: 教师模型的目标标签
            rank: 当前进程的rank（用于调试）
        返回:
            (total_loss, crossentropy_loss, distillation_loss): 总损失、交叉熵损失、蒸馏损失的元组"""
        # 提取学生模型的logits（未归一化的预测分数）
        student = student_predictions.logits
        # 提取教师模型的logits
        teacher = teacher_predictions.logits

        # 获取答案的起始位置和长度
        # 这是为了只对答案部分计算蒸馏损失，忽略提示部分
        student_answer_index, student_answer_size = self.__get_start_and_size_answers(
            student_targets)
        teacher_answer_index, teacher_answer_size = self.__get_start_and_size_answers(
            teacher_targets)

        # 如果需要，避免包含EOS token（句子结束标记）
        # 将答案长度减1以排除EOS token
        if self.skip_student_eos: student_answer_size = [size-1 for size in student_answer_size]
        if self.skip_teacher_eos: teacher_answer_size = [size-1 for size in teacher_answer_size]

        # 对齐答案的起始token，右侧填充，并计算softmax
        # 遍历批次中的每个样本
        for i in range(student.size(0)):
            # 获取答案起始位置
            shift = student_answer_index[i]
            # 获取答案长度
            size = student_answer_size[i]
            # 计算答案结束位置
            end_shift = shift+size
            # 对学生模型的答案部分应用温度缩放的softmax，然后右侧填充零
            # 温度缩放：logits/temperature 使概率分布更平滑（温度>1）或更尖锐（温度<1）
            student[i] = torch.cat((
                torch.nn.functional.softmax(student[i, shift:end_shift, :]/self.student_temperature, dim=-1),
                torch.zeros_like(student[i, :(student.size(1)-size), :])), dim=0
            )
        # 对教师模型执行相同的操作
        for i in range(teacher.size(0)):
            shift = teacher_answer_index[i]
            size = teacher_answer_size[i]
            end_shift = shift+size
            # 对教师模型的答案部分应用温度缩放的softmax，然后右侧填充零
            teacher[i] = torch.cat((
                torch.nn.functional.softmax(teacher[i, shift:end_shift, :]/self.teacher_temperature, dim=-1),
                torch.zeros_like(teacher[i, :(teacher.size(1)-size), :])), dim=0
            )

        # 裁剪到最大答案长度
        # 找到学生和教师答案中的最大长度
        mex_length = max(max(student_answer_size), max(teacher_answer_size))
        # 只保留答案部分，去除多余的序列长度
        student = student[:, :mex_length, :]
        teacher = teacher[:, :mex_length, :]

        # 如果启用调试模式且当前rank匹配调试rank
        if self.debug and rank == self.debug_rank:
            print("\n\n----------------------------------")
            print("------- Label / Prediction -------")
            print("----------------------------------")
            # 提取学生和教师的标签（去除填充的-100值）
            student_labels = [row[row != -100] for row in student_targets]
            teacher_labels = [row[row != -100] for row in teacher_targets]
            print("------- Label shape -------")
            # 打印学生和教师标签的形状
            print(f"Student label shape: {student_answer_size[0]}")
            print(f"Teacher label shape: {teacher_answer_size[0]}")
            print("------- Student Label -> Prediction -------")
            # 打印学生的标签文本
            print(self.student_tokenizer.batch_decode(student_labels[0]))
            # 打印学生的预测文本（取概率最高的token）
            print(self.student_tokenizer.batch_decode(torch.argmax(
                student[0][:student_answer_size[0]], dim=-1)))
            print("------- Teacher Label -> Prediction -------")
            # 打印教师的标签文本
            print(self.teacher_tokenizer.batch_decode(teacher_labels[0]))
            # 打印教师的预测文本（取概率最高的token）
            print(self.teacher_tokenizer.batch_decode(torch.argmax(
                teacher[0][:teacher_answer_size[0]], dim=-1)))
            print("------- Prediction Teacher -> Student  -------")
            # 打印教师预测的文本（使用教师分词器）
            print(self.teacher_tokenizer.batch_decode(torch.argmax(
                teacher[0][:teacher_answer_size[0]], dim=-1)))
            # 打印学生预测的文本（使用学生分词器）
            print(self.student_tokenizer.batch_decode(torch.argmax(
                student[0][:student_answer_size[0]], dim=-1)))
            print("------- Shape -------")
            # 打印学生和教师张量的形状
            print(f"Student shape: {student.size()}")
            print(f"Teacher shape: {teacher.size()}\n")

        # 按降序排序以对齐概率分布
        # 这是跨分词器蒸馏的关键步骤：通过排序使不同词汇表的概率可比较
        # 排序后，最高概率的token在最前面，无论其在原始词汇表中的位置
        student = student.sort(dim=-1, descending=True).values
        teacher = teacher.sort(dim=-1, descending=True).values

        # 填充以获得相同的词汇表大小
        # 计算学生和教师词汇表大小的差异
        diff_size = student.size(2) - teacher.size(2)
        # 如果学生词汇表更大，填充教师的概率分布
        if diff_size > 0:
            teacher = F.pad(teacher, (0, diff_size), value=0)
        # 如果教师词汇表更大，填充学生的概率分布
        elif diff_size < 0:
            student = F.pad(student, (0, abs(diff_size)), value=0)

        # 如果启用调试模式且当前rank匹配调试rank
        if self.debug and rank == self.debug_rank:
            print("--------------------------------------------")
            print("---- Post-treatment tensor architecture ----")
            print("--------------------------------------------")
            print("------- Shape -------")
            # 打印处理后的张量形状
            print(f"Student shape: {student.size()}")
            print(f"Teacher shape: {teacher.size()}")
            print(" ------- First token -------")
            # 打印第一个token的前5个和后5个logits值
            print(f"Student first logits: {student[0][0][:5].tolist()}")
            print(f"Teacher first logits: {teacher[0][0][:5].tolist()}")
            print(f"Student last logits: {student[0][0][-5:].tolist()}")
            print(f"Teacher last logits: {teacher[0][0][-5:].tolist()}")
            print(" ------- Last token -------")
            # 打印最后一个token的前5个和后5个logits值
            print(f"Student first logits: {student[0][-1][:5].tolist()}")
            print(f"Teacher first logits: {teacher[0][-1][:5].tolist()}")
            print(f"Student last logits: {student[0][-1][-5:].tolist()}")
            print(f"Teacher last logits: {teacher[0][-1][-5:].tolist()}\n")

        # 计算交叉熵损失
        # 这是标准的语言模型损失，衡量学生模型预测与真实标签的差异
        crossentropy_loss = self.crossentropy_weight * student_predictions.loss

        # 初始化蒸馏损失张量（每个样本一个损失值）
        distillation_loss = torch.zeros(student.size(0), device=student.device)
        # 遍历批次中的每个样本
        for i in range(student.size(0)):
            # 取学生和教师答案长度的最小值（确保不超出范围）
            size = min(student_answer_size[i], teacher_answer_size[i])
            # 计算学生和教师概率分布的L1距离（绝对值差异）
            # 对词汇表维度求和，对序列长度求平均
            # 这衡量了学生模型的输出分布与教师模型的输出分布的相似度
            distillation_loss[i] = abs(student[i][:size] - teacher[i][:size]).sum(-1).mean(-1)
        # 对批次中的所有样本求平均
        distillation_loss = distillation_loss.mean()
        # 应用蒸馏损失权重
        distillation_loss = self.distillation_weight * (distillation_loss)

        # 如果启用调试模式且当前rank匹配调试rank
        if self.debug and rank == self.debug_rank:
            print("--------------------------------------")
            print("---------------- Loss ----------------")
            print("--------------------------------------")
            # 打印交叉熵损失值
            print(f"Crossentropy loss: {crossentropy_loss}")
            # 打印蒸馏损失值
            print(f"Distillation loss: {distillation_loss}")
            # 打印总损失值
            print(f"Total loss: {crossentropy_loss + distillation_loss}")

        # 返回总损失、交叉熵损失和蒸馏损失
        return crossentropy_loss + distillation_loss, crossentropy_loss, distillation_loss

    def __get_start_and_size_answers(self, answer_tensors):
        """获取答案的起始位置和长度
        参数:
            answer_tensors: 包含答案标签的张量列表，其中ignore_index表示非答案部分
        返回:
            (answers_index, answers_size): 答案起始索引列表和答案长度列表的元组"""
        # 初始化答案起始索引列表
        answers_index = []
        # 初始化答案长度列表
        answers_size = []

        # 遍历每个答案张量
        for answer in answer_tensors:
            # 创建布尔掩码，标记哪些位置是ignore_index（非答案部分）
            is_value = answer.eq(self.ignore_index)
            # 计算答案长度：总长度减去ignore_index的数量
            answers_size.append(len(answer) - int(is_value.sum()))
            # 找到所有ignore_index的位置索引
            indices = is_value.nonzero(as_tuple=True)[0]
            # 如果没有ignore_index，或者第一个元素不是ignore_index
            if len(indices) == 0 or indices[0] != 0:
                # 答案从位置0开始
                answers_index.append(0)
            else:
                # 计算相邻ignore_index位置的差值
                diff_indices = indices[1:] - indices[:-1]
                # 找到第一个不连续的位置（差值不等于1）
                break_index = (diff_indices != 1).nonzero()
                # 如果找到不连续位置，计算连续ignore_index的长度
                # 否则，所有ignore_index都是连续的
                length = (break_index[0].item() +
                          1) if len(break_index) > 0 else len(indices)
                # 答案起始位置是连续ignore_index结束后的位置
                answers_index.append(length-1)
        # 返回答案起始索引和长度
        return answers_index, answers_size