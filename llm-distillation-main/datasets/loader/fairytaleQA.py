# 导入操作系统模块
import os
# 导入系统模块，用于修改Python路径
import sys
# 从datasets库导入从磁盘加载数据集的函数
from datasets import load_from_disk

# 将llm-distillation目录添加到Python搜索路径
sys.path.append(f"{os.getenv('HOME')}/llm-distillation")
# 从prompt模块导入创建对话提示的函数
from prompt.prompt import create_chat_prompt
# 从prompt模块导入创建普通提示的函数
from prompt.prompt import create_prompt

# 定义函数：对单个数据项进行分词处理
def tokenize(item, tokenizer):
    """将童话故事问答数据项转换为模型输入的token序列
    参数:
        item: 数据项，包含context（故事内容）、question（问题）、answers_generated（生成的答案）
        tokenizer: 分词器对象
    返回:
        包含input_ids、labels和attention_mask的字典"""
    # 判断是否为对话模型（名称中包含'chat'或'instruct'）
    is_chat = True if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower() else False
    # 设置任务类型为生成式问答（童话故事）
    task = "qa_generative"

    # 根据不同的模型设置few-shot示例数量
    if tokenizer.name_or_path == "meta-llama/Llama-2-7b-chat-hf":
        shot = 2  # Llama-2-chat使用2个示例
    elif tokenizer.name_or_path == "mistralai/Mistral-7B-Instruct-v0.2":
        shot = 4  # Mistral-Instruct使用4个示例
    elif tokenizer.name_or_path == "tiiuae/falcon-7b-instruct":
        shot = 2  # Falcon-instruct使用2个示例

    # 如果是对话模型，创建对话格式的提示
    if is_chat:
        prompt = create_chat_prompt(
            task, shot,
            context = item['context'],  # 童话故事内容
            question = item['question'],  # 问题
            # Mistral模型需要将系统提示与用户消息合并
            sys_user = True if "mistralai/Mistral-7B-Instruct-v0.2" in tokenizer.name_or_path else False,
            chat_template = tokenizer.apply_chat_template  # 使用分词器的对话模板
        )
    # 如果是普通模型，创建标准格式的提示（不使用few-shot）
    else:
        prompt = create_prompt(
            task, 0,  # 普通模型不使用few-shot示例
            context = item['context'],
            question = item['question'],
        )

    # 对提示进行初步编码（包含BOS token）
    context_tokens = tokenizer.encode(f"{tokenizer.bos_token} {prompt}", add_special_tokens=False)

    # 对话模型的特殊处理
    if 'chat' in tokenizer.name_or_path.lower() or "instruct" in tokenizer.name_or_path.lower():
        # 对话模型不需要手动添加BOS token
        context_tokens = tokenizer.encode(f"{prompt}", add_special_tokens=False)
        # Falcon模型需要在答案前添加空格
        if tokenizer.name_or_path == "tiiuae/falcon-7b-instruct":
            answer_tokens = tokenizer.encode(f" {item['answers_generated']}", add_special_tokens=False)
        # 其他对话模型不需要前置空格
        else:
            answer_tokens = tokenizer.encode(f"{item['answers_generated']}", add_special_tokens=False)
    # 普通语言模型的处理
    else:
        # 手动添加BOS token
        context_tokens = tokenizer.encode(f"{tokenizer.bos_token}{prompt}", add_special_tokens=False)
        # 答案前添加空格，后添加EOS token
        answer_tokens = tokenizer.encode(f" {item['answers_generated']}{tokenizer.eos_token}", add_special_tokens=False)

    # 将提示和答案拼接成完整的输入序列
    prompt_tokens = context_tokens+answer_tokens
    # 创建标签序列：提示部分用-100填充（不计算损失），答案部分保留原token
    labels_tokens = (len(context_tokens)*[-100,])+answer_tokens

    # 构建模型输入字典
    combined_tokens = {
        "input_ids": prompt_tokens,  # 输入token序列
        "labels": labels_tokens  # 标签序列（用于计算损失）
    }
    # 添加注意力掩码（全1表示所有token都参与计算）并返回
    return dict(combined_tokens, attention_mask=[1]*len(combined_tokens["input_ids"]))


# 定义函数：获取指定划分的数据集
def get_split(dataset_config, tokenizer, split):
    """加载并预处理FairytaleQA数据集的指定划分
    参数:
        dataset_config: 数据集配置对象
        tokenizer: 分词器
        split: 数据集划分（'train'或'validation'）
    返回:
        处理后的数据集对象"""
    # 从磁盘加载预生成的FairytaleQA数据集
    # 数据集名称格式：{生成模型名称}-FairytaleQA
    dataset = load_from_disk(f"{os.getenv('HOME')}/llm-distillation/datasets/hf/{dataset_config.generated_by.split('/')[-1]}-FairytaleQA")
    # 选择指定的数据集划分
    dataset = dataset[split]
    # 如果配置的数据集大小小于1，则只使用部分数据
    if dataset_config.size < 1: dataset = dataset.select(range(int(len(dataset)*dataset_config.size)))
    # 对数据集中的每个样本应用tokenize函数，并移除原始特征列
    dataset = dataset.map(lambda item: tokenize(item, tokenizer), remove_columns=list(dataset.features))
    # 返回处理后的数据集
    return dataset
