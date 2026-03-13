# 导入操作系统模块
import os
# 导入系统模块
import sys
# 导入JSON模块
import json
# 导入评分模块
import score
# 导入PyTorch深度学习框架
import torch
# 导入日志模块
import logging
# 导入命令行参数解析模块
import argparse
# 从transformers导入分词器和因果语言模型
from transformers import AutoTokenizer, AutoModelForCausalLM
# 从PyTorch导入数据加载器
from torch.utils.data import DataLoader
# 从datasets导入数据集加载函数
from datasets import load_dataset, load_from_disk
# 从itertools导入chain，用于展平嵌套列表
from itertools import chain
# 从tqdm导入进度条
from tqdm import tqdm
# 导入CSV模块，用于保存结果
import csv

# 将llm-distillation目录添加到Python搜索路径
sys.path.append(f"{os.getenv('HOME')}/llm-distillation")
# 启用分词器并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# 定义函数：获取可用的计算设备
def get_device():
    """检测并返回可用的计算设备（CUDA、MPS或CPU）
    返回:
        device: PyTorch设备对象"""
    # 默认使用CPU
    device = "cpu"
    # 如果CUDA可用，使用GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    # 如果MPS（Apple Silicon）可用，使用MPS
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    return device

# 定义函数：对数据项进行分词
def tokenization(items, tokenizer):
    """使用分词器对提示文本进行分词
    参数:
        items: 包含'prompt'字段的数据项字典
        tokenizer: 分词器对象
    返回:
        分词后的结果（包含input_ids和attention_mask）"""
    # 对提示文本进行分词，填充到最长序列
    return tokenizer(items["prompt"], padding='longest')

# 定义函数：根据映射文件重命名数据集列
def mapping(path, ds):
    """根据JSON映射文件重命名数据集的列名
    参数:
        path: JSON映射文件路径
        ds: 数据集对象
    返回:
        重命名后的数据集"""
    # 读取映射文件
    with open(path, 'r') as f: mapping = json.load(f)
    # 遍历映射，重命名列
    for key, value in mapping.items():
        ds = ds.rename_column(key, value)
    return ds

# 程序入口点
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Script to benchmark a model on a dataset.")
    # 添加模型ID参数
    parser.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf", help="Model ID")
    # 添加模型分词器参数（默认使用model_id）
    parser.add_argument("--model_tokenizer", type=str, help="Model tokenizer (default: model_id)")
    # 添加数据集ID参数
    parser.add_argument("--dataset_id", type=str, help="Dataset hugging face ID")
    # 添加数据集划分名称参数
    parser.add_argument("--split_name", type=str, default="test", help="Dataset split name")
    # 添加是否包含任务说明的参数
    parser.add_argument("--context", action="store_true", help="To pre prompt an explanation of the task")
    # 添加是否保留标题的参数
    parser.add_argument("--title", action="store_true", help="To keep title in the prompt")
    # 添加few-shot示例数量参数
    parser.add_argument("--number_few_shot", type=int, default=0, help="Number of few-shot examples")
    # 添加批次大小参数
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    # 添加数据加载器工作线程数参数
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    # 添加是否使用bfloat16精度的参数
    parser.add_argument("--bfloat", action="store_true", help="Load model in bf16")
    # 添加是否保存预测结果的参数
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions in txt file")
    # 添加是否从磁盘加载数据集的参数
    parser.add_argument("--from_disk", action="store_true", help="Load dataset from disk")
    # 添加任务类型参数
    parser.add_argument("--task", type=str, default="qa", help="Benchmark type (qa, qa_generative, summarization)")
    # 添加列名映射文件参数
    parser.add_argument("--mapping", type=str, default="", help="JSON file to map dataset column name")
    # 添加答案字典字段名参数
    parser.add_argument("--mapping_dict", type=str, default="text", help="Field name in the answer dictionary.")
    # 添加是否计算BERTScore的参数
    parser.add_argument("--bert_score", action="store_true", help="To compute bert score")
    # 添加输出路径参数
    parser.add_argument("--output_path", type=str, default="", help="Output path")
    # 添加上下文长度限制参数
    parser.add_argument("--context_length", type=int, default=None, help="Delete dataset row with length > context_length")
    # 解析命令行参数
    args = parser.parse_args()

    # 根据模型类型选择提示创建函数
    if 'chat' in args.model_id.split('/n')[:-2] or "instruct" in args.model_id.lower().split('/n')[:-2]:
        # 对话模型使用对话提示
        from prompt.prompt import create_chat_prompt as create_prompt
        is_chat = True
    else :
        # 普通模型使用标准提示
        from prompt.prompt import create_prompt
        is_chat = False

    # 定义函数：为数据项创建提示列（包含答案）
    def create_prompt_column(task, few_shot, item, has_title, tokenizer):
        """根据任务类型为数据项创建提示文本（包含答案）
        用于计算模型对正确答案的置信度
        参数:
            task: 任务类型（qa、qa_generative、qa_medical、summary_dialogue）
            few_shot: few-shot示例数量
            item: 数据项
            has_title: 是否包含标题
            tokenizer: 分词器
        返回:
            添加了'prompt'和'size'字段的数据项"""
        # 如果是问答任务
        if task == "qa" or task == "qa_generative":
            item['prompt'] = create_prompt(
                task, few_shot,
                title = item['title'] if has_title else "",  # 可选的标题
                context = item['context'],  # 上下文
                question = item['question'],  # 问题
                # Mistral模型需要将系统提示与用户消息合并
                sys_user = True if "mistralai" in args.model_id or args.context else False,
                chat_template = tokenizer.apply_chat_template if is_chat else None  # 对话模板
            )
        # 如果是医学问答任务
        elif task == "qa_medical":
             item['prompt'] = create_prompt(
                task, few_shot,
                context = item['context'],  # 医学论文内容
                question = item['question'],  # 问题
                sys_user = True if "mistralai" in args.model_id or args.context else False,
                chat_template = tokenizer.apply_chat_template if is_chat else None
            )
        # 如果是对话摘要任务
        elif task == "summary_dialogue":
            item['prompt'] = create_prompt(
                task, few_shot,
                context = item['context'],  # 对话内容
                sys_user = True if "mistralai" in args.model_id or args.context else False,
                chat_template = tokenizer.apply_chat_template if is_chat else None
            )
        # 计算提示的长度（不包含答案）
        item['size'] = len(tokenizer(item["prompt"], padding='longest', add_special_tokens=False)['input_ids'])

        # 提取答案（处理不同的答案格式）
        if isinstance(item['answers'], dict): answers = item['answers'][args.mapping_dict]
        elif isinstance(item['answers'][0], dict): answers = item[0][args.mapping_dict]
        else: answers = item['answers']

        # 如果答案是列表，取第一个
        if type(answers) == list: answers = answers[0]
        # 将答案附加到提示后面（用于计算置信度）
        item['prompt'] = item['prompt'] + " " + answers
        return item

    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logging.info('Start')
    # 获取计算设备
    device = get_device()
    logging.info(f'Device: {device}')

    # 加载分词器
    logging.info(f'Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer if args.model_tokenizer else args.model_id)
    # 添加填充token
    tokenizer.add_special_tokens({"pad_token":"<pad>"})
    # 设置填充方向为左侧
    tokenizer.padding_side = 'left'
    logging.info(f'Tokenizer loaded.')

    # 加载模型
    logging.info('Loading model...')
    # 如果使用bfloat16且不是CPU，以bfloat16精度加载
    if args.bfloat and device != "cpu": model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(device)
    # 否则以默认精度加载
    else: model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    # 调整token嵌入大小以匹配分词器
    model.resize_token_embeddings(len(tokenizer))
    # 设置模型的填充token ID
    model.config.pad_token_id = tokenizer.pad_token_id
    logging.info('Model loaded.')

    # 处理数据集
    logging.info('Processing dataset...')
    # 如果从磁盘加载
    if args.from_disk:
        dataset = load_from_disk(args.dataset_id)
        # 选择指定的划分
        if args.split_name: dataset = dataset[args.split_name]
    # 否则从Hugging Face加载
    else: dataset = load_dataset(args.dataset_id, split=args.split_name)
    # 如果提供了列名映射，应用映射
    if args.mapping: dataset = mapping(args.mapping, dataset)
    # 检查是否有标题列且需要保留标题
    has_title = True if 'title' in dataset.column_names and args.title else False
    # 为每个数据项创建提示列（包含答案）
    dataset = dataset.map(lambda item: create_prompt_column(args.task, args.number_few_shot, item, has_title, tokenizer=tokenizer))
    # 对提示进行分词
    dataset = dataset.map(lambda items: tokenization(items, tokenizer=tokenizer), batched=True, batch_size=args.batch_size)
    # 打印第一个样本的提示长度
    print(dataset['size'][0])
    # 设置数据集格式为PyTorch tensor（保留input_ids、attention_mask和size）
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", 'size'])
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # 初始化结果列表
    results = []
    # 创建Softmax层，用于计算概率
    m = torch.nn.Softmax(dim=-1)
    # 禁用梯度计算（推理模式）
    with torch.no_grad():
        # 遍历数据加载器
        for batch in tqdm(dataloader):
            # 获取模型输出的logits（第一个样本）
            output = model(batch['input_ids'].to(device)).logits[0]

            # 遍历答案部分的每个token（从提示结束位置到序列结束）
            for index in range(batch['size'], len(output-1)):
                # 检查模型预测的token是否与实际token相同
                same = False
                if batch['input_ids'][0][index].item() == torch.argmax(output[index-1]):
                    same = True
                # 记录结果：(是否预测正确, 预测的置信度)
                results.append((
                    same,  # 是否预测正确
                    m(output[index-1])[torch.argmax(output[index-1])].item(),  # 最高概率（置信度）
                ))

    # 确定输出路径
    titled_folder = "titled" if has_title else "untitled"
    output = args.output_path if args.output_path else f"{os.getenv('HOME')}/llm-distillation/benchmark/results/{args.model_id.split('/')[-1]}/{args.dataset_id.split('/')[-1]}/{titled_folder}/confidence.csv"
    # 将结果保存到CSV文件
    with open(output, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # 写入每一行结果
        for row in results:
            writer.writerow(row)
