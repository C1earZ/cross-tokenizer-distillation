# 导入操作系统模块
import os
# 导入PyTorch深度学习框架
import torch
# 导入模块动态加载工具
import importlib

# 从pathlib导入Path类，用于路径操作
from pathlib import Path
# 从data.concatenator导入数据集拼接类
from data.concatenator import ConcatDataset
# 从configs.configs_utils导入数据加载器参数获取函数
from configs.configs_utils import get_dataloader_kwargs

# 全局变量：训练集排序索引
sort_index = []
# 全局变量：验证集排序索引
sort_index_val = []

# 定义函数：从Python文件动态加载模块
def load_module_from_py_file(py_file: str) -> object:
    """从指定的Python文件路径动态加载模块
    该方法用于加载不在Python路径中的模块
    参数:
        py_file: Python文件的路径
    返回:
        加载的模块对象"""
    # 从文件路径提取模块名称
    module_name = Path(py_file).name
    # 创建源文件加载器
    loader = importlib.machinery.SourceFileLoader(module_name, py_file)
    # 从加载器创建模块规范
    spec = importlib.util.spec_from_loader(module_name, loader)
    # 从规范创建模块对象
    module = importlib.util.module_from_spec(spec)

    # 执行模块代码
    loader.exec_module(module)
    # 返回加载的模块
    return module


# 定义函数：获取指定划分的数据集
def get_dataset(dataset_config, tokenizer, split: str) -> torch.utils.data.Dataset:
    """根据配置加载指定划分的数据集
    参数:
        dataset_config: 数据集配置对象
        tokenizer: 分词器
        split: 数据集划分（'train'或'validation'）
    返回:
        PyTorch数据集对象"""
    # 如果未指定数据集文件，抛出错误
    if not dataset_config.file:
        raise ValueError(
            f"Dataset not specified. Please select a dataset path with the parameter '--dataset.file'.")

    # 如果数据集文件是Python文件
    if dataset_config.file.endswith('.py'):
        # 直接使用该文件路径，函数名为get_split
        module_path, func_name = Path(dataset_config.file), "get_split"
    # 如果数据集文件是目录
    else:
        # 使用目录下的load.py文件，函数名为get_split
        module_path, func_name = Path(
            dataset_config.file+"/load.py"), "get_split"

    # 如果模块文件不存在，抛出错误
    if not os.path.isfile(module_path):
        raise ValueError(
            f"The load.py file in the dataset folder or the path to a python loading file doesn't exist. {module_path}")
    # 动态加载模块
    module = load_module_from_py_file(module_path.as_posix())

    # 尝试调用模块中的get_split函数
    try:
        return getattr(module, func_name)(dataset_config, tokenizer, split)
    except Exception as err:
        # 如果调用失败，打印错误并抛出异常
        print(err)
        raise ValueError(f"It seems like the given method name ({func_name}) is not present in the load.py file ({module_path.as_posix()}).")


# 定义函数：获取训练和验证数据加载器
def get_dataloader(dataset_config, train_config, tokenizer, rank, distil_config=None):
    """创建训练和验证数据加载器
    参数:
        dataset_config: 数据集配置
        train_config: 训练配置
        tokenizer: 分词器
        rank: 进程排名
        distil_config: 蒸馏配置（可选）
    返回:
        (train_dataloader, eval_dataloader): 训练和验证数据加载器的元组"""
    # 声明使用全局变量
    global sort_index
    global sort_index_val

    # 加载训练数据集
    dataset_train = get_dataset(
        dataset_config,
        tokenizer,
        split="train",
    )
    # 如果使用packing批处理策略，将数据集拼接成固定长度的块
    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(
            dataset_train, chunk_size=train_config.context_length)

    # 如果设置了上下文长度限制且尚未创建排序索引
    if train_config.context_length and not sort_index:
        # 创建排序索引：只保留长度不超过上下文长度的样本
        sort_index = [idx for idx, ex in enumerate(dataset_train) if len(ex['input_ids']) <= train_config.context_length]
    # 如果已有排序索引，使用它过滤数据集
    if train_config.context_length and sort_index:
        dataset_train = dataset_train.select(sort_index)

    # 获取训练数据加载器的参数
    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train", distil_config)
    # 创建训练数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,  # 数据加载工作进程数
        pin_memory=True,  # 将数据固定在内存中以加速GPU传输
        shuffle=False,  # 不打乱数据（FSDP会自动处理）
        **train_dl_kwargs,  # 其他参数（批次大小、采样器等）
    )
    # 只在主进程打印训练集大小
    if rank == 0:
        print(f"--> Training Set Length = {len(dataset_train)}")

    # 如果需要运行验证
    if (train_config.run_validation):
        # 加载验证数据集
        dataset_val = get_dataset(
            dataset_config,
            tokenizer,
            split="validation",
        )

        # 如果设置了上下文长度限制且尚未创建验证集排序索引
        if train_config.context_length and not sort_index_val:
            # 创建验证集排序索引：只保留长度不超过上下文长度的样本
            sort_index_val = [idx for idx, ex in enumerate(dataset_val) if len(ex['input_ids']) <= train_config.context_length]
        # 如果已有验证集排序索引，使用它过滤数据集
        if sort_index_val:
            dataset_val = dataset_val.select(sort_index_val)

        # 如果使用packing批处理策略，将验证数据集拼接成固定长度的块
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(
                dataset_val, chunk_size=train_config.context_length)

        # 获取验证数据加载器的参数
        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val", distil_config)
        # 创建验证数据加载器
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,  # 数据加载工作进程数
            pin_memory=True,  # 将数据固定在内存中以加速GPU传输
            shuffle=False,  # 不打乱验证数据
            **val_dl_kwargs,  # 其他参数（批次大小、采样器等）
        )
        # 只在主进程打印验证集大小
        if rank == 0:
            print(f"--> Validation Set Length = {len(dataset_val)}")
        # 返回训练和验证数据加载器
        return train_dataloader, eval_dataloader
    # 如果不需要验证，只返回训练数据加载器
    else:
        return train_dataloader, None


# 定义函数：获取蒸馏训练所需的数据加载器
def get_distillation_dataloader(dataset_config, train_config, distil_config, student_tokenizer, teacher_tokenizer, rank):
    """为知识蒸馏创建学生和教师模型的数据加载器
    参数:
        dataset_config: 数据集配置
        train_config: 训练配置
        distil_config: 蒸馏配置
        student_tokenizer: 学生模型的分词器
        teacher_tokenizer: 教师模型的分词器
        rank: 进程排名
    返回:
        (student_train_dataloader, teacher_train_dataloader, student_eval_dataloader, teacher_eval_dataloader):
        学生和教师的训练和验证数据加载器"""
    # 设置数据集由教师模型生成（用于加载预生成的教师答案）
    dataset_config.generated_by = teacher_tokenizer.name_or_path

    # 获取学生模型的数据加载器
    student_train_dataloader, student_eval_dataloader = get_dataloader(dataset_config, train_config, student_tokenizer, rank, distil_config)
    # 如果教师是编码器-解码器架构，设置标志
    dataset_config.encoder_decoder = True if distil_config.encoder_decoder else False
    # 获取教师模型的数据加载器
    teacher_train_dataloader, teacher_eval_dataloader = get_dataloader(dataset_config, train_config, teacher_tokenizer, rank, distil_config)
    # 恢复编码器-解码器标志为学生模型的设置
    dataset_config.encoder_decoder = train_config.encoder_decoder
    # 返回学生和教师的训练和验证数据加载器
    return student_train_dataloader, teacher_train_dataloader, student_eval_dataloader, teacher_eval_dataloader
