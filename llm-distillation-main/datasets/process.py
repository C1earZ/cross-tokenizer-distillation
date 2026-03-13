# 从datasets库导入DatasetDict和load_from_disk函数
# DatasetDict用于管理包含多个划分（train/validation/test）的数据集
# load_from_disk用于从磁盘加载已保存的数据集
from datasets import DatasetDict, load_from_disk
# 导入命令行参数解析模块
import argparse

# 定义函数：解析命令行参数
def parse_args():
    # 创建参数解析器，描述脚本的功能
    parser = argparse.ArgumentParser(description="Split and save a dataset")
    # 添加数据集路径参数（必需）
    parser.add_argument("--dataset_path", required=True, type=str, help="Path to the dataset to be split")
    # 添加验证集大小参数，默认为0.1（10%）
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation size fraction (default: 0.1%)")
    # 返回解析后的参数
    return parser.parse_args()

# 定义主函数：处理数据集的划分和保存
def main():
    # 解析命令行参数
    args = parse_args()

    # 从磁盘加载数据集
    ds = load_from_disk(args.dataset_path)
    # 过滤数据集，只保留answers_generated字段不为空的样本
    # 这确保只使用有效的生成答案进行训练
    ds = ds.filter(lambda example: example['answers_generated'] is not None and example['answers_generated'] != "")
    # 将数据集划分为训练集和测试集（这里的test实际上是验证集）
    # test_size指定验证集的比例，seed确保划分的可重复性
    ds = ds.train_test_split(test_size=args.val_size, seed=42)

    # 创建DatasetDict，将划分重命名为标准的train和validation
    ds = DatasetDict({
        'train': ds['train'],  # 训练集
        'validation': ds['test']  # 验证集（从train_test_split的test重命名）
    })

    # 将处理后的数据集保存回原路径
    ds.save_to_disk(args.dataset_path)

# 脚本入口点
if __name__ == "__main__":
    # 运行主函数
    main()
