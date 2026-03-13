# 导入命令行参数解析模块
import argparse
# 导入JSON模块，用于读取预测结果文件
import json
# 从datasets库导入数据集加载和评估指标加载函数
from datasets import load_dataset, load_metric

# 定义主函数：评估SQuAD数据集的预测结果
def main(args):
    """使用官方SQuAD评估指标计算模型预测的准确性
    参数:
        args: 命令行参数对象，包含数据集名称、数据集划分和预测文件路径"""
    # 加载SQuAD官方评估指标（包括EM和F1分数）
    squad_metric = load_metric("squad")

    # 加载指定的数据集和划分（如squad的validation集）
    ds = load_dataset(args.dataset, split=args.split)
    # 从数据集中提取参考答案，构建包含id和answers的字典列表
    references = [{"id": item['id'], "answers": item['answers']} for item in ds]

    # 打开预测结果文件并读取所有预测
    with open(args.predictions_file, 'r') as file:
        # 每行是一个JSON对象，解析为Python字典列表
        predictions = [json.loads(line) for line in file]

    # 使用SQuAD评估指标计算预测结果与参考答案的匹配度
    # 返回EM（精确匹配）和F1分数
    results = squad_metric.compute(predictions=predictions, references=references)
    # 打印评估结果
    print(results)

# 程序入口点
if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Evaluate SQuAD predictions")
    # 添加数据集名称参数（必需），例如'squad'
    parser.add_argument("--dataset", required=True, help="Name of the dataset (e.g., 'squad')")
    # 添加数据集划分参数（必需），例如'validation'
    parser.add_argument("--split", required=True, help="Name of the split (e.g., 'validation')")
    # 添加预测文件路径参数（必需）
    parser.add_argument("--predictions_file", required=True, help="Path to the predictions file")

    # 解析命令行参数
    args = parser.parse_args()
    # 调用主函数执行评估
    main(args)
