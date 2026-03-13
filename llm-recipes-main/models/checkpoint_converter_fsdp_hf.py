# 导入操作系统模块
import os
# 导入系统模块
import sys
# 导入YAML模块，用于读取配置文件
import yaml
# 导入fire模块，用于创建命令行接口
import fire

# 从transformers导入自动配置、分词器和因果语言模型
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# 获取当前文件所在目录
current_directory = os.path.dirname(os.path.abspath(__file__))
# 获取父目录
parent_directory = os.path.dirname(current_directory)
# 将父目录添加到Python搜索路径
sys.path.append(parent_directory)
# 从models.checkpoint_handler导入加载分片模型的函数
from models.checkpoint_handler import load_sharded_model_single_gpu

# 定义函数：从配置文件加载模型
def load_model_from_config(config_path):
    """从配置文件创建模型（不加载权重）
    参数:
        config_path: 模型配置路径或Hugging Face模型名称
    返回:
        model: 创建的模型对象（未初始化权重）"""
    # 从预训练模型加载配置
    model_config = AutoConfig.from_pretrained(config_path)
    # 根据配置创建模型（权重随机初始化）
    model = AutoModelForCausalLM.from_config(config=model_config)
    return model

# 定义主函数：将FSDP检查点转换为Hugging Face格式
def main(
    fsdp_checkpoint_path="", # FSDP分片模型检查点的路径
    consolidated_model_path="", # 保存转换后的HF模型检查点的路径
    HF_model_path_or_name="" # HF模型的路径或名称，包含config.json和tokenizer_config.json（例如meta-llama/Llama-2-7b-chat-hf）
    ):
    """将FSDP分片检查点转换为Hugging Face格式
    参数:
        fsdp_checkpoint_path: FSDP检查点目录
        consolidated_model_path: 输出的HF模型目录
        HF_model_path_or_name: 原始HF模型名称或路径"""

    # 尝试从训练参数文件中读取模型名称
    try:
        # 训练参数文件名
        file_name = 'train_params.yaml'
        # 构建训练参数文件的完整路径
        train_params_path = os.path.join(fsdp_checkpoint_path, file_name)
        # 打开并读取YAML文件
        with open(train_params_path, 'r') as file:
            data = yaml.safe_load(file)

            # 从YAML文件中获取模型名称
            HF_model_path_or_name = data.get('model_name')

            print(f"Model name: {HF_model_path_or_name}")
    # 如果文件不存在
    except FileNotFoundError:
        # 如果没有提供模型名称
        if not HF_model_path_or_name:
            print(f"The file {train_params_path} does not exist.")
            # 提示用户输入模型名称
            HF_model_path_or_name = input("Please enter the model name: ")
            print(f"Model name: {HF_model_path_or_name}")
    # 捕获其他异常
    except Exception as e:
        print(f"An error occurred: {e}")

    # 从配置加载模型定义（不加载权重）
    model_def = load_model_from_config(HF_model_path_or_name)
    print("model is loaded from config")
    # 从FSDP检查点加载模型权重到单个GPU
    model = load_sharded_model_single_gpu(model_def, fsdp_checkpoint_path)
    print("model is loaded from FSDP checkpoints")
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(HF_model_path_or_name)
    # 保存分词器到输出目录
    tokenizer.save_pretrained(consolidated_model_path)
    # 保存模型到输出目录（Hugging Face格式）
    model.save_pretrained(consolidated_model_path)
    print(f"HuggingFace model checkpoints has been saved in {consolidated_model_path}")

# 程序入口点
if __name__ == "__main__":
    # 使用fire创建命令行接口
    fire.Fire(main)
