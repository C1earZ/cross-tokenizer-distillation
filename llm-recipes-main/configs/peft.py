# 导入类型提示模块中的List类型
# 用于声明列表类型的参数
from typing import List
# 导入dataclasses模块中的dataclass装饰器和field函数
# dataclass用于简化类定义，field用于设置字段的默认值工厂函数
from dataclasses import dataclass, field

# 使用dataclass装饰器定义LoRA（低秩适应）配置类
# LoRA通过在模型中插入低秩矩阵来实现参数高效微调
@dataclass
class lora_config:
     # LoRA的秩（rank），默认为8
     # 秩越小，可训练参数越少，但表达能力也越弱
     r: int=8

     # LoRA的缩放因子alpha，默认为32
     # 用于控制LoRA更新的幅度，通常设置为r的倍数
     lora_alpha: int=32

     # 目标模块列表，默认为查询和值投影层
     # 指定在模型的哪些层应用LoRA，这里选择注意力机制中的q_proj和v_proj
     target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

     # 偏置项处理方式，默认为"none"
     # 可选值："none"（不训练偏置）、"all"（训练所有偏置）、"lora_only"（只训练LoRA的偏置）
     bias= "none"

     # 任务类型，默认为因果语言模型
     # 指定模型的任务类型，用于正确配置LoRA
     task_type: str= "CAUSAL_LM"

     # LoRA的dropout率，默认为0.05
     # 用于防止过拟合，在LoRA层中随机丢弃5%的神经元
     lora_dropout: float=0.05

     # 是否为推理模式，默认为False
     # True表示只进行推理不训练，False表示训练模式
     inference_mode: bool = False

# 使用dataclass装饰器定义Llama适配器配置类
# Llama Adapter是一种轻量级的微调方法，通过在模型中添加可学习的适配器层
@dataclass
class llama_adapter_config:
     # 适配器的长度（token数量），默认为10
     # 指定每个适配器包含多少个可学习的token
     adapter_len: int= 10

     # 应用适配器的层数，默认为30
     # 指定在模型的前30层应用适配器
     adapter_layers: int= 30

     # 任务类型，默认为因果语言模型
     task_type: str= "CAUSAL_LM"

# 使用dataclass装饰器定义前缀微调配置类
# Prefix Tuning通过在输入前添加可训练的虚拟token来适应新任务
@dataclass
class prefix_config:
     # 虚拟token的数量，默认为30
     # 指定在输入序列前添加多少个可学习的前缀token
     num_virtual_tokens: int=30

     # 任务类型，默认为因果语言模型
     task_type: str= "CAUSAL_LM"
