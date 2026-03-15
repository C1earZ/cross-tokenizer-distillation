# 导入PyTorch深度学习框架
import torch
# 导入dataclasses模块，用于处理数据类
import dataclasses
# 导入PyTorch优化器模块
import torch.optim as optim

# 从policies导入任意精度AdamW优化器
from policies import AnyPrecisionAdamW
# 从policies导入FSDP激活检查点应用函数
from policies import apply_fsdp_checkpointing
# 从models.fsdp导入FSDP自动包装策略
from models.fsdp import fsdp_auto_wrap_policy
# 从configs导入FSDP配置类
from configs import fsdp_config as FSDP_CONFIG
# 从models.distillation_model导入蒸馏模型类
from models.distillation_model import DistillationModel
# BetterTransformer 已被 PyTorch 2.x 内置的 SDPA 取代，不再需要 optimum
# from optimum.bettertransformer import BetterTransformer
# 从transformers导入自动因果语言模型、MT5条件生成模型和分词器
from transformers import AutoModelForCausalLM, MT5ForConditionalGeneration, AutoTokenizer
# 从configs.configs_utils导入PEFT配置生成和配置更新函数
from configs.configs_utils import generate_peft_config, update_config
# 从peft导入PEFT模型获取函数
# 新版peft已将prepare_model_for_int8_training重命名为prepare_model_for_kbit_training
from peft import get_peft_model, prepare_model_for_kbit_training as prepare_model_for_int8_training
# 从transformers导入各种模型的解码器层类
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
# 从PyTorch FSDP导入CPU卸载类
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

# 从PyTorch FSDP导入完全分片数据并行类
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)

# 从models.tools导入模型工具函数
from models.tools import (
    freeze_transformer_layers,  # 冻结Transformer层
    print_model_size,  # 打印模型大小
    get_policies  # 获取FSDP策略
)

# 定义函数：加载分词器
def load_tokenizer(name, encoder_decoder):
    """从预训练模型加载分词器
    参数:
        name: 模型名称或路径
        encoder_decoder: 是否为编码器-解码器模型
    返回:
        tokenizer: 分词器对象"""
    # 从预训练模型加载分词器
    tokenizer = AutoTokenizer.from_pretrained(name)
    # 如果不是编码器-解码器模型，将填充token设置为EOS token
    if not encoder_decoder:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

# 定义函数：加载模型
def load_model(train_config, rank):
    """根据配置加载预训练模型
    参数:
        train_config: 训练配置对象
        rank: 进程排名
    返回:
        model: 加载的模型对象"""
    # 如果启用FSDP，禁用缓存以节省内存；否则启用缓存
    use_cache = False if train_config.enable_fsdp else True
    # 定义内部函数：实际加载模型
    def load():
        # 如果是MT0模型（多语言T5）
        if "mt0" in train_config.model_name:
            kwargs = {
                "use_cache": use_cache,
            }
            if train_config.quantization:
                kwargs["load_in_8bit"] = True
                kwargs["device_map"] = "auto"
            return MT5ForConditionalGeneration.from_pretrained(
                train_config.model_name,
                **kwargs,
            )
        # 否则加载因果语言模型（如GPT、Llama）
        else:
            # 新版transformers不再支持load_in_8bit直接传参，需通过BitsAndBytesConfig
            kwargs = {
                "use_cache": use_cache,
            }
            if train_config.quantization:
                kwargs["load_in_8bit"] = True
                kwargs["device_map"] = "auto"
            return AutoModelForCausalLM.from_pretrained(
                train_config.model_name,
                **kwargs,
            )

    # 如果不启用FSDP，直接加载模型
    if not train_config.enable_fsdp:
        model = load()

    # 如果启用FSDP
    elif train_config.enable_fsdp:
        # 如果使用低CPU内存模式
        if train_config.low_cpu_fsdp:
            # 只在主进程（rank 0）加载完整模型
            if rank == 0:
                model = load()
            # 其他进程使用meta设备创建空模型（节省内存）
            else:
                model_config = AutoModelForCausalLM.from_pretrained(
                    train_config.model_name)
                model_config.use_cache = use_cache
                # 在meta设备上创建模型（不分配实际内存）
                with torch.device("meta"):
                    model = AutoModelForCausalLM.from_config(model_config)
        # 否则所有进程都加载完整模型
        else:
            model = load()

        # 如果使用快速内核（Flash Attention或Xformer）
        if train_config.use_fast_kernels:
            """
            对于FSDP和FSDP+PEFT，设置'use_fast_kernels'将启用
            Flash Attention或Xformer内存高效内核
            基于使用的硬件。这将加速微调。
            """
            pass  # BetterTransformer 已被 PyTorch 2.x SDPA 取代，无需转换

    # 打印模型大小信息
    print_model_size(model, train_config, rank)
    return model

# 定义函数：设置模型（应用PEFT、量化、FSDP等）
def set_model(model, train_config, fsdp_config, rank, kwargs):
    """对模型应用各种配置（量化、PEFT、FSDP等）
    参数:
        model: 原始模型
        train_config: 训练配置
        fsdp_config: FSDP配置
        rank: 进程排名
        kwargs: 其他参数
    返回:
        model: 配置后的模型"""
    # 如果启用量化，准备模型进行INT8训练
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # 如果使用PEFT（参数高效微调）
    if train_config.use_peft:
        # 生成PEFT配置
        peft_config = generate_peft_config(train_config, kwargs)
        # 将模型转换为PEFT模型
        model = get_peft_model(model, peft_config)
        # 打印可训练参数数量
        model.print_trainable_parameters()
    # 如果冻结部分层
    elif train_config.freeze_layers:
        freeze_transformer_layers(train_config.num_freeze_layers)

    # 如果启用FSDP
    if train_config.enable_fsdp:
        # 如果使用纯BF16精度，将模型转换为bfloat16
        if fsdp_config.pure_bf16: model.to(torch.bfloat16)

        # 获取混合精度策略和包装策略
        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        # 创建自动包装策略（指定哪些层需要被FSDP包装）
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, [LlamaDecoderLayer, GPTNeoXLayer, MistralDecoderLayer, FalconDecoderLayer])

        # 使用FSDP包装模型
        model = FSDP(
            model,
            # 如果使用PEFT则使用自定义包装策略，否则使用默认策略
            auto_wrap_policy=my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            # 如果启用CPU卸载，将参数卸载到CPU
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            # 如果不使用纯BF16，应用混合精度策略
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            # 分片策略（完全分片、混合分片等）
            sharding_strategy=fsdp_config.sharding_strategy,
            # 设备ID（当前CUDA设备）
            device_id=torch.cuda.current_device(),
            # 限制all_gather操作以节省内存
            limit_all_gathers=True,
            # 同步模块状态（用于低CPU内存模式）
            sync_module_states=train_config.low_cpu_fsdp,
            # 参数初始化函数（用于低CPU内存模式）
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )

        # 如果启用FSDP激活检查点，应用检查点
        if fsdp_config.fsdp_activation_checkpointing: apply_fsdp_checkpointing(model)
        return model
    # 如果不使用FSDP
    else:
        # 如果使用量化，直接返回模型（已在CPU上）
        if train_config.quantization: return model
        # 否则将模型移动到指定GPU
        else:
            return model.to(f"cuda:{rank}")

# 定义函数：获取模型和分词器
def get_model(train_config, fsdp_config, rank, kwargs):
    """加载并配置模型和分词器
    参数:
        train_config: 训练配置
        fsdp_config: FSDP配置
        rank: 进程排名
        kwargs: 其他参数
    返回:
        (tokenizer, model): 分词器和模型的元组"""
    # 加载模型
    model = load_model(train_config, rank)
    # 设置模型（应用PEFT、FSDP等）
    model = set_model(model, train_config, fsdp_config, rank, kwargs)
    # 加载分词器
    tokenizer = load_tokenizer(train_config.model_name, train_config.encoder_decoder)
    # 设置填充token为EOS token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer, model

# 定义函数：获取蒸馏模型（学生和教师）
def get_distillation_models(train_config, distil_config, fsdp_config, rank, kwargs):
    """加载并配置学生和教师模型用于知识蒸馏
    参数:
        train_config: 训练配置（学生模型）
        distil_config: 蒸馏配置（教师模型）
        fsdp_config: FSDP配置
        rank: 进程排名
        kwargs: 其他参数
    返回:
        (student_tokenizer, teacher_tokenizer, distillation_model): 学生分词器、教师分词器和蒸馏模型"""
    # 获取学生模型和分词器
    student_tokenizer, student_model = get_model(train_config, fsdp_config, rank, kwargs)

    # 创建教师模型的FSDP配置
    teacher_fsdp_config = FSDP_CONFIG()
    # 使用蒸馏配置更新教师FSDP配置
    update_config((teacher_fsdp_config), **dataclasses.asdict(distil_config))
    # 获取教师模型和分词器
    teacher_tokenizer, teacher_model = get_model(distil_config, distil_config, rank, kwargs)

    # 返回学生分词器、教师分词器和蒸馏模型（包装学生和教师）
    return student_tokenizer, teacher_tokenizer, DistillationModel(student_model, teacher_model)

# 定义函数：获取优化器
def get_optimizer(model, train_config, fsdp_config):
    """根据配置创建优化器
    参数:
        model: 模型对象
        train_config: 训练配置
        fsdp_config: FSDP配置
    返回:
        optimizer: 优化器对象"""
    # 如果使用纯BF16且优化器类型为anyprecision
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        # 使用任意精度AdamW优化器（支持BF16动量和方差）
        return AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,  # 学习率
            momentum_dtype=torch.bfloat16,  # 动量使用BF16
            variance_dtype=torch.bfloat16,  # 方差使用BF16
            use_kahan_summation=False,  # 不使用Kahan求和（提高数值稳定性）
            weight_decay=train_config.weight_decay,  # 权重衰减
        )
    # 否则使用标准AdamW优化器
    else:
        return optim.AdamW(
            model.parameters(),
            lr=train_config.lr,  # 学习率
            weight_decay=train_config.weight_decay,  # 权重衰减
        )
