# 导入操作系统模块
import os
# 导入JSON模块，用于读取配置文件
import json
# 从importlib导入模块加载相关工具
from importlib import machinery, util
# 从pathlib导入Path类，用于路径操作
from pathlib import Path


# 定义私有函数：从Python文件动态加载模块
def __load_module_from_py_file(py_file: str) -> object:
    """从指定的Python文件路径动态加载模块
    参数:
        py_file: Python文件的路径
    返回:
        加载的模块对象"""
    # 从文件路径提取模块名称
    module_name = Path(py_file).name
    # 创建源文件加载器
    loader = machinery.SourceFileLoader(module_name, py_file)
    # 从加载器创建模块规范
    spec = util.spec_from_loader(module_name, loader)
    # 从规范创建模块对象
    module = util.module_from_spec(spec)

    # 执行模块代码
    loader.exec_module(module)
    # 返回加载的模块
    return module


# 定义函数：创建提示词（用于非对话模型）
def create_prompt(task: str, few_shot: int, **args):
    """为指定任务创建提示词
    参数:
        task: 任务类型（如"qa"、"summary"等）
        few_shot: few-shot示例数量
        **args: 其他参数，包括sys_user、title、context、question等
    返回:
        完整的提示词字符串"""
    # 初始化提示词为空字符串
    prompt = ""
    # 如果需要系统用户提示（sys_user），添加任务上下文说明
    if args.get('sys_user', False):
        prompt += json.load(open(f"{os.getenv('HOME')}/llm-distillation/prompt/context.json"))[task] + "\n\n"

    # 动态加载对应任务的few-shot模块
    module = __load_module_from_py_file(f"{os.getenv('HOME')}/llm-distillation/prompt/few_shot/{task}.py")
    # 调用模块的create_request函数创建请求部分
    request = '\n'.join(getattr(module, "create_request")(**args))

    # 如果需要few-shot示例
    if few_shot:
        # 调用模块的create_few_shot函数创建示例
        shot = '\n\n'.join(['\n'.join(s) for s in getattr(module, "create_few_shot")(few_shot, **args)])
        # 将示例和请求组合成完整提示词
        prompt += f"{shot}\n\n{request}"
    # 如果不需要few-shot示例
    else:
        # 只使用请求部分
        prompt += request
    # 返回完整提示词
    return prompt


# 定义函数：创建对话提示词（用于对话模型）
def create_chat_prompt(task: str, few_shot: int, **args):
    """为指定任务创建对话格式的提示词
    参数:
        task: 任务类型
        few_shot: few-shot示例数量
        **args: 其他参数，必须包含chat_template函数
    返回:
        格式化后的对话提示词字符串"""
    # 初始化对话列表和系统提示
    chat, sys_prompt = [], json.load(open(f"{os.getenv('HOME')}/llm-distillation/prompt/context.json"))[task]
    # 动态加载对应任务的few-shot模块
    module = __load_module_from_py_file(f"{os.getenv('HOME')}/llm-distillation/prompt/few_shot/{task}.py")
    # 调用模块的create_request函数创建请求部分
    request = getattr(module, "create_request")(**args)

    # 如果不使用sys_user模式，将系统提示作为单独的系统消息
    if not args.get('sys_user', False): chat.append({"role": "system", "content": sys_prompt})

    # 如果需要few-shot示例
    if few_shot:
        # 获取few-shot示例
        shot = getattr(module, "create_few_shot")(few_shot, **args)
        # 如果使用sys_user模式，将系统提示与第一个示例合并
        if args.get('sys_user', False):
            chat.extend([{"role": "user", "content": f"{sys_prompt}\n\n{shot[0][0]}"}, {"role": "assistant", "content": shot[0][1]}])
            # 移除已使用的第一个示例
            shot = shot[1:]
        # 将剩余的few-shot示例添加为用户-助手对话
        for s in shot: chat.extend([{"role": "user", "content": s[0]}, {"role": "assistant", "content": s[1]}])
        # 添加当前请求
        chat.extend([{"role": "user", "content": request[0]}, {"role": "assistant", "content": request[1]}])
    # 如果不需要few-shot示例
    else:
        # 如果使用sys_user模式，将系统提示与请求合并
        if args.get('sys_user', False): chat.extend([{"role": "user", "content": f"{sys_prompt}\n\n{request[0]}"}, {"role": "assistant", "content": request[1]}])
        # 否则直接添加请求
        else: chat.extend([{"role": "user", "content": request[0]}, {"role": "assistant", "content": request[1]}])
    # 使用聊天模板函数格式化对话
    prompt = args['chat_template'](chat, tokenize=False)

    # 根据任务类型截断提示词，只保留到答案提示部分
    if "qa" in task: return prompt[:prompt.rfind("Answer:") + 7]
    elif "summary" in task: return prompt[:prompt.rfind("Summary:") + 8]
