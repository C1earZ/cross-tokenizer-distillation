# 导入正则表达式模块
import re
# 导入字符串处理模块
import string
# 导入安全随机数生成模块
import secrets
# 导入Hugging Face评估库
import evaluate
# 从collections导入Counter，用于统计词频
from collections import Counter

# 定义私有函数：标准化文本
def _normalize(s):
  """对文本进行标准化处理，包括去除冠词、标点、转小写和修复空格
  参数:
      s: 输入文本字符串
  返回:
      标准化后的文本"""
  # 定义内部函数：去除冠词（a, an, the）
  def remove_articles(text):
    # 创建正则表达式，匹配单词边界内的冠词
    regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
    # 用空格替换冠词
    return re.sub(regex, ' ', text)
  # 定义内部函数：修复空格（去除多余空格）
  def white_space_fix(text):
    # 分割后重新连接，自动去除多余空格
    return ' '.join(text.split())
  # 定义内部函数：去除标点符号
  def remove_punc(text):
    # 获取所有标点符号集合
    exclude = set(string.punctuation)
    # 过滤掉所有标点符号
    return ''.join(ch for ch in text if ch not in exclude)
  # 定义内部函数：转换为小写
  def lower(text):
    return text.lower()
  # 依次应用所有标准化操作：小写 -> 去标点 -> 去冠词 -> 修复空格
  return white_space_fix(remove_articles(remove_punc(lower(s))))

# 定义私有函数：计算单个句子的F1分数
def _f1_score_sentence(prediction, answer):
    """计算预测和答案之间的F1分数、精确率和召回率
    参数:
        prediction: 预测文本
        answer: 参考答案文本
    返回:
        (f1, precision, recall): F1分数、精确率、召回率的元组"""
    # 将预测和答案分词
    prediction_tokens = prediction.split()
    answer_tokens = answer.split()

    # 计算共同词汇（使用Counter的交集操作）
    common = Counter(prediction_tokens) & Counter(answer_tokens)
    # 统计共同词汇的总数
    num_common = sum(common.values())

    # 如果没有共同词汇，返回0
    if num_common == 0:
        return 0, 0, 0

    # 计算精确率：共同词汇数 / 预测词汇总数
    precision = num_common / len(prediction_tokens)
    # 计算召回率：共同词汇数 / 答案词汇总数
    recall = num_common / len(answer_tokens)
    # 计算F1分数：精确率和召回率的调和平均数
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall

# 定义函数：计算批量预测的F1分数
def f1_score(predictions, answers):
    """计算预测列表和答案列表之间的平均F1分数、精确率和召回率
    参数:
        predictions: 预测文本列表
        answers: 答案列表（每个答案可以是字符串或字符串列表）
    返回:
        包含平均f1、precision、recall的字典"""
    # 初始化分数列表
    f1_scores, precision_scores, recall_scores = [], [], []

    # 遍历每对预测和答案
    for prediction, answer_list in zip(predictions, answers):
        # 标准化预测文本
        prediction = _normalize(prediction)
        # 初始化最大分数
        max_f1, max_precision, max_recall = 0, 0, 0

        # 如果答案列表为空
        if not answer_list:
          # 如果预测也为空或包含"no response"，则完全匹配
          if prediction == "" or 'no response' in prediction:
            max_f1 = max_precision = max_recall = 1
          # 否则完全不匹配
          else:
            max_f1 = max_precision = max_recall = 0
        # 如果答案列表不为空
        else:
          # 如果答案是字符串，转换为列表
          if isinstance(answer_list, str): answer_list = [answer_list]
          # 遍历所有可能的答案，取最高分数
          for answer in answer_list:
            # 标准化答案文本
            answer = _normalize(answer)
            # 计算F1分数
            f1, precision, recall = _f1_score_sentence(prediction, answer)
            # 更新最大分数
            max_f1, max_precision, max_recall = max(f1, max_f1), max(precision, max_precision), max(recall, max_recall)

        # 记录该样本的最大分数
        f1_scores.append(max_f1)
        precision_scores.append(max_precision)
        recall_scores.append(max_recall)

    # 计算平均分数
    average_f1 = sum(f1_scores) / len(f1_scores)
    average_precision = sum(precision_scores) / len(precision_scores)
    average_recall = sum(recall_scores) / len(recall_scores)

    # 返回平均分数字典
    return {'f1': average_f1, 'precision': average_precision, 'recall': average_recall}

# 定义函数：计算精确匹配率
def exact_match(predictions, answers):
    """计算预测和答案的精确匹配率（EM）
    参数:
        predictions: 预测文本列表
        answers: 答案列表（每个答案可以是字符串或字符串列表）
    返回:
        精确匹配率（0到1之间的浮点数）"""
    # 初始化精确匹配分数列表
    exact_match_scores = []
    # 遍历每对预测和答案
    for prediction, answer_list in zip(predictions, answers):
        # 标准化预测文本
        prediction = _normalize(prediction)
        # 如果答案是字符串，转换为列表
        if isinstance(answer_list, str): answer_list = [answer_list]
        # 标准化所有答案
        answer_list = [_normalize(item) for item in answer_list]
        # 如果答案列表为空且预测也为空或包含"no response"，则匹配
        if not answer_list and prediction == "" or "no response" in prediction: exact_match_scores.append(1)
        # 如果预测在答案列表中，则匹配
        if prediction in answer_list: exact_match_scores.append(1)
        # 否则不匹配
        else: exact_match_scores.append(0)
    # 返回平均精确匹配率
    return sum(exact_match_scores)/len(exact_match_scores)

# 定义函数：计算ROUGE分数
def rouge(predictions, answers):
    """使用ROUGE指标评估预测和答案的相似度
    参数:
        predictions: 预测文本列表
        answers: 参考答案列表
    返回:
        ROUGE分数字典（包含rouge1, rouge2, rougeL等）"""
    # 加载ROUGE评估指标，使用随机实验ID避免冲突
    rouge_metric = evaluate.load('rouge', experiment_id=f"{secrets.randbelow(10000)}")
    # 计算并返回ROUGE分数
    return rouge_metric.compute(predictions=predictions, references=answers)

# 定义函数：计算BERTScore
def bert_score(predictions, answers):
    """使用BERTScore评估预测和答案的语义相似度
    参数:
        predictions: 预测文本列表
        answers: 参考答案列表（可以是字符串列表或字典列表）
    返回:
        BERTScore字典（包含f1, precision, recall）"""
    # 加载BERTScore评估指标，使用随机实验ID避免冲突
    bertscore = evaluate.load("bertscore", experiment_id=f"{secrets.randbelow(10000)}")
    # 如果答案是字典格式（多个参考答案）
    if isinstance(answers[0], dict):
      # 初始化分数列表
      f1, precision, recall = [0]*len(predictions), [0]*len(predictions), [0]*len(predictions)
      # 遍历每个样本
      for i, row in enumerate(answers):
        # 遍历该样本的所有参考答案
        for answer in row:
          # 计算BERTScore
          tmp = bertscore.compute(predictions=predictions[i], references=answer, lang="en", rescale_with_baseline=True)
          # 取最高分数
          f1[i] = max(f1[i], tmp['f1'])
          precision[i] = max(precision[i], tmp['precision'])
          recall[i] = max(recall[i], tmp['recall'])
      # 返回分数字典
      return {'f1': f1, 'precision': precision, 'recall': recall}
    # 如果答案是简单列表格式
    else:
      # 直接计算并返回BERTScore
      return bertscore.compute(predictions=predictions, references=answers, lang="en", rescale_with_baseline=True)
