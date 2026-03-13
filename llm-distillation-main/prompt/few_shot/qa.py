# 定义问答few-shot示例数据列表
# 包含5个来自不同领域的问答示例，用于few-shot学习
data = [
  {
    # 第一个示例：关于电视剧角色的问题
    "id": 72736,
    "title": "Christine's boyfriend",
    "context": "Patrick Harris (Tim DeKay), Old Christine's new boyfriend, who she meets in a video store and starts dating.",
    "question": "Who played patrick on new adventures of old christine?",
    "answers": "Tim DeKay",
    "sources": "Lexi/spanextract"
  },
  {
    # 第二个示例：关于死刑犯人数的问题
    "id": 0,
    "title": "June 14, 2018: Death Row Inmates",
    "context": "As of June 14, 2018, there were 2,718 death row inmates in the United States.",
    "question": "Total number of death row inmates in the us?",
    "answers": "2,718",
    "sources": "Lexi/spanextract"

  },
  {
      # 第三个示例：关于共产主义起源的问题
      "id": 419,
      "title": "Modern Communism",
      "context": "Most modern forms of communism are grounded at least nominally in Marxism, an ideology conceived by noted sociologist Karl Marx during the mid nineteenth century.",
      "question": "Who came up with the idea of communism ?",
      "answers": "Karl Marx",
      "sources": "Lexi/spanextract"
  },
  {
    # 第四个示例：关于滑铁卢战役指挥官的问题
    "id": 225,
    "title": "Napoleon's Defeat by Seventh Coalition",
    "context": "A French army under the command of Napoleon Bonaparte was defeated by two of the armies of the Seventh Coalition : a British-led Allied army under the command of the Duke of Wellington, and a Prussian army under the command of Gebhard Leberecht von Blücher, Prince of Wahlstatt.",
    "question": "Who commanded british forces at the battle of waterloo?",
    "answers": "The Duke of Wellington",
    "sources": "Lexi/spanextract"
  },
  {
  # 第五个示例：关于动画片角色的问题
  "id": 620,
  "title": "Canine character",
  "context": "Astro is a canine character on the Hanna-Barbera cartoon, The Jetsons.",
  "question": "What was the dog's name on the jetsons?",
  "answers": "Astro",
  "sources": "Lexi/spanextract"
  }
]

# 定义函数：创建问答few-shot示例
def create_few_shot(number_few_shot: int, **args):
  """根据指定数量创建问答few-shot示例列表
  参数:
      number_few_shot: 需要的示例数量
      **args: 可选参数，包括title（是否包含标题）
  返回:
      包含上下文、问题和答案的列表"""
  # 初始化示例列表
  shot = []
  # 遍历指定数量的示例
  for i in range(number_few_shot):
    # 如果参数中指定包含标题
    if args.get('title', False):
      # 添加包含标题的问答对
      shot.append(
        [
          f"Title: {data[i]['title']}\nContext: {data[i]['context']}\nQuestion: {data[i]['question']}",  # 标题、上下文和问题
          f"Answer: {data[i]['answers']}"  # 答案
        ]
      )
    # 如果不包含标题
    else:
      # 添加不包含标题的问答对
      shot.append(
        [
          f"Context: {data[i]['context']}\nQuestion: {data[i]['question']}",  # 上下文和问题
          f"Answer: {data[i]['answers']}"  # 答案
        ]
      )
  # 返回示例列表
  return shot

# 定义函数：创建问答请求提示
def create_request(title="", context="", question="", **args):
  """创建问答任务的请求提示
  参数:
      title: 可选的标题
      context: 上下文信息
      question: 待回答的问题
  返回:
      包含上下文、问题和答案提示的列表"""
  # 如果提供了标题，返回包含标题的提示
  if title: return [f"Title: {title}\nContext: {context}\nQuestion: {question}", "Answer:"]
  # 否则返回不包含标题的提示
  else: return [f"Context: {context}\nQuestion: {question}", "Answer:"]
