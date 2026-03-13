# 定义few-shot示例数据列表
# 包含5个对话摘要的示例，用于few-shot学习
data = [
    {
        # 第一个示例：关于租赁手提包的对话
        "context": "#Person1#: John, shall we go to Sun Store? I have decided to buy that Murrberry handbag. Anyway,I'm not carrying this one to Mary's wedding.\n#Person2#: But, Jane, why not rent one with Handbag Hire? Instead of $ 990,pay $ 50,and you have it for a whole week.\n#Person1#: Sounds great, but I never knew I can rent a handbag.\n#Person2#: Handbag Hire is a new business. It was founded two months ago. Its collection covers many designer handbags.\n#Person1#: So... for the price of one Murrberry, I can use a different bag each week for twenty weeks?\n#Person2#: Absolutely. And if you like one of them, you can choose to buy it at a discounted rate. Of course the price varies by age and condition. For example, a $ 1500 Murrberry bag can sell for just $750.\n#Person1#: Great, but how do I rent? By telephone? Or in person?\n#Person2#: Either. And more conveniently, it accepts online orders.\n#Person1#: I'll do it on line now. I still have one more question. Mary's wedding is next Saturday. There are only five days left. Do I have enough time?\n#Person2#: Don't worry. It promises that customers receive their orders by post within two days. Three more days to go.\n#Person1#: Oh, I'd better order one right now.",
        # 对话摘要：Jane想买手提包，John建议租赁并介绍了服务细节
        "answers": "Jane wants to buy that Murrberry handbag to carry to Mary's wedding, but John suggests renting one with Handbag Hire and tells her about the service in detail. Jane is pleased to have a try."
    },
    {
        # 第二个示例：关于度假计划的对话
        "context": "#Person1#: The summers are so great here! Not hot at all. I love the cooling breezes, the clear air, all the greenery.\n#Person2#: This really has been a wonderful holiday for us. Shall we take a walk around the pond or into those woods for a while?\n#Person1#: Let's do both! Are we in a rush or anything\n#Person2#: No, not really. I had thought we'd stay in Hamburg tonight, but we can't unless we rush it. Let's stay in Bremen instead. Tomorrow we can have lunch in Hamburg, then check into a hostel in Copenhagen and have dinner there.\n#Person1#: Sounds fine to me. Whatever, let's enjoy this pond first.\n#Person2#: Sure. We can walk around to that path that leads into the woods there. Hey, look! There are some wild ducks over there in the reeds.\n#Person1#: I see them! Wow! How do you know they're wild?\n#Person2#: I used to go hunting with my uncle, that's how.\n#Person1#: They're neat. Now Let's take that path into the woods and see what we can see. . .",
        # 对话摘要：两人在享受池塘，决定改变住宿计划
        "answers": "#Person1# and #Person2# are enjoying a pond. #Person1# and #Person2# had planned to stay in Hamburg tonight, but they decide to stay in Bremen since they are not in a rush."
    },
    {
        # 第三个示例：关于面试后续的对话
        "context": "#Person1#: Well Rebecca, is there anything else you need to know for now?\n#Person2#: I don't think so, Mr. Parsons. I think you have covered all the main points for me.\n#Person1#: Okay well listen, here is my business card with my mobile number. If any other questions spring to mind don't hesitate to contact me. Of course you can also call Miss Childs too.\n#Person2#: Great. Rmm, when can I expect to hear from you?\n#Person1#: Well, we are finishing the shortlist interviews tomorrow, so we will certainly have a decision made by early next week. Miss Childs will call you to discuss more on Monday or Tuesday. How does that so\n#Person2#: That sounds perfect. Thank you very much for taking the time to speak to me Mr. Parsons.\n#Person1#: The pleasure's all mine, Rebecca.\n#Person2#: I hope to hear from you very soon.\n#Person1#: Absolutely. Thanks for coming Rebecca. Goodbye.",
        # 对话摘要：面试后，Mr. Parsons告知Rebecca下周会有结果
        "answers": "Mr. Parsons gives Rebecca his business card after the interview and tells Rebecca the decision will be made by early next week and Miss Childs will contact Rebecca."
    },
    {
        # 第四个示例：关于求婚和婚礼计划的对话
        "context": "#Person1#: Trina, will you marry me?\n#Person2#: Yes! Yes! And yes! Jared, of course I'll marry you!\n#Person1#: Oh, Babe, I can't wait to spend the rest of my life with you! I can't wait for all the adventures we're going to have, for all the fights and the laughter. I can't wait to grow old and wrinkly with you.\n#Person2#: Oh, Jared! I can't wait for our wedding! I hope you don't mind, but I'Ve already chosen a date! Six months from now in the summer! Melissa saw you buying the ring last month so I'Ve had plenty of time to start planning!\n#Person1#: She what?\n#Person2#: Oh don't worry, sweetie, I didn't know when you were going to propose. It was still a nice surprise! As I was saying, I'Ve got it all planned out. There's almost nothing left to do! I wrote up our guest list and we will have roughly four hundred guests attending.\n#Person1#: Four hundred?\n#Person2#: No need to sweat it. My parents agreed to pay for most of the wedding, which is going to be low-budget anyway. So roughly four hundred people, which means that the hall at Northwoods Heights will be our reception venue. I thought it would be nice if we had the wedding at your parents'church and my uncle of course would be officiating. We'll meet with him soon for some pre-wedding counseling. The music for the wedding ceremony was a no-brainer. My step-sister and her string quartet will take care of that. My cousin will be the official photographer. I thought it would also be nice if his daughter could sing a solo. Did you know that she's going to be a professional opera singer?\n#Person1#: Ah. . .\n#Person2#: And then of course the ladies at the church would love to be our caterers for the banquet and we'll get the Youth Group to serve us. I was thinking that your friend's band could be our entertainment for the night. though they might have to tone it down a bit. Or we could hire a DJ. Your sister's husband could get us a discount with that company that does the decor at weddings. what's their name again? I was thinking that we could have an island paradise-themed wedding and our theme color would be a soothing blue like Aquamarine. And there will be a huge seashell on the wall behind the podium where we'll make our toasts! What do you think of small packages of drink mixes for our wedding favors? Who else am I missing? Oh, your uncle could be our florist and his wife could make our wedding cake!\n#Person1#: Wow.\n#Person2#: See? It's going to be wonderful! Oh this wedding is going to be everything I ever dreamed of.\n#Person1#: If I survive the next six months.",
        # 对话摘要：Trina接受求婚，已经计划好了婚礼的所有细节
        "answers": "Trina accepts Jared's proposal. Then, Jared is astonished to know that Trina already knew from Melissa who saw him buying the ring that he was planning this. Trina has chosen a date and has made a list of four hundred guests and she tells Jared about her arrangements in an ecstasy. Jared finds it hard to get through."
    },
    {
        # 第五个示例：关于梦想和现实的对话
        "context": "#Person1#: Isabelle, you know I'm not interested in fame. \n#Person2#: Well, you don't seem to be interested in getting a real job, either. \n#Person1#: You know I'm interested in teaching. I'm looking for jazz students. . . \n#Person2#: Yeah, and every high school student in town is banging on your door, right? \n#Person1#: I know they're out there. I'll find them. \n#Person2#: You're such a dreamer! You think that you can spread the word of jazz in an underpass? ",
        # 对话摘要：Isabelle认为#Person1#是梦想家
        "answers": "Isabelle thinks #Person1# is a dreamer because #Person1# doesn't do real things."
    }
]

# 定义函数：创建few-shot示例
def create_few_shot(number_few_shot: int, **args):
  """根据指定数量创建few-shot示例列表
  参数:
      number_few_shot: 需要的示例数量
  返回:
      包含对话和摘要对的列表"""
  # 初始化示例列表
  shot = []
  # 遍历指定数量的示例
  for i in range(number_few_shot):
    # 添加对话和摘要对
    shot.append(
      [
        f"Dialogue: {data[i]['context']}",  # 对话内容
        f"Summary: {data[i]['answers']}"    # 对话摘要
      ]
    )
  # 返回示例列表
  return shot

# 定义函数：创建请求提示
def create_request(context="", **args):
  """创建对话摘要任务的请求提示
  参数:
      context: 待摘要的对话内容
  返回:
      包含对话和摘要提示的列表"""
  # 返回对话和摘要提示
  return [f"Dialogue: {context}", "Summary:"]
