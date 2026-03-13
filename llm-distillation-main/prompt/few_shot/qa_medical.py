# 定义医学问答few-shot示例数据列表
# 包含5个来自PubMed的医学研究问答示例，用于few-shot学习
data = [
  {
    # 第一个示例：关于白细胞计数与损伤严重程度的相关性
    "context": "Injury severity score (ISS), Glasgow coma score (GCS), and revised trauma score (RTS) are the most frequently used methods to evaluate the severity of injury in blunt trauma patients. ISS is too complicated to assess easily and GCS and RTS are easy to assess but somewhat subjective. White blood cell count (WBC) is an easy, quick and objective test. This study was performed to evaluate the significance of the WBC count at presentation in the blunt trauma patients. 713 blunt trauma patients, who were admitted to the Uludag University Medical Center Emergency Department between 01.04.2000-31.12.2000, were retrospectively evaluated in terms of ISS, GCS, RTS and white blood cell count at presentation. Statistical analysis revealed that WBC was correlated positively with ISS, but negatively with GCS and RTS.",
    "question": "Does the leukocyte count correlate with the severity of injury?",
    "answers": "The leukocyte count at presentation can be used as an adjunct in the evaluation of the severity of injury in blunt trauma patients.",
    "sources": "pubmed_qa"
  },
  {
    # 第二个示例：关于超声在颞下颌关节内部紊乱诊断中的应用
    "context": "The aim of this study was to assess the diagnostic value of articular sounds, standardized clinical examination, and standardized articular ultrasound in the detection of internal derangements of the temporomandibular joint. Forty patients and 20 asymptomatic volunteers underwent a standardized interview, physical examination, and static and dynamic articular ultrasound. Sensitivity, specificity, and predictive values were calculated using magnetic resonance as the reference test. A total of 120 temporomandibular joints were examined. Based on our findings, the presence of articular sounds and physical signs are often insufficient to detect disk displacement. Imaging by static and dynamic high-resolution ultrasound demonstrates considerably lower sensitivity when compared with magnetic resonance. Some of the technical difficulties resulted from a limited access because of the presence of surrounding bone structures.",
    "question": "Internal derangement of the temporomandibular joint: is there still a place for ultrasound?",
    "answers": "The present study does not support the recommendation of ultrasound as a conclusive diagnostic tool for internal derangements of the temporomandibular joint.",
    "sources": "pubmed_qa"
  },
  {
    # 第三个示例：关于英国军队新兵牙齿健康的差异
    "context": "Figures from the British Defence Dental Services reveal that serving personnel in the British Army have a persistently lower level of dental fitness than those in the Royal Navy or the Royal Air Force. No research had been undertaken to ascertain if this reflects the oral health of recruits joining each Service. This study aimed to pilot a process for collecting dental and sociodemographic data from new recruits to each Service and examine the null hypothesis that no differences in dental health existed. Diagnostic criteria were developed, a sample size calculated and data collected at the initial training establishments of each Service. Data for 432 participants were entered into the analysis. Recruits in the Army sample had a significantly greater prevalence of dental decay and greater treatment resource need than either of the other two Services. Army recruits had a mean number of 2.59 (2.08, 3.09) decayed teeth per recruit, compared to 1.93 (1.49, 2.39 p<0.01) in Royal Navy recruits and 1.26 (0.98, 1.53 p<0.001) in Royal Air Force recruits. Among Army recruits 62.7% were from the two most deprived quintiles of the Index of Multiple Deprivation compared to 42.5% of Royal Naval recruits and 36.6% of Royal Air Force recruits.",
    "question": "Is there a differential in the dental health of new recruits to the British Armed Forces?",
    "answers": "A significant difference in dental health between recruits to each Service does exist and is a likely to be a reflection of the sociodemographic background from which they are drawn.",
    "sources": "pubmed_qa"
  },
  {
    # 第四个示例：关于不同肠道准备方法对腺瘤检测率的影响
    "context": "In recent clinical trials (RCT) of bowel preparation, Golytely was more efficacious than MiraLAX. We hypothesised that there is a difference in adenoma detection between Golytely and MiraLAX. To compare the adenoma detection rate (ADR) between these bowel preparations, and to identify independent predictors of bowel preparation quality and adenoma detection. This was a post hoc analysis of an RCT that assessed efficacy and patient tolerability of Golytely vs. MiraLAX/Gatorade in average risk screening colonoscopy patients. Bowel preparation quality was measured with the Boston Bowel Preparation Scale (BBPS). An excellent/good equivalent BBPS score was defined as ≥ 7. Polyp pathology review was performed. ADR was defined as the proportion of colonoscopies with an adenoma. Univariate and multivariate analyses were conducted. One hundred and ninety patients were prospectively enrolled (87 MiraLAX, 103 Golytely). Golytely had a higher rate of a BBPS score ≥ 7 (82.5% vs. MiraLAX 67.8%, P=0.02). The ADR in the Golytely cohort was 26.2% (27/103), and was 16.1% (14/87) for MiraLAX (P = 0.091). On multivariate analyses, Golytely was 2.13 × more likely to be associated with a BBPS ≥ 7 (95% CI 1.05-4.32, P = 0.04) and 2.28 × more likely to be associated with adenoma detection (95% CI 1.05-4.98, P = 0.04) than MiraLAX.",
    "question": "MiraLAX vs. Golytely: is there a significant difference in the adenoma detection rate?",
    "answers": "Golytely was more efficacious than MiraLAX in bowel cleansing, and was independently associated with both bowel prep quality (BBPS ≥ 7) and higher adenoma detection. Golytely should be used as first line for bowel prep for colonoscopy. Studies with larger populations are needed to confirm these results.",
    "sources": "pubmed_qa"
  },
  {
    # 第五个示例：关于乳腺X光检查质量标准法案的影响
    "context": "The United States Food and Drug Administration implemented federal regulations governing mammography under the Mammography Quality Standards Act (MQSA) of 1992. During 1995, its first year in implementation, we examined the impact of the MQSA on the quality of mammography in North Carolina. All mammography facilities were inspected during 1993-1994, and again in 1995. Both inspections evaluated mean glandular radiation dose, phantom image evaluation, darkroom fog, and developer temperature. Two mammography health specialists employed by the North Carolina Division of Radiation Protection performed all inspections and collected and codified data. The percentage of facilities that met quality standards increased from the first inspection to the second inspection. Phantom scores passing rate was 31.6% versus 78.2%; darkroom fog passing rate was 74.3% versus 88.5%; and temperature difference passing rate was 62.4% versus 86.9%.",
    "question": "Has the mammography quality standards act affected the mammography quality in North Carolina?",
    "answers": "In 1995, the first year that the MQSA was in effect, there was a significant improvement in the quality of mammography in North Carolina. This improvement probably resulted from facilities' compliance with federal regulations.",
    "sources": "pubmed_qa"
  },
]

# 定义函数：创建医学问答few-shot示例
def create_few_shot(number_few_shot: int, **args):
  """根据指定数量创建医学问答few-shot示例列表
  参数:
      number_few_shot: 需要的示例数量
  返回:
      包含医学论文、问题和答案的列表"""
  # 初始化示例列表
  shot = []
  # 遍历指定数量的示例
  for i in range(number_few_shot):
    # 添加医学论文、问题和答案对
    shot.append(
      [
        f"Medical paper: {data[i]['context']}\nQuestion: {data[i]['question']}",  # 医学论文内容和问题
        f"Answer: {data[i]['answers']}"  # 答案
      ]
    )
  # 返回示例列表
  return shot

# 定义函数：创建医学问答请求提示
def create_request(context="", question="", **args):
  """创建医学问答任务的请求提示
  参数:
      context: 医学论文内容
      question: 待回答的问题
  返回:
      包含医学论文、问题和答案提示的列表"""
  # 返回医学论文、问题和答案提示
  return [f"Medical paper: {context}\nQuestion: {question}", "Answer:"]
