# Cross-Tokenizer Distillation 复现指南

> 论文: *Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs* (arXiv:2402.12030)
>
> 数据集: SQuAD | 教师模型: Llama-2-7b-chat | 学生模型: Pythia-410m

---

## 0. 环境准备

### 0.1 目录结构约定

项目代码中大量使用 `$HOME/llm-distillation` 和 `$HOME/llm-recipes` 作为硬编码路径，需要先创建软链接：

```bash
ln -s /home/user/cross-tokenizer-distillation/llm-distillation-main ~/llm-distillation
ln -s /home/user/cross-tokenizer-distillation/llm-recipes-main ~/llm-recipes
```

### 0.2 安装依赖

```bash
pip install torch transformers datasets accelerate peft fire tqdm wandb evaluate sentencepiece protobuf bert-score rouge-score
```

### 0.3 HuggingFace 登录

Llama-2 需要申请访问权限，登录你的 HuggingFace 账号：

```bash
huggingface-cli login
```

确保你已在 https://huggingface.co/meta-llama/Llama-2-7b-chat-hf 申请并获得了访问权限。

### 0.4 硬件需求

| 阶段 | 最低显存 | 建议配置 |
|------|---------|---------|
| Step 1: 教师生成预测 | ~16GB (bf16) | 单卡 A100/V100 |
| Step 3: 蒸馏训练 | ~40GB (教师+学生同时加载) | 多卡 FSDP 或单卡 A100-80G |
| Step 4: 评测 | ~2GB (仅学生模型) | 单卡即可 |

---

## 1. Step 1: 用教师模型生成 SQuAD 预测数据

教师模型在 SQuAD 训练集上生成答案（`answers_generated`），作为后续蒸馏的训练目标。

```bash
cd ~/llm-distillation

python datasets/generator.py \
    --model_id meta-llama/Llama-2-7b-chat-hf \
    --dataset_id squad \
    --split_name train \
    --task qa \
    --number_few_shot 1 \
    --batch_size 4 \
    --bfloat
```

生成结果保存到：
```
~/llm-distillation/datasets/generated/Llama-2-7b-chat-hf/squad/train/
```

> **说明**: `generator.py` 会加载教师模型，对 SQuAD 的每个样本用 `model.generate()` 生成答案，保存为 HuggingFace Dataset 格式（含 `answers_generated` 字段）。

---

## 2. Step 2: 数据集切分（训练集/验证集）

将生成的数据集按 90/10 切分为训练集和验证集：

```bash
cd ~/llm-distillation

python datasets/process.py \
    --dataset_path datasets/generated/Llama-2-7b-chat-hf/squad/train \
    --val_size 0.1
```

然后将处理后的数据集移动到训练代码期望的路径：

```bash
mkdir -p ~/llm-distillation/datasets/hf
cp -r datasets/generated/Llama-2-7b-chat-hf/squad/train ~/llm-distillation/datasets/hf/Llama-2-7b-chat-hf-squad
```

> **说明**: `process.py` 会过滤掉空答案，并用 seed=42 进行 train_test_split。蒸馏训练的 `squad.py` loader 会从 `~/llm-distillation/datasets/hf/{teacher_name}-squad` 路径读取数据。

---

## 3. Step 3: 运行蒸馏训练

### 3.1 单卡运行（如果显存足够）

```bash
cd ~/llm-recipes

python finetuning.py \
    --model_name EleutherAI/pythia-410m-deduped \
    --dataset.file ~/llm-distillation/datasets/loader/squad.py \
    --dataset.generated_by meta-llama/Llama-2-7b-chat-hf \
    --lr 1e-6 \
    --num_epochs 5 \
    --batch_size_training 4 \
    --val_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --output_dir ~/llm-distillation/output/pythia-410m-squad \
    --distillation \
    --distillation_config.model_name meta-llama/Llama-2-7b-chat-hf \
    --distillation_config.distil_factor 1.5 \
    --distillation_config.cross_entropy_factor 1 \
    --distillation_config.student_temperature 1 \
    --distillation_config.teacher_temperature 1 \
    --save_step 100
```

### 3.2 多卡 FSDP 运行（推荐）

```bash
cd ~/llm-recipes

torchrun --nproc_per_node 2 finetuning.py \
    --model_name EleutherAI/pythia-410m-deduped \
    --dataset.file ~/llm-distillation/datasets/loader/squad.py \
    --dataset.generated_by meta-llama/Llama-2-7b-chat-hf \
    --lr 1e-6 \
    --num_epochs 5 \
    --batch_size_training 4 \
    --val_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --output_dir ~/llm-distillation/output/pythia-410m-squad \
    --distillation \
    --distillation_config.model_name meta-llama/Llama-2-7b-chat-hf \
    --distillation_config.enable_fsdp \
    --distillation_config.pure_bf16 \
    --distillation_config.distil_factor 1.5 \
    --distillation_config.cross_entropy_factor 1 \
    --save_step 100
```

> **关键参数说明**:
> - `distil_factor=1.5`: 蒸馏损失权重（论文推荐值）
> - `cross_entropy_factor=1`: 交叉熵损失权重
> - `dataset.generated_by`: 必须与 Step 1 的教师模型一致，用于定位正确的数据集路径
> - 总损失 = CE_weight × CrossEntropy + Distil_weight × DistillationLoss
> - 蒸馏损失核心: 将 student/teacher 的 softmax 概率按降序排序后计算 L1 距离（解决跨词表对齐问题）

训练 checkpoint 保存在 `--output_dir` 指定的目录。

---

## 4. Step 4: 评测蒸馏后的学生模型

### 4.1 生成预测 + 计算指标（F1 / EM / ROUGE）

```bash
cd ~/llm-distillation/benchmark

# 确保结果目录存在
mkdir -p results/pythia-410m-squad/squad/untitled

python benchmark.py \
    --model_id ~/llm-distillation/output/pythia-410m-squad \
    --model_tokenizer EleutherAI/pythia-410m-deduped \
    --dataset_id squad \
    --split_name validation \
    --task qa \
    --number_few_shot 0 \
    --batch_size 4 \
    --bfloat \
    --save_predictions \
    --output_path results/pythia-410m-squad/squad/untitled
```

输出示例:
```json
{
    "model": "pythia-410m-squad",
    "dataset": "squad",
    "f1": 45.23,
    "precision": 48.10,
    "recall": 43.55,
    "em": 32.10,
    "squad": 38.67,
    "rouge1": 42.30,
    "rouge2": 20.15,
    "rougeL": 39.80
}
```

### 4.2 用 SQuAD 官方指标重新评分（可选）

```bash
cd ~/llm-distillation/benchmark

python official_metrics/squad.py \
    --dataset squad \
    --split validation \
    --predictions_file results/pythia-410m-squad/squad/untitled/predictions_0shots.json
```

---

## 5. Step 5: 对比基线（可选但推荐）

为了验证蒸馏的效果，建议同时评测以下基线：

### 5.1 评测教师模型（上限参考）

```bash
cd ~/llm-distillation/benchmark

mkdir -p results/Llama-2-7b-chat-hf/squad/untitled

python benchmark.py \
    --model_id meta-llama/Llama-2-7b-chat-hf \
    --dataset_id squad \
    --split_name validation \
    --task qa \
    --number_few_shot 1 \
    --batch_size 4 \
    --bfloat \
    --save_predictions \
    --output_path results/Llama-2-7b-chat-hf/squad/untitled
```

### 5.2 评测原始学生模型（下限参考）

```bash
cd ~/llm-distillation/benchmark

mkdir -p results/pythia-410m-deduped/squad/untitled

python benchmark.py \
    --model_id EleutherAI/pythia-410m-deduped \
    --dataset_id squad \
    --split_name validation \
    --task qa \
    --number_few_shot 0 \
    --batch_size 4 \
    --save_predictions \
    --output_path results/pythia-410m-deduped/squad/untitled
```

### 5.3 预期结果对比

| 模型 | 参数量 | F1 | EM |
|------|--------|----|----|
| Llama-2-7b-chat (教师) | 7B | ~70+ | ~55+ |
| Pythia-410m (原始学生) | 410M | ~5-10 | ~1-3 |
| Pythia-410m (蒸馏后) | 410M | ~30-45 | ~20-35 |

> 蒸馏后的学生模型应当显著优于原始模型，但低于教师模型。

---

## 完整流水线总结

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: generator.py                                       │
│  教师模型(Llama-2-7b-chat) 在 SQuAD 训练集上生成预测答案      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: process.py                                         │
│  数据集切分 (90% train / 10% validation)                     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: finetuning.py                                      │
│  跨词表蒸馏训练 (Llama→Pythia, sorted logit L1 loss)         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: benchmark.py                                       │
│  评测蒸馏后的学生模型 (F1 / EM / ROUGE)                      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: 对比基线                                            │
│  教师模型 vs 原始学生 vs 蒸馏学生                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 常见问题

### Q: 显存不足怎么办？
- Step 1 (生成): 使用 `--bfloat` 以 bf16 加载模型，减小 batch_size
- Step 3 (蒸馏): 使用 FSDP 多卡并行 (`--distillation_config.enable_fsdp --distillation_config.pure_bf16`)，增大 `gradient_accumulation_steps` 并减小 `batch_size_training`

### Q: `dataset.generated_by` 有什么用？
它决定了数据集的读取路径。`squad.py` loader 会从 `~/llm-distillation/datasets/hf/{generated_by.split('/')[-1]}-squad` 加载数据。必须与 Step 1 的教师模型一致。

### Q: 教师模型和学生模型的 tokenizer 不同怎么处理？
这正是本论文的核心贡献。`DistillationLoss` 将两个模型的 softmax 概率按降序排序后计算 L1 距离，不依赖 token-level 对齐，因此可以跨不同词表进行蒸馏。

### Q: 为什么学生模型用 0-shot 而教师模型用 1-shot？
这是代码中针对不同模型的 hardcoded 设置（见 `datasets/loader/squad.py:13-21`）。Llama-2-chat 使用 1-shot，而 Pythia 作为非 chat/instruct 模型走 else 分支，使用 0-shot。
