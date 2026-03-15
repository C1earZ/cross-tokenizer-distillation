# Cross-Tokenizer Distillation 复现指南（AutoDL 轻量版）

> 论文: *Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs* (arXiv:2402.12030)
>
> 数据集: SQuAD（使用 1/10 数据） | 教师模型: Llama-2-7b-chat | 学生模型: Pythia-410m
>
> 目标平台: AutoDL | 不追求精度，仅验证流程跑通

---

## 0. 环境准备

### 0.1 上传代码到 AutoDL

将项目上传到 AutoDL 实例（建议放到数据盘以节省系统盘空间）：

```bash
# 在 AutoDL 实例上，假设代码已上传到 /root/autodl-tmp/
cd /root/autodl-tmp/cross-tokenizer-distillation
```

### 0.2 目录结构约定

代码中硬编码了 `$HOME/llm-distillation` 和 `$HOME/llm-recipes` 路径，需要创建软链接：

```bash
# AutoDL 的 $HOME 是 /root
ln -s /root/autodl-tmp/cross-tokenizer-distillation/llm-distillation-main ~/llm-distillation
ln -s /root/autodl-tmp/cross-tokenizer-distillation/llm-recipes-main ~/llm-recipes
```

> **验证**: 运行 `ls ~/llm-distillation` 和 `ls ~/llm-recipes` 确认软链接正确。

### 0.3 安装依赖

```bash
pip install torch transformers datasets accelerate peft fire tqdm wandb evaluate sentencepiece protobuf bert-score rouge-score
```

> AutoDL 镜像通常预装了 PyTorch 和 CUDA，无需额外安装。选镜像时建议选带 PyTorch 2.x + CUDA 的版本。
> **RTX 5090 用户**: Blackwell 架构需要 PyTorch >= 2.6 和 CUDA >= 12.8，请确认镜像版本。

### 0.4 HuggingFace 登录

Llama-2 需要申请访问权限：

```bash
pip install huggingface_hub
huggingface-cli login
```

确保你已在 https://huggingface.co/meta-llama/Llama-2-7b-chat-hf 申请并获得了访问权限。

> **AutoDL 网络提示**: 如果 HuggingFace 下载慢，可使用 AutoDL 自带的学术加速：
> ```bash
> source /etc/network_turbo
> ```

### 0.5 硬件需求（轻量版）

使用 1/10 数据 + 低 batch_size 后的显存需求：

| 阶段 | 预估显存 | 建议 AutoDL 机型 |
|------|---------|-----------------|
| Step 1: 教师生成预测 | ~14GB (bf16, batch=1) | RTX 3090 24G / V100 16G |
| Step 3: 蒸馏训练 | ~18-24GB (bf16, batch=1) | RTX 3090 24G / RTX 4090 24G |
| Step 4: 评测 | ~2GB (仅学生模型) | 任意 GPU |

> **RTX 5090 32G**: 显存充足，Step 1 可用 `batch_size=4`，Step 3 可用 `batch_size_training=4`（`gradient_accumulation_steps` 相应调整为 2），训练速度大幅提升。

> 如果只有 16G 显存（如 V100），Step 3 可能需要用 FSDP 多卡，或进一步减小 batch_size。

---

## 1. Step 1: 用教师模型生成 SQuAD 预测数据

教师模型在 SQuAD 训练集上生成答案。

**RTX 5090 32G**（显存充足，使用 `batch_size=4` 加速生成）：

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

**其他显卡（24G 及以下）**，使用 `batch_size=1` 降低显存：

```bash
cd ~/llm-distillation

python datasets/generator.py \
    --model_id meta-llama/Llama-2-7b-chat-hf \
    --dataset_id squad \
    --split_name train \
    --task qa \
    --number_few_shot 1 \
    --batch_size 1 \
    --bfloat
```

生成结果保存到：
```
~/llm-distillation/datasets/generated/Llama-2-7b-chat-hf/squad/train/
```

> **说明**:
> - `batch_size=1` 大幅降低显存占用，代价是生成速度变慢
> - `--bfloat` 以 bf16 精度加载模型，显存减半
> - 这一步会处理完整的 SQuAD 训练集（~87k 条），因为生成阶段无法跳过数据
> - 如果时间太长，可以手动中断，后续用已生成的部分数据

---

## 2. Step 2: 数据集切分（训练集/验证集）

将生成的数据集按 90/10 切分：

```bash
cd ~/llm-distillation

python datasets/process.py \
    --dataset_path datasets/generated/Llama-2-7b-chat-hf/squad/train \
    --val_size 0.1
```

然后将处理后的数据集复制到训练代码期望的路径：

```bash
mkdir -p ~/llm-distillation/datasets/hf
cp -r datasets/generated/Llama-2-7b-chat-hf/squad/train ~/llm-distillation/datasets/hf/Llama-2-7b-chat-hf-squad
```

> **说明**: `process.py` 会过滤掉空答案，并用 seed=42 进行 train_test_split。

---

## 3. Step 3: 运行蒸馏训练（轻量配置）

### 3.1 单卡运行（RTX 5090 32G，推荐）

32GB 显存可同时加载教师和学生模型，使用 `batch_size=4` 提升训练速度：

```bash
cd ~/llm-recipes

python finetuning.py \
    --model_name EleutherAI/pythia-410m-deduped \
    --dataset.file ~/llm-distillation/datasets/loader/squad.py \
    --dataset.generated_by meta-llama/Llama-2-7b-chat-hf \
    --dataset.training_size 0.1 \
    --lr 1e-6 \
    --num_epochs 3 \
    --batch_size_training 4 \
    --val_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --output_dir ~/llm-distillation/output/pythia-410m-squad \
    --distillation \
    --distillation_config.model_name meta-llama/Llama-2-7b-chat-hf \
    --distillation_config.pure_bf16 \
    --distillation_config.distil_factor 1.5 \
    --distillation_config.cross_entropy_factor 1 \
    --distillation_config.student_temperature 1 \
    --distillation_config.teacher_temperature 1 \
    --save_step 50
```

### 3.2 单卡运行（其他 24G 显卡）

```bash
cd ~/llm-recipes

python finetuning.py \
    --model_name EleutherAI/pythia-410m-deduped \
    --dataset.file ~/llm-distillation/datasets/loader/squad.py \
    --dataset.generated_by meta-llama/Llama-2-7b-chat-hf \
    --dataset.training_size 0.1 \
    --lr 1e-6 \
    --num_epochs 3 \
    --batch_size_training 1 \
    --val_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --output_dir ~/llm-distillation/output/pythia-410m-squad \
    --distillation \
    --distillation_config.model_name meta-llama/Llama-2-7b-chat-hf \
    --distillation_config.pure_bf16 \
    --distillation_config.distil_factor 1.5 \
    --distillation_config.cross_entropy_factor 1 \
    --distillation_config.student_temperature 1 \
    --distillation_config.teacher_temperature 1 \
    --save_step 50
```

### 3.3 多卡 FSDP 运行（如果有多卡或单卡显存不够）

```bash
cd ~/llm-recipes

torchrun --nproc_per_node 2 finetuning.py \
    --model_name EleutherAI/pythia-410m-deduped \
    --dataset.file ~/llm-distillation/datasets/loader/squad.py \
    --dataset.generated_by meta-llama/Llama-2-7b-chat-hf \
    --dataset.training_size 0.1 \
    --lr 1e-6 \
    --num_epochs 3 \
    --batch_size_training 1 \
    --val_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --output_dir ~/llm-distillation/output/pythia-410m-squad \
    --distillation \
    --distillation_config.model_name meta-llama/Llama-2-7b-chat-hf \
    --distillation_config.enable_fsdp \
    --distillation_config.pure_bf16 \
    --distillation_config.distil_factor 1.5 \
    --distillation_config.cross_entropy_factor 1 \
    --save_step 50
```

> **轻量版参数说明**:
> - `dataset.training_size=0.1`: **只使用 1/10 的训练数据**（约 7.8k 条 → 约 780 条）
> - RTX 5090: `batch_size_training=4` + `gradient_accumulation_steps=2` = 等效 batch_size 8
> - 其他显卡: `batch_size_training=1` + `gradient_accumulation_steps=8` = 等效 batch_size 8
> - `num_epochs=3`: 减少训练轮数，不追求精度
> - `pure_bf16`: bf16 精度训练，显存减半
> - `save_step=50`: 更频繁保存 checkpoint（数据少，总步数也少）

---

## 4. Step 4: 评测蒸馏后的学生模型

### 4.1 生成预测 + 计算指标（F1 / EM / ROUGE）

```bash
cd ~/llm-distillation/benchmark

mkdir -p results/pythia-410m-squad/squad/untitled

python benchmark.py \
    --model_id ~/llm-distillation/output/pythia-410m-squad \
    --model_tokenizer EleutherAI/pythia-410m-deduped \
    --dataset_id squad \
    --split_name validation \
    --task qa \
    --number_few_shot 0 \
    --batch_size 16 \
    --bfloat \
    --save_predictions \
    --output_path results/pythia-410m-squad/squad/untitled
```

> 其他显卡使用 `--batch_size 4`。

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

### 5.1 评测原始学生模型（下限参考）

```bash
cd ~/llm-distillation/benchmark

mkdir -p results/pythia-410m-deduped/squad/untitled

python benchmark.py \
    --model_id EleutherAI/pythia-410m-deduped \
    --dataset_id squad \
    --split_name validation \
    --task qa \
    --number_few_shot 0 \
    --batch_size 16 \
    --save_predictions \
    --output_path results/pythia-410m-deduped/squad/untitled
```

> 其他显卡使用 `--batch_size 4`。

### 5.2 评测教师模型（上限参考，需要较大显存）

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

> RTX 5090 32G 评测教师模型（~14GB）仍有充足余量，可使用 `--batch_size 4`；其他显卡使用 `--batch_size 1`。

### 5.3 预期结果对比（轻量版，仅供参考）

| 模型 | 参数量 | F1 | EM |
|------|--------|----|----|
| Llama-2-7b-chat (教师) | 7B | ~70+ | ~55+ |
| Pythia-410m (原始学生) | 410M | ~5-10 | ~1-3 |
| Pythia-410m (蒸馏后, 1/10数据) | 410M | ~15-30 | ~10-20 |

> 使用 1/10 数据训练，蒸馏效果会弱于全量数据，但应当仍能看到相对原始学生的明显提升。核心目的是验证流程可行。

---

## 完整流水线总结

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: generator.py                                       │
│  教师模型(Llama-2-7b-chat) 在 SQuAD 训练集上生成预测答案      │
│  (batch_size=1, bf16)                                       │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: process.py                                         │
│  数据集切分 (90% train / 10% validation)                     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: finetuning.py                                      │
│  跨词表蒸馏训练 (training_size=0.1, batch=1, bf16)           │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: benchmark.py                                       │
│  评测蒸馏后的学生模型 (F1 / EM / ROUGE)                      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: 对比基线                                            │
│  原始学生 vs 蒸馏学生 (教师评测可选)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## AutoDL 常见问题

### Q: 显存不足 (OOM) 怎么办？
- 确保使用了 `--bfloat` 或 `--distillation_config.pure_bf16`
- 将 `batch_size` 降到 1
- 增大 `gradient_accumulation_steps` 来弥补
- 如果 Step 3 仍然 OOM，考虑使用多卡 FSDP

### Q: HuggingFace 下载模型很慢？
AutoDL 提供学术网络加速：
```bash
source /etc/network_turbo
```
或者使用 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 路径报错 FileNotFoundError？
检查软链接是否正确：
```bash
ls -la ~/llm-distillation
ls -la ~/llm-recipes
```
确保指向 `/root/autodl-tmp/cross-tokenizer-distillation/` 下对应的目录。

### Q: `dataset.generated_by` 有什么用？
它决定了数据集的读取路径。`squad.py` loader 会从 `~/llm-distillation/datasets/hf/{generated_by.split('/')[-1]}-squad` 加载数据。必须与 Step 1 的教师模型一致。

### Q: 教师模型和学生模型的 tokenizer 不同怎么处理？
这正是本论文的核心贡献。`DistillationLoss` 将两个模型的 softmax 概率按降序排序后计算 L1 距离，不依赖 token-level 对齐，因此可以跨不同词表进行蒸馏。

### Q: 为什么学生模型用 0-shot 而教师模型用 1-shot？
这是代码中针对不同模型的 hardcoded 设置（见 `datasets/loader/squad.py:13-21`）。Llama-2-chat 使用 1-shot，而 Pythia 作为非 chat/instruct 模型走 else 分支，使用 0-shot。

### Q: 数据盘 vs 系统盘？
AutoDL 的 `/root/autodl-tmp/` 是数据盘（大容量），`/root/` 是系统盘（通常较小）。建议把代码和模型都放在 `/root/autodl-tmp/` 下。
