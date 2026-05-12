# hate-speech-ft 中文使用指南

本文面向需要在 Colab 中运行仇恨言论检测实验的团队成员，说明当前
`hate-speech-ft` 项目的用途、运行入口、依赖准备、数据处理、训练/验证/测试
流程、W&B 记录方式、结果文件位置和常见问题。

本文重点结合当前已经实现的：

```text
standalone_distilbert_full_ft/
```

也会说明主项目实验框架中 `src/run_experiment.py` 的用途。两者的定位不同：

- `standalone_distilbert_full_ft/`：最小可运行 full fine-tuning 实验，适合先跑通
  DistilBERT full fine-tuning。
- `src/` + `configs/`：更完整的团队实验框架，适合后续多方法比较、HPO、seed
  runs 和聚合结果。

## 1. 项目用途

本项目用于比较不同方法在 HateXplain 仇恨言论三分类任务上的表现与计算成本。
核心研究目标不是只看分数，而是同时关注：

- validation / test macro-F1
- precision / recall
- 训练时间
- 峰值显存
- 可训练参数量
- W&B 与本地 JSON 记录是否可复现

当前已经真正可运行的主方法是：

```text
DistilBERT full fine-tuning
```

其他方法，例如 TF-IDF、Bi-LSTM、LoRA、partial FT、LP-FT，目前在主项目里是
`planned` 状态，需要后续分别实现。

## 2. 运行入口怎么选

### 推荐新手先用 standalone

如果你只是想跑通 DistilBERT full fine-tuning，使用：

```text
standalone_distilbert_full_ft/distilbert_full_ft_colab.ipynb
```

它会一步步完成：

1. 检查 GPU。
2. 挂载 Google Drive。
3. clone 或更新 repo。
4. 安装依赖。
5. 登录 W&B。
6. 检查脚本中的硬编码参数。
7. 运行训练、验证、测试。
8. 查看 metrics 和 predictions。
9. 把输出复制到 Google Drive。

### 团队正式实验用主 launcher

如果要跑主项目 catalog 中的实验，使用：

```text
notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb
```

或者命令行：

```bash
python src/run_experiment.py --list
python src/run_experiment.py --validate_protocol
python src/run_experiment.py --experiment distilbert_full_smoke --dry_run
python src/run_experiment.py --experiment distilbert_full_smoke
```

主 launcher 适合：

- 多个 experiment entry
- HPO trial command generation
- confirmation / final seed command generation
- result aggregation
- 团队统一命名和输出目录

standalone 适合：

- 快速理解 full fine-tuning 全流程
- 之后拆成独立 repo
- 手动改源文件参数
- 不需要复杂框架

## 3. 环境与依赖准备

### Colab runtime

建议使用 GPU runtime：

```text
Runtime -> Change runtime type -> GPU
```

A100、L4、T4 都可以运行，但时间和显存会不同。由于 Colab GPU 类型不稳定，
实验记录中必须保留 GPU name 和显存数据。

### standalone 依赖

在 Colab 或本地执行：

```bash
pip install -r standalone_distilbert_full_ft/requirements.txt
```

当前 standalone 依赖包括：

```text
torch
transformers
datasets
accelerate
scikit-learn
wandb
```

### 主项目依赖

主项目 Colab 使用：

```bash
pip install -r requirements-colab.txt
```

根目录的 `requirements.txt` 现在只是指向同一份轻量依赖：

```text
-r requirements-colab.txt
```

## 4. W&B 准备

W&B 用于在线记录实验指标、配置和运行过程，但**本地 JSON 文件仍然是最终可复现
记录的 source of truth**。

### Colab Secret

在 Colab 左侧 Secrets 中添加：

```text
WANDB_API_KEY
```

standalone notebook 会尝试读取这个 secret 并执行 W&B login。

### standalone 脚本中的 W&B 参数

在：

```text
standalone_distilbert_full_ft/train_distilbert_hatexplain.py
```

顶部可以修改：

```text
USE_WANDB
WANDB_PROJECT
WANDB_ENTITY
WANDB_MODE
WANDB_RUN_NAME
```

常见选择：

```text
USE_WANDB = True
WANDB_MODE = "online"
```

如果只是测试，不想联网：

```text
WANDB_MODE = "offline"
```

如果完全不用 W&B：

```text
USE_WANDB = False
```

## 5. 数据集准备

项目使用 Hugging Face Datasets 自动下载：

```text
Hate-speech-CNERG/hatexplain
```

不需要手动下载数据文件。

当前 full fine-tuning 使用的数据策略是：

- 使用官方 `train` / `validation` / `test` split。
- 输入文本由 `post_tokens` 拼接：

```text
" ".join(post_tokens)
```

- 标签采用 strict majority vote。
- 三名标注者没有严格多数时丢弃样本。
- 三分类标签为：

```text
0 = hatespeech
1 = normal
2 = offensive
```

standalone 脚本会在运行时打印：

```text
Train examples
Validation examples
Test examples
```

这些数量应该保存在 `run_summary.json` 和 `metrics.json` 中。

## 6. standalone full fine-tuning 训练流程

### 打开 notebook

使用：

```text
standalone_distilbert_full_ft/distilbert_full_ft_colab.ipynb
```

按顺序运行每个 cell。

### 直接命令行运行

也可以直接运行：

```bash
python standalone_distilbert_full_ft/train_distilbert_hatexplain.py
```

默认输出目录：

```text
standalone_distilbert_full_ft/outputs/distilbert_full_ft
```

### 训练过程中会做什么

脚本会执行：

1. 设置随机种子。
2. 保存 `config_snapshot.json`。
3. 初始化 W&B。
4. 加载 HateXplain。
5. 构造 train / validation / test records。
6. 加载 tokenizer。
7. tokenize dataset。
8. 加载 `DistilBERTForSequenceClassification`。
9. 统计参数量。
10. 创建 Hugging Face `Trainer`。
11. 训练。
12. 验证集评估。
13. 保存 validation predictions。
14. 如果 `RUN_TEST=True`，进行 test 评估并保存 test predictions。
15. 保存 final model。
16. 保存 metrics、runtime、memory、model-selection summary。
17. 打印 manual record fields。

## 7. 关键配置项和参数作用

所有 standalone 配置都在：

```text
standalone_distilbert_full_ft/train_distilbert_hatexplain.py
```

文件顶部。改变实验设置需要直接改这个文件。

### 实验身份

```text
METHOD
TRIAL_ID
SEARCH_STAGE
SEED
```

建议：

- tuning run：`SEARCH_STAGE = "tuning"`，`RUN_TEST = False`
- final run：`SEARCH_STAGE = "final"`，`RUN_TEST = True`
- 每次正式运行前修改 `TRIAL_ID`，避免不同 run 混在一起

### 数据与模型

```text
DATASET_NAME
MODEL_NAME
OUTPUT_DIR
MAX_LENGTH
```

默认：

```text
DATASET_NAME = "Hate-speech-CNERG/hatexplain"
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
```

如果要换模型，改 `MODEL_NAME`。如果 tokenizer 和模型不一致，不建议在 standalone
里拆开改，避免最小脚本变复杂。

### 训练超参数

```text
LEARNING_RATE
TRAIN_BATCH_SIZE
EVAL_BATCH_SIZE
NUM_EPOCHS
WEIGHT_DECAY
WARMUP_RATIO
```

full fine-tuning 默认建议：

```text
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
```

如果显存不够，优先调小：

```text
TRAIN_BATCH_SIZE
EVAL_BATCH_SIZE
```

### 样本上限

```text
MAX_TRAIN_SAMPLES
MAX_EVAL_SAMPLES
MAX_TEST_SAMPLES
```

完整运行：

```text
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None
MAX_TEST_SAMPLES = None
```

快速 smoke run：

```text
MAX_TRAIN_SAMPLES = 64
MAX_EVAL_SAMPLES = 64
MAX_TEST_SAMPLES = 64
NUM_EPOCHS = 1
RUN_TEST = False
```

### 是否跑测试集

```text
RUN_TEST
```

规则：

- 调参、debug、HPO：`RUN_TEST = False`
- 最终固定配置后：`RUN_TEST = True`

不要在 tuning 阶段根据 test 指标改参数。

## 8. 不同场景应该怎么设置

### 场景 A：检查环境是否能跑通

修改：

```text
MAX_TRAIN_SAMPLES = 64
MAX_EVAL_SAMPLES = 64
MAX_TEST_SAMPLES = 64
NUM_EPOCHS = 1
RUN_TEST = False
WANDB_MODE = "offline"
```

用途：

- 验证依赖安装
- 验证 dataset 下载
- 验证 GPU 可用
- 验证输出文件能生成

### 场景 B：正式 full fine-tuning validation run

修改：

```text
SEARCH_STAGE = "tuning"
RUN_TEST = False
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None
NUM_EPOCHS = 3
```

用途：

- 比较 validation macro-F1
- 不看 test
- 用于选择超参数

### 场景 C：最终 test run

修改：

```text
SEARCH_STAGE = "final"
RUN_TEST = True
SEED = 42
TRIAL_ID = "standalone_distilbert_full_ft_final_seed42"
```

之后分别跑：

```text
SEED = 43
SEED = 44
```

最终报告应使用 3 个 seed 的 mean ± std。

### 场景 D：W&B 不稳定或没有 key

修改：

```text
WANDB_MODE = "offline"
```

或者：

```text
USE_WANDB = False
```

本地 JSON 文件仍然会保存完整结果。

## 9. 评估指标在哪里看

### W&B 中会看到什么

如果 `USE_WANDB=True`，W&B 会记录：

- config
- train metrics
- validation metrics
- test metrics
- runtime
- memory
- parameter counts
- model selection summary

W&B 适合看曲线和在线比较。

### 本地 JSON 中会看到什么

最终更可靠的记录在：

```text
standalone_distilbert_full_ft/outputs/distilbert_full_ft/
```

主要文件：

```text
config_snapshot.json
metrics.json
run_summary.json
trainer_log_history.json
```

`metrics.json` 中包含：

```text
train
validation
test
runtime
model_selection
parameters
```

关键字段包括：

```text
eval_f1_macro
eval_accuracy
test_f1_macro
test_accuracy
gpu_synchronized_train_time_sec
total_runtime_sec
peak_mem_allocated_mb
peak_mem_reserved_mb
trainable_params
total_params
best_epoch
best_model_checkpoint
```

### 哪些需要手动记录

脚本最后会打印：

```text
Manual record fields
```

这段 printout 是为手动记录准备的，包含：

```text
method
trial_id
seed
hparams_json
best_epoch
val_macro_f1
train_time_s
peak_mem_allocated_mb
peak_mem_reserved_mb
trainable_params
total_params
status
```

如果你在写表格或论文实验记录，优先复制这段，或者从 `run_summary.json` 中提取。

## 10. 预测结果在哪里看

validation predictions：

```text
predictions_validation.json
```

test predictions：

```text
predictions_test.json
```

每条 prediction 包含：

```text
id
text
label
label_name
prediction
prediction_name
probabilities
```

这些预测文件**默认不会完整上传到 W&B**。原因是它们可能比较大，而且文本内容可能不适合全部进入在线 dashboard。

如果需要人工错误分析，应直接下载或查看本地 JSON 文件。

## 11. 最终模型保存位置

standalone 最终模型保存到：

```text
standalone_distilbert_full_ft/outputs/distilbert_full_ft/final_model/
```

里面包括：

```text
config.json
model.safetensors 或 pytorch_model.bin
tokenizer files
```

中间 checkpoint 保存到：

```text
standalone_distilbert_full_ft/outputs/distilbert_full_ft/checkpoints/
```

由于脚本设置了：

```text
load_best_model_at_end = True
metric_for_best_model = "eval_f1_macro"
```

最终保存的模型应该来自 validation macro-F1 最好的 checkpoint，而不一定是最后一个 epoch。

## 12. 如何复制结果到 Google Drive

standalone Colab notebook 最后有一个 cell，会把输出复制到：

```text
/content/drive/MyDrive/hate_speech_ft/standalone_distilbert_full_ft/run_<timestamp>/
```

建议每次正式运行后都执行这个 cell。否则 Colab runtime 断开后，本地 VM 文件可能丢失。

## 13. 主项目 catalog 方式

如果不使用 standalone，而是使用主项目 runner：

```bash
python src/run_experiment.py --list
python src/run_experiment.py --validate_protocol
python src/run_experiment.py --experiment distilbert_full_smoke --dry_run
python src/run_experiment.py --experiment distilbert_full_smoke
```

HPO command generation：

```bash
python src/run_experiment.py \
  --experiment distilbert_full_tuning \
  --suggest_trials 3 \
  --search_space full_ft \
  --hpo_seed 42
```

final seed command generation：

```bash
python src/run_experiment.py \
  --experiment distilbert_full_tuning \
  --suggest_seed_runs final \
  --set learning_rate=2e-5
```

主项目方式适合团队比较多个方法。standalone 方式适合最小 full fine-tuning 复现。

## 14. 常见问题与排查

### 没有 GPU

现象：

```text
CUDA available: False
```

处理：

- Colab 菜单选择 GPU runtime。
- 重新运行 runtime check cell。
- 如果仍无 GPU，可能是 Colab 配额或 availability 问题。

### W&B login 失败

处理：

- 确认 Colab Secrets 中有 `WANDB_API_KEY`。
- 确认 notebook access permission 已打开。
- 或改成：

```text
WANDB_MODE = "offline"
```

### Hugging Face 下载慢或 rate limit

处理：

- 设置 Hugging Face token。
- 重试 runtime。
- 确认网络可访问 Hugging Face。

### CUDA out of memory

处理：

优先调小：

```text
TRAIN_BATCH_SIZE
EVAL_BATCH_SIZE
```

如果还不够，可以降低：

```text
MAX_LENGTH
```

但降低 `MAX_LENGTH` 会改变实验设置，需要记录。

### 输出目录被旧结果覆盖

standalone 当前会写入固定目录：

```text
outputs/distilbert_full_ft
```

如果要保留多次运行，修改：

```text
OUTPUT_DIR
TRIAL_ID
WANDB_RUN_NAME
```

或者每次运行后先复制到 Google Drive。

### test 指标不应该用于调参

如果当前是 tuning/debug：

```text
RUN_TEST = False
```

只有最终配置冻结后才设置：

```text
RUN_TEST = True
```

### 结果到底看 W&B 还是 JSON

建议：

- W&B：看训练过程、曲线、快速比较。
- JSON：作为最终记录、论文表格、错误分析和复现依据。

如果两边不一致，以本地 JSON 为准。

## 15. 最小检查清单

正式运行前确认：

- `SEED` 正确。
- `TRIAL_ID` 唯一。
- `SEARCH_STAGE` 与 `RUN_TEST` 一致。
- `MAX_TRAIN_SAMPLES` / `MAX_EVAL_SAMPLES` / `MAX_TEST_SAMPLES` 是否为预期。
- `WANDB_PROJECT` / `WANDB_ENTITY` 是否正确。
- Colab 使用 GPU runtime。
- 输出目录会被保存到 Google Drive。

运行后确认：

- `metrics.json` 存在。
- `run_summary.json` 存在。
- `predictions_validation.json` 存在。
- final run 时 `predictions_test.json` 存在。
- `final_model/` 存在。
- `Manual record fields` 已打印或已从 JSON 复制。
