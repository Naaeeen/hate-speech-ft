# Standalone DistilBERT Full Fine-Tuning 中文使用指南

本文只说明 `standalone_distilbert_full_ft/` 这个独立实验文件夹，不涉及本仓库其他
目录或其他实验入口。

这个 standalone 文件夹的目标很明确：用最少的工程结构，完成 HateXplain 上
DistilBERT full fine-tuning 的训练、验证、最终测试、W&B 记录、本地 JSON 记录、
预测文件保存和最终模型保存。它之后可以单独拆成一个新 repo。

## 1. 文件说明

```text
standalone_distilbert_full_ft/
  distilbert_full_ft_colab.ipynb   Colab 全流程 notebook
  train_distilbert_hatexplain.py   单文件训练 / 验证 / 测试脚本
  requirements.txt                 最小依赖
  README.md                        英文简要说明
  USAGE_GUIDE_ZH.md                本中文指南
```

核心原则：

- 不 import 这个文件夹外的项目代码。
- 不使用复杂 config system。
- 不使用自动化 experiment framework。
- 不使用 reusable pipeline abstraction。
- 所有关键参数都在 `train_distilbert_hatexplain.py` 顶部用常量写死。
- 想改实验，就直接改 Python 文件顶部常量。

## 2. 什么时候使用 standalone

使用 standalone 的情况：

- 你只负责 DistilBERT full fine-tuning。
- 你想先跑通一个 minimum viable deep research 实验。
- 你希望训练、验证、测试、预测文件和模型保存都能端到端完成。
- 你可以接受手动改代码、手动复制 HPO 表格、手动对比结果。

不适合 standalone 的情况：

- 你要同时管理很多 method 的统一实验平台。
- 你需要自动生成大量 trial command。
- 你需要统一聚合所有 teammate 的结果。
- 你不希望手动改 Python 常量。

如果当前任务只是 full fine-tuning，standalone 更容易读懂，也更容易交付。

## 3. Colab 环境准备

在 Colab 中打开：

```text
standalone_distilbert_full_ft/distilbert_full_ft_colab.ipynb
```

建议 runtime：

```text
Runtime -> Change runtime type -> GPU
```

然后按 notebook 顺序执行：

1. 检查 GPU。
2. 挂载 Google Drive。
3. clone 或更新 repo。
4. 安装依赖。
5. 登录 W&B。
6. 检查脚本顶部常量。
7. 运行训练脚本。
8. 查看 metrics、predictions、model。
9. 把输出复制到 Google Drive。

依赖安装命令：

```bash
pip install -r standalone_distilbert_full_ft/requirements.txt
```

## 4. W&B 设置

W&B 用来在线记录 config、训练曲线、validation/test metrics、runtime、显存和参数量。
但是最终可复现记录仍然以本地 JSON 文件为准。

Colab 推荐在 Secrets 中添加：

```text
WANDB_API_KEY
```

并打开 notebook access permission。脚本里不要硬编码 API key。

如果在线记录：

```python
USE_WANDB = True
WANDB_MODE = "online"
WANDB_PROJECT = "hate-speech-ft"
WANDB_ENTITY = None  # 或你的 team/entity 名称
```

如果只想本地测试：

```python
USE_WANDB = True
WANDB_MODE = "offline"
```

如果完全不用 W&B：

```python
USE_WANDB = False
```

HateXplain 和 `distilbert-base-uncased` 是公开资源，通常不需要 Hugging Face token。
如果遇到 Hugging Face rate limit，可以在 Colab Secrets 里额外保存 `HF_TOKEN`，并在
notebook 中登录 Hugging Face；这属于可选加速/稳定性步骤，不是默认必需项。

## 5. 数据集与标签策略

脚本自动从 Hugging Face Datasets 下载：

```text
Hate-speech-CNERG/hatexplain
```

固定策略：

- 使用官方 `train` / `validation` / `test` split。
- 输入文本由 `post_tokens` 拼接：`" ".join(post_tokens)`。
- 标签来自三个 annotators 的 strict majority vote。
- 没有严格多数的样本会被丢弃。
- 三分类标签：

```text
0 = hatespeech
1 = normal
2 = offensive
```

不要把 rationales、targets、post id、annotator id 加进模型输入，否则就不是同一个
实验设定。

## 6. 关键参数在哪里改

所有实验参数都在：

```text
standalone_distilbert_full_ft/train_distilbert_hatexplain.py
```

文件顶部。

### 实验身份

每次正式运行前至少检查：

```python
METHOD = "full-ft"
TRIAL_ID = "standalone_distilbert_full_ft_tuning_seed42"
SEARCH_STAGE = "tuning"
SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "distilbert_full_ft"
OVERWRITE_OUTPUT_DIR = False
```

建议命名规则：

```text
smoke:  standalone_distilbert_full_ft_smoke_seed42
tuning: standalone_distilbert_full_ft_lr2e-5_seed42
final:  standalone_distilbert_full_ft_final_lr2e-5_seed42
```

每个重要 run 使用不同 `TRIAL_ID` 和不同 `OUTPUT_DIR`，避免覆盖结果。脚本默认
`OVERWRITE_OUTPUT_DIR = False`，如果目录里已经有 `metrics.json`、predictions、
checkpoint 或 `final_model/`，会直接报错。只有确认要替换旧结果时才改成 `True`。

### 模型与训练参数

常用参数：

```python
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
MAX_GRAD_NORM = 1.0
OPTIM = "adamw_torch"
LR_SCHEDULER_TYPE = "linear"
EARLY_STOPPING_PATIENCE = 2
EARLY_STOPPING_THRESHOLD = 0.001
MIXED_PRECISION = "none"
GRADIENT_CHECKPOINTING = False
RUN_TEST = False
```

Full fine-tuning HPO 主要改：

```python
LEARNING_RATE
```

建议候选：

```text
1e-5
2e-5
3e-5
```

除非有明确原因，不要在同一轮 HPO 里同时改很多参数，否则很难解释哪个因素导致
结果变化。

### 样本上限

完整实验：

```python
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None
MAX_TEST_SAMPLES = None
```

快速 smoke：

```python
MAX_TRAIN_SAMPLES = 64
MAX_EVAL_SAMPLES = 64
MAX_TEST_SAMPLES = 64
NUM_EPOCHS = 1
RUN_TEST = False
```

调试时可以设样本上限；正式 tuning/final 应使用完整 train/validation/test。

默认脚本是 validation-only tuning：

```python
SEARCH_STAGE = "tuning"
RUN_TEST = False
```

脚本会阻止 `SEARCH_STAGE` 为 `smoke`、`tuning` 或 `confirm` 时运行 test set。只有：

```python
SEARCH_STAGE = "final"
RUN_TEST = True
```

才会生成 test metrics 和 `predictions_test.json`。

## 7. 输出文件在哪里

默认输出目录：

```text
standalone_distilbert_full_ft/outputs/distilbert_full_ft/
```

主要输出：

```text
config_snapshot.json          本次运行的配置、seed、设备、依赖版本
metrics.json                  train / validation / test / runtime / memory / params / dataset audit
run_summary.json              一次运行的总摘要
trainer_log_history.json      Hugging Face Trainer 原始日志
predictions_validation.json   validation 预测、真实标签、概率
predictions_test.json         test 预测、真实标签、概率；仅 RUN_TEST=True 时生成
failure.json                  失败时保存错误信息
final_model/                  最终保存的模型和 tokenizer
checkpoints/                  中间 checkpoint
```

实验完成后应把整个输出目录复制到 Google Drive。Colab VM 断开后，本地文件可能丢失。

## 8. 指标看 W&B 还是 JSON

建议分工：

- W&B：看曲线、在线对比、快速检查 run 是否正常。
- JSON：最终实验记录、论文表格、复现和错误分析的 source of truth。

如果 W&B 和本地 JSON 不一致，以本地 JSON 为准。

必须重点看：

```text
validation.eval_f1_macro 或 eval_f1_macro
test.test_f1_macro
runtime.gpu_synchronized_train_time_sec
runtime.total_runtime_sec
runtime.peak_mem_allocated_mb
runtime.peak_mem_reserved_mb
runtime.run_peak_mem_allocated_mb
runtime.run_peak_mem_reserved_mb
parameters.trainable_params
parameters.total_params
model_selection.best_epoch
model_selection.best_model_checkpoint
```

预测文件默认不完整上传到 W&B。需要错误分析时，看本地：

```text
predictions_validation.json
predictions_test.json
```

## 9. 时间和显存怎么算

脚本会在训练前后做 CUDA synchronize，因此 GPU 训练时间更可靠：

```text
gpu_synchronized_train_time_sec
```

含义：

- 只围绕 `trainer.train()` 计时。
- 训练前同步 GPU。
- 训练后再次同步 GPU。
- 不包含下载数据、tokenize、保存 JSON、复制到 Drive 的时间。

总 runtime：

```text
total_runtime_sec
```

含义：

- 从脚本主流程开始到结束。
- 包含数据加载、tokenization、训练、验证、测试、保存文件等。
- 适合估算一次完整 run 在 Colab 中占用多久。

显存：

```text
peak_mem_allocated_mb
peak_mem_reserved_mb
run_peak_mem_allocated_mb
run_peak_mem_reserved_mb
```

含义：

- `peak_mem_allocated_mb` / `peak_mem_reserved_mb`：按 setup 要求，在训练前 reset，
  训练结束后读取的训练峰值显存。
- `run_peak_mem_allocated_mb` / `run_peak_mem_reserved_mb`：训练、验证、测试和保存过程
  全部结束后的整体峰值。
- `allocated`：PyTorch 实际分配给 tensor 的峰值显存。
- `reserved`：PyTorch caching allocator 向 GPU 预留的峰值显存。
- 论文或实验表一般记录训练峰值两者，至少记录 `peak_mem_allocated_mb`。

GPU 型号也必须记录：

```text
gpu_type
```

因为 A100、L4、T4 的时间不能直接公平比较。

数据预处理审计保存在：

```text
metrics.json -> dataset_audit
config_snapshot.json -> dataset_audit
```

它记录每个 split 的 raw examples、processed examples、kept examples、因为没有严格多数而
丢弃的数量，以及是否因为 sample cap 提前停止。

## 10. Case Walkthrough A：Smoke Run

目的：检查环境、依赖、数据、W&B、输出文件是否正常。

什么时候跑：

- 第一次打开 Colab。
- 换 GPU runtime 后。
- 改了脚本之后。
- W&B 或依赖升级后。

修改：

```python
SEARCH_STAGE = "smoke"
TRIAL_ID = "standalone_distilbert_full_ft_smoke_seed42"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "distilbert_full_ft_smoke_seed42"
SEED = 42
NUM_EPOCHS = 1
MAX_TRAIN_SAMPLES = 64
MAX_EVAL_SAMPLES = 64
MAX_TEST_SAMPLES = 64
RUN_TEST = False
WANDB_MODE = "offline"  # 可选；如果想测试线上 W&B，则用 online
```

运行：

```bash
python standalone_distilbert_full_ft/train_distilbert_hatexplain.py
```

什么时候停止：

- 脚本能完成。
- `metrics.json` 存在。
- `predictions_validation.json` 存在。
- W&B run 能看到，或 offline 文件能生成。

不要根据 smoke 的 F1 做任何实验结论。样本太少，分数没有研究意义。

## 11. Case Walkthrough B：单个 Validation Tuning Run

目的：用完整 train/validation 检查一个候选超参数是否合理。

什么时候跑：

- smoke 通过之后。
- 你要正式比较 learning rate 之前。

修改：

```python
SEARCH_STAGE = "tuning"
TRIAL_ID = "standalone_distilbert_full_ft_lr2e-5_seed42"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "distilbert_full_ft_lr2e-5_seed42"
SEED = 42
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None
MAX_TEST_SAMPLES = None
RUN_TEST = False
WANDB_MODE = "online"
```

运行后看：

```text
validation macro-F1
best_epoch
train time
peak memory
failure or completed status
```

什么时候停止并修改：

- 如果训练直接 OOM，先减小 `TRAIN_BATCH_SIZE`。
- 如果 validation loss/F1 明显异常，例如 F1 接近随机且不动，检查 label、数据量、
  学习率和 W&B config。
- 如果训练时间超出预算，记录当前时间后停止，不要静默丢弃失败 trial。
- 如果成功完成，进入 HPO 或 confirmation。

## 12. Case Walkthrough C：Manual HPO

Standalone 不自动调参。HPO 需要手动跑多个候选，并手动记录表格。

推荐 HPO 目标：

```text
选择 validation macro-F1 最好的 learning rate
```

推荐候选：

```text
LEARNING_RATE = 1e-5
LEARNING_RATE = 2e-5
LEARNING_RATE = 3e-5
```

固定不变：

```text
MODEL_NAME
MAX_LENGTH
TRAIN_BATCH_SIZE
EVAL_BATCH_SIZE
NUM_EPOCHS
WEIGHT_DECAY
WARMUP_RATIO
MAX_GRAD_NORM
OPTIM
LR_SCHEDULER_TYPE
EARLY_STOPPING_PATIENCE
EARLY_STOPPING_THRESHOLD
MIXED_PRECISION
GRADIENT_CHECKPOINTING
SEED
RUN_TEST = False
```

每个 HPO trial 的步骤：

1. 改 `LEARNING_RATE`。
2. 改 `TRIAL_ID`，把 lr 写进名字。
3. 改 `OUTPUT_DIR`，避免覆盖前一个 trial。
4. 确认 `SEARCH_STAGE = "tuning"`。
5. 确认 `RUN_TEST = False`。
6. 运行脚本。
7. 从 `run_summary.json` 或最后 printout 复制记录。
8. 保存输出目录到 Drive。
9. 再跑下一个 lr。

HPO 表格至少记录：

```text
method
trial_id
seed
learning_rate
num_epochs
batch_size
max_length
best_epoch
val_macro_f1
train_time_s
total_runtime_s
peak_mem_allocated_mb
peak_mem_reserved_mb
trainable_params
total_params
gpu_type
status
output_dir
wandb_url
notes
```

什么时候停止 HPO：

- 所有预先声明的候选都完成。
- 或某个候选失败，但已经记录 `status=failed` 和失败原因。
- 不要因为某个 lr 第一个 epoch 看起来不好就提前取消，除非已经出现 OOM、NaN、
  runtime 预算明显超标，或脚本错误。

怎么选择：

- 主排序：最高 validation macro-F1。
- 若差距很小，优先选更稳定、更快、显存更低的配置。
- 不要用 test set 选择 learning rate。
- 如果预算允许，保留 validation 排名前 2 的 configs 进入 confirmation；如果预算不足，
  至少清楚记录只 confirmation 了 top-1。

## 13. Case Walkthrough D：Confirmation Runs

目的：确认 HPO 选出来的配置不是 seed luck。严格按 setup 方案，应该对 HPO top-2
configs 都做 confirmation；如果时间不够，至少对 top-1 做 confirmation，并在报告里说明预算限制。

什么时候跑：

- HPO 完成并选定一个 learning rate 后。
- 还没有碰 test set 之前。

建议：

```text
SEED = 42
SEED = 43
```

每个 seed 修改：

```python
SEARCH_STAGE = "confirm"
TRIAL_ID = "standalone_distilbert_full_ft_confirm_lr2e-5_seed42"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "distilbert_full_ft_confirm_lr2e-5_seed42"
LEARNING_RATE = 2e-5
SEED = 42
RUN_TEST = False
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None
```

记录：

```text
config_rank_from_hpo
seed
val_macro_f1
best_epoch
train_time_s
peak_mem_allocated_mb
status
```

什么时候改变配置：

- 如果两个 seed 都明显低于 HPO 最佳 trial，回头检查 HPO 是否偶然、是否输出目录混淆、
  是否改错参数。
- 如果一个 seed 正常，一个 seed 失败，先修复失败原因，再重跑同一个 seed。
- 如果 confirmation 稳定，再进入 final test。

## 14. Case Walkthrough E：Final Test Runs

目的：冻结配置后，只做最终测试集评估。

规则：

- 只能使用已经由 validation 选定的配置。
- 不要根据 test 指标再改 learning rate、epoch、batch size。
- 每个 final seed 都应完整保存 `predictions_test.json`。

建议 final seeds：

```text
42, 43, 44
```

每个 seed 修改：

```python
SEARCH_STAGE = "final"
TRIAL_ID = "standalone_distilbert_full_ft_final_lr2e-5_seed42"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "distilbert_full_ft_final_lr2e-5_seed42"
LEARNING_RATE = 2e-5
SEED = 42
RUN_TEST = True
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None
MAX_TEST_SAMPLES = None
```

最终报告记录：

```text
final_test_macro_f1_mean
final_test_macro_f1_std
final_test_accuracy_mean
final_train_time_mean
final_train_time_std
peak_mem_allocated_mb_mean
peak_mem_allocated_mb_std
trainable_params
total_params
trainable_pct
```

可以用下面的方式手动计算 mean/std：

```python
import statistics as stats

scores = [0.71, 0.70, 0.72]
print("mean", stats.mean(scores))
print("std", stats.stdev(scores))
```

如果只有一个 final seed，只能报告 single-run result，不能写 mean ± std。

## 15. Case Walkthrough F：OOM 或失败时怎么办

如果出现 CUDA out of memory：

1. 记录失败 trial 的 `TRIAL_ID`、GPU、batch size 和错误。
2. 不要删除失败输出目录；保留 `failure.json`。
3. 优先减小 `TRAIN_BATCH_SIZE`。
4. 如 eval 也 OOM，再减小 `EVAL_BATCH_SIZE`。
5. 重新跑同一个 lr/seed，但 `TRIAL_ID` 或 `OUTPUT_DIR` 要能看出是 rerun。

如果出现 NaN 或 loss 异常：

1. 确认 learning rate 是否写错，例如 `2e-5` 误写成 `2e-3`。
2. 确认样本上限是否意外太小。
3. 确认标签策略没有改。
4. 先做 smoke，再回到完整 tuning。

如果 W&B 失败：

1. 改成 `WANDB_MODE = "offline"` 或 `USE_WANDB = False`。
2. 继续确保本地 JSON 生成。
3. 后续可以手动上传或只用本地记录。

失败也要记录，因为 HPO 成本应包括 failed trials。

## 16. Case Walkthrough G：预测文件错误分析

什么时候看预测文件：

- validation F1 低于预期。
- 某一类 precision/recall 很差。
- final test 后需要写 qualitative error analysis。

看哪个文件：

```text
predictions_validation.json   tuning/confirmation 阶段
predictions_test.json         final 阶段
```

每条记录应关注：

```text
id
text
label_name
prediction_name
probabilities
```

建议人工记录：

- 模型把 hatespeech 误判成 offensive 的例子。
- 模型把 offensive 误判成 normal 的例子。
- 低置信度样本。
- 文本很短、讽刺、拼写异常或上下文不足的样本。

不要在看 test 错误后继续调参；test 错误分析只能用于报告，不用于选择配置。

## 17. 最终模型保存和交付

最终模型位置：

```text
outputs/distilbert_full_ft/final_model/
```

应确认里面至少有：

```text
config.json
model.safetensors 或 pytorch_model.bin
tokenizer files
```

最终交付应包含：

```text
config_snapshot.json
metrics.json
run_summary.json
trainer_log_history.json
predictions_validation.json
predictions_test.json
final_model/
```

如果 final run 没有 `predictions_test.json`，通常说明 `RUN_TEST=False`，需要检查后重跑。

## 18. 最小记录模板

每个 trial 都记录一行：

```text
method:
trial_id:
search_stage:
seed:
learning_rate:
epochs:
batch_size:
max_length:
optim:
lr_scheduler_type:
max_grad_norm:
early_stopping_patience:
early_stopping_threshold:
mixed_precision:
gradient_checkpointing:
run_test:
best_epoch:
val_macro_f1:
test_macro_f1:
gpu_synchronized_train_time_sec:
total_runtime_sec:
peak_mem_allocated_mb:
peak_mem_reserved_mb:
trainable_params:
total_params:
gpu_type:
status:
output_dir:
wandb_url:
notes:
```

HPO 阶段 `test_macro_f1` 应为空，因为 `RUN_TEST=False`。

## 19. 正式运行前 Checklist

运行前：

- `TRIAL_ID` 是否唯一。
- `OUTPUT_DIR` 是否唯一。
- `SEARCH_STAGE` 是否正确。
- `RUN_TEST` 是否符合阶段。
- `LEARNING_RATE` 是否是当前候选。
- `SEED` 是否正确。
- 样本上限是否为预期。
- W&B mode 是否为预期。
- Colab 是否是 GPU runtime。

运行后：

- `metrics.json` 存在。
- `run_summary.json` 存在。
- `config_snapshot.json` 存在。
- `predictions_validation.json` 存在。
- final run 时 `predictions_test.json` 存在。
- `final_model/` 存在。
- `Manual record fields` 已复制到实验表。
- 输出目录已复制到 Google Drive。

## 20. 常见问题

### 分数很低是不是失败？

Smoke run 分数低是正常的。完整 validation/final run 才有研究意义。

### 可以一边 HPO 一边看 test 吗？

不可以。HPO 和 confirmation 只看 validation。test 只在 final 阶段使用。

### W&B 里没有 prediction 文件怎么办？

正常。预测文件主要保存在本地 JSON。W&B 负责曲线、指标和配置对比。

### 为什么同样配置时间差很多？

可能 GPU 型号不同，也可能 Colab runtime 状态不同。报告时间时必须同时记录
`gpu_type`，不要直接比较不同 GPU 的 wall time。

### 改 batch size 会不会影响实验？

会。batch size 是超参数和计算成本的一部分。如果为了 OOM 调小 batch size，必须记录。

### 可以改 `MAX_LENGTH` 吗？

可以，但这会改变输入截断策略。除非是新的明确实验，否则 full fine-tuning 主实验建议固定
`MAX_LENGTH = 128`。

### 什么时候可以删输出目录？

只有确认结果已经复制到 Drive、W&B 已记录、实验表已填好之后，才可以清理本地输出。
不要删除失败 trial 的记录，除非已经在表格中记录失败原因。
