# Bi-LSTM Pipeline 集成说明

这份文档说明 Minh 的 Bi-LSTM baseline 现在如何接入 shared experiment
pipeline。

## 改了什么

原始 Bi-LSTM 更像一个方法内部独立脚本：自己加载 HateXplain，只支持部分
launcher 参数，输出文件格式不完全统一，也没有完整的 output directory 保护、
失败记录和 final-only test 约束。

现在的版本保留 Bi-LSTM 自己的模型和 torch training loop，但接入了和
DistilBERT full FT、DistilBERT LP-FT、TF-IDF LogReg 一样的实验协议：

- `configs/experiments.json` 里有 ready entries：
  `bilstm_smoke`、`bilstm_quick`、`bilstm_tuning`、
  `bilstm_final_seed42`。
- `configs/search_spaces.json` 里有 Bi-LSTM 的 HPO search space 和 8-trial
  cap。
- `src/run_experiment.py` 可以 list、preview、run、生成 HPO trials，也可以
  生成 final seed runs。
- 现在没有方法内部的 `src/methods/bilstm/hpo.py`；HPO 搜索空间、trial cap、
  seed policy、输出路径和 config hash 都由共享 launcher 配置统一管理。
- Bi-LSTM 复用 shared final-only test policy、output-dir overwrite guard、
  W&B settings、result file names、failure summary 和 HPO identity fields。
- Bi-LSTM 目前只在本地保存 model artifacts；`--wandb_log_model` 必须保持
  `false`。在这个自定义 torch runner 实现 W&B artifact upload 之前，CLI
  和 Colab launcher 会拒绝 `end` 或 `checkpoint`。

## 文件职责

`src/methods/bilstm/args.py`

- 通过 `src.methods.common.add_common_method_arguments()` 接入 shared CLI flags。
- 增加 Bi-LSTM 自己的参数，例如 `embedding_size`、`hidden_size`、
  `num_layers`、`dropout`、`learning_rate`、`batch_size`、
  `eval_batch_size`、`epochs`。
- 参数校验不依赖训练库，所以即使还没装完整训练依赖，
  `python src/methods/bilstm/train.py --help` 也能正常工作。

`src/methods/bilstm/data.py`

- 使用 `src.data.preprocessing` 的统一 HateXplain 数据策略。
- 使用官方 `train`、`validation`、`test` split。
- 应用 strict-majority label filtering、确定性的 `data_fraction` 和
  `max_*_samples` 截断。
- 防止 validation split 和 test split 意外混用。

`src/methods/bilstm/model.py`

- 定义 torch `BiLSTMClassifier`。
- 模型结构仍然属于 Bi-LSTM 方法内部，不写进 shared pipeline。

`src/methods/bilstm/tokenizer.py`

- 包装 DistilBERT tokenizer，让 Bi-LSTM 使用可比较的 token ids，但 embedding
  仍然是随机初始化并从头训练。

`src/methods/bilstm/training.py`

- 负责 torch training loop、class weights、AdamW、linear scheduler、
  gradient clipping、epoch checkpoints、early stopping、validation/test
  metrics、final model save 和 prediction file writer。
- 不在 module import 阶段加载 `datasets` 或 `transformers`，避免 preview/help
  被训练依赖卡住。

`src/methods/bilstm/config.py`

- 生成 `resolved_config.json`、runtime metadata、W&B run name 和 model
  selection summary。
- runtime cost 只有在实际 training device 是 `cuda` 时才统计 GPU-hours。CPU
  Bi-LSTM run 仍会记录可用 GPU type 作为环境信息，但 compute cost 记录为 CPU。

`src/methods/bilstm/train.py`

- 是 catalog 调用的入口。
- 做启动校验、output directory 保护、W&B 初始化、数据加载、训练、标准结果
  文件写入；如果失败，会写 `failure_summary.json`。

## 运行路径

当你运行：

```bash
python src/run_experiment.py --experiment bilstm_smoke --dry_run
```

launcher 会读取 `configs/experiments.json`，定位 `bilstm_smoke`，合并
command defaults、`neural-scratch` family defaults、entry args 和所有 `--set`
overrides，然后打印一个指向下面脚本的命令：

```text
src/methods/bilstm/train.py
```

真正运行该命令时，`train.py` 会：

1. 解析 shared flags 和 Bi-LSTM-specific flags。
2. 校验 final/test policy 和 output directory safety。
3. 如果传了 `--use_wandb`，初始化 W&B。
4. 加载 HateXplain 并应用 shared preprocessing。
5. 创建 tokenizer 和 Bi-LSTM model。
6. 用 validation macro-F1 做 checkpoint selection。
7. 保存 final model、tokenizer、runtime、metrics、summary，以及 final-stage
   prediction files。

## 输出文件约定

每个成功的 Bi-LSTM run 会写：

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
model.pt                 # 除非使用 --no_save_final_model
tokenizer/
checkpoint-epoch*/       # 受 save_total_limit 控制
```

final-stage runs 还会写：

```text
eval_predictions.json
test_predictions.json
```

失败的 run 会写 `failure_summary.json`，并先清理目标目录里旧的 managed
success artifacts，避免旧 metrics 被误认为本次结果。

## 多次运行安全性

- HPO trial generation 生成的 `trial_id` 和 `output_dir` 会包含 HPO seed、
  trial index 和最终 `config_hash`。
- confirm 和 final seed generation 会按 selected config hash 和 seed 生成独立
  output directory。
- 如果 `output_dir` 已经有 managed artifacts，默认拒绝启动；只有显式传
  `--overwrite_output_dir` 才会覆盖。
- overwrite cleanup 包括 Bi-LSTM 的 `model.pt`、旧的 `finalmodel.pt`、
  tokenizer directory、checkpoints、summary files 和 prediction files。
- test set 只能在 `search_stage=final` 时运行。
