# DistilBERT LP+FT 集成说明

这份文档说明 Chris 的 DistilBERT LP+FT 代码是如何接入当前统一实验
pipeline 的。

## 修改了什么

- 新增并启用这些 catalog 实验：
  - `distilbert_lp_ft_smoke`
  - `distilbert_lp_ft_quick`
  - `distilbert_lp_ft_tuning`
  - `distilbert_lp_ft_final_seed42`
- LP+FT 的训练逻辑仍然保留在
  `src/methods/distilbert_lp_ft/` 方法包内。
- 复用项目已有的统一机制：实验 catalog、命令构造、HPO 元数据、W&B
  设置、输出目录保护、final-only test policy、结果 JSON 文件格式。
- 新增共享 helper：
  - `src/methods/transformer_data.py`：Transformer 方法共用的 HateXplain
    tokenization 和 split accounting。
  - `src/methods/predictions.py`：final 阶段共用的逐样本 prediction JSON
    写出逻辑。
- 更新 protocol validation，让 LP+FT 作为 ready method 使用 `lp_ft`
  search space。
- 更新测试和文档，使 Colab notebook 可以像 full FT 一样列出、preview 和
  运行 LP+FT。

## 为什么这样重构

原来的 LP+FT 脚本虽然能接收一部分共享 CLI 参数，但有些参数没有真正被
训练流程使用；同时它会在非 final 阶段读取并评估 test split，这会造成
调参阶段的数据泄漏。

重构后的边界是：

- 共享 pipeline 负责：catalog、命令、HPO/W&B 元数据、数据策略、输出文件、
  aggregation 兼容性。
- LP+FT 方法包负责：stage 1 冻结策略、stage 2 解冻策略、两阶段 Trainer
  流程、LP+FT 自己的超参数。

这样可以让 LP+FT 和 full FT 可比较，同时不会把方法细节硬编码进
`src/run_experiment.py` 或 Colab notebook。

## 当前 LP+FT 如何训练

Stage 1：linear probing

- 冻结 DistilBERT backbone。
- 只训练 `pre_classifier` 和 `classifier`。
- 使用 `stage1_head_learning_rate` 和 `stage1_epochs`。
- checkpoint 写到 `output_dir/stage1_linear_probe/`。

Stage 2：full fine-tuning

- 解冻全部参数。
- 从 stage 1 训练后的模型继续训练。
- 使用 `stage2_learning_rate` 和 `stage2_epochs`。
- checkpoint 写到 `output_dir/stage2_full_ft/`。
- final model 和 tokenizer 保存到 `output_dir` 根目录。

非 final 阶段只使用 train/validation。Final 阶段必须带 `--run_test`，
并会写出 validation 和 test 的 prediction 文件。

## 如何运行

列出实验：

```bash
python src/run_experiment.py --list
```

Preview LP+FT smoke：

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_smoke \
  --dry_run
```

运行 LP+FT smoke：

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_smoke
```

生成 HPO trial 命令：

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_tuning \
  --suggest_trials 4 \
  --search_space lp_ft \
  --hpo_seed 42
```

选出最佳配置后生成 final seed 命令：

```bash
python src/run_experiment.py \
  --experiment distilbert_lp_ft_tuning \
  --suggest_seed_runs final \
  --set stage1_head_learning_rate=1e-4 \
  --set stage1_epochs=5 \
  --set stage2_learning_rate=2e-5 \
  --set stage2_epochs=2
```

Colab notebook 读取同一个 `configs/experiments.json`，所以这些 LP+FT
实验会自动出现在下拉菜单里，不需要在 notebook 里硬编码方法逻辑。

## 需要查看和记录什么

每次 completed run 会在 `output_dir` 写：

```text
resolved_config.json
metrics.json
runtime.json
result_summary.json
```

Final run 还会写：

```text
eval_predictions.json
test_predictions.json
```

如果启用 `--use_wandb`，W&B 会记录同一套 config 和 metrics。但本地 JSON
仍然是 aggregation 和最终报告的主要依据。

## 后续如何扩展

如果要改 LP+FT：

- 主流程：`src/methods/distilbert_lp_ft/train.py`
- 两阶段训练 helper：`src/methods/distilbert_lp_ft/training.py`
- resolved config 结构：`src/methods/distilbert_lp_ft/config.py`
- HPO 搜索空间：`configs/search_spaces.json` 里的 `lp_ft`
- catalog 入口：`configs/experiments.json`
- 共享 launcher 行为：`src/run_experiment.py` 和 `src/experiments/*`

不要把 LP+FT 的 stage 逻辑放进 `src/run_experiment.py` 或 Colab notebook。
共享层只负责调度和记录，方法细节应该留在方法自己的目录里。
