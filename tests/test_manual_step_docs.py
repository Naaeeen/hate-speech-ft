from pathlib import Path
import unittest


STEP_DOCS = {
    "tfidf-logreg": {
        "file": "tfidf_logreg_steps.md",
        "config": "src/methods/tfidf_logreg/manual_config.py",
        "train": "src/methods/tfidf_logreg/train.py",
        "wandb_keys": ("eval_f1_macro", "test_f1_macro"),
    },
    "bilstm": {
        "file": "bilstm_steps.md",
        "config": "src/methods/bilstm/manual_config.py",
        "train": "src/methods/bilstm/train.py",
        "wandb_keys": ("eval_f1_macro", "test_f1_macro"),
    },
    "full-ft": {
        "file": "fullfttsteps.md",
        "config": "src/methods/distilbert_full/manual_config.py",
        "train": "src/methods/distilbert_full/train.py",
        "wandb_keys": ("eval/f1_macro", "test/f1_macro"),
    },
    "frozen-backbone": {
        "file": "frozen_distilbert_steps.md",
        "config": "src/methods/frozen_distilbert/manual_config.py",
        "train": "src/methods/frozen_distilbert/train.py",
        "wandb_keys": ("eval/f1_macro", "test/f1_macro"),
    },
    "lora": {
        "file": "distilbert_lora_steps.md",
        "config": "src/methods/distilbert_lora/manual_config.py",
        "train": "src/methods/distilbert_lora/train.py",
        "wandb_keys": ("eval/f1_macro", "test/f1_macro"),
    },
    "lp-ft": {
        "file": "distilbert_lp_ft_steps.md",
        "config": "src/methods/distilbert_lp_ft/manual_config.py",
        "train": "src/methods/distilbert_lp_ft/train.py",
        "wandb_keys": ("eval/f1_macro", "test/f1_macro", "stage1/eval/f1_macro"),
    },
    "efficient-head-ft": {
        "file": "distilbert_efficient_head_steps.md",
        "config": "src/methods/distilbert_efficient_head/manual_config.py",
        "train": "src/methods/distilbert_efficient_head/train.py",
        "wandb_keys": ("eval/f1_macro", "test/f1_macro", "stage1/eval/f1_macro"),
    },
}


OLD_AUTOMATION_TERMS = (
    "src/run_experiment.py",
    "experiment_launcher",
    "aggregate_results",
    "--experiment",
    "--set",
    "failure_summary",
    "best-effort",
)


class ManualStepDocsTests(unittest.TestCase):
    def test_each_step_doc_points_to_one_manual_config_and_train_script(self):
        for method, spec in STEP_DOCS.items():
            with self.subTest(method=method):
                text = Path(spec["file"]).read_text(encoding="utf-8")

                self.assertIn(spec["config"], text)
                self.assertIn(f"python {spec['train']}", text)
                self.assertIn(f"method = {method}", text)
                self.assertIn("dataset_name = Hate-speech-CNERG/hatexplain", text)
                self.assertIn("seed = 42", text)
                self.assertIn("run_test = True", text)

    def test_step_docs_describe_the_current_output_contract(self):
        required_outputs = (
            "resolved_config.json",
            "metrics.json",
            "runtime.json",
            "result_summary.json",
            "eval_predictions.json        # when run_test=True",
            "test_predictions.json        # when run_test=True",
            "selected_hyperparams_json <- result_summary.config.hyperparameters",
        )

        for method, spec in STEP_DOCS.items():
            with self.subTest(method=method):
                text = Path(spec["file"]).read_text(encoding="utf-8")
                for output in required_outputs:
                    self.assertIn(output, text)

    def test_step_docs_keep_hpo_as_print_only_suggestions(self):
        for method, spec in STEP_DOCS.items():
            with self.subTest(method=method):
                text = Path(spec["file"]).read_text(encoding="utf-8")

                self.assertIn("python src/hpo_random_search.py", text)
                self.assertIn("manual_config_updates", text)
                for old_term in OLD_AUTOMATION_TERMS:
                    self.assertNotIn(old_term, text)

    def test_step_docs_use_the_right_wandb_key_shape(self):
        for method, spec in STEP_DOCS.items():
            with self.subTest(method=method):
                text = Path(spec["file"]).read_text(encoding="utf-8")
                for key in spec["wandb_keys"]:
                    self.assertIn(key, text)

    def test_two_stage_docs_match_the_actual_model_selection_shape(self):
        for method in ("lp-ft", "efficient-head-ft"):
            with self.subTest(method=method):
                text = Path(STEP_DOCS[method]["file"]).read_text(encoding="utf-8")

                self.assertIn("stage1_best_model_checkpoint", text)
                self.assertIn("best_model_checkpoint", text)
                self.assertIn("stage2_best_metric", text)
                self.assertNotIn("stage2_best_model_checkpoint", text)

    def test_training_args_bin_is_not_documented_as_required(self):
        for filename in ("fullfttsteps.md", "frozen_distilbert_steps.md"):
            with self.subTest(file=filename):
                text = Path(filename).read_text(encoding="utf-8")

                self.assertIn("training_args.bin", text)
                self.assertIn("may also appear", text)
                self.assertNotIn("training_args.bin\ncheckpoint-*", text)

    def test_colab_docs_do_not_pull_over_manual_config_edits(self):
        notebook_text = Path("notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb").read_text(
            encoding="utf-8"
        )
        readme_text = Path("notebooks/README.md").read_text(encoding="utf-8")

        self.assertNotIn("pull --ff-only", notebook_text)
        self.assertIn("skipping git pull", notebook_text)
        self.assertIn("manual_config.py edits stay untouched", notebook_text)
        self.assertIn("should not\nrun `git pull`", readme_text)

    def test_wandb_docs_include_disabled_mode(self):
        text = Path("docs/WANDB.md").read_text(encoding="utf-8")

        self.assertIn('"wandb_mode": "disabled"', text)
        self.assertIn("skip W&B completely", text)

    def test_root_docs_name_the_checked_in_historical_csvs(self):
        for filename in ("README.md", "docs/MANUAL_METRICS.md"):
            with self.subTest(file=filename):
                text = Path(filename).read_text(encoding="utf-8")

                self.assertIn("results/all/final_runs (1).csv", text)
                self.assertIn("results/all/hpo_runs.csv", text)
                self.assertIn("results/all/method_summary (1).csv", text)


if __name__ == "__main__":
    unittest.main()
