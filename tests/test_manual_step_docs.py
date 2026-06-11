import json
from pathlib import Path
import unittest


STEP_DOCS = {
    "tfidf-logreg": {
        "file": "tfidf_logreg_steps.md",
        "config": "src/methods/tfidf_logreg/manual_config.py",
        "train": "src/methods/tfidf_logreg/train.py",
        "wandb_keys": ("eval/f1_macro", "test/f1_macro"),
    },
    "bilstm": {
        "file": "bilstm_steps.md",
        "config": "src/methods/bilstm/manual_config.py",
        "train": "src/methods/bilstm/train.py",
        "wandb_keys": ("train/loss", "eval/f1_macro", "test/f1_macro"),
    },
    "full-ft": {
        "file": "distilbert_full_steps.md",
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
        "wandb_keys": (
            "eval/f1_macro",
            "test/f1_macro",
            "stage1/train/loss",
            "stage2/train/loss",
            "stage1/eval/f1_macro",
        ),
    },
    "efficient-head-ft": {
        "file": "distilbert_efficient_head_steps.md",
        "config": "src/methods/distilbert_efficient_head/manual_config.py",
        "train": "src/methods/distilbert_efficient_head/train.py",
        "wandb_keys": (
            "eval/f1_macro",
            "test/f1_macro",
            "stage1/train/loss",
            "stage2/train/loss",
            "stage1/eval/f1_macro",
        ),
    },
}


OLD_AUTOMATION_TERMS = (
    "src/run_" "experiment.py",
    "experiment_" "laun" "cher",
    "aggregate_" "results",
    "--" "experiment",
    "--" "set",
    "failure_" "summary",
    "best-" "effort",
)

NO_REFERENCE_COMPARISON_TERMS = (
    "hist" "orical_sampled_hparams_json",
    "hist" "orical",
    "lega" "cy",
    "main " "branch",
)

DOC_AND_SOURCE_TEXT_FILES = [
    "README.md",
    "docs/README.md",
    "docs/WANDB.md",
    "docs/MANUAL_METRICS.md",
    "docs/ADDING_METHOD.md",
    "notebooks/README.md",
    "src/colab/README.md",
    "src/methods/README.md",
    "src/methods/distilbert_efficient_head/README.md",
    "src/methods/distilbert_full/README.md",
    "src/methods/distilbert_lora/README.md",
    "src/methods/distilbert_lp_ft/README.md",
    "src/methods/tfidf_logreg/README.md",
    *[spec["file"] for spec in STEP_DOCS.values()],
    "src/hpo_random_search.py",
    "src/methods/transformer_config.py",
    "src/methods/transformer_data.py",
    "src/methods/transformer_runner.py",
]


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

    def test_each_step_doc_has_colab_notebook_walkthrough(self):
        required_text = (
            "## Colab Notebook Walkthrough",
            "notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb",
            "METHOD_SCRIPT",
            "MANUAL_CONFIG_MODULE",
            "MANUAL_CONFIG_FILE",
            "manual_config.py",
            "output_dir",
            "metrics.json",
            "runtime.json",
            "result_summary.json",
        )

        for method, spec in STEP_DOCS.items():
            with self.subTest(method=method):
                text = Path(spec["file"]).read_text(encoding="utf-8")
                for expected in required_text:
                    self.assertIn(expected, text)

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

    def test_method_readmes_match_current_method_behavior(self):
        methods_readme = Path("src/methods/README.md").read_text(encoding="utf-8")
        self.assertIn("train-split word tokenizer used by Bi-LSTM", methods_readme)
        self.assertIn("lowercase word vocabulary", methods_readme)
        self.assertNotIn("DistilBERT tokenizer wrapper used by Bi-LSTM", methods_readme)

        for filename in (
            "src/methods/distilbert_lp_ft/README.md",
            "src/methods/distilbert_efficient_head/README.md",
        ):
            with self.subTest(file=filename):
                text = Path(filename).read_text(encoding="utf-8")
                self.assertIn("parent run", text)
                self.assertIn("stage1/train/loss", text)
                self.assertIn("stage2/eval/f1_macro", text)

    def test_training_args_bin_is_not_documented_as_required(self):
        for filename in ("distilbert_full_steps.md", "frozen_distilbert_steps.md"):
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

    def test_colab_notebook_explains_the_manual_workflow(self):
        notebook = json.loads(
            Path("notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb").read_text(
                encoding="utf-8"
            )
        )
        text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

        self.assertIn("# Hate Speech FT Colab Run Sheet", text)
        self.assertIn("## Optional: Get HPO Suggestions", text)
        self.assertIn("manual_config_updates", text)
        self.assertIn("## If I Already Have Hyperparameters", text)
        self.assertIn("METHOD_SCRIPT", text)
        self.assertIn("MANUAL_CONFIG_MODULE", text)
        self.assertIn("MANUAL_CONFIG_FILE", text)
        self.assertIn('"wandb_mode": "offline"', text)
        self.assertIn('"wandb_mode": "disabled"', text)
        self.assertIn("## 12. Where To Look After Training", text)
        self.assertIn("result_summary.json", text)
        self.assertIn("runtime.json", text)
        self.assertIn("W&B should show the live tracking view", text)

    def test_colab_notebook_has_text_for_every_code_cell(self):
        notebook = json.loads(
            Path("notebooks/hate_speech_ft_COLAB_EXAMPLE.ipynb").read_text(
                encoding="utf-8"
            )
        )
        cells = notebook["cells"]
        for index, cell in enumerate(cells):
            if cell["cell_type"] != "code":
                continue
            with self.subTest(code_cell=index):
                self.assertGreater(index, 0)
                self.assertEqual(cells[index - 1]["cell_type"], "markdown")

    def test_docs_and_comments_do_not_use_reference_comparison_language(self):
        for filename in DOC_AND_SOURCE_TEXT_FILES:
            with self.subTest(file=filename):
                text = Path(filename).read_text(encoding="utf-8").lower()
                for term in NO_REFERENCE_COMPARISON_TERMS:
                    self.assertNotIn(term, text)

    def test_wandb_docs_include_disabled_mode(self):
        text = Path("docs/WANDB.md").read_text(encoding="utf-8")

        self.assertIn('"wandb_mode": "disabled"', text)
        self.assertIn("skip W&B completely", text)


if __name__ == "__main__":
    unittest.main()
