import subprocess
import sys
import unittest
from argparse import ArgumentParser
from pathlib import Path
from tempfile import TemporaryDirectory

from src.methods.common import add_common_method_arguments
from src.methods.scaffold import build_catalog_snippet, scaffold_method


class MethodScaffoldTests(unittest.TestCase):
    def test_common_method_arguments_include_tracking_and_wandb_flags(self):
        parser = add_common_method_arguments(ArgumentParser())
        option_strings = {
            option
            for action in parser._actions
            for option in action.option_strings
        }

        for option in (
            "--method",
            "--search_stage",
            "--trial_id",
            "--dataset_name",
            "--output_dir",
            "--overwrite_output_dir",
            "--use_wandb",
            "--wandb_entity",
            "--wandb_project",
            "--wandb_group",
            "--wandb_tags",
            "--wandb_mode",
            "--wandb_log_model",
            "--run_test",
            "--max_length",
            "--weight_decay",
            "--warmup_ratio",
            "--max_grad_norm",
            "--optim",
            "--lr_scheduler_type",
            "--logging_strategy",
            "--logging_steps",
            "--eval_steps",
            "--save_steps",
            "--mixed_precision",
            "--gradient_checkpointing",
            "--class_weighting",
        ):
            self.assertIn(option, option_strings)

    def test_scaffold_method_creates_package_without_touching_catalog(self):
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp)

            created = scaffold_method(
                repo_root=repo_root,
                method_package="distilbert_lora",
                method_id="lora",
                family="transformer-peft",
                description="LoRA DistilBERT method.",
            )

            train_path = repo_root / "src" / "methods" / "distilbert_lora" / "train.py"
            readme_path = repo_root / "src" / "methods" / "distilbert_lora" / "README.md"
            catalog_path = repo_root / "configs" / "experiments.json"

            self.assertIn(train_path, created)
            self.assertIn(readme_path, created)
            self.assertFalse(catalog_path.exists())
            self.assertIn(
                '_METHOD_ID_PLACEHOLDER = "lora"',
                train_path.read_text(encoding="utf-8"),
            )
            self.assertIn("LoRA DistilBERT method.", readme_path.read_text(encoding="utf-8"))

    def test_scaffold_method_refuses_to_overwrite_existing_package(self):
        with TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            scaffold_method(
                repo_root=repo_root,
                method_package="bilstm",
                method_id="bilstm",
                family="neural-scratch",
                description="Bi-LSTM baseline.",
            )

            with self.assertRaises(FileExistsError):
                scaffold_method(
                    repo_root=repo_root,
                    method_package="bilstm",
                    method_id="bilstm",
                    family="neural-scratch",
                    description="Bi-LSTM baseline.",
                )

    def test_catalog_snippet_points_to_scaffolded_train_script(self):
        snippet = build_catalog_snippet(
            method_package="tfidf_logreg",
            method_id="tfidf-logreg",
            family="classical",
            description="TF-IDF + Logistic Regression baseline.",
        )

        self.assertEqual(snippet["status"], "planned")
        self.assertEqual(snippet["method"], "tfidf-logreg")
        self.assertEqual(snippet["script"], "src/methods/tfidf_logreg/train.py")
        self.assertEqual(snippet["args"]["output_dir"], "outputs/tfidf_logreg_template")

    def test_template_help_runs_from_repo_root(self):
        result = subprocess.run(
            [sys.executable, "src/methods/_template/train.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        )

        self.assertIn("--method", result.stdout)
        self.assertIn("--wandb_project", result.stdout)


if __name__ == "__main__":
    unittest.main()
