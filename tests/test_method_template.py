import subprocess
import sys
import unittest
from pathlib import Path


class MethodTemplateTests(unittest.TestCase):
    def test_template_help_exposes_shared_method_contract(self):
        result = subprocess.run(
            [sys.executable, "src/methods/_template/train.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        )

        for option in (
            "--method",
            "--search_stage",
            "--trial_id",
            "--dataset_name",
            "--output_dir",
            "--use_wandb",
            "--wandb_project",
            "--mixed_precision",
            "--gradient_checkpointing",
            "--class_weighting",
        ):
            self.assertIn(option, result.stdout)


if __name__ == "__main__":
    unittest.main()
