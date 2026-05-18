import unittest
from pathlib import Path

from src.experiments.hpo import load_hpo_config
from src.experiments.protocol import (
    EXPECTED_METHODS,
    validate_experiment_protocol,
)
from src.experiments.registry import load_experiment_registry


class ProtocolConformanceTests(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]
        self.registry = load_experiment_registry(
            self.repo_root / "configs" / "experiments.json"
        )
        self.hpo_config = load_hpo_config(
            self.repo_root / "configs" / "search_spaces.json"
        )

    def test_project_protocol_config_is_valid(self):
        report = validate_experiment_protocol(
            self.registry,
            self.hpo_config,
            repo_root=self.repo_root,
        )

        self.assertEqual(report.errors, [])
        self.assertTrue(report.is_valid)

    def test_every_expected_method_has_catalog_template_and_search_space(self):
        experiment_methods = {spec.method for spec in self.registry.experiments}
        search_spaces = set(self.hpo_config["search_spaces"])

        for method in EXPECTED_METHODS:
            self.assertIn(method.method_id, experiment_methods)
            self.assertIn(method.search_space, search_spaces)

    def test_validation_catches_non_final_test_evaluation(self):
        bad_registry = load_experiment_registry(
            self.repo_root / "configs" / "experiments.json"
        )
        target = bad_registry.get("distilbert_full_tuning")
        target.args["run_test"] = True

        report = validate_experiment_protocol(
            bad_registry,
            self.hpo_config,
            repo_root=self.repo_root,
        )

        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("run_test" in message and "final" in message for message in report.errors)
        )

    def test_validation_catches_final_without_test_evaluation(self):
        bad_registry = load_experiment_registry(
            self.repo_root / "configs" / "experiments.json"
        )
        target = bad_registry.get("distilbert_full_final_seed42")
        target.args["run_test"] = False

        report = validate_experiment_protocol(
            bad_registry,
            self.hpo_config,
            repo_root=self.repo_root,
        )

        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("Final experiment" in message and "run_test" in message for message in report.errors)
        )

    def test_validation_catches_missing_expected_method_stage(self):
        bad_registry = load_experiment_registry(
            self.repo_root / "configs" / "experiments.json"
        )
        target = bad_registry.get("random_init_distilbert_template")
        object.__setattr__(target, "stage", "smoke")

        report = validate_experiment_protocol(
            bad_registry,
            self.hpo_config,
            repo_root=self.repo_root,
        )

        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("random-init-distilbert" in message and "template" in message for message in report.errors)
        )

    def test_validation_catches_sample_caps_on_tuning_entries(self):
        bad_registry = load_experiment_registry(
            self.repo_root / "configs" / "experiments.json"
        )
        target = bad_registry.get("distilbert_full_tuning")
        target.args["max_train_samples"] = 64

        report = validate_experiment_protocol(
            bad_registry,
            self.hpo_config,
            repo_root=self.repo_root,
        )

        self.assertFalse(report.is_valid)
        self.assertTrue(
            any("max_train_samples" in message and "tuning" in message for message in report.errors)
        )

    def test_validation_catches_invalid_time_caps(self):
        bad_config = {
            **self.hpo_config,
            "time_caps_gpu_hours": {"full_ft": 0},
        }

        report = validate_experiment_protocol(
            self.registry,
            bad_config,
            repo_root=self.repo_root,
        )

        self.assertFalse(report.is_valid)
        self.assertTrue(any("time_caps_gpu_hours.full_ft" in message for message in report.errors))


if __name__ == "__main__":
    unittest.main()
