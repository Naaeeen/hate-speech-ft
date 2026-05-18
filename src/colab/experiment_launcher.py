from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from src.experiments.aggregate_results import (
    build_aggregate_report,
    write_aggregate_report,
)
from src.experiments.hpo import (
    DEFAULT_SEARCH_SPACE_PATH,
    build_seed_run_overrides,
    build_trial_overrides,
    default_search_space_name,
    get_seed_policy,
    get_trial_cap,
    get_search_space,
    load_hpo_config,
    merge_trial_overrides,
    SEED_RUN_PROTECTED_USER_OVERRIDE_KEYS,
    shared_fixed_command_overrides,
    validate_seed_run_base_stage,
)
from src.experiments.registry import (
    DEFAULT_REGISTRY_PATH,
    REPO_ROOT,
    build_experiment_command,
    format_command,
    load_experiment_registry,
    parse_override_text,
)


class ExperimentLauncher:
    def __init__(
        self,
        *,
        config_path: str | Path = DEFAULT_REGISTRY_PATH,
        search_config_path: str | Path = DEFAULT_SEARCH_SPACE_PATH,
        default_entity: str = "",
        default_project: str = "hate-speech-ft",
        include_planned: bool = False,
    ) -> None:
        import ipywidgets as widgets

        self._widgets = widgets
        self.config_path = Path(config_path)
        self.search_config_path = Path(search_config_path)
        self.registry = load_experiment_registry(self.config_path)
        self.search_config = load_hpo_config(self.search_config_path)
        experiments = (
            self.registry.experiments
            if include_planned
            else self.registry.ready_experiments()
        )
        if not experiments:
            raise ValueError(f"No experiments found in {self.config_path}")

        self.experiment = widgets.Dropdown(
            options=[
                (
                    f"{spec.experiment_id} [{spec.status}] - {spec.description}",
                    spec.experiment_id,
                )
                for spec in experiments
            ],
            description="Experiment",
            layout=widgets.Layout(width="760px"),
        )
        self.use_wandb = widgets.Checkbox(value=True, description="Use W&B")
        self.wandb_mode = widgets.Dropdown(
            options=["online", "offline", "disabled"],
            value="online",
            description="Mode",
        )
        self.wandb_entity = widgets.Text(
            value=default_entity,
            placeholder="team or username",
            description="Entity",
        )
        self.wandb_project = widgets.Text(
            value=default_project,
            description="Project",
        )
        self.wandb_group = widgets.Text(value="", description="Group")
        self.wandb_tags = widgets.Text(
            value="",
            placeholder="leave blank to use experiment tags",
            description="Tags",
        )
        self.wandb_log_model = widgets.Dropdown(
            options=["false", "end", "checkpoint"],
            value="false",
            description="Log model",
        )
        self.overwrite_output_dir = widgets.Checkbox(
            value=False,
            description="Overwrite output",
        )
        self.overrides = widgets.Textarea(
            value="",
            placeholder="Optional one per line, e.g.\nlearning_rate=3e-5\nseed=43\noutput_dir=/content/drive/MyDrive/hate_speech_ft/outputs/my_run",
            description="Overrides",
            layout=widgets.Layout(width="760px", height="120px"),
        )
        self.suggest_trials = widgets.IntText(
            value=0,
            description="Trials",
        )
        self.search_space = widgets.Text(
            value="",
            placeholder="blank uses method name",
            description="Search",
        )
        self.hpo_seed = widgets.IntText(value=42, description="HPO seed")
        self.trial_output_root = widgets.Text(
            value="/content/drive/MyDrive/hate_speech_ft/outputs/hpo",
            description="Trial root",
            layout=widgets.Layout(width="520px"),
        )
        self.seed_run_stage = widgets.Dropdown(
            options=["none", "confirm", "final"],
            value="none",
            description="Seed runs",
        )
        self.seed_output_root = widgets.Text(
            value="",
            placeholder="blank uses confirm or final output root",
            description="Seed root",
            layout=widgets.Layout(width="520px"),
        )
        self.aggregate_input = widgets.Text(
            value="",
            placeholder="blank follows active run root",
            description="Agg input",
            layout=widgets.Layout(width="520px"),
        )
        self.aggregate_output = widgets.Text(
            value="",
            placeholder="blank writes aggregate_summary.json under Agg input",
            description="Agg output",
            layout=widgets.Layout(width="620px"),
        )
        self.aggregate_group_by = widgets.Text(
            value="method search_stage config_hash",
            description="Group by",
            layout=widgets.Layout(width="520px"),
        )
        self.aggregate_metrics = widgets.Text(
            value="eval_f1_macro,training_time_sec,best_epoch",
            description="Metrics",
            layout=widgets.Layout(width="520px"),
        )
        self.view = widgets.VBox(
            [
                self.experiment,
                widgets.HBox([self.use_wandb, self.wandb_mode, self.wandb_log_model]),
                self.overwrite_output_dir,
                widgets.HBox([self.wandb_entity, self.wandb_project]),
                widgets.HBox([self.wandb_group, self.wandb_tags]),
                self.overrides,
                widgets.HBox([self.suggest_trials, self.search_space, self.hpo_seed]),
                self.trial_output_root,
                widgets.HBox([self.seed_run_stage, self.seed_output_root]),
                widgets.HBox([self.aggregate_input, self.aggregate_group_by]),
                widgets.HBox([self.aggregate_output, self.aggregate_metrics]),
            ]
        )

    def get_config(self) -> dict[str, Any]:
        return {
            "experiment": self.experiment.value,
            "overrides": parse_override_text(self.overrides.value),
            "use_wandb": self.use_wandb.value,
            "wandb_entity": self.wandb_entity.value,
            "wandb_project": self.wandb_project.value,
            "wandb_group": self.wandb_group.value or None,
            "wandb_tags": self.wandb_tags.value or None,
            "wandb_mode": self.wandb_mode.value,
            "wandb_log_model": self.wandb_log_model.value,
            "overwrite_output_dir": self.overwrite_output_dir.value,
            "suggest_trials": self.suggest_trials.value,
            "search_space": self.search_space.value or None,
            "hpo_seed": self.hpo_seed.value,
            "trial_output_root": self.trial_output_root.value,
            "seed_run_stage": self.seed_run_stage.value,
            "seed_output_root": self.seed_output_root.value,
        }

    def get_aggregate_config(self) -> dict[str, Any]:
        group_by = [
            item.strip()
            for item in (self.aggregate_group_by.value or "").replace(",", " ").split()
            if item.strip()
        ]
        metrics = [
            item.strip()
            for item in (self.aggregate_metrics.value or "").replace("\n", ",").split(",")
            if item.strip()
        ]
        explicit_input = self.aggregate_input.value
        seed_stage = getattr(getattr(self, "seed_run_stage", None), "value", "none")
        seed_output_root = getattr(getattr(self, "seed_output_root", None), "value", "")
        if explicit_input:
            aggregate_input = explicit_input.rstrip("/")
        elif seed_stage != "none":
            aggregate_input = (
                seed_output_root
                or f"/content/drive/MyDrive/hate_speech_ft/outputs/{seed_stage}"
            ).rstrip("/")
        else:
            aggregate_input = self.trial_output_root.value.rstrip("/")
        return {
            "input": aggregate_input,
            "output": self.aggregate_output.value
            or f"{aggregate_input.rstrip('/')}/aggregate_summary.json",
            "group_by": group_by or ["method", "search_stage", "config_hash"],
            "metrics": metrics or ["eval_f1_macro", "training_time_sec"],
        }

    def build_command(self) -> list[str]:
        config = self.get_config()
        spec = self.registry.get(config["experiment"])
        overrides = dict(config["overrides"])
        if config.get("overwrite_output_dir"):
            overrides["overwrite_output_dir"] = True
        return build_experiment_command(
            spec,
            overrides=overrides,
            use_wandb=config["use_wandb"],
            wandb_entity=config["wandb_entity"],
            wandb_project=config["wandb_project"],
            wandb_group=config["wandb_group"],
            wandb_tags=config["wandb_tags"],
            wandb_mode=config["wandb_mode"],
            wandb_log_model=config["wandb_log_model"],
        )

    def preview_command(self) -> str:
        config = self.get_config()
        if int(config["suggest_trials"] or 0) > 0:
            return "\n".join(self.preview_trial_commands())
        if config.get("seed_run_stage") not in {None, "none"}:
            return "\n".join(self.preview_seed_run_commands())
        command = self.build_command()
        rendered = format_command(command)
        print(rendered)
        return rendered

    def build_trial_commands(self) -> list[list[str]]:
        config = self.get_config()
        n_trials = int(config["suggest_trials"] or 0)
        if n_trials <= 0:
            return []

        spec = self.registry.get(config["experiment"])
        user_overrides = dict(config["overrides"])
        if config.get("overwrite_output_dir"):
            user_overrides["overwrite_output_dir"] = True
        if spec.stage == "smoke":
            raise ValueError(
                "HPO trial generation should use a tuning experiment, not a smoke "
                "experiment with sample caps. Select distilbert_full_tuning for real "
                "trial suggestions."
            )
        search_space_name = config["search_space"] or default_search_space_name(spec.method)
        search_space = get_search_space(self.search_config, search_space_name)
        trial_overrides = build_trial_overrides(
            base_experiment_id=spec.experiment_id,
            method=spec.method,
            search_space=search_space,
            n_trials=n_trials,
            hpo_seed=int(config["hpo_seed"]),
            output_root=config["trial_output_root"],
            trial_cap=get_trial_cap(self.search_config, search_space_name),
            fixed_overrides=shared_fixed_command_overrides(self.search_config),
        )
        commands = []
        for overrides in trial_overrides:
            merged_overrides = merge_trial_overrides(
                base_args=spec.args,
                user_overrides=user_overrides,
                trial_overrides=overrides,
            )
            commands.append(
                build_experiment_command(
                    spec,
                    overrides=merged_overrides,
                    use_wandb=config["use_wandb"],
                    wandb_entity=config["wandb_entity"],
                    wandb_project=config["wandb_project"],
                    wandb_group=config["wandb_group"],
                    wandb_tags=config["wandb_tags"],
                    wandb_mode=config["wandb_mode"],
                    wandb_log_model=config["wandb_log_model"],
                )
            )
        return commands

    def build_seed_run_commands(self) -> list[list[str]]:
        config = self.get_config()
        seed_stage = config.get("seed_run_stage")
        if seed_stage == "none":
            return []

        spec = self.registry.get(config["experiment"])
        validate_seed_run_base_stage(spec.stage)
        user_overrides = dict(config["overrides"])
        if config.get("overwrite_output_dir"):
            user_overrides["overwrite_output_dir"] = True
        seed_overrides = build_seed_run_overrides(
            base_experiment_id=spec.experiment_id,
            method=spec.method,
            seeds=get_seed_policy(self.search_config, seed_stage),
            output_root=config["seed_output_root"]
            or f"/content/drive/MyDrive/hate_speech_ft/outputs/{seed_stage}",
            search_stage=seed_stage,
            fixed_overrides=shared_fixed_command_overrides(self.search_config),
        )
        commands = []
        for overrides in seed_overrides:
            merged_overrides = merge_trial_overrides(
                base_args=spec.args,
                user_overrides=user_overrides,
                trial_overrides=overrides,
                protected_user_override_keys=SEED_RUN_PROTECTED_USER_OVERRIDE_KEYS,
            )
            commands.append(
                build_experiment_command(
                    spec,
                    overrides=merged_overrides,
                    use_wandb=config["use_wandb"],
                    wandb_entity=config["wandb_entity"],
                    wandb_project=config["wandb_project"],
                    wandb_group=config["wandb_group"],
                    wandb_tags=config["wandb_tags"],
                    wandb_mode=config["wandb_mode"],
                    wandb_log_model=config["wandb_log_model"],
                )
            )
        return commands

    def preview_trial_commands(self) -> list[str]:
        rendered_commands = [format_command(command) for command in self.build_trial_commands()]
        for rendered in rendered_commands:
            print(rendered)
        return rendered_commands

    def preview_seed_run_commands(self) -> list[str]:
        rendered_commands = [
            format_command(command) for command in self.build_seed_run_commands()
        ]
        for rendered in rendered_commands:
            print(rendered)
        return rendered_commands

    def run(self):
        config = self.get_config()
        if int(config["suggest_trials"] or 0) > 0:
            return self.run_trial_commands()
        if config.get("seed_run_stage") not in {None, "none"}:
            return self.run_seed_run_commands()
        command = self.build_command()
        print(format_command(command))
        return subprocess.run(command, check=True, cwd=REPO_ROOT)

    def run_trial_commands(self):
        results = []
        for command in self.build_trial_commands():
            print(format_command(command))
            results.append(subprocess.run(command, check=True, cwd=REPO_ROOT))
        return results

    def run_seed_run_commands(self):
        results = []
        for command in self.build_seed_run_commands():
            print(format_command(command))
            results.append(subprocess.run(command, check=True, cwd=REPO_ROOT))
        return results

    def build_aggregate_command(self) -> list[str]:
        config = self.get_aggregate_config()
        command = [
            sys.executable,
            "src/aggregate_results.py",
            config["input"],
            "--output",
            config["output"],
            "--group_by",
            *config["group_by"],
        ]
        for metric in config["metrics"]:
            command.extend(["--metric", metric])
        return command

    def preview_aggregate_command(self) -> str:
        rendered = format_command(self.build_aggregate_command())
        print(rendered)
        return rendered

    def aggregate_results(self) -> dict[str, Any]:
        config = self.get_aggregate_config()
        report = build_aggregate_report(
            [config["input"]],
            group_by=config["group_by"],
            metrics=config["metrics"],
        )
        output_path = write_aggregate_report(config["output"], report)
        print(f"Wrote aggregate report: {output_path}")
        print(
            f"Runs: {report['total_runs']} "
            f"completed={report['completed_runs']} failed={report['failed_runs']}"
        )
        return report
