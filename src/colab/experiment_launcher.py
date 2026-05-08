from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

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
        default_entity: str = "",
        default_project: str = "hate-speech-ft",
        include_planned: bool = False,
    ) -> None:
        import ipywidgets as widgets

        self._widgets = widgets
        self.config_path = Path(config_path)
        self.registry = load_experiment_registry(self.config_path)
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
        self.overrides = widgets.Textarea(
            value="",
            placeholder="Optional one per line, e.g.\nlearning_rate=3e-5\nseed=43\noutput_dir=/content/drive/MyDrive/hate_speech_ft/outputs/my_run",
            description="Overrides",
            layout=widgets.Layout(width="760px", height="120px"),
        )
        self.view = widgets.VBox(
            [
                self.experiment,
                widgets.HBox([self.use_wandb, self.wandb_mode, self.wandb_log_model]),
                widgets.HBox([self.wandb_entity, self.wandb_project]),
                widgets.HBox([self.wandb_group, self.wandb_tags]),
                self.overrides,
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
        }

    def build_command(self) -> list[str]:
        config = self.get_config()
        spec = self.registry.get(config["experiment"])
        return build_experiment_command(
            spec,
            overrides=config["overrides"],
            use_wandb=config["use_wandb"],
            wandb_entity=config["wandb_entity"],
            wandb_project=config["wandb_project"],
            wandb_group=config["wandb_group"],
            wandb_tags=config["wandb_tags"],
            wandb_mode=config["wandb_mode"],
            wandb_log_model=config["wandb_log_model"],
        )

    def preview_command(self) -> str:
        command = self.build_command()
        rendered = format_command(command)
        print(rendered)
        return rendered

    def run(self):
        command = self.build_command()
        print(format_command(command))
        return subprocess.run(command, check=True, cwd=REPO_ROOT)
