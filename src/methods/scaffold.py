from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = Path(__file__).resolve().parent / "_template"
VALID_FAMILIES = {
    "classical",
    "neural-scratch",
    "transformer",
    "transformer-peft",
    "transformer-two-stage",
}
PACKAGE_RE = re.compile(r"^[a-z][a-z0-9_]*$")
METHOD_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _validate_name(value: str, pattern: re.Pattern[str], label: str) -> None:
    if not pattern.fullmatch(value):
        raise ValueError(
            f"{label} must match {pattern.pattern!r}. Got: {value!r}"
        )


def build_catalog_snippet(
    *,
    method_package: str,
    method_id: str,
    family: str,
    description: str,
    stage: str = "template",
) -> dict[str, Any]:
    _validate_name(method_package, PACKAGE_RE, "method_package")
    _validate_name(method_id, METHOD_ID_RE, "method_id")
    if family not in VALID_FAMILIES:
        allowed = ", ".join(sorted(VALID_FAMILIES))
        raise ValueError(f"family must be one of: {allowed}")

    return {
        "status": "planned",
        "method": method_id,
        "family": family,
        "stage": stage,
        "script": f"src/methods/{method_package}/train.py",
        "description": description,
        "tags": [method_id, stage],
        "args": {
            "seed": 42,
            "output_dir": f"outputs/{method_package}_{stage}",
        },
    }


def _render_template(template_text: str, replacements: dict[str, str]) -> str:
    rendered = template_text
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def scaffold_method(
    *,
    repo_root: str | Path = REPO_ROOT,
    method_package: str,
    method_id: str,
    family: str,
    description: str,
    overwrite: bool = False,
) -> list[Path]:
    _validate_name(method_package, PACKAGE_RE, "method_package")
    _validate_name(method_id, METHOD_ID_RE, "method_id")
    if family not in VALID_FAMILIES:
        allowed = ", ".join(sorted(VALID_FAMILIES))
        raise ValueError(f"family must be one of: {allowed}")

    root = Path(repo_root)
    target_dir = root / "src" / "methods" / method_package
    if target_dir.exists() and not overwrite:
        raise FileExistsError(
            f"{target_dir} already exists. Pass --overwrite only if replacing a scaffold."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    train_template = (TEMPLATE_DIR / "train.py").read_text(encoding="utf-8")
    readme_template = (
        "# {{METHOD_PACKAGE}}\n\n"
        "{{DESCRIPTION}}\n\n"
        "Implement the method in `train.py`, then register a `planned` or "
        "`ready` experiment in `configs/experiments.json`.\n"
    )
    replacements = {
        "METHOD_ID": method_id,
        "METHOD_PACKAGE": method_package,
        "DESCRIPTION": description,
    }

    files = {
        target_dir / "__init__.py": f'"""Method package for {method_id}."""\n',
        target_dir / "train.py": _render_template(train_template, replacements),
        target_dir / "README.md": _render_template(readme_template, replacements),
    }
    for path, content in files.items():
        path.write_text(content, encoding="utf-8")
    return list(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a new method package scaffold.")
    parser.add_argument("--method-package", required=True, help="Python package name, e.g. distilbert_lora")
    parser.add_argument("--method-id", required=True, help="Experiment method id, e.g. lora")
    parser.add_argument("--family", required=True, choices=sorted(VALID_FAMILIES))
    parser.add_argument("--description", required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    created = scaffold_method(
        repo_root=args.repo_root,
        method_package=args.method_package,
        method_id=args.method_id,
        family=args.family,
        description=args.description,
        overwrite=args.overwrite,
    )
    snippet = build_catalog_snippet(
        method_package=args.method_package,
        method_id=args.method_id,
        family=args.family,
        description=args.description,
    )
    experiment_id = f"{args.method_package}_template"
    print("Created:")
    for path in created:
        print(f"- {path}")
    print("\nAdd this under configs/experiments.json -> experiments:")
    print(json.dumps({experiment_id: snippet}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
