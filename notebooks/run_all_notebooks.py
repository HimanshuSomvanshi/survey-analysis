"""
Execute the project's notebooks in dependency order and stop on the first failure.

Usage:
    python notebooks/run_all_notebooks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


NOTEBOOK_SEQUENCE = (
    "01_eda.ipynb",
    "02_preprocessing.ipynb",
    "03_model_training.ipynb",
)


def find_project_root(notebook_dir: Path) -> Path:
    candidate = notebook_dir.parent
    if (candidate / "data").exists():
        return candidate
    raise FileNotFoundError(
        f"Could not locate project root from notebook directory: {notebook_dir}"
    )


def execute_notebook(
    notebook_path: Path,
    project_root: Path,
    timeout_seconds: int = 1800,
) -> None:
    print(f"\n[RUNNING] {notebook_path.relative_to(project_root)}")

    notebook = nbformat.read(notebook_path, as_version=4)
    executor = ExecutePreprocessor(
        timeout=timeout_seconds,
        kernel_name="python3",
        allow_errors=False,
    )

    executor.preprocess(
        notebook,
        {
            "metadata": {
                "path": str(notebook_path.parent),
            },
        },
    )

    nbformat.write(notebook, notebook_path)
    print(f"[SUCCESS] {notebook_path.relative_to(project_root)}")


def main() -> int:
    notebook_dir = Path(__file__).resolve().parent
    project_root = find_project_root(notebook_dir)

    print(f"Project root: {project_root}")
    print(f"Python executable: {sys.executable}")

    for notebook_name in NOTEBOOK_SEQUENCE:
        notebook_path = notebook_dir / notebook_name

        if not notebook_path.is_file():
            print(
                f"[ERROR] Missing required notebook: "
                f"{notebook_path.relative_to(project_root)}",
                file=sys.stderr,
            )
            return 1

        try:
            execute_notebook(notebook_path, project_root)
        except Exception as exc:
            print(
                f"[FAILED] {notebook_path.relative_to(project_root)}",
                file=sys.stderr,
            )
            print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    print("\nAll notebooks completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
