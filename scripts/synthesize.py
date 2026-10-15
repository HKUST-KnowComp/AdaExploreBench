#!/usr/bin/env python3
"""Unified entry point for template synthesis and dataset generation."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for entry in (str(REPOSITORY_ROOT), str(SOURCE_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from synthesis.datasets.cli import main as datasets_main  # noqa: E402
from synthesis.datasets.final_review_cli import main as final_review_main  # noqa: E402
from synthesis.datasets.selection_cli import main as selection_main  # noqa: E402
from synthesis.templates.cli import main as templates_main  # noqa: E402


def main() -> None:
    commands = ("templates", "dataset", "dataset-review", "dataset-select")
    usage = f"usage: synthesize.py {{{','.join(commands)}}} ..."
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(usage)
        print("\ncommands:")
        print("  templates  Synthesize canonical domain templates")
        print("  dataset    Generate a static or dynamic episode dataset")
        print("  dataset-review  Score a published dynamic episode dataset")
        print("  dataset-select  Select balanced top-scoring dynamic episodes")
        raise SystemExit(0 if len(sys.argv) >= 2 else 2)
    if sys.argv[1] not in commands:
        print(usage, file=sys.stderr)
        raise SystemExit(2)
    command = sys.argv.pop(1)
    if command == "templates":
        templates_main()
    elif command == "dataset":
        datasets_main()
    elif command == "dataset-review":
        final_review_main()
    else:
        selection_main()


if __name__ == "__main__":
    main()
