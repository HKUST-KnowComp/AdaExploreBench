#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from synthesis.datasets.template_pool import load_template_pool  # noqa: E402
from synthesis.templates.reporting import markdown, rationale_rows  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Render canonical template rationales")
    parser.add_argument("--template", type=Path, default=Path("datasets/templates/domain_templates.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = rationale_rows(load_template_pool(args.template))
    if not rows:
        raise ValueError("canonical template contains no dynamic rationales")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown(rows))
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rationale events to {args.output}")


if __name__ == "__main__":
    main()
