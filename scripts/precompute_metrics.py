#!/usr/bin/env python3
"""Precompute full run metrics for subsequent viewer publication."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from viewer.run_summary import precompute_summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    args = parser.parse_args()
    precompute_summaries(args.dataset, args.run_dir)


if __name__ == "__main__":
    main()
