from __future__ import annotations

import argparse
import signal
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .config import load_config
from .pipeline import run_generation


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        prog="synthesize.py dataset",
        description="Generate a configured AdaExploreBench static or dynamic dataset",
    )
    value.add_argument(
        "--config",
        type=Path,
        default=Path("configs/synthesis/dynamic.yaml"),
    )
    value.add_argument(
        "--run-id",
        help="Stable generation run ID; required to resume an interrupted run.",
    )
    value.add_argument("--resume", action="store_true")
    value.add_argument(
        "--operation",
        choices=("fresh", "append"),
        help="Override dataset.operation for this official generation run.",
    )
    value.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow a fresh run to replace an existing dataset output.",
    )
    return value


def main() -> None:
    def interrupt(signum: int, _frame: object) -> None:
        raise InterruptedError(f"received signal {signal.Signals(signum).name}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, interrupt)
    value = parser()
    args = value.parse_args()
    config = load_config(args.config)
    if args.operation is not None:
        config = replace(
            config,
            dataset=replace(config.dataset, operation=args.operation),
        )
    if args.overwrite:
        if config.dataset.operation != "fresh":
            value.error("--overwrite is only valid for a fresh operation")
        config = replace(
            config,
            dataset=replace(config.dataset, overwrite=True),
        )
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if args.resume and not args.run_id:
        value.error("--resume requires --run-id")
    dataset = run_generation(
        config,
        run_id=run_id,
        resume=args.resume,
    )
    print(
        f"Wrote {dataset.episode_count} new {dataset.mode} episodes to "
        f"{config.dataset.output_dir}"
    )
