from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from pathlib import Path

from synthesis.datasets.template_pool import load_template_pool

from .config import load_config, validate_resources
from .contracts import Contracts
from .pipeline import TemplateSynthesisPipeline


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        prog="synthesize.py templates",
        description="Synthesize canonical AdaExploreBench domain templates",
    )
    value.add_argument("--config", type=Path, default=Path("configs/synthesis/templates.yaml"))
    value.add_argument("--run-id")
    value.add_argument("--domains", nargs="+")
    value.add_argument("--resume", action="store_true")
    value.add_argument("--fresh", action="store_true", help="start from the fixed skeleton instead of appending")
    value.add_argument("--publish", action="store_true", help="atomically publish only after all audits pass")
    value.add_argument(
        "--stop-after", choices=("field_constraints",),
        help="accept the selected stage for every requested domain, then exit without assembling later stages",
    )
    value.add_argument("--check", action="store_true", help="validate config, schemas, prompts, and current canonical without model calls")
    return value


def check(config_path: Path) -> None:
    config = load_config(config_path)
    validate_resources(config)
    contracts = Contracts(config.paths.canonical_schema, config.paths.stage_schemas)
    contracts.check_all()
    if config.paths.canonical.exists():
        pool = load_template_pool(config.paths.canonical)
        contracts.validate("canonical", pool)
    print("template synthesis resources and canonical template are valid")


async def run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.publish and args.stop_after:
        raise ValueError("--publish cannot be combined with --stop-after")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pipeline = TemplateSynthesisPipeline(config, run_id=run_id, resume=args.resume)
    await pipeline.run(
        domains=tuple(args.domains) if args.domains else None,
        append=not args.fresh,
        publish=args.publish,
        stop_after=args.stop_after,
    )
    if args.stop_after:
        print(f"stage accepted for all selected domains: {args.stop_after}")
        return
    print(f"candidate accepted: {pipeline.state.candidate}")
    if args.publish:
        print(f"canonical published: {config.paths.canonical}")


def main() -> None:
    args = parser().parse_args()
    if args.check:
        check(args.config)
        return
    asyncio.run(run(args))
