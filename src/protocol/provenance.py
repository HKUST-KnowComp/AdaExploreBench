from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any


DATASET_MANIFEST_SCHEMA_VERSION = "adaexplorebench_dataset_manifest_v1"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def code_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "working_tree_clean": not bool(status.stdout.strip()) if status.returncode == 0 else None,
        "working_tree_status_sha256": sha256_bytes(status.stdout.encode("utf-8")),
    }
