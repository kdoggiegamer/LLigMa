#!/usr/bin/env python3
"""Publish the Kaggle code dataset straight from the repository.

The kernels import their code from a Kaggle dataset. That dataset used to be a
hand-maintained staging directory (kaggle_upload_code/) refreshed by remembering
to copy lligma/, scripts/ and configs/ into it -- and it drifted, so the code
that actually ran was not provably the code in this repository.

This stages a fresh copy in a throwaway directory outside the project, prints
the hash of every file it is about to upload, refuses to continue if a staged
file differs from its source, and removes the staging directory on exit. There
is no copy to keep in sync because there is no copy.

    python scripts/publish_code_dataset.py --dry-run
    python scripts/publish_code_dataset.py -m "what changed"
    python scripts/publish_code_dataset.py --create -m "first version"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "angelodellamorte/lligma-code"
DIRECTORIES = ("lligma", "scripts", "configs")
FILES = ("requirements.txt", "README.md")
METADATA = "dataset-metadata.json"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stage(target: Path, message: str) -> None:
    for name in DIRECTORIES:
        shutil.copytree(ROOT / name, target / name, ignore=IGNORE)
    for name in FILES:
        shutil.copy2(ROOT / name, target / name)
    (target / METADATA).write_text(json.dumps({
        "title": DATASET_ID.split("/")[1],
        "id": DATASET_ID,
        "licenses": [{"name": "other"}],
        "description": message,
    }, indent=2), encoding="utf-8")


def mismatches(staged: Path) -> list[str]:
    """Every staged file must be byte-identical to the file it came from."""
    wrong = []
    for path in sorted(staged.rglob("*")):
        if not path.is_file() or path.name == METADATA:
            continue
        original = ROOT / path.relative_to(staged)
        if not original.is_file() or digest(original) != digest(path):
            wrong.append(str(path.relative_to(staged)))
    return wrong


def kaggle_executable() -> str:
    found = shutil.which("kaggle")
    if found:
        return found
    sibling = Path(sys.executable).with_name("kaggle.exe" if os.name == "nt" else "kaggle")
    if sibling.is_file():
        return str(sibling)
    raise SystemExit("kaggle CLI not found: install it or run from the project virtualenv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-m", "--message", default="sync code from repository")
    parser.add_argument("--create", action="store_true", help="create the dataset instead of a new version")
    parser.add_argument("--dry-run", action="store_true", help="stage and verify, upload nothing")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="lligma-code-") as temporary:
        target = Path(temporary)
        stage(target, args.message)

        files = sorted(path for path in target.rglob("*") if path.is_file())
        for path in files:
            print(f"{digest(path)[:16]}  {path.relative_to(target)}")

        wrong = mismatches(target)
        if wrong:
            raise SystemExit(f"staged files differ from the repository: {wrong}")
        print(f"verified: {len(files)} staged files are byte-identical to the repository")

        if args.dry_run:
            print("dry run: nothing uploaded")
            return 0

        command = [kaggle_executable(), "datasets", "create" if args.create else "version",
                   "-p", str(target), "--dir-mode", "zip"]
        if not args.create:
            command += ["-m", args.message]
        print("RUN " + " ".join(command))
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
