#!/usr/bin/env python3
"""Update HADROS3 scientific release metadata.

This is the official entry point for changing VERSION.json. It also keeps the
Theory LaTeX metadata macros synchronized so the regenerated PDF describes the
same release metadata as provenance.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VERSION_PATH = ROOT / "VERSION.json"
DEFAULT_THEORY_TEX = ROOT / "docs" / "Theory" / "HADROS3_Physics_Theory.tex"
REQUIRED_KEYS = {
    "software_version": "0.9.0",
    "physics_version": "1.0",
    "pipeline_version": "H3-W9a",
    "theory_version": "1.0",
    "theory_document": "docs/Theory/HADROS3_Physics_Theory.pdf",
}


def _short_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _load_version(path: Path) -> dict[str, Any]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {}
    for key, value in REQUIRED_KEYS.items():
        payload.setdefault(key, value)
    return payload


def _write_version(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _major_minor_patch(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) == 2:
        major, minor = parts
        patch = "0"
    elif len(parts) == 3:
        major, minor, patch = parts
    else:
        raise ValueError(f"Expected major.minor or major.minor.patch version, got {version!r}")
    return int(major), int(minor), int(patch)


def _bump_patch(version: str) -> str:
    major, minor, patch = _major_minor_patch(version)
    return f"{major}.{minor}.{patch + 1}"


def _bump_minor(version: str) -> str:
    major, minor, _patch = _major_minor_patch(version)
    return f"{major}.{minor + 1}"


def _replace_macro(text: str, macro: str, value: str) -> str:
    pattern = rf"(\\newcommand{{\\{re.escape(macro)}}}{{)[^}}]*(}})"
    replacement = rf"\g<1>{value}\g<2>"
    updated, count = re.subn(pattern, replacement, text, count=1)
    if count == 0:
        raise ValueError(f"Theory macro \\{macro} not found")
    return updated


def _sync_theory_tex(path: Path, payload: dict[str, Any]) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    macro_values = {
        "SoftwareVersion": payload["software_version"],
        "PhysicsVersion": payload["physics_version"],
        "TheoryVersion": payload["theory_version"],
        "TheoryCompatibleCommit": payload["git_commit"],
        "TheoryGenerationDate": payload["last_release_date"],
        "TheoryPipelineVersion": payload["pipeline_version"],
    }
    for macro, value in macro_values.items():
        text = _replace_macro(text, macro, str(value))
    path.write_text(text, encoding="utf-8")


def update_version(
    *,
    version_path: Path,
    theory_tex: Path,
    root: Path,
    mode: str,
    pipeline: str | None = None,
) -> dict[str, Any]:
    payload = _load_version(version_path)

    if mode == "software":
        payload["software_version"] = _bump_patch(str(payload["software_version"]))
    elif mode == "physics":
        payload["physics_version"] = _bump_minor(str(payload["physics_version"]))
        payload["theory_version"] = _bump_minor(str(payload["theory_version"]))
    elif mode == "pipeline":
        if not pipeline:
            raise ValueError("--pipeline requires a pipeline version, for example H3-W9b")
        payload["pipeline_version"] = pipeline
        payload["current_stage"] = pipeline
    else:
        raise ValueError(f"Unknown release mode {mode!r}")

    payload["last_release_date"] = date.today().isoformat()
    payload["git_commit"] = _short_commit(root)

    _write_version(version_path, payload)
    _sync_theory_tex(theory_tex, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--software", action="store_true", help="increment software_version patch")
    group.add_argument("--physics", action="store_true", help="increment physics_version and theory_version minor")
    group.add_argument("--pipeline", metavar="STAGE", help="set pipeline_version/current_stage")
    parser.add_argument("--version-file", type=Path, default=DEFAULT_VERSION_PATH, help=argparse.SUPPRESS)
    parser.add_argument("--theory-tex", type=Path, default=DEFAULT_THEORY_TEX, help=argparse.SUPPRESS)
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.software:
        mode = "software"
        pipeline = None
    elif args.physics:
        mode = "physics"
        pipeline = None
    else:
        mode = "pipeline"
        pipeline = args.pipeline

    payload = update_version(
        version_path=args.version_file,
        theory_tex=args.theory_tex,
        root=args.root,
        mode=mode,
        pipeline=pipeline,
    )
    print(json.dumps(payload, indent=2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
