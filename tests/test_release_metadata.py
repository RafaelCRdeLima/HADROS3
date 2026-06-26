from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from hadros3.config import defaults
from hadros3.provenance import build_provenance
from scripts.release.update_version import update_version


REQUIRED_VERSION_KEYS = {
    "software_version",
    "physics_version",
    "pipeline_version",
    "theory_version",
    "theory_document",
}


def _write_version(path: Path) -> dict[str, str]:
    payload = {
        "software_version": "0.9.0",
        "physics_version": "1.0",
        "pipeline_version": "H3-W9a",
        "theory_version": "1.0",
        "theory_document": "docs/Theory/HADROS3_Physics_Theory.pdf",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _write_theory_tex(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                r"\newcommand{\SoftwareVersion}{0.9.0}",
                r"\newcommand{\PhysicsVersion}{1.0}",
                r"\newcommand{\TheoryVersion}{1.0}",
                r"\newcommand{\TheoryCompatibleCommit}{old}",
                r"\newcommand{\TheoryGenerationDate}{2026-01-01}",
                r"\newcommand{\TheoryPipelineVersion}{H3-W9a}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_version_json_exists_and_has_required_keys() -> None:
    payload = json.loads(Path("VERSION.json").read_text(encoding="utf-8"))
    assert REQUIRED_VERSION_KEYS <= payload.keys()


def test_update_version_software_increments_only_software_version(tmp_path: Path) -> None:
    version_path = tmp_path / "VERSION.json"
    theory_tex = tmp_path / "Theory.tex"
    original = _write_version(version_path)
    _write_theory_tex(theory_tex)

    updated = update_version(
        version_path=version_path,
        theory_tex=theory_tex,
        root=Path.cwd(),
        mode="software",
    )

    assert updated["software_version"] == "0.9.1"
    assert updated["physics_version"] == original["physics_version"]
    assert updated["theory_version"] == original["theory_version"]
    assert updated["pipeline_version"] == original["pipeline_version"]
    assert updated["last_release_date"]
    assert updated["git_commit"]
    assert r"\newcommand{\SoftwareVersion}{0.9.1}" in theory_tex.read_text(encoding="utf-8")


def test_update_version_physics_increments_physics_and_theory_versions(tmp_path: Path) -> None:
    version_path = tmp_path / "VERSION.json"
    theory_tex = tmp_path / "Theory.tex"
    original = _write_version(version_path)
    _write_theory_tex(theory_tex)

    updated = update_version(
        version_path=version_path,
        theory_tex=theory_tex,
        root=Path.cwd(),
        mode="physics",
    )

    assert updated["software_version"] == original["software_version"]
    assert updated["physics_version"] == "1.1"
    assert updated["theory_version"] == "1.1"
    assert updated["pipeline_version"] == original["pipeline_version"]
    tex = theory_tex.read_text(encoding="utf-8")
    assert r"\newcommand{\PhysicsVersion}{1.1}" in tex
    assert r"\newcommand{\TheoryVersion}{1.1}" in tex


def test_update_version_pipeline_updates_pipeline_and_current_stage(tmp_path: Path) -> None:
    version_path = tmp_path / "VERSION.json"
    theory_tex = tmp_path / "Theory.tex"
    original = _write_version(version_path)
    _write_theory_tex(theory_tex)

    updated = update_version(
        version_path=version_path,
        theory_tex=theory_tex,
        root=Path.cwd(),
        mode="pipeline",
        pipeline="H3-W9b",
    )

    assert updated["software_version"] == original["software_version"]
    assert updated["physics_version"] == original["physics_version"]
    assert updated["theory_version"] == original["theory_version"]
    assert updated["pipeline_version"] == "H3-W9b"
    assert updated["current_stage"] == "H3-W9b"
    assert r"\newcommand{\TheoryPipelineVersion}{H3-W9b}" in theory_tex.read_text(encoding="utf-8")


def test_update_version_cli_uses_temporary_version_file(tmp_path: Path) -> None:
    version_path = tmp_path / "VERSION.json"
    theory_tex = tmp_path / "Theory.tex"
    _write_version(version_path)
    _write_theory_tex(theory_tex)

    subprocess.run(
        [
            sys.executable,
            "scripts/release/update_version.py",
            "--software",
            "--version-file",
            str(version_path),
            "--theory-tex",
            str(theory_tex),
            "--root",
            str(Path.cwd()),
        ],
        check=True,
    )

    payload = json.loads(version_path.read_text(encoding="utf-8"))
    assert payload["software_version"] == "0.9.1"
    assert json.loads(Path("VERSION.json").read_text(encoding="utf-8"))["software_version"] == "0.9.0"


def test_provenance_reads_scientific_release_from_version_json(tmp_path: Path) -> None:
    version_path = tmp_path / "VERSION.json"
    payload = _write_version(version_path)
    payload["software_version"] = "9.9.9"
    payload["physics_version"] = "8.7"
    payload["pipeline_version"] = "H3-WX"
    payload["theory_version"] = "8.7"
    payload["last_release_date"] = "2026-06-26"
    version_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    provenance = build_provenance(
        root=tmp_path,
        values=defaults(),
        products={},
        validation={"expensive_event_generation_invoked": False},
    )

    release = provenance["scientific_release"]
    for key in REQUIRED_VERSION_KEYS | {"last_release_date"}:
        assert release[key] == payload[key]
    assert "git_commit" in release
    assert provenance["scientific_theory"]["physics_version"] == payload["physics_version"]
    assert provenance["scientific_theory"]["software_version"] == payload["software_version"]
