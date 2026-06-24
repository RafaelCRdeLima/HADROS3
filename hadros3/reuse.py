"""Discovery of validated HADROS components reused by the HADROS3 web shell."""

from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent


def discover_original_hadros() -> dict[str, Any]:
    candidates = [PARENT / "HADROS", PARENT / "HADROS-CASCADE"]
    roots = [path for path in candidates if path.exists()]
    preferred = roots[0] if roots else None

    def rel(path: Path | None) -> str | None:
        if path is None or not path.exists():
            return None
        try:
            return str(path.resolve().relative_to(ROOT.resolve()))
        except ValueError:
            return str(path.resolve())

    if preferred is None:
        return {
            "status": "not_found",
            "message": "No sibling HADROS/HADROS-CASCADE checkout was found.",
            "roots": [],
            "components": {},
        }

    components = {
        "config_web_final_py": preferred / "scripts" / "config_web_final.py",
        "config_web_py": preferred / "scripts" / "config_web.py",
        "kerr_metric_header": preferred / "include" / "kerr_metric.hpp",
        "kerr_geodesic_header": preferred / "include" / "kerr_geodesic.hpp",
        "kerr_camera_header": preferred / "include" / "kerr_camera.hpp",
        "kerr_metric_source": preferred / "src" / "kerr_metric.cpp",
        "kerr_geodesic_source": preferred / "src" / "kerr_geodesic.cpp",
        "kerr_camera_source": preferred / "src" / "kerr_camera.cpp",
        "geodesic_preview_bin": preferred / "hadros_geodesic_preview",
        "geodesic_preview_cuda_bin": preferred / "hadros_geodesic_preview_cuda",
        "camera_preview_bin": preferred / "hadros_camera_preview",
        "geometry_schematic_script": preferred / "scripts" / "plot_geometry_schematic.py",
        "last_camera_config": preferred / "configs" / "cameras" / "last_camera.json",
    }
    return {
        "status": "found",
        "preferred_root": rel(preferred),
        "roots": [rel(path) for path in roots],
        "reuse_policy": "import/port validated interfaces; do not run expensive event generation",
        "components": {name: rel(path) for name, path in components.items()},
    }
