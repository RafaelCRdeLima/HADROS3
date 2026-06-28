from __future__ import annotations

import json
from pathlib import Path

from hadros3.config import defaults
from hadros3.observer_image_branches import generate_observer_image_branch_products


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _candidate(candidate_id: str, rank: int, score: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "interaction_id": candidate_id,
        "event_id": f"event-{candidate_id}",
        "source_sample_id": f"source-{candidate_id}",
        "interaction_E_nu_local_gev": 1.0e9 + rank,
        "physics_weight": 0.5,
        "observer_weight": score,
        "final_observation_score": score,
        "selection_policy": "top_n",
        "selection_rank": rank,
        "selection_reason": "test",
    }


def _cluster(index: int, x: float, y: float, n_rays: int, approach: float, spread: float = 1.0) -> dict[str, object]:
    pixels = [[x + spread * i, y] for i in range(n_rays)]
    return {
        "cluster_index": index,
        "n_rays": n_rays,
        "centroid_pixel": [x, y],
        "closest_approach_rg": approach,
        "pixels": pixels,
        "ray_indices": list(range(index * 100, index * 100 + n_rays)),
    }


def test_observer_image_branch_analysis_selects_primary_by_branch_score(tmp_path: Path) -> None:
    bridge_dir = tmp_path / "ObserverBridge"
    rows = [_candidate("single", 1, 0.8), _candidate("double", 2, 0.7)]
    _write_jsonl(bridge_dir / "observer_bridge_selected_candidates.jsonl", rows)
    _write_jsonl(bridge_dir / "observer_bridge_ranked_events.jsonl", rows)
    _write_jsonl(
        bridge_dir / "observer_candidate_kerr_pixel_map.jsonl",
        [
            {"candidate_id": "single", "matched_pixel_x": 10.0, "matched_pixel_y": 20.0, "closest_approach_rg": 0.5},
            {"candidate_id": "double", "matched_pixel_x": 30.0, "matched_pixel_y": 40.0, "closest_approach_rg": 0.5},
        ],
    )
    _write_jsonl(
        bridge_dir / "candidate_multi_image_audit.jsonl",
        [
            {
                "candidate_id": "single",
                "image_clusters": [_cluster(1, 10.0, 20.0, 4, 0.5)],
                "all_matching_rays": [{"ray_index": 100 + i, "closest_approach_rg": 0.5} for i in range(4)],
            },
            {
                "candidate_id": "double",
                "image_clusters": [
                    _cluster(1, 30.0, 40.0, 2, 0.2, spread=5.0),
                    _cluster(2, 90.0, 80.0, 8, 0.4, spread=1.0),
                ],
                "all_matching_rays": (
                    [{"ray_index": 100 + i, "closest_approach_rg": 0.2} for i in range(2)]
                    + [{"ray_index": 200 + i, "closest_approach_rg": 0.4} for i in range(8)]
                ),
            },
        ],
    )

    summary = generate_observer_image_branch_products(defaults(), run_output_dir=tmp_path)
    branch_rows = [json.loads(line) for line in (tmp_path / "ObserverImageBranches" / "observer_image_branches.jsonl").read_text(encoding="utf-8").splitlines()]
    primary_rows = [json.loads(line) for line in (tmp_path / "ObserverImageBranches" / "observer_image_primary_branches.jsonl").read_text(encoding="utf-8").splitlines()]

    assert summary["observer_image_branch_analysis_invoked"] is True
    assert summary["n_candidates"] == 2
    assert summary["n_branches"] == 3
    assert summary["n_single_image"] == 1
    assert summary["n_double_image"] == 1
    assert summary["fraction_multiple_images"] == 0.5
    assert {row["candidate_id"] for row in branch_rows} == {"single", "double"}
    double_primary = next(row for row in primary_rows if row["candidate_id"] == "double")
    assert double_primary["number_of_image_branches"] == 2
    assert double_primary["primary_branch_id"] == "double:branch-02"
    assert double_primary["primary_branch_selection_model"] == "argmax_branch_score"
    assert double_primary["primary_branch_selection_proxy"] is True

    for filename in [
        "observer_image_branch_summary.json",
        "observer_image_branch_report.json",
        "observer_image_statistics.json",
        "observer_branch_score_distribution.png",
        "observer_branch_cluster_map.png",
        "observer_branch_primary_vs_secondary.png",
        "observer_branch_statistics.csv",
        "observer_branch_view.html",
        "observer_viewpoint_convention_audit.json",
        "observer_viewpoint_convention_diagnostic.png",
    ]:
        assert (tmp_path / "ObserverImageBranches" / filename).exists()

    audit = json.loads((tmp_path / "ObserverImageBranches" / "observer_viewpoint_convention_audit.json").read_text())
    assert audit["inclination_convention"] == "theta_0_north_pi_over_2_equator"
    assert audit["expected_observer_hemisphere"] == "north"
    assert audit["camera_preview_observer_z"] > 0
    assert audit["observer_overlay_observer_z"] > 0
    assert audit["observer_interactive_view_camera_z"] > 0
    assert audit["observer_branch_view_camera_z"] > 0
    assert audit["medium_renderer_z_convention"] == "z = r cos(theta)"
    assert audit["all_views_hemisphere_consistent"] is True
