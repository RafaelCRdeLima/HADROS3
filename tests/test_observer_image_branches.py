from __future__ import annotations

import json
import csv
from pathlib import Path

from hadros3.config import defaults
from hadros3.observer_image_branches import generate_observer_image_branch_products
from hadros3.provenance import build_provenance


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
    assert summary["gravitational_image_analysis_invoked"] is True
    assert summary["visual_stage_name"] == "Gravitational Image Analysis"
    assert summary["branch_score_is_proxy"] is True
    assert summary["branch_score_not_true_magnification"] is True
    assert summary["branch_selection_model"] == "argmax_branch_score"
    assert summary["powheg_forwarding_uses_primary_branch"] is True
    provenance = build_provenance(
        root=Path.cwd(),
        values=defaults(),
        products={},
        validation={"expensive_event_generation_invoked": False},
        observer_image_branch_summary=summary,
    )
    assert provenance["observer_image_branches"]["branch_score_is_proxy"] is True
    assert provenance["observer_image_branches"]["branch_score_not_true_magnification"] is True
    assert provenance["observer_image_branches"]["visual_stage_name"] == "Gravitational Image Analysis"
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
        "observer_image_branches_orientation_audit.json",
        "gravitational_image_multiplicity.png",
        "gravitational_image_branch_score_breakdown.png",
        "gravitational_image_candidate_story.png",
        "gravitational_image_primary_selection_table.csv",
        "gravitational_image_powheg_forwarding_table.csv",
        "gravitational_image_candidate_view.html",
    ]:
        assert (tmp_path / "ObserverImageBranches" / filename).exists()

    primary_table = list(csv.DictReader((tmp_path / "ObserverImageBranches" / "gravitational_image_primary_selection_table.csv").open(encoding="utf-8")))
    double_primary_table = next(row for row in primary_table if row["candidate_id"] == "double")
    assert double_primary_table["primary_branch_id"] == "double:branch-02"
    assert double_primary_table["why_selected"] == "highest branch_score by argmax(branch_score)"

    powheg_table = list(csv.DictReader((tmp_path / "ObserverImageBranches" / "gravitational_image_powheg_forwarding_table.csv").open(encoding="utf-8")))
    double_powheg_row = next(row for row in powheg_table if row["powheg_forwarded_candidate_id"] == "double")
    assert double_powheg_row["powheg_forwarded_branch_id"] == "double:branch-02"
    assert double_powheg_row["powheg_forwarded_role"] == "primary"
    assert double_powheg_row["powheg_forwarding_reason"] == "selected primary branch by argmax(branch_score)"

    candidate_view_html = (tmp_path / "ObserverImageBranches" / "gravitational_image_candidate_view.html").read_text(encoding="utf-8")
    assert "Gravitational Image Analysis" in candidate_view_html
    assert "This primary branch is the one forwarded to POWHEG" in candidate_view_html

    branch_view_html = (tmp_path / "ObserverImageBranches" / "observer_branch_view.html").read_text(encoding="utf-8")
    assert ".stage img { display:block; max-width:100%; transform:scaleY(-1); }" in branch_view_html
    assert "imageHeight - Number(branch.pixel_centroid_y)" in branch_view_html
    assert ".north { left:12px; top:12px; color:#38bdf8; }" in branch_view_html
    assert ".south { left:12px; bottom:12px; color:#f97316; }" in branch_view_html

    orientation_audit = json.loads((tmp_path / "ObserverImageBranches" / "observer_image_branches_orientation_audit.json").read_text())
    spatial_products = {
        row["product_name"]: row
        for row in orientation_audit["products"]
        if row["product_name"]
        in {
            "observer_branch_view.html",
            "observer_branch_cluster_map.png",
            "observer_viewpoint_convention_diagnostic.png",
        }
    }
    assert set(spatial_products) == {
        "observer_branch_view.html",
        "observer_branch_cluster_map.png",
        "observer_viewpoint_convention_diagnostic.png",
    }
    for row in spatial_products.values():
        assert row["north_marker_screen_position"] == "top"
        assert row["south_marker_screen_position"] == "bottom"
        assert row["visual_convention"] == "north_up"
        assert row["visual_image_transform"] == "flip_y"
        assert row["display_coordinate_transform"] == "display_y = image_height - source_y"
        assert row["matches_expected"] is True

    audit = json.loads((tmp_path / "ObserverImageBranches" / "observer_viewpoint_convention_audit.json").read_text())
    assert audit["inclination_convention"] == "theta_0_north_pi_over_2_equator"
    assert audit["expected_observer_hemisphere"] == "north"
    assert audit["camera_preview_observer_z"] > 0
    assert audit["observer_overlay_observer_z"] > 0
    assert audit["observer_interactive_view_camera_z"] > 0
    assert audit["observer_branch_view_camera_z"] > 0
    assert audit["medium_renderer_z_convention"] == "z = r cos(theta)"
    assert audit["all_views_hemisphere_consistent"] is True
