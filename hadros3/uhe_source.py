"""HADROS3 H3-W5 UHE Source Monte Carlo layer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .config import validate_values
from .source_models import sample_polar_cone
from .source_outputs import (
    draw_source_preview,
    source_summary,
    write_source_samples_jsonl,
    write_source_summary_csv,
    write_source_summary_json,
)


def validate_source_records(records: list[dict[str, Any]], values: dict[str, dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    source = values["uhe_neutrino_source"]
    r_min = float(source["r_min_rg"])
    r_max = float(source["r_max_rg"])
    theta_min = float(source["theta_min_deg"])
    theta_max = float(source["theta_max_deg"])
    if len(records) != int(float(source["n_samples"])):
        problems.append("source record count does not match uhe_neutrino_source.n_samples")
    for index, record in enumerate(records):
        position = record["position"]
        r = float(position["r_rg"])
        theta = float(position["theta_deg"])
        phi = float(position["phi_rad"])
        physical_pdf = float(record["source_physical_pdf"])
        sampling_pdf = float(record["source_sampling_pdf"])
        weight = float(record["source_weight"])
        if not (r_min <= r <= r_max):
            problems.append(f"sample {index} has r outside configured source volume")
        if not (theta_min <= theta <= theta_max):
            problems.append(f"sample {index} has theta outside configured source volume")
        if not (0.0 <= phi < 2.0 * math.pi):
            problems.append(f"sample {index} has phi outside [0, 2pi)")
        if not (math.isfinite(physical_pdf) and physical_pdf > 0.0):
            problems.append(f"sample {index} has non-positive source_physical_pdf")
        if not (math.isfinite(sampling_pdf) and sampling_pdf > 0.0):
            problems.append(f"sample {index} has non-positive source_sampling_pdf")
        if not math.isfinite(weight):
            problems.append(f"sample {index} has non-finite source_weight")
        if record["momentum_is_physical_kerr"] is not False:
            problems.append(f"sample {index} incorrectly marks H3-W5 proxy momentum as physical Kerr")
    return problems


def generate_uhe_source_products(
    values: dict[str, dict[str, Any]], *, output_dir: Path
) -> dict[str, Any]:
    config_problems = validate_values(values)
    if config_problems:
        raise ValueError("Invalid HADROS3 configuration:\n- " + "\n- ".join(config_problems))
    output_dir.mkdir(parents=True, exist_ok=True)
    records = sample_polar_cone(values)
    record_problems = validate_source_records(records, values)
    if record_problems:
        raise ValueError("Invalid UHE source samples:\n- " + "\n- ".join(record_problems[:20]))

    samples_path = output_dir / "uhe_neutrino_source_samples.jsonl"
    csv_path = output_dir / "uhe_neutrino_source_summary.csv"
    json_path = output_dir / "uhe_neutrino_source_summary.json"
    preview_path = output_dir / "uhe_neutrino_source_preview.png"
    summary = source_summary(records, values)
    summary["validation_errors"] = []
    summary["products"] = {
        "uhe_source_samples": str(samples_path),
        "uhe_source_summary": str(csv_path),
        "uhe_source_summary_json": str(json_path),
        "uhe_source_preview": str(preview_path),
    }

    write_source_samples_jsonl(records, samples_path)
    write_source_summary_csv(summary, csv_path)
    write_source_summary_json(summary, json_path)
    draw_source_preview(records, values, preview_path)
    return summary
