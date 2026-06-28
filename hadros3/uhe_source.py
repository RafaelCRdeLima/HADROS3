"""HADROS3 H3-W5 UHE Source Monte Carlo layer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .config import validate_values
from .paths import UHE_SOURCE_DIR
from .source_models import sample_polar_cone
from .source_outputs import (
    draw_source_preview,
    source_summary,
    write_source_samples_jsonl,
    write_source_summary_csv,
    write_source_summary_json,
)


UNIFORMITY_BINS = 20


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
        if float(record["x_emit_r"]) != r:
            problems.append(f"sample {index} x_emit_r does not match position.r_rg")
        if float(record["x_emit_theta"]) != float(position["theta_rad"]):
            problems.append(f"sample {index} x_emit_theta does not match position.theta_rad")
        if float(record["x_emit_phi"]) != phi:
            problems.append(f"sample {index} x_emit_phi does not match position.phi_rad")
        physical_pdf = float(record["source_physical_pdf"])
        sampling_pdf = float(record["source_sampling_pdf"])
        weight = float(record["source_weight"])
        direction_sampling_pdf = float(record["direction_sampling_pdf"])
        direction_physical_pdf = float(record["direction_physical_pdf"])
        direction_weight = float(record["direction_weight"])
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
        if record["direction_model"] not in {"coordinate_radial_outward", "isotropic_local"}:
            problems.append(f"sample {index} has unsupported direction_model")
        direction = record["direction_local_components"]
        if direction.get("basis") not in {"Boyer-Lindquist_coordinate_direction", "ZAMO_orthonormal"}:
            problems.append(f"sample {index} has unsupported direction basis")
        if record["direction_model"] == "coordinate_radial_outward":
            if not (
                float(direction.get("dr", 0.0)) > 0.0
                and float(direction.get("dtheta", math.inf)) == 0.0
                and float(direction.get("dphi", math.inf)) == 0.0
            ):
                problems.append(f"sample {index} does not encode coordinate radial outward direction")
        if record["direction_model"] == "isotropic_local":
            norm = math.sqrt(
                float(direction.get("n_r", math.inf)) ** 2
                + float(direction.get("n_theta", math.inf)) ** 2
                + float(direction.get("n_phi", math.inf)) ** 2
            )
            if abs(norm - 1.0) > 1.0e-12:
                problems.append(f"sample {index} isotropic local direction is not unit normalized")
        if not (math.isfinite(direction_physical_pdf) and direction_physical_pdf > 0.0):
            problems.append(f"sample {index} has non-positive direction_physical_pdf")
        if not (math.isfinite(direction_sampling_pdf) and direction_sampling_pdf > 0.0):
            problems.append(f"sample {index} has non-positive direction_sampling_pdf")
        if not math.isfinite(direction_weight):
            problems.append(f"sample {index} has non-finite direction_weight")
        if record["momentum_is_physical_kerr"] is not False:
            problems.append(f"sample {index} incorrectly marks H3-W5 proxy momentum as physical Kerr")
    return problems


def _source_uniform_variables(records: list[dict[str, Any]], values: dict[str, dict[str, Any]]) -> dict[str, list[float]]:
    source = values["uhe_neutrino_source"]
    r_min = float(source["r_min_rg"])
    r_max = float(source["r_max_rg"])
    theta_min = math.radians(float(source["theta_min_deg"]))
    theta_max = math.radians(float(source["theta_max_deg"]))
    r_denominator = r_max**3 - r_min**3
    theta_denominator = math.cos(theta_min) - math.cos(theta_max)
    variables = {"u_r": [], "u_theta": [], "u_phi": []}
    for record in records:
        position = record["position"]
        r = float(position["r_rg"])
        theta = float(position["theta_rad"])
        phi = float(position["phi_rad"]) % (2.0 * math.pi)
        variables["u_r"].append((r**3 - r_min**3) / r_denominator)
        variables["u_theta"].append((math.cos(theta_min) - math.cos(theta)) / theta_denominator)
        variables["u_phi"].append(phi / (2.0 * math.pi))
    return variables


def _ks_uniform_statistic(values: list[float]) -> float:
    if not values:
        return math.inf
    ordered = sorted(values)
    n = len(ordered)
    return max(max((index + 1) / n - value, value - index / n) for index, value in enumerate(ordered))


def _uniform_variable_stats(values: list[float], bins: int = UNIFORMITY_BINS) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        return {"mean": math.inf, "variance": math.inf, "max_bin_deviation": math.inf, "ks_statistic": math.inf}
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / n
    counts = [0 for _ in range(bins)]
    for value in values:
        index = min(bins - 1, max(0, int(value * bins)))
        counts[index] += 1
    expected = n / bins
    max_bin_deviation = max(abs(count - expected) / expected for count in counts) if expected > 0.0 else math.inf
    return {
        "mean": mean,
        "variance": variance,
        "max_bin_deviation": max_bin_deviation,
        "ks_statistic": _ks_uniform_statistic(values),
        "histogram_bins": bins,
        "histogram_counts": counts,
    }


def source_sampling_uniformity_report(records: list[dict[str, Any]], values: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = values["uhe_neutrino_source"]
    variables = _source_uniform_variables(records, values)
    stats = {name: _uniform_variable_stats(samples) for name, samples in variables.items()}
    expected_mean = 0.5
    expected_variance = 1.0 / 12.0
    mean_tolerance = 0.08
    variance_tolerance = 0.04
    max_ks = 0.18
    max_bin_deviation = 0.75
    pass_flag = all(
        0.0 <= value <= 1.0
        for samples in variables.values()
        for value in samples
    ) and all(
        abs(entry["mean"] - expected_mean) <= mean_tolerance
        and abs(entry["variance"] - expected_variance) <= variance_tolerance
        and entry["ks_statistic"] <= max_ks
        and entry["max_bin_deviation"] <= max_bin_deviation
        for entry in stats.values()
    )
    return {
        "n_samples": len(records),
        "r_min_rg": float(source["r_min_rg"]),
        "r_max_rg": float(source["r_max_rg"]),
        "theta_min_deg": float(source["theta_min_deg"]),
        "theta_max_deg": float(source["theta_max_deg"]),
        "u_r_mean": stats["u_r"]["mean"],
        "u_r_variance": stats["u_r"]["variance"],
        "u_r_max_bin_deviation": stats["u_r"]["max_bin_deviation"],
        "u_r_ks_statistic": stats["u_r"]["ks_statistic"],
        "u_theta_mean": stats["u_theta"]["mean"],
        "u_theta_variance": stats["u_theta"]["variance"],
        "u_theta_max_bin_deviation": stats["u_theta"]["max_bin_deviation"],
        "u_theta_ks_statistic": stats["u_theta"]["ks_statistic"],
        "u_phi_mean": stats["u_phi"]["mean"],
        "u_phi_variance": stats["u_phi"]["variance"],
        "u_phi_max_bin_deviation": stats["u_phi"]["max_bin_deviation"],
        "u_phi_ks_statistic": stats["u_phi"]["ks_statistic"],
        "expected_uniform_mean": expected_mean,
        "expected_uniform_variance": expected_variance,
        "mean_tolerance": mean_tolerance,
        "variance_tolerance": variance_tolerance,
        "ks_tolerance": max_ks,
        "max_bin_deviation_tolerance": max_bin_deviation,
        "sampling_uniformity_pass": pass_flag,
        "variables": variables,
        "histograms": {
            name: {
                "bins": stats[name]["histogram_bins"],
                "counts": stats[name]["histogram_counts"],
            }
            for name in stats
        },
    }


def draw_source_sampling_uniformity(report: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    variables = report["variables"]
    n_samples = int(report["n_samples"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True, facecolor="#f8fafc")
    for ax, key, title in zip(
        axes,
        ["u_r", "u_theta", "u_phi"],
        [r"$u_r$ histogram", r"$u_{\theta}$ histogram", r"$u_{\phi}$ histogram"],
    ):
        ax.hist(variables[key], bins=UNIFORMITY_BINS, range=(0.0, 1.0), density=True, color="#2563eb", alpha=0.72, edgecolor="#0f172a", linewidth=0.35)
        ax.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.2, label="uniform expectation")
        ax.set_title(f"{title}\nn={n_samples}")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel({"u_r": r"$u_r$", "u_theta": r"$u_{\theta}$", "u_phi": r"$u_{\phi}$"}[key])
        ax.grid(True, color="#cbd5e1", alpha=0.5, linewidth=0.55)
    axes[0].set_ylabel("normalized density")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _direction_uniform_variables(records: list[dict[str, Any]]) -> dict[str, list[float]]:
    variables = {
        "cos_alpha": [],
        "beta": [],
        "u_alpha": [],
        "u_beta": [],
        "n_x": [],
        "n_y": [],
        "n_z": [],
    }
    for record in records:
        components = record["direction_local_components"]
        if record["direction_model"] == "isotropic_local":
            cos_alpha = float(components["cos_alpha"])
            beta = float(components["beta_rad"]) % (2.0 * math.pi)
            n_x = float(components["n_theta"])
            n_y = float(components["n_phi"])
            n_z = float(components["n_r"])
        else:
            cos_alpha = 1.0
            beta = 0.0
            n_x = 0.0
            n_y = 0.0
            n_z = 1.0
        variables["cos_alpha"].append(cos_alpha)
        variables["beta"].append(beta)
        variables["u_alpha"].append((cos_alpha + 1.0) / 2.0)
        variables["u_beta"].append(beta / (2.0 * math.pi))
        variables["n_x"].append(n_x)
        variables["n_y"].append(n_y)
        variables["n_z"].append(n_z)
    return variables


def source_direction_uniformity_report(records: list[dict[str, Any]], values: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = values["uhe_neutrino_source"]
    variables = _direction_uniform_variables(records)
    u_alpha_stats = _uniform_variable_stats(variables["u_alpha"])
    u_beta_stats = _uniform_variable_stats(variables["u_beta"])
    cos_alpha = variables["cos_alpha"]
    beta = variables["beta"]
    n = len(records)
    expected_mean = 0.5
    expected_variance = 1.0 / 12.0
    mean_tolerance = 0.08
    variance_tolerance = 0.04
    max_ks = 0.18
    max_bin_deviation = 0.75
    pass_flag = (
        records
        and str(source["direction_model"]) == "isotropic_local"
        and all(0.0 <= value <= 1.0 for value in variables["u_alpha"] + variables["u_beta"])
        and abs(u_alpha_stats["mean"] - expected_mean) <= mean_tolerance
        and abs(u_alpha_stats["variance"] - expected_variance) <= variance_tolerance
        and abs(u_beta_stats["mean"] - expected_mean) <= mean_tolerance
        and abs(u_beta_stats["variance"] - expected_variance) <= variance_tolerance
        and u_alpha_stats["ks_statistic"] <= max_ks
        and u_beta_stats["ks_statistic"] <= max_ks
        and u_alpha_stats["max_bin_deviation"] <= max_bin_deviation
        and u_beta_stats["max_bin_deviation"] <= max_bin_deviation
    )
    beta_mean = sum(beta) / n if n else math.inf
    cos_mean = sum(cos_alpha) / n if n else math.inf
    cos_variance = sum((value - cos_mean) ** 2 for value in cos_alpha) / n if n else math.inf
    return {
        "n_samples": n,
        "direction_model": str(source["direction_model"]),
        "direction_seed": int(float(source["direction_seed"])),
        "cos_alpha_mean": cos_mean,
        "cos_alpha_variance": cos_variance,
        "beta_mean": beta_mean,
        "u_alpha_mean": u_alpha_stats["mean"],
        "u_alpha_variance": u_alpha_stats["variance"],
        "u_beta_mean": u_beta_stats["mean"],
        "u_beta_variance": u_beta_stats["variance"],
        "expected_u_mean": expected_mean,
        "expected_u_variance": expected_variance,
        "u_alpha_max_bin_deviation": u_alpha_stats["max_bin_deviation"],
        "u_beta_max_bin_deviation": u_beta_stats["max_bin_deviation"],
        "u_alpha_ks_statistic": u_alpha_stats["ks_statistic"],
        "u_beta_ks_statistic": u_beta_stats["ks_statistic"],
        "mean_tolerance": mean_tolerance,
        "variance_tolerance": variance_tolerance,
        "ks_tolerance": max_ks,
        "max_bin_deviation_tolerance": max_bin_deviation,
        "direction_uniformity_pass": bool(pass_flag),
        "variables": variables,
        "histograms": {
            "u_alpha": {"bins": u_alpha_stats["histogram_bins"], "counts": u_alpha_stats["histogram_counts"]},
            "u_beta": {"bins": u_beta_stats["histogram_bins"], "counts": u_beta_stats["histogram_counts"]},
        },
    }


def draw_source_direction_uniformity(report: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    variables = report["variables"]
    n_samples = int(report["n_samples"])
    panels = [
        ("cos_alpha", r"$\cos(\alpha)$", (-1.0, 1.0), 0.5),
        ("beta", r"$\beta$", (0.0, 2.0 * math.pi), 1.0 / (2.0 * math.pi)),
        ("u_alpha", r"$u_{\alpha}$", (0.0, 1.0), 1.0),
        ("u_beta", r"$u_{\beta}$", (0.0, 1.0), 1.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), facecolor="#f8fafc")
    for ax, (key, label, value_range, expected_density) in zip(axes.flat, panels):
        ax.hist(variables[key], bins=UNIFORMITY_BINS, range=value_range, density=True, color="#7c3aed", alpha=0.70, edgecolor="#0f172a", linewidth=0.35)
        ax.axhline(expected_density, color="#dc2626", linestyle="--", linewidth=1.2, label="uniform expectation")
        ax.set_title(f"{label} histogram\nn={n_samples}")
        ax.set_xlabel(label)
        ax.set_ylabel("normalized density")
        ax.set_xlim(*value_range)
        ax.grid(True, color="#cbd5e1", alpha=0.5, linewidth=0.55)
    axes.flat[-1].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def draw_source_direction_sphere(report: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    output_path.parent.mkdir(parents=True, exist_ok=True)
    variables = report["variables"]
    fig, ax = plt.subplots(figsize=(7.4, 7.0), facecolor="#f8fafc")
    scatter = ax.scatter(
        variables["n_x"],
        variables["n_y"],
        s=7,
        c=variables["n_z"],
        cmap="coolwarm",
        alpha=0.58,
        edgecolors="none",
        vmin=-1.0,
        vmax=1.0,
    )
    ax.add_patch(Circle((0.0, 0.0), 1.0, fill=False, edgecolor="#0f172a", linewidth=1.0, alpha=0.70))
    ax.axhline(0.0, color="#475569", alpha=0.35, linewidth=0.8)
    ax.axvline(0.0, color="#475569", alpha=0.35, linewidth=0.8)
    ax.set_title(f"Local direction sphere projection\nn={report['n_samples']}")
    ax.set_xlabel(r"$n_x = \sin(\alpha)\cos(\beta)$")
    ax.set_ylabel(r"$n_y = \sin(\alpha)\sin(\beta)$")
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)
    ax.set_aspect("equal", adjustable="box")
    colorbar = fig.colorbar(scatter, ax=ax, shrink=0.82)
    colorbar.set_label(r"$n_z = \cos(\alpha)$")
    ax.grid(True, color="#cbd5e1", alpha=0.45, linewidth=0.55)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def generate_uhe_source_products(
    values: dict[str, dict[str, Any]], *, output_dir: Path
) -> dict[str, Any]:
    config_problems = validate_values(values)
    if config_problems:
        raise ValueError("Invalid HADROS3 configuration:\n- " + "\n- ".join(config_problems))
    if output_dir.name != UHE_SOURCE_DIR:
        output_dir = output_dir / UHE_SOURCE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    records = sample_polar_cone(values)
    record_problems = validate_source_records(records, values)
    if record_problems:
        raise ValueError("Invalid UHE source samples:\n- " + "\n- ".join(record_problems[:20]))

    samples_path = output_dir / "uhe_neutrino_source_samples.jsonl"
    csv_path = output_dir / "uhe_neutrino_source_summary.csv"
    json_path = output_dir / "uhe_neutrino_source_summary.json"
    preview_path = output_dir / "uhe_neutrino_source_preview.png"
    uniformity_path = output_dir / "uhe_source_sampling_uniformity.png"
    uniformity_report_path = output_dir / "uhe_source_sampling_uniformity_report.json"
    direction_uniformity_path = output_dir / "uhe_source_direction_uniformity.png"
    direction_uniformity_report_path = output_dir / "uhe_source_direction_uniformity_report.json"
    direction_sphere_path = output_dir / "uhe_source_direction_sphere.png"
    summary = source_summary(records, values)
    uniformity_report = source_sampling_uniformity_report(records, values)
    direction_uniformity_report = source_direction_uniformity_report(records, values)
    summary["validation_errors"] = []
    summary["sampling_uniformity_pass"] = uniformity_report["sampling_uniformity_pass"]
    summary["direction_uniformity_pass"] = direction_uniformity_report["direction_uniformity_pass"]
    summary["products"] = {
        "uhe_source_samples": str(samples_path),
        "uhe_source_summary": str(csv_path),
        "uhe_source_summary_json": str(json_path),
        "uhe_source_preview": str(preview_path),
        "uhe_source_sampling_uniformity": str(uniformity_path),
        "uhe_source_sampling_uniformity_report": str(uniformity_report_path),
        "uhe_source_direction_uniformity": str(direction_uniformity_path),
        "uhe_source_direction_uniformity_report": str(direction_uniformity_report_path),
        "uhe_source_direction_sphere": str(direction_sphere_path),
    }

    write_source_samples_jsonl(records, samples_path)
    write_source_summary_csv(summary, csv_path)
    write_source_summary_json(summary, json_path)
    draw_source_preview(records, values, preview_path)
    draw_source_sampling_uniformity(uniformity_report, uniformity_path)
    write_source_summary_json(uniformity_report, uniformity_report_path)
    draw_source_direction_uniformity(direction_uniformity_report, direction_uniformity_path)
    draw_source_direction_sphere(direction_uniformity_report, direction_sphere_path)
    write_source_summary_json(direction_uniformity_report, direction_uniformity_report_path)
    return summary
