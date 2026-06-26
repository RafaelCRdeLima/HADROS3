"""Shared visual renderer for the HADROS3 analytic medium."""

from __future__ import annotations

import math
from typing import Any, Callable

from .medium_model import analytic_torus_density_g_cm3, medium_metadata

Vec3 = tuple[float, float, float]
Projector2D = Callable[[Vec3], tuple[float, float]]
CameraProjector = Callable[[Vec3], tuple[float, float, bool]]


def spherical_to_cartesian(r_rg: float, theta_rad: float, phi_rad: float) -> Vec3:
    sin_theta = math.sin(theta_rad)
    return (
        r_rg * sin_theta * math.cos(phi_rad),
        r_rg * sin_theta * math.sin(phi_rad),
        r_rg * math.cos(theta_rad),
    )


def rz_from_spherical(r_rg: float, theta_rad: float) -> tuple[float, float]:
    return r_rg * math.sin(theta_rad), r_rg * math.cos(theta_rad)


class MediumRenderer:
    """Renderer for diagnostic views of the analytic torus density field."""

    @staticmethod
    def metadata() -> dict[str, Any]:
        payload = medium_metadata()
        payload["medium_renderer_used"] = True
        return payload

    @staticmethod
    def density(r_rg: float, theta_rad: float, values: dict[str, dict[str, Any]], *, density_floor_g_cm3: float = 0.0) -> float:
        return analytic_torus_density_g_cm3(r_rg, theta_rad, values, density_floor_g_cm3=density_floor_g_cm3)

    @staticmethod
    def density_grid_Rz(
        values: dict[str, dict[str, Any]],
        *,
        limit_rg: float,
        n_r: int = 260,
        n_z: int = 320,
        density_floor_g_cm3: float = 0.0,
    ) -> tuple[list[float], list[float], list[list[float]], list[float]]:
        r_axis = [limit_rg * i / (n_r - 1) for i in range(n_r)]
        z_axis = [-limit_rg + 2.0 * limit_rg * i / (n_z - 1) for i in range(n_z)]
        density_grid: list[list[float]] = []
        positive: list[float] = []
        for z in z_axis:
            row: list[float] = []
            for cylindrical_r in r_axis:
                radius = math.hypot(cylindrical_r, z)
                theta = math.atan2(cylindrical_r, z) if radius > 0.0 else 0.0
                rho = MediumRenderer.density(radius, theta, values, density_floor_g_cm3=density_floor_g_cm3)
                row.append(rho)
                if rho > 0.0:
                    positive.append(rho)
            density_grid.append(row)
        return r_axis, z_axis, density_grid, positive

    @staticmethod
    def draw_density_map_Rz(
        ax: Any,
        values: dict[str, dict[str, Any]],
        *,
        limit_rg: float,
        density_floor_g_cm3: float = 0.0,
        n_r: int = 260,
        n_z: int = 320,
        cmap: str = "magma",
        alpha: float = 0.78,
        add_colorbar_to: Any | None = None,
        colorbar_label: str = r"$\rho$ [g cm$^{-3}$]",
    ) -> dict[str, Any]:
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        from matplotlib.patches import Circle

        torus = values["analytic_torus"]
        r_inner = float(torus["r_inner_rg"])
        r_outer = float(torus["r_outer_rg"])
        half_angle = math.radians(float(torus["half_opening_angle_deg"]))
        r_axis, z_axis, density_grid, positive = MediumRenderer.density_grid_Rz(
            values,
            limit_rg=limit_rg,
            n_r=n_r,
            n_z=n_z,
            density_floor_g_cm3=density_floor_g_cm3,
        )
        if positive:
            image = ax.imshow(
                density_grid,
                origin="lower",
                extent=[min(r_axis), max(r_axis), min(z_axis), max(z_axis)],
                aspect="equal",
                cmap=cmap,
                norm=LogNorm(vmin=max(min(positive), 1.0e-30), vmax=max(positive)),
                alpha=alpha,
                zorder=0,
            )
            if add_colorbar_to is not None:
                cbar = add_colorbar_to.colorbar(image, ax=ax, shrink=0.82)
                cbar.set_label(colorbar_label)
            contour_levels = [max(positive) * factor for factor in (1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)]
            ax.contour(
                r_axis,
                z_axis,
                density_grid,
                levels=[level for level in contour_levels if min(positive) < level < max(positive)],
                colors="#f8fafc",
                linewidths=0.55,
                alpha=0.66,
                zorder=2,
            )
        else:
            ax.imshow(
                density_grid,
                origin="lower",
                extent=[min(r_axis), max(r_axis), min(z_axis), max(z_axis)],
                aspect="equal",
                cmap="Greys",
                alpha=alpha,
                zorder=0,
            )
        ax.add_patch(Circle((0.0, 0.0), r_inner, fill=False, edgecolor="#38bdf8", linestyle="--", linewidth=1.2, alpha=0.85, label="hard radial cuts"))
        ax.add_patch(Circle((0.0, 0.0), r_outer, fill=False, edgecolor="#38bdf8", linestyle="--", linewidth=1.2, alpha=0.85))
        for sign in (-1.0, 1.0):
            theta = 0.5 * math.pi + sign * half_angle
            ax.plot(
                [0.0, limit_rg * math.sin(theta)],
                [0.0, limit_rg * math.cos(theta)],
                color="#fde047",
                linestyle=":",
                linewidth=1.0,
                alpha=0.78,
                label="Gaussian width, not boundary" if sign < 0.0 else None,
            )
        return MediumRenderer.metadata()

    @staticmethod
    def proxy_shell_rings(values: dict[str, dict[str, Any]], *, phi_steps: int = 128) -> list[dict[str, Any]]:
        torus = values["analytic_torus"]
        r_inner = float(torus["r_inner_rg"])
        r_peak = float(torus["r_peak_rg"])
        r_outer = float(torus["r_outer_rg"])
        theta_width = math.radians(float(torus["half_opening_angle_deg"]))
        rings: list[dict[str, Any]] = []
        for radius, radius_label, radius_alpha in [(r_inner, "hard inner radial cut", 0.36), (r_peak, "peak density radius", 0.54), (r_outer, "hard outer radial cut", 0.36)]:
            for multiple in (-2.0, -1.0, 0.0, 1.0, 2.0):
                theta = min(math.pi - 1.0e-6, max(1.0e-6, 0.5 * math.pi + multiple * theta_width))
                angular_alpha = math.exp(-0.5 * multiple * multiple)
                points = [spherical_to_cartesian(radius, theta, 2.0 * math.pi * i / phi_steps) for i in range(phi_steps + 1)]
                rings.append(
                    {
                        "points": points,
                        "radius_rg": radius,
                        "theta_rad": theta,
                        "density_relative": angular_alpha,
                        "label": radius_label if multiple == 0.0 else "Gaussian angular density level",
                        "alpha": max(0.06, radius_alpha * angular_alpha),
                        "hard_radial_cut": radius in {r_inner, r_outer},
                    }
                )
        return rings

    @staticmethod
    def draw_shell_3d_proxy(
        ax: Any,
        values: dict[str, dict[str, Any]],
        project: Projector2D,
        *,
        color: str = "#64748b",
        zorder: int = 3,
    ) -> dict[str, Any]:
        for ring in MediumRenderer.proxy_shell_rings(values):
            projected = [project(point) for point in ring["points"]]
            xs = [point[0] for point in projected]
            ys = [point[1] for point in projected]
            linewidth = 1.0 if ring["hard_radial_cut"] else 0.65
            ax.plot(xs, ys, color=color, linewidth=linewidth, alpha=float(ring["alpha"]), zorder=zorder)
        return MediumRenderer.metadata()

    @staticmethod
    def draw_camera_projection_proxy(
        ax: Any,
        values: dict[str, dict[str, Any]],
        project_camera: CameraProjector,
        *,
        color: str = "0.55",
    ) -> dict[str, Any]:
        for ring in MediumRenderer.proxy_shell_rings(values, phi_steps=160):
            pts = [project_camera(point) for point in ring["points"]]
            xs = [pt[0] for pt in pts if pt[2]]
            ys = [pt[1] for pt in pts if pt[2]]
            if xs:
                ax.plot(xs, ys, color=color, alpha=float(ring["alpha"]), linewidth=0.85 if ring["hard_radial_cut"] else 0.55, zorder=3)
        return MediumRenderer.metadata()

    @staticmethod
    def draw_polar_cones_3d_proxy(ax: Any, values: dict[str, dict[str, Any]], project: Projector2D, *, zorder: int = 2) -> None:
        cone = values["polar_cone"]
        if not bool(cone.get("enabled", True)):
            return
        opening = math.radians(float(cone["opening_angle_deg"]))
        r_min = float(cone["r_min_rg"])
        r_max = float(cone["r_max_rg"])
        signs = (1.0, -1.0) if str(cone.get("draw_mode")) == "bipolar_funnel" else (1.0,)
        for sign in signs:
            for phi in [2.0 * math.pi * i / 24.0 for i in range(24)]:
                p0 = spherical_to_cartesian(r_min, opening if sign > 0 else math.pi - opening, phi)
                p1 = spherical_to_cartesian(r_max, opening if sign > 0 else math.pi - opening, phi)
                x0, y0 = project(p0)
                x1, y1 = project(p1)
                ax.plot([x0, x1], [y0, y1], color="#ca8a04", linewidth=0.55, alpha=0.24, zorder=zorder)
