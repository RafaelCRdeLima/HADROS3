#include "hadros/cascade/kerr_local_tetrad.hpp"

#include <cmath>

namespace hadros::cascade {

LocalDirection normalize_local_direction(LocalDirection direction)
{
    const double n = std::sqrt(
        direction.nr * direction.nr
        + direction.ntheta * direction.ntheta
        + direction.nphi * direction.nphi
    );
    if (!std::isfinite(n) || n <= 0.0) {
        return {1.0, 0.0, 0.0};
    }
    direction.nr /= n;
    direction.ntheta /= n;
    direction.nphi /= n;
    return direction;
}

double covariant_null_norm(
    const KerrMetric& metric,
    double r,
    double theta,
    const double p_contravariant[4]
)
{
    double g[4][4];
    metric.metric(r, theta, g);
    double norm = 0.0;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            norm += g[mu][nu] * p_contravariant[mu] * p_contravariant[nu];
        }
    }
    return norm;
}

double zamo_packet_energy(
    const KerrMetric& metric,
    double r,
    double theta,
    double p_cov_t,
    double p_cov_phi
)
{
    const double alpha = metric.lapse(r, theta);
    const double omega = metric.omega_frame_drag(r, theta);
    return -(p_cov_t + omega * p_cov_phi) / alpha;
}

KerrTetradInitialization initialize_zamo_null_packet(
    const KerrMetric& metric,
    double r,
    double theta,
    double phi,
    LocalDirection direction
)
{
    KerrTetradInitialization out;
    out.state.t = 0.0;
    out.state.r = r;
    out.state.theta = theta;
    out.state.phi = phi;

    if (r <= metric.horizon_radius()) {
        out.status = "INSIDE_HORIZON";
        return out;
    }

    double g[4][4];
    metric.metric(r, theta, g);
    const double alpha = metric.lapse(r, theta);
    const double omega = metric.omega_frame_drag(r, theta);
    const double grr = g[1][1];
    const double gthth = g[2][2];
    const double gphph = g[3][3];
    if (
        !std::isfinite(alpha) || alpha <= 0.0
        || !std::isfinite(grr) || grr <= 0.0
        || !std::isfinite(gthth) || gthth <= 0.0
        || !std::isfinite(gphph) || gphph <= 0.0
    ) {
        out.status = "BAD_METRIC";
        return out;
    }

    direction = normalize_local_direction(direction);

    out.p_contravariant[0] = 1.0 / alpha;
    out.p_contravariant[1] = direction.nr / std::sqrt(grr);
    out.p_contravariant[2] = direction.ntheta / std::sqrt(gthth);
    out.p_contravariant[3] =
        direction.nphi / std::sqrt(gphph)
        + omega * out.p_contravariant[0];

    for (int mu = 0; mu < 4; ++mu) {
        out.p_covariant[mu] = 0.0;
        for (int nu = 0; nu < 4; ++nu) {
            out.p_covariant[mu] += g[mu][nu] * out.p_contravariant[nu];
        }
    }

    out.state.pt = out.p_covariant[0];
    out.state.pr = out.p_covariant[1];
    out.state.ptheta = out.p_covariant[2];
    out.state.pphi = out.p_covariant[3];
    out.null_norm = covariant_null_norm(metric, r, theta, out.p_contravariant);
    out.zamo_energy = zamo_packet_energy(metric, r, theta, out.state.pt, out.state.pphi);
    out.valid = std::isfinite(out.null_norm)
        && std::isfinite(out.zamo_energy)
        && out.zamo_energy > 0.0;
    out.status = out.valid ? "OK" : "BAD_TETRAD";
    return out;
}

}  // namespace hadros::cascade
