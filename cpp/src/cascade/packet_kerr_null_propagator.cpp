#include "hadros/cascade/packet_kerr_null_propagator.hpp"

#include "hadros/cascade/kerr_local_tetrad.hpp"
#include "kerr_geodesic.hpp"
#include "kerr_metric.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace hadros::cascade {
namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

double dot(const Vec3& a, const Vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

double norm(const Vec3& v)
{
    return std::sqrt(dot(v, v));
}

bool normalize(Vec3& v)
{
    const double n = norm(v);
    if (!std::isfinite(n) || n <= 0.0) {
        return false;
    }
    v.x /= n;
    v.y /= n;
    v.z /= n;
    return true;
}

Vec3 spherical_position(double r, double theta, double phi)
{
    return {
        r * std::sin(theta) * std::cos(phi),
        r * std::sin(theta) * std::sin(phi),
        r * std::cos(theta)
    };
}

void cartesian_to_spherical(const Vec3& pos, double& r, double& theta, double& phi)
{
    r = norm(pos);
    if (!std::isfinite(r) || r <= 0.0) {
        theta = 0.5 * PI;
        phi = 0.0;
        return;
    }
    theta = std::acos(std::clamp(pos.z / r, -1.0, 1.0));
    phi = std::atan2(pos.y, pos.x);
}

LocalDirection local_spherical_direction(const Vec3& pos, const Vec3& dir)
{
    double r = 0.0;
    double theta = 0.0;
    double phi = 0.0;
    cartesian_to_spherical(pos, r, theta, phi);
    const double st = std::sin(theta);
    const double ct = std::cos(theta);
    const double sp = std::sin(phi);
    const double cp = std::cos(phi);
    const Vec3 e_r{st * cp, st * sp, ct};
    const Vec3 e_theta{ct * cp, ct * sp, -st};
    const Vec3 e_phi{-sp, cp, 0.0};
    return {dot(dir, e_r), dot(dir, e_theta), dot(dir, e_phi)};
}

}  // namespace

bool is_effective_null_packet_class(const std::string& classification)
{
    return classification == "MASSLESS_NULL" || classification == "ULTRARELATIVISTIC_NULL_OK";
}

PacketKerrNullPropagator::PacketKerrNullPropagator(PacketKerrNullPropagationConfig config)
    : config_(std::move(config))
{
}

const PacketKerrNullPropagationConfig& PacketKerrNullPropagator::config() const noexcept
{
    return config_;
}

PacketKerrNullPropagationResult PacketKerrNullPropagator::propagate(
    const EscapingParticlePacket& packet,
    const std::string& classification
) const
{
    PacketKerrNullPropagationResult result;
    result.event_id = packet.event_id;
    result.pdg_id = packet.pdg_id;
    result.classification = classification;
    result.energy_gev = packet.energy_gev;
    result.weighted_energy_gev = packet.energy_gev * packet.weight;
    result.x0 = packet.x;
    result.y0 = packet.y;
    result.z0 = packet.z;
    result.kerr_init_mode = config_.kerr_init_mode;
    result.observed_energy_proxy_gev = packet.energy_gev;
    result.backend_label = "REAL_HADROS_KERR_GEODESIC";

    if (!is_effective_null_packet_class(classification)) {
        result.final_status = "SKIPPED_CLASS";
        return result;
    }

    Vec3 pos{packet.x, packet.y, packet.z};
    Vec3 dir{packet.px_gev, packet.py_gev, packet.pz_gev};
    if (!normalize(dir)) {
        result.final_status = "FAILED_INTEGRATION";
        return result;
    }
    result.dir_x = dir.x;
    result.dir_y = dir.y;
    result.dir_z = dir.z;

    KerrMetric metric(config_.spin);
    KerrGeodesic geodesic(metric, config_.step_rg, 1.0e-6, KerrDerivativeMode::FiniteDifference);
    double r0 = 0.0;
    double theta0 = 0.0;
    double phi0 = 0.0;
    cartesian_to_spherical(pos, r0, theta0, phi0);
    constexpr double theta_eps = 1.0e-6;
    theta0 = std::clamp(theta0, theta_eps, PI - theta_eps);
    if (r0 <= metric.horizon_radius()) {
        result.final_status = "HIT_HORIZON";
        return result;
    }

    const auto init = initialize_zamo_null_packet(
        metric,
        r0,
        theta0,
        phi0,
        local_spherical_direction(pos, dir)
    );
    if (!init.valid) {
        result.final_status = "FAILED_INTEGRATION";
        return result;
    }

    GeodesicState y = init.state;
    result.initial_hamiltonian = geodesic.hamiltonian(y);
    result.redshift_factor = 1.0;
    result.final_status = "FAILED_INTEGRATION";

    const double r_stop = metric.horizon_radius() + 1.0e-3;
    GeodesicState previous = y;
    for (std::size_t step = 0; step < config_.max_steps; ++step) {
        if (y.r <= r_stop) {
            result.final_status = "HIT_HORIZON";
            break;
        }
        if (y.r >= config_.domain_radius_rg && step > 0) {
            result.final_status = "ESCAPED_DOMAIN";
            break;
        }
        previous = y;
        geodesic.step_adaptive(y);
        if (!std::isfinite(y.r) || !std::isfinite(y.theta) || !std::isfinite(y.phi)) {
            result.final_status = "FAILED_INTEGRATION";
            break;
        }
        const Vec3 p0 = spherical_position(previous.r, previous.theta, previous.phi);
        const Vec3 p1 = spherical_position(y.r, y.theta, y.phi);
        result.path_length_rg += norm({p1.x - p0.x, p1.y - p0.y, p1.z - p0.z});
        result.affine_steps = step + 1;
    }

    result.final_r = y.r;
    result.final_theta = y.theta;
    result.final_phi = y.phi;
    const Vec3 final_pos = spherical_position(y.r, y.theta, y.phi);
    result.final_x = final_pos.x;
    result.final_y = final_pos.y;
    result.final_z = final_pos.z;
    result.observer_theta = y.theta;
    result.observer_phi = y.phi;
    result.final_hamiltonian = geodesic.hamiltonian(y);
    return result;
}

}  // namespace hadros::cascade
