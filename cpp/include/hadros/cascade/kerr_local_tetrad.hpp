#pragma once

#include "geodesic_state.hpp"
#include "kerr_metric.hpp"

#include <string>

namespace hadros::cascade {

struct LocalDirection {
    double nr = 1.0;
    double ntheta = 0.0;
    double nphi = 0.0;
};

struct KerrTetradInitialization {
    GeodesicState state{};
    double p_contravariant[4] = {0.0, 0.0, 0.0, 0.0};
    double p_covariant[4] = {0.0, 0.0, 0.0, 0.0};
    double null_norm = 0.0;
    double zamo_energy = 1.0;
    bool valid = false;
    std::string status = "UNINITIALIZED";
};

LocalDirection normalize_local_direction(LocalDirection direction);

KerrTetradInitialization initialize_zamo_null_packet(
    const KerrMetric& metric,
    double r,
    double theta,
    double phi,
    LocalDirection direction
);

double covariant_null_norm(
    const KerrMetric& metric,
    double r,
    double theta,
    const double p_contravariant[4]
);

double zamo_packet_energy(
    const KerrMetric& metric,
    double r,
    double theta,
    double p_cov_t,
    double p_cov_phi
);

}  // namespace hadros::cascade
