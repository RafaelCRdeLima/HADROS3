#pragma once

#include "hadros/cascade/types.hpp"

#include <cstddef>
#include <string>

namespace hadros::cascade {

struct PacketKerrNullPropagationConfig {
    double spin = 0.0;
    double domain_radius_rg = 200.0;
    double horizon_radius_rg = 1.2;
    double step_rg = 0.05;
    std::size_t max_steps = 200000;
    std::string kerr_init_mode = "flat_local";
};

struct PacketKerrNullPropagationResult {
    std::uint64_t event_id = 0;
    int pdg_id = 0;
    std::string classification = "UNKNOWN";
    double energy_gev = 0.0;
    double weighted_energy_gev = 0.0;
    double x0 = 0.0;
    double y0 = 0.0;
    double z0 = 0.0;
    double dir_x = 0.0;
    double dir_y = 0.0;
    double dir_z = 0.0;
    double final_r = 0.0;
    double final_theta = 0.0;
    double final_phi = 0.0;
    double final_x = 0.0;
    double final_y = 0.0;
    double final_z = 0.0;
    double observer_theta = 0.0;
    double observer_phi = 0.0;
    std::string final_status = "FAILED_INTEGRATION";
    double path_length_rg = 0.0;
    std::size_t affine_steps = 0;
    double redshift_factor = 1.0;
    double observed_energy_proxy_gev = 0.0;
    std::string kerr_init_mode = "flat_local";
    double initial_hamiltonian = 0.0;
    double final_hamiltonian = 0.0;
    std::string backend_label = "REAL_HADROS_KERR_GEODESIC";
};

class PacketKerrNullPropagator {
public:
    explicit PacketKerrNullPropagator(PacketKerrNullPropagationConfig config = {});

    PacketKerrNullPropagationResult propagate(
        const EscapingParticlePacket& packet,
        const std::string& classification
    ) const;

    const PacketKerrNullPropagationConfig& config() const noexcept;

private:
    PacketKerrNullPropagationConfig config_;
};

bool is_effective_null_packet_class(const std::string& classification);

}  // namespace hadros::cascade
