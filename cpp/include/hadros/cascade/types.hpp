#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace hadros::cascade {

struct PrimaryParticle {
    int pdg_id = 0;
    double energy_gev = 0.0;
    double px_gev = 0.0;
    double py_gev = 0.0;
    double pz_gev = 0.0;
    double mass_gev = 0.0;
    double weight = 1.0;
    std::uint64_t event_id = 0;
    std::uint64_t seed = 0;
    std::string particle_label = "unspecified";
};

struct InteractionPoint {
    std::uint64_t event_id = 0;
    double x_cm = 0.0;
    double y_cm = 0.0;
    double z_cm = 0.0;
    double r_cm = 0.0;
    double theta_rad = 0.0;
    double phi_rad = 0.0;
    double density_g_cm3 = 0.0;
    double temperature_mev = 0.0;
    double temperature_proxy = 0.0;
    double composition_proxy = 0.0;
    double electron_fraction = 0.5;
    double column_before_cm2 = 0.0;
    double tau_before = 0.0;
    double weight = 1.0;
    std::string region_label = "unspecified";
    std::string region_class = "unspecified";
};

struct PrimaryInteractionEvent {
    std::uint64_t event_id = 0;
    PrimaryParticle primary;
    InteractionPoint point;
    std::string interaction_model = "unspecified";
    std::string backend_name = "unspecified";
    double x_bjorken = -1.0;
    double q2_gev2 = -1.0;
    double y_inelasticity = -1.0;
    std::string metadata = "";
};

struct PrimaryNeutrinoEvent {
    std::uint64_t event_id = 0;
    int neutrino_pdg = 14;
    double energy_gev = 0.0;
    double weight = 1.0;
    bool charged_current = true;
    InteractionPoint interaction;
};

struct SecondaryParticle {
    std::uint64_t event_id = 0;
    std::uint64_t parent_event_id = 0;
    int pdg = 0;
    int pdg_id = 0;
    double energy_gev = 0.0;
    double px_gev = 0.0;
    double py_gev = 0.0;
    double pz_gev = 0.0;
    double mass_gev = 0.0;
    double weight = 1.0;
    bool stable = true;
    std::string origin = "analytic";
    std::string origin_backend = "analytic";
    double interaction_x_rg = std::numeric_limits<double>::quiet_NaN();
    double interaction_y_rg = std::numeric_limits<double>::quiet_NaN();
    double interaction_z_rg = std::numeric_limits<double>::quiet_NaN();
    double geant4_box_origin_x_rg = std::numeric_limits<double>::quiet_NaN();
    double geant4_box_origin_y_rg = std::numeric_limits<double>::quiet_NaN();
    double geant4_box_origin_z_rg = std::numeric_limits<double>::quiet_NaN();
    double exit_x_rg = std::numeric_limits<double>::quiet_NaN();
    double exit_y_rg = std::numeric_limits<double>::quiet_NaN();
    double exit_z_rg = std::numeric_limits<double>::quiet_NaN();
    double geant4_box_origin_x_cm = std::numeric_limits<double>::quiet_NaN();
    double geant4_box_origin_y_cm = std::numeric_limits<double>::quiet_NaN();
    double geant4_box_origin_z_cm = std::numeric_limits<double>::quiet_NaN();
    double geant4_local_exit_x_cm = std::numeric_limits<double>::quiet_NaN();
    double geant4_local_exit_y_cm = std::numeric_limits<double>::quiet_NaN();
    double geant4_local_exit_z_cm = std::numeric_limits<double>::quiet_NaN();
    double geant4_local_cm_per_rg = std::numeric_limits<double>::quiet_NaN();
    std::string position_status = "MISSING_PARTICLE_POSITION";
};

struct InteractionResult {
    std::uint64_t event_id = 0;
    double input_energy_gev = 0.0;
    double visible_energy_gev = 0.0;
    double invisible_energy_gev = 0.0;
    double escaped_energy_gev = 0.0;
    double deposited_energy_gev = 0.0;
    std::vector<SecondaryParticle> secondaries;
    std::string metadata = "";
};

struct CascadeResult {
    std::uint64_t event_id = 0;
    double weight = 1.0;
    double deposited_em_gev = 0.0;
    double deposited_hadronic_gev = 0.0;
    double escaped_muon_gev = 0.0;
    double escaped_neutrino_gev = 0.0;
    std::vector<SecondaryParticle> escaped_particles;

    double deposited_energy_gev() const {
        return deposited_em_gev + deposited_hadronic_gev;
    }

    double escaped_energy_gev() const {
        return escaped_muon_gev + escaped_neutrino_gev;
    }

    double total_accounted_energy_gev() const {
        return deposited_energy_gev() + escaped_energy_gev();
    }
};

struct EscapingParticlePacket {
    std::uint64_t event_id = 0;
    int pdg_id = 0;
    double energy_gev = 0.0;
    double px_gev = 0.0;
    double py_gev = 0.0;
    double pz_gev = 0.0;
    double weight = 1.0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double r = 0.0;
    double theta = 0.0;
    double phi = 0.0;
    std::string origin_backend = "unspecified";
};

struct EscapingPacketCollection {
    std::vector<EscapingParticlePacket> packets;
    double total_energy_gev = 0.0;
    double total_weighted_energy_gev = 0.0;
};

class PacketPropagator {
public:
    virtual ~PacketPropagator() = default;
    virtual void propagate(EscapingParticlePacket& packet) = 0;
};

class NullPacketPropagator final : public PacketPropagator {
public:
    void propagate(EscapingParticlePacket& packet) override { (void)packet; }
};

class MassivePacketPropagator final : public PacketPropagator {
public:
    void propagate(EscapingParticlePacket& packet) override { (void)packet; }
};

inline bool is_neutrino_pdg(int pdg) {
    const int a = pdg < 0 ? -pdg : pdg;
    return a == 12 || a == 14 || a == 16;
}

inline bool is_charged_lepton_pdg(int pdg) {
    const int a = pdg < 0 ? -pdg : pdg;
    return a == 11 || a == 13 || a == 15;
}

inline bool is_muon_pdg(int pdg) {
    const int a = pdg < 0 ? -pdg : pdg;
    return a == 13;
}

inline bool is_electron_or_photon_pdg(int pdg) {
    const int a = pdg < 0 ? -pdg : pdg;
    return a == 11 || a == 22;
}

}  // namespace hadros::cascade
