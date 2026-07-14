#include "G4Box.hh"
#include "G4Event.hh"
#include "G4LogicalVolume.hh"
#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleDefinition.hh"
#include "G4ParticleTable.hh"
#include "G4VProcess.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include "G4RunManagerFactory.hh"
#include "G4Step.hh"
#include "G4SystemOfUnits.hh"
#include "G4Threading.hh"
#include "G4Track.hh"
#include "G4UserEventAction.hh"
#include "G4UserSteppingAction.hh"
#include "G4VModularPhysicsList.hh"
#include "G4VPhysicalVolume.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4Version.hh"
#include "FTFP_BERT.hh"
#include "Randomize.hh"

#include "HepMC3/GenEvent.h"
#include "HepMC3/GenParticle.h"
#include "HepMC3/ReaderAscii.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Options {
  fs::path input;
  fs::path output_dir;
  std::string mode = "import_check";
  std::string material = "G4_Galactic";
  std::string physics_list = "FTFP_BERT";
  double density_g_cm3 = 1.0;
  double hydrogen_mass_fraction = 0.75;
  double half_size_mm = 10.0;
  double world_margin_mm = 10.0;
  double production_cut_mm = 0.1;
  double max_energy_gev = 1.0e5;
  int max_events = 2;
  std::size_t max_recorded_steps = 50000;
  std::uint64_t seed = 59001;
};

std::string json_escape(const std::string &s) {
  std::ostringstream out;
  for (const char c : s) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << c;
    }
  }
  return out.str();
}

std::string q(const std::string &s) { return "\"" + json_escape(s) + "\""; }

std::string require_value(int argc, char **argv, int &i) {
  if (++i >= argc) throw std::runtime_error(std::string("missing value after ") + argv[i - 1]);
  return argv[i];
}

Options parse_options(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--input") o.input = require_value(argc, argv, i);
    else if (a == "--output-dir") o.output_dir = require_value(argc, argv, i);
    else if (a == "--mode") o.mode = require_value(argc, argv, i);
    else if (a == "--material") o.material = require_value(argc, argv, i);
    else if (a == "--physics-list") o.physics_list = require_value(argc, argv, i);
    else if (a == "--density-g-cm3") o.density_g_cm3 = std::stod(require_value(argc, argv, i));
    else if (a == "--hydrogen-mass-fraction") o.hydrogen_mass_fraction = std::stod(require_value(argc, argv, i));
    else if (a == "--half-size-mm") o.half_size_mm = std::stod(require_value(argc, argv, i));
    else if (a == "--world-margin-mm") o.world_margin_mm = std::stod(require_value(argc, argv, i));
    else if (a == "--production-cut-mm") o.production_cut_mm = std::stod(require_value(argc, argv, i));
    else if (a == "--max-energy-gev") o.max_energy_gev = std::stod(require_value(argc, argv, i));
    else if (a == "--max-events") o.max_events = std::stoi(require_value(argc, argv, i));
    else if (a == "--max-recorded-steps") o.max_recorded_steps = std::stoull(require_value(argc, argv, i));
    else if (a == "--seed") o.seed = std::stoull(require_value(argc, argv, i));
    else if (a == "--help") {
      std::cout << "hadros3_geant4_transport --input FILE --output-dir DIR "
                   "[--mode import_check|vacuum_smoke|material_smoke|real_free]\n";
      std::exit(0);
    } else throw std::runtime_error("unknown argument: " + a);
  }
  if (o.input.empty()) throw std::runtime_error("--input is required");
  if (o.output_dir.empty()) throw std::runtime_error("--output-dir is required");
  if (o.max_events <= 0 || o.max_recorded_steps == 0 || o.half_size_mm <= 0.0 || o.world_margin_mm <= 0.0 ||
      o.production_cut_mm <= 0.0 || o.max_energy_gev <= 0.0 || o.density_g_cm3 <= 0.0)
    throw std::runtime_error("numeric limits, dimensions, density and cuts must be positive");
  if (o.hydrogen_mass_fraction < 0.0 || o.hydrogen_mass_fraction > 1.0)
    throw std::runtime_error("hydrogen mass fraction must lie in [0,1]");
  static const std::unordered_set<std::string> modes{
      "import_check", "vacuum_smoke", "material_smoke", "real_free"};
  if (!modes.count(o.mode)) throw std::runtime_error("unsupported mode: " + o.mode);
  if (o.physics_list != "FTFP_BERT") throw std::runtime_error("only FTFP_BERT is validated in H3-W11 v1");
  return o;
}

struct Primary {
  int pdg = 0;
  int source_id = 0;
  double px_gev = 0.0;
  double py_gev = 0.0;
  double pz_gev = 0.0;
  double energy_gev = 0.0;
  double mass_gev = 0.0;
};

struct InputEvent {
  int hepmc_event_number = 0;
  std::vector<Primary> primaries;
  std::vector<double> weights;
};

struct ImportAudit {
  std::vector<InputEvent> events;
  std::size_t particles_seen = 0;
  std::size_t final_particles = 0;
  std::size_t unsupported_species = 0;
  std::size_t unsupported_energy = 0;
  double maximum_energy_gev = 0.0;
  std::vector<std::string> violations;
};

class HadrosPhysicsList final : public FTFP_BERT {
 public:
  explicit HadrosPhysicsList(double cut) : FTFP_BERT(0) { defaultCutValue = cut; }
};

bool finite_primary(const Primary &p) {
  return std::isfinite(p.px_gev) && std::isfinite(p.py_gev) && std::isfinite(p.pz_gev) &&
         std::isfinite(p.energy_gev) && std::isfinite(p.mass_gev) && p.energy_gev > 0.0 && p.mass_gev >= 0.0;
}

ImportAudit read_input(const Options &o) {
  HepMC3::ReaderAscii reader(o.input.string());
  if (reader.failed()) throw std::runtime_error("could not open HepMC3 input: " + o.input.string());
  ImportAudit audit;
  while (!reader.failed() && static_cast<int>(audit.events.size()) < o.max_events) {
    HepMC3::GenEvent event;
    reader.read_event(event);
    if (reader.failed()) break;
    InputEvent dst;
    dst.hepmc_event_number = event.event_number();
    dst.weights = event.weights();
    for (const auto &particle : event.particles()) {
      ++audit.particles_seen;
      if (particle->status() != 1 || particle->end_vertex()) continue;
      ++audit.final_particles;
      const auto &m = particle->momentum();
      const double generated_mass = particle->is_generated_mass_set() ? particle->generated_mass() : std::abs(m.m());
      Primary p{particle->pid(), particle->id(), m.px(), m.py(), m.pz(), m.e(), generated_mass};
      audit.maximum_energy_gev = std::max(audit.maximum_energy_gev, p.energy_gev);
      if (!finite_primary(p)) {
        audit.violations.push_back("event " + std::to_string(dst.hepmc_event_number) +
                                   " particle " + std::to_string(p.source_id) + " has non-finite/invalid four-vector");
        continue;
      }
      const double p2 = p.px_gev*p.px_gev + p.py_gev*p.py_gev + p.pz_gev*p.pz_gev;
      const double shell_scale = std::max(1.0, p.energy_gev*p.energy_gev);
      const double shell_residual = std::abs(p.energy_gev*p.energy_gev - p2 - p.mass_gev*p.mass_gev) / shell_scale;
      if (shell_residual > 5.0e-8) {
        audit.violations.push_back("event " + std::to_string(dst.hepmc_event_number) +
                                   " particle " + std::to_string(p.source_id) + " is off shell");
      }
      if (p.energy_gev > o.max_energy_gev) {
        ++audit.unsupported_energy;
        audit.violations.push_back("unsupported_energy event=" + std::to_string(dst.hepmc_event_number) +
                                   " particle=" + std::to_string(p.source_id) + " pdg=" + std::to_string(p.pdg) +
                                   " energy_gev=" + std::to_string(p.energy_gev) +
                                   " maximum_gev=" + std::to_string(o.max_energy_gev));
      }
      if (!G4ParticleTable::GetParticleTable()->FindParticle(p.pdg)) {
        ++audit.unsupported_species;
        audit.violations.push_back("unsupported_species event=" + std::to_string(dst.hepmc_event_number) +
                                   " particle=" + std::to_string(p.source_id) + " pdg=" + std::to_string(p.pdg));
      }
      dst.primaries.push_back(p);
    }
    if (dst.primaries.empty()) audit.violations.push_back("event " + std::to_string(dst.hepmc_event_number) + " has no final particles");
    audit.events.push_back(std::move(dst));
  }
  reader.close();
  if (audit.events.empty()) throw std::runtime_error("HepMC3 input contains no readable events");
  return audit;
}

void write_import_report(const Options &o, const ImportAudit &a, const std::string &status) {
  fs::create_directories(o.output_dir);
  std::ofstream f(o.output_dir / "geant4_import_report.json");
  f << std::setprecision(17) << "{\n"
    << "  \"status\": " << q(status) << ",\n"
    << "  \"mode\": " << q(o.mode) << ",\n"
    << "  \"generator_frame\": \"local_matter_tetrad\",\n"
    << "  \"momentum_unit\": \"GeV\",\n"
    << "  \"length_unit\": \"mm\",\n"
    << "  \"events\": " << a.events.size() << ",\n"
    << "  \"particles_seen\": " << a.particles_seen << ",\n"
    << "  \"final_particles\": " << a.final_particles << ",\n"
    << "  \"unsupported_species\": " << a.unsupported_species << ",\n"
    << "  \"unsupported_energy\": " << a.unsupported_energy << ",\n"
    << "  \"maximum_energy_gev\": " << a.maximum_energy_gev << ",\n"
    << "  \"validated_maximum_energy_gev\": " << o.max_energy_gev << ",\n"
    << "  \"violations\": [";
  for (std::size_t i = 0; i < a.violations.size(); ++i) f << (i ? "," : "") << "\n    " << q(a.violations[i]);
  if (!a.violations.empty()) f << '\n' << "  ";
  f << "]\n}\n";
}

class DetectorConstruction final : public G4VUserDetectorConstruction {
 public:
  explicit DetectorConstruction(Options options) : options_(std::move(options)) {}

  G4VPhysicalVolume *Construct() override {
    auto *nist = G4NistManager::Instance();
    auto *vacuum = nist->FindOrBuildMaterial("G4_Galactic");
    const double world_half = (options_.half_size_mm + options_.world_margin_mm) * mm;
    auto *world_solid = new G4Box("WorldSolid", world_half, world_half, world_half);
    auto *world_logical = new G4LogicalVolume(world_solid, vacuum, "World");
    auto *world = new G4PVPlacement(nullptr, {}, world_logical, "World", nullptr, false, 0, true);

    G4Material *patch_material = vacuum;
    if (options_.mode == "material_smoke" || options_.mode == "real_free") {
      if (options_.material == "HADROS3_H_HE") {
        auto *hydrogen = nist->FindOrBuildElement("H");
        auto *helium = nist->FindOrBuildElement("He");
        patch_material = new G4Material("HADROS3_H_HE", options_.density_g_cm3 * g / cm3, 2);
        patch_material->AddElement(hydrogen, options_.hydrogen_mass_fraction);
        patch_material->AddElement(helium, 1.0 - options_.hydrogen_mass_fraction);
      } else {
        patch_material = nist->FindOrBuildMaterial(options_.material);
        if (!patch_material) throw std::runtime_error("unknown NIST material: " + options_.material);
      }
    }
    const double h = options_.half_size_mm * mm;
    auto *patch_solid = new G4Box("PatchSolid", h, h, h);
    auto *patch_logical = new G4LogicalVolume(patch_solid, patch_material, "Patch");
    new G4PVPlacement(nullptr, {}, patch_logical, "Patch", world_logical, false, 0, true);
    return world;
  }

 private:
  Options options_;
};

class PrimaryGenerator final : public G4VUserPrimaryGeneratorAction {
 public:
  explicit PrimaryGenerator(const std::vector<InputEvent> &events) : events_(events) {}

  void GeneratePrimaries(G4Event *event) override {
    const auto index = static_cast<std::size_t>(event->GetEventID());
    if (index >= events_.size()) throw std::runtime_error("Geant4 requested an event beyond imported input");
    auto *vertex = new G4PrimaryVertex(G4ThreeVector(), 0.0);
    for (const auto &src : events_[index].primaries) {
      auto *definition = G4ParticleTable::GetParticleTable()->FindParticle(src.pdg);
      if (!definition) throw std::runtime_error("missing G4 particle definition after domain audit");
      auto *p = new G4PrimaryParticle(definition, src.px_gev * GeV, src.py_gev * GeV, src.pz_gev * GeV);
      p->SetMass(src.mass_gev * GeV);
      p->SetUserInformation(nullptr);
      vertex->SetPrimary(p);
    }
    event->AddPrimaryVertex(vertex);
  }

 private:
  const std::vector<InputEvent> &events_;
};

struct EventLedger {
  double initial_total_gev = 0.0;
  double deposited_gev = 0.0;
  double escaped_total_gev = 0.0;
  std::size_t escaped_particles = 0;
  std::size_t steps = 0;
  double signed_step_balance_gev = 0.0;
  double absolute_step_balance_gev = 0.0;
};

class OutputState {
 public:
  OutputState(const fs::path &dir, const std::vector<InputEvent> &events, double half_size_mm,
              bool strict_vacuum_ledger, std::size_t max_recorded_steps)
      : events_(events), half_size_mm_(half_size_mm), strict_vacuum_ledger_(strict_vacuum_ledger),
        max_recorded_steps_(max_recorded_steps), event_file_(dir / "geant4_events_raw.jsonl"),
        escaped_file_(dir / "geant4_escaped_particles_raw.jsonl"), step_file_(dir / "geant4_steps_raw.jsonl") {
    if (!event_file_ || !escaped_file_ || !step_file_) throw std::runtime_error("could not create Geant4 raw output files");
  }

  void begin(int event_id) {
    ledger_ = {};
    for (const auto &p : events_.at(static_cast<std::size_t>(event_id)).primaries) ledger_.initial_total_gev += p.energy_gev;
  }

  void deposit(double energy) { ledger_.deposited_gev += energy / GeV; }
  void step() { ++ledger_.steps; }
  void step_balance(double value_gev) {
    ledger_.signed_step_balance_gev += value_gev;
    ledger_.absolute_step_balance_gev += std::abs(value_gev);
  }

  void record_step(int event_id, const G4Step &step) {
    const auto *pre = step.GetPreStepPoint();
    const auto *post = step.GetPostStepPoint();
    const auto *pre_volume = pre->GetPhysicalVolume();
    if (!pre_volume || pre_volume->GetName() != "Patch") return;
    if (recorded_steps_ >= max_recorded_steps_) {
      steps_truncated_ = true;
      return;
    }
    ++recorded_steps_;
    const auto &track = *step.GetTrack();
    const auto &x0 = pre->GetPosition();
    const auto &x1 = post->GetPosition();
    const auto *process = post->GetProcessDefinedStep();
    const auto *creator = track.GetCreatorProcess();
    const std::string process_name = process ? process->GetProcessName() : "unknown";
    const auto &secondaries = *step.GetSecondaryInCurrentStep();
    const bool interaction = process_name != "Transportation" && process_name != "CoupledTransportation" && process_name != "unknown";
    step_file_ << std::setprecision(17)
      << "{\"geant4_event_id\":" << event_id
      << ",\"hepmc_event_number\":" << events_.at(static_cast<std::size_t>(event_id)).hepmc_event_number
      << ",\"event_step_index\":" << ledger_.steps
      << ",\"track_id\":" << track.GetTrackID() << ",\"parent_track_id\":" << track.GetParentID()
      << ",\"pdg_id\":" << track.GetParticleDefinition()->GetPDGEncoding()
      << ",\"pre_position_local_mm\":[" << x0.x()/mm << ',' << x0.y()/mm << ',' << x0.z()/mm << ']'
      << ",\"post_position_local_mm\":[" << x1.x()/mm << ',' << x1.y()/mm << ',' << x1.z()/mm << ']'
      << ",\"pre_total_energy_gev\":" << pre->GetTotalEnergy()/GeV
      << ",\"post_total_energy_gev\":" << post->GetTotalEnergy()/GeV
      << ",\"pre_kinetic_energy_gev\":" << pre->GetKineticEnergy()/GeV
      << ",\"post_kinetic_energy_gev\":" << post->GetKineticEnergy()/GeV
      << ",\"deposited_energy_gev\":" << step.GetTotalEnergyDeposit()/GeV
      << ",\"step_length_mm\":" << step.GetStepLength()/mm
      << ",\"global_time_ns\":" << post->GetGlobalTime()/ns
      << ",\"process_name\":" << q(process_name)
      << ",\"creator_process\":" << q(creator ? creator->GetProcessName() : "primary")
      << ",\"pre_volume\":" << q(pre_volume->GetName())
      << ",\"post_volume\":" << q(post->GetPhysicalVolume() ? post->GetPhysicalVolume()->GetName() : "outside_world")
      << ",\"is_boundary\":" << (post->GetStepStatus() == fGeomBoundary ? "true" : "false")
      << ",\"is_interaction\":" << (interaction ? "true" : "false")
      << ",\"secondaries_created\":" << secondaries.size()
      << ",\"geant4_statistical_weight\":" << track.GetWeight() << "}\n";
  }

  void escape(int event_id, const G4Track &track, const G4StepPoint &point) {
    ++ledger_.escaped_particles;
    ledger_.escaped_total_gev += track.GetTotalEnergy() / GeV;
    const auto &x = point.GetPosition();
    const auto &p = track.GetMomentum();
    const double kinetic = track.GetKineticEnergy() / GeV;
    const double mass = track.GetParticleDefinition()->GetPDGMass() / GeV;
    double nx = 0.0, ny = 0.0, nz = 0.0;
    const double ax = std::abs(x.x()/mm), ay = std::abs(x.y()/mm), az = std::abs(x.z()/mm);
    if (ax >= ay && ax >= az) nx = x.x() >= 0.0 ? 1.0 : -1.0;
    else if (ay >= ax && ay >= az) ny = x.y() >= 0.0 ? 1.0 : -1.0;
    else nz = x.z() >= 0.0 ? 1.0 : -1.0;
    const auto *creator = track.GetCreatorProcess();
    escaped_file_ << std::setprecision(17)
      << "{\"geant4_event_id\":" << event_id
      << ",\"hepmc_event_number\":" << events_.at(static_cast<std::size_t>(event_id)).hepmc_event_number
      << ",\"pdg_id\":" << track.GetParticleDefinition()->GetPDGEncoding()
      << ",\"track_id\":" << track.GetTrackID() << ",\"parent_track_id\":" << track.GetParentID()
      << ",\"position_local_mm\":[" << x.x()/mm << ',' << x.y()/mm << ',' << x.z()/mm << ']'
      << ",\"momentum_local_gev\":[" << p.x()/GeV << ',' << p.y()/GeV << ',' << p.z()/GeV << ']'
      << ",\"energy_local_gev\":" << track.GetTotalEnergy()/GeV
      << ",\"kinetic_energy_local_gev\":" << kinetic
      << ",\"mass_gev\":" << mass
      << ",\"time_local_ns\":" << track.GetGlobalTime()/ns
      << ",\"boundary_normal_local\":[" << nx << ',' << ny << ',' << nz << ']'
      << ",\"creator_process\":" << q(creator ? creator->GetProcessName() : "primary")
      << ",\"geant4_statistical_weight\":" << track.GetWeight() << "}\n";
  }

  void end(int event_id) {
    const double raw_balance = ledger_.initial_total_gev - ledger_.deposited_gev - ledger_.escaped_total_gev;
    const double inferred_medium_rest_exchange = strict_vacuum_ledger_ ? 0.0 : raw_balance;
    const double residual = raw_balance - inferred_medium_rest_exchange;
    const double normalized = residual / std::max(1.0, std::abs(ledger_.initial_total_gev));
    const double normalized_raw = raw_balance / std::max(1.0, std::abs(ledger_.initial_total_gev));
    max_abs_normalized_residual_ = std::max(max_abs_normalized_residual_, std::abs(normalized));
    max_abs_normalized_raw_balance_ = std::max(max_abs_normalized_raw_balance_, std::abs(normalized_raw));
    total_deposited_gev_ += ledger_.deposited_gev;
    total_escaped_gev_ += ledger_.escaped_total_gev;
    total_escaped_particles_ += ledger_.escaped_particles;
    total_steps_ += ledger_.steps;
    event_file_ << std::setprecision(17)
      << "{\"geant4_event_id\":" << event_id
      << ",\"hepmc_event_number\":" << events_.at(static_cast<std::size_t>(event_id)).hepmc_event_number
      << ",\"initial_total_gev\":" << ledger_.initial_total_gev
      << ",\"deposited_gev\":" << ledger_.deposited_gev
      << ",\"escaped_total_gev\":" << ledger_.escaped_total_gev
      << ",\"escaped_particles\":" << ledger_.escaped_particles
      << ",\"steps\":" << ledger_.steps
      << ",\"signed_step_balance_gev\":" << ledger_.signed_step_balance_gev
      << ",\"absolute_step_balance_gev\":" << ledger_.absolute_step_balance_gev
      << ",\"raw_energy_balance_gev\":" << raw_balance
      << ",\"inferred_medium_rest_mass_and_binding_exchange_gev\":" << inferred_medium_rest_exchange
      << ",\"unexplained_residual_gev\":" << residual
      << ",\"normalized_unexplained_residual\":" << normalized << "}\n";
  }

  double total_deposited_gev() const { return total_deposited_gev_; }
  double total_escaped_gev() const { return total_escaped_gev_; }
  std::size_t total_escaped_particles() const { return total_escaped_particles_; }
  std::size_t total_steps() const { return total_steps_; }
  std::size_t recorded_steps() const { return recorded_steps_; }
  bool steps_truncated() const { return steps_truncated_; }
  double max_residual() const { return max_abs_normalized_residual_; }
  double max_raw_balance() const { return max_abs_normalized_raw_balance_; }

 private:
  const std::vector<InputEvent> &events_;
  double half_size_mm_ = 0.0;
  bool strict_vacuum_ledger_ = true;
  std::size_t max_recorded_steps_ = 0;
  std::ofstream event_file_;
  std::ofstream escaped_file_;
  std::ofstream step_file_;
  EventLedger ledger_;
  double total_deposited_gev_ = 0.0;
  double total_escaped_gev_ = 0.0;
  std::size_t total_escaped_particles_ = 0;
  std::size_t total_steps_ = 0;
  std::size_t recorded_steps_ = 0;
  bool steps_truncated_ = false;
  double max_abs_normalized_residual_ = 0.0;
  double max_abs_normalized_raw_balance_ = 0.0;
};

class EventAction final : public G4UserEventAction {
 public:
  explicit EventAction(OutputState &output) : output_(output) {}
  void BeginOfEventAction(const G4Event *event) override { output_.begin(event->GetEventID()); }
  void EndOfEventAction(const G4Event *event) override { output_.end(event->GetEventID()); }
 private:
  OutputState &output_;
};

class SteppingAction final : public G4UserSteppingAction {
 public:
  explicit SteppingAction(OutputState &output) : output_(output) {}
  void UserSteppingAction(const G4Step *step) override {
    output_.step();
    output_.deposit(step->GetTotalEnergyDeposit());
    double secondary_total = 0.0;
    for (const auto *secondary : *step->GetSecondaryInCurrentStep()) secondary_total += secondary->GetTotalEnergy();
    const double local_balance = (step->GetPreStepPoint()->GetTotalEnergy() - step->GetPostStepPoint()->GetTotalEnergy()
                                  - secondary_total - step->GetTotalEnergyDeposit()) / GeV;
    output_.step_balance(local_balance);
    output_.record_step(G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID(), *step);
    const auto *pre_volume = step->GetPreStepPoint()->GetPhysicalVolume();
    const auto *post = step->GetPostStepPoint();
    if (pre_volume && pre_volume->GetName() == "Patch" && post->GetStepStatus() == fGeomBoundary) {
      output_.escape(G4RunManager::GetRunManager()->GetCurrentEvent()->GetEventID(), *step->GetTrack(), *post);
    }
  }
 private:
  OutputState &output_;
};

void write_summary(const Options &o, const ImportAudit &audit, const OutputState &output) {
  std::ofstream f(o.output_dir / "geant4_backend_summary.json");
  f << std::setprecision(17) << "{\n"
    << "  \"status\": \"ok\",\n"
    << "  \"geant4_version\": " << q(G4Version) << ",\n"
    << "  \"physics_list\": " << q(o.physics_list) << ",\n"
    << "  \"mode\": " << q(o.mode) << ",\n"
    << "  \"events_transported\": " << audit.events.size() << ",\n"
    << "  \"final_primaries\": " << audit.final_particles << ",\n"
    << "  \"total_steps\": " << output.total_steps() << ",\n"
    << "  \"recorded_steps\": " << output.recorded_steps() << ",\n"
    << "  \"steps_truncated\": " << (output.steps_truncated() ? "true" : "false") << ",\n"
    << "  \"escaped_particles\": " << output.total_escaped_particles() << ",\n"
    << "  \"deposited_gev\": " << output.total_deposited_gev() << ",\n"
    << "  \"escaped_total_gev\": " << output.total_escaped_gev() << ",\n"
    << "  \"max_abs_normalized_raw_energy_balance\": " << output.max_raw_balance() << ",\n"
    << "  \"max_abs_normalized_unexplained_residual\": " << output.max_residual() << "\n}\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Options options = parse_options(argc, argv);
    fs::create_directories(options.output_dir);
    auto *run_manager = G4RunManagerFactory::CreateRunManager(G4RunManagerType::Serial);
    run_manager->SetUserInitialization(new DetectorConstruction(options));
    auto *physics = new HadrosPhysicsList(options.production_cut_mm * mm);
    run_manager->SetUserInitialization(physics);
    // Particle definitions are required for the domain audit. Geant4 requires the
    // physics list to be owned by a run manager before the particle table is queried.
    physics->ConstructParticle();
    ImportAudit audit = read_input(options);
    const bool domain_failure = audit.unsupported_energy != 0 || audit.unsupported_species != 0;
    const bool structural_failure = std::any_of(audit.violations.begin(), audit.violations.end(), [](const std::string &v) {
      return v.rfind("unsupported_", 0) != 0;
    });
    write_import_report(options, audit, domain_failure ? "unsupported_domain" : (structural_failure ? "invalid_input" : "ok"));
    if (domain_failure) {
      std::cerr << "H3-W11 domain guard refused " << audit.unsupported_energy << " particles above "
                << options.max_energy_gev << " GeV and " << audit.unsupported_species << " unsupported species\n";
      return 3;
    }
    if (structural_failure) {
      std::cerr << "H3-W11 import audit found structural violations\n";
      return 4;
    }
    if (options.mode == "import_check") {
      delete run_manager;
      return 0;
    }

    G4Random::setTheSeed(static_cast<long>(options.seed));
    OutputState output(options.output_dir, audit.events, options.half_size_mm, options.mode == "vacuum_smoke",
                       options.max_recorded_steps);
    run_manager->SetUserAction(new PrimaryGenerator(audit.events));
    run_manager->SetUserAction(new EventAction(output));
    run_manager->SetUserAction(new SteppingAction(output));
    run_manager->Initialize();
    run_manager->BeamOn(static_cast<G4int>(audit.events.size()));
    write_summary(options, audit, output);
    delete run_manager;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "hadros3_geant4_transport: " << e.what() << '\n';
    return 2;
  }
}
