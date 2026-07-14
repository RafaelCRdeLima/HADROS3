#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"
#include "Pythia8Plugins/PowhegHooks.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

struct Config {
  fs::path lhe;
  fs::path output_dir;
  std::string request_id;
  std::string mode = "real_smoke";
  int seed = 48001;
  int max_events = 2;
  bool isr = false;
  bool fsr = true;
  bool hadronization = true;
  bool decays = true;
  bool mpi = false;
  bool write_hepmc = true;
  double target_lepton_energy = 0.0;
  double target_mass = 0.0;
};

static std::string quote(const std::string& value) {
  std::string out = "\"";
  for (char ch : value) {
    if (ch == '\\' || ch == '"') out += '\\';
    out += ch;
  }
  return out + "\"";
}

static bool parse_bool(const std::string& value) {
  if (value == "1" || value == "true" || value == "on") return true;
  if (value == "0" || value == "false" || value == "off") return false;
  throw std::runtime_error("invalid boolean: " + value);
}

static Config parse_args(int argc, char** argv) {
  Config cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") {
      std::cout << "hadros3_event_generator --lhe FILE --output-dir DIR --request-id ID "
                   "[--mode parton_check|real_smoke|real_free] [--seed N] [--max-events N]\n";
      std::exit(0);
    }
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
    const std::string value = argv[++i];
    if (key == "--lhe") cfg.lhe = value;
    else if (key == "--output-dir") cfg.output_dir = value;
    else if (key == "--request-id") cfg.request_id = value;
    else if (key == "--mode") cfg.mode = value;
    else if (key == "--seed") cfg.seed = std::stoi(value);
    else if (key == "--max-events") cfg.max_events = std::stoi(value);
    else if (key == "--isr") cfg.isr = parse_bool(value);
    else if (key == "--fsr") cfg.fsr = parse_bool(value);
    else if (key == "--hadronization") cfg.hadronization = parse_bool(value);
    else if (key == "--decays") cfg.decays = parse_bool(value);
    else if (key == "--mpi") cfg.mpi = parse_bool(value);
    else if (key == "--write-hepmc") cfg.write_hepmc = parse_bool(value);
    else if (key == "--target-lepton-energy") cfg.target_lepton_energy = std::stod(value);
    else if (key == "--target-mass") cfg.target_mass = std::stod(value);
    else throw std::runtime_error("unknown argument: " + key);
  }
  if (cfg.lhe.empty() || cfg.output_dir.empty() || cfg.request_id.empty())
    throw std::runtime_error("--lhe, --output-dir and --request-id are required");
  if (cfg.max_events <= 0 || cfg.seed <= 0 || cfg.seed > 900000000)
    throw std::runtime_error("seed/max-events outside supported range");
  if (cfg.mode != "parton_check" && cfg.mode != "real_smoke" && cfg.mode != "real_free")
    throw std::runtime_error("unsupported mode: " + cfg.mode);
  return cfg;
}

int main(int argc, char** argv) {
  try {
    const Config cfg = parse_args(argc, argv);
    if (!fs::is_regular_file(cfg.lhe) || fs::file_size(cfg.lhe) == 0)
      throw std::runtime_error("LHE input is absent or empty: " + cfg.lhe.string());
    fs::create_directories(cfg.output_dir);
    std::ofstream event_out(cfg.output_dir / "events_summary.jsonl");
    std::ofstream particle_out(cfg.output_dir / "final_particles.jsonl");
    if (!event_out || !particle_out) throw std::runtime_error("cannot create JSONL outputs");

    Pythia8::Pythia pythia;
    // POWHEG's fixed-target EBMUP(2) is the target rest mass.  PYTHIA's
    // default proton mass is rounded to 0.938270 GeV, whereas HADROS3/POWHEG
    // uses 0.938272 GeV.  Without aligning them PYTHIA assigns the nominally
    // stationary target a spurious 1.94 MeV/c momentum.
    if (cfg.target_mass > 0.0) {
      std::ostringstream setting;
      setting << std::setprecision(17) << "2212:m0 = " << cfg.target_mass;
      pythia.readString(setting.str());
    }
    pythia.readString("Beams:frameType = 4");
    pythia.readString("Beams:LHEF = " + cfg.lhe.string());
    // The incoming neutrino is elementary and carries the complete beam
    // momentum. Leaving the generic lepton PDF enabled silently removes a
    // finite energy fraction without a physical neutrino beam remnant.
    pythia.readString("PDF:lepton = off");
    pythia.readString("Random:setSeed = on");
    pythia.readString("Random:seed = " + std::to_string(cfg.seed));
    pythia.readString("Print:quiet = on");
    pythia.readString("Next:numberShowInfo = 0");
    pythia.readString("Next:numberShowProcess = 0");
    pythia.readString("Next:numberShowEvent = 0");

    const bool parton_check = cfg.mode == "parton_check";
    std::shared_ptr<Pythia8::PowhegHooks> powheg_hooks;
    if (parton_check) {
      pythia.readString("PartonLevel:ISR = off");
      pythia.readString("PartonLevel:FSR = off");
      pythia.readString("PartonLevel:MPI = off");
      pythia.readString("PartonLevel:Remnants = off");
      pythia.readString("HadronLevel:all = off");
    } else {
      pythia.readString(std::string("PartonLevel:ISR = ") + (cfg.isr ? "on" : "off"));
      pythia.readString(std::string("PartonLevel:FSR = ") + (cfg.fsr ? "on" : "off"));
      pythia.readString(std::string("PartonLevel:MPI = ") + (cfg.mpi ? "on" : "off"));
      pythia.readString(std::string("HadronLevel:Hadronize = ") + (cfg.hadronization ? "on" : "off"));
      pythia.readString(std::string("HadronLevel:Decay = ") + (cfg.decays ? "on" : "off"));
      pythia.readString("POWHEG:veto = 1");
      pythia.readString("POWHEG:pThard = 0");
      pythia.readString("POWHEG:pTemt = 0");
      pythia.readString("POWHEG:emitted = 0");
      pythia.readString("POWHEG:pTdef = 1");
      pythia.readString("POWHEG:MPIveto = 0");
      powheg_hooks = std::make_shared<Pythia8::PowhegHooks>();
      pythia.setUserHooksPtr(powheg_hooks);
    }
    if (!pythia.init()) throw std::runtime_error("PYTHIA failed to initialize the LHE input");

    std::unique_ptr<Pythia8::Pythia8ToHepMC> hepmc;
    if (cfg.write_hepmc)
      hepmc = std::make_unique<Pythia8::Pythia8ToHepMC>((cfg.output_dir / "events.hepmc3").string());

    int generated = 0;
    int failures = 0;
    double max_residual = 0.0;
    double max_onshell = 0.0;
    while (generated < cfg.max_events) {
      if (!pythia.next()) {
        if (pythia.info.atEndOfFile()) break;
        if (++failures > 10) throw std::runtime_error("too many PYTHIA event failures");
        continue;
      }
      if (pythia.event.size() < 3) throw std::runtime_error("PYTHIA returned an empty event record");
      int hard_lepton = -1;
      for (int i = 0; i < pythia.event.size(); ++i) {
        if (pythia.event[i].status() == -21 && std::abs(pythia.event[i].id()) >= 11 &&
            std::abs(pythia.event[i].id()) <= 16 && pythia.event[i].pz() > 0.0) {
          hard_lepton = i;
          break;
        }
      }
      if (cfg.target_lepton_energy > 0.0 && hard_lepton >= 0) {
        const double rapidity = std::log(pythia.event[hard_lepton].e() / cfg.target_lepton_energy);
        const double ch = std::cosh(rapidity);
        const double sh = std::sinh(rapidity);
        for (int i = 0; i < pythia.event.size(); ++i) {
          auto p = pythia.event[i].p();
          const double pz = ch * p.pz() - sh * p.e();
          const double energy = ch * p.e() - sh * p.pz();
          pythia.event[i].p(Pythia8::Vec4(p.px(), p.py(), pz, energy));
        }
      }
      if (generated == 0 && std::getenv("HADROS3_DEBUG_EVENT")) pythia.event.list();
      ++generated;
      if (hepmc && !hepmc->writeNextEvent(pythia)) throw std::runtime_error("HepMC3 conversion failed");

      Pythia8::Vec4 initial;
      double target_beam_pz = 0.0;
      double target_beam_energy = 0.0;
      for (int i = 0; i < pythia.event.size(); ++i) {
        if (pythia.event[i].status() != -12) continue;
        initial += pythia.event[i].p();
        if (std::abs(pythia.event[i].id()) == 2212) {
          target_beam_pz = pythia.event[i].pz();
          target_beam_energy = pythia.event[i].e();
        }
      }
      if (initial.e() <= 0.0) initial = pythia.event[1].p() + pythia.event[2].p();
      Pythia8::Vec4 final_sum;
      int final_count = 0;
      int final_partons = 0;
      int final_hadrons = 0;
      int final_charge3 = 0;
      double max_shower_scale = 0.0;
      for (int i = 0; i < pythia.event.size(); ++i) {
        const auto& particle = pythia.event[i];
        if (std::abs(particle.status()) == 51 || std::abs(particle.status()) == 52)
          max_shower_scale = std::max(max_shower_scale, particle.scale());
        if (!particle.isFinal()) continue;
        ++final_count;
        if (particle.isParton()) ++final_partons;
        if (particle.isHadron()) ++final_hadrons;
        final_charge3 += particle.chargeType();
        final_sum += particle.p();
        const double shell = std::abs(particle.e() * particle.e() - particle.pAbs2() - particle.m() * particle.m()) /
                             std::max(particle.e() * particle.e(), 1.0);
        max_onshell = std::max(max_onshell, shell);
        particle_out << std::setprecision(17)
          << "{\"event_generation_event_id\":" << quote(cfg.request_id + ":" + std::to_string(generated))
          << ",\"particle_index\":" << i << ",\"pdg_id\":" << particle.id()
          << ",\"particle_name\":" << quote(pythia.particleData.name(particle.id()))
          << ",\"status\":" << particle.status() << ",\"mother1\":" << particle.mother1()
          << ",\"mother2\":" << particle.mother2() << ",\"daughter1\":" << particle.daughter1()
          << ",\"daughter2\":" << particle.daughter2() << ",\"px_gev\":" << particle.px()
          << ",\"py_gev\":" << particle.py() << ",\"pz_gev\":" << particle.pz()
          << ",\"energy_gev\":" << particle.e() << ",\"mass_gev\":" << particle.m()
          << ",\"charge_e3\":" << particle.chargeType() << ",\"is_hadron\":" << (particle.isHadron() ? "true" : "false")
          << ",\"is_parton\":" << (particle.isParton() ? "true" : "false")
          << ",\"generator_frame\":\"local_matter_tetrad\",\"momentum_unit\":\"GeV\",\"length_unit\":\"mm\"}\n";
      }
      const auto delta = final_sum - initial;
      const double scale = std::max(std::abs(initial.e()), 1.0);
      const double residual = std::max({std::abs(delta.e()), std::abs(delta.px()), std::abs(delta.py()), std::abs(delta.pz())}) / scale;
      max_residual = std::max(max_residual, residual);
      const int initial_charge3 = pythia.event[1].chargeType() + pythia.event[2].chargeType();
      event_out << std::setprecision(17)
        << "{\"powheg_request_id\":" << quote(cfg.request_id)
        << ",\"lhe_event_index\":" << generated
        << ",\"event_generation_event_id\":" << quote(cfg.request_id + ":" + std::to_string(generated))
        << ",\"seed\":" << cfg.seed << ",\"xwgtup\":" << pythia.info.weight()
        << ",\"scalup_gev\":" << pythia.info.scalup() << ",\"n_final_particles\":" << final_count
        << ",\"n_final_partons\":" << final_partons << ",\"n_final_hadrons\":" << final_hadrons
        << ",\"initial_charge_e3\":" << initial_charge3 << ",\"final_charge_e3\":" << final_charge3
        << ",\"charge_conservation_pass\":" << (initial_charge3 == final_charge3 ? "true" : "false")
        << ",\"four_momentum_residual_relative\":" << residual
        << ",\"initial_energy_gev\":" << initial.e() << ",\"initial_px_gev\":" << initial.px()
        << ",\"initial_py_gev\":" << initial.py() << ",\"initial_pz_gev\":" << initial.pz()
        << ",\"target_beam_energy_gev\":" << target_beam_energy << ",\"target_beam_pz_gev\":" << target_beam_pz
        << ",\"target_rest_frame_residual\":" << std::abs(target_beam_pz) / std::max(std::abs(target_beam_energy), 1.0)
        << ",\"final_energy_gev\":" << final_sum.e() << ",\"final_px_gev\":" << final_sum.px()
        << ",\"final_py_gev\":" << final_sum.py() << ",\"final_pz_gev\":" << final_sum.pz()
        << ",\"shower_invoked\":" << (!parton_check && (cfg.isr || cfg.fsr) ? "true" : "false")
        << ",\"hadronization_invoked\":" << (!parton_check && cfg.hadronization ? "true" : "false")
        << ",\"decays_invoked\":" << (!parton_check && cfg.decays ? "true" : "false")
        << ",\"matching_policy\":" << quote(parton_check ? "off" : "powheg_vetoed_scalup")
        << ",\"max_shower_scale_gev\":" << max_shower_scale
        << ",\"matching_scale_pass\":" << (parton_check || max_shower_scale <= pythia.info.scalup() * (1.0 + 1e-10) ? "true" : "false")
        << ",\"generator_frame\":\"local_matter_tetrad\",\"status\":\"ok\"}\n";
    }
    if (generated == 0) throw std::runtime_error("no LHE events were generated");

    std::ofstream summary(cfg.output_dir / "backend_summary.json");
    summary << std::setprecision(17)
      << "{\n  \"status\": \"ok\",\n  \"backend\": \"pythia8\",\n"
      << "  \"pythia_version\": \"" << PYTHIA_VERSION << "\",\n"
      << "  \"hepmc3_version\": \"3.03.01\",\n"
      << "  \"n_events_generated\": " << generated << ",\n"
      << "  \"n_event_failures\": " << failures << ",\n"
      << "  \"four_momentum_residual_relative_max\": " << max_residual << ",\n"
      << "  \"onshell_residual_relative_max\": " << max_onshell << "\n}\n";
    std::cout << "generated " << generated << " events for " << cfg.request_id << "\n";
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "hadros3_event_generator: " << exc.what() << "\n";
    return 1;
  }
}
