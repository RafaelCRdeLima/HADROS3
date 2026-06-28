#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Config {
  std::string backend = "local_powheg";
  std::string process = "nudis";
  int events_per_candidate = 1;
  int random_seed = 12345;
  std::string seed_mode = "base_plus_candidate_rank";
  std::string run_mode = "dry_run";
};

struct Candidate {
  std::size_t input_index = 0;
  int candidate_rank = 0;
  std::string interaction_id;
  std::string event_id;
  std::string source_sample_id;
  double energy_gev = 0.0;
  double physics_weight = 0.0;
  double observer_weight = 0.0;
  double final_score = 0.0;
  std::string selection_policy;
  std::string selection_reason;
};

struct Request {
  std::string request_id;
  Candidate candidate;
  fs::path card_path;
  int seed = 0;
};

static std::string read_text(const fs::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static std::string quote(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out.push_back('\\');
    if (c == '\n') {
      out += "\\n";
    } else {
      out.push_back(c);
    }
  }
  out.push_back('"');
  return out;
}

static std::string json_string(const std::string& text, const std::string& key, const std::string& fallback = "") {
  const std::string needle = "\"" + key + "\"";
  auto pos = text.find(needle);
  if (pos == std::string::npos) return fallback;
  pos = text.find(':', pos);
  if (pos == std::string::npos) return fallback;
  pos = text.find('"', pos);
  if (pos == std::string::npos) return fallback;
  std::string out;
  bool escape = false;
  for (std::size_t i = pos + 1; i < text.size(); ++i) {
    const char c = text[i];
    if (escape) {
      out.push_back(c);
      escape = false;
    } else if (c == '\\') {
      escape = true;
    } else if (c == '"') {
      return out;
    } else {
      out.push_back(c);
    }
  }
  return fallback;
}

static double json_number(const std::string& text, const std::string& key, double fallback) {
  const std::string needle = "\"" + key + "\"";
  auto pos = text.find(needle);
  if (pos == std::string::npos) return fallback;
  pos = text.find(':', pos);
  if (pos == std::string::npos) return fallback;
  ++pos;
  while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) ++pos;
  const char* begin = text.c_str() + pos;
  char* end = nullptr;
  const double value = std::strtod(begin, &end);
  if (end == begin || !std::isfinite(value)) return fallback;
  return value;
}

static std::string json_scalar_string(const std::string& text, const std::string& key, const std::string& fallback = "") {
  const std::string direct = json_string(text, key, "");
  if (!direct.empty()) return direct;
  const std::string needle = "\"" + key + "\"";
  auto pos = text.find(needle);
  if (pos == std::string::npos) return fallback;
  pos = text.find(':', pos);
  if (pos == std::string::npos) return fallback;
  ++pos;
  while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) ++pos;
  std::size_t end = pos;
  while (end < text.size() && text[end] != ',' && text[end] != '}') ++end;
  std::string value = text.substr(pos, end - pos);
  while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) value.pop_back();
  return value.empty() ? fallback : value;
}

static std::string section_text(const std::string& text, const std::string& section) {
  const std::string needle = "\"" + section + "\"";
  auto pos = text.find(needle);
  if (pos == std::string::npos) return text;
  pos = text.find('{', pos);
  if (pos == std::string::npos) return text;
  int depth = 0;
  for (std::size_t i = pos; i < text.size(); ++i) {
    if (text[i] == '{') ++depth;
    if (text[i] == '}') {
      --depth;
      if (depth == 0) return text.substr(pos, i - pos + 1);
    }
  }
  return text;
}

static Config load_config(const fs::path& path) {
  Config c;
  const std::string powheg = section_text(read_text(path), "powheg");
  c.backend = json_string(powheg, "powheg_backend", c.backend);
  c.process = json_string(powheg, "powheg_process", c.process);
  c.events_per_candidate = std::max(1, static_cast<int>(json_number(powheg, "events_per_candidate", c.events_per_candidate)));
  c.random_seed = static_cast<int>(json_number(powheg, "random_seed", c.random_seed));
  c.seed_mode = json_string(powheg, "powheg_seed_mode", c.seed_mode);
  c.run_mode = json_string(powheg, "run_mode", c.run_mode);
  if (c.backend != "local_powheg") throw std::runtime_error("powheg_backend must be local_powheg");
  if (c.process != "nudis") throw std::runtime_error("powheg_process must be nudis");
  if (c.run_mode != "dry_run" && c.run_mode != "real_smoke" && c.run_mode != "real_free") {
    throw std::runtime_error("POWHEG run_mode must be dry_run, real_smoke, or real_free");
  }
  if (c.run_mode == "real_smoke") {
    c.events_per_candidate = std::min(c.events_per_candidate, 2);
  }
  return c;
}

static Candidate candidate_from_line(const std::string& line, std::size_t index) {
  Candidate c;
  c.input_index = index;
  c.candidate_rank = static_cast<int>(json_number(line, "selection_rank", static_cast<double>(index + 1)));
  c.interaction_id = json_scalar_string(line, "interaction_id", "interaction-" + std::to_string(index + 1));
  c.event_id = json_scalar_string(line, "event_id", "event-" + std::to_string(index + 1));
  c.source_sample_id = json_scalar_string(line, "source_sample_id", "0");
  c.energy_gev = json_number(line, "interaction_E_nu_local_gev", 0.0);
  c.physics_weight = json_number(line, "physics_weight", 0.0);
  c.observer_weight = json_number(line, "observer_weight", 0.0);
  c.final_score = json_number(line, "final_observation_score", 0.0);
  c.selection_policy = json_scalar_string(line, "selection_policy", "");
  c.selection_reason = json_scalar_string(line, "selection_reason", "");
  return c;
}

static std::vector<Candidate> read_candidates(const fs::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("ObserverBridge selected candidates not found. Run Observer Bridge with downstream selection first: " + path.string());
  std::vector<Candidate> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    rows.push_back(candidate_from_line(line, rows.size()));
  }
  return rows;
}

static double qmax_for_energy(double energy_gev) {
  // Q_max = sqrt(2 * m_p * E_nu): kinematic limit for fixed-target DIS
  return std::min(std::sqrt(2.0 * 0.938272 * std::max(energy_gev, 0.0)), 1.0e5);
}

static std::string fortran_double(double value, int precision = 10) {
  std::ostringstream ss;
  ss << std::uppercase << std::scientific << std::setprecision(precision) << value;
  std::string s = ss.str();
  auto pos = s.find('E');
  if (pos != std::string::npos) s[pos] = 'D';
  return s;
}

static std::string powheg_card(const Candidate& c, const Config& cfg, int seed) {
  const double qmax = qmax_for_energy(c.energy_gev);
  std::ostringstream out;
  out << "! HADROS3 POWHEG DIS card.\n";
  if (cfg.run_mode == "real_smoke") {
    out << "! H3-W9b real smoke mode: pwhg_main executes locally for this card.\n";
  } else if (cfg.run_mode == "real_free") {
    out << "! H3-W9b real free mode: pwhg_main executes locally for this card.\n";
  } else {
    out << "! H3-W9a dry run: pwhg_main is NOT executed in this stage.\n";
  }
  out << "! interaction_id=" << c.interaction_id << " event_id=" << c.event_id << "\n";
  out << "! final_observation_score=" << std::setprecision(17) << c.final_score << "\n";
  out << "LOevents 1\n";
  out << "numevts " << cfg.events_per_candidate << "\n";
  out << "ih1 12\n";
  out << "ih2 1\n";
  out << "ebeam1 " << fortran_double(c.energy_gev) << "\n";
  out << "ebeam2 0.938272d0\n";
  out << "bornktmin 0d0\n";
  out << "bornsuppfact 0d0\n";
  out << "Qmin 10d0\n";
  out << "Qmax " << fortran_double(qmax) << "\n";
  out << "xmin 0d0\nxmax 1d0\nymin 0d0\nymax 1d0\n";
  out << "q2suppr 200d0\n";
  out << "lhans1 303400\nlhans2 303400\nalphas_from_pdf 1\n";
  out << "renscfact 1d0\nfacscfact 1d0\n";
  out << "use-old-grid 0\nuse-old-ubound 0\n";
  if (cfg.run_mode == "real_smoke") {
    out << "ncall1 100\nitmx1 1\nncall2 200\nitmx2 1\n";
  } else {
    out << "ncall1 1000\nitmx1 1\nncall2 2000\nitmx2 1\n";
  }
  out << "foldcsi 1\nfoldy 1\nfoldphi 1\nnubound 1000\n";
  out << "iupperfsr 1\nfastbtlbound 1\nstoremintupb 1\nubexcess_correct 1\nstoreinfo_rwgt 1\n";
  out << "hdamp 0\nbornzerodamp 1\nwithnegweights 1\nflg_jacsing 1\ntestplots 0\nxupbound 2d0\n";
  out << "iseed " << seed << "\n";
  out << "manyseeds 0\ndoublefsr 0\nrunningscales 1\nolddij 0\n";
  out << "channel_type 3\nvtype 2\n";
  out << "smartsig 1\nnores 1\nparallelstage 0\nxgriditeration 1\n";
  out << "py8QED 0\npy8MPI 1\npy8had 2\npy8shower 1\ncolltest 0\nsofttest 0\n";
  return out.str();
}

static void write_text_file(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path.string());
  out << text;
}

static std::string rel_to(const fs::path& path, const fs::path& base) {
  return fs::relative(path, base).generic_string();
}

static void write_requests(const fs::path& path, const std::vector<Request>& requests, const fs::path& run_output, const Config& cfg) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path.string());
  out << std::setprecision(17);
  for (const auto& r : requests) {
    const std::string status = cfg.run_mode == "real_smoke" ? "real_smoke_ready" : (cfg.run_mode == "real_free" ? "real_free_ready" : "dry_run_ready");
    out << "{"
        << "\"powheg_request_id\":" << quote(r.request_id) << ","
        << "\"candidate_rank\":" << r.candidate.candidate_rank << ","
        << "\"interaction_id\":" << quote(r.candidate.interaction_id) << ","
        << "\"event_id\":" << quote(r.candidate.event_id) << ","
        << "\"source_sample_id\":" << quote(r.candidate.source_sample_id) << ","
        << "\"interaction_E_nu_local_gev\":" << r.candidate.energy_gev << ","
        << "\"physics_weight\":" << r.candidate.physics_weight << ","
        << "\"observer_weight\":" << r.candidate.observer_weight << ","
        << "\"final_observation_score\":" << r.candidate.final_score << ","
        << "\"powheg_candidate_source\":\"ObserverBridge/observer_bridge_selected_candidates.jsonl\","
        << "\"powheg_selection_performed_by\":\"ObserverBridge\","
        << "\"powheg_selection_policy\":" << quote(r.candidate.selection_policy) << ","
        << "\"selection_reason\":" << quote(r.candidate.selection_reason) << ","
        << "\"powheg_input_path\":" << quote(rel_to(r.card_path, run_output)) << ","
        << "\"powheg_seed\":" << r.seed << ","
        << "\"powheg_status\":" << quote(status) << ","
        << "\"powheg_invoked\":false"
        << "}\n";
  }
}

static void write_summary_csv(const fs::path& path, const std::vector<Request>& requests, const Config& cfg) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path.string());
  out << "powheg_request_id,candidate_rank,interaction_id,event_id,interaction_E_nu_local_gev,final_observation_score,powheg_seed,powheg_status\n";
  out << std::setprecision(17);
  for (const auto& r : requests) {
    const std::string status = cfg.run_mode == "real_smoke" ? "real_smoke_ready" : (cfg.run_mode == "real_free" ? "real_free_ready" : "dry_run_ready");
    out << r.request_id << "," << r.candidate.candidate_rank << "," << r.candidate.interaction_id << ","
        << r.candidate.event_id << "," << r.candidate.energy_gev << "," << r.candidate.final_score << ","
        << r.seed << "," << status << "\n";
  }
}

static void write_summary_json(const fs::path& path, const Config& cfg, int input_count, const std::vector<Request>& requests, const fs::path& output_dir) {
  double min_score = requests.empty() ? 0.0 : requests.front().candidate.final_score;
  double max_score = requests.empty() ? 0.0 : requests.front().candidate.final_score;
  double min_energy = requests.empty() ? 0.0 : requests.front().candidate.energy_gev;
  double max_energy = requests.empty() ? 0.0 : requests.front().candidate.energy_gev;
  for (const auto& r : requests) {
    min_score = std::min(min_score, r.candidate.final_score);
    max_score = std::max(max_score, r.candidate.final_score);
    min_energy = std::min(min_energy, r.candidate.energy_gev);
    max_energy = std::max(max_energy, r.candidate.energy_gev);
  }
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path.string());
  const bool real_smoke = cfg.run_mode == "real_smoke";
  const bool real_free = cfg.run_mode == "real_free";
  const std::string stage_name = real_smoke ? "H3-W9b POWHEG Real Run Smoke Mode" : (real_free ? "H3-W9b POWHEG Real Free Mode" : "H3-W9a POWHEG Integration Dry Run");
  const std::string selection_policy = requests.empty() ? "" : requests.front().candidate.selection_policy;
  out << std::setprecision(17);
  out << "{\n"
      << "  \"stage_name\": " << quote(stage_name) << ",\n"
      << "  \"powheg_backend\": " << quote(cfg.backend) << ",\n"
      << "  \"powheg_process\": " << quote(cfg.process) << ",\n"
      << "  \"powheg_run_mode\": " << quote(cfg.run_mode) << ",\n"
      << "  \"powheg_dry_run_invoked\": " << ((real_smoke || real_free) ? "false" : "true") << ",\n"
      << "  \"powheg_real_smoke_invoked\": " << (real_smoke ? "true" : "false") << ",\n"
      << "  \"powheg_real_free_invoked\": " << (real_free ? "true" : "false") << ",\n"
      << "  \"powheg_invoked\": false,\n"
      << "  \"pwhg_main_executed\": false,\n"
      << "  \"powheg_cards_generated\": " << requests.size() << ",\n"
      << "  \"powheg_lhe_generated\": false,\n"
      << "  \"powheg_runtime_self_contained\": true,\n"
      << "  \"backend_language\": \"C++17\",\n"
      << "  \"backend_executable\": \"bin/hadros3_powheg_driver\",\n"
      << "  \"powheg_candidate_source\": \"ObserverBridge/observer_bridge_selected_candidates.jsonl\",\n"
      << "  \"powheg_n_selected_candidates_input\": " << input_count << ",\n"
      << "  \"powheg_selection_performed_by\": \"ObserverBridge\",\n"
      << "  \"powheg_selection_policy\": " << quote(selection_policy) << ",\n"
      << "  \"powheg_jobs_prepared\": " << requests.size() << ",\n"
      << "  \"n_candidates_input\": " << input_count << ",\n"
      << "  \"n_powheg_jobs\": " << requests.size() << ",\n"
      << "  \"n_lhe_events\": 0,\n"
      << "  \"lhe_found\": false,\n"
      << "  \"events_per_candidate\": " << cfg.events_per_candidate << ",\n"
      << "  \"random_seed\": " << cfg.random_seed << ",\n"
      << "  \"powheg_seed_mode\": " << quote(cfg.seed_mode) << ",\n"
      << "  \"top_powheg_request_id\": " << quote(requests.empty() ? "" : requests.front().request_id) << ",\n"
      << "  \"min_score\": " << min_score << ",\n"
      << "  \"max_score\": " << max_score << ",\n"
      << "  \"min_energy_gev\": " << min_energy << ",\n"
      << "  \"max_energy_gev\": " << max_energy << ",\n"
      << "  \"pythia_invoked\": false,\n"
      << "  \"geant4_invoked\": false,\n"
      << "  \"photon_transport_invoked\": false,\n"
      << "  \"expensive_event_generation_invoked\": false,\n"
      << "  \"products\": {\n"
      << "    \"powheg_event_requests\": " << quote((output_dir / "powheg_event_requests.jsonl").string()) << ",\n"
      << "    \"powheg_summary_json\": " << quote((output_dir / "powheg_summary.json").string()) << ",\n"
      << "    \"powheg_summary_csv\": " << quote((output_dir / "powheg_summary.csv").string()) << ",\n"
      << "    \"powheg_report\": " << quote((output_dir / "powheg_report.json").string()) << "\n"
      << "  }\n"
      << "}\n";
}

static void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0 << " --run-output output/<run-name>\n";
}

int main(int argc, char** argv) {
  try {
    fs::path run_output;
    for (int i = 1; i < argc; ++i) {
      const std::string key = argv[i];
      if (key == "--run-output" && i + 1 < argc) {
        run_output = argv[++i];
      } else if (key == "--help" || key == "-h") {
        usage(argv[0]);
        return 0;
      } else {
        throw std::runtime_error("unknown argument: " + key);
      }
    }
    if (run_output.empty()) throw std::runtime_error("--run-output is required");
    const fs::path config_path = run_output / "RunMetadata" / "hadros3_config.json";
    const fs::path input_path = run_output / "ObserverBridge" / "observer_bridge_selected_candidates.jsonl";
    const fs::path output_dir = run_output / "POWHEG";
    fs::create_directories(output_dir);
    fs::create_directories(output_dir / "powheg_input_cards");

    const Config cfg = load_config(config_path);
    const std::vector<Candidate> candidates = read_candidates(input_path);
    std::vector<Candidate> selected = candidates;
    if (cfg.run_mode == "real_smoke" && selected.size() > 1) {
      selected.resize(1);
    }
    std::vector<Request> requests;
    for (std::size_t i = 0; i < selected.size(); ++i) {
      std::ostringstream id;
      id << "H3PWHG-" << std::setw(6) << std::setfill('0') << (i + 1);
      Request req;
      req.request_id = id.str();
      req.candidate = selected[i];
      req.seed = cfg.random_seed + selected[i].candidate_rank;
      req.card_path = output_dir / "powheg_input_cards" / req.request_id / "powheg.input";
      write_text_file(req.card_path, powheg_card(req.candidate, cfg, req.seed));
      requests.push_back(req);
    }
    write_requests(output_dir / "powheg_event_requests.jsonl", requests, run_output, cfg);
    write_summary_csv(output_dir / "powheg_summary.csv", requests, cfg);
    write_summary_json(output_dir / "powheg_summary.json", cfg, static_cast<int>(candidates.size()), requests, output_dir);
    fs::copy_file(output_dir / "powheg_summary.json", output_dir / "powheg_report.json", fs::copy_options::overwrite_existing);
    if (cfg.run_mode == "real_smoke") {
      std::cout << "H3-W9b POWHEG real-smoke prepared " << requests.size()
                << " job; local pwhg_main execution is delegated to the Python wrapper\n";
    } else if (cfg.run_mode == "real_free") {
      std::cout << "H3-W9b POWHEG real-free prepared " << requests.size()
                << " jobs; local pwhg_main execution is delegated to the Python wrapper\n";
    } else {
      std::cout << "H3-W9a POWHEG dry run prepared " << requests.size() << " jobs; pwhg_main NOT executed\n";
    }
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "hadros3_powheg_driver failed: " << exc.what() << "\n";
    return 1;
  }
}
