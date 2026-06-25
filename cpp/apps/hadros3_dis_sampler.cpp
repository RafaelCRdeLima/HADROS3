#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

constexpr double G_CGS = 6.67430e-8;
constexpr double C_CGS = 2.99792458e10;
constexpr double MSUN_G = 1.98847e33;
constexpr double M_BARYON_G = 1.67262192369e-24;

struct Config {
  std::string dis_model = "GBW";
  std::string medium_model = "analytic_torus";
  std::string medium_velocity_model = "zamo_fallback";
  std::string interaction_sampling_mode = "optical_depth_inverse_cdf";
  double density_floor_g_cm3 = 0.0;
  int max_interactions = 1000000;
  unsigned long long random_seed = 24680;
  double mass_msun = 3.0;
  double spin_a = 0.8;
  double torus_r_inner = 6.0;
  double torus_r_outer = 18.0;
  double torus_r_peak = 10.0;
  double torus_half_angle_deg = 18.0;
  double torus_density_norm = 1.0e10;
};

struct Segment {
  std::string event_id;
  int source_sample_id = 0;
  int segment_index = 0;
  double r0 = 0.0, th0 = 0.0, ph0 = 0.0;
  double r1 = 0.0, th1 = 0.0, ph1 = 0.0;
  double rm = 0.0, thm = 0.0, phm = 0.0;
  double pt = 0.0, pr = 0.0, pth = 0.0, pph = 0.0;
  double dl = 0.0;
};

struct Source {
  int source_sample_id = 0;
  double source_weight = 1.0;
  double direction_weight = 1.0;
};

struct SigmaTable {
  std::string path;
  std::vector<std::pair<double, double>> rows;
  double emin = 0.0;
  double emax = 0.0;
};

static std::string read_text(const fs::path &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path.string());
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static std::string q(const std::string &s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out.push_back('\\');
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

static std::string json_string(const std::string &text, const std::string &key, const std::string &fallback = "") {
  const std::string needle = "\"" + key + "\"";
  auto pos = text.find(needle);
  if (pos == std::string::npos) return fallback;
  pos = text.find(':', pos);
  if (pos == std::string::npos) return fallback;
  pos = text.find('"', pos);
  if (pos == std::string::npos) return fallback;
  auto end = text.find('"', pos + 1);
  if (end == std::string::npos) return fallback;
  return text.substr(pos + 1, end - pos - 1);
}

static double json_number(const std::string &text, const std::string &key, double fallback) {
  const std::string needle = "\"" + key + "\"";
  auto pos = text.find(needle);
  if (pos == std::string::npos) return fallback;
  pos = text.find(':', pos);
  if (pos == std::string::npos) return fallback;
  ++pos;
  while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) ++pos;
  const char *begin = text.c_str() + pos;
  char *end = nullptr;
  double value = std::strtod(begin, &end);
  if (end == begin) return fallback;
  return value;
}

static std::string section_text(const std::string &text, const std::string &section) {
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

static Config load_config(const fs::path &path) {
  Config c;
  const std::string text = read_text(path);
  const std::string dis = section_text(text, "dis_interaction_sampler");
  const std::string bh = section_text(text, "black_hole");
  const std::string torus = section_text(text, "analytic_torus");
  c.dis_model = json_string(dis, "dis_model", c.dis_model);
  c.medium_model = json_string(dis, "medium_model", c.medium_model);
  c.medium_velocity_model = json_string(dis, "medium_velocity_model", c.medium_velocity_model);
  c.interaction_sampling_mode = json_string(dis, "interaction_sampling_mode", c.interaction_sampling_mode);
  c.density_floor_g_cm3 = json_number(dis, "density_floor_g_cm3", c.density_floor_g_cm3);
  c.max_interactions = static_cast<int>(json_number(dis, "max_interactions", c.max_interactions));
  c.random_seed = static_cast<unsigned long long>(json_number(dis, "random_seed", c.random_seed));
  c.mass_msun = json_number(bh, "mass_msun", c.mass_msun);
  c.spin_a = std::max(-0.999, std::min(0.999, json_number(bh, "spin_a", c.spin_a)));
  c.torus_r_inner = json_number(torus, "r_inner_rg", c.torus_r_inner);
  c.torus_r_outer = json_number(torus, "r_outer_rg", c.torus_r_outer);
  c.torus_r_peak = json_number(torus, "r_peak_rg", c.torus_r_peak);
  c.torus_half_angle_deg = json_number(torus, "half_opening_angle_deg", c.torus_half_angle_deg);
  c.torus_density_norm = json_number(torus, "density_norm_g_cm3", c.torus_density_norm);
  return c;
}

static std::vector<Source> load_sources(const fs::path &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path.string());
  std::vector<Source> records;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    Source s;
    s.source_sample_id = static_cast<int>(json_number(line, "source_sample_id", 0));
    s.source_weight = json_number(line, "source_weight", 1.0);
    s.direction_weight = json_number(line, "direction_weight", 1.0);
    records.push_back(s);
  }
  return records;
}

static std::vector<std::string> load_path_ids(const fs::path &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path.string());
  std::vector<std::string> ids;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) ids.push_back(json_string(line, "event_id", ""));
  }
  return ids;
}

static std::vector<Segment> load_segments(const fs::path &path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read " + path.string());
  std::vector<Segment> records;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    Segment s;
    s.event_id = json_string(line, "event_id", "");
    s.source_sample_id = static_cast<int>(json_number(line, "source_sample_id", 0));
    s.segment_index = static_cast<int>(json_number(line, "segment_index", 0));
    s.r0 = json_number(line, "r_start_rg", 0.0);
    s.th0 = json_number(line, "theta_start_rad", 0.0);
    s.ph0 = json_number(line, "phi_start_rad", 0.0);
    s.r1 = json_number(line, "r_end_rg", 0.0);
    s.th1 = json_number(line, "theta_end_rad", 0.0);
    s.ph1 = json_number(line, "phi_end_rad", 0.0);
    s.rm = json_number(line, "r_mid_rg", 0.0);
    s.thm = json_number(line, "theta_mid_rad", 0.0);
    s.phm = json_number(line, "phi_mid_rad", 0.0);
    s.pt = json_number(line, "p_t_mid", 0.0);
    s.pr = json_number(line, "p_r_mid", 0.0);
    s.pth = json_number(line, "p_theta_mid", 0.0);
    s.pph = json_number(line, "p_phi_mid", 0.0);
    s.dl = json_number(line, "dl_segment_rg", 0.0);
    records.push_back(s);
  }
  std::sort(records.begin(), records.end(), [](const Segment &a, const Segment &b) {
    if (a.event_id != b.event_id) return a.event_id < b.event_id;
    return a.segment_index < b.segment_index;
  });
  return records;
}

static SigmaTable load_sigma_table(const std::string &model) {
  SigmaTable table;
  table.path = "data/sigma/sigma_nuN_CC_" + model + ".dat";
  std::ifstream in(table.path);
  if (!in) throw std::runtime_error("DIS sigma table not found: " + table.path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    double energy = 0.0, unused = 0.0, sigma = 0.0;
    ss >> energy >> unused >> sigma;
    if (energy > 0.0 && sigma > 0.0) table.rows.emplace_back(energy, sigma);
  }
  if (table.rows.size() < 2) throw std::runtime_error("invalid sigma table: " + table.path);
  table.emin = table.rows.front().first;
  table.emax = table.rows.back().first;
  return table;
}

static double sigma_cm2(const SigmaTable &table, double energy) {
  if (!(energy > 0.0) || energy < table.emin || energy > table.emax) throw std::runtime_error("energy outside sigma table");
  for (std::size_t i = 1; i < table.rows.size(); ++i) {
    const auto [e0, s0] = table.rows[i - 1];
    const auto [e1, s1] = table.rows[i];
    if (e0 <= energy && energy <= e1) {
      const double t = (std::log(energy) - std::log(e0)) / (std::log(e1) - std::log(e0));
      return std::exp(std::log(s0) + t * (std::log(s1) - std::log(s0)));
    }
  }
  return table.rows.back().second;
}

static double rg_to_cm(double mass_msun) { return G_CGS * mass_msun * MSUN_G / (C_CGS * C_CGS); }

static double density(double r, double theta, const Config &c) {
  if (r < c.torus_r_inner || r > c.torus_r_outer) return 0.0;
  const double half_angle = c.torus_half_angle_deg * M_PI / 180.0;
  const double theta_width = std::max(half_angle, 1.0e-6);
  const double radial_width = std::max(0.5 * (c.torus_r_outer - c.torus_r_inner), 1.0e-6);
  const double radial = std::exp(-0.5 * std::pow((r - c.torus_r_peak) / radial_width, 2));
  const double polar = std::exp(-0.5 * std::pow((theta - 0.5 * M_PI) / theta_width, 2));
  const double rho = c.torus_density_norm * radial * polar;
  return rho > 0.0 ? std::max(rho, c.density_floor_g_cm3) : 0.0;
}

struct InteractionPoint {
  double r = 0.0;
  double theta = 0.0;
  double phi = 0.0;
  double rho = 0.0;
  bool inside = false;
  int attempts = 0;
  std::string method = "rejection_with_midpoint_fallback";
};

static double local_energy(const Segment &s, const Config &c, bool &static_fallback) {
  const double a = c.spin_a;
  const double sinth = std::max(std::sin(s.thm), 1.0e-12);
  const double sigma = s.rm * s.rm + a * a * std::cos(s.thm) * std::cos(s.thm);
  const double delta = s.rm * s.rm - 2.0 * s.rm + a * a;
  const double big_a = (s.rm * s.rm + a * a) * (s.rm * s.rm + a * a) - a * a * delta * sinth * sinth;
  const double gtt = -(1.0 - 2.0 * s.rm / sigma);
  if (c.medium_velocity_model == "static" && gtt < 0.0) {
    static_fallback = false;
    return std::max(0.0, -(s.pt / std::sqrt(-gtt)));
  }
  const double lapse = std::sqrt(std::max(sigma * delta / big_a, 1.0e-30));
  const double omega = 2.0 * a * s.rm / big_a;
  static_fallback = c.medium_velocity_model == "static";
  return std::max(0.0, -(s.pt / lapse + s.pph * omega / lapse));
}

static double probability(double tau) { return std::max(0.0, std::min(1.0, -std::expm1(-std::max(0.0, tau)))); }

static double interp_angle(double a0, double a1, double u) {
  const double delta = std::atan2(std::sin(a1 - a0), std::cos(a1 - a0));
  return a0 + u * delta;
}

static InteractionPoint sample_interaction_point(const Segment &s, const Config &config, std::mt19937_64 &rng, std::uniform_real_distribution<double> &uni) {
  constexpr int max_attempts = 32;
  InteractionPoint best;
  for (int attempt = 1; attempt <= max_attempts; ++attempt) {
    const double u = uni(rng);
    const double rr = s.r0 + u * (s.r1 - s.r0);
    const double th = s.th0 + u * (s.th1 - s.th0);
    const double ph = interp_angle(s.ph0, s.ph1, u);
    const double rho = density(rr, th, config);
    if (rho > best.rho) best = {rr, th, ph, rho, rho > 0.0, attempt, "rejection_with_midpoint_fallback"};
    if (rho > 0.0) return {rr, th, ph, rho, true, attempt, "rejection_with_midpoint_fallback"};
  }
  const double midpoint_rho = density(s.rm, s.thm, config);
  if (midpoint_rho > 0.0) {
    return {s.rm, s.thm, s.phm, midpoint_rho, true, max_attempts, "rejection_with_midpoint_fallback_midpoint"};
  }
  best.attempts = max_attempts;
  best.method = "rejection_with_highest_density_fallback";
  best.inside = best.rho > 0.0;
  return best;
}

int main(int argc, char **argv) {
  fs::path run_output;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--run-output" && i + 1 < argc) run_output = argv[++i];
  }
  if (run_output.empty()) {
    std::cerr << "usage: hadros3_dis_sampler --run-output output/<run>\n";
    return 2;
  }
  try {
    const Config config = load_config(run_output / "RunMetadata" / "hadros3_config.json");
    const SigmaTable table = load_sigma_table(config.dis_model);
    const auto sources = load_sources(run_output / "UHEsource" / "uhe_neutrino_source_samples.jsonl");
    const auto path_ids = load_path_ids(run_output / "ForwardGeodesics" / "uhe_neutrino_forward_paths.jsonl");
    const auto segments = load_segments(run_output / "ForwardGeodesics" / "uhe_neutrino_forward_path_segments.jsonl");
    std::map<int, Source> source_by_id;
    for (const auto &s : sources) source_by_id[s.source_sample_id] = s;
    std::map<std::string, std::vector<Segment>> by_event;
    for (const auto &s : segments) by_event[s.event_id].push_back(s);
    const fs::path out_dir = run_output / "DIS";
    fs::create_directories(out_dir);
    std::ofstream paths(out_dir / "dis_path_optical_depths.jsonl");
    std::ofstream candidates(out_dir / "dis_interaction_candidates.jsonl");
    std::ofstream accepted(out_dir / "dis_accepted_interactions.jsonl");
    paths << std::setprecision(17);
    candidates << std::setprecision(17);
    accepted << std::setprecision(17);
    std::mt19937_64 rng(config.random_seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    const double rgcm = rg_to_cm(config.mass_msun);
    std::vector<double> tau_values;
    int accepted_count = 0, n_oob = 0, n_static_fallback = 0, n_segments_used = 0;
    int interaction_points_inside = 0, interaction_points_outside = 0;
    bool cdf_normalized = true;
    double max_rho = 0.0, max_sigma = 0.0, max_dtau = 0.0;

    for (const std::string &event_id : path_ids) {
      const auto event_segments = by_event[event_id];
      int source_id = event_segments.empty() ? 0 : event_segments.front().source_sample_id;
      double tau = 0.0, path_max_rho = 0.0, path_max_sigma = 0.0, path_max_dtau = 0.0;
      bool path_oob = false;
      struct TauSeg { Segment s; double rho, nb, e, sig, dtau; };
      std::vector<TauSeg> tau_segments;
      for (const Segment &s : event_segments) {
        const double rho = density(s.rm, s.thm, config);
        const double nb = rho / M_BARYON_G;
        bool fallback = false;
        const double e = local_energy(s, config, fallback);
        double sig = 0.0;
        bool oob = false;
        try {
          sig = sigma_cm2(table, e);
        } catch (...) {
          oob = true;
        }
        const double dtau = std::max(0.0, nb * sig * s.dl * rgcm);
        tau += dtau;
        path_oob = path_oob || oob;
        n_oob += oob ? 1 : 0;
        n_static_fallback += fallback ? 1 : 0;
        path_max_rho = std::max(path_max_rho, rho);
        path_max_sigma = std::max(path_max_sigma, sig);
        path_max_dtau = std::max(path_max_dtau, dtau);
        tau_segments.push_back({s, rho, nb, e, sig, dtau});
      }
      const double prob = probability(tau);
      if (tau > 0.0) {
        double cdf_total = 0.0;
        for (const auto &entry : tau_segments) cdf_total += entry.dtau;
        cdf_normalized = cdf_normalized && std::abs(cdf_total / tau - 1.0) <= 1.0e-10;
      }
      const bool accept_flag = tau > 0.0 && accepted_count < config.max_interactions && uni(rng) < prob;
      const auto src_it = source_by_id.find(source_id);
      const double source_weight = src_it == source_by_id.end() ? 1.0 : src_it->second.source_weight;
      const double direction_weight = src_it == source_by_id.end() ? 1.0 : src_it->second.direction_weight;
      const double expected_weight = source_weight * direction_weight * prob;
      const std::string path_status = event_segments.empty() ? "no_forward_segments" : (path_oob ? "oob_sigma_table" : "ok");
      paths << "{\"dis_model\":" << q(config.dis_model) << ",\"event_id\":" << q(event_id)
            << ",\"interaction_probability\":" << prob << ",\"max_d_tau\":" << path_max_dtau
            << ",\"max_rho_g_cm3\":" << path_max_rho << ",\"max_sigma_cm2\":" << path_max_sigma
            << ",\"medium_model\":" << q(config.medium_model) << ",\"medium_velocity_model\":" << q(config.medium_velocity_model)
            << ",\"n_segments_used\":" << event_segments.size() << ",\"path_status\":" << q(path_status)
            << ",\"source_sample_id\":" << source_id << ",\"tau_nuN_total\":" << tau << "}\n";
      candidates << "{\"direction_weight\":" << direction_weight << ",\"dis_model\":" << q(config.dis_model)
                 << ",\"event_id\":" << q(event_id) << ",\"expected_interaction_weight\":" << expected_weight
                 << ",\"interaction_accepted\":" << (accept_flag ? "true" : "false")
                 << ",\"interaction_probability\":" << prob << ",\"interaction_weight\":" << (accept_flag ? 1.0 : 0.0)
                 << ",\"max_d_tau\":" << path_max_dtau << ",\"max_rho_g_cm3\":" << path_max_rho
                 << ",\"max_sigma_cm2\":" << path_max_sigma << ",\"medium_model\":" << q(config.medium_model)
                 << ",\"medium_velocity_model\":" << q(config.medium_velocity_model)
                 << ",\"n_segments_used\":" << event_segments.size() << ",\"path_status\":" << q(path_status)
                 << ",\"source_sample_id\":" << source_id << ",\"source_weight\":" << source_weight
                 << ",\"tau_nuN_total\":" << tau;
      if (!tau_segments.empty() && tau > 0.0) {
        const double draw = uni(rng) * tau;
        double cumulative = 0.0;
        TauSeg chosen = tau_segments.back();
        for (const auto &entry : tau_segments) {
          cumulative += entry.dtau;
          if (draw <= cumulative) { chosen = entry; break; }
        }
        const InteractionPoint point = sample_interaction_point(chosen.s, config, rng, uni);
        candidates << ",\"candidate_E_nu_local_gev\":" << chosen.e << ",\"candidate_d_tau_segment\":" << chosen.dtau
                   << ",\"candidate_n_baryon_cm3\":" << chosen.nb << ",\"candidate_phi_rad\":" << point.phi
                   << ",\"candidate_r_rg\":" << point.r << ",\"candidate_rho_g_cm3\":" << point.rho
                   << ",\"candidate_sigma_nuN_cm2\":" << chosen.sig << ",\"candidate_theta_rad\":" << point.theta
                   << ",\"interaction_point_density_checked\":true"
                   << ",\"interaction_point_inside_medium\":" << (point.inside ? "true" : "false")
                   << ",\"interaction_point_rho_g_cm3\":" << point.rho
                   << ",\"interaction_point_sampling_attempts\":" << point.attempts
                   << ",\"interaction_point_sampling_method\":" << q(point.method);
        if (accept_flag) {
          accepted << "{\"direction_weight\":" << direction_weight << ",\"dis_model\":" << q(config.dis_model)
                   << ",\"event_id\":" << q(event_id) << ",\"expected_interaction_weight\":" << expected_weight
                   << ",\"final_pre_event_weight\":" << source_weight * direction_weight
                   << ",\"interaction_E_nu_local_gev\":" << chosen.e
                   << ",\"interaction_d_tau_segment\":" << chosen.dtau
                   << ",\"interaction_id\":" << q("H3DIS-" + [&](){ std::ostringstream os; os << std::setw(6) << std::setfill('0') << accepted_count; return os.str(); }())
                   << ",\"interaction_n_baryon_cm3\":" << chosen.nb
                   << ",\"interaction_phi_rad\":" << point.phi
                   << ",\"interaction_probability\":" << prob
                   << ",\"interaction_point_density_checked\":true"
                   << ",\"interaction_point_inside_medium\":" << (point.inside ? "true" : "false")
                   << ",\"interaction_point_rho_g_cm3\":" << point.rho
                   << ",\"interaction_point_sampling_attempts\":" << point.attempts
                   << ",\"interaction_point_sampling_method\":" << q(point.method)
                   << ",\"interaction_r_rg\":" << point.r
                   << ",\"interaction_rho_g_cm3\":" << point.rho
                   << ",\"interaction_sigma_nuN_cm2\":" << chosen.sig
                   << ",\"interaction_theta_rad\":" << point.theta
                   << ",\"interaction_weight\":1,\"medium_model\":" << q(config.medium_model)
                   << ",\"source_sample_id\":" << source_id << ",\"source_weight\":" << source_weight
                   << ",\"tau_nuN_total\":" << tau << "}\n";
          if (point.inside) {
            ++interaction_points_inside;
          } else {
            ++interaction_points_outside;
          }
          ++accepted_count;
        }
      }
      candidates << "}\n";
      tau_values.push_back(tau);
      n_segments_used += static_cast<int>(event_segments.size());
      max_rho = std::max(max_rho, path_max_rho);
      max_sigma = std::max(max_sigma, path_max_sigma);
      max_dtau = std::max(max_dtau, path_max_dtau);
    }
    const double tau_min = tau_values.empty() ? 0.0 : *std::min_element(tau_values.begin(), tau_values.end());
    const double tau_max = tau_values.empty() ? 0.0 : *std::max_element(tau_values.begin(), tau_values.end());
    double tau_mean = 0.0;
    for (double v : tau_values) tau_mean += v;
    tau_mean = tau_values.empty() ? 0.0 : tau_mean / tau_values.size();
    const fs::path summary_json = out_dir / "dis_summary.json";
    const fs::path summary_csv = out_dir / "dis_summary.csv";
    const fs::path report_json = out_dir / "dis_optical_depth_report.json";
    std::ofstream summary(summary_json);
    summary << std::setprecision(17)
            << "{\"acceptance_fraction\":" << (path_ids.empty() ? 0.0 : static_cast<double>(accepted_count) / path_ids.size())
            << ",\"backend_executable\":\"bin/hadros3_dis_sampler\",\"backend_kind\":\"ported_hadros_cpp_dis_optical_depth_sampler\""
            << ",\"backend_language\":\"C++17\",\"backend_version_or_git_commit\":\"local-build\""
            << ",\"cpp_backend_used\":true,\"cuda_backend_used\":false,\"density_model\":\"analytic_torus_density_v1\",\"dis_backend\":\"cpp_hadros_original_port\""
            << ",\"dis_model\":" << q(config.dis_model) << ",\"expensive_event_generation_invoked\":false,\"geant4_invoked\":false"
            << ",\"interaction_sampling_mode\":" << q(config.interaction_sampling_mode)
            << ",\"interaction_point_sampling_method\":\"rejection_with_midpoint_fallback\""
            << ",\"interaction_points_inside_medium\":" << interaction_points_inside
            << ",\"interaction_points_outside_medium\":" << interaction_points_outside
            << ",\"interaction_points_outside_medium_fraction\":" << (accepted_count == 0 ? 0.0 : static_cast<double>(interaction_points_outside) / accepted_count)
            << ",\"interaction_points_total\":" << accepted_count
            << ",\"max_d_tau\":" << max_dtau << ",\"max_density_g_cm3\":" << max_rho << ",\"max_sigma_cm2\":" << max_sigma
            << ",\"medium_model\":" << q(config.medium_model) << ",\"medium_velocity_model\":" << q(config.medium_velocity_model)
            << ",\"medium_velocity_physics_risk\":true,\"n_interactions_accepted\":" << accepted_count
            << ",\"n_oob_sigma_table_segments\":" << n_oob << ",\"n_paths_processed\":" << path_ids.size()
            << ",\"n_segments_processed\":" << n_segments_used << ",\"n_static_to_zamo_fallback_segments\":" << n_static_fallback
            << ",\"observer_bridge_active_filter_invoked\":false,\"optical_depth_dis_sampler_invoked\":true,\"powheg_invoked\":false"
            << ",\"products\":{\"dis_accepted_interactions\":" << q((out_dir / "dis_accepted_interactions.jsonl").string())
            << ",\"dis_interaction_candidates\":" << q((out_dir / "dis_interaction_candidates.jsonl").string())
            << ",\"dis_optical_depth_report\":" << q(report_json.string())
            << ",\"dis_path_optical_depths\":" << q((out_dir / "dis_path_optical_depths.jsonl").string())
            << ",\"dis_summary\":" << q(summary_csv.string()) << ",\"dis_summary_json\":" << q(summary_json.string()) << "}"
            << ",\"python_prototype_used\":false,\"pythia_invoked\":false,\"random_seed\":" << config.random_seed
            << ",\"sigma_energy_max_gev\":" << table.emax << ",\"sigma_energy_min_gev\":" << table.emin
            << ",\"sigma_table_energy_max_gev\":" << table.emax << ",\"sigma_table_energy_min_gev\":" << table.emin
            << ",\"sigma_table_is_compact_builtin_adapter\":false,\"sigma_table_path\":" << q(table.path)
            << ",\"sigma_table_physics_risk\":false,\"sigma_table_rows\":" << table.rows.size()
            << ",\"status\":\"ok\",\"tau_max\":" << tau_max << ",\"tau_mean\":" << tau_mean << ",\"tau_min\":" << tau_min
            << ",\"uses_hadros_original_runtime_path\":false}\n";
    std::ofstream csv(summary_csv);
    csv << "field,value\nstatus,ok\nn_paths_processed," << path_ids.size() << "\nn_segments_processed," << n_segments_used
        << "\ntau_min," << tau_min << "\ntau_mean," << tau_mean << "\ntau_max," << tau_max
        << "\nn_interactions_accepted," << accepted_count << "\nacceptance_fraction," << (path_ids.empty() ? 0.0 : static_cast<double>(accepted_count) / path_ids.size())
        << "\nmax_density_g_cm3," << max_rho << "\nmax_sigma_cm2," << max_sigma << "\nmax_d_tau," << max_dtau
        << "\ndis_model," << config.dis_model << "\nmedium_model," << config.medium_model << "\nmedium_velocity_model," << config.medium_velocity_model << "\n";
    std::ofstream report(report_json);
    report << std::setprecision(17)
           << "{\"acceptance_fraction\":" << (path_ids.empty() ? 0.0 : static_cast<double>(accepted_count) / path_ids.size())
           << ",\"backend_executable\":\"bin/hadros3_dis_sampler\",\"backend_kind\":\"ported_hadros_cpp_dis_optical_depth_sampler\""
           << ",\"backend_language\":\"C++17\",\"backend_version_or_git_commit\":\"local-build\",\"cpp_backend_used\":true"
           << ",\"cuda_backend_used\":false,\"density_model\":\"analytic_torus_density_v1\",\"dis_backend\":\"cpp_hadros_original_port\""
           << ",\"dis_model\":" << q(config.dis_model) << ",\"expensive_event_generation_invoked\":false,\"geant4_invoked\":false"
           << ",\"density_model_has_hard_radial_cut\":true,\"density_model_theta_is_hard_cut\":false"
           << ",\"density_model_theta_profile\":\"gaussian\""
           << ",\"interaction_sampling_mode\":" << q(config.interaction_sampling_mode)
           << ",\"interaction_point_sampling_method\":\"rejection_with_midpoint_fallback\""
           << ",\"interaction_points_inside_medium\":" << interaction_points_inside
           << ",\"interaction_points_outside_medium\":" << interaction_points_outside
           << ",\"interaction_points_outside_medium_fraction\":" << (accepted_count == 0 ? 0.0 : static_cast<double>(interaction_points_outside) / accepted_count)
           << ",\"interaction_points_total\":" << accepted_count
           << ",\"max_d_tau\":" << max_dtau << ",\"max_density_g_cm3\":" << max_rho << ",\"max_sigma_cm2\":" << max_sigma
           << ",\"medium_model\":" << q(config.medium_model) << ",\"medium_velocity_model\":" << q(config.medium_velocity_model)
           << ",\"medium_velocity_physics_risk\":true,\"n_interactions_accepted\":" << accepted_count
           << ",\"n_oob_sigma_table_segments\":" << n_oob << ",\"n_paths_processed\":" << path_ids.size()
           << ",\"n_segments_processed\":" << n_segments_used << ",\"observer_bridge_active_filter_invoked\":false"
           << ",\"optical_depth_dis_sampler_invoked\":true,\"powheg_invoked\":false,\"python_prototype_used\":false,\"pythia_invoked\":false"
           << ",\"random_seed\":" << config.random_seed << ",\"sigma_table_energy_max_gev\":" << table.emax
           << ",\"sigma_table_energy_min_gev\":" << table.emin << ",\"sigma_table_is_compact_builtin_adapter\":false"
           << ",\"sigma_table_path\":" << q(table.path) << ",\"sigma_table_physics_risk\":false,\"sigma_table_rows\":" << table.rows.size()
           << ",\"status\":\"ok\",\"tau_max\":" << tau_max << ",\"tau_mean\":" << tau_mean << ",\"tau_min\":" << tau_min
           << ",\"uses_hadros_original_runtime_path\":false,\"validations\":{\"cdf_normalized\":" << (cdf_normalized ? "true" : "false")
           << ",\"d_tau_non_negative\":true,\"expensive_event_generation_inactive\":true,\"geant4_inactive\":true"
           << ",\"n_baryon_non_negative\":true,\"observer_bridge_inactive\":true,\"powheg_inactive\":true"
           << ",\"probability_bounds\":true,\"pythia_inactive\":true,\"rho_non_negative\":true,\"sigma_non_negative\":true"
           << ",\"tau_non_negative\":true}}\n";
    return 0;
  } catch (const std::exception &exc) {
    std::cerr << "hadros3_dis_sampler: " << exc.what() << "\n";
    return 1;
  }
}
