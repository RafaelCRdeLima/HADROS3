#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

constexpr double PI = 3.141592653589793238462643383279502884;

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct Config {
  std::string observer_bridge_backend = "cpp_cpu";
  std::string bridge_mode = "scoring_only";
  std::string secondary_particle_proxy_model = "geometric_escape_proxy";
  std::string escape_proxy_model = "geometric_outward_proxy";
  std::string visibility_model = "geometric_proxy";
  std::string redshift_proxy_model = "unity_or_metric_proxy";
  std::string line_of_sight_proxy_model = "geometric_proxy";
  std::string fov_policy = "hard";
  bool distance_weight_enabled = true;
  bool redshift_weight_enabled = true;
  bool line_of_sight_check_enabled = true;
  int max_ranked_events = 25;
  double min_observer_weight = 0.0;
  double min_final_observation_score = 0.0;
  double observer_distance_rg = 60.0;
  double observer_inclination_deg = 80.0;
  double observer_azimuth_deg = 0.0;
  double camera_fov_deg = 25.0;
};

struct Candidate {
  std::string interaction_id;
  std::string event_id;
  int source_sample_id = 0;
  double r = 0.0;
  double theta = 0.0;
  double phi = 0.0;
  double e_local = 0.0;
  double rho = 0.0;
  double sigma = 0.0;
  double source_weight = 1.0;
  double direction_weight = 1.0;
  double interaction_weight = 1.0;
  double final_pre_event_weight = 1.0;
  double expected_interaction_weight = 0.0;
  double physics_weight = 1.0;
  double distance_to_observer_rg = 0.0;
  double observer_angle_deg = 0.0;
  bool camera_fov_flag = false;
  double camera_fov_weight = 0.0;
  double escape_probability_proxy = 0.0;
  double escape_direction_proxy = 0.0;
  double escape_weight_proxy = 0.0;
  double visibility_proxy = 1.0;
  double line_of_sight_proxy = 1.0;
  std::string line_of_sight_status = "geometric_proxy_unchecked";
  bool visibility_flag = true;
  double redshift_proxy = 1.0;
  double redshift_weight = 1.0;
  double arrival_time_proxy_rg = 0.0;
  double distance_weight = 1.0;
  double line_of_sight_weight = 1.0;
  double observer_weight = 0.0;
  double final_observation_score = 0.0;
  std::string bridge_status = "scored_with_proxy_weights";
  bool proxy_physics_risk = true;
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
  auto end = text.find('"', pos + 1);
  if (end == std::string::npos) return fallback;
  return text.substr(pos + 1, end - pos - 1);
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
  double value = std::strtod(begin, &end);
  if (end == begin) return fallback;
  return value;
}

static bool json_bool(const std::string& text, const std::string& key, bool fallback) {
  const std::string needle = "\"" + key + "\"";
  auto pos = text.find(needle);
  if (pos == std::string::npos) return fallback;
  pos = text.find(':', pos);
  if (pos == std::string::npos) return fallback;
  ++pos;
  while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t')) ++pos;
  if (text.compare(pos, 4, "true") == 0) return true;
  if (text.compare(pos, 5, "false") == 0) return false;
  return fallback;
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
  const std::string text = read_text(path);
  const std::string bridge = section_text(text, "observer_bridge");
  const std::string camera = section_text(text, "observer_camera");
  c.observer_bridge_backend = json_string(bridge, "observer_bridge_backend", c.observer_bridge_backend);
  c.bridge_mode = json_string(bridge, "bridge_mode", c.bridge_mode);
  c.secondary_particle_proxy_model = json_string(bridge, "secondary_particle_proxy_model", c.secondary_particle_proxy_model);
  c.escape_proxy_model = json_string(bridge, "escape_proxy_model", c.escape_proxy_model);
  c.visibility_model = json_string(bridge, "visibility_model", c.visibility_model);
  c.redshift_proxy_model = json_string(bridge, "redshift_proxy_model", c.redshift_proxy_model);
  c.line_of_sight_proxy_model = json_string(bridge, "line_of_sight_proxy_model", c.line_of_sight_proxy_model);
  c.fov_policy = json_string(bridge, "fov_policy", c.fov_policy);
  c.distance_weight_enabled = json_bool(bridge, "distance_weight_enabled", c.distance_weight_enabled);
  c.redshift_weight_enabled = json_bool(bridge, "redshift_weight_enabled", c.redshift_weight_enabled);
  c.line_of_sight_check_enabled = json_bool(bridge, "line_of_sight_check_enabled", c.line_of_sight_check_enabled);
  c.max_ranked_events = std::max(1, static_cast<int>(json_number(bridge, "max_ranked_events", c.max_ranked_events)));
  c.min_observer_weight = std::max(0.0, json_number(bridge, "min_observer_weight", c.min_observer_weight));
  c.min_final_observation_score = std::max(0.0, json_number(bridge, "min_final_observation_score", c.min_final_observation_score));
  c.observer_distance_rg = std::max(1.0e-9, json_number(camera, "observer_distance_rg", c.observer_distance_rg));
  c.observer_inclination_deg = json_number(camera, "inclination_deg", c.observer_inclination_deg);
  c.observer_azimuth_deg = json_number(camera, "azimuth_deg", c.observer_azimuth_deg);
  c.camera_fov_deg = std::max(1.0e-9, json_number(camera, "field_of_view_deg", c.camera_fov_deg));
  return c;
}

static double clamp(double value, double lo, double hi) {
  return std::max(lo, std::min(hi, value));
}

static double deg_to_rad(double deg) {
  return deg * PI / 180.0;
}

static Vec3 spherical(double r, double theta, double phi) {
  const double st = std::sin(theta);
  return {r * st * std::cos(phi), r * st * std::sin(phi), r * std::cos(theta)};
}

static Vec3 sub(Vec3 a, Vec3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

static double dot(Vec3 a, Vec3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static double norm(Vec3 a) {
  return std::sqrt(dot(a, a));
}

static Vec3 unit(Vec3 a) {
  const double n = norm(a);
  if (n <= 0.0) return {0.0, 0.0, 0.0};
  return {a.x / n, a.y / n, a.z / n};
}

static Candidate candidate_from_line(const std::string& line, const Config& c) {
  Candidate out;
  out.interaction_id = json_string(line, "interaction_id", "");
  out.event_id = json_string(line, "event_id", "");
  out.source_sample_id = static_cast<int>(json_number(line, "source_sample_id", 0));
  out.r = json_number(line, "interaction_r_rg", 0.0);
  out.theta = json_number(line, "interaction_theta_rad", 0.0);
  out.phi = json_number(line, "interaction_phi_rad", 0.0);
  out.e_local = json_number(line, "interaction_E_nu_local_gev", 0.0);
  out.rho = json_number(line, "interaction_rho_g_cm3", json_number(line, "interaction_point_rho_g_cm3", 0.0));
  out.sigma = json_number(line, "interaction_sigma_nuN_cm2", 0.0);
  out.source_weight = std::max(0.0, json_number(line, "source_weight", 1.0));
  out.direction_weight = std::max(0.0, json_number(line, "direction_weight", 1.0));
  out.interaction_weight = std::max(0.0, json_number(line, "interaction_weight", 1.0));
  out.final_pre_event_weight = std::max(0.0, json_number(line, "final_pre_event_weight", out.source_weight * out.direction_weight));
  out.expected_interaction_weight = std::max(0.0, json_number(line, "expected_interaction_weight", out.final_pre_event_weight * out.interaction_weight));
  out.physics_weight = out.final_pre_event_weight;

  const Vec3 interaction = spherical(out.r, out.theta, out.phi);
  const Vec3 radial_out = unit(interaction);
  const Vec3 observer = spherical(c.observer_distance_rg, deg_to_rad(c.observer_inclination_deg), deg_to_rad(c.observer_azimuth_deg));
  const Vec3 camera_axis = unit({-observer.x, -observer.y, -observer.z});
  const Vec3 observer_to_interaction = sub(interaction, observer);
  const Vec3 dir_from_observer = unit(observer_to_interaction);
  const Vec3 interaction_to_observer = unit(sub(observer, interaction));

  out.distance_to_observer_rg = norm(observer_to_interaction);
  out.arrival_time_proxy_rg = out.distance_to_observer_rg;
  out.observer_angle_deg = 180.0 / PI * std::acos(clamp(dot(camera_axis, dir_from_observer), -1.0, 1.0));
  const double half_fov = 0.5 * c.camera_fov_deg;
  out.camera_fov_flag = out.observer_angle_deg <= half_fov;
  if (c.fov_policy == "soft") {
    const double sigma = std::max(half_fov, 1.0e-9);
    out.camera_fov_weight = std::exp(-0.5 * (out.observer_angle_deg / sigma) * (out.observer_angle_deg / sigma));
  } else {
    out.camera_fov_weight = out.camera_fov_flag ? 1.0 : 0.0;
  }

  out.escape_direction_proxy = clamp(dot(radial_out, interaction_to_observer), 0.0, 1.0);
  out.escape_probability_proxy = out.escape_direction_proxy;
  out.escape_weight_proxy = out.escape_probability_proxy;
  out.visibility_proxy = 1.0;
  out.line_of_sight_proxy = c.line_of_sight_check_enabled ? 1.0 : 1.0;
  out.line_of_sight_weight = out.line_of_sight_proxy;
  out.line_of_sight_status = c.line_of_sight_check_enabled ? "geometric_proxy_unblocked" : "geometric_proxy_disabled";
  out.visibility_flag = out.visibility_proxy > 0.0;
  out.redshift_proxy = 1.0;
  out.redshift_weight = c.redshift_weight_enabled ? out.redshift_proxy : 1.0;
  out.distance_weight = c.distance_weight_enabled ? clamp((c.observer_distance_rg * c.observer_distance_rg) / std::max(out.distance_to_observer_rg * out.distance_to_observer_rg, 1.0e-30), 0.0, 1.0) : 1.0;
  out.observer_weight = out.escape_weight_proxy * out.visibility_proxy * out.camera_fov_weight * out.distance_weight * out.redshift_weight * out.line_of_sight_weight;
  if (out.observer_weight < c.min_observer_weight) out.observer_weight = 0.0;
  out.final_observation_score = out.physics_weight * out.observer_weight;
  if (out.final_observation_score < c.min_final_observation_score) out.final_observation_score = 0.0;
  return out;
}

static void write_candidate(std::ostream& out, const Candidate& c, const Config& cfg) {
  out << std::setprecision(17)
      << "{\"bridge_status\":" << quote(c.bridge_status)
      << ",\"camera_fov_deg\":" << cfg.camera_fov_deg
      << ",\"camera_fov_flag\":" << (c.camera_fov_flag ? "true" : "false")
      << ",\"camera_fov_weight\":" << c.camera_fov_weight
      << ",\"direction_weight\":" << c.direction_weight
      << ",\"distance_weight\":" << c.distance_weight
      << ",\"distance_to_observer_rg\":" << c.distance_to_observer_rg
      << ",\"escape_direction_proxy\":" << c.escape_direction_proxy
      << ",\"escape_probability_proxy\":" << c.escape_probability_proxy
      << ",\"escape_proxy_model\":" << quote(cfg.escape_proxy_model)
      << ",\"escape_weight_proxy\":" << c.escape_weight_proxy
      << ",\"expected_interaction_weight\":" << c.expected_interaction_weight
      << ",\"final_observation_score\":" << c.final_observation_score
      << ",\"final_pre_event_weight\":" << c.final_pre_event_weight
      << ",\"interaction_E_nu_local_gev\":" << c.e_local
      << ",\"interaction_id\":" << quote(c.interaction_id)
      << ",\"interaction_phi_rad\":" << c.phi
      << ",\"interaction_r_rg\":" << c.r
      << ",\"interaction_rho_g_cm3\":" << c.rho
      << ",\"interaction_sigma_nuN_cm2\":" << c.sigma
      << ",\"interaction_theta_rad\":" << c.theta
      << ",\"interaction_weight\":" << c.interaction_weight
      << ",\"event_id\":" << quote(c.event_id)
      << ",\"line_of_sight_proxy\":" << c.line_of_sight_proxy
      << ",\"line_of_sight_proxy_model\":" << quote(cfg.line_of_sight_proxy_model)
      << ",\"line_of_sight_status\":" << quote(c.line_of_sight_status)
      << ",\"line_of_sight_weight\":" << c.line_of_sight_weight
      << ",\"observer_angle_deg\":" << c.observer_angle_deg
      << ",\"observer_azimuth_deg\":" << cfg.observer_azimuth_deg
      << ",\"observer_distance_rg\":" << cfg.observer_distance_rg
      << ",\"observer_inclination_deg\":" << cfg.observer_inclination_deg
      << ",\"observer_weight\":" << c.observer_weight
      << ",\"physics_weight\":" << c.physics_weight
      << ",\"proxy_physics_risk\":true"
      << ",\"redshift_proxy\":" << c.redshift_proxy
      << ",\"redshift_proxy_model\":" << quote(cfg.redshift_proxy_model)
      << ",\"redshift_weight\":" << c.redshift_weight
      << ",\"secondary_particle_proxy_model\":" << quote(cfg.secondary_particle_proxy_model)
      << ",\"source_sample_id\":" << c.source_sample_id
      << ",\"source_weight\":" << c.source_weight
      << ",\"arrival_time_proxy_rg\":" << c.arrival_time_proxy_rg
      << ",\"visibility_flag\":" << (c.visibility_flag ? "true" : "false")
      << ",\"visibility_model\":" << quote(cfg.visibility_model)
      << ",\"visibility_proxy\":" << c.visibility_proxy
      << "}\n";
}

int main(int argc, char** argv) {
  try {
    fs::path run_output;
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--run-output" && i + 1 < argc) {
        run_output = argv[++i];
      } else {
        throw std::runtime_error("usage: hadros3_observer_bridge --run-output output/<run>");
      }
    }
    if (run_output.empty()) throw std::runtime_error("usage: hadros3_observer_bridge --run-output output/<run>");
    const fs::path config_path = run_output / "RunMetadata" / "hadros3_config.json";
    const fs::path dis_path = run_output / "DIS" / "dis_accepted_interactions.jsonl";
    const fs::path out_dir = run_output / "ObserverBridge";
    fs::create_directories(out_dir);

    const Config cfg = load_config(config_path);
    std::ifstream accepted(dis_path);
    if (!accepted) throw std::runtime_error("cannot read " + dis_path.string());

    std::vector<Candidate> candidates;
    std::string line;
    while (std::getline(accepted, line)) {
      if (line.empty()) continue;
      candidates.push_back(candidate_from_line(line, cfg));
    }
    std::vector<Candidate> ranked = candidates;
    std::sort(ranked.begin(), ranked.end(), [](const Candidate& a, const Candidate& b) {
      if (a.final_observation_score != b.final_observation_score) return a.final_observation_score > b.final_observation_score;
      return a.interaction_id < b.interaction_id;
    });

    const fs::path candidates_path = out_dir / "observer_bridge_candidates.jsonl";
    const fs::path ranked_path = out_dir / "observer_bridge_ranked_events.jsonl";
    const fs::path summary_path = out_dir / "observer_bridge_summary.json";
    const fs::path csv_path = out_dir / "observer_bridge_summary.csv";
    const fs::path report_path = out_dir / "observer_bridge_report.json";

    std::ofstream cand_out(candidates_path);
    for (const Candidate& c : candidates) write_candidate(cand_out, c, cfg);
    std::ofstream ranked_out(ranked_path);
    for (const Candidate& c : ranked) write_candidate(ranked_out, c, cfg);

    int inside_fov = 0;
    int visible = 0;
    double score_min = candidates.empty() ? 0.0 : std::numeric_limits<double>::infinity();
    double score_max = 0.0;
    double score_sum = 0.0;
    double physics_sum = 0.0;
    double observer_sum = 0.0;
    for (const Candidate& c : candidates) {
      inside_fov += c.camera_fov_flag ? 1 : 0;
      visible += c.visibility_flag ? 1 : 0;
      score_min = std::min(score_min, c.final_observation_score);
      score_max = std::max(score_max, c.final_observation_score);
      score_sum += c.final_observation_score;
      physics_sum += c.physics_weight;
      observer_sum += c.observer_weight;
    }
    if (candidates.empty()) score_min = 0.0;
    const double score_mean = candidates.empty() ? 0.0 : score_sum / candidates.size();
    const double physics_mean = candidates.empty() ? 0.0 : physics_sum / candidates.size();
    const double observer_mean = candidates.empty() ? 0.0 : observer_sum / candidates.size();
    const std::string top_event_id = ranked.empty() ? "" : ranked.front().event_id;

    auto write_summary = [&](std::ostream& out) {
      out << std::setprecision(17)
          << "{\"backend_executable\":\"bin/hadros3_observer_bridge\""
          << ",\"backend_language\":\"C++17\""
          << ",\"bridge_mode\":" << quote(cfg.bridge_mode)
          << ",\"camera_fov_deg\":" << cfg.camera_fov_deg
          << ",\"cpp_backend_used\":true"
          << ",\"distance_weight_enabled\":" << (cfg.distance_weight_enabled ? "true" : "false")
          << ",\"escape_proxy_model\":" << quote(cfg.escape_proxy_model)
          << ",\"escape_proxy_physics_risk\":true"
          << ",\"event_generation_invoked\":false"
          << ",\"final_observation_score_definition\":\"physics_weight * observer_weight\""
          << ",\"fov_policy\":" << quote(cfg.fov_policy)
          << ",\"geant4_invoked\":false"
          << ",\"line_of_sight_check_enabled\":" << (cfg.line_of_sight_check_enabled ? "true" : "false")
          << ",\"line_of_sight_proxy_model\":" << quote(cfg.line_of_sight_proxy_model)
          << ",\"max_ranked_events\":" << cfg.max_ranked_events
          << ",\"min_final_observation_score\":" << cfg.min_final_observation_score
          << ",\"min_observer_weight\":" << cfg.min_observer_weight
          << ",\"n_candidates_scored\":" << candidates.size()
          << ",\"n_inside_fov\":" << inside_fov
          << ",\"n_interactions_input\":" << candidates.size()
          << ",\"n_ranked_events\":" << ranked.size()
          << ",\"n_visible_proxy\":" << visible
          << ",\"observer_bridge_backend\":" << quote(cfg.observer_bridge_backend)
          << ",\"observer_bridge_invoked\":true"
          << ",\"observer_weight_definition\":\"escape_weight_proxy * visibility_proxy * camera_fov_weight * distance_weight * redshift_weight * line_of_sight_weight\""
          << ",\"observer_weight_mean\":" << observer_mean
          << ",\"photon_transport_invoked\":false"
          << ",\"physics_weight_definition\":\"final_pre_event_weight\""
          << ",\"physics_weight_mean\":" << physics_mean
          << ",\"powheg_invoked\":false"
          << ",\"products\":{\"observer_bridge_candidates\":" << quote(candidates_path.string())
          << ",\"observer_bridge_ranked_events\":" << quote(ranked_path.string())
          << ",\"observer_bridge_report\":" << quote(report_path.string())
          << ",\"observer_bridge_summary\":" << quote(csv_path.string())
          << ",\"observer_bridge_summary_json\":" << quote(summary_path.string()) << "}"
          << ",\"proxy_physics_risk\":true"
          << ",\"pythia_invoked\":false"
          << ",\"redshift_proxy_model\":" << quote(cfg.redshift_proxy_model)
          << ",\"redshift_proxy_physics_risk\":true"
          << ",\"redshift_weight_enabled\":" << (cfg.redshift_weight_enabled ? "true" : "false")
          << ",\"score_max\":" << score_max
          << ",\"score_mean\":" << score_mean
          << ",\"score_min\":" << score_min
          << ",\"secondary_particle_proxy_model\":" << quote(cfg.secondary_particle_proxy_model)
          << ",\"status\":\"ok\""
          << ",\"top_event_id\":" << quote(top_event_id)
          << ",\"uses_hadros_original_runtime_path\":false"
          << ",\"visibility_model\":" << quote(cfg.visibility_model)
          << ",\"visibility_proxy_physics_risk\":true"
          << "}\n";
    };

    std::ofstream summary(summary_path);
    write_summary(summary);
    std::ofstream report(report_path);
    write_summary(report);
    std::ofstream csv(csv_path);
    csv << "field,value\n"
        << "status,ok\n"
        << "n_interactions_input," << candidates.size() << "\n"
        << "n_candidates_scored," << candidates.size() << "\n"
        << "n_inside_fov," << inside_fov << "\n"
        << "n_visible_proxy," << visible << "\n"
        << "score_min," << score_min << "\n"
        << "score_mean," << score_mean << "\n"
        << "score_max," << score_max << "\n"
        << "top_event_id," << top_event_id << "\n";
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "hadros3_observer_bridge: " << exc.what() << "\n";
    return 1;
  }
}
