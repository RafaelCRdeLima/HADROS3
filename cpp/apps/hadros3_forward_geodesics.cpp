#include "hadros/cascade/kerr_local_tetrad.hpp"
#include "hadros/cascade/packet_kerr_null_propagator.hpp"
#include "kerr_geodesic.hpp"
#include "kerr_metric.hpp"

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
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

struct Config {
  std::string geodesic_backend = "cpp_hadros_original_port";
  int n_samples_to_propagate = 64;
  int max_steps = 256;
  double initial_step_rg = 1.0;
  double outer_radius_rg = 80.0;
  double horizon_tolerance_rg = 1.0e-3;
  double null_invariant_tolerance = 1.0e-6;
  double killing_energy_tolerance = 1.0e-10;
  double lz_tolerance = 1.0e-10;
  double spin_a = 0.8;
};

struct Sample {
  std::string event_id;
  int source_sample_id = 0;
  double r = 0.0;
  double theta = 0.0;
  double phi = 0.0;
  double energy_gev = 0.0;
  std::string direction_model;
  std::string direction_generator;
  std::string basis;
  double n_r = 1.0;
  double n_theta = 0.0;
  double n_phi = 0.0;
  double dr = 1.0;
  double dtheta = 0.0;
  double dphi = 0.0;
};

std::string read_text(const fs::path& path)
{
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot read " + path.string());
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string json_string(const std::string& text, const std::string& key, const std::string& fallback = "")
{
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

double json_number(const std::string& text, const std::string& key, double fallback)
{
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
  if (end == begin) return fallback;
  return value;
}

std::string section_text(const std::string& text, const std::string& section)
{
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

std::string quote(const std::string& s)
{
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

Config load_config(const fs::path& path)
{
  Config c;
  const std::string text = read_text(path);
  const std::string forward = section_text(text, "forward_geodesics");
  const std::string black_hole = section_text(text, "black_hole");
  c.geodesic_backend = json_string(forward, "geodesic_backend", c.geodesic_backend);
  c.n_samples_to_propagate = static_cast<int>(json_number(forward, "n_samples_to_propagate", c.n_samples_to_propagate));
  c.max_steps = static_cast<int>(json_number(forward, "max_steps", c.max_steps));
  c.initial_step_rg = json_number(forward, "initial_step_rg", c.initial_step_rg);
  c.outer_radius_rg = json_number(forward, "outer_radius_rg", c.outer_radius_rg);
  c.horizon_tolerance_rg = json_number(forward, "horizon_tolerance_rg", c.horizon_tolerance_rg);
  c.null_invariant_tolerance = json_number(forward, "null_invariant_tolerance", c.null_invariant_tolerance);
  c.killing_energy_tolerance = json_number(forward, "killing_energy_tolerance", c.killing_energy_tolerance);
  c.lz_tolerance = json_number(forward, "lz_tolerance", c.lz_tolerance);
  c.spin_a = std::clamp(json_number(black_hole, "spin_a", c.spin_a), -0.999, 0.999);
  c.max_steps = std::max(c.max_steps, 1);
  c.n_samples_to_propagate = std::max(c.n_samples_to_propagate, 0);
  c.initial_step_rg = std::max(c.initial_step_rg, 1.0e-6);
  return c;
}

std::vector<Sample> load_samples(const fs::path& path)
{
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot read " + path.string());
  }
  std::vector<Sample> samples;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    Sample s;
    s.event_id = json_string(line, "event_id", "H3UHE-unknown");
    s.source_sample_id = static_cast<int>(json_number(line, "source_sample_id", 0));
    s.r = json_number(line, "x_emit_r", json_number(line, "r_rg", 0.0));
    s.theta = json_number(line, "x_emit_theta", json_number(line, "theta_rad", 0.0));
    s.phi = json_number(line, "x_emit_phi", json_number(line, "phi_rad", 0.0));
    s.energy_gev = json_number(line, "E_nu_emit_gev", 0.0);
    s.direction_model = json_string(line, "direction_model", "coordinate_radial_outward");
    s.direction_generator = json_string(line, "direction_generator", "CoordinateRadialOutwardDirectionGenerator");
    s.basis = json_string(line, "basis", s.direction_model == "isotropic_local" ? "ZAMO_orthonormal" : "Boyer-Lindquist_coordinate_direction");
    s.n_r = json_number(line, "n_r", 1.0);
    s.n_theta = json_number(line, "n_theta", 0.0);
    s.n_phi = json_number(line, "n_phi", 0.0);
    s.dr = json_number(line, "dr", 1.0);
    s.dtheta = json_number(line, "dtheta", 0.0);
    s.dphi = json_number(line, "dphi", 0.0);
    samples.push_back(s);
  }
  return samples;
}

GeodesicState initialize_state(const Sample& sample, const KerrMetric& metric)
{
  const hadros::cascade::LocalDirection direction{
      sample.direction_model == "isotropic_local" ? sample.n_r : sample.dr,
      sample.direction_model == "isotropic_local" ? sample.n_theta : sample.dtheta,
      sample.direction_model == "isotropic_local" ? sample.n_phi : sample.dphi,
  };
  const auto init = hadros::cascade::initialize_zamo_null_packet(
      metric,
      sample.r,
      std::clamp(sample.theta, 1.0e-6, PI - 1.0e-6),
      sample.phi,
      direction);
  if (!init.valid) {
    throw std::runtime_error("ZAMO tetrad initialization failed for " + sample.event_id + ": " + init.status);
  }
  return init.state;
}

GeodesicState normalize_poles(GeodesicState s)
{
  while (s.theta < 0.0 || s.theta > PI) {
    if (s.theta < 0.0) {
      s.theta = -s.theta;
      s.phi += PI;
      s.ptheta = -s.ptheta;
    } else {
      s.theta = 2.0 * PI - s.theta;
      s.phi += PI;
      s.ptheta = -s.ptheta;
    }
  }
  s.theta = std::clamp(s.theta, 1.0e-6, PI - 1.0e-6);
  return s;
}

double proper_path_distance(const KerrMetric& metric, const GeodesicState& a, const GeodesicState& b)
{
  const double rmid = std::max(0.5 * (a.r + b.r), 1.0e-6);
  const double thmid = 0.5 * (a.theta + b.theta);
  const double dphi = std::atan2(std::sin(b.phi - a.phi), std::cos(b.phi - a.phi));
  const double sigma = metric.Sigma(rmid, thmid);
  const double delta = std::max(metric.Delta(rmid), 1.0e-12);
  const double sin2 = std::max(std::sin(thmid) * std::sin(thmid), 1.0e-10);
  const double ds2 = (sigma / delta) * (b.r - a.r) * (b.r - a.r)
      + sigma * (b.theta - a.theta) * (b.theta - a.theta)
      + (metric.A(rmid, thmid) * sin2 / sigma) * dphi * dphi;
  return std::sqrt(std::max(ds2, 0.0));
}

double zamo_energy_gev(const KerrMetric& metric, const GeodesicState& state, double source_energy_gev)
{
  return hadros::cascade::zamo_packet_energy(metric, state.r, state.theta, state.pt, state.pphi) * source_energy_gev;
}

std::string backend_json_fields()
{
  return "\"backend_kind\":\"ported_hadros_kerr_engine\","
         "\"backend_language\":\"C++17\","
         "\"backend_executable\":\"bin/hadros3_forward_geodesics\","
         "\"backend_version_or_git_commit\":\"local-build\","
         "\"cpp_backend_used\":true,"
         "\"cuda_backend_used\":false,"
         "\"python_prototype_used\":false,"
         "\"uses_hadros_original_runtime_path\":false,"
         "\"uses_kerr_metric\":true,"
         "\"uses_hamiltonian\":true,"
         "\"uses_zamo_tetrad\":true,"
         "\"uses_christoffel_or_hamiltonian\":true";
}

void write_stop_csv(const fs::path& path, const std::map<std::string, int>& stop_counts, std::size_t n_paths)
{
  std::ofstream out(path);
  out << "stop_condition,count,fraction\n";
  for (const std::string key : {"outer_escape_radius", "horizon_crossing", "max_steps", "invalid_invariant"}) {
    const auto it = stop_counts.find(key);
    const int count = it == stop_counts.end() ? 0 : it->second;
    out << key << "," << count << "," << (n_paths == 0 ? 0.0 : static_cast<double>(count) / static_cast<double>(n_paths)) << "\n";
  }
}

}  // namespace

int main(int argc, char** argv)
{
  fs::path run_output;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--run-output" && i + 1 < argc) {
      run_output = argv[++i];
    }
  }
  if (run_output.empty()) {
    std::cerr << "usage: hadros3_forward_geodesics --run-output output/<run>\n";
    return 2;
  }

  try {
    const fs::path config_path = run_output / "RunMetadata" / "hadros3_config.json";
    const fs::path source_path = run_output / "UHEsource" / "uhe_neutrino_source_samples.jsonl";
    const fs::path out_dir = run_output / "ForwardGeodesics";
    fs::create_directories(out_dir);

    const Config config = load_config(config_path);
    const KerrMetric metric(config.spin_a);
    const double integrator_tolerance = std::min(config.null_invariant_tolerance * 0.01, 1.0e-8);
    const KerrGeodesic geodesic(metric, config.initial_step_rg, integrator_tolerance, KerrDerivativeMode::Analytic);

    std::vector<Sample> all_samples = load_samples(source_path);
    const std::size_t n_input_samples = all_samples.size();
    if (static_cast<int>(all_samples.size()) > config.n_samples_to_propagate) {
      all_samples.resize(config.n_samples_to_propagate);
    }

    std::ofstream paths(out_dir / "uhe_neutrino_forward_paths.jsonl");
    std::ofstream segments(out_dir / "uhe_neutrino_forward_path_segments.jsonl");
    if (!paths || !segments) {
      throw std::runtime_error("cannot open forward geodesic outputs");
    }
    paths << std::setprecision(17);
    segments << std::setprecision(17);

    std::map<std::string, int> stop_counts;
    int n_segments = 0;
    double null_max = 0.0;
    double eerr_max = 0.0;
    double lzerr_max = 0.0;
    double dtheta_max = 0.0;
    double dphi_max = 0.0;
    double curvature_max = 0.0;
    bool all_valid = true;
    const double r_horizon_stop = metric.horizon_radius() + config.horizon_tolerance_rg;
    const std::string first_direction_model = all_samples.empty() ? "" : all_samples.front().direction_model;
    const std::string first_direction_generator = all_samples.empty() ? "" : all_samples.front().direction_generator;

    for (const Sample& sample : all_samples) {
      GeodesicState state = initialize_state(sample, metric);
      const GeodesicState initial = state;
      const double p_t0 = state.pt;
      const double p_phi0 = state.pphi;
      double local_null_max = 0.0;
      double local_energy_error_max = 0.0;
      double local_lz_error_max = 0.0;
      double local_dtheta_max = 0.0;
      double local_dphi_max = 0.0;
      double local_curvature = 0.0;
      std::string stop = "max_steps";
      std::string status = "propagated_forward_no_interaction";
      int local_segments = 0;
      GeodesicState previous_rhs = geodesic.rhs(state);

      for (int segment_index = 0; segment_index < config.max_steps; ++segment_index) {
        if (state.r <= r_horizon_stop) {
          stop = "horizon_crossing";
          break;
        }
        if (state.r >= config.outer_radius_rg && segment_index > 0) {
          stop = "outer_escape_radius";
          break;
        }

        const GeodesicState start = state;
        geodesic.step_adaptive(state);
        state = normalize_poles(state);
        if (!std::isfinite(state.r) || !std::isfinite(state.theta) || !std::isfinite(state.phi)) {
          stop = "invalid_invariant";
          status = "invalid_invariant";
          break;
        }

        const double dl = proper_path_distance(metric, start, state);
        if (!(dl > 0.0) || !std::isfinite(dl)) {
          stop = "invalid_invariant";
          status = "invalid_invariant";
          break;
        }

        const double null_abs = std::abs(2.0 * geodesic.hamiltonian(state));
        const double energy_error = std::abs((-state.pt) - (-p_t0)) / std::max(std::abs(-p_t0), 1.0);
        const double lz_error = std::abs(state.pphi - p_phi0);
        local_null_max = std::max(local_null_max, null_abs);
        local_energy_error_max = std::max(local_energy_error_max, energy_error);
        local_lz_error_max = std::max(local_lz_error_max, lz_error);
        if (null_abs > config.null_invariant_tolerance || energy_error > config.killing_energy_tolerance || lz_error > config.lz_tolerance) {
          stop = "invalid_invariant";
          status = "invalid_invariant";
          break;
        }

        const GeodesicState current_rhs = geodesic.rhs(state);
        local_curvature = std::max({
            local_curvature,
            std::abs(current_rhs.r - previous_rhs.r),
            std::abs(current_rhs.theta - previous_rhs.theta),
            std::abs(current_rhs.phi - previous_rhs.phi),
        });
        previous_rhs = current_rhs;
        const double dphi_total = std::atan2(std::sin(state.phi - initial.phi), std::cos(state.phi - initial.phi));
        local_dtheta_max = std::max(local_dtheta_max, std::abs(state.theta - initial.theta));
        local_dphi_max = std::max(local_dphi_max, std::abs(dphi_total));

        const double rmid = 0.5 * (start.r + state.r);
        const double thmid = 0.5 * (start.theta + state.theta);
        const double phimid = start.phi + 0.5 * std::atan2(std::sin(state.phi - start.phi), std::cos(state.phi - start.phi));
        const GeodesicState mid{
            0.5 * (start.t + state.t),
            rmid,
            thmid,
            phimid,
            0.5 * (start.pt + state.pt),
            0.5 * (start.pr + state.pr),
            0.5 * (start.ptheta + state.ptheta),
            0.5 * (start.pphi + state.pphi),
        };

        segments << "{\"E_nu_local_gev_mid\":" << zamo_energy_gev(metric, mid, sample.energy_gev)
                 << ",\"dl_segment_rg\":" << dl
                 << ",\"event_id\":" << quote(sample.event_id)
                 << ",\"full_kerr_geodesic\":true,\"geodesic_backend\":\"cpp_hadros_original_port\",\"geodesic_status\":" << quote(status)
                 << ",\"p_phi_mid\":" << mid.pphi * sample.energy_gev
                 << ",\"p_r_mid\":" << mid.pr * sample.energy_gev
                 << ",\"p_t_mid\":" << mid.pt * sample.energy_gev
                 << ",\"p_theta_mid\":" << mid.ptheta * sample.energy_gev
                 << ",\"phi_end_rad\":" << state.phi
                 << ",\"phi_mid_rad\":" << phimid
                 << ",\"phi_start_rad\":" << start.phi
                 << ",\"r_end_rg\":" << state.r
                 << ",\"r_mid_rg\":" << rmid
                 << ",\"r_start_rg\":" << start.r
                 << ",\"segment_index\":" << segment_index
                 << ",\"source_sample_id\":" << sample.source_sample_id
                 << ",\"theta_end_rad\":" << state.theta
                 << ",\"theta_mid_rad\":" << thmid
                 << ",\"theta_phi_evolution\":true,\"theta_start_rad\":" << start.theta
                 << ",\"uses_christoffel_or_hamiltonian\":true,\"uses_hamiltonian\":true,\"uses_kerr_metric\":true}\n";
        ++n_segments;
        ++local_segments;
      }

      const bool valid = local_segments > 0
          && stop != "invalid_invariant"
          && local_null_max <= config.null_invariant_tolerance
          && local_energy_error_max <= config.killing_energy_tolerance
          && local_lz_error_max <= config.lz_tolerance;
      all_valid = all_valid && valid;
      ++stop_counts[stop];
      null_max = std::max(null_max, local_null_max);
      eerr_max = std::max(eerr_max, local_energy_error_max);
      lzerr_max = std::max(lzerr_max, local_lz_error_max);
      dtheta_max = std::max(dtheta_max, local_dtheta_max);
      dphi_max = std::max(dphi_max, local_dphi_max);
      curvature_max = std::max(curvature_max, local_curvature);

      paths << "{" << backend_json_fields()
            << ",\"coordinate_radial_preview\":false,\"curvature_indicator_max\":" << local_curvature
            << ",\"direction_generator\":" << quote(sample.direction_generator)
            << ",\"direction_model\":" << quote(sample.direction_model)
            << ",\"emission_direction\":{\"direction_generator\":" << quote(sample.direction_generator)
            << ",\"direction_model\":" << quote(sample.direction_model)
            << ",\"direction_local_components\":{\"basis\":" << quote(sample.basis)
            << ",\"n_r\":" << sample.n_r << ",\"n_theta\":" << sample.n_theta << ",\"n_phi\":" << sample.n_phi
            << ",\"dr\":" << sample.dr << ",\"dtheta\":" << sample.dtheta << ",\"dphi\":" << sample.dphi
            << "}},\"event_id\":" << quote(sample.event_id)
            << ",\"expensive_event_generation_invoked\":false,\"forward_backend\":\"cpp_hadros_original_port\""
            << ",\"forward_geodesics_consumes_source_direction\":true,\"four_momentum_constructed_from_source_direction\":true,\"four_momentum_sampled_in_source\":false"
            << ",\"full_kerr_geodesic\":true,\"geodesic_backend\":\"cpp_hadros_original_port\",\"geodesic_status\":" << quote(status)
            << ",\"initial_momentum\":{\"direction_generator\":" << quote(sample.direction_generator)
            << ",\"direction_model\":" << quote(sample.direction_model)
            << ",\"energy_gev\":" << sample.energy_gev
            << ",\"four_momentum\":{\"basis\":\"Boyer-Lindquist_covariant_from_ZAMO_orthonormal\""
            << ",\"p_phi\":" << initial.pphi * sample.energy_gev
            << ",\"p_r\":" << initial.pr * sample.energy_gev
            << ",\"p_t\":" << initial.pt * sample.energy_gev
            << ",\"p_theta\":" << initial.ptheta * sample.energy_gev
            << "},\"generator\":\"KerrNullMomentumGenerator\",\"killing_energy_gev\":" << -initial.pt * sample.energy_gev
            << ",\"local_tetrad\":\"ZAMO\",\"lz\":" << initial.pphi * sample.energy_gev
            << ",\"momentum_is_physical_kerr\":true,\"status\":\"physical_kerr_null_momentum_for_H3_W6_forward_geodesics\"}"
            << ",\"initial_position\":{\"phi_rad\":" << sample.phi << ",\"r_rg\":" << sample.r << ",\"t\":0,\"theta_rad\":" << sample.theta << "}"
            << ",\"killing_energy_max_error\":" << local_energy_error_max
            << ",\"lz_max_error\":" << local_lz_error_max
            << ",\"max_delta_phi_rad\":" << local_dphi_max
            << ",\"max_delta_theta_rad\":" << local_dtheta_max
            << ",\"momentum_generator\":\"KerrNullMomentumGenerator\",\"momentum_is_physical_kerr\":true"
            << ",\"n_segments\":" << local_segments
            << ",\"null_norm_max_abs\":" << local_null_max
            << ",\"observer_bridge_active_filter_invoked\":false,\"optical_depth_dis_sampler_invoked\":false"
            << ",\"source_sample_id\":" << sample.source_sample_id
            << ",\"stop_condition\":" << quote(stop)
            << ",\"theta_phi_evolution\":true,\"validation_pass\":" << (valid ? "true" : "false")
            << "}\n";
    }

    const fs::path summary_json = out_dir / "uhe_neutrino_forward_summary.json";
    const fs::path summary_csv = out_dir / "uhe_neutrino_forward_summary.csv";
    const fs::path validation_json = out_dir / "geodesic_validation_report.json";
    const fs::path stop_csv = out_dir / "stop_condition_statistics.csv";

    std::ofstream summary(summary_json);
    summary << std::setprecision(17)
            << "{" << backend_json_fields()
            << ",\"coordinate_radial_preview\":false,\"curvature_indicator_max\":" << curvature_max
            << ",\"direction_generator\":" << quote(first_direction_generator)
            << ",\"direction_model\":" << quote(first_direction_model)
            << ",\"expensive_event_generation_invoked\":false,\"forward_backend\":\"cpp_hadros_original_port\",\"forward_neutrino_geodesics_invoked\":true"
            << ",\"four_momentum_constructed_from_source_direction\":true,\"four_momentum_sampled_in_source\":false,\"forward_geodesics_consumes_source_direction\":true"
            << ",\"full_kerr_geodesic\":true,\"geodesic_backend\":\"cpp_hadros_original_port\",\"horizon_tolerance_rg\":" << config.horizon_tolerance_rg
            << ",\"initial_step_rg\":" << config.initial_step_rg << ",\"input_source_samples\":" << quote(source_path.string())
            << ",\"killing_energy_max_error\":" << eerr_max << ",\"killing_energy_tolerance\":" << config.killing_energy_tolerance
            << ",\"lz_max_error\":" << lzerr_max << ",\"lz_tolerance\":" << config.lz_tolerance
            << ",\"max_delta_phi_rad\":" << dphi_max << ",\"max_delta_theta_rad\":" << dtheta_max
            << ",\"max_steps\":" << config.max_steps
            << ",\"momentum_generator\":\"KerrNullMomentumGenerator\",\"momentum_is_physical_kerr\":true"
            << ",\"n_input_samples\":" << n_input_samples
            << ",\"n_paths\":" << all_samples.size()
            << ",\"n_samples_propagated\":" << all_samples.size()
            << ",\"n_samples_requested\":" << config.n_samples_to_propagate
            << ",\"n_segments\":" << n_segments
            << ",\"null_invariant_tolerance\":" << config.null_invariant_tolerance
            << ",\"null_norm_max\":" << null_max
            << ",\"observer_bridge_active_filter_invoked\":false,\"optical_depth_dis_sampler_invoked\":false"
            << ",\"outer_radius_rg\":" << config.outer_radius_rg
            << ",\"ported_hadros_files\":["
            << "\"include/geodesic_state.hpp\","
            << "\"include/kerr_metric.hpp\","
            << "\"include/kerr_metric_derivatives.hpp\","
            << "\"include/kerr_geodesic.hpp\","
            << "\"include/hadros/cascade/kerr_local_tetrad.hpp\","
            << "\"include/hadros/cascade/packet_kerr_null_propagator.hpp\","
            << "\"include/hadros/cascade/types.hpp\","
            << "\"src/kerr_metric.cpp\","
            << "\"src/kerr_geodesic.cpp\","
            << "\"src/cascade/kerr_local_tetrad.cpp\","
            << "\"src/cascade/packet_kerr_null_propagator.cpp\"]"
            << ",\"products\":{\"forward_path_segments\":" << quote((out_dir / "uhe_neutrino_forward_path_segments.jsonl").string())
            << ",\"forward_paths\":" << quote((out_dir / "uhe_neutrino_forward_paths.jsonl").string())
            << ",\"forward_summary\":" << quote(summary_csv.string())
            << ",\"forward_summary_json\":" << quote(summary_json.string())
            << ",\"geodesic_validation_report\":" << quote(validation_json.string())
            << ",\"stop_condition_statistics\":" << quote(stop_csv.string()) << "}"
            << ",\"status\":" << quote(all_valid ? "ok" : "validation_failed")
            << ",\"stop_condition_counts\":{";
    bool first = true;
    for (const auto& kv : stop_counts) {
      if (!first) summary << ",";
      summary << quote(kv.first) << ":" << kv.second;
      first = false;
    }
    summary << "},\"theta_phi_evolution\":true,\"validation_errors\":[],\"validation_pass\":"
            << (all_valid ? "true" : "false") << "}\n";

    std::ofstream csv(summary_csv);
    csv << "status,geodesic_backend,n_samples_requested,n_samples_propagated,n_paths,n_segments,direction_model,momentum_generator,momentum_is_physical_kerr,full_kerr_geodesic,theta_phi_evolution,uses_kerr_metric,uses_christoffel_or_hamiltonian,coordinate_radial_preview,max_delta_theta_rad,max_delta_phi_rad,curvature_indicator_max,null_norm_max,killing_energy_max_error,lz_max_error,validation_pass\n";
    csv << (all_valid ? "ok" : "validation_failed") << ",cpp_hadros_original_port," << config.n_samples_to_propagate << ","
        << all_samples.size() << "," << all_samples.size() << "," << n_segments << "," << first_direction_model
        << ",KerrNullMomentumGenerator,true,true,true,true,true,false," << dtheta_max << "," << dphi_max << ","
        << curvature_max << "," << null_max << "," << eerr_max << "," << lzerr_max << "," << (all_valid ? "true" : "false") << "\n";

    std::ofstream validation(validation_json);
    validation << std::setprecision(17)
               << "{\"killing_energy_max_error\":" << eerr_max
               << ",\"lz_max_error\":" << lzerr_max
               << ",\"null_norm_max\":" << null_max
               << ",\"validation_pass\":" << (all_valid ? "true" : "false") << "}\n";
    write_stop_csv(stop_csv, stop_counts, all_samples.size());
    return all_valid ? 0 : 1;
  } catch (const std::exception& exc) {
    std::cerr << "hadros3_forward_geodesics: " << exc.what() << "\n";
    return 1;
  }
}
