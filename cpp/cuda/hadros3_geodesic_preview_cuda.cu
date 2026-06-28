#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef HADROS_CUDA_PREVIEW_GLFW
#include <GLFW/glfw3.h>
#endif

namespace fs = std::filesystem;

namespace {

constexpr double PI = 3.141592653589793238462643383279502884;

fs::path preview_output_dir()
{
    const char* env = std::getenv("HADROS_PREVIEW_OUTPUT_DIR");
    if (env && *env) return fs::path(env);
    return fs::path("output/camera_preview");
}

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::ostringstream oss__;                                          \
            oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__          \
                  << ": " << cudaGetErrorString(err__);                       \
            throw std::runtime_error(oss__.str());                             \
        }                                                                      \
    } while (0)

struct Rgb {
    std::uint8_t r = 0;
    std::uint8_t g = 0;
    std::uint8_t b = 0;
};
static_assert(sizeof(Rgb) == 3, "Rgb must match packed PPM RGB bytes");

struct PreviewParams {
    double spin = 0.0001;
    double requested_spin = 0.0001;
    double r_obs_rg = 60.0;
    double theta_obs_rad = 80.0 * PI / 180.0;
    double phi_obs_rad = 0.0;
    double fov_rad = 25.0 * PI / 180.0;
    double aspect_ratio = 16.0 / 9.0;
    double r_max_rg = 80.0;
    double step_size = 0.45;
    double horizon_eps = 0.02;
    double disk_r_min_rg = 3.0;
    double disk_r_max_rg = 80.0;
    double disk_thickness_rg = 0.2;
    double near_clip_rg = 1.0;
    double torus_r0_rg = 10.0;
    double torus_sigma_r_rg = 5.0;
    double torus_h_rg = 2.0;
    double torus_alpha = 0.16;
    double torus_brightness = 1.0;
    double torus_max_alpha_step = 0.14;
    double torus_emissivity_cutoff = 1.0e-4;
    double funnel_theta_rad = 20.0 * PI / 180.0;
    double funnel_sigma_theta_rad = 4.0 * PI / 180.0;
    double funnel_alpha = 0.13;
    double funnel_brightness = 1.0;
    double funnel_emissivity_cutoff = 1.0e-4;
    int max_steps = 1200;
    int nx = 256;
    int ny = 144;
    int final_nx = 256;
    int final_ny = 144;
    int interactive_nx = 128;
    int interactive_ny = 72;
    int nav_mode = 5;
    int disk_geometry = 0;
    int disk_hit_mode = 0;
    int aspect_mode = 1;
    int sky_mode = 0;
    int sky_texture_width = 0;
    int sky_texture_height = 0;
    int funnel_enabled = 0;
    int opaque_structures = 0;
    int geodesic_model = 0;
    int spin_convention = 1;
    int allow_expensive_preview = 0;
    double still_refine_delay_s = 0.40;
    double rot_speed_rad_s = 90.0 * PI / 180.0;
    double zoom_speed_rg_s = 80.0;
    double fov_speed_rad_s = 80.0 * PI / 180.0;
};

enum PreviewNavMode {
    NAV_DETAILED = 0,
    NAV_SHADOW_DISK = 1,
    NAV_DISK_RADIUS_DEBUG = 2,
    NAV_HIT_REASON = 3,
    NAV_HIT_DISTANCE_DEBUG = 4,
    NAV_CELESTIAL_PLUS_TORUS_VOLUME = 5,
    NAV_TORUS_VOLUME = 6,
    NAV_VOLUME_EMISSIVITY_DEBUG = 7,
    NAV_FIRST_HIT_DISK_DEBUG = 8,
    NAV_OPAQUE_DISK_DEBUG = NAV_FIRST_HIT_DISK_DEBUG,
    NAV_PAINT_SWATCH_DISK = 9
};

enum PreviewDiskGeometry {
    DISK_THIN = 0,
    DISK_THICK_TORUS = 1
};

enum PreviewDiskHitMode {
    DISK_FIRST_HIT = 0,
    DISK_TRANSPARENT_OVERLAY = 1
};

enum PreviewAspectMode {
    ASPECT_FIXED = 0,
    ASPECT_WINDOW = 1
};

enum PreviewSkyMode {
    SKY_PROCEDURAL = 0,
    SKY_TEXTURE = 1,
    SKY_INTERSTELLAR_COORDINATE_GRID = 2
};

enum PreviewGeodesicModel {
    GEODESIC_KERR_LIKE = 0,
    GEODESIC_FULL_KERR = 1
};

enum PreviewSpinConvention {
    SPIN_CONVENTION_HADROS = 0,
    SPIN_CONVENTION_THORNE = 1
};

struct SkyTexture {
    int width = 0;
    int height = 0;
    std::vector<Rgb> pixels;

    bool loaded() const
    {
        return width > 0 && height > 0 && !pixels.empty();
    }
};

struct HitInfo {
    int klass = 2;
    int reason = 2;
    int step = 0;
    double path_length = 0.0;
    double r_cyl = 0.0;
    double z = 0.0;
    double intensity = 0.0;
};

struct PerfResult {
    double seconds = 0.0;
    double fps = 0.0;
    double kernel_ms = 0.0;
    double copy_ms = 0.0;
    double upload_ms = 0.0;
    double frame_ms = 0.0;
};

struct LatencyStats {
    double input_poll_ms = 0.0;
    double camera_update_ms = 0.0;
    double cuda_kernel_ms = 0.0;
    double cuda_copy_ms = 0.0;
    double gl_texture_upload_ms = 0.0;
    double draw_quad_ms = 0.0;
    double glfw_swap_buffers_ms = 0.0;
    double total_loop_ms = 0.0;
    double camera_to_texture_ms = 0.0;
    double fps = 0.0;
    bool camera_dirty = false;
    bool rendered = false;
};

struct Counts {
    int shadow = 0;
    int disk = 0;
    int sky = 0;
};

double fixed_aspect_ratio(const PreviewParams& p);
double vertical_fov_rad(double fov_x_rad, double aspect);
int height_for_aspect(int width, double aspect);
std::string aspect_label(double aspect);

struct CudaPreviewBuffers {
    Rgb* d_pixels = nullptr;
    Rgb* d_sky_pixels = nullptr;
    unsigned char* d_classes = nullptr;
    double* d_hit_distances = nullptr;
    cudaEvent_t start_event = nullptr;
    cudaEvent_t stop_event = nullptr;
    int nx = 0;
    int ny = 0;
    int sky_width = 0;
    int sky_height = 0;

    void allocate(int width, int height)
    {
        release_frame();
        nx = width;
        ny = height;
        CUDA_CHECK(cudaMalloc(&d_pixels, static_cast<std::size_t>(nx) * ny * sizeof(Rgb)));
        CUDA_CHECK(cudaMalloc(&d_classes, static_cast<std::size_t>(nx) * ny * sizeof(unsigned char)));
        CUDA_CHECK(cudaMalloc(&d_hit_distances, static_cast<std::size_t>(nx) * ny * sizeof(double)));
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    void upload_sky(const SkyTexture& sky)
    {
        if (!sky.loaded()) {
            if (d_sky_pixels) CUDA_CHECK(cudaFree(d_sky_pixels));
            d_sky_pixels = nullptr;
            sky_width = 0;
            sky_height = 0;
            return;
        }
        if (d_sky_pixels && sky_width == sky.width && sky_height == sky.height) return;
        if (d_sky_pixels) CUDA_CHECK(cudaFree(d_sky_pixels));
        sky_width = sky.width;
        sky_height = sky.height;
        CUDA_CHECK(cudaMalloc(&d_sky_pixels, sky.pixels.size() * sizeof(Rgb)));
        CUDA_CHECK(cudaMemcpy(
            d_sky_pixels,
            sky.pixels.data(),
            sky.pixels.size() * sizeof(Rgb),
            cudaMemcpyHostToDevice
        ));
    }

    void release_frame()
    {
        if (d_pixels) CUDA_CHECK(cudaFree(d_pixels));
        if (d_classes) CUDA_CHECK(cudaFree(d_classes));
        if (d_hit_distances) CUDA_CHECK(cudaFree(d_hit_distances));
        if (start_event) CUDA_CHECK(cudaEventDestroy(start_event));
        if (stop_event) CUDA_CHECK(cudaEventDestroy(stop_event));
        d_pixels = nullptr;
        d_classes = nullptr;
        d_hit_distances = nullptr;
        start_event = nullptr;
        stop_event = nullptr;
        nx = 0;
        ny = 0;
    }

    void release()
    {
        release_frame();
        if (d_sky_pixels) CUDA_CHECK(cudaFree(d_sky_pixels));
        d_sky_pixels = nullptr;
        sky_width = 0;
        sky_height = 0;
    }

    ~CudaPreviewBuffers()
    {
        release();
    }
};

__host__ __device__ double clampd(double value, double lo, double hi)
{
    return value < lo ? lo : (value > hi ? hi : value);
}

__host__ __device__ double3 add3(double3 a, double3 b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ double3 mul3(double3 a, double s)
{
    return make_double3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ double dot3(double3 a, double3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ double norm3(double3 a)
{
    return sqrt(dot3(a, a));
}

__host__ __device__ double3 normalize3(double3 a)
{
    const double n = fmax(norm3(a), 1.0e-30);
    return mul3(a, 1.0 / n);
}

__host__ __device__ double3 cross3(double3 a, double3 b)
{
    return make_double3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__host__ __device__ double horizon_radius(double spin)
{
    const double a = clampd(fabs(spin), 0.0, 0.999);
    return 1.0 + sqrt(fmax(0.0, 1.0 - a * a));
}

__host__ __device__ Rgb procedural_sky(double3 pos)
{
    const double r = fmax(norm3(pos), 1.0e-30);
    double lon = atan2(pos.y, pos.x);
    if (lon < 0.0) lon += 2.0 * PI;
    const double lat = asin(clampd(pos.z / r, -1.0, 1.0));

    const bool grid =
        fabs(sin(12.0 * lon)) < 0.055 ||
        fabs(sin(12.0 * (lat + 0.5 * PI))) < 0.055;
    if (grid) return {245, 248, 255};
    if (fabs(lat) < 0.04) return {238, 116, 58};
    if (fabs(lon) < 0.04 || fabs(lon - 0.5 * PI) < 0.04 || fabs(lon - PI) < 0.04) {
        return {80, 168, 255};
    }

    const double t = 0.5 + 0.5 * sin(3.0 * lon + 2.0 * lat);
    return {
        static_cast<std::uint8_t>(18 + 45 * t),
        static_cast<std::uint8_t>(32 + 55 * t),
        static_cast<std::uint8_t>(75 + 95 * t)
    };
}

__host__ __device__ Rgb blend_rgb(Rgb a, Rgb b, double t);

__host__ __device__ Rgb interstellar_coordinate_grid_sky(double3 pos)
{
    const double r = fmax(norm3(pos), 1.0e-30);
    double lon = atan2(pos.y, pos.x);
    if (lon < 0.0) lon += 2.0 * PI;
    const double theta = acos(clampd(pos.z / r, -1.0, 1.0));
    const double u = lon / (2.0 * PI);
    const double v = theta / PI;

    const int nx = 18;
    const int ny = 8;
    const double gx = u * nx;
    const double gy = v * ny;
    const int ix = static_cast<int>(floor(gx));
    const int iy = static_cast<int>(floor(gy));
    const double fx = gx - floor(gx);
    const double fy = gy - floor(gy);

    const Rgb column_palette[18] = {
        {25, 130, 55},    // green
        {92, 130, 70},    // grey green
        {142, 128, 38},   // ochre
        {192, 31, 28},    // red
        {198, 0, 70},     // crimson
        {190, 0, 132},    // magenta
        {153, 22, 155},   // purple
        {106, 44, 164},   // violet
        {58, 44, 150},    // indigo
        {18, 26, 128},    // deep blue
        {17, 86, 160},    // blue
        {0, 142, 174},    // cyan
        {0, 160, 132},    // teal
        {0, 154, 86},     // emerald
        {0, 142, 55},     // green
        {28, 132, 60},    // muted green
        {68, 148, 82},    // light green
        {92, 166, 108}    // pale green
    };
    Rgb base = column_palette[max(0, min(nx - 1, ix))];

    const double row = clampd((static_cast<double>(iy) + 0.5) / ny, 0.0, 1.0);
    const double top_desaturate = clampd(0.78 - 0.95 * row, 0.0, 0.50);
    const double top_darken = 0.78 + 0.30 * row;
    const Rgb grey = {122, 130, 124};
    base = blend_rgb(base, grey, top_desaturate);
    base = {
        static_cast<std::uint8_t>(clampd(base.r * top_darken, 0.0, 255.0)),
        static_cast<std::uint8_t>(clampd(base.g * top_darken, 0.0, 255.0)),
        static_cast<std::uint8_t>(clampd(base.b * top_darken, 0.0, 255.0))
    };

    const bool cell_line = fx < 0.050 || fx > 0.950 || fy < 0.070 || fy > 0.930;
    const bool major_lon = (ix % 3 == 0) && fx < 0.080;
    const bool major_lat = (iy % 2 == 0) && fy < 0.100;
    if (major_lon || major_lat) return {252, 252, 247};
    if (cell_line) return {232, 234, 226};
    return base;
}

__host__ __device__ Rgb cheap_sky(double3 pos)
{
    const double r = fmax(norm3(pos), 1.0e-30);
    const double z = clampd(pos.z / r, -1.0, 1.0);
    const double t = 0.5 + 0.5 * z;
    return {
        static_cast<std::uint8_t>(18 + 28 * t),
        static_cast<std::uint8_t>(32 + 45 * t),
        static_cast<std::uint8_t>(72 + 82 * t)
    };
}

__host__ __device__ Rgb blend_rgb(Rgb a, Rgb b, double t)
{
    const double u = clampd(t, 0.0, 1.0);
    return {
        static_cast<std::uint8_t>(clampd((1.0 - u) * a.r + u * b.r, 0.0, 255.0)),
        static_cast<std::uint8_t>(clampd((1.0 - u) * a.g + u * b.g, 0.0, 255.0)),
        static_cast<std::uint8_t>(clampd((1.0 - u) * a.b + u * b.b, 0.0, 255.0))
    };
}

__host__ __device__ Rgb average_rgb(Rgb a, Rgb b)
{
    return {
        static_cast<std::uint8_t>((static_cast<int>(a.r) + static_cast<int>(b.r)) / 2),
        static_cast<std::uint8_t>((static_cast<int>(a.g) + static_cast<int>(b.g)) / 2),
        static_cast<std::uint8_t>((static_cast<int>(a.b) + static_cast<int>(b.b)) / 2)
    };
}

__host__ __device__ Rgb sample_sky_texture_bilinear(
    double lon,
    double theta,
    const Rgb* sky_pixels,
    int width,
    int height
)
{
    if (!sky_pixels || width <= 0 || height <= 0) return {0, 0, 0};
    const double fx = (lon / (2.0 * PI)) * static_cast<double>(width) - 0.5;
    const double fy = (theta / PI) * static_cast<double>(height - 1);
    const int x0_raw = static_cast<int>(floor(fx));
    const int y0 = max(0, min(height - 1, static_cast<int>(floor(fy))));
    const int x0 = ((x0_raw % width) + width) % width;
    const int x1 = (x0 + 1) % width;
    const int y1 = max(0, min(height - 1, y0 + 1));
    const double tx = fx - floor(fx);
    const double ty = fy - floor(fy);
    const Rgb c00 = sky_pixels[y0 * width + x0];
    const Rgb c10 = sky_pixels[y0 * width + x1];
    const Rgb c01 = sky_pixels[y1 * width + x0];
    const Rgb c11 = sky_pixels[y1 * width + x1];
    const double w00 = (1.0 - tx) * (1.0 - ty);
    const double w10 = tx * (1.0 - ty);
    const double w01 = (1.0 - tx) * ty;
    const double w11 = tx * ty;
    return {
        static_cast<std::uint8_t>(clampd(w00 * c00.r + w10 * c10.r + w01 * c01.r + w11 * c11.r, 0.0, 255.0)),
        static_cast<std::uint8_t>(clampd(w00 * c00.g + w10 * c10.g + w01 * c01.g + w11 * c11.g, 0.0, 255.0)),
        static_cast<std::uint8_t>(clampd(w00 * c00.b + w10 * c10.b + w01 * c01.b + w11 * c11.b, 0.0, 255.0))
    };
}

__host__ __device__ Rgb sample_sky_texture(double3 pos, const Rgb* sky_pixels, int width, int height)
{
    if (!sky_pixels || width <= 0 || height <= 0) return procedural_sky(pos);
    const double r = fmax(norm3(pos), 1.0e-30);
    double lon = atan2(pos.y, pos.x);
    if (lon < 0.0) lon += 2.0 * PI;
    const double theta = acos(clampd(pos.z / r, -1.0, 1.0));
    const Rgb base = sample_sky_texture_bilinear(lon, theta, sky_pixels, width, height);
    const double seam_width = 3.0 * PI / 180.0;
    const double seam_dist = fmin(lon, 2.0 * PI - lon);
    if (seam_dist >= seam_width) return base;
    const Rgb left = sample_sky_texture_bilinear(2.0 * PI - seam_width, theta, sky_pixels, width, height);
    const Rgb right = sample_sky_texture_bilinear(seam_width, theta, sky_pixels, width, height);
    const Rgb seam = average_rgb(left, right);
    const double x = clampd(seam_dist / seam_width, 0.0, 1.0);
    const double smooth = x * x * (3.0 - 2.0 * x);
    return blend_rgb(seam, base, smooth);
}

__host__ __device__ Rgb celestial_sky(
    const PreviewParams& p,
    double3 pos,
    const Rgb* sky_pixels,
    int sky_width,
    int sky_height
)
{
    if (p.nav_mode == NAV_SHADOW_DISK) return cheap_sky(pos);
    if (p.sky_mode == SKY_TEXTURE) return sample_sky_texture(pos, sky_pixels, sky_width, sky_height);
    if (p.sky_mode == SKY_INTERSTELLAR_COORDINATE_GRID) return interstellar_coordinate_grid_sky(pos);
    return procedural_sky(pos);
}

__host__ __device__ Rgb disk_radius_color(double r_cyl, const PreviewParams& p)
{
    const double t = clampd((r_cyl - p.disk_r_min_rg) / fmax(p.disk_r_max_rg - p.disk_r_min_rg, 1.0e-9), 0.0, 1.0);
    return {
        static_cast<std::uint8_t>(40 + 215 * t),
        static_cast<std::uint8_t>(230 * (1.0 - fabs(2.0 * t - 1.0))),
        static_cast<std::uint8_t>(255 * (1.0 - t))
    };
}

__host__ __device__ Rgb paint_swatch_disk_color(double r_cyl, double phi, const PreviewParams& p)
{
    double lon = phi;
    if (lon < 0.0) lon += 2.0 * PI;
    const double u = lon / (2.0 * PI);
    const double rr = clampd(
        (r_cyl - p.disk_r_min_rg) / fmax(p.disk_r_max_rg - p.disk_r_min_rg, 1.0e-9),
        0.0,
        1.0
    );
    const int n_phi = 18;
    const int n_r = 8;
    const double gx = u * n_phi;
    const double gy = rr * n_r;
    const int ix = max(0, min(n_phi - 1, static_cast<int>(floor(gx))));
    const int iy = max(0, min(n_r - 1, static_cast<int>(floor(gy))));
    const double fx = gx - floor(gx);
    const double fy = gy - floor(gy);
    const Rgb column_palette[18] = {
        {25, 130, 55}, {92, 130, 70}, {142, 128, 38}, {192, 31, 28},
        {198, 0, 70}, {190, 0, 132}, {153, 22, 155}, {106, 44, 164},
        {58, 44, 150}, {18, 26, 128}, {17, 86, 160}, {0, 142, 174},
        {0, 160, 132}, {0, 154, 86}, {0, 142, 55}, {28, 132, 60},
        {68, 148, 82}, {92, 166, 108}
    };
    const double row = (static_cast<double>(iy) + 0.5) / n_r;
    const double top_desaturate = clampd(0.70 - 0.90 * row, 0.0, 0.45);
    const double brightness = 0.78 + 0.30 * row;
    Rgb base = blend_rgb(column_palette[ix], {122, 130, 124}, top_desaturate);
    base = {
        static_cast<std::uint8_t>(clampd(base.r * brightness, 0.0, 255.0)),
        static_cast<std::uint8_t>(clampd(base.g * brightness, 0.0, 255.0)),
        static_cast<std::uint8_t>(clampd(base.b * brightness, 0.0, 255.0))
    };
    const bool radial_line = fy < 0.075 || fy > 0.925;
    const bool azimuth_line = fx < 0.045 || fx > 0.955;
    const bool major_spoke = (ix % 3 == 0) && fx < 0.075;
    if (major_spoke || radial_line) return {252, 252, 246};
    if (azimuth_line) return {232, 234, 226};
    return base;
}

__host__ __device__ Rgb hit_distance_color(double distance, const PreviewParams& p)
{
    const double t = clampd((distance - p.near_clip_rg) / fmax(p.r_max_rg - p.near_clip_rg, 1.0e-9), 0.0, 1.0);
    return {
        static_cast<std::uint8_t>(255 * (1.0 - t)),
        static_cast<std::uint8_t>(80 + 175 * t),
        static_cast<std::uint8_t>(255 * t)
    };
}

__host__ __device__ Rgb hit_reason_color(int reason)
{
    if (reason == 0) return {0, 0, 0};        // horizon
    if (reason == 1) return {230, 130, 35};   // disk
    if (reason == 2) return {45, 120, 255};   // sky
    if (reason == 3) return {255, 0, 255};    // r_max exit
    return {255, 32, 32};                     // numerical/max-steps
}

__host__ __device__ bool volume_mode(int nav_mode)
{
    return nav_mode == NAV_DETAILED ||
           nav_mode == NAV_CELESTIAL_PLUS_TORUS_VOLUME ||
           nav_mode == NAV_TORUS_VOLUME ||
           nav_mode == NAV_VOLUME_EMISSIVITY_DEBUG;
}

__host__ __device__ double torus_emissivity(const PreviewParams& p, double3 pos)
{
    const double r_cyl = sqrt(pos.x * pos.x + pos.y * pos.y);
    const double dr = (r_cyl - p.torus_r0_rg) / fmax(p.torus_sigma_r_rg, 1.0e-9);
    const double dz = pos.z / fmax(p.torus_h_rg, 1.0e-9);
    return exp(-(dr * dr) - (dz * dz));
}

__host__ __device__ double funnel_emissivity(const PreviewParams& p, double3 pos)
{
    if (!p.funnel_enabled) return 0.0;
    const double r = fmax(norm3(pos), 1.0e-30);
    const double theta = acos(clampd(pos.z / r, -1.0, 1.0));
    const double theta0 = clampd(p.funnel_theta_rad, 1.0e-6, 0.5 * PI - 1.0e-6);
    const double d_north = fabs(theta - theta0);
    const double d_south = fabs((PI - theta) - theta0);
    const double polar = fmin(theta, PI - theta);
    const double d = fmin(d_north, d_south) / fmax(p.funnel_sigma_theta_rad, 1.0e-9);
    const double radial_fade = 1.0 / (1.0 + 0.015 * r * r);
    const double wall = exp(-(d * d));
    const double fill_width = fmax(0.75 * theta0, p.funnel_sigma_theta_rad);
    const double core = 0.35 * exp(-((polar / fill_width) * (polar / fill_width)));
    return fmax(wall, core) * radial_fade;
}

__host__ __device__ Rgb compose_volume_color(Rgb background, double cr, double cg, double cb, double alpha)
{
    const double inv = 1.0 - clampd(alpha, 0.0, 1.0);
    return {
        static_cast<std::uint8_t>(clampd(cr + inv * background.r, 0.0, 255.0)),
        static_cast<std::uint8_t>(clampd(cg + inv * background.g, 0.0, 255.0)),
        static_cast<std::uint8_t>(clampd(cb + inv * background.b, 0.0, 255.0))
    };
}

__host__ __device__ Rgb emissivity_debug_color(double intensity)
{
    const double t = clampd(intensity, 0.0, 1.0);
    return {
        static_cast<std::uint8_t>(255 * t),
        static_cast<std::uint8_t>(180 * sqrt(t)),
        static_cast<std::uint8_t>(40 * (1.0 - t))
    };
}

__host__ __device__ void accumulate_preview_volume_sample(
    const PreviewParams& p,
    double3 sample_pos,
    double ds,
    double* accum_r,
    double* accum_g,
    double* accum_b,
    double* accum_alpha,
    double* accum_intensity
)
{
    if (*accum_alpha >= 0.995) return;
    const double j = torus_emissivity(p, sample_pos);
    const double jf = funnel_emissivity(p, sample_pos);
    if (j > p.torus_emissivity_cutoff) {
        const double alpha_step = clampd(
            p.opaque_structures
                ? fmax(p.torus_alpha * p.torus_brightness * j * ds, 0.65 * j)
                : p.torus_alpha * p.torus_brightness * j * ds,
            0.0,
            p.opaque_structures ? 0.90 : p.torus_max_alpha_step
        );
        const double weight = (1.0 - *accum_alpha) * alpha_step;
        *accum_r += weight * 205.0;
        *accum_g += weight * 230.0;
        *accum_b += weight * 255.0;
        *accum_alpha += weight;
        *accum_intensity += j * ds * p.torus_brightness;
    }
    if (jf > p.funnel_emissivity_cutoff && *accum_alpha < 0.995) {
        const double alpha_step = clampd(
            p.opaque_structures
                ? fmax(p.funnel_alpha * p.funnel_brightness * jf * ds, 0.65 * jf)
                : p.funnel_alpha * p.funnel_brightness * jf * ds,
            0.0,
            p.opaque_structures ? 0.90 : p.torus_max_alpha_step
        );
        const double weight = (1.0 - *accum_alpha) * alpha_step;
        *accum_r += weight * 232.0;
        *accum_g += weight * 236.0;
        *accum_b += weight * 218.0;
        *accum_alpha += weight;
        *accum_intensity += jf * ds * p.funnel_brightness;
    }
}

__host__ __device__ void accumulate_preview_volume_segment(
    const PreviewParams& p,
    double3 pos,
    double3 dir,
    double h,
    double* accum_r,
    double* accum_g,
    double* accum_b,
    double* accum_alpha,
    double* accum_intensity
)
{
    const double r = norm3(pos);
    const int samples = r < 8.0 ? 5 : (r < 20.0 ? 3 : 2);
    const double ds = h / static_cast<double>(samples);
    for (int s = 0; s < samples; ++s) {
        const double u = (static_cast<double>(s) + 0.5) / static_cast<double>(samples);
        const double3 sample_pos = add3(pos, mul3(dir, h * u));
        accumulate_preview_volume_sample(
            p,
            sample_pos,
            ds,
            accum_r,
            accum_g,
            accum_b,
            accum_alpha,
            accum_intensity
        );
    }
}

__host__ __device__ double adaptive_step(const PreviewParams& p, double r)
{
    const double t = clampd((r - 6.0) / 34.0, 0.0, 1.0);
    const double smooth = t * t * (3.0 - 2.0 * t);
    return p.step_size * (1.0 + 3.0 * smooth);
}

struct KerrPreviewState {
    double t;
    double r;
    double theta;
    double phi;
    double pt;
    double pr;
    double ptheta;
    double pphi;
};

__device__ double pk_kerr_sigma(double r, double th, double a)
{
    const double c = cos(th);
    return r * r + a * a * c * c;
}

__device__ double pk_kerr_delta(double r, double a)
{
    return r * r - 2.0 * r + a * a;
}

__device__ double pk_kerr_big_a(double r, double th, double a)
{
    const double s = sin(th);
    const double s2 = s * s;
    const double rr_aa = r * r + a * a;
    return rr_aa * rr_aa - a * a * pk_kerr_delta(r, a) * s2;
}

__device__ double pk_kerr_horizon(double a)
{
    const double aa = clampd(fabs(a), 0.0, 0.999);
    return 1.0 + sqrt(fmax(1.0 - aa * aa, 0.0));
}

__device__ double pk_full_kerr_adaptive_step(const PreviewParams& p, const KerrPreviewState& y)
{
    const double r_h = pk_kerr_horizon(p.spin);
    const double r = fmax(y.r, r_h + 1.0e-5);
    double factor = 4.0;
    if (r < r_h + 0.25) factor = 0.75;
    else if (r < 2.2) factor = 0.85;
    else if (r < 4.0) factor = 0.90;
    else if (r < 8.0) factor = 1.00;
    else if (r < 16.0) factor = 1.70;
    else if (r < 35.0) factor = 2.70;

    const double polar = fmin(y.theta, PI - y.theta);
    if (polar < 0.02) factor = fmin(factor, 0.70);
    else if (polar < 0.08) factor = fmin(factor, 0.85);

    return clampd(p.step_size * factor, 0.22, 3.00);
}

__device__ void pk_kerr_metric(double r, double th, double a, double g[4][4])
{
    const double sig = pk_kerr_sigma(r, th, a);
    const double del = fmax(pk_kerr_delta(r, a), 1.0e-12);
    const double s = sin(th);
    const double s2 = s * s;
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) g[mu][nu] = 0.0;
    }
    g[0][0] = -(1.0 - 2.0 * r / sig);
    g[0][3] = -2.0 * a * r * s2 / sig;
    g[3][0] = g[0][3];
    g[1][1] = sig / del;
    g[2][2] = sig;
    g[3][3] = (r * r + a * a + 2.0 * a * a * r * s2 / sig) * s2;
}

__device__ void pk_kerr_inverse_metric(double r, double th, double a, double ginv[4][4])
{
    const double sig = pk_kerr_sigma(r, th, a);
    const double del = fmax(pk_kerr_delta(r, a), 1.0e-12);
    const double s = sin(th);
    const double s2 = fmax(s * s, 1.0e-10);
    const double bigA = pk_kerr_big_a(r, th, a);
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) ginv[mu][nu] = 0.0;
    }
    ginv[0][0] = -bigA / (sig * del);
    ginv[0][3] = -2.0 * a * r / (sig * del);
    ginv[3][0] = ginv[0][3];
    ginv[1][1] = del / sig;
    ginv[2][2] = 1.0 / sig;
    ginv[3][3] = (del - a * a * s2) / (sig * del * s2);
}

__device__ double pk_kerr_lapse(double r, double th, double a)
{
    const double sig = pk_kerr_sigma(r, th, a);
    const double del = fmax(pk_kerr_delta(r, a), 1.0e-12);
    const double bigA = pk_kerr_big_a(r, th, a);
    return sqrt(fmax(sig * del / bigA, 1.0e-300));
}

__device__ double pk_kerr_omega(double r, double th, double a)
{
    return 2.0 * a * r / pk_kerr_big_a(r, th, a);
}

__device__ double pk_wrapped_delta_phi(double phi_new, double phi_old)
{
    return remainder(phi_new - phi_old, 2.0 * PI);
}

__device__ void pk_normalize_polar_coordinate(KerrPreviewState* y)
{
    if (!y || !isfinite(y->theta) || !isfinite(y->phi) || !isfinite(y->ptheta)) return;
    if (y->theta < 0.0) {
        y->theta = -y->theta;
        y->phi += PI;
        y->ptheta = -y->ptheta;
    }
    if (y->theta > PI) {
        y->theta = 2.0 * PI - y->theta;
        y->phi += PI;
        y->ptheta = -y->ptheta;
    }
    y->theta = clampd(y->theta, 1.0e-4, PI - 1.0e-4);
    y->phi = remainder(y->phi, 2.0 * PI);
}

__device__ double pk_zamo_spatial_interval_rg(
    const PreviewParams& p,
    const KerrPreviewState& current,
    const KerrPreviewState& previous
)
{
    const double r_mid = 0.5 * (current.r + previous.r);
    const double theta_mid = clampd(0.5 * (current.theta + previous.theta), 1.0e-4, PI - 1.0e-4);
    double g[4][4];
    pk_kerr_metric(r_mid, theta_mid, p.spin, g);
    const double dr = current.r - previous.r;
    const double dtheta = current.theta - previous.theta;
    const double dphi = pk_wrapped_delta_phi(current.phi, previous.phi);
    const double dl2 = g[1][1] * dr * dr + g[2][2] * dtheta * dtheta + g[3][3] * dphi * dphi;
    return sqrt(fmax(dl2, 0.0));
}

__device__ double pk_quotient_derivative(double n, double dn, double d, double dd)
{
    return (dn * d - n * dd) / fmax(d * d, 1.0e-300);
}

__device__ void pk_kerr_inverse_metric_derivatives(
    double r,
    double th,
    double a,
    double dgdr[4][4],
    double dgdth[4][4]
)
{
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            dgdr[mu][nu] = 0.0;
            dgdth[mu][nu] = 0.0;
        }
    }

    const double s = sin(th);
    const double c = cos(th);
    const double s2 = fmax(s * s, 1.0e-10);
    const double ds2_dth = 2.0 * s * c;
    const double a2 = a * a;
    const double rr = r * r;
    const double rr_a2 = rr + a2;
    const double sig = pk_kerr_sigma(r, th, a);
    const double del = fmax(pk_kerr_delta(r, a), 1.0e-12);
    const double bigA = pk_kerr_big_a(r, th, a);

    const double dsig_dr = 2.0 * r;
    const double dsig_dth = -2.0 * a2 * s * c;
    const double ddel_dr = 2.0 * r - 2.0;
    const double dbigA_dr = 4.0 * r * rr_a2 - a2 * ddel_dr * s2;
    const double dbigA_dth = -a2 * del * ds2_dth;

    const double sig_del = sig * del;
    const double dsig_del_dr = dsig_dr * del + sig * ddel_dr;
    const double dsig_del_dth = dsig_dth * del;

    const double n_tt = -bigA;
    dgdr[0][0] = pk_quotient_derivative(n_tt, -dbigA_dr, sig_del, dsig_del_dr);
    dgdth[0][0] = pk_quotient_derivative(n_tt, -dbigA_dth, sig_del, dsig_del_dth);

    const double n_tphi = -2.0 * a * r;
    dgdr[0][3] = pk_quotient_derivative(n_tphi, -2.0 * a, sig_del, dsig_del_dr);
    dgdth[0][3] = pk_quotient_derivative(n_tphi, 0.0, sig_del, dsig_del_dth);
    dgdr[3][0] = dgdr[0][3];
    dgdth[3][0] = dgdth[0][3];

    dgdr[1][1] = pk_quotient_derivative(del, ddel_dr, sig, dsig_dr);
    dgdth[1][1] = pk_quotient_derivative(del, 0.0, sig, dsig_dth);

    dgdr[2][2] = -dsig_dr / fmax(sig * sig, 1.0e-300);
    dgdth[2][2] = -dsig_dth / fmax(sig * sig, 1.0e-300);

    const double n_phiphi = del - a2 * s2;
    const double den_phiphi = sig_del * s2;
    const double dden_phiphi_dr = dsig_del_dr * s2;
    const double dden_phiphi_dth = dsig_del_dth * s2 + sig_del * ds2_dth;
    dgdr[3][3] = pk_quotient_derivative(n_phiphi, ddel_dr, den_phiphi, dden_phiphi_dr);
    dgdth[3][3] = pk_quotient_derivative(n_phiphi, -a2 * ds2_dth, den_phiphi, dden_phiphi_dth);
}

__device__ KerrPreviewState pk_geodesic_rhs(const PreviewParams& p, const KerrPreviewState& y)
{
    const double theta = clampd(y.theta, 1.0e-4, PI - 1.0e-4);
    const double r = fmax(y.r, pk_kerr_horizon(p.spin) + 1.0e-5);
    double ginv[4][4];
    pk_kerr_inverse_metric(r, theta, p.spin, ginv);

    const double pcov[4] = {y.pt, y.pr, y.ptheta, y.pphi};
    KerrPreviewState dydl{};
    for (int nu = 0; nu < 4; ++nu) {
        dydl.t += ginv[0][nu] * pcov[nu];
        dydl.r += ginv[1][nu] * pcov[nu];
        dydl.theta += ginv[2][nu] * pcov[nu];
        dydl.phi += ginv[3][nu] * pcov[nu];
    }

    double dgdr[4][4];
    double dgdth[4][4];
    pk_kerr_inverse_metric_derivatives(r, theta, p.spin, dgdr, dgdth);
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            const double pp = pcov[mu] * pcov[nu];
            dydl.pr -= 0.5 * dgdr[mu][nu] * pp;
            dydl.ptheta -= 0.5 * dgdth[mu][nu] * pp;
        }
    }
    return dydl;
}

__device__ KerrPreviewState pk_add_scaled5(
    const KerrPreviewState& y,
    const KerrPreviewState& k1, double a1,
    const KerrPreviewState& k2, double a2,
    const KerrPreviewState& k3, double a3,
    const KerrPreviewState& k4, double a4,
    const KerrPreviewState& k5, double a5
)
{
    return {
        y.t + a1 * k1.t + a2 * k2.t + a3 * k3.t + a4 * k4.t + a5 * k5.t,
        y.r + a1 * k1.r + a2 * k2.r + a3 * k3.r + a4 * k4.r + a5 * k5.r,
        y.theta + a1 * k1.theta + a2 * k2.theta + a3 * k3.theta + a4 * k4.theta + a5 * k5.theta,
        y.phi + a1 * k1.phi + a2 * k2.phi + a3 * k3.phi + a4 * k4.phi + a5 * k5.phi,
        y.pt + a1 * k1.pt + a2 * k2.pt + a3 * k3.pt + a4 * k4.pt + a5 * k5.pt,
        y.pr + a1 * k1.pr + a2 * k2.pr + a3 * k3.pr + a4 * k4.pr + a5 * k5.pr,
        y.ptheta + a1 * k1.ptheta + a2 * k2.ptheta + a3 * k3.ptheta + a4 * k4.ptheta + a5 * k5.ptheta,
        y.pphi + a1 * k1.pphi + a2 * k2.pphi + a3 * k3.pphi + a4 * k4.pphi + a5 * k5.pphi
    };
}

__device__ double pk_state_error_norm(const KerrPreviewState& a, const KerrPreviewState& b)
{
    double err = 0.0;
    err = fmax(err, fabs(a.r - b.r));
    err = fmax(err, fabs(a.theta - b.theta));
    err = fmax(err, fabs(remainder(a.phi - b.phi, 2.0 * PI)));
    err = fmax(err, fabs(a.pr - b.pr));
    err = fmax(err, fabs(a.ptheta - b.ptheta));
    err = fmax(err, fabs(a.pphi - b.pphi));
    return err;
}

__device__ bool pk_rkf45_step_preview(
    const PreviewParams& p,
    KerrPreviewState& y,
    double h0,
    double tolerance,
    double* accepted_h
)
{
    double h = h0;
    const double h_min = fmax(0.02, h0 * 1.0e-3);
    KerrPreviewState best = y;

    for (int attempt = 0; attempt < 8; ++attempt) {
        const KerrPreviewState k1 = pk_geodesic_rhs(p, y);
        const KerrPreviewState k2 = pk_geodesic_rhs(p, pk_add_scaled5(y, k1, h * 1.0 / 4.0, KerrPreviewState{}, 0.0, KerrPreviewState{}, 0.0, KerrPreviewState{}, 0.0, KerrPreviewState{}, 0.0));
        const KerrPreviewState k3 = pk_geodesic_rhs(p, pk_add_scaled5(y, k1, h * 3.0 / 32.0, k2, h * 9.0 / 32.0, KerrPreviewState{}, 0.0, KerrPreviewState{}, 0.0, KerrPreviewState{}, 0.0));
        const KerrPreviewState k4 = pk_geodesic_rhs(p, pk_add_scaled5(y, k1, h * 1932.0 / 2197.0, k2, h * -7200.0 / 2197.0, k3, h * 7296.0 / 2197.0, KerrPreviewState{}, 0.0, KerrPreviewState{}, 0.0));
        const KerrPreviewState k5 = pk_geodesic_rhs(p, pk_add_scaled5(y, k1, h * 439.0 / 216.0, k2, h * -8.0, k3, h * 3680.0 / 513.0, k4, h * -845.0 / 4104.0, KerrPreviewState{}, 0.0));
        const KerrPreviewState k6 = pk_geodesic_rhs(p, pk_add_scaled5(y, k1, h * -8.0 / 27.0, k2, h * 2.0, k3, h * -3544.0 / 2565.0, k4, h * 1859.0 / 4104.0, k5, h * -11.0 / 40.0));

        KerrPreviewState y4 = pk_add_scaled5(
            y,
            k1, h * 25.0 / 216.0,
            k3, h * 1408.0 / 2565.0,
            k4, h * 2197.0 / 4104.0,
            k5, h * -1.0 / 5.0,
            KerrPreviewState{}, 0.0
        );
        KerrPreviewState y5 = pk_add_scaled5(
            y,
            k1, h * 16.0 / 135.0,
            k3, h * 6656.0 / 12825.0,
            k4, h * 28561.0 / 56430.0,
            k5, h * -9.0 / 50.0,
            k6, h * 2.0 / 55.0
        );
        pk_normalize_polar_coordinate(&y4);
        pk_normalize_polar_coordinate(&y5);
        best = y5;

        const double err = pk_state_error_norm(y4, y5);
        if (err < tolerance || h <= h_min) {
            y = y5;
            if (accepted_h) *accepted_h = h;
            return true;
        }

        const double factor = 0.8 * pow(tolerance / fmax(err, 1.0e-30), 0.20);
        h *= clampd(factor, 0.18, 2.0);
        h = fmax(h, h_min);
    }

    y = best;
    pk_normalize_polar_coordinate(&y);
    if (accepted_h) *accepted_h = h;
    return false;
}

__host__ __device__ bool thin_disk_hit(
    const PreviewParams& p,
    double3 prev_pos,
    double3 pos,
    double* hit_r_cyl,
    double* hit_z,
    double* hit_phi = nullptr
)
{
    const bool crossed_midplane = (prev_pos.z <= 0.0 && pos.z >= 0.0) || (prev_pos.z >= 0.0 && pos.z <= 0.0);
    const bool inside_thickness = fabs(pos.z) <= p.disk_thickness_rg;
    double hx = pos.x;
    double hy = pos.y;
    double hz = pos.z;

    if (crossed_midplane && fabs(pos.z - prev_pos.z) > 1.0e-12) {
        const double t = clampd(-prev_pos.z / (pos.z - prev_pos.z), 0.0, 1.0);
        hx = prev_pos.x + t * (pos.x - prev_pos.x);
        hy = prev_pos.y + t * (pos.y - prev_pos.y);
        hz = 0.0;
    } else if (!inside_thickness) {
        return false;
    }

    const double r_cyl = sqrt(hx * hx + hy * hy);
    if (hit_r_cyl) *hit_r_cyl = r_cyl;
    if (hit_z) *hit_z = hz;
    if (hit_phi) *hit_phi = atan2(hy, hx);
    return r_cyl >= p.disk_r_min_rg && r_cyl <= p.disk_r_max_rg;
}

__host__ __device__ bool thick_torus_hit(
    const PreviewParams& p,
    double3 pos,
    double* hit_r_cyl,
    double* hit_z,
    double* hit_phi = nullptr
)
{
    const double r_cyl = sqrt(pos.x * pos.x + pos.y * pos.y);
    const double tube = sqrt((r_cyl - p.torus_r0_rg) * (r_cyl - p.torus_r0_rg) + pos.z * pos.z);
    if (hit_r_cyl) *hit_r_cyl = r_cyl;
    if (hit_z) *hit_z = pos.z;
    if (hit_phi) *hit_phi = atan2(pos.y, pos.x);
    return tube <= p.torus_h_rg && r_cyl >= p.disk_r_min_rg && r_cyl <= p.disk_r_max_rg;
}

__host__ __device__ int trace_preview_pixel(
    const PreviewParams& p,
    int i,
    int j,
    Rgb* color_out,
    const Rgb* sky_pixels = nullptr,
    int sky_width = 0,
    int sky_height = 0,
    HitInfo* hit_out = nullptr
)
{
    const double theta = clampd(p.theta_obs_rad, 1.0e-6, PI - 1.0e-6);
    const double sin_t = sin(theta);
    const double cos_t = cos(theta);
    const double sin_p = sin(p.phi_obs_rad);
    const double cos_p = cos(p.phi_obs_rad);

    const double3 e_r = make_double3(sin_t * cos_p, sin_t * sin_p, cos_t);
    const double3 e_theta = make_double3(cos_t * cos_p, cos_t * sin_p, -sin_t);
    const double3 e_phi = make_double3(-sin_p, cos_p, 0.0);
    double3 pos = mul3(e_r, p.r_obs_rg);

    const double aspect = fmax(p.aspect_ratio, 1.0e-9);
    const double tan_half_fov_x = tan(0.5 * p.fov_rad);
    const double tan_half_fov_y = tan_half_fov_x / aspect;
    const double u = (2.0 * (static_cast<double>(i) + 0.5) / p.nx - 1.0) * tan_half_fov_x;
    const double v = (2.0 * (static_cast<double>(j) + 0.5) / p.ny - 1.0) * tan_half_fov_y;
    double3 dir = normalize3(add3(add3(mul3(e_r, -1.0), mul3(e_phi, u)), mul3(e_theta, v)));

    const double r_h = horizon_radius(p.spin);
    double3 prev_pos = pos;
    double path_length = 0.0;
    double accum_r = 0.0;
    double accum_g = 0.0;
    double accum_b = 0.0;
    double accum_alpha = 0.0;
    double accum_intensity = 0.0;
    const bool use_volume = volume_mode(p.nav_mode);
    HitInfo first_disk;
    bool has_disk_hit = false;
    int klass = 2;

    for (int step = 0; step < p.max_steps; ++step) {
        const double r = norm3(pos);
        if (r <= r_h + p.horizon_eps) {
            klass = 0;
            if (color_out) {
                if (p.nav_mode == NAV_HIT_REASON) *color_out = hit_reason_color(0);
                else if (p.nav_mode == NAV_VOLUME_EMISSIVITY_DEBUG) *color_out = emissivity_debug_color(accum_intensity);
                else *color_out = compose_volume_color({0, 0, 0}, accum_r, accum_g, accum_b, accum_alpha);
            }
            if (hit_out) *hit_out = {klass, 0, step, path_length, sqrt(pos.x * pos.x + pos.y * pos.y), pos.z, accum_intensity};
            return klass;
        }

        const double h = adaptive_step(p, r);
        if (use_volume && path_length > p.near_clip_rg && accum_alpha < 0.995) {
            accumulate_preview_volume_segment(
                p,
                pos,
                dir,
                h,
                &accum_r,
                &accum_g,
                &accum_b,
                &accum_alpha,
                &accum_intensity
            );
        }

        double hit_cyl = 0.0;
        double hit_z = 0.0;
        double hit_phi = 0.0;
        const bool past_near_clip = path_length > p.near_clip_rg;
        const bool disk_hit = !use_volume && past_near_clip && (
            p.disk_geometry == DISK_THICK_TORUS
                ? thick_torus_hit(p, pos, &hit_cyl, &hit_z, &hit_phi)
                : thin_disk_hit(p, prev_pos, pos, &hit_cyl, &hit_z, &hit_phi)
        );
        if (disk_hit && !has_disk_hit) {
            first_disk = {1, 1, step, path_length, hit_cyl, hit_z};
            has_disk_hit = true;
            if (p.disk_hit_mode == DISK_FIRST_HIT) {
                klass = 1;
                if (color_out) {
                    if (p.nav_mode == NAV_DISK_RADIUS_DEBUG) *color_out = disk_radius_color(hit_cyl, p);
                    else if (p.nav_mode == NAV_PAINT_SWATCH_DISK) *color_out = paint_swatch_disk_color(hit_cyl, hit_phi, p);
                    else if (p.nav_mode == NAV_HIT_DISTANCE_DEBUG) *color_out = hit_distance_color(path_length, p);
                    else if (p.nav_mode == NAV_HIT_REASON) *color_out = hit_reason_color(1);
                    else *color_out = {230, 130, 35};
                }
                if (hit_out) *hit_out = first_disk;
                return klass;
            }
        }

        if (r >= p.r_max_rg) {
            if (has_disk_hit && p.disk_hit_mode == DISK_TRANSPARENT_OVERLAY) {
                klass = 1;
                if (color_out) {
                    if (p.nav_mode == NAV_DISK_RADIUS_DEBUG) *color_out = disk_radius_color(first_disk.r_cyl, p);
                    else if (p.nav_mode == NAV_PAINT_SWATCH_DISK) *color_out = paint_swatch_disk_color(first_disk.r_cyl, 0.0, p);
                    else if (p.nav_mode == NAV_HIT_DISTANCE_DEBUG) *color_out = hit_distance_color(first_disk.path_length, p);
                    else if (p.nav_mode == NAV_HIT_REASON) *color_out = hit_reason_color(1);
                    else *color_out = {230, 130, 35};
                }
                if (hit_out) *hit_out = first_disk;
                return klass;
            }
            klass = 2;
            if (color_out) {
                if (p.nav_mode == NAV_HIT_REASON) *color_out = hit_reason_color(3);
                else if (p.nav_mode == NAV_VOLUME_EMISSIVITY_DEBUG) *color_out = emissivity_debug_color(accum_intensity);
                else {
                    const Rgb bg = p.nav_mode == NAV_TORUS_VOLUME
                        ? Rgb{0, 0, 0}
                        : celestial_sky(p, pos, sky_pixels, sky_width, sky_height);
                    *color_out = use_volume ? compose_volume_color(bg, accum_r, accum_g, accum_b, accum_alpha) : bg;
                }
            }
            if (hit_out) *hit_out = {klass, 3, step, path_length, sqrt(pos.x * pos.x + pos.y * pos.y), pos.z, accum_intensity};
            return klass;
        }

        const double inv_r3 = 1.0 / fmax(r * r * r, 1.0e-9);
        const double3 gravity = mul3(pos, -2.0 * inv_r3);
        const double3 z_axis = make_double3(0.0, 0.0, 1.0);
        const double3 drag = mul3(cross3(z_axis, dir), 0.10 * p.spin * inv_r3);

        const double3 acc1 = add3(gravity, drag);
        const double3 mid_dir = normalize3(add3(dir, mul3(acc1, 0.5 * h)));
        const double3 mid_pos = add3(pos, mul3(dir, 0.5 * h));
        const double mid_r = norm3(mid_pos);
        const double mid_inv_r3 = 1.0 / fmax(mid_r * mid_r * mid_r, 1.0e-9);
        const double3 acc2 = add3(
            mul3(mid_pos, -2.0 * mid_inv_r3),
            mul3(cross3(z_axis, mid_dir), 0.10 * p.spin * mid_inv_r3)
        );

        prev_pos = pos;
        dir = normalize3(add3(dir, mul3(acc2, h)));
        pos = add3(pos, mul3(mid_dir, h));
        path_length += h;
    }

    if (has_disk_hit && p.disk_hit_mode == DISK_TRANSPARENT_OVERLAY) {
        klass = 1;
        if (color_out) {
            if (p.nav_mode == NAV_DISK_RADIUS_DEBUG) *color_out = disk_radius_color(first_disk.r_cyl, p);
            else if (p.nav_mode == NAV_HIT_DISTANCE_DEBUG) *color_out = hit_distance_color(first_disk.path_length, p);
            else if (p.nav_mode == NAV_HIT_REASON) *color_out = hit_reason_color(1);
            else *color_out = {230, 130, 35};
        }
        if (hit_out) *hit_out = first_disk;
        return klass;
    }
    if (color_out) {
        if (p.nav_mode == NAV_HIT_REASON) *color_out = hit_reason_color(4);
        else if (p.nav_mode == NAV_VOLUME_EMISSIVITY_DEBUG) *color_out = emissivity_debug_color(accum_intensity);
        else {
            const Rgb bg = p.nav_mode == NAV_TORUS_VOLUME
                ? Rgb{0, 0, 0}
                : celestial_sky(p, pos, sky_pixels, sky_width, sky_height);
            *color_out = use_volume ? compose_volume_color(bg, accum_r, accum_g, accum_b, accum_alpha) : bg;
        }
    }
    if (hit_out) *hit_out = {klass, 4, p.max_steps, path_length, sqrt(pos.x * pos.x + pos.y * pos.y), pos.z, accum_intensity};
    return klass;
}

__device__ double3 pk_bl_to_cart(const KerrPreviewState& y, double spin)
{
    const double theta = clampd(y.theta, 1.0e-6, PI - 1.0e-6);
    const double st = sin(theta);
    const double rho = sqrt(fmax(y.r * y.r + spin * spin, 0.0));
    return make_double3(
        rho * st * cos(y.phi),
        rho * st * sin(y.phi),
        y.r * cos(theta)
    );
}

__device__ bool pk_initial_state_sample(
    const PreviewParams& p,
    double pixel_x,
    double pixel_y,
    KerrPreviewState* y_out
)
{
    const double r = fmax(p.r_obs_rg, pk_kerr_horizon(p.spin) + 1.0 + p.horizon_eps);
    const double theta = clampd(p.theta_obs_rad, 1.0e-6, PI - 1.0e-6);
    const double aspect = fmax(p.aspect_ratio, 1.0e-9);
    const double tan_half_fov_x = tan(0.5 * p.fov_rad);
    const double tan_half_fov_y = tan_half_fov_x / aspect;
    const double u = (2.0 * (pixel_x + 0.5) / p.nx - 1.0) * tan_half_fov_x;
    const double v = (2.0 * (pixel_y + 0.5) / p.ny - 1.0) * tan_half_fov_y;
    const double norm = sqrt(1.0 + u * u + v * v);
    const double n_r = -1.0 / norm;
    const double n_theta = v / norm;
    const double n_phi = u / norm;

    double g[4][4];
    pk_kerr_metric(r, theta, p.spin, g);
    const double alpha = pk_kerr_lapse(r, theta, p.spin);
    const double omega = pk_kerr_omega(r, theta, p.spin);
    const double p_contra[4] = {
        1.0 / alpha,
        n_r / sqrt(fmax(g[1][1], 1.0e-300)),
        n_theta / sqrt(fmax(g[2][2], 1.0e-300)),
        n_phi / sqrt(fmax(g[3][3], 1.0e-300)) + omega / alpha
    };
    double p_cov[4] = {0.0, 0.0, 0.0, 0.0};
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) p_cov[mu] += g[mu][nu] * p_contra[nu];
    }

    *y_out = {0.0, r, theta, p.phi_obs_rad, p_cov[0], p_cov[1], p_cov[2], p_cov[3]};
    return isfinite(p_cov[0]) && isfinite(p_cov[1]) && isfinite(p_cov[2]) && isfinite(p_cov[3]);
}

__device__ int trace_preview_pixel_full_kerr(
    const PreviewParams& p,
    int i,
    int j,
    Rgb* color_out,
    const Rgb* sky_pixels = nullptr,
    int sky_width = 0,
    int sky_height = 0,
    HitInfo* hit_out = nullptr,
    double pixel_offset_x = 0.0,
    double pixel_offset_y = 0.0
)
{
    KerrPreviewState y;
    if (!pk_initial_state_sample(
            p,
            static_cast<double>(i) + pixel_offset_x,
            static_cast<double>(j) + pixel_offset_y,
            &y
        )) {
        if (color_out) {
            *color_out = p.nav_mode == NAV_HIT_REASON ? hit_reason_color(4) : Rgb{0, 0, 0};
        }
        if (hit_out) *hit_out = {2, 4, 0, 0.0, 0.0, 0.0, 0.0};
        return 2;
    }

    const double r_h = pk_kerr_horizon(p.spin);
    double3 pos = pk_bl_to_cart(y, p.spin);
    double3 prev_pos = pos;
    double path_length = 0.0;
    double accum_r = 0.0;
    double accum_g = 0.0;
    double accum_b = 0.0;
    double accum_alpha = 0.0;
    double accum_intensity = 0.0;
    const bool use_volume = volume_mode(p.nav_mode);
    HitInfo first_disk;
    bool has_disk_hit = false;
    int klass = 2;

    for (int step = 0; step < p.max_steps; ++step) {
        if (!isfinite(y.r) || !isfinite(y.theta) || !isfinite(y.phi) ||
            !isfinite(y.pr) || !isfinite(y.ptheta)) {
            if (color_out) {
                if (p.nav_mode == NAV_HIT_REASON) *color_out = hit_reason_color(4);
                else if (p.nav_mode == NAV_VOLUME_EMISSIVITY_DEBUG) *color_out = emissivity_debug_color(accum_intensity);
                else {
                    const Rgb bg = p.nav_mode == NAV_TORUS_VOLUME
                        ? Rgb{0, 0, 0}
                        : celestial_sky(p, pos, sky_pixels, sky_width, sky_height);
                    *color_out = use_volume ? compose_volume_color(bg, accum_r, accum_g, accum_b, accum_alpha) : bg;
                }
            }
            if (hit_out) *hit_out = {klass, 4, step, path_length, sqrt(pos.x * pos.x + pos.y * pos.y), pos.z, accum_intensity};
            return klass;
        }
        pk_normalize_polar_coordinate(&y);

        const KerrPreviewState y_prev = y;
        pos = pk_bl_to_cart(y, p.spin);

        if (y.r <= r_h + p.horizon_eps) {
            klass = 0;
            if (color_out) {
                if (p.nav_mode == NAV_HIT_REASON) *color_out = hit_reason_color(0);
                else if (p.nav_mode == NAV_VOLUME_EMISSIVITY_DEBUG) *color_out = emissivity_debug_color(accum_intensity);
                else *color_out = compose_volume_color({0, 0, 0}, accum_r, accum_g, accum_b, accum_alpha);
            }
            if (hit_out) *hit_out = {klass, 0, step, path_length, sqrt(pos.x * pos.x + pos.y * pos.y), pos.z, accum_intensity};
            return klass;
        }

        const double h = pk_full_kerr_adaptive_step(p, y);
        double hit_cyl = 0.0;
        double hit_z = 0.0;
        double hit_phi = 0.0;
        const bool disk_hit = !use_volume && path_length > p.near_clip_rg && (
            p.disk_geometry == DISK_THICK_TORUS
                ? thick_torus_hit(p, pos, &hit_cyl, &hit_z, &hit_phi)
                : thin_disk_hit(p, prev_pos, pos, &hit_cyl, &hit_z, &hit_phi)
        );
        if (disk_hit && !has_disk_hit) {
            first_disk = {1, 1, step, path_length, hit_cyl, hit_z};
            has_disk_hit = true;
            if (p.disk_hit_mode == DISK_FIRST_HIT) {
                klass = 1;
                if (color_out) {
                    if (p.nav_mode == NAV_DISK_RADIUS_DEBUG) *color_out = disk_radius_color(hit_cyl, p);
                    else if (p.nav_mode == NAV_PAINT_SWATCH_DISK) *color_out = paint_swatch_disk_color(hit_cyl, hit_phi, p);
                    else if (p.nav_mode == NAV_HIT_DISTANCE_DEBUG) *color_out = hit_distance_color(path_length, p);
                    else if (p.nav_mode == NAV_HIT_REASON) *color_out = hit_reason_color(1);
                    else *color_out = {230, 130, 35};
                }
                if (hit_out) *hit_out = first_disk;
                return klass;
            }
        }

        if (y.r >= p.r_max_rg) {
            if (has_disk_hit && p.disk_hit_mode == DISK_TRANSPARENT_OVERLAY) {
                klass = 1;
                if (color_out) {
                    if (p.nav_mode == NAV_DISK_RADIUS_DEBUG) *color_out = disk_radius_color(first_disk.r_cyl, p);
                    else if (p.nav_mode == NAV_PAINT_SWATCH_DISK) *color_out = paint_swatch_disk_color(first_disk.r_cyl, 0.0, p);
                    else if (p.nav_mode == NAV_HIT_DISTANCE_DEBUG) *color_out = hit_distance_color(first_disk.path_length, p);
                    else if (p.nav_mode == NAV_HIT_REASON) *color_out = hit_reason_color(1);
                    else *color_out = {230, 130, 35};
                }
                if (hit_out) *hit_out = first_disk;
                return klass;
            }
            klass = 2;
            if (color_out) {
                if (p.nav_mode == NAV_HIT_REASON) *color_out = hit_reason_color(3);
                else if (p.nav_mode == NAV_VOLUME_EMISSIVITY_DEBUG) *color_out = emissivity_debug_color(accum_intensity);
                else {
                    const Rgb bg = p.nav_mode == NAV_TORUS_VOLUME
                        ? Rgb{0, 0, 0}
                        : celestial_sky(p, pos, sky_pixels, sky_width, sky_height);
                    *color_out = use_volume ? compose_volume_color(bg, accum_r, accum_g, accum_b, accum_alpha) : bg;
                }
            }
            if (hit_out) *hit_out = {klass, 3, step, path_length, sqrt(pos.x * pos.x + pos.y * pos.y), pos.z, accum_intensity};
            return klass;
        }

        prev_pos = pos;
        double accepted_h = h;
        pk_rkf45_step_preview(p, y, h, 2.0e-3, &accepted_h);
        pk_normalize_polar_coordinate(&y);
        const double3 new_pos = pk_bl_to_cart(y, p.spin);
        const double ds = pk_zamo_spatial_interval_rg(p, y, y_prev);
        if (use_volume && path_length > p.near_clip_rg && ds > 0.0) {
            const double3 delta = make_double3(new_pos.x - pos.x, new_pos.y - pos.y, new_pos.z - pos.z);
            const double euclidean = norm3(delta);
            if (euclidean > 1.0e-12) {
                accumulate_preview_volume_segment(
                    p,
                    pos,
                    mul3(delta, 1.0 / euclidean),
                    ds,
                    &accum_r,
                    &accum_g,
                    &accum_b,
                    &accum_alpha,
                    &accum_intensity
                );
            }
        }
        path_length += ds;
    }

    if (has_disk_hit && p.disk_hit_mode == DISK_TRANSPARENT_OVERLAY) {
        klass = 1;
        if (color_out) {
            if (p.nav_mode == NAV_DISK_RADIUS_DEBUG) *color_out = disk_radius_color(first_disk.r_cyl, p);
            else if (p.nav_mode == NAV_HIT_DISTANCE_DEBUG) *color_out = hit_distance_color(first_disk.path_length, p);
            else if (p.nav_mode == NAV_HIT_REASON) *color_out = hit_reason_color(1);
            else *color_out = {230, 130, 35};
        }
        if (hit_out) *hit_out = first_disk;
        return klass;
    }
    if (color_out) {
        if (p.nav_mode == NAV_HIT_REASON) *color_out = hit_reason_color(4);
        else if (p.nav_mode == NAV_VOLUME_EMISSIVITY_DEBUG) *color_out = emissivity_debug_color(accum_intensity);
        else {
            const Rgb bg = p.nav_mode == NAV_TORUS_VOLUME
                ? Rgb{0, 0, 0}
                : celestial_sky(p, pos, sky_pixels, sky_width, sky_height);
            *color_out = use_volume ? compose_volume_color(bg, accum_r, accum_g, accum_b, accum_alpha) : bg;
        }
    }
    if (hit_out) *hit_out = {klass, 4, p.max_steps, path_length, sqrt(pos.x * pos.x + pos.y * pos.y), pos.z, accum_intensity};
    return klass;
}

__global__ void preview_kernel(
    PreviewParams params,
    Rgb* pixels,
    unsigned char* classes,
    double* hit_distances,
    const Rgb* sky_pixels
)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= params.nx || j >= params.ny) return;

    Rgb color;
    HitInfo hit;
    int klass = 2;
    if (params.geodesic_model == GEODESIC_FULL_KERR) {
        klass = trace_preview_pixel_full_kerr(
            params,
            i,
            j,
            &color,
            sky_pixels,
            params.sky_texture_width,
            params.sky_texture_height,
            &hit
        );
    } else {
        klass = trace_preview_pixel(
            params,
            i,
            j,
            &color,
            sky_pixels,
            params.sky_texture_width,
            params.sky_texture_height,
            &hit
        );
    }
    const int out_j = params.ny - 1 - j;
    const int idx = out_j * params.nx + i;
    pixels[idx] = color;
    if (classes) classes[idx] = static_cast<unsigned char>(klass);
    if (hit_distances) {
        hit_distances[idx] = volume_mode(params.nav_mode)
            ? hit.intensity
            : (klass == 1 ? hit.path_length : -1.0);
    }
}

void write_ppm(const fs::path& path, const std::vector<Rgb>& pixels, int nx, int ny)
{
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out << "P6\n" << nx << " " << ny << "\n255\n";
    for (const Rgb& p : pixels) {
        out.put(static_cast<char>(p.r));
        out.put(static_cast<char>(p.g));
        out.put(static_cast<char>(p.b));
    }
}

void skip_ppm_ws_and_comments(std::istream& in)
{
    while (in) {
        in >> std::ws;
        if (in.peek() != '#') break;
        std::string ignored;
        std::getline(in, ignored);
    }
}

SkyTexture load_sky_texture(const fs::path& path)
{
    SkyTexture texture;
    if (path.empty()) return texture;
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "Sky texture not found: " << path << ". Using procedural sky.\n";
        return texture;
    }
    std::string magic;
    in >> magic;
    if (magic != "P6") {
        std::cerr << "Sky texture must be binary PPM P6: " << path << ". Using procedural sky.\n";
        return texture;
    }
    skip_ppm_ws_and_comments(in);
    in >> texture.width;
    skip_ppm_ws_and_comments(in);
    in >> texture.height;
    skip_ppm_ws_and_comments(in);
    int max_value = 0;
    in >> max_value;
    in.get();
    if (!in || texture.width <= 0 || texture.height <= 0 || max_value != 255) {
        std::cerr << "Invalid sky texture: " << path << ". Using procedural sky.\n";
        return SkyTexture{};
    }
    texture.pixels.resize(static_cast<std::size_t>(texture.width) * texture.height);
    in.read(
        reinterpret_cast<char*>(texture.pixels.data()),
        static_cast<std::streamsize>(texture.pixels.size() * sizeof(Rgb))
    );
    if (!in) {
        std::cerr << "Could not read sky texture pixels: " << path << ". Using procedural sky.\n";
        return SkyTexture{};
    }
    std::cout << "Loaded CUDA sky texture: " << path << " (" << texture.width << "x" << texture.height << ")\n";
    return texture;
}

Counts count_classes(const std::vector<unsigned char>& classes)
{
    Counts counts;
    for (unsigned char c : classes) {
        if (c == 0) ++counts.shadow;
        else if (c == 1) ++counts.disk;
        else ++counts.sky;
    }
    return counts;
}

double color_luma(const Rgb& c)
{
    return 0.2126 * static_cast<double>(c.r)
        + 0.7152 * static_cast<double>(c.g)
        + 0.0722 * static_cast<double>(c.b);
}

void smooth_volume_preview_pixels(
    std::vector<Rgb>& pixels,
    const std::vector<unsigned char>& classes,
    int nx,
    int ny
)
{
    if (nx < 3 || ny < 3 || pixels.size() != static_cast<std::size_t>(nx * ny)) return;
    const std::vector<Rgb> src = pixels;
    const double spatial[3][3] = {
        {0.55, 0.75, 0.55},
        {0.75, 1.00, 0.75},
        {0.55, 0.75, 0.55},
    };
    for (int y = 1; y < ny - 1; ++y) {
        for (int x = 1; x < nx - 1; ++x) {
            const int idx = y * nx + x;
            const double center_luma = color_luma(src[idx]);
            double wr = 0.0;
            double rr = 0.0;
            double gg = 0.0;
            double bb = 0.0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    const int nidx = (y + dy) * nx + (x + dx);
                    const double dl = fabs(color_luma(src[nidx]) - center_luma);
                    const double sigma = classes.empty() || classes[nidx] == classes[idx] ? 24.0 : 10.0;
                    const double range = exp(-(dl * dl) / (2.0 * sigma * sigma));
                    const double w = spatial[dy + 1][dx + 1] * range;
                    wr += w;
                    rr += w * static_cast<double>(src[nidx].r);
                    gg += w * static_cast<double>(src[nidx].g);
                    bb += w * static_cast<double>(src[nidx].b);
                }
            }
            if (wr > 0.0) {
                pixels[idx] = {
                    static_cast<std::uint8_t>(clampd(rr / wr, 0.0, 255.0)),
                    static_cast<std::uint8_t>(clampd(gg / wr, 0.0, 255.0)),
                    static_cast<std::uint8_t>(clampd(bb / wr, 0.0, 255.0))
                };
            }
        }
    }
    const std::vector<Rgb> mid = pixels;
    const double hweights[5] = {0.10, 0.20, 0.40, 0.20, 0.10};
    for (int y = 0; y < ny; ++y) {
        for (int x = 2; x < nx - 2; ++x) {
            const int idx = y * nx + x;
            if (!classes.empty() && classes[idx] == 0) continue;
            const double center_luma = color_luma(mid[idx]);
            double wr = 0.0;
            double rr = 0.0;
            double gg = 0.0;
            double bb = 0.0;
            for (int dx = -2; dx <= 2; ++dx) {
                const int nidx = y * nx + x + dx;
                if (!classes.empty() && classes[nidx] == 0) continue;
                const double dl = fabs(color_luma(mid[nidx]) - center_luma);
                const double range = exp(-(dl * dl) / (2.0 * 70.0 * 70.0));
                const double w = hweights[dx + 2] * range;
                wr += w;
                rr += w * static_cast<double>(mid[nidx].r);
                gg += w * static_cast<double>(mid[nidx].g);
                bb += w * static_cast<double>(mid[nidx].b);
            }
            if (wr > 0.0) {
                pixels[idx] = {
                    static_cast<std::uint8_t>(clampd(rr / wr, 0.0, 255.0)),
                    static_cast<std::uint8_t>(clampd(gg / wr, 0.0, 255.0)),
                    static_cast<std::uint8_t>(clampd(bb / wr, 0.0, 255.0))
                };
            }
        }
    }
}

std::vector<unsigned char> render_cpu_reference(const PreviewParams& params)
{
    PreviewParams kernel_params = params;
    if (kernel_params.aspect_ratio <= 0.0) kernel_params.aspect_ratio = fixed_aspect_ratio(kernel_params);
    std::vector<unsigned char> classes(static_cast<std::size_t>(kernel_params.nx) * kernel_params.ny);
    for (int j = 0; j < kernel_params.ny; ++j) {
        for (int i = 0; i < kernel_params.nx; ++i) {
            const int out_j = kernel_params.ny - 1 - j;
            classes[static_cast<std::size_t>(out_j * kernel_params.nx + i)] =
                static_cast<unsigned char>(trace_preview_pixel(kernel_params, i, j, nullptr));
        }
    }
    return classes;
}

PerfResult render_cuda(
    const PreviewParams& params,
    std::vector<Rgb>& pixels,
    std::vector<unsigned char>& classes,
    std::vector<double>& hit_distances,
    const SkyTexture* sky = nullptr
)
{
    PreviewParams kernel_params = params;
    if (kernel_params.aspect_ratio <= 0.0) kernel_params.aspect_ratio = fixed_aspect_ratio(kernel_params);
    const bool use_sky_texture = kernel_params.sky_mode == SKY_TEXTURE && sky && sky->loaded();
    if (use_sky_texture) {
        kernel_params.sky_texture_width = sky->width;
        kernel_params.sky_texture_height = sky->height;
    } else {
        kernel_params.sky_texture_width = 0;
        kernel_params.sky_texture_height = 0;
    }
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        throw std::runtime_error("No CUDA device found.");
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "CUDA preview device 0: " << prop.name << "\n";

    pixels.assign(static_cast<std::size_t>(kernel_params.nx) * kernel_params.ny, Rgb{});
    classes.assign(static_cast<std::size_t>(kernel_params.nx) * kernel_params.ny, 2);
    hit_distances.assign(static_cast<std::size_t>(kernel_params.nx) * kernel_params.ny, -1.0);

    Rgb* d_pixels = nullptr;
    Rgb* d_sky_pixels = nullptr;
    unsigned char* d_classes = nullptr;
    double* d_hit_distances = nullptr;
    const std::size_t pixel_bytes = pixels.size() * sizeof(Rgb);
    const std::size_t class_bytes = classes.size() * sizeof(unsigned char);
    const std::size_t distance_bytes = hit_distances.size() * sizeof(double);
    const auto frame_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMalloc(&d_pixels, pixel_bytes));
    CUDA_CHECK(cudaMalloc(&d_classes, class_bytes));
    CUDA_CHECK(cudaMalloc(&d_hit_distances, distance_bytes));
    if (use_sky_texture) {
        CUDA_CHECK(cudaMalloc(&d_sky_pixels, sky->pixels.size() * sizeof(Rgb)));
        CUDA_CHECK(cudaMemcpy(
            d_sky_pixels,
            sky->pixels.data(),
            sky->pixels.size() * sizeof(Rgb),
            cudaMemcpyHostToDevice
        ));
    }

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const dim3 block(16, 16);
    const dim3 grid((kernel_params.nx + block.x - 1) / block.x, (kernel_params.ny + block.y - 1) / block.y);

    CUDA_CHECK(cudaEventRecord(start));
    preview_kernel<<<grid, block>>>(kernel_params, d_pixels, d_classes, d_hit_distances, d_sky_pixels);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    const auto copy_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(pixels.data(), d_pixels, pixel_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(classes.data(), d_classes, class_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hit_distances.data(), d_hit_distances, distance_bytes, cudaMemcpyDeviceToHost));
    const auto copy_end = std::chrono::steady_clock::now();
    if (volume_mode(kernel_params.nav_mode)) {
        smooth_volume_preview_pixels(pixels, classes, kernel_params.nx, kernel_params.ny);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_pixels));
    if (d_sky_pixels) CUDA_CHECK(cudaFree(d_sky_pixels));
    CUDA_CHECK(cudaFree(d_classes));
    CUDA_CHECK(cudaFree(d_hit_distances));
    const auto frame_end = std::chrono::steady_clock::now();

    PerfResult perf;
    perf.kernel_ms = static_cast<double>(ms);
    perf.copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
    perf.frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
    perf.seconds = perf.frame_ms / 1000.0;
    perf.fps = perf.seconds > 0.0 ? 1.0 / perf.seconds : 0.0;
    return perf;
}

PerfResult render_cuda_reuse(
    const PreviewParams& params,
    CudaPreviewBuffers& buffers,
    std::vector<Rgb>& pixels,
    std::vector<unsigned char>& classes,
    std::vector<double>& hit_distances,
    const SkyTexture* sky = nullptr
)
{
    PreviewParams kernel_params = params;
    if (kernel_params.aspect_ratio <= 0.0) kernel_params.aspect_ratio = fixed_aspect_ratio(kernel_params);
    const bool use_sky_texture = kernel_params.sky_mode == SKY_TEXTURE && sky && sky->loaded();
    if (use_sky_texture) {
        kernel_params.sky_texture_width = sky->width;
        kernel_params.sky_texture_height = sky->height;
        buffers.upload_sky(*sky);
    } else {
        kernel_params.sky_texture_width = 0;
        kernel_params.sky_texture_height = 0;
        buffers.upload_sky(SkyTexture{});
    }
    if (buffers.nx != kernel_params.nx || buffers.ny != kernel_params.ny || !buffers.d_pixels || !buffers.d_classes) {
        buffers.allocate(kernel_params.nx, kernel_params.ny);
    }

    pixels.resize(static_cast<std::size_t>(kernel_params.nx) * kernel_params.ny);
    classes.resize(static_cast<std::size_t>(kernel_params.nx) * kernel_params.ny);
    hit_distances.resize(static_cast<std::size_t>(kernel_params.nx) * kernel_params.ny);

    const dim3 block(16, 16);
    const dim3 grid((kernel_params.nx + block.x - 1) / block.x, (kernel_params.ny + block.y - 1) / block.y);
    const auto frame_start = std::chrono::steady_clock::now();

    CUDA_CHECK(cudaEventRecord(buffers.start_event));
    preview_kernel<<<grid, block>>>(
        kernel_params,
        buffers.d_pixels,
        buffers.d_classes,
        buffers.d_hit_distances,
        buffers.d_sky_pixels
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(buffers.stop_event));
    CUDA_CHECK(cudaEventSynchronize(buffers.stop_event));

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, buffers.start_event, buffers.stop_event));

    const auto copy_start = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpy(
        pixels.data(),
        buffers.d_pixels,
        pixels.size() * sizeof(Rgb),
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        classes.data(),
        buffers.d_classes,
        classes.size() * sizeof(unsigned char),
        cudaMemcpyDeviceToHost
    ));
    CUDA_CHECK(cudaMemcpy(
        hit_distances.data(),
        buffers.d_hit_distances,
        hit_distances.size() * sizeof(double),
        cudaMemcpyDeviceToHost
    ));
    const auto copy_end = std::chrono::steady_clock::now();
    if (volume_mode(kernel_params.nav_mode)) {
        smooth_volume_preview_pixels(pixels, classes, kernel_params.nx, kernel_params.ny);
    }

    const auto frame_end = std::chrono::steady_clock::now();
    PerfResult perf;
    perf.kernel_ms = static_cast<double>(kernel_ms);
    perf.copy_ms = std::chrono::duration<double, std::milli>(copy_end - copy_start).count();
    perf.frame_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
    perf.seconds = perf.frame_ms / 1000.0;
    perf.fps = perf.seconds > 0.0 ? 1.0 / perf.seconds : 0.0;
    return perf;
}

void append_perf_log(const PreviewParams& params, const PerfResult& perf)
{
    const fs::path out_dir = preview_output_dir();
    fs::create_directories(out_dir);
    std::ofstream out(out_dir / "performance_log.txt", std::ios::app);
    out << "cuda_preview"
        << " resolution=" << params.nx << "x" << params.ny
        << " geodesic_model=" << (params.geodesic_model == GEODESIC_FULL_KERR ? "full_kerr" : "kerr_like")
        << " aspect=" << std::setprecision(6) << params.aspect_ratio
        << " fov_x_deg=" << params.fov_rad * 180.0 / PI
        << " fov_y_deg=" << vertical_fov_rad(params.fov_rad, params.aspect_ratio) * 180.0 / PI
        << " rays=" << params.nx * params.ny
        << " render_seconds=" << std::fixed << std::setprecision(6) << perf.seconds
        << " fps=" << std::setprecision(2) << perf.fps
        << " kernel_ms=" << std::setprecision(4) << perf.kernel_ms
        << " copy_ms=" << perf.copy_ms
        << " upload_ms=" << perf.upload_ms
        << " frame_ms=" << perf.frame_ms
        << " max_steps=" << params.max_steps
        << " step=" << params.step_size
        << " r_max=" << params.r_max_rg
        << "\n";
}

void write_validation(
    const PreviewParams& params,
    const Counts& cuda_counts,
    const Counts& cpu_counts,
    const std::vector<unsigned char>& cuda_classes,
    const std::vector<unsigned char>& cpu_classes,
    const PerfResult& perf
)
{
    fs::create_directories("output/camera_preview");
    int same = 0;
    for (std::size_t i = 0; i < cuda_classes.size() && i < cpu_classes.size(); ++i) {
        if (cuda_classes[i] == cpu_classes[i]) ++same;
    }
    const double agreement = cuda_classes.empty()
        ? 0.0
        : 100.0 * static_cast<double>(same) / static_cast<double>(cuda_classes.size());

    std::ofstream out("output/camera_preview/cuda_vs_cpu_preview_validation.txt");
    out << "# HADROS CUDA preview validation\n\n";
    out << "This compares the CUDA preview kernel with the scalar CPU implementation of\n";
    out << "the same preview-only low-order geodesic model. It is not a production\n";
    out << "radiative-transfer validation.\n\n";
    out << "resolution: " << params.nx << "x" << params.ny << "\n";
    out << "aspect_ratio: " << std::fixed << std::setprecision(6) << params.aspect_ratio << "\n";
    out << "fov_x_deg: " << params.fov_rad * 180.0 / PI << "\n";
    out << "fov_y_deg: " << vertical_fov_rad(params.fov_rad, params.aspect_ratio) * 180.0 / PI << "\n";
    out << "cuda_shadow_pixels: " << cuda_counts.shadow << "\n";
    out << "cuda_disk_pixels: " << cuda_counts.disk << "\n";
    out << "cuda_sky_pixels: " << cuda_counts.sky << "\n";
    out << "cpu_shadow_pixels: " << cpu_counts.shadow << "\n";
    out << "cpu_disk_pixels: " << cpu_counts.disk << "\n";
    out << "cpu_sky_pixels: " << cpu_counts.sky << "\n";
    out << "classification_agreement_percent: " << std::fixed << std::setprecision(2) << agreement << "\n";
    out << "cuda_render_seconds: " << std::setprecision(6) << perf.seconds << "\n";
    out << "cuda_fps: " << std::setprecision(2) << perf.fps << "\n";
    out << "max_steps: " << params.max_steps << "\n";
    out << "step_size: " << params.step_size << "\n";
    out << "r_max_rg: " << params.r_max_rg << "\n";
}

void write_torus_volume_validation(
    const PreviewParams& params,
    const double observer_distances[3],
    const std::vector<Counts>& counts_list,
    const std::vector<double>& mean_intensity,
    const std::vector<double>& max_intensity
)
{
    fs::create_directories("output/camera_preview");
    std::ofstream out("output/camera_preview/torus_volume_preview_validation.txt");
    out << "# HADROS CUDA preview torus-volume validation\n\n";
    out << "preview_only: true\n";
    out << "nav_mode: celestial_plus_torus_volume\n";
    out << "first_hit_disk_default: disabled\n";
    out << "torus_r0_rg: " << params.torus_r0_rg << "\n";
    out << "torus_sigma_r_rg: " << params.torus_sigma_r_rg << "\n";
    out << "torus_h_rg: " << params.torus_h_rg << "\n";
    out << "torus_alpha: " << params.torus_alpha << "\n";
    out << "torus_brightness: " << params.torus_brightness << "\n\n";
    out << "observer_r_rg,shadow_fraction,sky_fraction,mean_accumulated_torus_intensity,max_accumulated_torus_intensity\n";
    out << std::fixed << std::setprecision(6);
    for (int i = 0; i < 3; ++i) {
        const double total = static_cast<double>(std::max(1, counts_list[i].shadow + counts_list[i].disk + counts_list[i].sky));
        out << observer_distances[i] << ","
            << counts_list[i].shadow / total << ","
            << counts_list[i].sky / total << ","
            << mean_intensity[i] << ","
            << max_intensity[i] << "\n";
    }
}

std::string timestamp_compact()
{
    const auto now = std::chrono::system_clock::now();
    const std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream out;
    out << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return out.str();
}

void save_camera_json(const PreviewParams& params)
{
    fs::create_directories("configs/cameras");
    const fs::path camera_path = fs::path("configs/cameras") / ("camera_cuda_" + timestamp_compact() + ".json");
    const fs::path last_json = fs::path("configs/cameras") / "last_camera.json";
    auto write = [&](const fs::path& path) {
        std::ofstream out(path);
        out << std::setprecision(12);
        out << "{\n";
        out << "  \"camera_name\": \"hadros_geodesic_preview_cuda\",\n";
        out << "  \"backend\": \"cuda\",\n";
        out << "  \"geodesic_model\": \"" << (params.geodesic_model == GEODESIC_FULL_KERR ? "full_kerr" : "kerr_like") << "\",\n";
        out << "  \"observer_distance_rg\": " << params.r_obs_rg << ",\n";
        out << "  \"inclination_deg\": " << params.theta_obs_rad * 180.0 / PI << ",\n";
        out << "  \"azimuth_deg\": " << params.phi_obs_rad * 180.0 / PI << ",\n";
        out << "  \"fov_deg\": " << params.fov_rad * 180.0 / PI << ",\n";
        out << "  \"fov_x_deg\": " << params.fov_rad * 180.0 / PI << ",\n";
        out << "  \"fov_y_deg\": " << vertical_fov_rad(params.fov_rad, params.aspect_ratio) * 180.0 / PI << ",\n";
        out << "  \"aspect_ratio\": " << params.aspect_ratio << ",\n";
        out << "  \"spin\": " << params.spin << ",\n";
        out << "  \"requested_spin\": " << params.requested_spin << ",\n";
        out << "  \"preview_spin_convention\": \""
            << (params.spin_convention == SPIN_CONVENTION_THORNE ? "thorne" : "hadros")
            << "\",\n";
        out << "  \"r_max_rg\": " << params.r_max_rg << ",\n";
        out << "  \"preview_nav_mode\": " << params.nav_mode << ",\n";
        out << "  \"preview_sky_mode\": \""
            << (params.sky_mode == SKY_TEXTURE ? "texture" :
                (params.sky_mode == SKY_INTERSTELLAR_COORDINATE_GRID ? "interstellar_coordinate_grid" : "procedural"))
            << "\",\n";
        out << "  \"preview_disk_r_in_rg\": " << params.disk_r_min_rg << ",\n";
        out << "  \"preview_disk_r_out_rg\": " << params.disk_r_max_rg << ",\n";
        out << "  \"preview_disk_thickness_rg\": " << params.disk_thickness_rg << ",\n";
        out << "  \"preview_step_size\": " << params.step_size << ",\n";
        out << "  \"preview_max_steps\": " << params.max_steps << ",\n";
        out << "  \"preview_resolution\": [" << params.nx << ", " << params.ny << "],\n";
        out << "  \"notes\": \"saved from CUDA preview; preview-only geodesic camera framing, not production radiative transfer\"\n";
        out << "}\n";
    };
    write(camera_path);
    write(last_json);
    std::cout << "Saved CUDA preview camera: " << camera_path << "\n";
}

void save_current_preview(
    const fs::path& output_path,
    const std::vector<Rgb>& pixels,
    const PreviewParams& params,
    const PerfResult& perf
)
{
    if (!pixels.empty()) {
        write_ppm(output_path, pixels, params.nx, params.ny);
    }
    save_camera_json(params);
    append_perf_log(params, perf);
    std::cout << "Saved CUDA preview image: " << output_path << "\n";
}

int parse_nav_mode(const std::string& value)
{
    if (value == "detailed" ||
        value == "celestial_plus_torus_volume" ||
        value == "celestial_sphere" ||
        value == "celestial") return NAV_CELESTIAL_PLUS_TORUS_VOLUME;
    if (value == "torus_volume") return NAV_TORUS_VOLUME;
    if (value == "volume_emissivity_debug" || value == "emissivity_debug") return NAV_VOLUME_EMISSIVITY_DEBUG;
    if (value == "paint_swatch_disk" || value == "interstellar_disk" || value == "thin_paint_disk") return NAV_PAINT_SWATCH_DISK;
    if (value == "shadow_disk" || value == "nav" || value == "fast") return NAV_SHADOW_DISK;
    if (value == "first_hit_disk_debug" || value == "opaque_disk_debug" || value == "opaque_disk") return NAV_FIRST_HIT_DISK_DEBUG;
    if (value == "disk_radius_debug" || value == "radius_debug") return NAV_DISK_RADIUS_DEBUG;
    if (value == "hit_reason" || value == "reason") return NAV_HIT_REASON;
    if (value == "hit_distance_debug" || value == "distance_debug") return NAV_HIT_DISTANCE_DEBUG;
    return NAV_CELESTIAL_PLUS_TORUS_VOLUME;
}

void write_nav_mode_validation()
{
    struct ModeCheck {
        const char* name;
        int mode;
    };
    const ModeCheck checks[] = {
        {"celestial_sphere", parse_nav_mode("celestial_sphere")},
        {"torus_volume", parse_nav_mode("torus_volume")},
        {"celestial_plus_torus_volume", parse_nav_mode("celestial_plus_torus_volume")},
        {"detailed", parse_nav_mode("detailed")},
        {"shadow_disk", parse_nav_mode("shadow_disk")},
        {"paint_swatch_disk", parse_nav_mode("paint_swatch_disk")},
        {"first_hit_disk_debug", parse_nav_mode("first_hit_disk_debug")},
        {"opaque_disk_debug", parse_nav_mode("opaque_disk_debug")},
        {"disk_radius_debug", parse_nav_mode("disk_radius_debug")},
        {"hit_reason", parse_nav_mode("hit_reason")},
        {"hit_distance_debug", parse_nav_mode("hit_distance_debug")},
    };

    fs::create_directories("output/camera_preview");
    std::ofstream out("output/camera_preview/nav_mode_volume_validation.txt");
    out << "# HADROS CUDA preview nav-mode validation\n\n";
    out << "mode,internal_id,uses_volume_path,debug_only,first_hit_disk_debug\n";
    for (const ModeCheck& check : checks) {
        const bool debug_only =
            check.mode == NAV_SHADOW_DISK ||
            check.mode == NAV_FIRST_HIT_DISK_DEBUG ||
            check.mode == NAV_OPAQUE_DISK_DEBUG ||
            check.mode == NAV_DISK_RADIUS_DEBUG ||
            check.mode == NAV_HIT_REASON ||
            check.mode == NAV_HIT_DISTANCE_DEBUG;
        const bool first_hit_disk =
            check.mode == NAV_SHADOW_DISK ||
            check.mode == NAV_FIRST_HIT_DISK_DEBUG ||
            check.mode == NAV_OPAQUE_DISK_DEBUG ||
            check.mode == NAV_DISK_RADIUS_DEBUG ||
            check.mode == NAV_HIT_REASON ||
            check.mode == NAV_HIT_DISTANCE_DEBUG;
        out << check.name << ","
            << check.mode << ","
            << (volume_mode(check.mode) ? "yes" : "no") << ","
            << (debug_only ? "yes" : "no") << ","
            << (first_hit_disk ? "yes" : "no") << "\n";
    }
}

void write_aspect_ratio_validation(const PreviewParams& params)
{
    struct AspectCase {
        const char* name;
        int nx;
        int ny;
        const char* path;
    };
    const AspectCase cases[] = {
        {"16:9", 256, 144, "output/camera_preview/aspect_16x9.ppm"},
        {"21:9", 336, 144, "output/camera_preview/aspect_21x9.ppm"},
        {"4:3", 192, 144, "output/camera_preview/aspect_4x3.ppm"},
        {"1:1", 144, 144, "output/camera_preview/aspect_1x1.ppm"},
    };

    fs::create_directories("output/camera_preview");
    std::ofstream out("output/camera_preview/aspect_ratio_validation.txt");
    out << "# HADROS CUDA preview aspect-ratio validation\n\n";
    out << "fov_definition: horizontal FOV_x\n";
    out << "projection: FOV_y = 2 atan(tan(FOV_x/2) / aspect)\n";
    out << "mode: fixed headless validation\n";
    out << "validation_fov_x_deg: " << std::max(params.fov_rad, 60.0 * PI / 180.0) * 180.0 / PI << "\n\n";
    out << "label,resolution,aspect,fov_x_deg,fov_y_deg,shadow_pixels,sky_pixels,shadow_bbox_w,shadow_bbox_h,shadow_bbox_w_over_h,image\n";

    for (const AspectCase& c : cases) {
        PreviewParams test = params;
        test.nx = test.final_nx = c.nx;
        test.ny = test.final_ny = c.ny;
        test.fov_rad = std::max(params.fov_rad, 60.0 * PI / 180.0);
        test.aspect_mode = ASPECT_FIXED;
        test.aspect_ratio = fixed_aspect_ratio(test);
        std::vector<Rgb> pixels;
        std::vector<unsigned char> classes;
        std::vector<double> distances;
        (void)render_cuda(test, pixels, classes, distances);
        write_ppm(c.path, pixels, test.nx, test.ny);

        int min_x = test.nx;
        int min_y = test.ny;
        int max_x = -1;
        int max_y = -1;
        for (int y = 0; y < test.ny; ++y) {
            for (int x = 0; x < test.nx; ++x) {
                const int idx = y * test.nx + x;
                if (classes[idx] == 0) {
                    min_x = std::min(min_x, x);
                    max_x = std::max(max_x, x);
                    min_y = std::min(min_y, y);
                    max_y = std::max(max_y, y);
                }
            }
        }
        const Counts counts = count_classes(classes);
        const int bbox_w = max_x >= min_x ? max_x - min_x + 1 : 0;
        const int bbox_h = max_y >= min_y ? max_y - min_y + 1 : 0;
        const double bbox_ratio = bbox_h > 0 ? static_cast<double>(bbox_w) / bbox_h : 0.0;
        out << c.name << ","
            << test.nx << "x" << test.ny << ","
            << std::fixed << std::setprecision(6) << test.aspect_ratio << ","
            << test.fov_rad * 180.0 / PI << ","
            << vertical_fov_rad(test.fov_rad, test.aspect_ratio) * 180.0 / PI << ","
            << counts.shadow << ","
            << counts.sky << ","
            << bbox_w << ","
            << bbox_h << ","
            << bbox_ratio << ","
            << c.path << "\n";
    }
}

int parse_disk_geometry(const std::string& value)
{
    if (value == "thick_torus" || value == "torus") return DISK_THICK_TORUS;
    return DISK_THIN;
}

int parse_disk_hit_mode(const std::string& value)
{
    if (value == "transparent_overlay" || value == "overlay") return DISK_TRANSPARENT_OVERLAY;
    return DISK_FIRST_HIT;
}

int parse_aspect_mode(const std::string& value)
{
    if (value == "fixed") return ASPECT_FIXED;
    if (value == "window") return ASPECT_WINDOW;
    throw std::runtime_error("Unknown aspect mode: " + value);
}

int parse_sky_mode(const std::string& value)
{
    if (value == "texture" || value == "eso" || value == "milky_way") return SKY_TEXTURE;
    if (value == "interstellar_coordinate_grid" || value == "interstellar_grid" ||
        value == "color_calibration_grid" || value == "calibration_grid" ||
        value == "thorne_grid" || value == "kip_thorne_grid" || value == "checker" ||
        value == "checkerboard") {
        return SKY_INTERSTELLAR_COORDINATE_GRID;
    }
    if (value == "procedural" || value == "grid") return SKY_PROCEDURAL;
    throw std::runtime_error("Unknown sky mode: " + value);
}

int parse_geodesic_model(const std::string& value)
{
    if (value == "full_kerr" || value == "kerr" || value == "full") return GEODESIC_FULL_KERR;
    if (value == "kerr_like" || value == "preview" || value == "approx") return GEODESIC_KERR_LIKE;
    throw std::runtime_error("Unknown geodesic model: " + value);
}

int parse_spin_convention(const std::string& value)
{
    if (value == "thorne" || value == "kip_thorne" || value == "interstellar") return SPIN_CONVENTION_THORNE;
    if (value == "hadros" || value == "native" || value == "legacy") return SPIN_CONVENTION_HADROS;
    throw std::runtime_error("Unknown preview spin convention: " + value);
}

double fixed_aspect_ratio(const PreviewParams& p)
{
    return static_cast<double>(std::max(1, p.nx)) / static_cast<double>(std::max(1, p.ny));
}

double vertical_fov_rad(double fov_x_rad, double aspect)
{
    return 2.0 * atan(tan(0.5 * fov_x_rad) / fmax(aspect, 1.0e-9));
}

int height_for_aspect(int width, double aspect)
{
    return std::max(1, static_cast<int>(std::lround(static_cast<double>(std::max(1, width)) / fmax(aspect, 1.0e-9))));
}

std::string aspect_label(double aspect)
{
    const int scale = 120;
    int w = std::max(1, static_cast<int>(std::lround(aspect * scale)));
    int h = scale;
    const int g = std::max(1, std::gcd(w, h));
    w /= g;
    h /= g;
    if (std::abs(aspect - 16.0 / 9.0) < 0.03) return "16:9";
    if (std::abs(aspect - 21.0 / 9.0) < 0.04) return "21:9";
    if (std::abs(aspect - 4.0 / 3.0) < 0.03) return "4:3";
    if (std::abs(aspect - 1.0) < 0.02) return "1:1";
    std::ostringstream out;
    out << w << ":" << h;
    return out.str();
}

void apply_quality_preset(PreviewParams& params, const std::string& quality, bool explicit_step, bool explicit_steps)
{
    if (quality == "fast") {
        if (!explicit_steps) params.max_steps = 400;
        if (!explicit_step) params.step_size = 1.20;
    } else if (quality == "medium") {
        if (!explicit_steps) params.max_steps = 800;
        if (!explicit_step) params.step_size = 0.75;
    } else if (quality == "high") {
        if (!explicit_steps) params.max_steps = 1200;
        if (!explicit_step) params.step_size = 0.45;
    }
}

void apply_full_kerr_quality_preset(
    PreviewParams& params,
    const std::string& quality,
    bool explicit_step,
    bool explicit_steps
)
{
    if (params.geodesic_model != GEODESIC_FULL_KERR) return;

    if (quality == "fast") {
        if (!explicit_steps) params.max_steps = 900;
        if (!explicit_step) params.step_size = 0.65;
    } else if (quality == "medium") {
        if (!explicit_steps) params.max_steps = 1400;
        if (!explicit_step) params.step_size = 0.35;
    } else if (quality == "high") {
        if (!explicit_steps) params.max_steps = 2400;
        if (!explicit_step) params.step_size = 0.20;
    }
}

void apply_full_kerr_interactive_safety(PreviewParams& params)
{
    if (params.geodesic_model != GEODESIC_FULL_KERR) return;
    if (params.allow_expensive_preview) return;

    const int safe_final_nx = 512;
    const int safe_interactive_nx = 256;
    const int safe_max_steps = 900;
    const double safe_step = 0.35;
    const double requested_aspect =
        static_cast<double>(std::max(1, params.final_nx)) /
        static_cast<double>(std::max(1, params.final_ny));

    bool changed = false;
    if (params.final_nx > safe_final_nx) {
        params.final_nx = safe_final_nx;
        changed = true;
    }
    if (params.interactive_nx > safe_interactive_nx) {
        params.interactive_nx = safe_interactive_nx;
        changed = true;
    }
    if (params.aspect_mode == ASPECT_FIXED) {
        params.final_ny = height_for_aspect(params.final_nx, requested_aspect);
        params.interactive_ny = height_for_aspect(params.interactive_nx, requested_aspect);
    } else {
        params.final_ny = std::min(params.final_ny, 288);
        params.interactive_ny = std::min(params.interactive_ny, 144);
    }
    if (params.max_steps > safe_max_steps) {
        params.max_steps = safe_max_steps;
        changed = true;
    }
    if (params.step_size > safe_step) {
        params.step_size = safe_step;
        changed = true;
    }
    params.nx = std::min(params.nx, params.interactive_nx);
    params.ny = std::min(params.ny, params.interactive_ny);
    if (changed) {
        std::cerr
            << "Full Kerr CUDA interactive safety: limiting preview to "
            << params.interactive_nx << "x" << params.interactive_ny
            << " while moving, " << params.final_nx << "x" << params.final_ny
            << " refined, max_steps=" << params.max_steps
            << ", step=" << params.step_size << ".\n";
    }
}

class LatencyCsvLogger {
public:
    LatencyCsvLogger()
    {
        const fs::path out_dir = preview_output_dir();
        fs::create_directories(out_dir);
        out_.open(out_dir / "interactive_latency_log.csv", std::ios::app);
        if (out_.tellp() == 0) {
            out_ << "frame,input_poll_ms,camera_update_ms,cuda_kernel_ms,cuda_copy_ms,"
                 << "gl_texture_upload_ms,draw_quad_ms,glfw_swap_buffers_ms,total_loop_ms,"
                 << "camera_to_texture_ms,fps,camera_dirty,rendered\n";
        }
    }

    void write(const LatencyStats& s)
    {
        if (!out_) return;
        out_ << frame_++ << ","
             << std::fixed << std::setprecision(4)
             << s.input_poll_ms << ","
             << s.camera_update_ms << ","
             << s.cuda_kernel_ms << ","
             << s.cuda_copy_ms << ","
             << s.gl_texture_upload_ms << ","
             << s.draw_quad_ms << ","
             << s.glfw_swap_buffers_ms << ","
             << s.total_loop_ms << ","
             << s.camera_to_texture_ms << ","
             << s.fps << ","
             << (s.camera_dirty ? 1 : 0) << ","
             << (s.rendered ? 1 : 0) << "\n";
        if (frame_ % 60 == 0) out_.flush();
    }

private:
    std::ofstream out_;
    std::size_t frame_ = 0;
};

#ifdef HADROS_CUDA_PREVIEW_GLFW
PreviewParams* g_params = nullptr;
bool g_needs_render = true;
bool g_camera_dirty = true;
bool g_dragging = false;
bool g_save_requested = false;
bool g_framebuffer_resized = false;
double g_last_x = 0.0;
double g_last_y = 0.0;
std::chrono::steady_clock::time_point g_last_camera_change_time = std::chrono::steady_clock::now();
bool g_camera_change_pending = false;

void request_render()
{
    g_needs_render = true;
    g_camera_dirty = true;
    g_camera_change_pending = true;
    g_last_camera_change_time = std::chrono::steady_clock::now();
}

void key_callback(GLFWwindow* window, int key, int, int action, int)
{
    if ((action != GLFW_PRESS && action != GLFW_REPEAT) || !g_params) return;
    PreviewParams& p = *g_params;
    if (key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q) {
        g_save_requested = true;
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    } else if (key == GLFW_KEY_S) {
        g_save_requested = true;
    } else if (key == GLFW_KEY_COMMA) {
        p.step_size = clampd(p.step_size * 0.8, 0.01, 10.0);
        request_render();
    } else if (key == GLFW_KEY_PERIOD) {
        p.step_size = clampd(p.step_size * 1.25, 0.01, 10.0);
        request_render();
    } else if (key == GLFW_KEY_R) {
        request_render();
    }
}

bool poll_continuous_input(GLFWwindow* window, PreviewParams& p, double dt)
{
    bool changed = false;
    const double angle_step = p.rot_speed_rad_s * dt;
    const double distance_step = p.zoom_speed_rg_s * dt;
    const double fov_step = p.fov_speed_rad_s * dt;
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS) {
        p.phi_obs_rad -= angle_step;
        changed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS) {
        p.phi_obs_rad += angle_step;
        changed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS) {
        p.theta_obs_rad = clampd(p.theta_obs_rad - angle_step, 1.0e-6, PI - 1.0e-6);
        changed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS) {
        p.theta_obs_rad = clampd(p.theta_obs_rad + angle_step, 1.0e-6, PI - 1.0e-6);
        changed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_KP_ADD) == GLFW_PRESS) {
        p.r_obs_rg = clampd(p.r_obs_rg - distance_step, 4.0, 1000.0);
        changed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS) {
        p.r_obs_rg = clampd(p.r_obs_rg + distance_step, 4.0, 1000.0);
        changed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_LEFT_BRACKET) == GLFW_PRESS) {
        p.fov_rad = clampd(p.fov_rad - fov_step, 1.0 * PI / 180.0, 160.0 * PI / 180.0);
        changed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_RIGHT_BRACKET) == GLFW_PRESS) {
        p.fov_rad = clampd(p.fov_rad + fov_step, 1.0 * PI / 180.0, 160.0 * PI / 180.0);
        changed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
        p.spin = clampd(p.spin - 0.6 * dt, -0.999, 0.999);
        changed = true;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
        p.spin = clampd(p.spin + 0.6 * dt, -0.999, 0.999);
        changed = true;
    }
    if (changed) request_render();
    return changed;
}

void cursor_callback(GLFWwindow*, double x, double y)
{
    if (!g_params) return;
    if (g_dragging) {
        g_params->phi_obs_rad += 0.25 * (x - g_last_x) * PI / 180.0;
        g_params->theta_obs_rad = clampd(
            g_params->theta_obs_rad + 0.25 * (y - g_last_y) * PI / 180.0,
            1.0e-6,
            PI - 1.0e-6
        );
        request_render();
    }
    g_last_x = x;
    g_last_y = y;
}

void mouse_callback(GLFWwindow*, int button, int action, int)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        g_dragging = action == GLFW_PRESS;
    }
}

void scroll_callback(GLFWwindow*, double, double y)
{
    if (!g_params) return;
    g_params->fov_rad = clampd(
        g_params->fov_rad - y * 2.0 * PI / 180.0,
        1.0 * PI / 180.0,
        160.0 * PI / 180.0
    );
    request_render();
}

void framebuffer_size_callback(GLFWwindow*, int, int)
{
    g_framebuffer_resized = true;
    request_render();
}

void init_texture(GLuint texture, const std::vector<Rgb>& pixels, int nx, int ny)
{
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, nx, ny, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
}

void resize_texture(GLuint texture, std::vector<Rgb>& pixels, std::vector<unsigned char>& classes, int nx, int ny)
{
    pixels.assign(static_cast<std::size_t>(nx) * ny, Rgb{10, 16, 28});
    classes.assign(static_cast<std::size_t>(nx) * ny, 2);
    init_texture(texture, pixels, nx, ny);
}

double update_texture(GLuint texture, const std::vector<Rgb>& pixels, int nx, int ny)
{
    const auto start = std::chrono::steady_clock::now();
    glBindTexture(GL_TEXTURE_2D, texture);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, nx, ny, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void draw_texture(GLFWwindow* window, GLuint texture, int nx, int ny)
{
    int fb_width = 0;
    int fb_height = 0;
    glfwGetFramebufferSize(window, &fb_width, &fb_height);
    glViewport(0, 0, fb_width, fb_height);

    const float scale = std::max(
        1.0f,
        std::min(
            static_cast<float>(fb_width) / static_cast<float>(std::max(1, nx)),
            static_cast<float>(fb_height) / static_cast<float>(std::max(1, ny))
        )
    );
    const float draw_width = scale * static_cast<float>(nx);
    const float draw_height = scale * static_cast<float>(ny);
    const float x0 = 0.5f * (static_cast<float>(fb_width) - draw_width);
    const float y0 = 0.5f * (static_cast<float>(fb_height) - draw_height);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, static_cast<double>(fb_width), 0.0, static_cast<double>(fb_height), -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(x0, y0);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(x0 + draw_width, y0);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(x0 + draw_width, y0 + draw_height);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(x0, y0 + draw_height);
    glEnd();
    glDisable(GL_TEXTURE_2D);
}

int run_window(
    PreviewParams& params,
    const fs::path& output_path,
    bool live,
    bool vsync,
    const SkyTexture* sky
)
{
    if (!glfwInit()) {
        std::cerr << "GLFW could not initialize; running CUDA headless preview.\n";
        return 2;
    }
    GLFWwindow* window = glfwCreateWindow(
        std::max(640, params.final_nx * 2),
        std::max(360, params.final_ny * 2),
        "HADROS CUDA geodesic preview",
        nullptr,
        nullptr
    );
    if (!window) {
        glfwTerminate();
        std::cerr << "GLFW window creation failed; running CUDA headless preview.\n";
        return 2;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(vsync ? 1 : 0);
    glfwSetKeyCallback(window, key_callback);
    glfwSetCursorPosCallback(window, cursor_callback);
    glfwSetMouseButtonCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    g_params = &params;

    GLuint texture = 0;
    glGenTextures(1, &texture);
    std::vector<Rgb> pixels(static_cast<std::size_t>(params.nx) * params.ny, Rgb{10, 16, 28});
    std::vector<unsigned char> classes;
    std::vector<double> hit_distances;
    init_texture(texture, pixels, params.nx, params.ny);
    CudaPreviewBuffers buffers;
    PerfResult last_perf;
    LatencyStats last_latency;
    Counts last_counts;
    const int static_nav_mode = params.nav_mode;
    auto last_frame = std::chrono::steady_clock::now();
    auto last_log = std::chrono::steady_clock::now();
    auto last_title = std::chrono::steady_clock::now() - std::chrono::seconds(1);
    auto last_camera_motion = std::chrono::steady_clock::now();
    bool final_frame_rendered = false;
    bool saved_this_session = false;
    LatencyCsvLogger latency_log;

    std::cout << "HADROS CUDA geodesic preview controls: arrows/mouse inclination, +/- distance, []/wheel FOV, A/D spin, </> step, R render, Q quit.\n";

    while (!glfwWindowShouldClose(window)) {
        const auto loop_start = std::chrono::steady_clock::now();
        LatencyStats latency;
        latency.camera_dirty = g_camera_dirty;

        const auto poll_start = std::chrono::steady_clock::now();
        glfwPollEvents();
        const auto poll_end = std::chrono::steady_clock::now();
        latency.input_poll_ms = std::chrono::duration<double, std::milli>(poll_end - poll_start).count();

        const auto loop_now = std::chrono::steady_clock::now();
        const double dt = std::chrono::duration<double>(loop_now - last_frame).count();
        last_frame = loop_now;
        bool input_changed = false;
        const auto camera_update_start = std::chrono::steady_clock::now();
        if (live) {
            input_changed = poll_continuous_input(window, params, std::min(dt, 0.05));
        }
        if (input_changed || g_dragging) {
            last_camera_motion = std::chrono::steady_clock::now();
            final_frame_rendered = false;
        }
        const auto camera_update_end = std::chrono::steady_clock::now();
        latency.camera_update_ms = std::chrono::duration<double, std::milli>(camera_update_end - camera_update_start).count();
        latency.camera_dirty = g_camera_dirty;

        int fb_width = 0;
        int fb_height = 0;
        glfwGetFramebufferSize(window, &fb_width, &fb_height);
        const double window_aspect =
            static_cast<double>(std::max(1, fb_width)) / static_cast<double>(std::max(1, fb_height));
        if (params.aspect_mode == ASPECT_WINDOW) {
            params.aspect_ratio = window_aspect;
        } else {
            params.aspect_ratio = fixed_aspect_ratio(params);
        }
        if (g_framebuffer_resized) {
            final_frame_rendered = false;
            g_needs_render = true;
            g_framebuffer_resized = false;
        }

        PreviewParams render_params = params;
        const bool camera_active = input_changed || g_dragging || g_camera_dirty;
        const double still_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - last_camera_motion).count();
        if (camera_active) {
            render_params.nx = params.interactive_nx;
            render_params.ny = params.aspect_mode == ASPECT_WINDOW
                ? height_for_aspect(params.interactive_nx, params.aspect_ratio)
                : params.interactive_ny;
            render_params.nav_mode = static_nav_mode;
        } else if (!final_frame_rendered && still_s >= params.still_refine_delay_s) {
            render_params.nx = params.final_nx;
            render_params.ny = params.aspect_mode == ASPECT_WINDOW
                ? height_for_aspect(params.final_nx, params.aspect_ratio)
                : params.final_ny;
            render_params.nav_mode = static_nav_mode;
            g_needs_render = true;
        }
        render_params.aspect_ratio = params.aspect_mode == ASPECT_WINDOW
            ? params.aspect_ratio
            : fixed_aspect_ratio(render_params);
        const int final_target_ny = params.aspect_mode == ASPECT_WINDOW
            ? height_for_aspect(params.final_nx, params.aspect_ratio)
            : params.final_ny;

        if (render_params.nx != params.nx || render_params.ny != params.ny) {
            params.nx = render_params.nx;
            params.ny = render_params.ny;
            resize_texture(texture, pixels, classes, params.nx, params.ny);
        }
        params.nav_mode = render_params.nav_mode;
        params.aspect_ratio = render_params.aspect_ratio;

        const bool should_render = live ? (g_needs_render || camera_active) : g_needs_render;
        if (should_render) {
            last_perf = render_cuda_reuse(params, buffers, pixels, classes, hit_distances, sky);
            last_counts = count_classes(classes);
            last_perf.upload_ms = update_texture(texture, pixels, params.nx, params.ny);
            latency.cuda_kernel_ms = last_perf.kernel_ms;
            latency.cuda_copy_ms = last_perf.copy_ms;
            latency.gl_texture_upload_ms = last_perf.upload_ms;
            if (g_camera_change_pending) {
                latency.camera_to_texture_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - g_last_camera_change_time
                ).count();
                g_camera_change_pending = false;
            }
            g_needs_render = false;
            g_camera_dirty = false;
            if (!camera_active && params.nx == params.final_nx && params.ny == final_target_ny) {
                final_frame_rendered = true;
            }
            latency.rendered = true;
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_log).count() >= 1.0) {
                append_perf_log(params, last_perf);
                last_log = now;
            }
        }
        if (g_save_requested) {
            save_current_preview(output_path, pixels, params, last_perf);
            g_save_requested = false;
            saved_this_session = true;
        }
        if (!live) {
            g_needs_render = false;
        }
        if (live && !input_changed && !g_dragging && !g_camera_dirty) {
            // Keep drawing the existing texture without launching CUDA work.
        }

        const auto title_now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(title_now - last_title).count() >= 0.10) {
            std::ostringstream title;
            const double fov_y = vertical_fov_rad(params.fov_rad, fmax(params.aspect_ratio, 1.0e-9));
            title << "HADROS CUDA preview | backend=cuda"
                  << " model=" << (params.geodesic_model == GEODESIC_FULL_KERR ? "full_kerr" : "kerr_like")
                  << " res=" << params.nx << "x" << params.ny
                  << " aspect=" << aspect_label(params.aspect_ratio)
                  << " FOVx=" << std::fixed << std::setprecision(1) << params.fov_rad * 180.0 / PI
                  << " FOVy=" << fov_y * 180.0 / PI
                  << " poll=" << std::fixed << std::setprecision(2) << last_latency.input_poll_ms << "ms"
                  << " upd=" << last_latency.camera_update_ms << "ms"
                  << " kernel=" << last_latency.cuda_kernel_ms << "ms"
                  << " copy=" << last_latency.cuda_copy_ms << "ms"
                  << " upload=" << last_latency.gl_texture_upload_ms << "ms"
                  << " draw=" << last_latency.draw_quad_ms << "ms"
                  << " swap=" << last_latency.glfw_swap_buffers_ms << "ms"
                  << " loop=" << last_latency.total_loop_ms << "ms"
                  << " fps=" << std::setprecision(1) << last_latency.fps
                  << " cam2tex=" << std::setprecision(2) << last_latency.camera_to_texture_ms << "ms"
                  << " max_steps=" << params.max_steps
                  << " step=" << params.step_size
                  << " dirty=" << (g_camera_dirty ? "1" : "0")
                  << " live=" << (live ? "1" : "0")
                  << " vsync=" << (vsync ? "1" : "0")
                  << " shadow=" << last_counts.shadow
                  << " disk=" << last_counts.disk
                  << " sky=" << last_counts.sky;
            glfwSetWindowTitle(window, title.str().c_str());
            last_title = title_now;
        }

        glClearColor(0.02f, 0.025f, 0.035f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        const auto draw_start = std::chrono::steady_clock::now();
        draw_texture(window, texture, params.nx, params.ny);
        const auto draw_end = std::chrono::steady_clock::now();
        latency.draw_quad_ms = std::chrono::duration<double, std::milli>(draw_end - draw_start).count();

        const auto swap_start = std::chrono::steady_clock::now();
        glfwSwapBuffers(window);
        const auto swap_end = std::chrono::steady_clock::now();
        latency.glfw_swap_buffers_ms = std::chrono::duration<double, std::milli>(swap_end - swap_start).count();
        const auto loop_end = std::chrono::steady_clock::now();
        latency.total_loop_ms = std::chrono::duration<double, std::milli>(loop_end - loop_start).count();
        latency.fps = latency.total_loop_ms > 0.0 ? 1000.0 / latency.total_loop_ms : 0.0;
        if (!latency.rendered) {
            latency.cuda_kernel_ms = 0.0;
            latency.cuda_copy_ms = 0.0;
            latency.gl_texture_upload_ms = 0.0;
        }
        last_latency = latency;
        latency_log.write(latency);
    }

    if (!saved_this_session) {
        save_current_preview(output_path, pixels, params, last_perf);
    }
    glDeleteTextures(1, &texture);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
#endif

void usage()
{
    std::cout
        << "Usage: hadros_geodesic_preview_cuda [--nx N] [--ny N]\n"
        << "       [--spin a] [--inclination deg] [--azimuth deg] [--fov deg] [--r-obs rg]\n"
        << "       [--spin-convention thorne|hadros]\n"
        << "       [--geodesic-model kerr_like|full_kerr]\n"
        << "       [--r-max rg] [--step h] [--max-steps N] [--horizon-eps eps]\n"
        << "       [--allow-expensive-preview 0|1]\n"
        << "       [--quality fast|medium|high]\n"
        << "       [--nav-mode celestial_plus_torus_volume|celestial_sphere|torus_volume|detailed|paint_swatch_disk]\n"
        << "       [--nav-mode first_hit_disk_debug|opaque_disk_debug|disk_radius_debug|hit_reason|hit_distance_debug]\n"
        << "       [--disk-r-in rg] [--disk-r-out rg] [--disk-thickness rg]\n"
        << "       [--near-clip rg] [--disk-geometry thin_disk|thick_torus]\n"
        << "       [--disk-hit-mode first_hit|transparent_overlay]\n"
        << "       [--torus-r0 rg] [--torus-sigma-r rg] [--torus-h rg]\n"
        << "       [--torus-alpha a] [--torus-brightness b]\n"
        << "       [--torus-max-alpha-step a] [--torus-emissivity-cutoff j]\n"
        << "       [--funnel 0|1] [--funnel-theta deg] [--funnel-width deg]\n"
        << "       [--funnel-alpha a] [--funnel-brightness b]\n"
        << "       [--opaque-structures 0|1]\n"
        << "       [--aspect-mode window|fixed]\n"
        << "       [--sky-mode procedural|interstellar_coordinate_grid|texture] [--sky assets/sky/eso0932a.ppm]\n"
        << "       [--interactive-nx N] [--interactive-ny N] [--still-delay seconds]\n"
        << "       [--rot-speed deg/s] [--zoom-speed rg/s] [--fov-speed deg/s]\n"
        << "       [--out output/camera_preview/geodesic_preview_cuda.ppm]\n"
        << "       [--validate] [--headless] [--live 0|1] [--vsync 0|1]\n";
}

} // namespace

int main(int argc, char* argv[])
{
    PreviewParams params;
    fs::path output_path = preview_output_dir() / "geodesic_preview_cuda.ppm";
    bool validate = false;
    bool headless = false;
    bool live = true;
    bool vsync = false;
    bool explicit_step = false;
    bool explicit_steps = false;
    std::string quality = "medium";
    fs::path sky_path = "assets/sky/eso0932a.ppm";

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) throw std::runtime_error("Missing value after " + arg);
            return argv[++i];
        };
        if (arg == "--nx") params.nx = params.final_nx = std::max(1, std::atoi(next()));
        else if (arg == "--ny") params.ny = params.final_ny = std::max(1, std::atoi(next()));
        else if (arg == "--interactive-nx") params.interactive_nx = std::max(1, std::atoi(next()));
        else if (arg == "--interactive-ny") params.interactive_ny = std::max(1, std::atoi(next()));
        else if (arg == "--spin") params.spin = params.requested_spin = std::atof(next());
        else if (arg == "--spin-convention") params.spin_convention = parse_spin_convention(next());
        else if (arg == "--geodesic-model") params.geodesic_model = parse_geodesic_model(next());
        else if (arg == "--inclination") params.theta_obs_rad = std::atof(next()) * PI / 180.0;
        else if (arg == "--azimuth") params.phi_obs_rad = std::atof(next()) * PI / 180.0;
        else if (arg == "--fov") params.fov_rad = std::atof(next()) * PI / 180.0;
        else if (arg == "--r-obs") params.r_obs_rg = std::atof(next());
        else if (arg == "--r-max") params.r_max_rg = std::atof(next());
        else if (arg == "--step") {
            const std::string value = next();
            if (value != "auto") {
                params.step_size = std::atof(value.c_str());
                explicit_step = true;
            }
        } else if (arg == "--max-steps") {
            const std::string value = next();
            if (value != "auto") {
                params.max_steps = std::max(1, std::atoi(value.c_str()));
                explicit_steps = true;
            }
        }
        else if (arg == "--horizon-eps") params.horizon_eps = std::atof(next());
        else if (arg == "--quality") quality = next();
        else if (arg == "--allow-expensive-preview") params.allow_expensive_preview = std::atoi(next()) != 0;
        else if (arg == "--nav-mode") params.nav_mode = parse_nav_mode(next());
        else if (arg == "--disk-r-in") params.disk_r_min_rg = std::atof(next());
        else if (arg == "--disk-r-out") params.disk_r_max_rg = std::atof(next());
        else if (arg == "--disk-thickness") params.disk_thickness_rg = std::atof(next());
        else if (arg == "--near-clip") params.near_clip_rg = std::atof(next());
        else if (arg == "--disk-geometry") params.disk_geometry = parse_disk_geometry(next());
        else if (arg == "--disk-hit-mode") params.disk_hit_mode = parse_disk_hit_mode(next());
        else if (arg == "--torus-r0") params.torus_r0_rg = std::atof(next());
        else if (arg == "--torus-sigma-r") params.torus_sigma_r_rg = std::atof(next());
        else if (arg == "--torus-h") params.torus_h_rg = std::atof(next());
        else if (arg == "--torus-alpha") params.torus_alpha = std::atof(next());
        else if (arg == "--torus-brightness") params.torus_brightness = std::atof(next());
        else if (arg == "--torus-max-alpha-step") params.torus_max_alpha_step = std::atof(next());
        else if (arg == "--torus-emissivity-cutoff") params.torus_emissivity_cutoff = std::atof(next());
        else if (arg == "--funnel") params.funnel_enabled = std::atoi(next()) != 0;
        else if (arg == "--funnel-theta") params.funnel_theta_rad = std::atof(next()) * PI / 180.0;
        else if (arg == "--funnel-width") params.funnel_sigma_theta_rad = std::atof(next()) * PI / 180.0;
        else if (arg == "--funnel-alpha") params.funnel_alpha = std::atof(next());
        else if (arg == "--funnel-brightness") params.funnel_brightness = std::atof(next());
        else if (arg == "--opaque-structures") params.opaque_structures = std::atoi(next()) != 0;
        else if (arg == "--aspect-mode") params.aspect_mode = parse_aspect_mode(next());
        else if (arg == "--sky-mode") params.sky_mode = parse_sky_mode(next());
        else if (arg == "--sky") sky_path = next();
        else if (arg == "--still-delay") params.still_refine_delay_s = std::max(0.0, std::atof(next()));
        else if (arg == "--rot-speed") params.rot_speed_rad_s = std::atof(next()) * PI / 180.0;
        else if (arg == "--zoom-speed") params.zoom_speed_rg_s = std::atof(next());
        else if (arg == "--fov-speed") params.fov_speed_rad_s = std::atof(next()) * PI / 180.0;
        else if (arg == "--out") output_path = next();
        else if (arg == "--validate") validate = true;
        else if (arg == "--headless") headless = true;
        else if (arg == "--live") live = std::atoi(next()) != 0;
        else if (arg == "--vsync") vsync = std::atoi(next()) != 0;
        else if (arg == "--help") {
            usage();
            return 0;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    params.requested_spin = params.spin;
    if (params.spin_convention == SPIN_CONVENTION_THORNE) {
        params.spin = -params.spin;
    }
    apply_quality_preset(params, quality, explicit_step, explicit_steps);
    apply_full_kerr_quality_preset(params, quality, explicit_step, explicit_steps);
    if (!headless && !validate) {
        apply_full_kerr_interactive_safety(params);
    }
    if (params.aspect_mode == ASPECT_FIXED || headless || validate) {
        params.aspect_ratio = fixed_aspect_ratio(params);
    }
    SkyTexture sky_texture;
    if (params.sky_mode == SKY_TEXTURE) {
        sky_texture = load_sky_texture(sky_path);
        if (sky_texture.loaded()) {
            params.sky_texture_width = sky_texture.width;
            params.sky_texture_height = sky_texture.height;
        } else {
            params.sky_mode = SKY_PROCEDURAL;
        }
    }

#ifdef HADROS_CUDA_PREVIEW_GLFW
    if (!headless && !validate) {
        const int status = run_window(params, output_path, live, vsync, &sky_texture);
        if (status == 0) return 0;
    }
#else
    if (!headless) {
        std::cout << "CUDA preview was built without GLFW/OpenGL; running headless.\n";
    }
#endif

    std::vector<Rgb> pixels;
    std::vector<unsigned char> cuda_classes;
    std::vector<double> hit_distances;
    const PerfResult perf = render_cuda(params, pixels, cuda_classes, hit_distances, &sky_texture);
    write_ppm(output_path, pixels, params.nx, params.ny);
    if (params.nav_mode == NAV_DISK_RADIUS_DEBUG) {
        write_ppm(preview_output_dir() / "disk_clipping_debug.ppm", pixels, params.nx, params.ny);
    }
    append_perf_log(params, perf);

    const Counts cuda_counts = count_classes(cuda_classes);
    std::cout << "CUDA geodesic preview wrote " << output_path << "\n";
    std::cout << "resolution=" << params.nx << "x" << params.ny
              << " seconds=" << std::fixed << std::setprecision(6) << perf.seconds
              << " fps=" << std::setprecision(2) << perf.fps
              << " kernel_ms=" << std::setprecision(4) << perf.kernel_ms
              << " copy_ms=" << perf.copy_ms
              << " frame_ms=" << perf.frame_ms
              << " shadow=" << cuda_counts.shadow
              << " disk=" << cuda_counts.disk
              << " sky=" << cuda_counts.sky
              << "\n";

    if (validate) {
        PreviewParams ref = params;
        ref.nx = 32;
        ref.ny = 32;
        std::vector<Rgb> ref_pixels;
        std::vector<unsigned char> ref_cuda_classes;
        std::vector<double> ref_hit_distances;
        const PerfResult ref_perf = render_cuda(ref, ref_pixels, ref_cuda_classes, ref_hit_distances, &sky_texture);
        const std::vector<unsigned char> cpu_classes = render_cpu_reference(ref);
        write_ppm(preview_output_dir() / "geodesic_preview_cuda_validation_32.ppm", ref_pixels, ref.nx, ref.ny);
        write_validation(
            ref,
            count_classes(ref_cuda_classes),
            count_classes(cpu_classes),
            ref_cuda_classes,
            cpu_classes,
            ref_perf
        );
        std::cout << "Validation written to output/camera_preview/cuda_vs_cpu_preview_validation.txt\n";

        std::vector<Counts> volume_counts;
        std::vector<double> volume_mean_intensity;
        std::vector<double> volume_max_intensity;
        const double observer_distances[3] = {40.0, 80.0, 160.0};
        const char* suffixes[3] = {"r40", "r80", "r160"};
        for (int idx = 0; idx < 3; ++idx) {
            PreviewParams disk_ref = params;
            disk_ref.r_obs_rg = observer_distances[idx];
            disk_ref.r_max_rg = std::max(params.r_max_rg, observer_distances[idx] + params.disk_r_max_rg + 20.0);
            disk_ref.nav_mode = NAV_CELESTIAL_PLUS_TORUS_VOLUME;
            disk_ref.nx = std::max(params.nx, 128);
            disk_ref.ny = std::max(params.ny, 128);
            std::vector<Rgb> disk_pixels;
            std::vector<unsigned char> disk_classes;
            std::vector<double> disk_distances;
            (void)render_cuda(disk_ref, disk_pixels, disk_classes, disk_distances, &sky_texture);
            const fs::path debug_path = preview_output_dir() /
                ("torus_volume_preview_" + std::string(suffixes[idx]) + ".ppm");
            write_ppm(debug_path, disk_pixels, disk_ref.nx, disk_ref.ny);
            volume_counts.push_back(count_classes(disk_classes));
            double sum_i = 0.0;
            double max_i = 0.0;
            for (double value : disk_distances) {
                sum_i += std::max(0.0, value);
                max_i = std::max(max_i, value);
            }
            volume_mean_intensity.push_back(disk_distances.empty() ? 0.0 : sum_i / disk_distances.size());
            volume_max_intensity.push_back(max_i);
        }
        write_torus_volume_validation(params, observer_distances, volume_counts, volume_mean_intensity, volume_max_intensity);
        write_nav_mode_validation();
        write_aspect_ratio_validation(params);
        std::cout << "Torus-volume validation written to output/camera_preview/torus_volume_preview_validation.txt\n";
        std::cout << "Nav-mode validation written to output/camera_preview/nav_mode_volume_validation.txt\n";
        std::cout << "Aspect-ratio validation written to output/camera_preview/aspect_ratio_validation.txt\n";
    }

    return 0;
}
