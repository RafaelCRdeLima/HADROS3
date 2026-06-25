#include "kerr_geodesic.hpp"
#include <algorithm>

#include <cmath>
#include <cstdlib>
#include <string>

namespace {
KerrDerivativeMode derivative_mode_from_env()
{
    const char* value = std::getenv("KERR_DERIVATIVE_MODE");
    if (!value || std::string(value).empty()) {
        return KerrDerivativeMode::FiniteDifference;
    }
    return parse_kerr_derivative_mode(value);
}
}

KerrGeodesic::KerrGeodesic(
    KerrMetric metric,
    double h,
    double tolerance,
    KerrDerivativeMode derivative_mode
)
    : metric_(metric),
      h_(h),
      tolerance_(tolerance),
      derivative_mode_(
          derivative_mode == KerrDerivativeMode::Environment
          ? derivative_mode_from_env()
          : derivative_mode
      )
{
}

double KerrGeodesic::hamiltonian(
    const GeodesicState& y
) const
{
    double ginv[4][4];

    metric_.inverse_metric(
        y.r,
        y.theta,
        ginv
    );

    const double p[4] = {
        y.pt,
        y.pr,
        y.ptheta,
        y.pphi
    };

    double H = 0.0;

    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            H += 0.5 * ginv[mu][nu] * p[mu] * p[nu];
        }
    }

    return H;
}

GeodesicState KerrGeodesic::rhs(
    const GeodesicState& y
) const
{
    double ginv[4][4];

    metric_.inverse_metric(
        y.r,
        y.theta,
        ginv
    );

    const double p[4] = {
        y.pt,
        y.pr,
        y.ptheta,
        y.pphi
    };

    GeodesicState dydl{};

    dydl.t = 0.0;
    dydl.r = 0.0;
    dydl.theta = 0.0;
    dydl.phi = 0.0;

    for (int nu = 0; nu < 4; ++nu) {
        dydl.t     += ginv[0][nu] * p[nu];
        dydl.r     += ginv[1][nu] * p[nu];
        dydl.theta += ginv[2][nu] * p[nu];
        dydl.phi   += ginv[3][nu] * p[nu];
    }

    dydl.pt = 0.0;
    dydl.pphi = 0.0;

    dydl.pr = 0.0;
    dydl.ptheta = 0.0;

    double dgdr[4][4];
    double dgdtheta[4][4];
    kerr_inverse_metric_derivatives(
        metric_,
        derivative_mode_,
        y.r,
        y.theta,
        dgdr,
        dgdtheta
    );

    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            dydl.pr -= 0.5 *
                dgdr[mu][nu]
                * p[mu] * p[nu];

            dydl.ptheta -= 0.5 *
                dgdtheta[mu][nu]
                * p[mu] * p[nu];
        }
    }

    return dydl;
}

void KerrGeodesic::step_rk4(
    GeodesicState& y
) const
{
    auto add_scaled =
        [](const GeodesicState& y0,
           const GeodesicState& k1, double a1,
           const GeodesicState& k2 = GeodesicState{}, double a2 = 0.0,
           const GeodesicState& k3 = GeodesicState{}, double a3 = 0.0,
           const GeodesicState& k4 = GeodesicState{}, double a4 = 0.0,
           const GeodesicState& k5 = GeodesicState{}, double a5 = 0.0)
    {
        GeodesicState y;

        y.t      = y0.t      + a1*k1.t      + a2*k2.t      + a3*k3.t      + a4*k4.t      + a5*k5.t;
        y.r      = y0.r      + a1*k1.r      + a2*k2.r      + a3*k3.r      + a4*k4.r      + a5*k5.r;
        y.theta  = y0.theta  + a1*k1.theta  + a2*k2.theta  + a3*k3.theta  + a4*k4.theta  + a5*k5.theta;
        y.phi    = y0.phi    + a1*k1.phi    + a2*k2.phi    + a3*k3.phi    + a4*k4.phi    + a5*k5.phi;

        y.pt     = y0.pt     + a1*k1.pt     + a2*k2.pt     + a3*k3.pt     + a4*k4.pt     + a5*k5.pt;
        y.pr     = y0.pr     + a1*k1.pr     + a2*k2.pr     + a3*k3.pr     + a4*k4.pr     + a5*k5.pr;
        y.ptheta = y0.ptheta + a1*k1.ptheta + a2*k2.ptheta + a3*k3.ptheta + a4*k4.ptheta + a5*k5.ptheta;
        y.pphi   = y0.pphi   + a1*k1.pphi   + a2*k2.pphi   + a3*k3.pphi   + a4*k4.pphi   + a5*k5.pphi;

        return y;
    };

    GeodesicState k1 = rhs(y);
    GeodesicState k2 = rhs(add_scaled(y, k1, 0.5 * h_));
    GeodesicState k3 = rhs(add_scaled(y, k2, 0.5 * h_));
    GeodesicState k4 = rhs(add_scaled(y, k3, h_));

    y.t += h_ *
        (k1.t + 2.0*k2.t + 2.0*k3.t + k4.t) / 6.0;

    y.r += h_ *
        (k1.r + 2.0*k2.r + 2.0*k3.r + k4.r) / 6.0;

    y.theta += h_ *
        (k1.theta + 2.0*k2.theta + 2.0*k3.theta + k4.theta) / 6.0;

    y.phi += h_ *
        (k1.phi + 2.0*k2.phi + 2.0*k3.phi + k4.phi) / 6.0;

    y.pt += h_ *
        (k1.pt + 2.0*k2.pt + 2.0*k3.pt + k4.pt) / 6.0;

    y.pr += h_ *
        (k1.pr + 2.0*k2.pr + 2.0*k3.pr + k4.pr) / 6.0;

    y.ptheta += h_ *
        (k1.ptheta + 2.0*k2.ptheta + 2.0*k3.ptheta + k4.ptheta) / 6.0;

    y.pphi += h_ *
        (k1.pphi + 2.0*k2.pphi + 2.0*k3.pphi + k4.pphi) / 6.0;
}

void KerrGeodesic::step_adaptive(GeodesicState& y) const
{
    auto add_scaled =
        [](const GeodesicState& y0,
           const GeodesicState& k1, double a1,
           const GeodesicState& k2, double a2 = 0.0,
           const GeodesicState& k3, double a3 = 0.0,
           const GeodesicState& k4, double a4 = 0.0,
           const GeodesicState& k5, double a5 = 0.0)
    {
        GeodesicState y;

        y.t      = y0.t      + a1*k1.t      + a2*k2.t      + a3*k3.t      + a4*k4.t      + a5*k5.t;
        y.r      = y0.r      + a1*k1.r      + a2*k2.r      + a3*k3.r      + a4*k4.r      + a5*k5.r;
        y.theta  = y0.theta  + a1*k1.theta  + a2*k2.theta  + a3*k3.theta  + a4*k4.theta  + a5*k5.theta;
        y.phi    = y0.phi    + a1*k1.phi    + a2*k2.phi    + a3*k3.phi    + a4*k4.phi    + a5*k5.phi;

        y.pt     = y0.pt     + a1*k1.pt     + a2*k2.pt     + a3*k3.pt     + a4*k4.pt     + a5*k5.pt;
        y.pr     = y0.pr     + a1*k1.pr     + a2*k2.pr     + a3*k3.pr     + a4*k4.pr     + a5*k5.pr;
        y.ptheta = y0.ptheta + a1*k1.ptheta + a2*k2.ptheta + a3*k3.ptheta + a4*k4.ptheta + a5*k5.ptheta;
        y.pphi   = y0.pphi   + a1*k1.pphi   + a2*k2.pphi   + a3*k3.pphi   + a4*k4.pphi   + a5*k5.pphi;

        return y;
    };

    auto error_norm =
        [](const GeodesicState& a,
           const GeodesicState& b)
    {
        double err = 0.0;

        err = std::max(err, std::abs(a.r      - b.r));
        err = std::max(err, std::abs(a.theta  - b.theta));
        err = std::max(err, std::abs(a.phi    - b.phi));
        err = std::max(err, std::abs(a.pr     - b.pr));
        err = std::max(err, std::abs(a.ptheta - b.ptheta));
        err = std::max(err, std::abs(a.pphi   - b.pphi));

        return err;
    };

    double h = h_;

    const double h_min = h_ * 1.0e-5;

    for (int attempt = 0; attempt < 50; ++attempt) {

        GeodesicState k1 = rhs(y);

        GeodesicState k2 = rhs(add_scaled(
            y,
            k1, h * 1.0/4.0,
            GeodesicState{}, 0.0,
            GeodesicState{}, 0.0,
            GeodesicState{}, 0.0,
            GeodesicState{}, 0.0
        ));

        GeodesicState k3 = rhs(add_scaled(
            y,
            k1, h * 3.0/32.0,
            k2, h * 9.0/32.0,
            GeodesicState{}, 0.0,
            GeodesicState{}, 0.0,
            GeodesicState{}, 0.0
        ));

        GeodesicState k4 = rhs(add_scaled(
            y,
            k1, h * 1932.0/2197.0,
            k2, h * -7200.0/2197.0,
            k3, h * 7296.0/2197.0,
            GeodesicState{}, 0.0,
            GeodesicState{}, 0.0
        ));

        GeodesicState k5 = rhs(add_scaled(
            y,
            k1, h * 439.0/216.0,
            k2, h * -8.0,
            k3, h * 3680.0/513.0,
            k4, h * -845.0/4104.0,
            GeodesicState{}, 0.0
        ));

        GeodesicState k6 = rhs(add_scaled(
            y,
            k1, h * -8.0/27.0,
            k2, h * 2.0,
            k3, h * -3544.0/2565.0,
            k4, h * 1859.0/4104.0,
            k5, h * -11.0/40.0
        ));

        GeodesicState y4;

        y4.t      = y.t      + h*(25.0/216.0*k1.t      + 1408.0/2565.0*k3.t      + 2197.0/4104.0*k4.t      - 1.0/5.0*k5.t);
        y4.r      = y.r      + h*(25.0/216.0*k1.r      + 1408.0/2565.0*k3.r      + 2197.0/4104.0*k4.r      - 1.0/5.0*k5.r);
        y4.theta  = y.theta  + h*(25.0/216.0*k1.theta  + 1408.0/2565.0*k3.theta  + 2197.0/4104.0*k4.theta  - 1.0/5.0*k5.theta);
        y4.phi    = y.phi    + h*(25.0/216.0*k1.phi    + 1408.0/2565.0*k3.phi    + 2197.0/4104.0*k4.phi    - 1.0/5.0*k5.phi);
        y4.pt     = y.pt     + h*(25.0/216.0*k1.pt     + 1408.0/2565.0*k3.pt     + 2197.0/4104.0*k4.pt     - 1.0/5.0*k5.pt);
        y4.pr     = y.pr     + h*(25.0/216.0*k1.pr     + 1408.0/2565.0*k3.pr     + 2197.0/4104.0*k4.pr     - 1.0/5.0*k5.pr);
        y4.ptheta = y.ptheta + h*(25.0/216.0*k1.ptheta + 1408.0/2565.0*k3.ptheta + 2197.0/4104.0*k4.ptheta - 1.0/5.0*k5.ptheta);
        y4.pphi   = y.pphi   + h*(25.0/216.0*k1.pphi   + 1408.0/2565.0*k3.pphi   + 2197.0/4104.0*k4.pphi   - 1.0/5.0*k5.pphi);

        GeodesicState y5;

        y5.t      = y.t      + h*(16.0/135.0*k1.t      + 6656.0/12825.0*k3.t      + 28561.0/56430.0*k4.t      - 9.0/50.0*k5.t      + 2.0/55.0*k6.t);
        y5.r      = y.r      + h*(16.0/135.0*k1.r      + 6656.0/12825.0*k3.r      + 28561.0/56430.0*k4.r      - 9.0/50.0*k5.r      + 2.0/55.0*k6.r);
        y5.theta  = y.theta  + h*(16.0/135.0*k1.theta  + 6656.0/12825.0*k3.theta  + 28561.0/56430.0*k4.theta  - 9.0/50.0*k5.theta  + 2.0/55.0*k6.theta);
        y5.phi    = y.phi    + h*(16.0/135.0*k1.phi    + 6656.0/12825.0*k3.phi    + 28561.0/56430.0*k4.phi    - 9.0/50.0*k5.phi    + 2.0/55.0*k6.phi);
        y5.pt     = y.pt     + h*(16.0/135.0*k1.pt     + 6656.0/12825.0*k3.pt     + 28561.0/56430.0*k4.pt     - 9.0/50.0*k5.pt     + 2.0/55.0*k6.pt);
        y5.pr     = y.pr     + h*(16.0/135.0*k1.pr     + 6656.0/12825.0*k3.pr     + 28561.0/56430.0*k4.pr     - 9.0/50.0*k5.pr     + 2.0/55.0*k6.pr);
        y5.ptheta = y.ptheta + h*(16.0/135.0*k1.ptheta + 6656.0/12825.0*k3.ptheta + 28561.0/56430.0*k4.ptheta - 9.0/50.0*k5.ptheta + 2.0/55.0*k6.ptheta);
        y5.pphi   = y.pphi   + h*(16.0/135.0*k1.pphi   + 6656.0/12825.0*k3.pphi   + 28561.0/56430.0*k4.pphi   - 9.0/50.0*k5.pphi   + 2.0/55.0*k6.pphi);

        const double err = error_norm(y4, y5);

        if (err < tolerance_ || h <= h_min) {
            y = y5;
            return;
        }

        const double safety = 0.8;
        const double factor =
            safety * std::pow(tolerance_ / std::max(err, 1.0e-30), 0.25);

        h *= std::clamp(factor, 0.1, 5.0);

        if (h < h_min) {
            h = h_min;
        }
    }

    step_rk4(y);
}
