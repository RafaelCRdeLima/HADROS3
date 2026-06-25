#ifndef KERR_METRIC_DERIVATIVES_HPP
#define KERR_METRIC_DERIVATIVES_HPP

#include "kerr_metric.hpp"

#include <algorithm>
#include <cmath>
#include <string>

enum class KerrDerivativeMode {
    Environment = -1,
    FiniteDifference = 0,
    Analytic = 1
};

inline KerrDerivativeMode parse_kerr_derivative_mode(const std::string& value)
{
    if (value == "analytic") {
        return KerrDerivativeMode::Analytic;
    }
    return KerrDerivativeMode::FiniteDifference;
}

inline const char* kerr_derivative_mode_name(KerrDerivativeMode mode)
{
    if (mode == KerrDerivativeMode::Environment) return "environment";
    return mode == KerrDerivativeMode::Analytic ? "analytic" : "finite_difference";
}

inline double kerr_inverse_metric_derivative_quotient(
    double n,
    double dn,
    double d,
    double dd
)
{
    return (dn * d - n * dd) / (d * d);
}

inline void kerr_inverse_metric_derivatives_analytic(
    const KerrMetric& metric,
    double r,
    double th,
    double dgdr[4][4],
    double dgdtheta[4][4]
)
{
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            dgdr[mu][nu] = 0.0;
            dgdtheta[mu][nu] = 0.0;
        }
    }

    const double a = metric.a;
    const double a2 = a * a;
    const double rr = r * r;
    const double rr_a2 = rr + a2;
    const double s = std::sin(th);
    const double c = std::cos(th);
    const double s2 = s * s;
    const double ds2_dtheta = 2.0 * s * c;
    const double sigma = metric.Sigma(r, th);
    const double delta = metric.Delta(r);
    const double big_a = metric.A(r, th);

    const double dsigma_dr = 2.0 * r;
    const double dsigma_dtheta = -2.0 * a2 * s * c;
    const double ddelta_dr = 2.0 * r - 2.0;
    const double dbig_a_dr = 4.0 * r * rr_a2 - a2 * ddelta_dr * s2;
    const double dbig_a_dtheta = -a2 * delta * ds2_dtheta;

    const double sigma_delta = sigma * delta;
    const double dsigma_delta_dr = dsigma_dr * delta + sigma * ddelta_dr;
    const double dsigma_delta_dtheta = dsigma_dtheta * delta;

    const double n_tt = -big_a;
    dgdr[0][0] = kerr_inverse_metric_derivative_quotient(
        n_tt,
        -dbig_a_dr,
        sigma_delta,
        dsigma_delta_dr
    );
    dgdtheta[0][0] = kerr_inverse_metric_derivative_quotient(
        n_tt,
        -dbig_a_dtheta,
        sigma_delta,
        dsigma_delta_dtheta
    );

    const double n_tphi = -2.0 * a * r;
    dgdr[0][3] = kerr_inverse_metric_derivative_quotient(
        n_tphi,
        -2.0 * a,
        sigma_delta,
        dsigma_delta_dr
    );
    dgdtheta[0][3] = kerr_inverse_metric_derivative_quotient(
        n_tphi,
        0.0,
        sigma_delta,
        dsigma_delta_dtheta
    );
    dgdr[3][0] = dgdr[0][3];
    dgdtheta[3][0] = dgdtheta[0][3];

    dgdr[1][1] = kerr_inverse_metric_derivative_quotient(
        delta,
        ddelta_dr,
        sigma,
        dsigma_dr
    );
    dgdtheta[1][1] = kerr_inverse_metric_derivative_quotient(
        delta,
        0.0,
        sigma,
        dsigma_dtheta
    );

    dgdr[2][2] = -dsigma_dr / (sigma * sigma);
    dgdtheta[2][2] = -dsigma_dtheta / (sigma * sigma);

    const double n_phiphi = delta - a2 * s2;
    const double den_phiphi = sigma_delta * s2;
    const double dden_phiphi_dr = dsigma_delta_dr * s2;
    const double dden_phiphi_dtheta =
        dsigma_delta_dtheta * s2 + sigma_delta * ds2_dtheta;

    dgdr[3][3] = kerr_inverse_metric_derivative_quotient(
        n_phiphi,
        ddelta_dr,
        den_phiphi,
        dden_phiphi_dr
    );
    dgdtheta[3][3] = kerr_inverse_metric_derivative_quotient(
        n_phiphi,
        -a2 * ds2_dtheta,
        den_phiphi,
        dden_phiphi_dtheta
    );
}

inline void kerr_inverse_metric_derivatives_finite_difference(
    const KerrMetric& metric,
    double r,
    double th,
    double dgdr[4][4],
    double dgdtheta[4][4]
)
{
    const double eps_r = 1.0e-5 * std::max(1.0, std::abs(r));
    const double eps_theta = 1.0e-5;
    double gp[4][4];
    double gm[4][4];

    metric.inverse_metric(r + eps_r, th, gp);
    metric.inverse_metric(r - eps_r, th, gm);
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            dgdr[mu][nu] = (gp[mu][nu] - gm[mu][nu]) / (2.0 * eps_r);
        }
    }

    metric.inverse_metric(r, th + eps_theta, gp);
    metric.inverse_metric(r, th - eps_theta, gm);
    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            dgdtheta[mu][nu] = (gp[mu][nu] - gm[mu][nu]) / (2.0 * eps_theta);
        }
    }
}

inline void kerr_inverse_metric_derivatives(
    const KerrMetric& metric,
    KerrDerivativeMode mode,
    double r,
    double th,
    double dgdr[4][4],
    double dgdtheta[4][4]
)
{
    if (mode == KerrDerivativeMode::Analytic) {
        kerr_inverse_metric_derivatives_analytic(metric, r, th, dgdr, dgdtheta);
    } else {
        kerr_inverse_metric_derivatives_finite_difference(metric, r, th, dgdr, dgdtheta);
    }
}

#endif
