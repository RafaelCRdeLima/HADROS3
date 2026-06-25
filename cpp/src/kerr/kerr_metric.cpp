#include "kerr_metric.hpp"

#include <cmath>

KerrMetric::KerrMetric(double a_spin)
    : a(a_spin)
{
}

double KerrMetric::horizon_radius() const
{
    return 1.0 + std::sqrt(1.0 - a*a);
}

double KerrMetric::Sigma(double r, double th) const
{
    return r*r + a*a * std::cos(th)*std::cos(th);
}

double KerrMetric::Delta(double r) const
{
    return r*r - 2.0*r + a*a;
}

double KerrMetric::A(double r, double th) const
{
    const double s2 = std::sin(th)*std::sin(th);

    return
        (r*r + a*a)*(r*r + a*a)
        - a*a * Delta(r) * s2;
}

void KerrMetric::metric(
    double r,
    double th,
    double g[4][4]
) const
{
    const double sig = Sigma(r, th);
    const double del = Delta(r);

    const double s = std::sin(th);
    const double s2 = s*s;

    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            g[mu][nu] = 0.0;
        }
    }

    // Boyer-Lindquist Kerr metric
    // signature (-,+,+,+)

    g[0][0] =
        -(1.0 - 2.0*r/sig);

    g[0][3] =
        -2.0*a*r*s2/sig;

    g[3][0] =
        g[0][3];

    g[1][1] =
        sig / del;

    g[2][2] =
        sig;

    g[3][3] =
        (
            r*r + a*a
            + 2.0*a*a*r*s2/sig
        ) * s2;
}

void KerrMetric::inverse_metric(
    double r,
    double th,
    double ginv[4][4]
) const
{
    const double sig = Sigma(r, th);
    const double del = Delta(r);

    const double s = std::sin(th);
    const double s2 = s*s;

    const double bigA = A(r, th);

    for (int mu = 0; mu < 4; ++mu) {
        for (int nu = 0; nu < 4; ++nu) {
            ginv[mu][nu] = 0.0;
        }
    }

    ginv[0][0] =
        -bigA / (sig * del);

    ginv[0][3] =
        -2.0*a*r / (sig * del);

    ginv[3][0] =
        ginv[0][3];

    ginv[1][1] =
        del / sig;

    ginv[2][2] =
        1.0 / sig;

    ginv[3][3] =
        (
            del - a*a*s2
        ) / (sig * del * s2);
}

double KerrMetric::lapse(
    double r,
    double th
) const
{
    const double sig = Sigma(r, th);
    const double del = Delta(r);
    const double bigA = A(r, th);

    return std::sqrt(sig * del / bigA);
}

double KerrMetric::omega_frame_drag(
    double r,
    double th
) const
{

    const double s2 =
        std::sin(th)*std::sin(th);

    return
        2.0*a*r /
        (
            (r*r + a*a)*(r*r + a*a)
            - a*a*Delta(r)*s2
        );
}