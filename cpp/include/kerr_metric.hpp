#ifndef KERR_METRIC_HPP
#define KERR_METRIC_HPP

struct KerrMetric {
    explicit KerrMetric(double a_spin);

    double a;

    double horizon_radius() const;

    double Sigma(double r, double th) const;
    double Delta(double r) const;
    double A(double r, double th) const;

    void metric(
        double r,
        double th,
        double g[4][4]
    ) const;

    void inverse_metric(
        double r,
        double th,
        double ginv[4][4]
    ) const;

    double lapse(double r, double th) const;
    double omega_frame_drag(double r, double th) const;
};

#endif