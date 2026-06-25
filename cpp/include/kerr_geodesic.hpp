#ifndef KERR_GEODESIC_HPP
#define KERR_GEODESIC_HPP

#include "kerr_metric.hpp"
#include "kerr_metric_derivatives.hpp"
#include "geodesic_state.hpp"

/**
 * @brief Hamiltonian Kerr null-geodesic stepper used by HADROS ray tracing.
 *
 * The class evaluates the geodesic right-hand side and advances states using
 * fixed or adaptive Runge--Kutta stepping.  It is shared by image, cache, and
 * validation executables.
 */
class KerrGeodesic {
public:
    /**
     * @brief Construct a Kerr geodesic integrator.
     * @param metric Kerr metric object with spin and coordinate convention.
     * @param h Default integration step.
     * @param tolerance Adaptive-step tolerance.
     * @param derivative_mode Choice of finite-difference or analytic derivatives.
     */
    explicit KerrGeodesic(
        KerrMetric metric,
        double h = 0.02,
        double tolerance = 1.0e-6,
        KerrDerivativeMode derivative_mode = KerrDerivativeMode::Environment
    );

    /**
     * @brief Evaluate the Hamiltonian constraint for a geodesic state.
     * @param y Current geodesic phase-space state.
     * @return Hamiltonian value for diagnostics and validation.
     */
    double hamiltonian(
        const GeodesicState& y
    ) const;

    /**
     * @brief Evaluate the geodesic equations of motion.
     * @param y Current geodesic phase-space state.
     * @return Time derivative of the state.
     */
    GeodesicState rhs(
        const GeodesicState& y
    ) const;

    /**
     * @brief Advance the geodesic state by one fixed RK4 step.
     * @param y State updated in place.
     */
    void step_rk4(
    GeodesicState& y
    ) const;

    /**
     * @brief Advance the geodesic state with adaptive step control.
     * @param y State updated in place.
     */
    void step_adaptive(
        GeodesicState& y
    ) const;

private:
    KerrMetric metric_;
    double h_;
    double tolerance_;
    KerrDerivativeMode derivative_mode_;
};

#endif
