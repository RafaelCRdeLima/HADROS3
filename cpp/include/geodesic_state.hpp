#ifndef GEODESIC_STATE_HPP
#define GEODESIC_STATE_HPP

struct GeodesicState {

    // Coordinates
    double t;
    double r;
    double theta;
    double phi;

    // Covariant momenta
    double pt;
    double pr;
    double ptheta;
    double pphi;
};

#endif