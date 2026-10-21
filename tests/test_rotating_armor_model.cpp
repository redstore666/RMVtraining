#include "RotatingArmorModel.hpp"

#include <cassert>
#include <cmath>
#include <iostream>

namespace
{
    bool near(double a, double b, double tol = 1e-8)
    {
        return std::abs(a - b) <= tol;
    }
}

int main()
{
    using namespace rotating_armor;

    State x;
    x << 1.0, 0.5, -2.0, 0.2, 1.5, 0.1, 0.3, 1.2, 0.25, 0.0, 0.0;
    constexpr double dt = 0.1;

    const State xp = transition(x, dt);
    assert(near(xp(0), 1.05));
    assert(near(xp(2), -1.98));
    assert(near(xp(4), 1.51));
    assert(near(xp(6), 0.42));
    assert(near(xp(1), x(1)));
    assert(near(xp(8), x(8)));

    const StateJacobian F = transitionJacobian(dt);
    assert(near(F(0, 1), dt));
    assert(near(F(2, 3), dt));
    assert(near(F(4, 5), dt));
    assert(near(F(6, 7), dt));

    // Check the analytic observation Jacobian against finite differences.
    constexpr int slot = 1;
    constexpr int count = 4;
    const MeasurementJacobian H = measurementJacobian(x, slot, count);
    constexpr double eps = 1e-6;
    for (int col = 0; col < kStateDim; ++col)
    {
        State plus = x;
        State minus = x;
        plus(col) += eps;
        minus(col) -= eps;
        const Measurement z_plus = measurement(plus, slot, count);
        const Measurement z_minus = measurement(minus, slot, count);
        for (int row = 0; row < kMeasurementDim; ++row)
        {
            const double numeric = (z_plus(row) - z_minus(row)) / (2.0 * eps);
            assert(near(H(row, col), numeric, 2e-6));
        }
    }

    const Covariance Q = processNoise(dt);
    assert(Q.allFinite());
    assert((Q - Q.transpose()).norm() < 1e-12);
    assert(Q(8, 8) >= 0.0);
    assert(near(wrapAngle(4.0), 4.0 - 2.0 * 3.14159265358979323846));

    std::cout << "RotatingArmorModel tests passed\n";
    return 0;
}
