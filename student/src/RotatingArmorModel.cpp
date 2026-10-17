#include "RotatingArmorModel.hpp"

#include <cmath>
#include <stdexcept>

namespace rotating_armor {
namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;

void validateDt(double dt) {
    if (!std::isfinite(dt) || dt < 0.0) {
        throw std::invalid_argument("RotatingArmorModel: dt must be finite and >= 0");
    }
}

void validateSlot(int plate_slot, int plate_count) {
    if (plate_count <= 0 || plate_slot < 0 || plate_slot >= plate_count) {
        throw std::invalid_argument("RotatingArmorModel: invalid armor plate slot/count");
    }
}

void addWhiteAccelerationNoise(Covariance& Q, int position_index,
                               int velocity_index, double spectral_density,
                               double dt) {
    if (!std::isfinite(spectral_density) || spectral_density < 0.0) {
        throw std::invalid_argument(
            "RotatingArmorModel: process-noise spectral density must be finite and >= 0");
    }

    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    // Continuous white-acceleration model:
    // q * [[dt^3/3, dt^2/2], [dt^2/2, dt]].
    Q(position_index, position_index) += spectral_density * dt3 / 3.0;
    Q(position_index, velocity_index) += spectral_density * dt2 / 2.0;
    Q(velocity_index, position_index) += spectral_density * dt2 / 2.0;
    Q(velocity_index, velocity_index) += spectral_density * dt;
}
} // namespace

double wrapAngle(double angle) {
    if (!std::isfinite(angle)) {
        return angle;
    }
    angle = std::fmod(angle + kPi, 2.0 * kPi);
    if (angle < 0.0) {
        angle += 2.0 * kPi;
    }
    return angle - kPi;
}

State transition(const State& x, double dt) {
    validateDt(dt);
    State next = x;
    next(0) = x(0) + x(1) * dt; // cx += vx * dt
    next(2) = x(2) + x(3) * dt; // cy += vy * dt
    next(4) = x(4) + x(5) * dt; // z  += vz * dt
    next(6) = wrapAngle(x(6) + x(7) * dt);
    // vx, vy, vz, yaw_rate, and radius follow constant-state assumptions.
    return next;
}

StateJacobian transitionJacobian(double dt) {
    validateDt(dt);
    StateJacobian F = StateJacobian::Identity();
    F(0, 1) = dt;
    F(2, 3) = dt;
    F(4, 5) = dt;
    F(6, 7) = dt;
    return F;
}

Measurement measurement(const State& x, int plate_slot, int plate_count) {
    validateSlot(plate_slot, plate_count);
    const double phase = x(6) + 2.0 * kPi * static_cast<double>(plate_slot) /
                                      static_cast<double>(plate_count);

    Measurement z;
    z(0) = x(0) - x(8) * std::cos(phase);
    z(1) = x(2) - x(8) * std::sin(phase);
    z(2) = x(4);
    return z;
}

double platePhaseToWorldYaw(double phase) {
    return wrapAngle(phase + kPi);
}

double worldYawToPlatePhase(double world_yaw) {
    return wrapAngle(world_yaw - kPi);
}

MeasurementJacobian measurementJacobian(
    const State& x, int plate_slot, int plate_count) {
    validateSlot(plate_slot, plate_count);
    const double phase = x(6) + 2.0 * kPi * static_cast<double>(plate_slot) /
                                      static_cast<double>(plate_count);
    const double radius = x(8);

    MeasurementJacobian H = MeasurementJacobian::Zero();
    H(0, 0) = 1.0;
    H(0, 6) = radius * std::sin(phase);
    H(0, 8) = -std::cos(phase);

    H(1, 2) = 1.0;
    H(1, 6) = -radius * std::cos(phase);
    H(1, 8) = -std::sin(phase);

    H(2, 4) = 1.0;
    return H;
}

Covariance processNoise(double dt, const ProcessNoiseConfig& config) {
    validateDt(dt);
    Covariance Q = Covariance::Zero();
    addWhiteAccelerationNoise(Q, 0, 1, config.accel_x, dt);
    addWhiteAccelerationNoise(Q, 2, 3, config.accel_y, dt);
    addWhiteAccelerationNoise(Q, 4, 5, config.accel_z, dt);
    addWhiteAccelerationNoise(Q, 6, 7, config.angular_accel, dt);

    if (!std::isfinite(config.radius_random_walk) || config.radius_random_walk < 0.0) {
        throw std::invalid_argument(
            "RotatingArmorModel: radius random-walk intensity must be finite and >= 0");
    }
    Q(8, 8) = config.radius_random_walk * dt;
    return Q;
}

} // namespace rotating_armor
