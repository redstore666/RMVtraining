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

// 奇数槽位是否要走 radius_diff / height_diff（只有四板车体才有前后对）。
bool usesPairOffset(int plate_slot, int plate_count) {
    return plate_count == 4 && (plate_slot % 2) == 1;
}

double plateRadius(const State& x, int plate_slot, int plate_count) {
    return usesPairOffset(plate_slot, plate_count) ? x(8) + x(9) : x(8);
}

double plateHeight(const State& x, int plate_slot, int plate_count) {
    return usesPairOffset(plate_slot, plate_count) ? x(4) + x(10) : x(4);
}

double platePhase(const State& x, int plate_slot, int plate_count) {
    return x(6) + 2.0 * kPi * static_cast<double>(plate_slot) /
                      static_cast<double>(plate_count);
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
    const double phase = platePhase(x, plate_slot, plate_count);
    const double radius = plateRadius(x, plate_slot, plate_count);

    Measurement z;
    z(0) = x(0) - radius * std::cos(phase);
    z(1) = x(2) - radius * std::sin(phase);
    z(2) = plateHeight(x, plate_slot, plate_count);
    return z;
}

SphericalMeasurement sphericalMeasurement(const State& x, int plate_slot, int plate_count,
                                           bool has_yaw) {
    validateSlot(plate_slot, plate_count);
    const double phase = platePhase(x, plate_slot, plate_count);
    const double radius = plateRadius(x, plate_slot, plate_count);

    const double px = x(0) - radius * std::cos(phase);
    const double py = x(2) - radius * std::sin(phase);
    const double pz = plateHeight(x, plate_slot, plate_count);
    const double rho = std::hypot(px, py);

    SphericalMeasurement z;
    z(0) = std::atan2(py, px);
    z(1) = std::atan2(pz, rho);
    z(2) = std::sqrt(rho * rho + pz * pz);
    // 板面外法向角：measurement() 把板放在 center - r*[cos, sin]，径向朝外即法向。
    z(3) = has_yaw ? platePhaseToWorldYaw(phase) : 0.0;
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
    const double phase = platePhase(x, plate_slot, plate_count);
    const double radius = plateRadius(x, plate_slot, plate_count);
    const double pair = usesPairOffset(plate_slot, plate_count) ? 1.0 : 0.0;
    const double c = std::cos(phase);
    const double s = std::sin(phase);

    MeasurementJacobian H = MeasurementJacobian::Zero();
    H(0, 0) = 1.0;
    H(0, 6) = radius * s;
    H(0, 8) = -c;
    H(0, 9) = -c * pair;

    H(1, 2) = 1.0;
    H(1, 6) = -radius * c;
    H(1, 8) = -s;
    H(1, 9) = -s * pair;

    H(2, 4) = 1.0;
    H(2, 10) = pair;
    return H;
}

SphericalJacobian sphericalJacobian(const State& x, int plate_slot, int plate_count,
                                    bool has_yaw) {
    validateSlot(plate_slot, plate_count);
    const double phase = platePhase(x, plate_slot, plate_count);
    const double radius = plateRadius(x, plate_slot, plate_count);
    const double pair = usesPairOffset(plate_slot, plate_count) ? 1.0 : 0.0;
    const double c = std::cos(phase);
    const double s = std::sin(phase);

    const double px = x(0) - radius * c;
    const double py = x(2) - radius * s;
    const double pz = plateHeight(x, plate_slot, plate_count);
    const double rho2 = px * px + py * py;
    const double rho = std::sqrt(rho2);
    const double dist2 = rho2 + pz * pz;
    const double dist = std::sqrt(dist2);
    if (!(rho2 > 1e-12) || !(dist2 > 1e-12)) {
        return SphericalJacobian::Zero();
    }

    // dp/dx：板心对状态的雅可比（与 measurementJacobian 同源）。
    Eigen::Matrix<double, 3, kStateDim> dp = Eigen::Matrix<double, 3, kStateDim>::Zero();
    dp(0, 0) = 1.0;
    dp(0, 6) = radius * s;
    dp(0, 8) = -c;
    dp(0, 9) = -c * pair;
    dp(1, 2) = 1.0;
    dp(1, 6) = -radius * c;
    dp(1, 8) = -s;
    dp(1, 9) = -s * pair;
    dp(2, 4) = 1.0;
    dp(2, 10) = pair;

    SphericalJacobian H = SphericalJacobian::Zero();
    // yaw = atan2(py, px)
    H.row(0) = (-py / rho2) * dp.row(0) + (px / rho2) * dp.row(1);
    // pitch = atan2(pz, rho)
    H.row(1) = (-pz * px / (rho * dist2)) * dp.row(0) +
               (-pz * py / (rho * dist2)) * dp.row(1) + (rho / dist2) * dp.row(2);
    // distance = |p|
    H.row(2) = (px / dist) * dp.row(0) + (py / dist) * dp.row(1) + (pz / dist) * dp.row(2);
    // 板面法向角 = phase + pi，只依赖车体 yaw。
    H(3, 6) = has_yaw ? 1.0 : 0.0;
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
    Q(9, 9) = config.pair_radius_random_walk * dt;
    Q(10, 10) = config.pair_height_random_walk * dt;
    return Q;
}

} // namespace rotating_armor
