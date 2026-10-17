#include "RotatingArmorEkf.hpp"

#include <algorithm>
#include <cmath>

namespace rotating_armor {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

using Gain = Eigen::Matrix<double, kStateDim, kMeasurementDim>;
using MeasurementCovariance = Eigen::Matrix3d;

// 先验里的 sigma 来自外部（配置文件/粘合层），可能没填、填 0、填负数或填 NaN。
// P0 一旦被污染就再也回不来（后续只有 F P F' + Q 与 Joseph 更新，没有任何
// "重新投影"步骤），所以统一兜底：宁可退回保守默认值，也不让非有限值进协方差。
double positiveOr(double value, double fallback) {
    return (std::isfinite(value) && value > 0.0) ? value : fallback;
}

// 把角度折叠到一个周期内（[-period/2, period/2)）。四板对称下 period = pi/2：
// 换到相邻那块板差一个整周期，折叠后残差为 0，这正是"板面法向能直接约束
// 车体 yaw"的数学形式。
double foldPeriod(double value, double period) {
    return std::remainder(value, period);
}

} // namespace

const char* toString(EkfInitStatus status) noexcept {
    switch (status) {
    case EkfInitStatus::Ok:
        return "Ok";
    case EkfInitStatus::InvalidRadiusPrior:
        return "InvalidRadiusPrior";
    case EkfInitStatus::InvalidObservation:
        return "InvalidObservation";
    case EkfInitStatus::InvalidPlateSlot:
        return "InvalidPlateSlot";
    case EkfInitStatus::SpinPriorUnspecified:
        return "SpinPriorUnspecified";
    case EkfInitStatus::InvalidSpinPrior:
        return "InvalidSpinPrior";
    }
    return "Unknown";
}

RotatingArmorEkf::RotatingArmorEkf(const EkfConfig& config) : config_(config) {
    reset();
}

void RotatingArmorEkf::reset() {
    state_.setZero();
    covariance_.setZero();
    initialized_ = false;
    phase_resolved_ = false;
    spin_pinned_ = false;
}

EkfInitStatus RotatingArmorEkf::initialize(const ArmorObservation& observation,
                                           const EkfInitPrior& prior) {
    if (config_.plate_count <= 0 || prior.plate_slot < 0 ||
        prior.plate_slot >= config_.plate_count) {
        return EkfInitStatus::InvalidPlateSlot;
    }
    if (!std::isfinite(prior.radius_m) || !(prior.radius_m > 0.0)) {
        return EkfInitStatus::InvalidRadiusPrior;
    }
    if (!observation.position.allFinite() || !std::isfinite(observation.sigma_position_m) ||
        !(observation.sigma_position_m > 0.0)) {
        return EkfInitStatus::InvalidObservation;
    }

    // 角速度先验必须由调用方显式声明：Unspecified 直接拒绝，
    // 不允许"没填"被静默解释成"目标不转"（那会在默认 Q 下锁进错误极小值）。
    switch (prior.spin) {
    case SpinPriorKind::Unspecified:
        return EkfInitStatus::SpinPriorUnspecified;
    case SpinPriorKind::Static:
        // 声明"目标不转"就得真的不转：同时给非零角速度是自相矛盾的配置。
        if (!std::isfinite(prior.yaw_rate_prior_rad_s) || prior.yaw_rate_prior_rad_s != 0.0) {
            return EkfInitStatus::InvalidSpinPrior;
        }
        break;
    case SpinPriorKind::Spinning:
        if (!std::isfinite(prior.yaw_rate_prior_rad_s) || prior.yaw_rate_prior_rad_s == 0.0) {
            return EkfInitStatus::InvalidSpinPrior;
        }
        break;
    }

    // 有板面法向 -> 初相被观测唯一确定（相位可解释）；
    // 否则是降级路径：相位不可由单帧确定，只能靠角速度先验 + 后续观测收敛。
    const bool phase_from_yaw =
        observation.has_world_yaw && std::isfinite(observation.world_yaw_rad);

    const double slot_phase = 2.0 * kPi * static_cast<double>(prior.plate_slot) /
                              static_cast<double>(config_.plate_count);

    // phase = yaw + slot_phase 是"从车体中心指向装甲板的反方向"的角，
    // 于是由观测反推车体中心：center = 板中心 + r * [cos(phase), sin(phase)]。
    // 降级路径取 phase = 0 作为任意相位参考，并把它写成"完全未知"的方差。
    const double phase = phase_from_yaw ? worldYawToPlatePhase(observation.world_yaw_rad) : 0.0;
    const double yaw = wrapAngle(phase - slot_phase);

    State x = State::Zero();
    x(0) = observation.position(0) + prior.radius_m * std::cos(phase);
    x(2) = observation.position(1) + prior.radius_m * std::sin(phase);
    x(4) = observation.position(2);
    x(6) = yaw;
    // Static 声明把角速度钉在 0；Spinning 用声明的先验值（上面已保证非零且有限）。
    x(7) = (prior.spin == SpinPriorKind::Spinning) ? prior.yaw_rate_prior_rad_s : 0.0;
    x(8) = prior.radius_m;

    const double position_sigma =
        phase_from_yaw ? positiveOr(prior.position_sigma_m, 0.5)
                       : std::max(positiveOr(prior.position_sigma_m, 0.5), prior.radius_m);
    const double velocity_sigma = positiveOr(prior.velocity_sigma_m_s, 1.0);
    // 相位未知时 yaw 的初始方差强制为 pi：用一个"看起来很小"的方差把错误的相位
    // 锁死，比不给相位更糟。
    const double yaw_sigma = phase_from_yaw ? positiveOr(prior.yaw_sigma_rad, 0.2) : kPi;
    const double yaw_rate_sigma =
        (prior.spin == SpinPriorKind::Static) ? 0.0 : positiveOr(prior.yaw_rate_sigma_rad_s, 3.0);
    const double radius_sigma = positiveOr(prior.radius_sigma_m, 0.05);

    Covariance p = Covariance::Zero();
    p.diagonal() << position_sigma * position_sigma, velocity_sigma * velocity_sigma,
        position_sigma * position_sigma, velocity_sigma * velocity_sigma,
        position_sigma * position_sigma, velocity_sigma * velocity_sigma,
        yaw_sigma * yaw_sigma, yaw_rate_sigma * yaw_rate_sigma, radius_sigma * radius_sigma;

    state_ = x;
    covariance_ = p;
    initialized_ = true;
    phase_resolved_ = phase_from_yaw;
    spin_pinned_ = (prior.spin == SpinPriorKind::Static);
    return EkfInitStatus::Ok;
}

bool RotatingArmorEkf::predict(double dt) {
    if (!initialized_ || !std::isfinite(dt) || dt < 0.0) {
        return false;
    }
    if (dt == 0.0) {
        return true; // 同一时间戳的重复帧：合法的空推进
    }
    const StateJacobian f = transitionJacobian(dt);
    state_ = transition(state_, dt);
    Covariance q = processNoise(dt, config_.process);
    if (spin_pinned_) {
        // Static 是硬约束：把角速度的整个噪声块一起清零。否则随机游走会在几十帧内
        // 把"声明不转"磨掉——合成真值测试里 300 帧可漂到 omega=0.10 rad/s、yaw 偏
        // 0.44rad（中心同步偏移 0.11m 抵消掉），而板残差只有 4mm，又是一次
        // "板对、状态错"。
        //
        // 注意必须连 (6,7)/(7,6) 交叉项一起清：白加速度模型给的是
        // [[dt^3/3, dt^2/2], [dt^2/2, dt]]，只把 (7,7) 归零会让该 2x2 块的行列式
        // 变成 -dt^4/4 < 0，即 Q 不再是半正定矩阵。
        q(6, 7) = 0.0;
        q(7, 6) = 0.0;
        q(7, 7) = 0.0;
    }
    covariance_ = f * covariance_ * f.transpose() + q;
    return true;
}

bool RotatingArmorEkf::update(const ArmorObservation& observation, int plate_slot) {
    if (!initialized_ || config_.plate_count <= 0 || plate_slot < 0 ||
        plate_slot >= config_.plate_count) {
        return false;
    }
    if (!observation.position.allFinite() || !std::isfinite(observation.sigma_position_m) ||
        !(observation.sigma_position_m > 0.0)) {
        return false;
    }

    const Measurement h = measurement(state_, plate_slot, config_.plate_count);
    const MeasurementJacobian h_jacobian =
        measurementJacobian(state_, plate_slot, config_.plate_count);
    const MeasurementCovariance r = observation.sigma_position_m *
                                     observation.sigma_position_m *
                                     MeasurementCovariance::Identity();

    // R 正定 => S = H P H' + R 正定，不需要额外的可逆性判断（NIS 门限不在此步）。
    const MeasurementCovariance s = h_jacobian * covariance_ * h_jacobian.transpose() + r;
    const Gain k = covariance_ * h_jacobian.transpose() * s.inverse();

    // 残差 (z - h) 走的是 cos/sin，本来就不需要 wrap；只有状态里的 yaw 需要。
    state_ = state_ + k * (observation.position - h);
    state_(6) = wrapAngle(state_(6));

    // Joseph 形式：数值上比 (I - K H) P 更稳，且保持对称。
    const Covariance i_kh = Covariance::Identity() - k * h_jacobian;
    covariance_ = i_kh * covariance_ * i_kh.transpose() + k * r * k.transpose();

    if (!state_.allFinite() || !covariance_.allFinite()) {
        reset();
        return false;
    }
    return true;
}

bool RotatingArmorEkf::updatePhase(double measured_world_yaw, double sigma_rad) {
    if (!initialized_ || !std::isfinite(measured_world_yaw) || !std::isfinite(sigma_rad) ||
        !(sigma_rad > 0.0) || config_.plate_count <= 0) {
        return false;
    }

    // 残差按 2pi/N 折叠：换到另一块板时差一个整周期，折叠后为 0。
    const double period = 2.0 * kPi / static_cast<double>(config_.plate_count);
    const double residual = foldPeriod(state_(6) - measured_world_yaw, period);

    // 只有 yaw 这一维参与，H 是一个行向量 e6。
    const double s = covariance_(6, 6) + sigma_rad * sigma_rad;
    if (!(s > 0.0)) {
        return false;
    }
    const Eigen::Matrix<double, 9, 1> k = covariance_.col(6) / s;

    state_ = state_ + k * residual;
    state_(6) = wrapAngle(state_(6));

    // Joseph 形式（标量观测）：P = (I - k e6') P (I - k e6')' + k sigma^2 k'
    Eigen::Matrix<double, 9, 9> i_kh = Eigen::Matrix<double, 9, 9>::Identity();
    i_kh.row(6) -= k.transpose();
    covariance_ = i_kh * covariance_ * i_kh.transpose() + sigma_rad * sigma_rad * (k * k.transpose());

    if (!state_.allFinite() || !covariance_.allFinite()) {
        reset();
        return false;
    }
    return true;
}

bool RotatingArmorEkf::updateRadius(double measured_radius, double sigma_radius) {
    if (!initialized_ || !std::isfinite(measured_radius) || !std::isfinite(sigma_radius) ||
        !(measured_radius > 0.0) || !(sigma_radius > 0.0)) {
        return false;
    }

    // 只有半径这一维参与，H = e8。
    const double s = covariance_(8, 8) + sigma_radius * sigma_radius;
    if (!(s > 0.0)) {
        return false;
    }
    const Eigen::Matrix<double, 9, 1> k = covariance_.col(8) / s;
    const double residual = measured_radius - state_(8);
    state_ = state_ + k * residual;
    // 半径是物理量：更新后夹回正区间，避免一条噪声观测把它推到 0 或负数。
    state_(8) = std::clamp(state_(8), 0.03, 0.80);

    Eigen::Matrix<double, 9, 9> i_kh = Eigen::Matrix<double, 9, 9>::Identity();
    i_kh.row(8) -= k.transpose();
    covariance_ = i_kh * covariance_ * i_kh.transpose() +
                  sigma_radius * sigma_radius * (k * k.transpose());

    if (!state_.allFinite() || !covariance_.allFinite()) {
        reset();
        return false;
    }
    return true;
}

bool RotatingArmorEkf::innovationNis(const ArmorObservation& observation, int plate_slot,
                                     double& nis) const {
    if (!initialized_ || config_.plate_count <= 0 || plate_slot < 0 ||
        plate_slot >= config_.plate_count) {
        return false;
    }
    if (!observation.position.allFinite() || !std::isfinite(observation.sigma_position_m) ||
        !(observation.sigma_position_m > 0.0)) {
        return false;
    }
    const Measurement h = measurement(state_, plate_slot, config_.plate_count);
    const MeasurementJacobian h_jacobian =
        measurementJacobian(state_, plate_slot, config_.plate_count);
    const MeasurementCovariance r = observation.sigma_position_m *
                                     observation.sigma_position_m *
                                     MeasurementCovariance::Identity();
    const MeasurementCovariance s = h_jacobian * covariance_ * h_jacobian.transpose() + r;
    const Measurement residual = observation.position - h;
    nis = residual.dot(s.ldlt().solve(residual));
    return std::isfinite(nis);
}

} // namespace rotating_armor
