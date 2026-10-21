#include "RotatingArmorEkf.hpp"

#include <algorithm>
#include <cmath>

namespace rotating_armor {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

// 车心到装甲板平面的水平距离（= 旋转半径）的物理区间。
// 下界 0.05 m：小于它模型退化成"无半径纯旋转"，ω 不可观测，而 update() 的
// 位置雅可比 H(:,8) 仍能把 R 一路压向 0。上界 0.60 m：远大于任何 RM 全向
// 底盘（大装甲板 ~0.28，自制道具实测 ~0.28），只用来兜数值发散。
constexpr double kRadiusMinM = 0.05;
constexpr double kRadiusMaxM = 0.60;

// radius_diff（l）/ height_diff（h）的物理区间：前后两对板相对左右两对最多差
// 半个车身，0.30 m 已经远超任何 RM 底盘，只用兜数值发散。
constexpr double kPairOffsetMaxM = 0.30;

using Gain = Eigen::Matrix<double, kStateDim, kSphericalMeasurementDim>;
using MeasurementCovariance = Eigen::Matrix<double, kSphericalMeasurementDim,
                                             kSphericalMeasurementDim>;

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

bool RotatingArmorEkf::restore(const State &state, const Covariance &covariance,
                               bool phase_resolved, bool spin_pinned) {
    if (!state.allFinite() || !covariance.allFinite() || !(state(8) > 0.0) ||
        std::fabs(state(9)) > kPairOffsetMaxM || std::fabs(state(10)) > kPairOffsetMaxM) {
        return false;
    }
    state_ = state;
    covariance_ = covariance;
    phase_resolved_ = phase_resolved;
    spin_pinned_ = spin_pinned;
    initialized_ = true;
    return true;
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
    case SpinPriorKind::Unknown:
        // 不表态：先验可以是 0，也可以非 0，方差由 yaw_rate_sigma_rad_s 给出。
        if (!std::isfinite(prior.yaw_rate_prior_rad_s)) {
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
    x(7) = (prior.spin == SpinPriorKind::Spinning || prior.spin == SpinPriorKind::Unknown)
               ? prior.yaw_rate_prior_rad_s
               : 0.0;
    x(8) = prior.radius_m;
    // 前后对偏移初值 0（"先假设四块板共圆共高"），方差给 0.10 m 让它自己走。
    x(9) = 0.0;
    x(10) = 0.0;
    radius_prior_m_ = prior.radius_m;
    radius_fixed_ = prior.radius_fixed;

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
    // 前后对半径差/高度差的先验不确定度。sp_vision_25 在 P0 里给的是 1.0（方差），
    // 即完全放开；这里给 0.10 m（方差 0.01），对 0.2~0.3 m 的半径来说已经足够
    // 让它在一个锁定窗口内收敛，又不至于单帧就把非对称性放大成噪声。
    const double pair_sigma = 0.10;

    Covariance p = Covariance::Zero();
    p.diagonal() << position_sigma * position_sigma, velocity_sigma * velocity_sigma,
        position_sigma * position_sigma, velocity_sigma * velocity_sigma,
        position_sigma * position_sigma, velocity_sigma * velocity_sigma,
        yaw_sigma * yaw_sigma, yaw_rate_sigma * yaw_rate_sigma, radius_sigma * radius_sigma,
        pair_sigma * pair_sigma, pair_sigma * pair_sigma;

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

    // 量测取球坐标 (yaw, pitch, distance) + 板面法向角，见 sphericalMeasurement()。
    // 板心 xyz 的直接量测已不再进入 update()：
    //   * 深度误差沿视线方向，在 xyz 里等于同时污染三个通道；
    //   * 在 yaw/pitch/distance 里，深度只进 distance 一维，可以用大方差单独表达。
    const bool has_yaw = observation.has_world_yaw && std::isfinite(observation.world_yaw_rad);
    const SphericalMeasurement h =
        sphericalMeasurement(state_, plate_slot, config_.plate_count, has_yaw);
    SphericalJacobian h_jacobian =
        sphericalJacobian(state_, plate_slot, config_.plate_count, has_yaw);
    if (radius_fixed_) {
        // 不让位置观测改半径：H 的第 8 列清零后，残差里"半径错了"这一项就
        // 只能由中心/相位去解释，R 保持先验。
        h_jacobian.col(8).setZero();
    }

    // 自适应量测噪声（形式借鉴 sp_vision_25 target.cpp:192-196）：
    //   * 角度维由位置噪声按距离折算：sigma_ang = sigma_pos / distance；
    //   * 测距方差随"板面偏离视线"变大（板越侧向，单帧 PnP 的深度越不可信），
    //     用 log(1 + |delta_angle|) 表达，delta_angle 是板法向与视线方位的夹角；
    //   * 法向角方差随距离变大，同样对数增长。
    // 这三项正好对应我们实测到的误差结构（侧向板深度飘、远处法向糊）。
    const double distance = std::max(h(2), 0.20);
    const double sigma_pos = observation.sigma_position_m;
    const double sigma_ang = sigma_pos / distance;
    const double bearing = std::atan2(state_(2), state_(0));
    const double delta_angle =
        has_yaw ? std::fabs(wrapAngle(observation.world_yaw_rad - bearing)) : 0.0;
    const double dist_var = sigma_pos * sigma_pos * (1.0 + std::log1p(delta_angle));
    const double normal_var = sigma_ang * sigma_ang * (1.0 + std::log1p(distance));

    MeasurementCovariance r = MeasurementCovariance::Zero();
    r(0, 0) = sigma_ang * sigma_ang;
    r(1, 1) = sigma_ang * sigma_ang;
    r(2, 2) = dist_var;
    if (has_yaw) {
        r(3, 3) = normal_var;
    } else {
        // 没有板面法向：把这一维方差放到无穷大，等价于不更新（H 行也是零）。
        r(3, 3) = 1.0e9;
    }

    // R 正定 => S = H P H' + R 正定，不需要额外的可逆性判断（NIS 门限不在此步）。
    const MeasurementCovariance s = h_jacobian * covariance_ * h_jacobian.transpose() + r;
    const Gain k = covariance_ * h_jacobian.transpose() * s.inverse();

    SphericalMeasurement residual = SphericalMeasurement::Zero();
    // 残差 = 观测的球坐标 - 预测的球坐标。yaw 与板法向角按 2pi 折叠；
    // pitch 物理上不会越过 +-pi，直接相减；distance 是普通标量。
    const double obs_px = observation.position(0);
    const double obs_py = observation.position(1);
    const double obs_pz = observation.position(2);
    const double obs_rho = std::hypot(obs_px, obs_py);
    residual(0) = wrapAngle(std::atan2(obs_py, obs_px) - h(0));
    residual(1) = std::atan2(obs_pz, obs_rho) - h(1);
    residual(2) = std::sqrt(obs_rho * obs_rho + obs_pz * obs_pz) - h(2);
    residual(3) = has_yaw ? wrapAngle(observation.world_yaw_rad - h(3)) : 0.0;

    // 先算到临时量里再提交：半径被推成 <=0 时整条更新必须丢弃，不能留下半个
    // 被污染的状态。
    State candidate = state_ + k * residual;
    candidate(6) = wrapAngle(candidate(6));

    if (!candidate.allFinite()) {
        return false;
    }
    // 半径是几何量，必须落在物理区间内（理由见文件头 kRadiusMinM 的注释）。
    // l / h 同样要有界：无界时单帧就能把"前后对偏移"推到米级，四块板散架。
    candidate(8) = std::clamp(candidate(8), kRadiusMinM, kRadiusMaxM);
    candidate(9) = std::clamp(candidate(9), -kPairOffsetMaxM, kPairOffsetMaxM);
    candidate(10) = std::clamp(candidate(10), -kPairOffsetMaxM, kPairOffsetMaxM);
    if (config_.plate_count == 4) {
        // 奇数槽位用的是 r + l，它也必须落在物理区间内。
        const double pair_radius = candidate(8) + candidate(9);
        if (pair_radius < kRadiusMinM) {
            candidate(9) = kRadiusMinM - candidate(8);
        } else if (pair_radius > kRadiusMaxM) {
            candidate(9) = kRadiusMaxM - candidate(8);
        }
    }
    state_ = candidate;

    // Joseph 形式：数值上比 (I - K H) P 更稳，且保持对称。
    const Covariance i_kh = Covariance::Identity() - k * h_jacobian;
    covariance_ = i_kh * covariance_ * i_kh.transpose() + k * r * k.transpose();

    if (radius_fixed_) {
        state_(8) = radius_prior_m_;
        covariance_.row(8).setZero();
        covariance_.col(8).setZero();
        covariance_(8, 8) = 1e-9;
    }

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
    const Eigen::Matrix<double, kStateDim, 1> k = covariance_.col(6) / s;

    state_ = state_ + k * residual;
    state_(6) = wrapAngle(state_(6));

    // Joseph 形式（标量观测）：P = (I - k e6') P (I - k e6')' + k sigma^2 k'
    Eigen::Matrix<double, kStateDim, kStateDim> i_kh =
        Eigen::Matrix<double, kStateDim, kStateDim>::Identity();
    i_kh.row(6) -= k.transpose();
    covariance_ = i_kh * covariance_ * i_kh.transpose() + sigma_rad * sigma_rad * (k * k.transpose());

    if (!state_.allFinite() || !covariance_.allFinite()) {
        reset();
        return false;
    }
    return true;
}

bool RotatingArmorEkf::updatePhaseForSlot(double measured_world_yaw, double sigma_rad,
                                         int plate_slot) {
    if (!initialized_ || !std::isfinite(measured_world_yaw) || !std::isfinite(sigma_rad) ||
        !(sigma_rad > 0.0) || config_.plate_count <= 0 || plate_slot < 0 ||
        plate_slot >= config_.plate_count) {
        return false;
    }

    // 已知槽位时，"车体 yaw" = 板面法向 - slot 的相位偏移，是唯一的，不需要
    // 折叠掉 2pi/N。位置更新用同一个槽位，两者一致。
    const double period = 2.0 * kPi / static_cast<double>(config_.plate_count);
    const double target_yaw = measured_world_yaw + static_cast<double>(plate_slot) * period;
    // 只折叠到 +-pi：这是 cos/sin 观测的固有周期，不是四板歧义。
    const double residual = foldPeriod(target_yaw - state_(6), 2.0 * kPi);
    if (!std::isfinite(residual)) {
        return false;
    }

    const double s = covariance_(6, 6) + sigma_rad * sigma_rad;
    if (!(s > 0.0)) {
        return false;
    }
    const Eigen::Matrix<double, kStateDim, 1> k = covariance_.col(6) / s;

    state_ = state_ + k * residual;
    state_(6) = wrapAngle(state_(6));
    Eigen::Matrix<double, kStateDim, kStateDim> i_kh =
        Eigen::Matrix<double, kStateDim, kStateDim>::Identity();
    i_kh.row(6) -= k.transpose();
    covariance_ =
        i_kh * covariance_ * i_kh.transpose() + sigma_rad * sigma_rad * (k * k.transpose());

    if (!state_.allFinite() || !covariance_.allFinite()) {
        reset();
        return false;
    }
    return true;
}

bool RotatingArmorEkf::updateCenter(const Eigen::Vector3d& measured_center_m, double sigma_m) {
    if (!initialized_ || !measured_center_m.allFinite() || !std::isfinite(sigma_m) ||
        !(sigma_m > 0.0)) {
        return false;
    }

    Eigen::Matrix<double, 3, kStateDim> h = Eigen::Matrix<double, 3, kStateDim>::Zero();
    h(0, 0) = 1.0;
    h(1, 2) = 1.0;
    h(2, 4) = 1.0;

    const Eigen::Vector3d predicted(state_(0), state_(2), state_(4));
    const Eigen::Vector3d residual = measured_center_m - predicted;
    const Eigen::Matrix3d s = h * covariance_ * h.transpose() +
                              sigma_m * sigma_m * Eigen::Matrix3d::Identity();
    const Eigen::Matrix<double, kStateDim, 3> k = covariance_ * h.transpose() * s.inverse();

    state_ = state_ + k * residual;

    // Joseph 形式，保持协方差对称正定。
    Eigen::Matrix<double, kStateDim, kStateDim> i_kh =
        Eigen::Matrix<double, kStateDim, kStateDim>::Identity();
    i_kh -= k * h;
    covariance_ = i_kh * covariance_ * i_kh.transpose() +
                  sigma_m * sigma_m * (k * k.transpose());

    if (!state_.allFinite() || !covariance_.allFinite()) {
        reset();
        return false;
    }
    return true;
}

bool RotatingArmorEkf::updateYawRate(double measured_rad_s, double sigma_rad_s) {
    if (!initialized_ || !std::isfinite(measured_rad_s) || !std::isfinite(sigma_rad_s) ||
        !(sigma_rad_s > 0.0)) {
        return false;
    }
    const double s = covariance_(7, 7) + sigma_rad_s * sigma_rad_s;
    if (!(s > 0.0) || !std::isfinite(s)) {
        return false;
    }
    // 标量更新：H = e7。S 是标量，增益就是 P 的第 7 列 / S。
    const Eigen::Matrix<double, kStateDim, 1> gain = covariance_.col(7) / s;
    state_ += gain * (measured_rad_s - state_(7));
    covariance_ -= gain * covariance_.row(7);
    // 数值上保持对称（只做下三角 -> 上三角的镜像，避免长期漂移）。
    covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval();
    state_(6) = wrapAngle(state_(6));
    return state_.allFinite();
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
    const Eigen::Matrix<double, kStateDim, 1> k = covariance_.col(8) / s;
    const double residual = measured_radius - state_(8);
    state_ = state_ + k * residual;
    // 半径是物理量：更新后夹回正区间，避免一条噪声观测把它推到 0 或负数。
    state_(8) = std::clamp(state_(8), 0.03, 0.80);

    Eigen::Matrix<double, kStateDim, kStateDim> i_kh =
        Eigen::Matrix<double, kStateDim, kStateDim>::Identity();
    i_kh.row(8) -= k.transpose();
    covariance_ = i_kh * covariance_ * i_kh.transpose() +
                  sigma_radius * sigma_radius * (k * k.transpose());

    if (!state_.allFinite() || !covariance_.allFinite()) {
        reset();
        return false;
    }
    return true;
}

bool RotatingArmorEkf::setRadius(double radius_m) {
    if (!initialized_ || !std::isfinite(radius_m) || !(radius_m > 0.0)) {
        return false;
    }
    state_(8) = std::clamp(radius_m, 0.03, 0.80);
    radius_prior_m_ = state_(8);
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

    // 必须和 update() 用同一条量测方程，否则门限判的不是滤波器真正吃进去的那个
    // 新息。update() 走的是球坐标 (yaw, pitch, distance) + 板法向角，这里照抄。
    const bool has_yaw = observation.has_world_yaw && std::isfinite(observation.world_yaw_rad);
    const SphericalMeasurement h =
        sphericalMeasurement(state_, plate_slot, config_.plate_count, has_yaw);
    SphericalJacobian h_jacobian =
        sphericalJacobian(state_, plate_slot, config_.plate_count, has_yaw);
    if (radius_fixed_) {
        h_jacobian.col(8).setZero();
    }

    const double distance = std::max(h(2), 0.20);
    const double sigma_pos = observation.sigma_position_m;
    const double sigma_ang = sigma_pos / distance;
    const double bearing = std::atan2(state_(2), state_(0));
    const double delta_angle =
        has_yaw ? std::fabs(wrapAngle(observation.world_yaw_rad - bearing)) : 0.0;

    MeasurementCovariance r = MeasurementCovariance::Zero();
    r(0, 0) = sigma_ang * sigma_ang;
    r(1, 1) = sigma_ang * sigma_ang;
    r(2, 2) = sigma_pos * sigma_pos * (1.0 + std::log1p(delta_angle));
    r(3, 3) = has_yaw ? sigma_ang * sigma_ang * (1.0 + std::log1p(distance)) : 1.0e9;

    SphericalMeasurement residual = SphericalMeasurement::Zero();
    const double obs_px = observation.position(0);
    const double obs_py = observation.position(1);
    const double obs_pz = observation.position(2);
    const double obs_rho = std::hypot(obs_px, obs_py);
    residual(0) = wrapAngle(std::atan2(obs_py, obs_px) - h(0));
    residual(1) = std::atan2(obs_pz, obs_rho) - h(1);
    residual(2) = std::sqrt(obs_rho * obs_rho + obs_pz * obs_pz) - h(2);
    residual(3) = has_yaw ? wrapAngle(observation.world_yaw_rad - h(3)) : 0.0;

    const MeasurementCovariance s = h_jacobian * covariance_ * h_jacobian.transpose() + r;
    // 没有板法向时第 4 维被 1e9 的方差压成 0 贡献，等价于只用前三维。
    nis = residual.dot(s.ldlt().solve(residual));
    return std::isfinite(nis);
}

} // namespace rotating_armor
