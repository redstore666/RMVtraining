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
// radius_diff / height_diff 的物理上限。真实整车的"前后对"差只有 1~2 cm，
// 给到 0.30 m 会让这两维在位置噪声下吸收半径误差（合成 G 场景实测 l 跑到
// 0.33 m，等于把奇数板推到半径 0.60 m 的圆上，瞄准点整体错一块板）。
constexpr double kPairOffsetMaxM = 0.05;

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
    // l/h 的可用范围极小（见 kPairOffsetMaxM），先验就该贴着 0：
    // 真实整车前后对差值只有 1~2 cm，给 0.03 m 已经足够让它自适应。
    const double pair_sigma = 0.03;

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
    if (config_.adaptive_process_noise)
    {
        // 平稳时缓慢回落；机动时由 update() 把它顶上去。
        maneuver_scale_ = 1.0 + config_.adaptive_decay * (maneuver_scale_ - 1.0);
        const double s = std::clamp(maneuver_scale_, 1.0, config_.adaptive_scale_max);
        if (s > 1.0)
        {
            // 只放大"速度/角速度"：位置行不动。位置行跟着放大等于允许状态瞬间
            // 跳变，静止目标会因此漂移（合成 ekf_init_contract 的 Static 收敛断言）。
            q(1, 1) *= s;
            q(3, 3) *= s;
            q(7, 7) *= s;
        }
    }
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

void RotatingArmorEkf::noteInnovation(double nis) {
    if (!config_.adaptive_process_noise || !std::isfinite(nis) || nis < 0.0) {
        return;
    }
    // NIS / floor > 1 说明"当前过程噪声解释不了这次观测" —— 车在机动。
    // 用 EMA 把它变成过程噪声的放大倍数：机动时立刻放大（跟得上瞬时变化），
    // 平稳时 predict() 里缓慢回落（半径/静态中心仍靠平均，不抖）。
    const double ratio = std::clamp(nis / std::max(1.0e-6, config_.adaptive_nis_floor),
                                    1.0, config_.adaptive_scale_max);
    maneuver_scale_ += config_.adaptive_alpha * (ratio - maneuver_scale_);
    maneuver_scale_ = std::clamp(maneuver_scale_, 1.0, config_.adaptive_scale_max);
}

bool RotatingArmorEkf::updateCartesian(const ArmorObservation& observation, int plate_slot) {
    if (!initialized_ || config_.plate_count <= 0 || plate_slot < 0 ||
        plate_slot >= config_.plate_count) {
        return false;
    }
    if (!observation.position.allFinite() || !std::isfinite(observation.sigma_position_m) ||
        !(observation.sigma_position_m > 0.0)) {
        return false;
    }

    // 笛卡尔 3 维量测用自己的一套维度（3），不能复用球坐标版本的 4 维别名。
    using CartesianJacobian = Eigen::Matrix<double, kMeasurementDim, kStateDim>;
    using CartesianCovariance = Eigen::Matrix<double, kMeasurementDim, kMeasurementDim>;
    using CartesianGain = Eigen::Matrix<double, kStateDim, kMeasurementDim>;
    const Measurement h = measurement(state_, plate_slot, config_.plate_count);
    CartesianJacobian h_jacobian = measurementJacobian(state_, plate_slot, config_.plate_count);
    if (radius_fixed_) {
        h_jacobian.col(8).setZero();
    }
    const CartesianCovariance r = observation.sigma_position_m * observation.sigma_position_m *
                                  CartesianCovariance::Identity();
    const CartesianCovariance s = h_jacobian * covariance_ * h_jacobian.transpose() + r;
    const Measurement residual_cart = observation.position - h;
    noteInnovation(residual_cart.dot(s.ldlt().solve(residual_cart)));
    const CartesianGain k = covariance_ * h_jacobian.transpose() * s.inverse();
    State candidate = state_ + k * (observation.position - h);
    candidate(6) = wrapAngle(candidate(6));
    if (!candidate.allFinite()) {
        return false;
    }
    candidate(8) = std::clamp(candidate(8), kRadiusMinM, kRadiusMaxM);
    state_ = candidate;
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

bool RotatingArmorEkf::update(const ArmorObservation& observation, int plate_slot) {
    static const bool spherical_on_by_env = std::getenv("EKF_SPHERICAL") != nullptr;
    if (!config_.spherical_measurement_enabled && !spherical_on_by_env) {
        // 默认走旧的笛卡尔 3 维量测：合成回归（F/G、首帧契约）全部按它标定，
        // 球坐标版本在 v4 上更快但会把高速自转的瞄准点打到相邻板上（见 docs）。
        return updateCartesian(observation, plate_slot);
    }
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
    static const bool pair_offsets_disabled_by_env = std::getenv("EKF_NO_PAIR") != nullptr;
    if (pair_offsets_disabled_by_env || !config_.pair_offset_enabled ||
        observation.plates_visible < 2) {
        // 单板可见时 l / h 不可观测（只观测得到 r+l 或 r），必须冻住；
        // 否则它们会吸收半径误差随机游走。pair_offset_enabled=false 是消融开关。
        h_jacobian.col(9).setZero();
        h_jacobian.col(10).setZero();
    }

    // 自适应量测噪声，直接采用 sp_vision_25 target.cpp:192-196 的三个量：
    //     sigma_yaw^2 = sigma_pitch^2 = 4e-3              (0.063 rad ≈ 3.6°)
    //     sigma_dist^2 = 0.10 * (log(|Δangle| + 1) + 1)   (≈0.10~0.19 m²)
    //     sigma_normal^2 = log(distance + 1)/200 + 9e-2   (≈0.30 rad ≈ 17°)
    // 其中 Δangle 是板法向与视线方位的夹角。
    //
    // 这组数字的物理含义很重要：**yaw/pitch 给得很紧，测距给得极松**。
    // 单帧 PnP 的深度误差本来就远大于横向误差（我们的实测：z 抖动 p90 0.095 m，
    // 横向 p50 0.033 m），若给测距一个小方差，滤波器会去追那条噪声最大的方向，
    // 车心被拖走。
    //
    // 但 sp_vision 原值（系数 1.0）在我们的场景里**太松**：实测 v4 上
    // 系数 1.0 -> 97.1 px、0.10 -> 84.3 px、0.03 -> 87.6 px（见
    // docs/借鉴sp_vision25.md），所以默认取 0.10。差别的原因是他们的目标在
    // 3.5 m 外、标准双灯条板，深度几乎不可观测；我们 1.4 m、板大，深度还有信息。
    const double distance = std::max(h(2), 0.20);
    const double bearing = std::atan2(state_(2), state_(0));
    const double delta_angle =
        has_yaw ? std::fabs(wrapAngle(observation.world_yaw_rad - bearing)) : 0.0;

    auto envOr = [](const char* name, double fallback) {
        const char* v = std::getenv(name);
        return (v != nullptr) ? std::atof(v) : fallback;
    };
    // 量测噪声（形式照 sp_vision_25 target.cpp:192-196，系数按我们的实测定）
    //
    //   角度维  sigma_ang^2    = 4e-3                        (0.063 rad ≈ 3.6°)
    //   测距维  sigma_dist^2   = depth_scale * (log(|Δangle|+1) + 1)
    //   法向维  sigma_normal^2 = 9e-2                        (0.30 rad ≈ 17°)
    //
    // 物理含义：**横向（角度）给紧，深度给松**。单帧 PnP 的四角共面平板对深度
    // 本来就有二义性，而重投影误差对它完全无感（v4 实测重投影 0.5 px，深度抖动
    // 却有 0.09 m）。若给测距一个小方差，滤波器就会去追这条噪声最大的方向，把
    // 车心拖走——这正是我们之前 90 px 误差的主因。
    //
    // 系数不是照抄：sp_vision 是 ≳1 m²（3.5 m 外、标准灯条板），我们实测
    //   depth_scale = 0.10  -> 84.3 px   （采用）
    //   depth_scale = 1.00  -> 97.1 px   （sp_vision 原值）
    //   depth_scale = 0.03  -> 87.6 px   （偏紧）
    //   depth_scale = 0.003 -> 98.9 px   （太紧，深度噪声进来）
    // 测距方差 = K * sigma_pos^2 * (log|Δangle|+1)：既保留"深度比横向松"的比例，
    // 又让干净数据（sigma_pos=0.01）自动收紧，不会像写死常数那样把合成用例带跑。
    const double depth_scale = envOr("SPV_DIST_SCALE", 0.10);
    const double sigma_dist = std::sqrt(depth_scale * (std::log1p(delta_angle) + 1.0));

    MeasurementCovariance r = MeasurementCovariance::Zero();
    r(0, 0) = 4.0e-3;
    r(1, 1) = 4.0e-3;
    r(2, 2) = sigma_dist * sigma_dist;
    r(3, 3) = has_yaw ? 9.0e-2 : 1.0e9;

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

    // NIS 必须和 update() 走同一条量测方程。update() 有两条路径
    // （笛卡尔 / 球坐标，见 EkfConfig::spherical_measurement_enabled），
    // 这里按同一个开关分支，否则门限判的不是滤波器真正吃进去的那个新息——
    // 实测后果是合成 G 场景在 omega=2 处把正确解判成离群、把相邻板判成内点，
    // 瞄准点整体错一块板（误差 0.37 m = 2R·sin45°）。
    static const bool spherical_on_by_env_nis = std::getenv("EKF_SPHERICAL") != nullptr;
    static const bool spherical_nis_by_env = std::getenv("EKF_SPHERICAL_NIS") != nullptr;
    if (!config_.spherical_measurement_enabled && !spherical_nis_by_env) {
        using CartesianJacobian = Eigen::Matrix<double, kMeasurementDim, kStateDim>;
        using CartesianCovariance = Eigen::Matrix<double, kMeasurementDim, kMeasurementDim>;
        const Measurement h = measurement(state_, plate_slot, config_.plate_count);
        CartesianJacobian h_jacobian =
            measurementJacobian(state_, plate_slot, config_.plate_count);
        if (radius_fixed_) {
            h_jacobian.col(8).setZero();
        }
        if (observation.plates_visible < 2) {
            // 和 updateCartesian() 保持一致：只看到一块板时 l/h 不可观测。
            h_jacobian.col(9).setZero();
            h_jacobian.col(10).setZero();
        }
        const CartesianCovariance r = observation.sigma_position_m *
                                      observation.sigma_position_m *
                                      CartesianCovariance::Identity();
        const CartesianCovariance s = h_jacobian * covariance_ * h_jacobian.transpose() + r;
        const Measurement residual = observation.position - h;
        nis = residual.dot(s.ldlt().solve(residual));
        return std::isfinite(nis);
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
    const double bearing = std::atan2(state_(2), state_(0));
    const double delta_angle =
        has_yaw ? std::fabs(wrapAngle(observation.world_yaw_rad - bearing)) : 0.0;

    MeasurementCovariance r = MeasurementCovariance::Zero();
    r(0, 0) = 4.0e-3;
    r(1, 1) = 4.0e-3;
    r(2, 2) = std::log1p(delta_angle) + 1.0;
    r(3, 3) = has_yaw ? (std::log1p(distance) / 200.0 + 9.0e-2) : 1.0e9;

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
