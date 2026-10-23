#include "RotatingArmorTracker.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>

namespace rotating_armor
{
namespace
{
const char *stable_reason(bool enough_updates, bool phase_ok, bool fresh, double age_s)
{
    if (!enough_updates)
        return "few_updates";
    if (!phase_ok)
        return "phase";
    if (!fresh)
        return "stale";
    if (age_s > 0.6)
        return "timeout";
    return "stable";
}

constexpr double kPi = 3.141592653589793238462643383279502884;
// 四块装甲板 90 度对称：相位只有 mod pi/2 的可辨识性。
constexpr double kHalfPi = 0.5 * kPi;
// 角速度用前向差分算，步长 1ms：比手推 d(atan2)/dt 不容易错，代价可忽略。
constexpr double kVelocityDt = 1.0e-3;

// 把角度折到 (-period/2, period/2]。
double foldPeriod(double value, double period)
{
    double wrapped = std::fmod(value + 0.5 * period, period);
    if (wrapped < 0.0)
    {
        wrapped += period;
    }
    return wrapped - 0.5 * period;
}

double clampd(double value, double low, double high)
{
    return std::min(std::max(value, low), high);
}

// Theil-Sen 斜率：取所有"时间跨度 >= min_pair_dt"点对斜率的中位数。
// 对 ≤30% 的野值免疫，配合 MAD 当误差用；点数不足或全非有限时返回 NaN。
double theilSenSlope(const std::vector<double> &times, const std::vector<double> &values,
                     double min_pair_dt, double *sigma_out)
{
    const int count = static_cast<int>(times.size());
    if (count < 4)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::vector<double> slopes;
    slopes.reserve(static_cast<std::size_t>(count) * 2);
    for (int i = 0; i < count; ++i)
    {
        for (int j = i + 1; j < count; ++j)
        {
            const double dt = times[j] - times[i];
            if (dt < min_pair_dt)
            {
                continue;
            }
            slopes.push_back((values[j] - values[i]) / dt);
        }
    }
    if (slopes.size() < 5)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    auto middle = slopes.begin() + static_cast<long>(slopes.size() / 2);
    std::nth_element(slopes.begin(), middle, slopes.end());
    const double slope = *middle;
    std::vector<double> deviations;
    deviations.reserve(slopes.size());
    for (const double value : slopes)
    {
        deviations.push_back(std::fabs(value - slope));
    }
    auto dev_middle = deviations.begin() + static_cast<long>(deviations.size() / 2);
    std::nth_element(deviations.begin(), dev_middle, deviations.end());
    if (sigma_out != nullptr)
    {
        *sigma_out = 1.4826 * (*dev_middle);
    }
    return slope;
}

// 某块板相对云台准星（光轴）的角距离：yaw 与 pitch 里取大的那个。
// 用来回答"现在准星压在谁身上"——自瞄应该瞄准星上的车，而不是检测列表里
// 碰巧排第一的那台（实拍 v4 两台车同时在画面里，旧逻辑会永远锁在右边那台）。
double boresightAngle(const Eigen::Vector3d &position, const GimbalState &gimbal)
{
    return std::max(std::fabs(wrapToPi(vectorYaw(position) - static_cast<double>(gimbal.yaw))),
                    std::fabs(vectorPitch(position) - static_cast<double>(gimbal.pitch)));
}

struct LinearFit
{
    double slope{0.0};
    double intercept{0.0};
    double slope_error{1.0e9};
    double span{0.0};
    bool ok{false};
};

// 最小二乘直线 y = slope*t + intercept。t 必须按时间升序。
LinearFit fitLinear(const std::vector<double> &t, const std::vector<double> &y)
{
    LinearFit result;
    const std::size_t n = t.size();
    if (n < 3 || y.size() != n)
    {
        return result;
    }
    const double dn = static_cast<double>(n);
    double st = 0.0, sy = 0.0, stt = 0.0, sty = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        st += t[i];
        sy += y[i];
        stt += t[i] * t[i];
        sty += t[i] * y[i];
    }
    const double denom = dn * stt - st * st;
    if (!(std::fabs(denom) > 1.0e-12))
    {
        return result;
    }
    result.slope = (dn * sty - st * sy) / denom;
    result.intercept = (sy - result.slope * st) / dn;
    double sse = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        const double error = y[i] - (result.slope * t[i] + result.intercept);
        sse += error * error;
    }
    const double sxx = stt - st * st / dn;
    if (sxx > 1.0e-12 && n > 2)
    {
        result.slope_error = std::sqrt(std::max(sse, 0.0) / (dn - 2.0) / sxx);
    }
    result.span = t.back() - t.front();
    result.ok = std::isfinite(result.slope) && std::isfinite(result.intercept);
    return result;
}

bool predictionFinite(const PredictionResult &prediction)
{
    const float values[] = {prediction.center_x_m,
                            prediction.center_y_m,
                            prediction.center_velocity_x_m_s,
                            prediction.center_velocity_y_m_s,
                            prediction.body_yaw,
                            prediction.body_yaw_velocity,
                            prediction.radius_1_m,
                            prediction.radius_2_m,
                            prediction.distance_m,
                            prediction.target_yaw,
                            prediction.target_pitch,
                            prediction.target_yaw_velocity,
                            prediction.target_pitch_velocity};
    for (float value : values)
    {
        if (!std::isfinite(value))
        {
            return false;
        }
    }
    return prediction.target_id != 0 && prediction.state != TrackingState::NoTarget &&
           prediction.radius_1_m > 0.0F && prediction.distance_m >= 0.0F;
}

// 跟踪器里的调试打印总开关（默认关）。
//
// 关掉的原因：这些行以前是无条件打到 stderr 的，实跑时控制台会被 [ev]/[drop]/
// [cleargap] 刷满，反而把每 150 帧一行的耗时统计和拒绝原因挤没了。需要看细节时
// 设 TRACK_DEBUG=1（其它专门的开关仍各自独立：RATE_DEBUG / CIRCLE_DEBUG /
// TRACKLET_DEBUG / BATCH_DEBUG）。
bool trackDebugEnabled()
{
    static const bool enabled = std::getenv("TRACK_DEBUG") != nullptr;
    return enabled;
}

} // namespace

Eigen::Vector3d platePosition(const State &state, int plate_slot, int plate_count)
{
    if (plate_count <= 0)
    {
        return Eigen::Vector3d(state(0), state(2), state(4));
    }
    return measurement(state, plate_slot, plate_count);
}

RotatingArmorTracker::RotatingArmorTracker(TrackerConfig config)
    : config_(config), ekf_(config.ekf)
{
    normal_history_per_slot_.resize(
        static_cast<std::size_t>(std::max(config_.ekf.plate_count, 1)));
    reset();
}

void RotatingArmorTracker::applyProcessNoise()
{
    const double scale = std::max(1e-6, config_.process_accel_scale);
    EkfConfig config = config_.ekf;
    config.process.accel_x = 0.01 * scale;
    config.process.accel_y = 0.01 * scale;
    config.process.accel_z = 0.01 * scale;
    static const double spin_env = [] {
        const char *v = std::getenv("SPIN_Q");
        return v ? std::atof(v) : -1.0;
    }();
    double spin_scale =
        spin_env > 0.0 ? spin_env : std::max(1e-6, config_.process_spin_scale);
    if (spin_env < 0.0 && config_.adaptive_spin_q_enabled)
    {
        // |ω| 越大，需要的角加速度噪声越大（Q ∝ |ω|²）。
        const double ratio = std::fabs(measured_rate_ema_) /
                             std::max(1e-3, config_.adaptive_spin_rate_ref_rad_s);
        const double adaptive = std::clamp(ratio * ratio, 1.0, config_.adaptive_spin_q_max);
        spin_scale = std::max(spin_scale, adaptive);
    }
    config.process.angular_accel = 0.01 * scale * spin_scale;
    ekf_.setConfig(config);
}

bool RotatingArmorTracker::applyPhaseMeasurement(double world_yaw_rad, double sigma_rad)
{
    if (!ekf_.initialized() || !std::isfinite(world_yaw_rad) || !(sigma_rad > 0.0))
    {
        return false;
    }
    const int slot = aim_slot_ >= 0 ? aim_slot_ : 0;
    const Eigen::Vector3d plate_before = platePosition(ekf_.state(), slot, config_.ekf.plate_count);
    if (!ekf_.updatePhase(world_yaw_rad, sigma_rad))
    {
        return false;
    }
    if (config_.phase_anti_anchor)
    {
        const Eigen::Vector3d plate_after = platePosition(ekf_.state(), slot, config_.ekf.plate_count);
        ekf_.shiftCenter(plate_before - plate_after);
    }
    return true;
}

void RotatingArmorTracker::reset()
{
    ekf_.reset();
    applyProcessNoise();
    diagnostics_ = TrackerDiagnostics{};
    has_timestamp_ = false;
    last_timestamp_s_ = 0.0;
    state_timestamp_s_ = 0.0;
    initialized_ = false;
    target_id_ = 0;
    aim_slot_ = -1;
    last_seen_s_.fill(-1.0);
    radius_updates_ = 0;
    aim_smoothed_valid_ = false;
    aim_smoothed_s_ = -1.0;
    aim_history_count_.fill(0);
    last_accept_s_ = 0.0;
    has_accept_ = false;
    consecutive_updates_ = 0;
    missed_frames_ = 0;
    other_id_streak_ = 0;
    centered_streak_ = 0;
    contradiction_streak_ = 0;
    relock_count_ = 0;
    lock_samples_.clear();
    for (auto &history : normal_history_per_slot_)
    {
        history.clear();
    }
    locked_radius_m_ = 0.0;
    locked_spin_rad_s_ = 0.0;
    has_pair_relock_ = false;
    last_pair_relock_s_ = 0.0;
    timestamp_s_ = 0.0;
    lock_radius_from_data_ = false;
    lock_phase_flipped_ = false;
    smoother_ = ObservationSmoother{};
    noise_samples_.clear();
    observation_noise_m_ = config_.association_sigma_floor_m;
    last_updated_slot_ = -1;
}

void RotatingArmorTracker::dropTarget(const char *reason)
{
    if (trackDebugEnabled())
        std::fprintf(stderr, "[drop] %s\n", reason);
    resetRateWindow();
    if (config_.relock_reuse_enabled && ekf_.initialized())
    {
        snapshot_state_ = ekf_.state();
        snapshot_covariance_ = ekf_.covariance();
        snapshot_s_ = state_timestamp_s_;
        snapshot_id_ = target_id_;
        snapshot_phase_resolved_ = ekf_.phase_resolved();
        snapshot_spin_pinned_ = ekf_.spinPinned();
        has_snapshot_ = true;
    }
    reset();
    ++relock_count_;
    diagnostics_.relock_count = relock_count_;
    diagnostics_.last_reject = (reason != nullptr) ? reason : "";
}

void RotatingArmorTracker::collectWorldObservations(
    const std::vector<ArmorPose> &observations, const GimbalState &gimbal,
    std::vector<WorldObservation> &out) const
{
    out.clear();
    // 整帧的 camera/云台 -> 世界，只算一次。
    const Eigen::Matrix3d rotation = gimbalToWorld(gimbal);
    for (const ArmorPose &pose : observations)
    {
        const Eigen::Vector3d p_gimbal{pose.position_gimbal_m[0], pose.position_gimbal_m[1],
                                       pose.position_gimbal_m[2]};
        if (!p_gimbal.allFinite() || p_gimbal.norm() < 1.0e-3 ||
            !std::isfinite(pose.reprojection_error))
        {
            continue;
        }
        WorldObservation observation;
        observation.position = rotation * p_gimbal;
        observation.target_id = pose.detection.target_id;
        observation.size = pose.detection.size;
        // 重投影误差越大，位置越不可信：1x ~ 3x 的噪声缩放。
        // 观测可信度按重投影误差分档：重投影越干净越可信（下限 0.35 倍），
        // 越脏越不可信。目的是让"好观测"真正拽得动状态，而不是被过程模型压住。
        const double scale = clampd(0.5 + pose.reprojection_error / config_.position_sigma_quality_gain,
                                    config_.position_sigma_quality_floor, 3.0);
        observation.sigma_position_m = config_.position_sigma_m * scale;

        const Eigen::Vector3d n_gimbal{pose.normal_gimbal[0], pose.normal_gimbal[1],
                                       pose.normal_gimbal[2]};
        if (n_gimbal.allFinite() && n_gimbal.norm() > 0.5)
        {
            const Eigen::Vector3d n_world = rotation * (n_gimbal / n_gimbal.norm());
            if (n_world.allFinite() && std::hypot(n_world.x(), n_world.y()) > 1.0e-3)
            {
                observation.has_yaw = true;
                observation.yaw = vectorYaw(n_world);
            }
        }
        if (!observation.position.allFinite())
        {
            continue;
        }
        out.push_back(observation);
    }
}

RotatingArmorTracker::CircleFit RotatingArmorTracker::fitCenterFixedRadius(
    const std::deque<LockSample> &samples, double radius, const CircleFit &initial) {
    // 半径钉住、只解圆心（Gauss-Newton，2 个未知量）。自由三参数拟合在短弧上
    // 病态：实拍 v4 末段 31 个样本只走 0.2 m，自由拟合把半径压到 0.101（真值
    // 0.20），圆心偏移让每步角增量正负相消，ω 落到 0.37 rad/s 而车实际以
    // ~8 rad/s 自转。半径是按板型的物理常数，钉住它之后条件数好得多。
    CircleFit fit;
    fit.radius = radius;
    if (samples.size() < 4 || !(radius > 0.0)) {
        return fit;
    }
    double cx = initial.ok ? initial.cx : 0.0;
    double cy = initial.ok ? initial.cy : 0.0;
    if (!initial.ok) {
        for (const LockSample &sample : samples) {
            cx += sample.position.x();
            cy += sample.position.y();
        }
        cx /= static_cast<double>(samples.size());
        cy /= static_cast<double>(samples.size());
    }
    for (int iteration = 0; iteration < 20; ++iteration) {
        Eigen::Matrix2d normal = Eigen::Matrix2d::Zero();
        Eigen::Vector2d rhs = Eigen::Vector2d::Zero();
        for (const LockSample &sample : samples) {
            const Eigen::Vector2d delta(cx - sample.position.x(), cy - sample.position.y());
            const double distance = delta.norm();
            if (distance < 1e-6) {
                continue;
            }
            const Eigen::Vector2d jacobian = delta / distance;
            const double residual = distance - radius;
            normal += jacobian * jacobian.transpose();
            rhs -= jacobian * residual;
        }
        if (std::fabs(normal.determinant()) < 1e-12) {
            break;
        }
        const Eigen::Vector2d step = normal.inverse() * rhs;
        cx += step.x();
        cy += step.y();
        if (step.norm() < 1e-6) {
            break;
        }
    }
    double sum = 0.0;
    for (const LockSample &sample : samples) {
        const double distance =
            (Eigen::Vector2d(cx, cy) - Eigen::Vector2d(sample.position.x(), sample.position.y()))
                .norm();
        sum += (distance - radius) * (distance - radius);
    }
    fit.cx = cx;
    fit.cy = cy;
    fit.residual = std::sqrt(sum / static_cast<double>(samples.size()));
    fit.ok = true;
    return fit;
}

RotatingArmorTracker::BodyBatchFit
RotatingArmorTracker::fitBodyBatch(const std::deque<LockSample> &samples, int plate_count,
                                  double omega_hint, double config_coverage_min,
                                  bool fold_phase, bool alias_gate)
{
    // 关键：一帧里看到的可能是四块板中的任意一块，所以板心角度只能定到
    // mod (2pi/N)。先按"整周展开"拟合转速是错的（实测会解出 ω≈0），
    // 正确做法是把残留折叠到 2pi/N 再拟合：对 ω 做网格搜索，
    // 每个 ω 用圆均值求 φ，取折叠残差最小的那一组。
    BodyBatchFit out;
    if (samples.size() < 4)
    {
        return out;
    }
    const double period = 2.0 * kPi / static_cast<double>(std::max(plate_count, 1));
    const double t_ref = samples.front().timestamp_s;
    // 板心角度只能定到 mod 2pi/plate_count，圆均值必须先把角度乘上 plate_count
    // 再平均（否则四块板各差 90°，求和会互相抵消，均值是任意的）。
    const double fold_factor = fold_phase ? static_cast<double>(std::max(plate_count, 1)) : 1.0;
    std::deque<LockSample> work = samples;
    CircleFit circle;
    double best_rate = 0.0;
    double best_phase = 0.0;
    double best_cost = 1.0e18;
    double best_rms = 0.0;
    for (int pass = 0; pass < 2; ++pass)
    {
        circle = fitArmorCircle(work);
        if (!circle.ok || !(circle.radius > 1.0e-3))
        {
            return out;
        }
        std::vector<double> theta(work.size(), 0.0);
        std::vector<double> tt(work.size(), 0.0);
        for (std::size_t i = 0; i < work.size(); ++i)
        {
            theta[i] = std::atan2(work[i].position.y() - circle.cy, work[i].position.x() - circle.cx);
            tt[i] = work[i].timestamp_s - t_ref;
        }
        best_cost = 1.0e18;
        std::vector<std::pair<double, double>> candidates; // (rate, cost)
        candidates.reserve(1300);
        for (double rate = -12.0; rate <= 12.0 + 1.0e-9; rate += 0.02)
        {
            double sum_sin = 0.0;
            double sum_cos = 0.0;
            for (std::size_t i = 0; i < theta.size(); ++i)
            {
                const double a = (theta[i] - rate * tt[i]) * fold_factor;
                sum_sin += std::sin(a);
                sum_cos += std::cos(a);
            }
            const double mean_angle = std::atan2(sum_sin, sum_cos) / fold_factor;
            double cost = 0.0;
            for (std::size_t i = 0; i < theta.size(); ++i)
            {
                const double r = wrapAngle(theta[i] - rate * tt[i] - mean_angle);
                const double rf = r - period * std::round(r / period);
                cost += rf * rf;
            }
            candidates.emplace_back(rate, cost);
            if (cost < best_cost)
            {
                best_cost = cost;
                best_rate = rate;
                best_phase = mean_angle;
            }
        }
        // ω 的折叠别名：四块板相隔 2pi/N，ω 与 ω ± (2pi/N)/Δt 在短窗口上几乎
        // 一样好。取所有代价接近最优的候选里、离 EKF 预测最近的那个，
        // 否则单板可见时会跳到别名上（合成回归 A/G/H 就是这么挂的）。
        if (alias_gate && std::isfinite(omega_hint) && !candidates.empty())
        {
            double best_score = 1.0e18;
            for (const auto &cand : candidates)
            {
                if (cand.second > best_cost * 1.05 + 1.0e-6)
                {
                    continue;
                }
                const double score = std::fabs(cand.first - omega_hint);
                if (score < best_score)
                {
                    best_score = score;
                    best_rate = cand.first;
                }
            }
        }
        {
            double sum_sin = 0.0;
            double sum_cos = 0.0;
            for (std::size_t i = 0; i < theta.size(); ++i)
            {
                const double a = (theta[i] - best_rate * tt[i]) * fold_factor;
                sum_sin += std::sin(a);
                sum_cos += std::cos(a);
            }
            best_phase = std::atan2(sum_sin, sum_cos) / fold_factor;
            double cost = 0.0;
            for (std::size_t i = 0; i < theta.size(); ++i)
            {
                const double rr = wrapAngle(theta[i] - best_rate * tt[i] - best_phase);
                const double r = rr - period * std::round(rr / period);
                cost += r * r;
            }
            best_cost = cost;
        }
        best_rms = std::sqrt(best_cost / static_cast<double>(theta.size()));
        // 剔除折叠残差离群的样本，用内点重拟合圆
        std::deque<LockSample> kept;
        for (std::size_t i = 0; i < theta.size(); ++i)
        {
            const double rr = wrapAngle(theta[i] - best_rate * tt[i] - best_phase);
            const double r = rr - period * std::round(rr / period);
            if (std::fabs(r) <= std::max(2.5 * best_rms, 0.10))
            {
                kept.push_back(work[i]);
            }
        }
        if (kept.size() < 6 || kept.size() == work.size())
        {
            break;
        }
        work.swap(kept);
    }
    // 角度覆盖度：静止目标所有样本在同一个角度上，覆盖度≈0；只有板真的
    // 转起来（覆盖 >= 1.2 rad ≈ 70°）时相位和 ω 才可辨识。
    double coverage = 0.0;
    {
        std::vector<double> ang;
        ang.reserve(work.size());
        for (const LockSample &sample : work)
        {
            ang.push_back(std::atan2(sample.position.y() - circle.cy,
                                     sample.position.x() - circle.cx));
        }
        std::sort(ang.begin(), ang.end());
        double max_gap = ang.empty() ? 2.0 * kPi : (ang.front() + 2.0 * kPi - ang.back());
        for (std::size_t i = 1; i < ang.size(); ++i)
        {
            max_gap = std::max(max_gap, ang[i] - ang[i - 1]);
        }
        coverage = 2.0 * kPi - max_gap;
    }
    out.ok = circle.residual <= 0.10 && std::isfinite(best_rate) && std::isfinite(best_phase) &&
             coverage >= config_coverage_min;
    out.coverage_rad = coverage;
    out.center = Eigen::Vector2d(circle.cx, circle.cy);
    out.radius = circle.radius;
    out.rate_rad_s = best_rate;
    // 模型里 plate = centre − R·[cos,sin](phase) ⇒ 板心角度 = phase + pi
    out.phase_rad = wrapAngle(best_phase - kPi);
    out.circle_residual_m = circle.residual;
    out.phase_residual_rad = best_rms;
    out.used = static_cast<int>(work.size());
    return out;
}

bool RotatingArmorTracker::estimateRateAlongTracklet(const std::deque<LockSample> &samples,
                                                     const CircleFit &centre, double *rate_rad_s,
                                                     double *arc_rad, double *residual_m)
{
    if (!centre.ok || samples.size() < 3)
    {
        return false;
    }
    const std::size_t n = samples.size();
    std::vector<double> t(n), th(n);
    double prev = 0.0;
    double unwrapped = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        const double dx = samples[i].position.x() - centre.cx;
        const double dy = samples[i].position.y() - centre.cy;
        const double angle = std::atan2(dy, dx);
        if (i == 0)
        {
            unwrapped = angle;
        }
        else
        {
            double step = angle - prev;
            step -= 2.0 * kPi * std::round(step / (2.0 * kPi));
            unwrapped += step;
        }
        prev = angle;
        t[i] = samples[i].timestamp_s - samples.front().timestamp_s;
        th[i] = unwrapped;
    }
    double mean_t = 0.0;
    double mean_th = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        mean_t += t[i];
        mean_th += th[i];
    }
    mean_t /= static_cast<double>(n);
    mean_th /= static_cast<double>(n);
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        num += (t[i] - mean_t) * (th[i] - mean_th);
        den += (t[i] - mean_t) * (t[i] - mean_t);
    }
    if (!(den > 1.0e-9))
    {
        return false;
    }
    const double rate = num / den;
    double sq = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        const double model = mean_th + rate * (t[i] - mean_t);
        sq += (th[i] - model) * (th[i] - model);
    }
    const double angle_rms = std::sqrt(sq / static_cast<double>(n));
    if (rate_rad_s != nullptr)
    {
        *rate_rad_s = rate;
    }
    if (arc_rad != nullptr)
    {
        *arc_rad = th.back() - th.front();
    }
    if (residual_m != nullptr)
    {
        *residual_m = centre.radius * angle_rms;
    }
    return true;
}

RotatingArmorTracker::CircleFit
RotatingArmorTracker::fitArmorCircle(const std::deque<LockSample> &samples)
{
    // Kasa 代数圆拟合：min sum (|p-c|^2 - R^2)^2，闭式解，对"接近共线"的
    // 短弧也稳定（Gauss-Newton 在短弧上会发散，实测给出 R=3142 m 这种值）。
    CircleFit fit;
    if (samples.size() < 4)
    {
        return fit;
    }
    Eigen::MatrixXd a = Eigen::MatrixXd::Zero(static_cast<int>(samples.size()), 3);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(static_cast<int>(samples.size()));
    for (std::size_t i = 0; i < samples.size(); ++i)
    {
        const double x = samples[i].position.x();
        const double y = samples[i].position.y();
        const int row = static_cast<int>(i);
        a(row, 0) = 2.0 * x;
        a(row, 1) = 2.0 * y;
        a(row, 2) = 1.0;
        b(row) = x * x + y * y;
    }
    const Eigen::Vector3d solution = a.colPivHouseholderQr().solve(b);
    if (!solution.allFinite())
    {
        return fit;
    }
    const double cx = solution(0);
    const double cy = solution(1);
    const double radius_sq = solution(2) + cx * cx + cy * cy;
    if (!(radius_sq > 0.0))
    {
        return fit;
    }
    const double radius = std::sqrt(radius_sq);
    double sum = 0.0;
    for (const LockSample &sample : samples)
    {
        const double residual = std::hypot(sample.position.x() - cx, sample.position.y() - cy) - radius;
        sum += residual * residual;
    }
    fit.cx = cx;
    fit.cy = cy;
    fit.radius = radius;
    fit.residual = std::sqrt(sum / static_cast<double>(samples.size()));
    fit.ok = std::isfinite(fit.cx) && std::isfinite(fit.cy) && std::isfinite(fit.radius) &&
             fit.radius > 0.0 && fit.residual < 1.0;
    return fit;
}

bool RotatingArmorTracker::tryLock(const std::vector<WorldObservation> &world,
                                   double timestamp_s, const GimbalState &gimbal)
{
    // 1) 攒样本：只收"带板面法向 + 有编号"的观测，并且跟上一帧挂上关系。
    //
    //    两种关系都允许：
    //    - 同板（法向差 <= 0.45 rad）：链上的正常一步；
    //    - 换板（法向差接近 pi/2 的整数倍）：旋转车每转 90 度就换一块板露出来，
    //      这时单板位置会跳、但车体相位是连续的。旧实现把换板直接丢掉，于是
    //      旋转目标的链永远长不到 lock_min_span_s（实测卡在 10 个样本，
    //      last_reject=lock:waiting_continuity），一次也锁不上。
    //
    //    换板的样本仍然保留（几何上它是另一块板的真实位置），但要打标记：
    //    算角速度时必须跳过它，否则 90 度的跳变会被当成"转得飞快"。
    const WorldObservation *pick = nullptr;
    bool pick_handoff = false;
    double best_angle = std::numeric_limits<double>::infinity();
    const bool have_last = !lock_samples_.empty();
    const double last_yaw = have_last ? lock_samples_.back().yaw : 0.0;
    const int last_id = have_last ? lock_samples_.back().target_id : 0;
    for (const WorldObservation &observation : world)
    {
        if (observation.target_id == 0 || !observation.has_yaw)
        {
            continue;
        }
        if (have_last && observation.target_id != last_id)
        {
            continue;
        }
        const double delta = wrapToPi(observation.yaw - last_yaw);
        bool handoff = false;
        if (have_last)
        {
            // 同板门限要比法向抖动宽：板面法向的单帧抖动在 0.05 rad 量级，
            // 而合成夹具里故意放到了 0.35 rad，0.45 rad 的门限会把 ~18% 的
            // 相邻帧判成"断链"，链永远长不到 10 个样本。1.2 rad 仍远小于
            // pi/2 = 1.57，换板照样能区分。
            if (std::fabs(delta) <= 1.2)
            {
                handoff = false;
            }
            else if (std::fabs(delta) < kHalfPi + 0.7)
            {
                const double residual =
                    delta - std::round(delta / kHalfPi) * kHalfPi;
                if (std::fabs(residual) > 0.6)
                {
                    continue; // 既不是同板也不是换板：多半是别的目标
                }
                handoff = true;
            }
            else
            {
                continue;
            }
        }
        if (!have_last)
        {
            // 链还没开始：选"离准星最近"的那块板。自瞄的语义是"瞄我现在正对着的
            // 那台车"，不是"瞄检测列表里排第一的那台"（实拍 v4 两台车同框时，
            // 旧实现永远锁到列表里的第一台，左边那台自转车根本没被看过）。
            const double angle = boresightAngle(observation.position, gimbal);
            if (pick == nullptr || angle < best_angle)
            {
                pick = &observation;
                pick_handoff = handoff;
                best_angle = angle;
            }
        }
        else if (pick == nullptr ||
                 (pick_handoff && !handoff) ||
                 (!pick_handoff && !handoff &&
                  std::fabs(delta) < std::fabs(wrapToPi(pick->yaw - last_yaw))))
        {
            pick = &observation;
            pick_handoff = handoff;
        }
    }
    if (pick == nullptr && have_last)
    {
        diagnostics_.lock_samples = static_cast<int>(lock_samples_.size());
        diagnostics_.last_reject = "lock:waiting_continuity";
        return false;
    }
    if (pick != nullptr)
    {
        lock_samples_.push_back(
            LockSample{timestamp_s, pick->position, pick->yaw, pick->target_id, pick->size,
                       pick_handoff});
    }
    while (!lock_samples_.empty() &&
           timestamp_s - lock_samples_.front().timestamp_s > config_.lock_max_age_s)
    {
        lock_samples_.pop_front();
    }

    // 双板直解：同一帧里同编号的两块相邻板（法向差约 90 度）可以同时解出中心与半径。
    //   p_i - p_j = R * (n_i - n_j)  =>  R = |p_i - p_j| / |n_i - n_j|，中心 = p_i - R*n_i
    // 这比"单帧板面法向 + 半径先验"稳得多：实拍里小目标法向抖动 10 度以上，单帧定出
    // 的中心会偏 ~R；而两块板是两条射线的交点，法向误差只会让交点沿径向轻微缩放。
    const int pair_target_id = pick != nullptr
                                   ? pick->target_id
                                   : (lock_samples_.empty() ? 0 : lock_samples_.back().target_id);
    Eigen::Vector3d pair_position(0.0, 0.0, 0.0);
    Eigen::Vector2d pair_center(0.0, 0.0);
    double pair_radius = 0.0;
    double pair_yaw = 0.0;
    bool pair_ok = false;
    for (std::size_t i = 0; i < world.size() && !pair_ok; ++i)
    {
        if (world[i].target_id != pair_target_id)
        {
            continue;
        }
        for (std::size_t j = i + 1; j < world.size(); ++j)
        {
            if (world[j].target_id != pair_target_id)
            {
                continue;
            }
            const Eigen::Vector2d ni(std::cos(world[i].yaw), std::sin(world[i].yaw));
            const Eigen::Vector2d nj(std::cos(world[j].yaw), std::sin(world[j].yaw));
            const Eigen::Vector2d ndiff = ni - nj;
            if (ndiff.norm() < 0.6)
            {
                continue; // 法向太接近：不是相邻槽位，解退化
            }
            const Eigen::Vector2d pi2(world[i].position.x(), world[i].position.y());
            const Eigen::Vector2d pj2(world[j].position.x(), world[j].position.y());
            // 用"模型构型"的半角而不是实测法向差：法向本身有 ±20° 噪声，直接
            // 除 |n_i-n_j| 会把半径算大（实拍 v4：弦 0.40、法向差 80° → 0.31，
            // 而相邻构型给出的真值是 0.283）。四板模型只有相邻(90°)和对置(180°)。
            const double dpsi_pair = std::fabs(foldPeriod(world[j].yaw - world[i].yaw, 2.0 * kPi));
            const double half_pair =
                dpsi_pair > config_.radius_update_opposite_yaw_rad ? 0.5 * kPi : 0.25 * kPi;
            const double candidate_radius = (pi2 - pj2).norm() / (2.0 * std::sin(half_pair));
            if (candidate_radius < config_.radius_min_m ||
                candidate_radius > config_.radius_max_m)
            {
                continue;
            }
            // 两块板解出的共同点：p_i + R*n_i = p_j + R*n_j。由"两块板都满足"
            // 这件事本身可知 n 是从板指向车体中心的（内向），故中心 = p + R*n。
            const Eigen::Vector2d candidate_center = pi2 + candidate_radius * ni;
            if ((candidate_center - candidate_radius * nj - pj2).norm() > 0.10)
            {
                continue; // 两块板解出的中心不自洽
            }
            pair_position = world[i].position;
            pair_center = candidate_center;
            pair_radius = candidate_radius;
            pair_yaw = std::atan2(pair_position.y() - pair_center.y(),
                                  pair_position.x() - pair_center.x());
            pair_ok = true;
            break;
        }
    }

    diagnostics_.lock_samples = static_cast<int>(lock_samples_.size());
    const int min_samples =
        pair_ok ? 3 : std::max(config_.lock_min_samples, 6);
    if (static_cast<int>(lock_samples_.size()) < min_samples)
    {
        diagnostics_.last_reject = "lock:waiting_samples";
        return false;
    }
    const double span =
        lock_samples_.back().timestamp_s - lock_samples_.front().timestamp_s;
    const double min_span = pair_ok ? 0.05 : config_.lock_min_span_s;
    if (span < min_span)
    {
        diagnostics_.last_reject = "lock:waiting_span";
        return false;
    }

    // 2) 相位序列：按 wrapToPi 解缠（几何正确，换板带来的 90 度就是真实的
    //    另一块板的位置）。角速度只用"同板步长"算，并对步长取中位数抗离群。
    const std::size_t count = lock_samples_.size();
    const double t_last = lock_samples_.back().timestamp_s;
    // 2b) 角速度：把"换板跳变"从相位里减掉，得到车体的连续相位，再整段拟合。
    //
    //     换板那一步的法向跳变是 pi/2 的整数倍，把这个整数倍扣掉之后剩下的
    //     才是车体真正转过的角度。扣完以后所有样本都能参与拟合（不必只挑一段
    //     同板的），166fps 下单帧 0.03 rad 的弱信号也能靠 0.3-0.5 s 的基线
    //     平均出来；否则一旦亮出来的板每几帧就换一次，最长同板段不足 5 个样本，
    //     速率永远测不出来（合成夹具里就是这个现象：reject=lock:rate_ambiguous）。
    std::vector<double> times(count, 0.0);
    std::vector<double> phases(count, 0.0);
    std::vector<double> body(count, 0.0);
    for (std::size_t i = 0; i < count; ++i)
    {
        times[i] = lock_samples_[i].timestamp_s - t_last;
        if (i == 0)
        {
            phases[0] = lock_samples_[0].yaw;
            body[0] = 0.0;
            continue;
        }
        const double delta = wrapToPi(lock_samples_[i].yaw - lock_samples_[i - 1].yaw);
        phases[i] = phases[i - 1] + delta;
        const double jump = lock_samples_[i].handoff
                                ? std::round(delta / kHalfPi) * kHalfPi
                                : 0.0;
        body[i] = body[i - 1] + (delta - jump);
    }
    const LinearFit fit = fitLinear(times, body);
    if (!fit.ok || count < 6)
    {
        diagnostics_.last_reject = "lock:rate_ambiguous";
        return false;
    }
    double omega = fit.slope;
    double slope_error = fit.slope_error;
    // 法向序列给出的速率"可用"的判据：斜率和标准误同量级时不可信。
    const bool yaw_rate_reliable =
        std::fabs(omega) > 1.0e-3 && slope_error < 0.5 * std::fabs(omega) + 0.05;
    // 但"法向序列可靠"只能说明它自己一致，不代表它能看见高速自转：板面法向
    // 每 90° 复位一次，10 rad/s 在 31 个样本的窗口里会被折回成 0.2~0.5 rad/s
    // 的假斜率（实测 v4 自转车锁定时刻法向斜率 -0.244，真实转速 10.4）。所以
    // 法向速率只有大到"不可能是折返"时才算数，否则让位置圆拟合说了算。
    const bool yaw_sees_fast_spin = yaw_rate_reliable && std::fabs(omega) >= 1.0;

    double phase_spread = 0.0;
    for (std::size_t i = 0; i < count; ++i)
    {
        phase_spread = std::max(phase_spread, std::fabs(body[i] - body[count - 1]));
    }

    // 2c) 位置圆拟合：高速自转目标的速率必须从"板心位置"估，不能从板面 yaw 估。
    //
    //     实拍 v4 的 3 号车自转时，单帧 yaw 抖动 ±0.3 rad，而单帧真实转角只有
    //     0.01-0.02 rad —— yaw 序列看起来就是噪声，线性拟合给出 |omega|≈0，
    //     于是锁定按 Static 初始化，转速永久钉在 0，预测永远滞后。
    //
    //     板心位置没有这个问题（重投影 0.5 px）：四块板共圆，圆拟合给出
    //     (c, R)，相邻板心的切向位移 delta_theta ≈ cross(r, dp)/R^2 累加给出
    //     omega 的大小和方向。换板那一步位移 ~2R，用步长门限剔除。
    // 半径先按板型取物理先验；自由三参数拟合只在它离先验不远时才采信半径值，
    // 否则改用"钉住半径、只解圆心"的拟合（短弧上条件数好得多）。
    // --radius / radius_prior_m 显式给值时必须真的生效：以前这里无条件从
    // radius_prior_small_m 起步，于是"小板"永远按 0.20 兜底，--radius 对小板无效
    // （实拍 v4 的 3 号道具半径 0.28，被钉在 0.20，模型四块板整体小一圈）。
    double size_prior = config_.radius_prior_from_size ? config_.radius_prior_small_m
                                                       : config_.radius_prior_m;
    if (config_.radius_prior_from_size && lock_samples_.back().size == ArmorSize::Large)
    {
        size_prior = config_.radius_prior_large_m;
    }
    size_prior = clampd(size_prior, config_.radius_min_m, config_.radius_max_m);
    CircleFit circle = fitArmorCircle(lock_samples_);
    bool used_fixed_radius = false;
    // 半径优先用"板心圆拟合"算出来的值：四块板共圆，(c, R) 是几何量，本来就该
    // 从识别到的板位置解出来，而不是拿板型先验去覆盖它。
    //
    // 以前这里只要 |circle.radius - size_prior| > 0.3*size_prior 就丢弃拟合半径、
    // 改用"钉住先验只解圆心"。实拍 v4 的 3 号道具真实半径 0.28，而小装甲板先验
    // 0.20，差了 40% 正好越过 30% 的门限，于是每帧都被钉回 0.20——模型四块板
    // 整体小一圈，圆心跟着偏，画面上就是"绿板转心偏高、蓝板转一半卡回去"。
    // 现在先验只当哨兵：拟合值落在物理区间内就用它。
    // 注意这是"拟合半径落在物理区间内就用拟合值"，不是反过来：
    // 写成 !(r > min) || !(r < max) 等于 (r <= min) || (r >= max)，含义完全反了
    // ——那样只有拟合失败时才用固定半径，而拟合成功时反而被钉回先验（实测 v4
    // R 拟合 0.284 却被钉在 0.20，模型四块板整体小一圈）。
    // 拟合半径落在物理区间内就保持"自由拟合"的结果；只有拟合值离谱（出区间）
    // 时才退回"钉住板型先验、只解圆心"。
    //
    // 这里踩过两次坑，写清楚免得再改错：
    //  1) 原写法是 !(r > min) || !(r < max)，等价于 (r <= min) || (r >= max)，
    //     即"拟合值出区间才进来"，与它自己的注释相反但**行为上是对的**（v4 拟合
    //     0.284、区间内，于是保持自由拟合）。我把它"改正"成小区间判断后，v4 的
    //     id=3 中位从 91.2 变 99.9 px，配对比较 96.0 -> 103.4，是回归。
    //  2) 又试了"跨锁对拟合半径做 EMA 再解圆心"，配对比较 91.9 -> 107.9，更差。
    // 所以 v4 上最优仍是：拟合值可用就照用，别平滑、别重解圆心。
    const bool fit_in_range = circle.ok && circle.radius > config_.radius_min_m &&
                              circle.radius < config_.radius_max_m;
    if (!fit_in_range)
    {
        const CircleFit fixed = fitCenterFixedRadius(lock_samples_, size_prior, circle);
        if (fixed.ok)
        {
            circle = fixed;
            used_fixed_radius = true;
        }
    }
    bool spin_from_positions = false;
    double circle_phase_last = 0.0;
    double circle_step_ratio = 0.0;
    if (circle.ok && circle.residual <= config_.lock_circle_residual_m &&
        circle.radius >= config_.radius_min_m && circle.radius <= config_.radius_max_m &&
        count >= 6)
    {
        double sum_dtheta = 0.0;
        double sum_dt = 0.0;
        double sum_dtheta2 = 0.0;
        int used_steps = 0;
        int total_steps = 0;
        for (std::size_t i = 1; i < count; ++i)
        {
            const Eigen::Vector2d prev(lock_samples_[i - 1].position.x(),
                                       lock_samples_[i - 1].position.y());
            const Eigen::Vector2d cur(lock_samples_[i].position.x(),
                                      lock_samples_[i].position.y());
            const double dt =
                lock_samples_[i].timestamp_s - lock_samples_[i - 1].timestamp_s;
            if (dt <= 1.0e-6)
            {
                continue;
            }
            ++total_steps;
            const Eigen::Vector2d delta = cur - prev;
            if (delta.norm() > config_.lock_circle_max_step_ratio * circle.radius)
            {
                continue; // 换板跳变（~2R）或漏检后的跳跃，不参与速率
            }
            const Eigen::Vector2d radius_vector =
                prev - Eigen::Vector2d(circle.cx, circle.cy);
            const double dtheta = (radius_vector.x() * delta.y() -
                                   radius_vector.y() * delta.x()) /
                                  (circle.radius * circle.radius);
            sum_dtheta += dtheta;
            sum_dtheta2 += dtheta * dtheta;
            sum_dt += dt;
            ++used_steps;
        }
        // 角跨度：短弧上圆拟合是病态的（实拍 v4 末段 31 个样本只走了 0.2 m，
        // 自由拟合把半径压到 0.101、每步角增量正负相消，ω 落在 0.37 rad/s 而
        // 真值约 8）。跨度不够时改用"弦长 + 先验半径"解整段转过的角度：
        //     chord = 2R sin(dtheta/2)  =>  dtheta = 2 asin(chord / 2R)
        // 方向取法向序列的净变化符号（1 rad 量级的转角远大于法向噪声）。
        double angular_span = 0.0;
        {
            const double reference = std::atan2(
                lock_samples_.front().position.y() - circle.cy,
                lock_samples_.front().position.x() - circle.cx);
            for (const LockSample &sample : lock_samples_)
            {
                const double angle =
                    std::atan2(sample.position.y() - circle.cy, sample.position.x() - circle.cx);
                angular_span = std::max(angular_span, std::fabs(wrapToPi(angle - reference)));
            }
        }
        const bool span_ok = angular_span >= config_.lock_circle_min_span_rad;
        // 弦长兜底只在"法向速率不可靠"时用：法向拟合可靠时它是更好的估计，
        // 不能被短弧兜底覆盖（合成场景 A 的角速度辨识会因此失效）。
        if (!span_ok && !yaw_rate_reliable)
        {
            const Eigen::Vector3d chord =
                lock_samples_.back().position - lock_samples_.front().position;
            const double chord_length = chord.norm();
            const double window_dt =
                lock_samples_.back().timestamp_s - lock_samples_.front().timestamp_s;
            double net_yaw = 0.0;
            for (std::size_t i = 1; i < count; ++i)
            {
                if (lock_samples_[i].handoff || lock_samples_[i - 1].handoff)
                {
                    continue;
                }
                net_yaw += wrapToPi(lock_samples_[i].yaw - lock_samples_[i - 1].yaw);
            }
            if (chord_length > 1e-3 && window_dt > 1e-4 && chord_length <= 2.0 * size_prior)
            {
                double dtheta = 2.0 * std::asin(
                    std::min(1.0, chord_length / (2.0 * size_prior)));
                dtheta *= (net_yaw < 0.0) ? -1.0 : 1.0;
                omega = dtheta / window_dt;
                sum_dtheta = dtheta;
                sum_dt = window_dt;
                used_steps = std::max(used_steps, 6);
                // 弦长只回答"这一段总共转了多少"，没回答"转得稳不稳"：手持
                // 抖动同样造得出弦长（回放 v1 的 23 个样本被解成 -4 rad/s，
                // 而车其实没转）。所以不能声称零误差——claim 零会把先验 sigma
                // 压到 1.0，窗口和衰减要花 0.6 s 才能把它掰回来。
                slope_error = clampd(0.5 * std::fabs(omega), 0.5, 3.0);
                spin_from_positions = true;
                circle_phase_last = std::atan2(
                    lock_samples_.back().position.y() - circle.cy,
                    lock_samples_.back().position.x() - circle.cx);
            }
        }
        const double used_ratio =
            total_steps > 0 ? static_cast<double>(used_steps) / total_steps : 0.0;
        circle_step_ratio = used_ratio;
        if (!spin_from_positions && span_ok && used_steps >= 5 && sum_dt > 0.0 &&
            used_ratio >= config_.lock_circle_trusted_ratio)
        {
            const double omega_positions = sum_dtheta / sum_dt;
            // 采信位置圆拟合的条件：
            //   0) 圆本身合理（平移目标拟合出来的圆半径会是几十米，不能拿它
            //      定相位/内外侧）：R 在板型先验附近；
            //   1) 法向序列本身不可靠；或
            //   2) 位置说"很快"（>3 rad/s 且是法向估计两倍以上）——这是 90°
            //      折返的典型症状（法向每 90° 复位，10 rad/s 在 31 样本窗口里
            //      会折成 0.2~0.5 rad/s 的假斜率）。
            const bool circle_plausible =
                std::isfinite(circle.radius) &&
                circle.radius >= 0.5 * config_.radius_min_m &&
                circle.radius <= 2.0 * config_.radius_max_m;
            const bool positions_win =
                circle_plausible &&
                (!yaw_rate_reliable ||
                 (std::fabs(omega_positions) > 3.0 &&
                  std::fabs(omega_positions) > 2.0 * std::fabs(omega)));
            if (positions_win)
            {
                const double mean_dtheta = sum_dtheta / static_cast<double>(used_steps);
                const double variance = std::max(
                    0.0,
                    sum_dtheta2 / static_cast<double>(used_steps) - mean_dtheta * mean_dtheta);
                // 单步角增量的散布折算到"速率标准误"
                const double mean_dt = sum_dt / static_cast<double>(used_steps);
                slope_error = std::sqrt(variance) / (std::max(1, used_steps) * mean_dt);
                omega = omega_positions;
                spin_from_positions = true;
                circle_phase_last = std::atan2(
                    lock_samples_.back().position.y() - circle.cy,
                    lock_samples_.back().position.x() - circle.cx);
            }
        }
    }

    const LockSample last = lock_samples_.back();

    // 半径是"按板型的物理常数"。短弧上圆拟合/相位回归对 R 是病态的（实拍见过
    // 0.055、0.145 这种解），所以任何"数据估出来的 R"都必须落在先验附近才采信；
    // 偏太远就退回先验——宁可板位整体偏一点，也不要让半径塌掉后整段跟踪都错。
    const double data_radius_tolerance = 0.35 * size_prior;

    double radius =
        clampd(config_.radius_prior_m, config_.radius_min_m, config_.radius_max_m);
    bool radius_from_data = false;

    // 3) 半径与"车体在板的哪一侧"：解线性最小二乘
    //      p_k = c + v*(t_k - t_last) - R*[cos(phi_k), sin(phi_k)]
    //    未知量 (cx, cy, vx, vy, R)。R 的符号直接回答法向是朝外还是朝内。
    //    目标不转（相位几乎不扫）时 R 与中心不可分辨，退回"车体在板背后"的
    //    几何先验 + 配置半径。
    const double ray = std::atan2(last.position.y(), last.position.x());
    const double inside = std::fabs(wrapToPi(wrapToPi(last.yaw) - ray));
    const double outside = std::fabs(wrapToPi(wrapToPi(last.yaw + kPi) - ray));
    bool flipped = outside < inside;
    if (spin_from_positions)
    {
        // 有了圆拟合中心，"板面外法向"可以直接定出来：(p - c)/R。板面法向的
        // 180 度歧义在这里一次解决，不再依赖"车体在板背后"的几何先验。
        const Eigen::Vector2d outward(last.position.x() - circle.cx,
                                      last.position.y() - circle.cy);
        const Eigen::Vector2d yaw_dir(std::cos(last.yaw), std::sin(last.yaw));
        flipped = outward.dot(yaw_dir) < 0.0;
    }

    if (phase_spread >= config_.phase_spread_for_radius_rad && count >= 12)
    {
        const int rows = static_cast<int>(count) * 2;
        Eigen::MatrixXd a = Eigen::MatrixXd::Zero(rows, 5);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(rows);
        for (std::size_t i = 0; i < count; ++i)
        {
            const double dt = times[i];
            const double c = std::cos(phases[i]);
            const double s = std::sin(phases[i]);
            const int row = static_cast<int>(i) * 2;
            a(row, 0) = 1.0;
            a(row, 2) = dt;
            a(row, 4) = -c;
            b(row) = lock_samples_[i].position.x();
            a(row + 1, 1) = 1.0;
            a(row + 1, 3) = dt;
            a(row + 1, 4) = -s;
            b(row + 1) = lock_samples_[i].position.y();
        }
        const Eigen::VectorXd solution = a.colPivHouseholderQr().solve(b);
        if (solution.allFinite())
        {
            const double fitted = solution(4);
            const bool fitted_flipped = fitted < 0.0;
            // 拟合出的相位方向必须和几何先验一致，否则说明这段数据还不足以
            // 把"半径 vs 中心"分开，宁可退回先验半径。
            if (std::fabs(fitted) >= config_.radius_min_m &&
                std::fabs(fitted) <= config_.radius_max_m && fitted_flipped == flipped &&
                std::fabs(std::fabs(fitted) - size_prior) <= data_radius_tolerance)
            {
                radius = std::fabs(fitted);
                radius_from_data = true;
            }
        }
    }

    // 位置圆拟合给出的相位已经是"从圆心指向板"的外法向，直接可用；否则用
    // 单帧 yaw 加 180 度歧义消解。
    const double phase_last =
        spin_from_positions ? wrapToPi(circle_phase_last)
                            : wrapToPi(last.yaw + (flipped ? kPi : 0.0));
    // 声明"不转"必须有据：速率接近 0 且拟合标准误小。速率本身测不准时
    // （噪声大、样本短）不能拿"static_rate_rad_s"兜底——那正是旧实现把
    // 2-8 rad/s 的旋转目标全判成 Static、然后硬钉住、整段跟踪 0 次更新的原因。
    // 转速接近 0 且满足下列任一条，才声明"不转"：
    //   - 拟合标准误小（速率测得准）；
    //   - 板面法向几乎没扫开（相位不动 ⇒ 车体没转，与速率估得准不准无关）。
    // 第二条是给"法向抖动比转速还大"的场合兜底的：抖动大时速率估不准，但
    // "压根没扫开"这个事实本身很强。
    if (spin_from_positions)
    {
        // 四块板共圆是整体几何约束，短弧上弧长/圆心都稳，但**半径**是病态的：
        // 实拍 v4 的 3 号车重锁时圆拟合给出 0.145、0.101（真值 ~0.20），
        // 半径一旦被拉小，冻结之后整段跟踪的板位都偏 —— 末段瞄准点飞到
        // 板外 0.24 m 就是这么来的。半径是"按板型的物理常数"，只在离先验
        // 不太远时才采信数据值；超出容差就保留先验。
        const double tolerance = data_radius_tolerance;
        if (std::fabs(circle.radius - size_prior) <= tolerance)
        {
            // 只有"自由拟合可信"时才把半径当数据估计；钉住半径的那次拟合里
            // circle.radius 就是先验本身，不能反过来宣称是数据估出来的
            // （合成场景 A 的半径辨识测试靠这个）。
            radius = used_fixed_radius ? size_prior : circle.radius;
            radius_from_data = !used_fixed_radius;
        }
        else
        {
            radius = size_prior;
            radius_from_data = false;
        }
    }
    const double static_confidence = std::max(1.0, 0.5 * config_.static_rate_rad_s);
    bool is_static =
        std::fabs(omega) <= config_.static_rate_rad_s &&
        (slope_error <= static_confidence || phase_spread < 0.15);
    EkfInitPrior prior;
    // 半径先验按板型选：小装甲车 0.20 m、大装甲车 0.25 m（实车 3 号是 0.197）。
    if (config_.radius_prior_from_size && !radius_from_data)
    {
        if (last.size == ArmorSize::Large)
        {
            radius = config_.radius_prior_large_m;
        }
        else
        {
            radius = config_.radius_prior_small_m;
        }
    }
    // 半径用"慢"估计：单次锁定解出的半径可能差 0.1 m（v4 实测在 0.309/0.287/0.200
    // 之间整段跳），直接拿来用会让蓝圈半径每隔几秒变一次。这里用一个很慢的
    // 一阶滤波把每次锁定的候选半径平均进去，锁定/重锁都不清掉，
    // 半径就只在长时间一致时才改变。
    if (config_.radius_track_smooth_alpha > 0.0)
    {
        if (!(radius_track_m_ > 0.0) ||
            radius < config_.radius_min_m || radius > config_.radius_max_m)
        {
            radius_track_m_ = radius;
        }
        else
        {
            radius_track_m_ +=
                config_.radius_track_smooth_alpha * (radius - radius_track_m_);
        }
        radius = radius_track_m_;
    }
    prior.radius_m = radius;
    prior.plate_slot = 0; // 槽位基准任意：相对相位才是物理量
    if (std::isfinite(config_.forced_omega_rad_s))
    {
        omega = config_.forced_omega_rad_s;
        is_static = std::fabs(omega) < 1.0e-6;
    }
    // 不再在锁定时把"转/不转"钉死：初始化只装一个先验，之后由窗口化的
    // 法向斜率（updateYawRateFromWindow -> ekf_.updateYawRate）持续修正。
    // 旧做法里 Static 会把 omega 的方差与过程噪声一起清零，一旦锁定那一帧
    // 判错（v1 f=95 双板误配、v4 法向抖 20°），后面就只能靠"声明被数据
    // 反驳 -> 重锁"来救，实测每 30 帧重锁一次、omega 在 0 与 10 rad/s 之间跳。
    prior.spin = SpinPriorKind::Unknown;
    prior.yaw_rate_prior_rad_s = is_static ? 0.0 : omega;
    prior.position_sigma_m = 0.15;
    prior.velocity_sigma_m_s = config_.velocity_sigma_m_s;
    // 首帧的相位来自单帧板面法向（实拍里小目标法向抖动可达 10 度以上），
    // 初值给宽容一点，让随后的更新把它拉正；给太紧会把 0.2-0.3m 的初值偏差
    // 全部判成离群，于是整段跟踪 0 次更新。
    prior.yaw_sigma_rad = 0.80;
    // 静态也留 1.0 rad/s 的方差：先验只是起点，斜率量测会把它拉到真值。
    prior.yaw_rate_sigma_rad_s = clampd(2.0 * slope_error, 1.0, 6.0);
    prior.radius_sigma_m = config_.radius_sigma_m;
    prior.radius_fixed = config_.freeze_radius;

    ArmorObservation observation;
    observation.position = last.position;
    if (pair_ok)
    {
        // 双板解出的几何直接覆盖：半径来自两块板的弦长，中心来自射线交点。
        prior.radius_m = pair_radius;
        prior.radius_sigma_m = std::min(config_.radius_sigma_m, 0.05);
        if (config_.radius_track_smooth_alpha > 0.0)
        {
            if (!(radius_track_m_ > 0.0))
                radius_track_m_ = pair_radius;
            else
                radius_track_m_ +=
                    config_.radius_track_smooth_alpha * (pair_radius - radius_track_m_);
            prior.radius_m = radius_track_m_;
        }
        prior.radius_fixed = config_.freeze_radius;
        observation.position = pair_position;
    }
    observation.sigma_position_m = config_.position_sigma_m;
    observation.has_world_yaw = true;
    // initialize() 的约定：world_yaw_rad 是板面"外法向"，方向 = 从车体中心指向板。
    //
    // 这里曾经多加了 pi（把"朝外的法向"当成"朝里的法向"），于是车体中心被放到
    // 板的另一侧、差 2R。静止目标能自己收敛回去（板位置不变，只是 yaw 差 pi），
    // 旋转目标就完全跟不上：实测 v4 的 3 号板锁定后 104 次更新全被门限拒绝。
    observation.world_yaw_rad =
        pair_ok ? wrapToPi(pair_yaw + kPi) : wrapToPi(phase_last + kPi);
    lock_start_s_ = timestamp_s;
    if (ekf_.initialize(observation, prior) != EkfInitStatus::Ok)
    {
        lock_samples_.clear();
        diagnostics_.last_reject = "lock:ekf_rejected";
        return false;
    }

    initialized_ = true;
    target_id_ = last.target_id;
    state_timestamp_s_ = timestamp_s;
    last_accept_s_ = timestamp_s;
    has_accept_ = true;
    consecutive_updates_ = 1;
    missed_frames_ = 0;
    other_id_streak_ = 0;
    contradiction_streak_ = 0;
    aim_slot_ = 0;
    locked_radius_m_ = radius;
    if (std::isfinite(config_.forced_omega_rad_s))
    {
        radius_from_data = true;
        spin_from_positions = false;
    }
    locked_spin_rad_s_ = is_static ? 0.0 : omega;
    diagnostics_.last_reject.clear();
    lock_radius_from_data_ = radius_from_data;
    diagnostics_.lock_rate_error_rad_s = slope_error;
    diagnostics_.lock_spin_from_positions = spin_from_positions;
    diagnostics_.lock_circle_radius_m = circle.ok ? circle.radius : 0.0;
    diagnostics_.lock_circle_residual_m = circle.ok ? circle.residual : 0.0;
    diagnostics_.lock_step_ratio = circle_step_ratio;
    lock_phase_flipped_ = flipped;
    diagnostics_.phase = LockPhase::Tracking;
    diagnostics_.accepted_updates = 1;
    diagnostics_.consecutive_updates = 1;
    diagnostics_.locked_radius_m = radius;
    diagnostics_.locked_spin_rad_s = locked_spin_rad_s_;
    diagnostics_.radius_from_data = radius_from_data;
    diagnostics_.phase_flipped = flipped;
    diagnostics_.target_id = target_id_;
    diagnostics_.aim_slot = aim_slot_;
    pushNormalHistory(0, timestamp_s, last.yaw);
    lock_samples_.clear();
    return true;
}

void RotatingArmorTracker::pushNormalHistory(int plate_slot, double timestamp_s,
                                             double world_yaw_rad)
{
    if (plate_slot < 0 ||
        plate_slot >= static_cast<int>(normal_history_per_slot_.size()))
    {
        return;
    }
    auto &history = normal_history_per_slot_[static_cast<std::size_t>(plate_slot)];
    const double unwrapped =
        history.empty() ? world_yaw_rad
                        : history.back().second + wrapToPi(world_yaw_rad - history.back().second);
    history.emplace_back(timestamp_s, unwrapped);
    while (!history.empty() && timestamp_s - history.front().first > 0.40)
    {
        history.pop_front();
    }
    while (history.size() > 32)
    {
        history.pop_front();
    }
}


// 窗口化的板面法向斜率 -> omega 伪量测。
//
// 折叠展开：车体相位只有 2*pi/N 的可辨识度（四板就是 90°），而 166 fps 下单帧
// 真实转角远小于 pi/N，所以"以上一个解缠样本为参考、按周期折叠增量"不会把
// 真实转动绕回去，却能容忍换板（换板带来的整周期增量折叠后为 0）。
// 拟合残差大 = 这一串法向根本不来自同一块/同一台车（同 id 异车、漏检跳变），
// 此时宁可不给量测，也不要喂一条错误斜率——v1 的 10 rad/s 就是这么来的。
void RotatingArmorTracker::updateYawRateFromWindow(const std::vector<WorldObservation> &world,
                                                   double timestamp_s)
{
    if (!config_.rate_update_enabled || !ekf_.initialized() || target_id_ == 0)
    {
        return;
    }
    const int plate_count = config_.ekf.plate_count;
    if (plate_count <= 0)
    {
        return;
    }
    const double period = 2.0 * kPi / static_cast<double>(plate_count);

    // 1) 只追"上一帧那块板"：用观测之间的最近邻续接，不用模型预测。
    //
    //    为什么不用预测：模型一旦把转速估错，预测位置会跟着转，最近邻就会在
    //    同一台车的两块板之间来回跳（实拍 v4 的斜置副板离正板 0.38 m），而
    //    两块板的法向差 26°、小于折叠周期 90° 的一半，解缠后变成"每帧几十度"
    //    的假高速——实测 wobs 冲到 -16 rad/s，EKF 跟着跑到 -10。
    //    观测间最近邻与模型无关，副板只在"目标板消失"时才会被误续接，
    //    而那种情况位置跳变 >0.15 m，直接重起窗口。
    const WorldObservation *chosen = nullptr;
    bool continuous = true;
    if (rate_samples_.empty())
    {
        for (const WorldObservation &observation : world)
        {
            if (observation.target_id == target_id_ && observation.has_yaw &&
                observation.position.allFinite())
            {
                chosen = &observation;
                break;
            }
        }
    }
    else
    {
        double best = config_.rate_associate_gate_m;
        for (const WorldObservation &observation : world)
        {
            if (observation.target_id != target_id_ || !observation.has_yaw ||
                !observation.position.allFinite())
            {
                continue;
            }
            const double distance = (observation.position - rate_samples_.back().position).norm();
            if (distance < best)
            {
                best = distance;
                chosen = &observation;
            }
        }
        if (chosen == nullptr)
        {
            // 换板（云台甩到另一块板上，快速自转时每几十帧一次）：位置跳变
            // 超过最近邻门限，但法向差恰好是 2π/N 的整数倍——四块板共享同一个
            // 车体朝向，折叠后是连续的。这种情况接上去，窗口不中断（G 组
            // 8 rad/s 靠这一条）。反例是 v4 的斜置副板：它离正板 0.38 m，
            // 但法向差只有 26°、不是 90° 的整数倍，不会被接上。
            double handoff = config_.position_gate_m;
            for (const WorldObservation &observation : world)
            {
                if (observation.target_id != target_id_ || !observation.has_yaw ||
                    !observation.position.allFinite())
                {
                    continue;
                }
                const double jump = std::fabs(foldPeriod(
                    observation.yaw - rate_samples_.back().unwrapped_yaw, period));
                if (jump > 0.30)
                {
                    continue;
                }
                const double distance =
                    (observation.position - rate_samples_.back().position).norm();
                if (distance < handoff)
                {
                    handoff = distance;
                    chosen = &observation;
                    continuous = false;
                }
            }
        }
        if (chosen == nullptr)
        {
            // 这块板这一帧没被看到（被挡/漏检）：窗口保留，等下一帧；
            // 间断太久由下面的 rate_max_gap_s 清空。
            if (timestamp_s - rate_samples_.back().timestamp_s > config_.rate_max_gap_s)
            {
                if (trackDebugEnabled())
                    std::fprintf(stderr, "[cleargap] ts=%.3f back=%.3f n=%zu\n", timestamp_s,
                                 rate_samples_.back().timestamp_s, rate_samples_.size());
                rate_samples_.clear();
                last_rate_update_s_ = -1.0e9;
            }
            return;
        }
    }
    if (chosen == nullptr)
    {
        if (trackDebugEnabled())
            std::fprintf(stderr, "[nopick] ts=%.3f n=%zu obs=%zu\n", timestamp_s,
                         rate_samples_.size(), world.size());
        diagnostics_.rate_window_samples = static_cast<int>(rate_samples_.size());
        return;
    }
    if (!rate_samples_.empty() &&
        timestamp_s - rate_samples_.back().timestamp_s > config_.rate_max_gap_s)
    {
        if (trackDebugEnabled())
            std::fprintf(stderr, "[cleargap2] ts=%.3f back=%.3f n=%zu\n", timestamp_s,
                         rate_samples_.back().timestamp_s, rate_samples_.size());
        rate_samples_.clear();
        last_rate_update_s_ = -1.0e9;
    }

    // 2) 解缠：折叠掉换板造成的 2pi/N 整数倍。
    double unwrapped = chosen->yaw;
    if (!rate_samples_.empty())
    {
        double reference = rate_samples_.back().unwrapped_yaw;
        if (!std::isfinite(reference))
        {
            for (auto it = rate_samples_.rbegin(); it != rate_samples_.rend(); ++it)
            {
                if (std::isfinite(it->unwrapped_yaw))
                {
                    reference = it->unwrapped_yaw;
                    break;
                }
            }
        }
        const double delta = std::isfinite(reference)
                                 ? foldPeriod(chosen->yaw - reference, period)
                                 : 0.0;
        // 单帧 0.6 rad 在 166 fps 下相当于 100 rad/s，物理上不可能；出现就说明
        // 续接跳到了别的板/别的车，窗口重起，别把跳变当成旋转。
        // 法向单帧跳 >0.6 rad（166fps 下等价 100 rad/s）说明这一帧的法向是野值
        // （实拍暗场景里板面法向抖动 ±20° 很常见，delta 经常到 ±0.3）。以前这里
        // 直接清窗，结果窗口永远长不到 min_samples：法向野值把"位置相位"这条
        // 更好的证据一起废掉了。现在只把这一帧的**法向**标成无效（NaN），
        // 位置样本照常保留，方位角证据继续累积。
        if (std::fabs(delta) > 0.60)
        {
            unwrapped = std::numeric_limits<double>::quiet_NaN();
        }
        else
        {
            unwrapped = rate_samples_.back().unwrapped_yaw + delta;
        }
    }
    rate_samples_.push_back({timestamp_s, unwrapped, chosen->position,
                             Eigen::Vector2d(ekf_.state()(0), ekf_.state()(2)), continuous});
    while (!rate_samples_.empty() &&
           timestamp_s - rate_samples_.front().timestamp_s > config_.rate_window_s)
    {
        rate_samples_.pop_front();
    }

    // 3) Theil-Sen 斜率：只用"时间跨度 >= 半个窗口"的点对。
    //
    //    法向不是白噪声，而是"每帧翻面/抖动"的野值序列，最小二乘会被单帧
    //    野值带飞（实测 v4 静态目标上 LS 给出 16 rad/s）。取点对斜率的**中位数**
    //    对 ≤30% 的野值免疫；误差取中位绝对偏差（MAD），野值多时报出的 sigma
    //    很大，滤波器自然不会被它拖走——这就是"由数据自己决定转不转"。
    const int count = static_cast<int>(rate_samples_.size());
    diagnostics_.rate_window_samples = count;
    if (count < config_.rate_min_samples)
    {
        return;
    }
    const double span = rate_samples_.back().timestamp_s - rate_samples_.front().timestamp_s;
    if (span < config_.rate_min_span_s)
    {
        return;
    }
    const double min_pair_dt = 0.5 * span;

    // 3a) 第一路证据：法向斜率（Theil-Sen + MAD）。
    std::vector<double> times(static_cast<std::size_t>(count));
    std::vector<double> yaws(static_cast<std::size_t>(count));
    std::vector<double> bearings(static_cast<std::size_t>(count));

    // 所有采样点用**同一个**极点算方位角（当前 EKF 车心），不能用每帧当时存下来的
    // 车心：车心本身在自转段会抖 ±0.1 m，0.1/0.28 ≈ 0.36 rad 的极点误差会被当成
    // 方位角变化，两两斜率散开、MAD 变大（实拍 v4 左车 sigma 3.3，滤波器不敢跟）。
    // 用同一个极点时，极点偏差只是一个常数偏移，求导后完全消掉。
    // 平移目标必须用"每帧当时的车心"：车心自己会跟着平移走，拿当前车心去算旧样本
    // 的方位角会把平移读成转动（H 组平移用例就是这么挂的）。自转目标反过来：
    // 车心固定、当时车心的抖动才是噪声源。按当前转速选极点。
    const bool spinning_now = std::fabs(ekf_.state()(7)) >= 0.5;
    const Eigen::Vector2d pole(ekf_.state()(0), ekf_.state()(2));
    // 方位角序列只收"位置连续的同一块板"的样本（见 RateSample::continuous）。
    std::vector<double> bearing_times;
    std::vector<double> bearing_values;
    bearing_times.reserve(static_cast<std::size_t>(count));
    bearing_values.reserve(static_cast<std::size_t>(count));
    double bearing_unwrapped = 0.0;
    bool bearing_started = false;
    for (int i = 0; i < count; ++i)
    {
        const RateSample &sample = rate_samples_[static_cast<std::size_t>(i)];
        times[static_cast<std::size_t>(i)] = sample.timestamp_s;
        yaws[static_cast<std::size_t>(i)] = sample.unwrapped_yaw;
        const Eigen::Vector2d &center = spinning_now ? pole : sample.center;
        const double dx = sample.position.x() - center.x();
        const double dy = sample.position.y() - center.y();
        const double bearing = std::atan2(-dy, -dx);
        if (!bearing_started)
        {
            bearing_unwrapped = bearing;
            bearing_started = true;
        }
        else
        {
            // 换板是几何上真实的 90° 跳变（折叠后≈0），同一块板的自转每帧只走
            // ω·dt（5 rad/s、166fps 时 0.03 rad）。折叠后仍然跳 >0.6 rad 的样本
            // 只可能来自"另一台车/非刚性副板"，从方位角序列里剔除，但它仍留在
            // 窗口里供法向证据使用——不需要清窗。
            const double step = foldPeriod(bearing - bearing_unwrapped, period);
            if (std::fabs(step) > 0.60)
            {
                continue;
            }
            bearing_unwrapped += step;
        }
        bearing_times.push_back(sample.timestamp_s);
        bearing_values.push_back(bearing_unwrapped);
        bearings[static_cast<std::size_t>(i)] = bearing_unwrapped;
    }
    // 法向序列里可能夹着 NaN（那一帧法向是野值，见上面的说明），只保留有效的。
    std::vector<double> yaw_times;
    std::vector<double> yaw_values;
    yaw_times.reserve(times.size());
    yaw_values.reserve(times.size());
    for (std::size_t i = 0; i < times.size(); ++i)
    {
        if (std::isfinite(yaws[i]))
        {
            yaw_times.push_back(times[i]);
            yaw_values.push_back(yaws[i]);
        }
    }
    double mad = 0.0;
    const double slope = yaw_times.size() >= 4
                             ? theilSenSlope(yaw_times, yaw_values, min_pair_dt, &mad)
                             : std::numeric_limits<double>::quiet_NaN();
    const double sigma = std::max(1.4826 * mad, config_.rate_sigma_floor_rad_s);

    // 3b) 第二路证据：板相对车心的方位角斜率。
    //
    //     法向在实拍小目标上会"整段地"漂（v4 的板心在 ±45° 折叠边界附近来回
    //     翻，解缠后是每帧 +5° 的锯齿，Theil-Sen 也救不了——实测斜率 -10 rad/s，
    //     而车其实没转）。位置没有这个问题：目标平移时板与车心一起动，方位角
    //     不变。两路证据打架时以"说不清"处理：不更新，让衰减项把 omega 停在 0。
    double bearing_mad = 0.0;
    const double bearing_slope =
        bearing_times.size() >= 8
            ? theilSenSlope(bearing_times, bearing_values, 0.5 * (bearing_times.back() - bearing_times.front()), &bearing_mad)
            : std::numeric_limits<double>::quiet_NaN();
    diagnostics_.rate_bearing_rad_s = std::isfinite(bearing_slope) ? bearing_slope : 0.0;

    // 3c) 以"位置相位"为主证据。
    //
    //     法向斜率是主证据这件事，在实拍小目标上不成立：v4 左车（真转速 ~5 rad/s）
    //     的法向在 ±45° 折叠边界来回翻，Theil-Sen 给出的是锯齿斜率，符号都可能反。
    //     位置相位没有这个问题——板心到车心的方位角每帧只走 ω·dt（166fps、5rad/s
    //     时 0.03 rad，远小于折叠半周期 45°），位置噪声 1-3cm 又比法向噪声
    //     （~20°）小一个量级。法向斜率只在它和位置相位一致时用来抬高可信度。
    const double bearing_sigma =
        std::max(1.4826 * bearing_mad, config_.rate_sigma_floor_rad_s);
    double used_slope = bearing_slope;
    double used_sigma = bearing_sigma;
    if (std::isfinite(slope) && std::isfinite(bearing_slope))
    {
        const double tolerance = std::max(config_.rate_agreement_rad_s,
                                          0.5 * std::fabs(bearing_slope));
        if (std::fabs(slope - bearing_slope) <= tolerance)
        {
            // 两路一致：按方差加权，瘦一下 sigma。
            const double w_n = 1.0 / (sigma * sigma);
            const double w_b = 1.0 / (bearing_sigma * bearing_sigma);
            used_slope = (w_n * slope + w_b * bearing_slope) / (w_n + w_b);
            used_sigma = std::sqrt(1.0 / (w_n + w_b));
        }
    }
    // 3d) 物理闸门。
    //
    //     上面那套"以位置相位为主证据"在实拍里仍然会被喂进伪转速：v4 左车
    //     （自转+平移）实测 bear 斜率 +19~+34 rad/s，远超 4 板车 8 rad/s 的物理
    //     上限——这些值来自"续接到了另一块板 / 极点选错 / 折叠算术出错"，不是
    //     转速。直接喂进 EKF 就会把 omega 打成 ±10 的方波（v4 实测 |omega|>=5
    //     占 20.6%），再反馈回预测，瞄准点整块飘走。所以先用物理上限筛一遍：
    //     主证据不可信就退到另一路，两路都不可信就这一窗不用（留给衰减项）。
    const auto plausible = [&](double value)
    {
        return std::isfinite(value) &&
               (config_.rate_max_abs_rad_s <= 0.0 ||
                std::fabs(value) <= config_.rate_max_abs_rad_s);
    };
    const char *source = "bear";
    if (!plausible(used_slope))
    {
        if (plausible(bearing_slope))
        {
            used_slope = bearing_slope;
            used_sigma = bearing_sigma;
            source = "bear";
        }
        else if (plausible(slope))
        {
            used_slope = slope;
            used_sigma = sigma;
            source = "yaw";
        }
        else
        {
            if (trackDebugEnabled())
                std::fprintf(stderr,
                             "[evskip] n=%d yaw=%+.2f(%.2f) bear=%+.2f(%.2f) (超物理闸门)\n",
                             count, slope, mad, bearing_slope, bearing_mad);
            return;
        }
    }
    if (used_sigma > config_.rate_max_sigma_rad_s)
    {
        if (trackDebugEnabled())
            std::fprintf(stderr, "[evskip] n=%d used=%+.2f(%.2f) src=%s (sigma 过大)\n",
                         count, used_slope, used_sigma, source);
        return;
    }

    if (trackDebugEnabled())
        std::fprintf(stderr, "[ev] n=%d yaw=%+.2f(%.2f) bear=%+.2f(%.2f) used=%+.2f(%.2f) src=%s\n",
                     count, slope, mad, bearing_slope, bearing_mad, used_slope, used_sigma, source);

    if (!ekf_.updateYawRate(used_slope, used_sigma))
    {
        return;
    }
    // 刚吃过转速证据，盖章：update() 末尾的衰减块以此为据判断"证据缺席"。
    last_rate_update_s_ = timestamp_s;
    diagnostics_.rate_obs_used = true;
    diagnostics_.rate_obs_rad_s = used_slope;
    diagnostics_.rate_obs_sigma_rad_s = used_sigma;
    diagnostics_.rate_obs_samples = count;
}

bool RotatingArmorTracker::reinitializeFromPair(const std::vector<WorldObservation> &world)
{
    if (!ekf_.initialized() || target_id_ == 0)
    {
        return false;
    }
    for (std::size_t i = 0; i < world.size(); ++i)
    {
        if (world[i].target_id != target_id_)
        {
            continue;
        }
        for (std::size_t j = i + 1; j < world.size(); ++j)
        {
            if (world[j].target_id != target_id_)
            {
                continue;
            }
            const Eigen::Vector2d ni(std::cos(world[i].yaw), std::sin(world[i].yaw));
            const Eigen::Vector2d nj(std::cos(world[j].yaw), std::sin(world[j].yaw));
            const Eigen::Vector2d ndiff = ni - nj;
            if (ndiff.norm() < 0.6)
            {
                continue;
            }
            const Eigen::Vector2d pi2(world[i].position.x(), world[i].position.y());
            const Eigen::Vector2d pj2(world[j].position.x(), world[j].position.y());
            const double radius = (pi2 - pj2).norm() / ndiff.norm();
            if (radius < config_.radius_min_m || radius > config_.radius_max_m)
            {
                continue;
            }
            const Eigen::Vector2d center = pi2 + radius * ni;
            if ((center - radius * nj - pj2).norm() > 0.10)
            {
                continue;
            }
            // 状态本来就和双板解一致时不要重定：长时间"没有可接收的更新"不等于
            // "几何错了"，每帧重定会把协方差反复打回大值，后面的观测更进不来。
            const Eigen::Vector2d state_center(ekf_.state()(0), ekf_.state()(2));
            const double state_radius = ekf_.state()(8);
            if ((state_center - center).norm() < 0.20 &&
                std::fabs(state_radius - radius) < 0.08)
            {
                return false;
            }
            // 重定限频：0.5 s 内只允许一次，避免"重定 -> 观测仍被门限拒 -> 再重定"。
            if (has_pair_relock_ && timestamp_s_ - last_pair_relock_s_ < 0.5)
            {
                return false;
            }
            EkfInitPrior prior;
            prior.radius_m = radius;
            prior.plate_slot = 0;
            prior.spin = SpinPriorKind::Spinning;
            prior.yaw_rate_prior_rad_s = ekf_.state()(7);
            prior.position_sigma_m = 0.10;
            prior.velocity_sigma_m_s = config_.velocity_sigma_m_s;
            prior.yaw_sigma_rad = 0.50;
            prior.yaw_rate_sigma_rad_s = 4.0;
            prior.radius_sigma_m = 0.05;
            ArmorObservation observation;
            observation.position = world[i].position;
            observation.sigma_position_m = world[i].sigma_position_m;
            observation.has_world_yaw = true;
            observation.world_yaw_rad = wrapToPi(std::atan2(world[i].position.y() - center.y(),
                                                          world[i].position.x() - center.x()) +
                                                 kPi);
            if (ekf_.initialize(observation, prior) != EkfInitStatus::Ok)
            {
                return false;
            }
            has_pair_relock_ = true;
            last_pair_relock_s_ = timestamp_s_;
            ++diagnostics_.relock_count;
            diagnostics_.last_reject = "relock:pair";
            return true;
        }
    }
    return false;
}

// 同帧双板弦长 -> 半径。两块板的法向差 Δψ 与弦长 chord 唯一确定半径：
// |n_i - n_j| = 2 sin(Δψ/2)，而 chord = R |n_i - n_j|，所以 R = chord / (2 sin(Δψ/2))。
// 每帧最多吃一条，避免同一帧两条配对观测重复约束同一维。
bool RotatingArmorTracker::updateBodyFromPair(const std::vector<WorldObservation> &world)
{
    if (!config_.pair_body_solve_enabled || target_id_ == 0 || !ekf_.initialized())
    {
        return false;
    }
    // 刚锁定那几帧车心和相位还没收敛，两支候选解分不开，硬套会把相位整体推歪
    // （合成 G 组 6/8 rad/s 的瞄准残差会从 1 cm 涨到 35 cm）。等状态站住再解。
    if (diagnostics_.accepted_updates < 8)
    {
        return false;
    }
    const int plate_count = config_.ekf.plate_count;
    if (plate_count < 2)
    {
        return false;
    }
    const double period = 2.0 * kPi / static_cast<double>(plate_count);
    if (std::fabs(ekf_.state()(7)) >= config_.pair_body_max_rate_rad_s)
    {
        return false;
    }

    // 找同帧里"两块板"构成的最佳一对：弦长最大、法向差最接近相邻槽位。
    int best_i = -1;
    int best_j = -1;
    int best_dslot = 0;
    double best_chord = 0.0;
    double best_score = -1.0;
    for (std::size_t i = 0; i < world.size(); ++i)
    {
        if (world[i].target_id != target_id_ || !world[i].has_yaw)
        {
            continue;
        }
        for (std::size_t j = i + 1; j < world.size(); ++j)
        {
            if (world[j].target_id != target_id_ || !world[j].has_yaw)
            {
                continue;
            }
            const double chord = (world[j].position - world[i].position).norm();
            if (!std::isfinite(chord) || chord < 0.15 || chord > 0.90)
            {
                continue;
            }
            const double dpsi = wrapToPi(world[j].yaw - world[i].yaw);
            int dslot = static_cast<int>(std::lround(dpsi / period));
            dslot %= plate_count;
            if (dslot < 0)
            {
                dslot += plate_count;
            }
            // 只解相邻板（相差 1 个槽位）：对置板的两条方程退化，R 和 theta 不唯一。
            if (dslot != 1 && dslot != plate_count - 1)
            {
                continue;
            }
            const double nslot = (dslot == 1) ? 1.0 : -1.0;
            const double score = chord - 0.02 * std::fabs(
                foldPeriod(world[j].yaw - world[i].yaw - nslot * period, 2.0 * kPi));
            if (score > best_score)
            {
                best_score = score;
                best_i = static_cast<int>(i);
                best_j = static_cast<int>(j);
                best_dslot = dslot;
                best_chord = chord;
            }
        }
    }
    if (best_i < 0)
    {
        return false;
    }

    const Eigen::Vector3d pa = world[static_cast<std::size_t>(best_i)].position;
    const Eigen::Vector3d pb = world[static_cast<std::size_t>(best_j)].position;
    const Eigen::Vector3d d = pb - pa;
    // 相邻槽位：|d| = 2 R sin(period/2) = sqrt(2) R（四板构型）。
    const double radius_meas = best_chord / (2.0 * std::sin(0.5 * period));
    if (!std::isfinite(radius_meas) || radius_meas < config_.radius_min_m ||
        radius_meas > config_.radius_max_m)
    {
        return false;
    }
    // theta_a：a 板相对车体相位的槽位角。上面推导的 +-135/45 度只在 period=90
    // 时成立，这里写成与 period 无关的形式。
    const double d_angle = std::atan2(d.y(), d.x());
    // 两个分支对应 b 板比 a 板多一个槽位(逆时针)或少一个槽位(顺时针)。法向差的
    // 符号能给出分支，但"世界系 yaw 的正方向"在不同数据源里可能镜像，符号并不
    // 总是可信。这里直接算两个分支、取与当前车心更一致的那个——几何上只有一支
    // 成立，这个判据不依赖任何符号约定。
    const double theta_plus = std::atan2(-d.y(), -d.x()) - (kPi - 0.5 * period);
    const double theta_minus = d_angle - 0.5 * period;
    const Eigen::Vector2d center_state(ekf_.state()(0), ekf_.state()(2));
    Eigen::Vector2d center_meas(0.0, 0.0);
    double theta_a = theta_plus;
    double best_dev = std::numeric_limits<double>::infinity();
    for (int branch = 0; branch < 2; ++branch)
    {
        const double theta = (branch == 0) ? theta_plus : theta_minus;
        const Eigen::Vector2d candidate(pa.x() + radius_meas * std::cos(theta),
                                        pa.y() + radius_meas * std::sin(theta));
        if (!candidate.allFinite())
        {
            continue;
        }
        const double dev = (candidate - center_state).norm();
        if (dev < best_dev)
        {
            best_dev = dev;
            theta_a = theta;
            center_meas = candidate;
        }
    }
    if (!center_meas.allFinite())
    {
        return false;
    }

    const double yaw_state = ekf_.state()(6);
    // 四板构型下个体槽位编号本身不可观测（整体转 90 度等价），取与当前相位最近的
    // 那个编号即可，保证相位修正量落在 +-period/2 内。
    const int slot_a = static_cast<int>(std::lround((theta_a - yaw_state) / period));
    const double yaw_meas = theta_a - static_cast<double>(slot_a) * period;

    if ((center_meas - center_state).norm() > 0.35)
    {
        return false;
    }
    const double dyaw = wrapToPi(yaw_meas - yaw_state);
    // 只修"小幅偏差"：相位差很大说明状态还没收敛（比如刚锁定时角速度还在爬升），
    // 这时闭式解的瞬时相位本身不可信，硬拧会把相位整体推歪——四板构型下相位
    // 有 90 度对称性，推歪之后位置更新再也拉不回来。
    if (std::fabs(dyaw) > config_.pair_body_max_phase_error_rad)
    {
        return false;
    }

    const Eigen::Vector3d center_delta(config_.pair_body_center_gain * (center_meas.x() - center_state.x()),
                                       config_.pair_body_center_gain * (center_meas.y() - center_state.y()),
                                       0.0);
    ekf_.shiftCenter(center_delta);
    ekf_.rotatePhase(config_.pair_body_yaw_gain * dyaw);
    if (config_.pair_body_radius_gain > 0.0)
    {
        const double radius_next = std::clamp(
            ekf_.state()(8) + config_.pair_body_radius_gain * (radius_meas - ekf_.state()(8)),
            config_.radius_min_m, config_.radius_max_m);
        applySlowRadius(radius_next);
    }
    ++diagnostics_.pair_body_solves;
    return true;
}

void RotatingArmorTracker::updateRadiusFromPairs(const std::vector<WorldObservation> &world)
{
    if (!config_.pair_radius_update_enabled || target_id_ == 0 || !ekf_.initialized())
    {
        return;
    }
    // 高速自转时不动半径：四块板轮流转，改 R 等于把同帧的相位残差硬塞给半径，
    // 中心会被拽走（合成 G 组 4/6/8 rad/s 的瞄准残差会从 1 cm 涨到 3 cm 以上）。
    // 静止/平移目标才需要它——实拍 v4 的自制道具半径 0.28 就是这种情况。
    if (std::fabs(ekf_.state()(7)) >= config_.static_rate_rad_s ||
        static_streak_ < 10)
    {
        return;
    }
    for (size_t i = 0; i < world.size(); ++i)
    {
        if (world[i].target_id != target_id_ || !world[i].has_yaw)
        {
            continue;
        }
        for (size_t j = i + 1; j < world.size(); ++j)
        {
            if (world[j].target_id != target_id_ || !world[j].has_yaw)
            {
                continue;
            }
            const double dpsi = std::fabs(foldPeriod(world[j].yaw - world[i].yaw, 2.0 * kPi));
            const double chord = (world[j].position - world[i].position).norm();
            // 只采信"两块不同板"的弦长：同一块板被重复检出时弦长很小，
            // 而法向差却可能几十度，弦长除噪会把半径拉到 0.14（实拍 v4 的 3 号车）。
            if (!std::isfinite(chord) || chord < config_.radius_update_min_chord_m)
            {
                continue;
            }
            // 用"模型构型"的半角而不是实测法向差：实测法向差本身有 ±20° 噪声，
            // 直接带进 sin() 会把半径算大（v4 实测法向差 80° → 半径 0.31，
            // 而弦长 0.40/相邻构型给出的是 0.283）。四板模型只有两种可能：
            // 相邻 90°（弦 = √2 R）或对置 180°（弦 = 2R）。
            double half_model = 0.0;
            if (dpsi > config_.radius_update_opposite_yaw_rad)
            {
                half_model = 0.5 * kPi;
            }
            else if (dpsi > config_.radius_update_adjacent_yaw_rad)
            {
                half_model = 0.25 * kPi;
            }
            else
            {
                continue;
            }
            const double radius = chord / (2.0 * std::sin(half_model));
            if (!std::isfinite(radius) || radius < config_.radius_min_m ||
                radius > config_.radius_max_m)
            {
                continue;
            }
            // 低通：单帧弦长还有几毫米到几厘米的噪声，别让它一步跳到位。
            // freeze_radius 打开时位置更新会把 R 拉回先验，所以 setRadius 会
            // 同时改先验（见 RotatingArmorEkf::setRadius）。
            double target_radius = radius;
            if (config_.freeze_radius)
            {
                // 第一次拿到弦长直接采信（先验 0.20 vs 实测 0.28，慢慢靠拢要几百帧）；
                // 之后低通，滤掉单帧噪声。
                // 第一次拿到弦长直接采信：先验 0.20 vs 实测 0.28 时慢慢靠拢要几百帧，
                // 而 static_streak>=10 已经保证目标确实是静止的，这一次不会被打脸。
                const double alpha =
                    radius_updates_ == 0 ? 1.0 : config_.radius_update_alpha;
                const double current = ekf_.state()(8);
                target_radius = current + alpha * (radius - current);
                applySlowRadius(target_radius);
                ++radius_updates_;
            }
            else
            {
                // 不走 updateRadius 的协方差更新：每次改半径都会重写整块协方差，
                // 高速自转场景里会连累 yaw 的估计（合成 G 组 2 rad/s 射击点偏 30 cm）。
                // 弦长本身已经是几何量，低通后直接写状态就够。
                applySlowRadius(target_radius);
            }
            return;
        }
    }
}

void RotatingArmorTracker::noteChordRadius(const std::vector<WorldObservation>& world,
                                           double timestamp_s)
{
    if (config_.radius_track_smooth_alpha <= 0.0)
    {
        return;
    }
    for (std::size_t i = 0; i < world.size(); ++i)
    {
        if (world[i].target_id == 0 || !world[i].has_yaw)
            continue;
        for (std::size_t j = i + 1; j < world.size(); ++j)
        {
            if (world[j].target_id != world[i].target_id || !world[j].has_yaw)
                continue;
            const double dpsi = std::fabs(foldPeriod(world[j].yaw - world[i].yaw, 2.0 * kPi));
            const bool adjacent =
                dpsi > 0.5 * kPi - 0.45 && dpsi < 0.5 * kPi + 0.45;
            const bool opposite =
                dpsi > kPi - 0.45;
            if (!adjacent && !opposite)
                continue;
            const Eigen::Vector3d d = world[j].position - world[i].position;
            const double chord = std::hypot(d.x(), d.y());
            const double half = adjacent ? 0.25 * kPi : 0.5 * kPi;
            const double radius = chord / (2.0 * std::sin(half));
            if (!(radius > config_.radius_min_m && radius < config_.radius_max_m))
                continue;
            if (radius < config_.radius_plausible_min_m ||
                radius > config_.radius_plausible_max_m)
                continue; // 法向判错导致的离谱半径，不进证据集
            LockSample sample{};
            sample.timestamp_s = timestamp_s;
            sample.position = Eigen::Vector3d(radius, 0.0, 0.0);
            sample.target_id = world[i].target_id;
            chord_radius_samples_.push_back(sample);
        }
    }
    while (!chord_radius_samples_.empty() &&
           timestamp_s - chord_radius_samples_.front().timestamp_s > config_.radius_chord_window_s)
    {
        chord_radius_samples_.pop_front();
    }
}

void RotatingArmorTracker::applySlowRadius(double candidate_radius)
{
    if (!std::isfinite(candidate_radius) || candidate_radius <= 0.0)
    {
        return;
    }
    if (!ekf_.initialized())
    {
        return;
    }
    const double alpha = config_.radius_track_smooth_alpha;
    if (alpha <= 0.0)
    {
        ekf_.setRadius(candidate_radius);
        return;
    }
    if (!(radius_track_m_ > 0.0))
    {
        radius_track_m_ = ekf_.state()(8);
    }
    double target = candidate_radius;
    if (chord_radius_samples_.size() >= 10)
    {
        // 有弦长证据时以它为准：它对法向噪声不敏感，是几何真值。
        std::vector<double> chords;
        chords.reserve(chord_radius_samples_.size());
        for (const LockSample& s : chord_radius_samples_)
            chords.push_back(s.position.x());
        // 低分位：错误法向解出来的大半径被排除在外，只信"一致偏小"的那一簇。
        const double q = std::clamp(config_.radius_chord_quantile, 0.0, 0.9);
        const std::size_t idx = static_cast<std::size_t>(q * static_cast<double>(chords.size() - 1));
        std::nth_element(chords.begin(), chords.begin() + static_cast<std::ptrdiff_t>(idx),
                         chords.end());
        target = chords[idx];
        // 弦长本身很准，收敛可以稍快一点。
        radius_track_m_ += std::min(1.0, 2.0 * alpha) * (target - radius_track_m_);
    }
    else
    {
        // 没有弦长证据时，拟合值只允许在既有估计附近小幅修正（±2 cm）。
        // 拟合值依赖板法向，法向错时会把圆心和半径一起解歪——正是要压制的那类噪声。
        const double lo = radius_track_m_ - config_.radius_fit_nudge_m;
        const double hi = radius_track_m_ + config_.radius_fit_nudge_m;
        target = std::clamp(target, lo, hi);
        radius_track_m_ += alpha * (target - radius_track_m_);
    }
    radius_track_m_ = std::max(config_.radius_min_m, std::min(config_.radius_max_m, radius_track_m_));
    ekf_.setRadius(radius_track_m_);
}

// 掉线快照续接：注销后不久、同一编号又出现时，直接恢复掉线前的状态，
// 跳过"重新收集 10 个样本 + 等跨度"的 Collecting。
bool RotatingArmorTracker::tryRestoreSnapshot(const std::vector<WorldObservation> &world,
                                              double timestamp_seconds)
{
    if (!config_.relock_reuse_enabled || !has_snapshot_ || snapshot_id_ == 0)
    {
        return false;
    }
    const double gap = timestamp_seconds - snapshot_s_;
    if (!(gap >= 0.0) || gap > config_.relock_reuse_s)
    {
        return false;
    }
    bool present = false;
    for (const WorldObservation &observation : world)
    {
        if (observation.target_id == snapshot_id_)
        {
            present = true;
            break;
        }
    }
    if (!present)
    {
        return false;
    }
    if (!ekf_.restore(snapshot_state_, snapshot_covariance_, snapshot_phase_resolved_,
                      snapshot_spin_pinned_))
    {
        return false;
    }
    target_id_ = snapshot_id_;
    // reset() 会把 diagnostics_ 清成默认值（target_id=0）；续接后必须把对外
    // 可见的编号/相位补回去，否则上层会以为"还在收集"，指标也会漏掉这些帧。
    diagnostics_.phase = LockPhase::Tracking;
    diagnostics_.target_id = target_id_;
    // 用真实间隔推进一次协方差（旋转目标在 gap 里可能转了几十度，滤波器的
    // 外推至少把不确定性放大到位，真接不上就让门限拒掉、再走正常重锁）。
    if (gap > 1.0e-9 && !ekf_.predict(gap))
    {
        ekf_.reset();
        return false;
    }
    state_timestamp_s_ = timestamp_seconds;
    // 续接帧不算"刚被观测"：新鲜度按真实空窗算（gap 内目标可能已经转过几十度），
    // 所以 Unstable 是诚实的；但把连续更新数留到"再吃一次更新就 Stable"，
    // 不然每次续接都要白等 10 帧。
    has_accept_ = true;
    last_accept_s_ = snapshot_s_;
    consecutive_updates_ = std::max(0, config_.stable_updates - 1);
    missed_frames_ = 0;
    has_snapshot_ = false;
    diagnostics_.last_reject = "snapshot_restore";
    return true;
}

bool RotatingArmorTracker::smoothObservation(const WorldObservation &observation, int slot,
                                              double timestamp_s, ArmorObservation &out)
{
    if (!config_.observation_smoothing_enabled)
    {
        return false;
    }
    // 换板、换目标、间隔太久都说明"上一段均值"不再代表这块板，重置。
    const bool same_plate = smoother_.slot == slot && smoother_.target_id == observation.target_id;
    const bool fresh_gap = timestamp_s - smoother_.timestamp_s > config_.smoothing_max_gap_s;
    if (!same_plate || fresh_gap || smoother_.count <= 0)
    {
        smoother_.mean = observation.position;
        smoother_.slot = slot;
        smoother_.target_id = observation.target_id;
        smoother_.count = 1;
        smoother_.timestamp_s = timestamp_s;
        return false; // 只有一帧时平滑没有意义，走原始观测
    }
    // 观测离均值太远：可能是同一槽位上的真实跳变（重锁/外推偏了），重置。
    const double spread_m = (observation.position - smoother_.mean).norm();
    const bool too_far = !(spread_m <= config_.smoothing_max_spread_m);
    if (too_far)
    {
        smoother_.mean = observation.position;
        smoother_.count = 1;
        smoother_.timestamp_s = timestamp_s;
        return false;
    }
    const int window = std::max(config_.smoothing_window, 1);
    const double alpha = 1.0 / static_cast<double>(std::min(smoother_.count + 1, window));
    smoother_.mean += alpha * (observation.position - smoother_.mean);
    smoother_.count = std::min(smoother_.count + 1, window);
    smoother_.timestamp_s = timestamp_s;

    out.position = smoother_.mean;
    // 平均 N 帧把白噪声压到 1/sqrt(N)，但公共系统误差（半径/相位偏差）不会
    // 被平均掉，所以地板项保留。
    const double averaged = observation.sigma_position_m / std::sqrt(static_cast<double>(smoother_.count));
    out.sigma_position_m = std::hypot(averaged, config_.association_sigma_floor_m);
    return true;
}

int RotatingArmorTracker::associateAndUpdate(const std::vector<WorldObservation> &world,
                                             double timestamp_s)
{
    if (!ekf_.initialized())
    {
        return 0;
    }
    const int plate_count = config_.ekf.plate_count;
    int accepted = 0;
    std::vector<bool> used(world.size(), false);
    std::vector<int> slot_of_observation(world.size(), -1);
    // 同一帧里两块不同的板不能更新到同一个槽位：v4 实拍中一台车同时露出两块
    // 相邻板（相距 0.40 m），若两帧都判给同一个槽位，中心会被来回拉、相位永远
    // 定不住。按距离贪心分配，已用过的槽位本帧不再接受第二块板。
    std::vector<bool> slot_used(static_cast<std::size_t>(plate_count), false);
    diagnostics_.rejected_this_frame = 0;
    diagnostics_.best_gate_distance_m = std::numeric_limits<double>::infinity();
    diagnostics_.best_gate_nis = std::numeric_limits<double>::infinity();

    // 同帧两块板的联合绑定：两块板是同一台车上的两块相邻面，它们的槽位差由
    // 几何决定，只有 4 种整体旋转可选。逐块"找最近的预测板"是各自独立决策，
    // 会把两块板绑到互相矛盾的槽位上（v4 实拍里 20 帧同槽、157 帧不同槽，
    // 一旦同槽就互相拉扯，相位永远定不住）。这里对 4 种槽位整体旋转求和代价，
    // 取最小的一组，用它把两块板的槽位钉住。
    std::vector<int> forced_slot(world.size(), -1);
    if (config_.joint_pair_binding_enabled)
    {
        std::vector<int> cand;
        for (std::size_t i = 0; i < world.size(); ++i)
        {
            if (world[i].target_id == target_id_)
            {
                cand.push_back(static_cast<int>(i));
            }
        }
        if (cand.size() == 2)
        {
            const WorldObservation &wa = world[static_cast<std::size_t>(cand[0])];
            const WorldObservation &wb = world[static_cast<std::size_t>(cand[1])];
            const double chord = (wb.position - wa.position).norm();
            if (wa.has_yaw && wb.has_yaw && std::isfinite(chord) && chord > 0.15 &&
                chord < 0.90)
            {
                const double period = 2.0 * kPi / static_cast<double>(plate_count);
                int dslot = static_cast<int>(
                    std::lround(wrapToPi(wb.yaw - wa.yaw) / period));
                dslot %= plate_count;
                if (dslot < 0)
                {
                    dslot += plate_count;
                }
                if (dslot == 1 || dslot == plate_count - 1)
                {
                    double best_cost = std::numeric_limits<double>::infinity();
                    int best_start = -1;
                    for (int start = 0; start < plate_count; ++start)
                    {
                        const int sb = (start + dslot) % plate_count;
                        const double cost =
                            (wa.position - platePosition(ekf_.state(), start, plate_count)).norm() +
                            (wb.position - platePosition(ekf_.state(), sb, plate_count)).norm();
                        if (cost < best_cost)
                        {
                            best_cost = cost;
                            best_start = start;
                        }
                    }
                    // 代价过大说明这一对本身就不可信（比如其中一块是同一块板的
                    // 重复检出），不强制绑定，退回原来的逐块贪心。
                    if (best_start >= 0 && best_cost < config_.joint_pair_binding_max_cost_m)
                    {
                        forced_slot[static_cast<std::size_t>(cand[0])] = best_start;
                        forced_slot[static_cast<std::size_t>(cand[1])] =
                            (best_start + dslot) % plate_count;
                    }
                }
            }
        }
    }

    while (accepted < std::max(config_.max_updates_per_frame, 1))
    {
        int best_observation = -1;
        int best_slot = -1;
        double best_distance = std::numeric_limits<double>::infinity();
        double best_nis = 0.0;
        for (std::size_t i = 0; i < world.size(); ++i)
        {
            if (used[i])
            {
                continue;
            }
            // 只更新"跟踪中的那台车"的观测。不按 target_id 过滤的话，另一台车
            // 的板只要离我们的预测板够近就会被当成自己的观测：实拍 v4 两台车
            // 同框时，id=3 的滤波器被 id=1 的板反复拉走（瞄准点直接跳到另一台
            // 车上 1.0 m，半径被撑到 0.34 m）。
            if (target_id_ != 0 && world[i].target_id != target_id_)
            {
                continue;
            }
            ArmorObservation observation;
            observation.position = world[i].position;
            observation.sigma_position_m = observationSigma(world[i].sigma_position_m);
            if (forced_slot[static_cast<std::size_t>(i)] >= 0)
            {
                // 已被联合绑定钉住槽位：只试它，别让逐块贪心把它改到别的面上去。
                const int only = forced_slot[static_cast<std::size_t>(i)];
                if (slot_used[static_cast<std::size_t>(only)])
                {
                    continue;
                }
            }
            for (int slot = 0; slot < plate_count; ++slot)
            {
                if (slot_used[static_cast<std::size_t>(slot)])
                {
                    continue;
                }
                if (forced_slot[static_cast<std::size_t>(i)] >= 0 &&
                    slot != forced_slot[static_cast<std::size_t>(i)])
                {
                    continue;
                }
                if (config_.obs_continuity_gate_enabled && has_last_accepted_position_ &&
                    (world[i].position - last_accepted_position_).norm() > config_.obs_continuity_gate_m)
                {
                    ++diagnostics_.continuity_rejects;
                    continue;
                }
                const Eigen::Vector3d predicted = platePosition(ekf_.state(), slot, plate_count);
                const double distance = (world[i].position - predicted).norm();
                // 换板（更新的不是当前瞄准槽位）时观测更不可信：板编号只到
                // 2*pi/N 的可辨识度，观测落到哪块板上本身带一个槽位的歧义。
                // 不放大 R 的话，静止时把协方差压得很小的滤波器会把换板观测
                // 全判成 NIS 离群（实拍 v4 旋转目标 416/718 帧就是这样丢的）。
                // 不再因为"换板"放大 sigma：观测已经分配给最近的预测板，
                // 板编号的 2*pi/N 歧义已经在关联里消化掉了。放大 sigma 等于
                // 告诉滤波器"这帧别信"，而实拍里单板可见时每帧都可能换板，
                // 结果就是状态永远被弱更新拖着走、ω 学不上去。
                const ArmorObservation &slot_observation = observation;
                diagnostics_.best_gate_distance_m =
                    std::min(diagnostics_.best_gate_distance_m, distance);
                const bool within_gate = distance <= config_.position_gate_m;
                // 重锚候选：状态已经漂出位置门，但观测仍可能是"同一台车的某块
                // 板"（同 target_id）。此时先按这块板把相位+中心对齐，再更新；
                // 不这样做的话位置门/ NIS 门会一起把状态锁死在错误相位上。
                // 重锚候选同样只在"已经卡住"时开放：否则任何 1.5 m 内的
                // 离群观测都会绕过 NIS 门被吃进来（回归测试 D 组要求单个
                // 离群观测必须被拒绝）。
                const bool realign_candidate =
                    !within_gate && config_.single_plate_realign_m > 0.0 &&
                    distance <= config_.single_plate_realign_gate_m &&
                    last_updates_ == 0 &&
                    missed_frames_ >= std::max(config_.realign_after_misses, 0);
                if (!within_gate && !realign_candidate)
                {
                    continue;
                }
                double nis = 0.0;
                if (!ekf_.innovationNis(slot_observation, slot, nis))
                {
                    continue;
                }
                // 锁定的头几次更新只靠位置门：初值来自单帧板面法向，实拍里
                // 小目标的法向抖 10 度以上，初值偏差常在 2R 量级，用 NIS 门
                // 会把它们全部判成离群，于是"锁定后 0 次更新"。放行几次让滤波器
                // 收敛，之后 NIS 门照常把关。
                const bool bootstrapping =
                    diagnostics_.accepted_updates <
                    static_cast<std::uint64_t>(std::max(config_.bootstrap_updates, 0));
                if (!bootstrapping && !realign_candidate && nis > config_.nis_gate)
                {
                    diagnostics_.best_gate_nis = std::min(diagnostics_.best_gate_nis, nis);
                    continue;
                }
                // 法向只用来"打破四板对称造成的槽位平局"，不用来拉整帧相位。
                // 四块板是同一个模子、彼此相差 90 度，位置门（0.6 m）对相隔
                // 0.4 m 的两块板同时放行——光看位置分不出"这一帧看到的是哪一面"，
                // 分错的后果就是画面上模型板停在相邻那一面上（相位差 90 度）。
                // 板面法向直接把"是哪一面"告诉了我们：按角度差加一项代价，
                // 让槽位选择同时满足位置近和朝向对。
                double score = distance;
                if (config_.normal_tiebreak_weight > 0.0 && world[i].has_yaw &&
                    std::isfinite(world[i].yaw))
                {
                    const double period = 2.0 * kPi / static_cast<double>(plate_count);
                    const double face_yaw =
                        ekf_.state()(6) + static_cast<double>(slot) * period;
                    score += config_.normal_tiebreak_weight *
                             std::fabs(wrapToPi(face_yaw - world[i].yaw));
                }
                // 角度项：法向差 + 方位差（见 TrackerConfig::association_by_angle_enabled）
                static const bool angle_assoc_disabled_by_env =
                    std::getenv("NO_ASSOC_ANGLE") != nullptr;
                if (config_.association_by_angle_enabled && !angle_assoc_disabled_by_env &&
                    world[i].has_yaw &&
                    std::isfinite(world[i].yaw))
                {
                    const double angle_period = 2.0 * kPi / static_cast<double>(plate_count);
                    const double face_yaw = rotating_armor::platePhaseToWorldYaw(
                        ekf_.state()(6) + static_cast<double>(slot) * angle_period);
                    const double normal_err = std::fabs(wrapToPi(world[i].yaw - face_yaw));
                    const double obs_bearing =
                        std::atan2(world[i].position.y(), world[i].position.x());
                    const double pred_bearing = std::atan2(predicted.y(), predicted.x());
                    const double bearing_err =
                        std::fabs(wrapToPi(obs_bearing - pred_bearing));
                    score += config_.association_angle_weight_m_per_rad *
                             (normal_err + bearing_err);
                }
                if (score < best_distance)
                {
                    best_distance = score;
                    best_observation = static_cast<int>(i);
                    best_slot = slot;
                    best_nis = nis;
                }
            }
        }
        if (best_observation < 0)
        {
            break;
        }
        const WorldObservation &chosen = world[static_cast<std::size_t>(best_observation)];
        slot_used[static_cast<std::size_t>(best_slot)] = true;
        // 学这个槽位相对车体中心的高度偏移（只在已初始化时才有意义）。纯显示量，
        // 不参与滤波；EMA 系数小一点，避免单帧深度野值把它带跑。
        if (ekf_.initialized() && config_.slot_offsets_enabled)
        {
            const double offset = chosen.position.z() - ekf_.state()(4);
            if (std::fabs(offset) < 0.20)
            {
                double &stored = slot_z_offset_[static_cast<std::size_t>(best_slot)];
                stored += 0.08 * (offset - stored);
            }
            // 每个槽位的实际半径也学一下（同上，只用于显示）。
            const Eigen::Vector2d radial(chosen.position.x() - ekf_.state()(0),
                                         chosen.position.y() - ekf_.state()(2));
            const double radius_offset = radial.norm() - ekf_.state()(8);
            if (std::fabs(radius_offset) < 0.12)
            {
                double &stored = slot_radius_offset_[static_cast<std::size_t>(best_slot)];
                stored += 0.08 * (radius_offset - stored);
            }
            if (radial.norm() > 0.05)
            {
                const double period = 2.0 * kPi / static_cast<double>(plate_count);
                const double observed_angle = std::atan2(radial.y(), radial.x());
                const double model_angle = ekf_.state()(6) +
                                           static_cast<double>(best_slot) * period + kPi;
                double deviation = wrapToPi(observed_angle - model_angle);
                while (std::fabs(deviation) > 0.5 * period)
                {
                    deviation -= std::copysign(period, deviation);
                }
                double &stored = slot_angle_offset_[static_cast<std::size_t>(best_slot)];
                stored += 0.08 * (deviation - stored);
            }
        }
        ArmorObservation observation;
        // 本帧同编号板数：只有 >=2 块时 l/h 才可观测（两只都看到才能比较半径差）。
        {
            int same_id_visible = 0;
            for (const WorldObservation &candidate_obs : world)
            {
                if (candidate_obs.target_id == chosen.target_id)
                    ++same_id_visible;
            }
            observation.plates_visible = same_id_visible;
        }
        if (!smoothObservation(chosen, best_slot, timestamp_s, observation))
        {
            observation.position = chosen.position;
            observation.sigma_position_m = observationSigma(chosen.sigma_position_m);
        }
        // 板面法向进量测（借鉴 sp_vision_25 target.cpp:187-222 的第 4 维）：
        // 这一维直接把"这块板朝哪儿"喂给滤波器，是打破四块板 90° 对称的信息来源。
        // 约定沿用 initialize()：world_yaw_rad = 板面外法向（由车心指向板）。
        // PnP 法向朝里朝外取决于解算，故用 NORMAL_SIGN 环境变量做正反两个方向的
        // 一次性标定（见 docs/借鉴sp_vision25.md）。
        if (config_.plate_normal_measurement_enabled && chosen.has_yaw &&
            std::isfinite(chosen.yaw))
        {
            observation.has_world_yaw = true;
            observation.world_yaw_rad =
                wrapToPi(chosen.yaw + (config_.plate_normal_sign < 0.0 ? kPi : 0.0));
        }
        // 单板重锚：状态和这块板差得太多时，用观测把相位与中心直接对齐。
        // 单板 + 钉死半径时 (中心, 相位) 只有一个自由度被位置观测约束，滤波器
        // 可能整块板漂走（实拍 v4 自转车 f=1076 起状态相位偏 2.16 rad，四块
        // 模型板全错位，瞄准圈停在另一台车上）。位置门 0.6 m 又刚好放它进来，
        // NIS 门再拒掉更新 → 状态永远修不回来。
        if (config_.single_plate_realign_m > 0.0 && ekf_.initialized() &&
            last_updates_ == 0 && missed_frames_ >= std::max(config_.realign_after_misses, 0))
        {
            const int count = config_.ekf.plate_count;
            const Eigen::Vector3d predicted =
                platePosition(ekf_.state(), best_slot, count);
            if ((chosen.position - predicted).norm() > config_.single_plate_realign_m)
            {
                const double center_x = ekf_.state()(0);
                const double center_y = ekf_.state()(2);
                const double slot_phase =
                    2.0 * kPi * static_cast<double>(best_slot) / static_cast<double>(count);
                const double phase_target =
                    std::atan2(center_y - chosen.position.y(), center_x - chosen.position.x());
                const double theta_target = wrapToPi(phase_target - slot_phase);
                const double delta_theta = wrapToPi(theta_target - ekf_.state()(6));
                ++diagnostics_.single_plate_realigns;
                ekf_.rotatePhase(delta_theta);
                const Eigen::Vector3d plate_after =
                    platePosition(ekf_.state(), best_slot, count);
                ekf_.shiftCenter(chosen.position - plate_after);
            }
        }
        // 相位残差的"上坡"就是角速度误差：模型转得比真车慢，残差就单调涨。
        // 记下更新前的预测板面朝向，用来算这一步的残差（更新后就看不到了）。
        const double period_now = 2.0 * kPi / static_cast<double>(std::max(config_.ekf.plate_count, 1));
        const double face_pred =
            (chosen.has_yaw && std::isfinite(chosen.yaw))
                ? wrapToPi(ekf_.state()(6) + static_cast<double>(best_slot) * period_now + kPi)
                : std::numeric_limits<double>::quiet_NaN();
        if (!ekf_.update(observation, best_slot))
        {
            // 更新失败（观测把状态推出物理区间等）：只放弃这一块板，别放弃本帧
            // 剩下的板。v4 实拍里同一台车常同帧露出两块相邻板（相距 0.394 m），
            // 第一块失败就 break 会让第二块也白白丢掉。
            used[static_cast<std::size_t>(best_observation)] = true;
            continue;
        }
        // 相位由板法向直接钉住：模型里 plate = centre − R·[cos,sin](phase)，
        // 外法向方向 = phase + pi；因为 updatePhase 的残差按 2pi/N 折叠，
        // pi = 2*(2pi/N)（N=4）恰好是整数个周期，所以直接用观测法向当相位量测。
        // 逐帧位置更新推相位会被"圆心偏 0.2 m"带歪，法向不受深度误差影响。
        // 相位用"板心在拟合圆上的角度"量测：phase = atan2(p−c) − pi。
        // 这个量测只用位置、不吃噪声很大的法向；而且它只动相位(第 6 行)，
        // 不会像逐帧位置更新那样把车心一起拖走（实测全片段漂 0.4 m）。
        if (config_.circle_phase_enabled && ekf_.initialized())
        {
            const double dx = chosen.position.x() - ekf_.state()(0);
            const double dy = chosen.position.y() - ekf_.state()(2);
            if (std::hypot(dx, dy) > 0.05)
            {
                const double phase_meas = std::atan2(dy, dx) - kPi;
                const Eigen::Vector2d now_c(ekf_.state()(0), ekf_.state()(2));
                const Eigen::Vector3d plate_before =
                    platePosition(ekf_.state(), best_slot, std::max(config_.ekf.plate_count, 1));
                if (ekf_.updatePhase(phase_meas, config_.circle_phase_sigma_rad) &&
                    config_.phase_anti_anchor)
                {
                    const Eigen::Vector3d plate_after =
                        platePosition(ekf_.state(), best_slot, std::max(config_.ekf.plate_count, 1));
                    ekf_.shiftCenter(Eigen::Vector3d(
                        (plate_before.x() - plate_after.x()) + (now_c.x() - ekf_.state()(0)),
                        (plate_before.y() - plate_after.y()) + (now_c.y() - ekf_.state()(2)), 0.0));
                }
            }
        }
        if (config_.normal_phase_enabled && chosen.has_yaw && std::isfinite(chosen.yaw))
        {
            bool facing = true;
            const Eigen::Vector3d n_w(std::cos(chosen.yaw), std::sin(chosen.yaw), 0.0);
            const Eigen::Vector3d to_cam = -chosen.position;
            const double norm_tc = std::hypot(to_cam.x(), to_cam.y());
            if (norm_tc > 1.0e-6)
            {
                const double c = std::fabs(n_w.x() * to_cam.x() + n_w.y() * to_cam.y()) / norm_tc;
                facing = c >= config_.normal_phase_min_facing;
            }
            if (facing)
            {
                ekf_.updatePhase(chosen.yaw, config_.normal_phase_sigma_rad);
            }
        }
        noteObservationForNoise(chosen.position, timestamp_s);
        // 同板角速度量测：只有当这一步和上一步确认是同一块板（位移很小）时，
        // 板心绕当前车心的转角差才是干净的 ω，不会掺进"换板"造成的 90° 跳变。
        static const double tracklet_alpha_env = [] {
            const char *v = std::getenv("RATE_TRACKLET_ALPHA");
            return v ? std::atof(v) : -1.0;
        }();
        const double tracklet_alpha =
            tracklet_alpha_env >= 0.0 ? tracklet_alpha_env : config_.rate_tracklet_alpha;
        if (config_.rate_from_tracklet_enabled && ekf_.initialized())
        {
            const Eigen::Vector2d p_now(chosen.position.x(), chosen.position.y());
            const Eigen::Vector2d c_now(ekf_.state()(0), ekf_.state()(2));
            if (rate_has_last_)
            {
                const double step = (p_now - rate_last_position_).norm();
                const double dt = timestamp_s - rate_last_time_s_;
                if (step <= config_.rate_tracklet_max_step_m && dt > 1e-3 && dt < 0.2)
                {
                    const double a_now = std::atan2(p_now.y() - c_now.y(), p_now.x() - c_now.x());
                    const double a_prev = std::atan2(rate_last_position_.y() - c_now.y(),
                                                     rate_last_position_.x() - c_now.x());
                    const double dtheta = wrapToPi(a_now - a_prev);
                    const double omega_meas = dtheta / dt;
                    bool rate_ema_updated = false;
                    // 有符号的滑动平均：真自转方向恒定，平均后仍在；云台抖动正负抵消，
                    // 平均后接近 0。用它区分"真在转"和"抖出来的大角速度"。
                    if (std::isfinite(omega_meas) && std::fabs(omega_meas) < 12.0)
                    {
                        rate_signed_ema_ += config_.adaptive_spin_alpha * (omega_meas - rate_signed_ema_);
                        // 真自转：每一步方向相同，连续计数涨；云台抖动：方向来回变，
                        // 计数被扣到 0。只有"连续同向"够久才认为真在转。
                        spin_streak_ = (omega_meas * rate_signed_ema_ > 0.0)
                                           ? std::min(spin_streak_ + 1, 60)
                                           : std::max(spin_streak_ - 3, 0);
                    }
                    // 注意：这里不能把"EKF 当前 |ω| 大"也算作 spinning。那样一旦 ω
                    // 被噪声推大，就会永久自我确认（正反馈），合成 H 的静止+抖动
                    // 场景就是这么漂到 -1.2 rad/s 的。
                    const bool spinning =
                        spin_streak_ >= config_.rate_tracklet_min_streak &&
                        std::fabs(rate_signed_ema_) >= config_.rate_tracklet_min_rate_rad_s;
                        std::fabs(ekf_.state()(7)) >= config_.rate_tracklet_min_rate_rad_s;
                    const bool warmed_up = diagnostics_.accepted_updates >=
                                           static_cast<std::uint64_t>(config_.rate_tracklet_min_updates);
                    // 车心还没收敛时，绕它算出来的角度差是垃圾：要求板心到车心的
                    // 距离确实接近当前半径，才认为这个角度有意义。
                    const double r_now = std::max(ekf_.state()(8), 0.05);
                    const double dist_now = (p_now - c_now).norm();
                    const bool on_circle = std::fabs(dist_now - r_now) <= 0.06;
                    // 量测必须和"持续趋势"同号、且不偏离太远：单帧抖动常常符号相反
                    // 或幅度离谱，放进来会把 omega 撑到 1 rad/s 以上（合成 H 静止+抖动）。
                    const bool consistent =
                        omega_meas * rate_signed_ema_ > 0.0 &&
                        std::fabs(omega_meas - rate_signed_ema_) <=
                            config_.rate_tracklet_consistency_rad_s;
                    // 车心自己在快速移动（云台抖动/世界系晃）时，绕它算的角度差不可信。
                    const bool centre_still =
                        std::hypot(ekf_.state()(1), ekf_.state()(3)) <=
                        config_.rate_tracklet_max_centre_speed_m_s;
                    if (warmed_up && on_circle && spinning && centre_still &&
                        std::isfinite(omega_meas) && std::fabs(omega_meas) < 12.0)
                    {
                        // 作为 EKF 的第 5 维量测（带方差）灌进去，而不是 setRate 硬写：
                        // 硬写会把 omega 的不确定度信息抹掉，合成 F/G/H 就是这么失准的。
                        // 角度差噪声约 sigma_theta，除以 dt 得到角速度标准差；
                        // 下限 0.35 rad/s，免得 dt 很小的时候量测被当成绝对真值。
                        static const double sig_theta_env = [] {
                            const char *v = std::getenv("RATE_SIGMA");
                            return v ? std::atof(v) : -1.0;
                        }();
                        static const double sig_floor_env = [] {
                            const char *v = std::getenv("RATE_FLOOR");
                            return v ? std::atof(v) : -1.0;
                        }();
                        const double sig_theta = sig_theta_env > 0.0
                                                    ? sig_theta_env
                                                    : config_.rate_tracklet_sigma_theta_rad;
                        const double sig_floor = sig_floor_env > 0.0
                                                    ? sig_floor_env
                                                    : config_.rate_tracklet_sigma_floor_rad_s;
                        const double sigma_omega = std::max(0.35, sig_theta / dt);
                        const double sigma_used = std::max(sigma_omega, sig_floor);
                        if (ekf_.updateRate(omega_meas, sigma_used))
                        {
                            last_rate_update_s_ = timestamp_s;
                        }
                        diagnostics_.rate_measurements += 1;
                        // 只有"真的在转 + 板心确实在半径上"的量测才用来调 Q：
                        // 静止目标遇到云台抖动时，噪声差分会一路灌到 12 rad/s，
                        // 把 Q 顶得很高，omega 就自己飘到 -1.4（合成 H 就是这样挂的）。
                        measured_rate_ema_ = std::fabs(rate_signed_ema_);
                        rate_ema_updated = true;
                        applyProcessNoise();
                    }
                    if (!rate_ema_updated)
                    {
                        // 没有可信量测时让 Q 慢慢回落，别把一次抖动的记忆一直留着。
                        const double decay = config_.adaptive_spin_decay_rad_s * std::max(0.0, dt);
                        if (measured_rate_ema_ > 0.0)
                        {
                            measured_rate_ema_ = std::max(0.0, measured_rate_ema_ - decay);
                            applyProcessNoise();
                        }
                    }
                    static const bool rate_dbg = std::getenv("RATE_DEBUG") != nullptr;
                    if (rate_dbg)
                    {
                        std::fprintf(stderr,
                                     "[rate] t=%.3f meas=%+.2f ekf=%+.2f step=%.3f on_circle=%d "
                                     "warm=%d spin=%d used=%d sigma=%.2f\n",
                                     timestamp_s, omega_meas, ekf_.state()(7), step,
                                     on_circle ? 1 : 0, warmed_up ? 1 : 0, spinning ? 1 : 0,
                                     (warmed_up && on_circle && spinning &&
                                      std::isfinite(omega_meas) && std::fabs(omega_meas) < 12.0)
                                         ? 1
                                         : 0,
                                     std::max(0.35, config_.rate_tracklet_sigma_theta_rad / dt));
                    }
                }
            }
            rate_last_position_ = p_now;
            rate_last_time_s_ = timestamp_s;
            rate_has_last_ = true;
        }
        // 角速度修正（PLL）：残差斜率 = 真角速度 − 当前估计角速度。
        if (config_.rate_from_residual_enabled && std::isfinite(face_pred) &&
            std::isfinite(chosen.yaw) && ekf_.initialized())
        {
            const double residual = wrapToPi(chosen.yaw - face_pred);
            rate_residual_samples_.push_back(RateResidual{timestamp_s, residual});
            while (!rate_residual_samples_.empty() &&
                   timestamp_s - rate_residual_samples_.front().timestamp_s >
                       config_.rate_residual_window_s)
            {
                rate_residual_samples_.pop_front();
            }
            if (static_cast<int>(rate_residual_samples_.size()) >= config_.rate_residual_min_samples)
            {
                double sum_t = 0.0, sum_r = 0.0, sum_tt = 0.0, sum_tr = 0.0;
                const double t0 = rate_residual_samples_.front().timestamp_s;
                double prev = 0.0, unwrapped = 0.0;
                bool first = true;
                for (const RateResidual &sample : rate_residual_samples_)
                {
                    // 残差本身按 2pi 折叠，时间相邻时步长很小，可以直接展开。
                    if (first)
                    {
                        unwrapped = sample.residual_rad;
                        first = false;
                    }
                    else
                    {
                        double step = sample.residual_rad - prev;
                        step = wrapToPi(step);
                        unwrapped += step;
                    }
                    prev = sample.residual_rad;
                    const double t = sample.timestamp_s - t0;
                    sum_t += t;
                    sum_r += unwrapped;
                    sum_tt += t * t;
                    sum_tr += t * unwrapped;
                }
                const double n = static_cast<double>(rate_residual_samples_.size());
                const double denom = n * sum_tt - sum_t * sum_t;
                if (denom > 1e-6)
                {
                    const double slope = (n * sum_tr - sum_t * sum_r) / denom;
                    const double corrected =
                        ekf_.state()(7) + config_.rate_residual_gain * slope;
                    ekf_.setRate(std::clamp(corrected, -12.0, 12.0));
                }
            }
        }
        // 整车重拟合用的滑窗样本（板心轨迹 = 一个圆）
        body_samples_.push_back(
            LockSample{timestamp_s, chosen.position, chosen.yaw, chosen.target_id, chosen.size, false});
        // 同板轨迹段：一旦相邻帧位移超过阈值就说明换板了，从新的一块开始
        if (!tracklet_samples_.empty() &&
            (chosen.position - tracklet_samples_.back().position).norm() > config_.tracklet_max_step_m)
        {
            tracklet_samples_.clear();
        }
        tracklet_samples_.push_back(
            LockSample{timestamp_s, chosen.position, chosen.yaw, chosen.target_id, chosen.size, false});
        while (!body_samples_.empty() &&
               timestamp_s - body_samples_.front().timestamp_s > config_.body_refit_window_s)
        {
            body_samples_.pop_front();
        }
        // 装甲板法向直接给出"从板心指向转轴"的方向：车心 = 板心 − R·n。
        // 这条式子不用槽位、不用相位，等于给车心一个只由板朝向量出来的独立量测。
        // 之前所有位置更新都是拿"滤波自己的相位"去算预测板位，法向只用于相位更新；
        // 现在把法向当几何约束直接反推车心，正是"整车建模"该用的量。
        if (config_.center_from_normal_enabled && chosen.has_yaw && std::isfinite(chosen.yaw))
        {
            const double radius_now = ekf_.state()(8);
            // 高速自转时法向更新的预测分支容易选错（合成 G 场景 omega>=4 rad/s 会翻车），
            // 只在转速不高时相信它。
            const bool rate_ok = std::fabs(ekf_.state()(7)) <= config_.center_from_normal_max_rate_rad_s;
            if (radius_now > 0.01 && rate_ok)
            {
                const Eigen::Vector2d n(std::cos(chosen.yaw), std::sin(chosen.yaw));
                const Eigen::Vector2d p_xy(chosen.position.x(), chosen.position.y());
                const Eigen::Vector2d c_pred(ekf_.state()(0), ekf_.state()(2));
                // PnP 法向符号可能翻，两只候选 c = p ∓ R·n 取离预测近的那只，
                // 这不是先验，只是在两个候选里做出选择。
                const Eigen::Vector2d c_a = p_xy - radius_now * n;
                const Eigen::Vector2d c_b = p_xy + radius_now * n;
                const Eigen::Vector2d c =
                    ((c_a - c_pred).norm() <= (c_b - c_pred).norm()) ? c_a : c_b;
                if ((c - c_pred).norm() <= config_.center_from_normal_gate_m)
                {
                    ekf_.updateCenter(Eigen::Vector3d(c.x(), c.y(), ekf_.state()(4)),
                                      config_.center_from_normal_sigma_m);
                    ++diagnostics_.center_from_normal_updates;
                }
            }
        }
        // "看见过"=这块板真的被用来更新了（或者观测就落在这块板的预测位置上）。
        // 只按位置门判定会把相邻板也算成看见：四块板相距 0.35 m，位置门 0.6 m，
        // 于是"没看见的板"也进了瞄准候选，又挑回离云台最近的那块去了。
        if (best_slot >= 0 && best_slot < static_cast<int>(last_seen_s_.size()))
        {
            last_seen_s_[static_cast<std::size_t>(best_slot)] = timestamp_s;
            last_seen_position_[static_cast<std::size_t>(best_slot)] = chosen.position;
            {
                const std::size_t index = static_cast<std::size_t>(best_slot);
                const int count = aim_history_count_[index];
                const int slot_in_ring = count % kAimHistory;
                aim_history_[index][static_cast<std::size_t>(slot_in_ring)] = chosen.position;
                aim_history_s_[index][static_cast<std::size_t>(slot_in_ring)] = timestamp_s;
                aim_history_count_[index] = std::min(count + 1, kAimHistory);
            }
        }
        // 瞄准槽位跟着观测到的板走：4 块板物理上等价，把瞄准点绑在"刚看到
        // 的那块"上误差最小；不重绑的话对外发布的瞄准点会指向模型外推 90 度
        // 的另一块板，误差随相位漂移一起放大。
        if (best_slot != aim_slot_)
        {
            aim_slot_ = best_slot;
            diagnostics_.aim_slot = best_slot;
        }
        used[static_cast<std::size_t>(best_observation)] = true;
        slot_of_observation[static_cast<std::size_t>(best_observation)] = best_slot;
        last_updated_slot_ = best_slot;
        if (config_.obs_continuity_gate_enabled)
        {
            last_accepted_position_ = chosen.position;
            has_last_accepted_position_ = true;
        }
        if (config_.tangent_rate_enabled && ekf_.initialized())
        {
            const Eigen::Vector2d p_now(chosen.position.x(), chosen.position.y());
            const Eigen::Vector2d c_now(ekf_.state()(0), ekf_.state()(2));
            const double radius = ekf_.state()(8);
            const double dt = tangent_has_last_ ? timestamp_s - tangent_last_time_s_ : 0.0;
            if (tangent_has_last_ && dt > 1.0e-3 && dt < 0.20 && radius > 1.0e-3)
            {
                const Eigen::Vector2d step = p_now - tangent_last_position_;
                const double step_norm = step.norm();
                if (step_norm < config_.tangent_rate_max_step_m && step_norm > 1.0e-5)
                {
                    // 观测处的切向单位向量：p = c - R*u(theta) =>
                    // u(theta) = (c - p)/R，omega>0 时板速度方向 = (sin, -cos)
                    const double ux = (c_now.x() - p_now.x()) / radius;
                    const double uy = (c_now.y() - p_now.y()) / radius;
                    const Eigen::Vector2d tangent_unit(uy, -ux);
                    const double rate_meas = step.dot(tangent_unit) / (radius * dt);
                    if (std::isfinite(rate_meas) && std::fabs(rate_meas) < 12.0)
                    {
                        const double alpha = std::clamp(config_.tangent_rate_alpha, 0.0, 1.0);
                        const double rate_new =
                            (1.0 - alpha) * ekf_.state()(7) + alpha * rate_meas;
                        ekf_.setRate(rate_new);
                    }
                }
            }
            tangent_last_position_ = p_now;
            tangent_last_time_s_ = timestamp_s;
            tangent_has_last_ = true;
        }
        ++accepted;
        diagnostics_.last_nis = best_nis;
        if (chosen.has_yaw)
        {
            // 相位观测：板面法向直接约束车体 yaw（折叠掉 2pi/N 的换板歧义）。
            // 它和位置观测吃的是同一帧的同一块板，但因为折叠，换板不会把
            // 相位带歪——这正是位置关联最脆弱的地方。
            const bool phase_from_yaw =
                config_.phase_update_enabled ||
                (config_.phase_update_when_static && best_slot == last_updated_slot_ &&
                 std::fabs(ekf_.state()(7)) < config_.static_rate_rad_s);
            // 按槽位一致的相位更新：best_slot 已经被位置更新用掉了，相位更新
            // 用同一个槽位，两者说的是同一件事。旋转目标上必须走这条——折叠版
            // 只能修 45 度以内，锁不住槽位，实测 v4 上 yaw 单帧跳 140~215 度。
            // 只在"同帧存在第二块同 id 板"时用槽位锁相位：那正是折叠版的
            // 90 度歧义真正发作的地方（两块板都能解释观测，相位在解之间翻转）。
            // 单板帧没有歧义，折叠版已经够用，不用槽位锁去打扰正常演化。
            // 同帧里被接受进滤波器的观测数（含本帧前面已接受的那几块板）。
            const bool two_plates_visible = accepted >= 2;
            const bool slot_phase =
                config_.slot_phase_update_enabled && best_slot >= 0 && two_plates_visible &&
                std::fabs(ekf_.state()(7)) >= config_.slot_phase_min_rate_rad_s;
            if (slot_phase || phase_from_yaw)
            {
                // 反锚：先记下这块板在模型里的位置，改完 yaw 再把中心平移回去，
                // 让板留在观测到的位置。单板可见时 (center, yaw) 是规范自由度，
                // 不反锚就是"yaw 更新把板挪走、位置更新再拉回来"互相打架。
                const Eigen::Vector3d plate_before =
                    rotating_armor::platePosition(ekf_.state(), best_slot,
                                                  ekf_.config().plate_count);
                const bool phase_ok =
                    slot_phase
                        ? ekf_.updatePhaseForSlot(chosen.yaw, config_.phase_update_sigma_rad,
                                                  best_slot)
                        : ekf_.updatePhase(chosen.yaw, config_.phase_update_sigma_rad);
                if (phase_ok)
                {
                    const Eigen::Vector3d plate_after =
                        rotating_armor::platePosition(ekf_.state(), best_slot,
                                                      ekf_.config().plate_count);
                    if (config_.phase_anti_anchor)
                    {
                        ekf_.shiftCenter(plate_before - plate_after);
                    }
                }
            }
            pushNormalHistory(best_slot, timestamp_s, chosen.yaw);
        }
    }

    // 同帧两块板的中点 = 车体中心量测（对置板：中点即圆心；相邻板：沿弦法向
    // 偏 R*cos45）。单板可见时中心与相位是一对规范自由度，滤波器只能二选一；
    // 双板帧把这两个自由度同时钉住，是实拍里唯一能压相位误差的硬约束。
    // 双板中心/相位更新：两块板的位置直接给出车心（中点）与姿态（弦方向），
    // 位置噪声 ±5cm 下相位误差约 10°，比单板法向（±25°）好得多。但高速自转时
    // 两板中点受相位滞后影响、弦方向也在转，单帧直接拉会把估计带偏（合成 G 组
    // 全挂），所以和半径学习用同一个门：目标连续 10 帧判为 static 才启用。
    // 双板弦方向 → 相位量测（不需要 ω）：两块相邻板连线方向只由相位决定，
    //   angle(p2-p1) = phi + s1*period - pi/4   (mod 2pi)
    // 所以 phi 的等价法向 yaw = angle(p2-p1) + pi/4 + pi，与"哪块是哪块"无关
    // （顺序反了差 2*period，折叠后为零）。弦长 0.40 m 的基线让方向噪声降到
    // ~0.1 rad，比单板法向的 ±25° 稳得多，而且它不吃 ω，不会像批处理那样发散。
    if (config_.pair_phase_update_enabled && ekf_.initialized() &&
        std::fabs(ekf_.state()(7)) >= config_.static_rate_rad_s)
    {
        int pair_index[2] = {-1, -1};
        int pair_count = 0;
        for (std::size_t i = 0; i < world.size() && pair_count < 2; ++i)
        {
            if (target_id_ != 0 && world[i].target_id != target_id_)
            {
                continue;
            }
            pair_index[pair_count++] = static_cast<int>(i);
        }
        if (pair_count == 2 && world.size() == 2)
        {
            const Eigen::Vector3d p1 = world[pair_index[0]].position;
            const Eigen::Vector3d p2 = world[pair_index[1]].position;
            const double chord = (p2 - p1).norm();
            const double radius = ekf_.state()(8);
            const double period = 2.0 * kPi / static_cast<double>(std::max(plate_count, 1));
            const double expected = radius * 2.0 * std::sin(period * 0.5);
            if (std::fabs(chord - expected) < 0.15)
            {
                const double ang = std::atan2(p2.y() - p1.y(), p2.x() - p1.x());
                const double measured_world_yaw =
                    rotating_armor::wrapAngle(ang + 0.25 * period + kPi);
                const Eigen::Vector3d plate_before =
                    rotating_armor::platePosition(ekf_.state(), aim_slot_ >= 0 ? aim_slot_ : 0,
                                                  plate_count);
                if (ekf_.updatePhase(measured_world_yaw, config_.pair_phase_sigma_rad) &&
                    config_.phase_anti_anchor)
                {
                    const Eigen::Vector3d plate_after =
                        rotating_armor::platePosition(ekf_.state(), aim_slot_ >= 0 ? aim_slot_ : 0,
                                                      plate_count);
                    ekf_.shiftCenter(plate_before - plate_after);
                }
                ++diagnostics_.pair_phase_updates;
            }
        }
    }

    if (config_.pair_center_update_enabled && static_streak_ >= config_.pair_center_min_static_frames)
    {
        // 注意：这里刻意不看 used[] / slot_of_observation[]。双板帧里第二块板
        // 常常因为"第一块板刚更新过、预测已经动过"而被 NIS 门拒掉，可它恰恰是
        // 最能钉死中心与相位的量测。所以按几何把每块同 id 观测分到最近的模型板
        // 槽位，只要两块落在不同槽位、弦长与模型构型相符就采用。
        const double radius = ekf_.state()(8);
        int index[2] = {-1, -1};
        int slots[2] = {-1, -1};
        int found = 0;
        for (std::size_t i = 0; i < world.size() && found < 2; ++i)
        {
            if (world[i].target_id != target_id_ || !world[i].position.allFinite())
            {
                continue;
            }
            int best_slot = -1;
            double best_distance = 1.0e9;
            for (int slot = 0; slot < plate_count; ++slot)
            {
                const Eigen::Vector3d predicted =
                    rotating_armor::platePosition(ekf_.state(), slot, plate_count);
                const double distance = (world[i].position - predicted).norm();
                if (distance < best_distance)
                {
                    best_distance = distance;
                    best_slot = slot;
                }
            }
            if (best_distance > config_.pair_center_gate_m || best_slot < 0)
            {
                continue;
            }
            if (found == 0)
            {
                index[0] = static_cast<int>(i);
                slots[0] = best_slot;
                found = 1;
            }
            else if (best_slot != slots[0])
            {
                index[1] = static_cast<int>(i);
                slots[1] = best_slot;
                found = 2;
            }
        }
        if (found == 2 && slots[0] != slots[1] && radius > 0.01)
        {
            const Eigen::Vector3d p1 = world[index[0]].position;
            const Eigen::Vector3d p2 = world[index[1]].position;
            const Eigen::Vector3d chord = p2 - p1;
            const double chord_length = chord.norm();
            const double ratio = chord_length / radius;
            // 板间距：相邻板 2R sin(pi/N)、对置板 2R。只认这两种构型。
            const double period = 2.0 * kPi / static_cast<double>(plate_count);
            const double expected_adjacent = 2.0 * radius * std::sin(period * 0.5);
            const bool opposite = std::fabs(chord_length - 2.0 * radius) <=
                                  0.25 * radius;
            const bool adjacent = std::fabs(chord_length - expected_adjacent) <=
                                  0.25 * radius;
            if ((opposite || adjacent) && chord_length > 1e-6)
            {
                // 弦方向 = 两块板角度的中值 - 90 度；中值 = theta + (s1+s2)/2*period
                const double chord_angle = std::atan2(chord.y(), chord.x());
                const double mid_angle = chord_angle + kHalfPi;
                const double slot_mid =
                    0.5 * static_cast<double>(slots[0] + slots[1]) * period;
                double theta_pair = wrapToPi(mid_angle - slot_mid);
                // 只有 period 的可辨识度：取离当前相位最近的一支。
                double delta = wrapToPi(theta_pair - ekf_.state()(6));
                // 相位只有 period（90°）可辨识：把 delta 折到 [-period/2, period/2)。
                // 这里必须 while——单次 if 在相位已经偏出半个周期以上时折不回来，
                // 实拍里会出现 -1.5 rad 的"合理"delta 把模型一下转 90°。
                while (std::fabs(delta) > 0.5 * period)
                {
                    delta -= std::copysign(period, delta);
                }
                // 双板几何配对是最干净的一对观测：顺手把两块板的槽位几何
                // （半径/角度/高度偏移）学一下，只影响显示，不参与滤波。
                // 受 slot_offsets_enabled 控制：打开时四块模型板各有圆心，
                // 与"同速度同方向同圆心"的要求冲突；默认关闭。
                for (int k = 0; config_.slot_offsets_enabled && k < 2; ++k)
                {
                    const Eigen::Vector3d &pp = world[index[k]].position;
                    const int sl = slots[k];
                    const Eigen::Vector2d radial(pp.x() - ekf_.state()(0), pp.y() - ekf_.state()(2));
                    const double dr = radial.norm() - ekf_.state()(8);
                    if (std::fabs(dr) < 0.12)
                    {
                        double &stored = slot_radius_offset_[static_cast<std::size_t>(sl)];
                        stored += 0.15 * (dr - stored);
                    }
                    if (radial.norm() > 0.05)
                    {
                        const double observed_angle = std::atan2(radial.y(), radial.x());
                        const double model_angle = ekf_.state()(6) +
                                                   static_cast<double>(sl) * period + kPi;
                        double dev = wrapToPi(observed_angle - model_angle);
                        while (std::fabs(dev) > 0.5 * period)
                        {
                            dev -= std::copysign(period, dev);
                        }
                        double &stored = slot_angle_offset_[static_cast<std::size_t>(sl)];
                        stored += 0.15 * (dev - stored);
                    }
                    const double dz = pp.z() - ekf_.state()(4);
                    if (std::fabs(dz) < 0.20)
                    {
                        double &stored = slot_z_offset_[static_cast<std::size_t>(sl)];
                        stored += 0.15 * (dz - stored);
                    }
                }
                ekf_.rotatePhase(delta);
                // 相位定了以后中心由任一块板反推：c = p + R*u(theta + s*period)
                const double angle1 =
                    ekf_.state()(6) + static_cast<double>(slots[0]) * period;
                // 用两块板各自反推车心再平均：只用一块板时，那块板的 PnP
                // 深度/横向偏差会原样进到车心里；两块反向平均能抵掉一部分。
                const double angle2 =
                    ekf_.state()(6) + static_cast<double>(slots[1]) * period;
                const Eigen::Vector3d center_measured =
                    0.5 * ((p1 + radius * Eigen::Vector3d(std::cos(angle1), std::sin(angle1), 0.0)) +
                           (p2 + radius * Eigen::Vector3d(std::cos(angle2), std::sin(angle2), 0.0)));
                const double sigma = std::max(
                    config_.pair_center_sigma_m,
                    std::hypot(world[index[0]].sigma_position_m,
                               world[index[1]].sigma_position_m) /
                        std::sqrt(2.0));
                ekf_.updateCenter(center_measured, sigma);
            }
        }
    }

    for (std::size_t i = 0; i < world.size(); ++i)
    {
        if (!used[i])
        {
            ++diagnostics_.rejected_this_frame;
        }
    }
    return accepted;
}

bool RotatingArmorTracker::declarationContradicted(
    const std::vector<WorldObservation> &world, double timestamp_s)
{
    (void)world;
    std::size_t best_index = 0;
    std::size_t best_size = 0;
    for (std::size_t slot = 0; slot < normal_history_per_slot_.size(); ++slot)
    {
        if (normal_history_per_slot_[slot].size() > best_size)
        {
            best_size = normal_history_per_slot_[slot].size();
            best_index = slot;
        }
    }
    if (best_size < 4)
    {
        return false;
    }
    const auto &history = normal_history_per_slot_[best_index];
    if (timestamp_s - history.front().first < config_.relock_min_span_s)
    {
        return false;
    }
    std::vector<double> times;
    std::vector<double> values;
    times.reserve(history.size());
    values.reserve(history.size());
    for (const auto &entry : history)
    {
        times.push_back(entry.first);
        values.push_back(entry.second);
    }
    const LinearFit fit = fitLinear(times, values);
    if (!fit.ok)
    {
        return false;
    }
    if (std::fabs(fit.slope - ekf_.state()(7)) > config_.relock_omega_diff_rad_s)
    {
        ++contradiction_streak_;
    }
    else
    {
        contradiction_streak_ = 0;
    }
    return contradiction_streak_ >= std::max(config_.relock_frames, 1);
}

PredictionResult RotatingArmorTracker::buildResult(const GimbalState &gimbal,
                                                   double timestamp_s)
{
    PredictionResult result;
    if (!ekf_.initialized())
    {
        return result;
    }
    const State &state = ekf_.state();

    // 运动类型完全由状态量判：这不是"声明"，是每帧从估计值读出来的结论，
    // 转速可信度看 rate_obs_sigma_rad_s，别把两者混在一起。
    const double speed = std::hypot(state(1), state(3));
    diagnostics_.speed_m_s = speed;
    const double spin = std::fabs(state(7));
    if (spin >= config_.static_rate_rad_s)
    {
        diagnostics_.motion = MotionMode::Spinning;
    }
    else if (speed >= config_.motion_translating_speed_m_s)
    {
        diagnostics_.motion = MotionMode::Translating;
    }
    else
    {
        diagnostics_.motion = MotionMode::Static;
    }
    // "不转"就够了：平移目标的弦长/中点同样能给出半径与相位，没必要连速度也压到
    // 静止。早期用 motion == Static，实拍里速度估计的噪声让 streak 反复归零，
    // 双板更新与半径学习 200 帧只触发 1 次。
    static_streak_ =
        std::fabs(ekf_.state()(7)) < config_.static_rate_rad_s ? static_streak_ + 1 : 0;

    // 新鲜度按"距上次被接收的更新多少秒"判，不按丢帧数：相机帧率变化时
    // 帧数门限的含义会漂移（166fps 下 2 帧只有 12ms，实拍里会长期卡在 Unstable）。
    const double stale_s = has_accept_ ? (timestamp_s - last_accept_s_) : 1e9;
    const double stale_window_s = staleWindowSeconds();
    const bool fresh_enough = has_accept_ && stale_s <= stale_window_s;
    diagnostics_.stability = stable_reason(consecutive_updates_ >= config_.stable_updates,
                                             ekf_.phase_resolved(), fresh_enough,
                                             timestamp_s - last_accept_s_);
    const bool lock_mature = lock_start_s_ < 0.0 ||
                             (timestamp_s - lock_start_s_) >= config_.stable_min_lock_s;
    const bool stable = lock_mature && consecutive_updates_ >= config_.stable_updates &&
                        ekf_.phase_resolved() && fresh_enough &&
                        (timestamp_s - last_accept_s_ <= config_.timeout_s);
    result.state = stable ? TrackingState::Stable : TrackingState::Unstable;
    result.target_id = static_cast<std::uint8_t>(std::min(std::max(target_id_, 0), 255));
    result.center_x_m = static_cast<float>(state(0));
    result.center_y_m = static_cast<float>(state(2));
    result.center_velocity_x_m_s = static_cast<float>(state(1));
    result.center_velocity_y_m_s = static_cast<float>(state(3));
    result.body_yaw = static_cast<float>(wrapToPi(state(6)));
    result.body_yaw_velocity = static_cast<float>(state(7));
    result.radius_1_m = static_cast<float>(state(8));
    result.radius_2_m = static_cast<float>(state(8));

    // 预测到"子弹该对准的时刻"：
    //   控制器下发的预测时间偏置（秒）+ 本帧实际测得的视觉处理延迟（秒）。
    // 两者是不同环节：bias 由电控给出（它知道弹道飞行时间与自身回路延迟），
    // measured_latency_s 是"取到这一帧到算出结果"之间已经花掉的时间。
    const double bias = std::isfinite(gimbal.prediction_bias_s)
                            ? clampd(static_cast<double>(gimbal.prediction_bias_s), 0.0,
                                     config_.max_bias_s)
                            : 0.0;
    const double latency = std::isfinite(measured_latency_s_)
                               ? clampd(measured_latency_s_, 0.0, config_.max_latency_s)
                               : 0.0;

    // 弹道飞行时间：云台要指向"子弹到靶那一刻板的位置"，只补视觉延迟不够。
    // 30 m/s、2 m 距离就是 67 ms，板以 3 rad/s 转着，这 67 ms 里已经跑了 5 cm ——
    // 超过小板半宽（6.75 cm）的一半。距离用预测后的车心估计，迭代两次收敛
    // （飞行时间对距离的依赖很弱，两次足够）。
    double flight_s = 0.0;
    if (config_.flight_compensation_enabled && config_.bullet_speed_m_s > 1.0)
    {
        for (int iteration = 0; iteration < 2; ++iteration)
        {
            RotatingArmorEkf stage = ekf_;
            if (!stage.predict(bias + latency + flight_s))
                break;
            const State &s = stage.state();
            const double range = std::hypot(s(0), s(2));
            if (!(range > 1.0e-3) || !std::isfinite(range))
                break;
            flight_s = clampd(range / config_.bullet_speed_m_s, 0.0, config_.max_flight_s);
        }
    }

    RotatingArmorEkf probe = ekf_;
    if (!probe.predict(bias + latency + flight_s))
    {
        return PredictionResult{};
    }
    const int plate_count = config_.ekf.plate_count;
    if (plate_count <= 0)
    {
        return PredictionResult{};
    }

    // 选瞄准的装甲板：预测时刻上"离当前云台指向最近"的那块，换板带迟滞。
    //
    // 但"离云台最近"只能用来兜底：四块板在 2.5 m 外只差几度，按角度挑+迟滞
    // 会永久锁在相邻那块上（实拍 v4 静止 1384 帧全部 aim=3，而每一帧被看到的
    // 都是 slot 0 —— 瞄准点因此画在板外 0.35 m 处，正是"紫圈跑到板外面"）。
    // 正确的瞄准点是"最近一次被接受更新关联上的那块板"在未来时刻的位置：
    // 同一块板按 ω 外推，换板由关联结果驱动（update 里 aim_slot_ = best_slot）。
    // 朝向过滤：外法向 = (板心 − 中心)/R。朝向云台那一面才有可能被看见/打中；
    // "离云台最近"的角度判据对"同一条视线上的近板与远板"完全打平（两者方位角
    // 一模一样），不按朝面过滤就会挑到车体背面那块（合成 8 rad/s 实测一半帧
    // 瞄在背面板上）。
    const Eigen::Vector3d body_center(probe.state()(0), probe.state()(2), probe.state()(4));
    int best_slot = -1;
    // 首选"刚刚被观测到的那块板"，而不是重新按角度挑。
    //
    // 关联结果已经把它写在 aim_slot_ 里（associateAndUpdate 中 aim_slot_ =
    // best_slot）。如果这里再用"离云台最近"重挑，四块板差几度时就会挑到相邻
    // 那块上——实拍 v4 自转车 id=3 上，47% 的帧 aim_slot 与真正看到的那块板
    // 不同，瞄准残差中位被抬到 159 mm（而观测到最近板只有 87 mm）。
    // 用户的要求也正是"瞄现在正对着的那块"，所以只要它还在"刚看见"窗口内就直接用。
    if (aim_slot_ >= 0 && aim_slot_ < plate_count &&
        aim_slot_ < static_cast<int>(last_seen_s_.size()))
    {
        const double seen = last_seen_s_[static_cast<std::size_t>(aim_slot_)];
        if (seen >= 0.0 && timestamp_s - seen <= config_.aim_visible_hold_s)
        {
            // 保持上次瞄准的槽位（不按角度重挑，避免四块板差几度时来回跳），但
            // 必须确认它在"开火时刻"仍然朝向相机：提前量里有处理延迟 + 弹丸飞行
            // 时间（3.9 m / 30 m/s = 0.13 s），8 rad/s 的板在这段时间里转 82°，
            // 刚看见的那块到子弹到达时可能已经转到背面（合成场景 G 实测 76%）。
            const Eigen::Vector3d held = platePosition(probe.state(), aim_slot_, plate_count);
            const Eigen::Vector3d held_outward = held - body_center;
            const double held_norm = held_outward.norm();
            const double held_facing =
                held_norm > 1.0e-6 ? held_outward.dot(-held.normalized()) / held_norm : 0.0;
            if (held_facing > 0.0)
            {
                best_slot = aim_slot_;
            }
        }
    }
    for (int pass = 0; pass < 2 && best_slot < 0; ++pass)
    {
        double best_error = std::numeric_limits<double>::infinity();
        for (int slot = 0; slot < plate_count && slot < static_cast<int>(last_seen_s_.size());
             ++slot)
        {
            const double seen = last_seen_s_[static_cast<std::size_t>(slot)];
            if (seen < 0.0 || timestamp_s - seen > config_.aim_visible_hold_s)
            {
                continue; // 最近没见过这块板，不能拿它当瞄准点
            }
            const Eigen::Vector3d candidate = platePosition(probe.state(), slot, plate_count);
            const Eigen::Vector3d outward = candidate - body_center;
            const double outward_norm = outward.norm();
            const double facing =
                outward_norm > 1.0e-6 ? outward.dot(-candidate.normalized()) / outward_norm : 0.0;
            if (pass == 0 && facing <= 0.0)
            {
                continue; // 背面板：先不考虑，找不到朝面板再退回纯角度
            }
            const double error =
                std::max(std::fabs(wrapToPi(vectorYaw(candidate) - gimbal.yaw)),
                         std::fabs(vectorPitch(candidate) - gimbal.pitch));
            {
                best_error = error;
                best_slot = slot;
            }
        }
    }
    if (best_slot < 0 && aim_slot_ >= 0 && aim_slot_ < plate_count)
    {
        best_slot = aim_slot_;
    }
    if (best_slot < 0)
    {
        double best_error = std::numeric_limits<double>::infinity();
        for (int slot = 0; slot < plate_count; ++slot)
        {
            const Eigen::Vector3d candidate = platePosition(probe.state(), slot, plate_count);
            const double error =
                std::max(std::fabs(wrapToPi(vectorYaw(candidate) - gimbal.yaw)),
                         std::fabs(vectorPitch(candidate) - gimbal.pitch));
            if (error < best_error)
            {
                best_error = error;
                best_slot = slot;
            }
        }
    }
    if (best_slot < 0)
    {
        return PredictionResult{};
    }
    aim_slot_ = best_slot;
    diagnostics_.aim_slot = best_slot;

    Eigen::Vector3d plate = platePosition(probe.state(), best_slot, plate_count);
    // 只有当结果不向前预测（零延迟、零 bias）时才直接用观测板心：一旦要外推
    // bias+latency，EKF 的四板模型相位比"把观测绕中心转一圈"更可靠（合成场景 G
    // 里 8 rad/s、50 ms 延迟下，观测外推 5 cm，EKF 8 mm）。
    const double prediction_horizon = bias + latency;
    if (config_.aim_observation_hold_s > 0.0 &&
        (prediction_horizon <= 0.002 || config_.aim_observation_any_horizon) &&
        best_slot < static_cast<int>(last_seen_s_.size()))
    {
        const std::size_t slot_index = static_cast<std::size_t>(best_slot);
        const double seen = last_seen_s_[slot_index];
        if (seen >= 0.0 && timestamp_s - seen <= config_.aim_observation_hold_s &&
            last_seen_position_[slot_index].allFinite())
        {
            // 用观测到的板心，并按 omega 沿圆周外推到当前时刻：外推是绕
            // 中心刚体旋转的严格解（中心不动时板心只转不缩），零延迟时
            // 外推量接近 0，有 10~30 ms 检测延迟时正好补掉。
            plate = last_seen_position_[slot_index];
            // 观测是在 seen 时刻量的，而结果要送到 bias+latency 之后的"开火时刻"，
            // 两段都要按 omega 补：只补 lag 会在有处理延迟时系统性落后半个身位。
            // 弹道飞行时间同样要补：子弹到靶时板已经又转了一段。
            const double horizon = bias + latency + flight_s + (timestamp_s - seen);
            // 刚体推进（常速度 + 常角速度）：
            //   p(seen+Δ) = (c(t) − v·(t−seen)) + v·Δ + Rz(ω·Δ)·(p_obs − c(seen)),  Δ = horizon
            // 旋转基点必须是"观测时刻的中心"。用已经推到未来的 probe.state() 当基点，
            // 会把观测这一段的相位又转回去（实测使瞄准点系统性落后飞行时间 40 ms，
            // 见 docs/实验报告.md 的时间偏移扫描）。
            const Eigen::Vector3d center(ekf_.state()(0), ekf_.state()(2), ekf_.state()(4));
            const Eigen::Vector3d velocity(ekf_.state()(1), ekf_.state()(3), 0.0);
            const double omega = ekf_.state()(7);
            if (config_.aim_observation_extrapolate && horizon > 0.0 && std::isfinite(omega) &&
                std::fabs(omega) > 1.0e-3)
            {
                const Eigen::Vector3d base = center - velocity * (timestamp_s - seen);
                const Eigen::Vector3d relative = plate - base;
                const double angle = omega * horizon;
                const double c = std::cos(angle);
                const double s_ = std::sin(angle);
                plate = base + velocity * horizon +
                        Eigen::Vector3d(c * relative.x() - s_ * relative.y(),
                                        s_ * relative.x() + c * relative.y(),
                                        relative.z());
            }
            else if (config_.aim_observation_extrapolate && horizon > 0.0)
            {
                // 不转也要补平移：车沿视线方向动 1 m/s，67 ms 就是 6.7 cm。
                plate += velocity * horizon;
            }
            // 运动补偿后的一阶平滑：先把上一次的平滑结果按同样规则推进到现在，
            // 再和新观测混合。旋转是刚体运动，推进是精确的，所以 α 只牺牲噪声、
            // 不引入相位滞后。
            if (config_.aim_smoothing_alpha > 0.0 && aim_smoothed_valid_ &&
                aim_smoothed_s_ >= 0.0 && timestamp_s > aim_smoothed_s_ &&
                timestamp_s - aim_smoothed_s_ <= 0.10 &&
                aim_smoothed_world_.allFinite())
            {
                const double step = timestamp_s - aim_smoothed_s_;
                Eigen::Vector3d previous = aim_smoothed_world_;
                if (std::isfinite(omega) && std::fabs(omega) > 1.0e-3)
                {
                    const Eigen::Vector3d relative_previous = previous - center;
                    const double angle_previous = omega * step;
                    const double cp = std::cos(angle_previous);
                    const double sp = std::sin(angle_previous);
                    previous = center + Eigen::Vector3d(
                                            cp * relative_previous.x() - sp * relative_previous.y(),
                                            sp * relative_previous.x() + cp * relative_previous.y(),
                                            relative_previous.z());
                }
                const double alpha = config_.aim_smoothing_alpha;
                plate = (1.0 - alpha) * previous + alpha * plate;
            }
            // 局部直线拟合（最小二乘，各自对 t 拟合 x/y/z），在 timestamp_s 处取值。
            // 比 EMA 好的地方：不需要 omega，且用的是"窗口中点前后"的样本，不滞后。
            if (config_.aim_fit_samples >= 3 && config_.aim_fit_window_s > 0.0)
            {
                const std::size_t index = static_cast<std::size_t>(best_slot);
                const int count = aim_history_count_[index];
                if (count >= 3)
                {
                    double sum_t = 0.0, sum_tt = 0.0;
                    Eigen::Vector3d sum_p = Eigen::Vector3d::Zero();
                    int used = 0;
                    for (int i = 0; i < count; ++i)
                    {
                        const double sample_t = aim_history_s_[index][static_cast<std::size_t>(i)];
                        if (timestamp_s - sample_t > config_.aim_fit_window_s || sample_t > timestamp_s)
                        {
                            continue;
                        }
                        sum_t += sample_t;
                        sum_tt += sample_t * sample_t;
                        sum_p += aim_history_[index][static_cast<std::size_t>(i)];
                        ++used;
                    }
                    if (used >= 3)
                    {
                        const double denominator = used * sum_tt - sum_t * sum_t;
                        if (std::fabs(denominator) > 1.0e-12)
                        {
                            Eigen::Vector3d sum_tp = Eigen::Vector3d::Zero();
                            for (int i = 0; i < count; ++i)
                            {
                                const double sample_t =
                                    aim_history_s_[index][static_cast<std::size_t>(i)];
                                if (timestamp_s - sample_t > config_.aim_fit_window_s ||
                                    sample_t > timestamp_s)
                                {
                                    continue;
                                }
                                sum_tp += sample_t * aim_history_[index][static_cast<std::size_t>(i)];
                            }
                            const Eigen::Vector3d slope =
                                (used * sum_tp - sum_t * sum_p) / denominator;
                            const Eigen::Vector3d intercept = (sum_p - slope * sum_t) / used;
                            const Eigen::Vector3d fitted = intercept + slope * timestamp_s;
                            // 拟合外推太远（样本都挤在窗口边缘）时不采信。
                            const double extrapolation =
                                std::fabs(timestamp_s - sum_t / used);
                            if (extrapolation <= 0.5 * config_.aim_fit_window_s &&
                                fitted.allFinite())
                            {
                                plate = fitted;
                            }
                        }
                    }
                }
            }
            aim_smoothed_world_ = plate;
            aim_smoothed_s_ = timestamp_s;
            aim_smoothed_valid_ = true;
        }
    }
    // 长时间没有观测：冻结在最后看到的板心上，不用速度外推（外推会让瞄准圈
    // 在整段丢失里飘出去）。
    if (config_.aim_freeze_after_s > 0.0 && best_slot >= 0 &&
        best_slot < static_cast<int>(last_seen_s_.size()))
    {
        const double seen = last_seen_s_[static_cast<std::size_t>(best_slot)];
        if (seen >= 0.0 && timestamp_s - seen > config_.aim_freeze_after_s &&
            last_seen_position_[static_cast<std::size_t>(best_slot)].allFinite())
        {
            plate = last_seen_position_[static_cast<std::size_t>(best_slot)];
        }
    }
    diagnostics_.predicted_aim_world = plate;
    const double distance = plate.norm();
    if (!(distance > 1.0e-6))
    {
        return PredictionResult{};
    }
    const double yaw = vectorYaw(plate);
    const double pitch = vectorPitch(plate);

    // 瞄准点的角速度：在预测时刻再做一次 1ms 前向差分。
    double yaw_velocity = 0.0;
    double pitch_velocity = 0.0;
    RotatingArmorEkf ahead = probe;
    if (ahead.predict(kVelocityDt))
    {
        const Eigen::Vector3d plate_ahead = platePosition(ahead.state(), best_slot, plate_count);
        yaw_velocity = wrapToPi(vectorYaw(plate_ahead) - yaw) / kVelocityDt;
        pitch_velocity = (vectorPitch(plate_ahead) - pitch) / kVelocityDt;
    }

    result.distance_m = static_cast<float>(distance);
    result.target_yaw = static_cast<float>(yaw);
    result.target_pitch = static_cast<float>(pitch);
    result.target_yaw_velocity = static_cast<float>(yaw_velocity);
    result.target_pitch_velocity = static_cast<float>(pitch_velocity);

    const double yaw_error = std::fabs(wrapToPi(yaw - gimbal.yaw));
    const double pitch_error = std::fabs(pitch - gimbal.pitch);

    // 开火判据：把"角度对准"换算成"板面内的线性脱靶量"。
    // 旧判据 fire_angle_error_rad=0.05 在 1.5 m 就是 7.5 cm，比小板半宽
    // 6.75 cm 还大，竖直方向更是 2.85 cm 的两倍半 —— 角度"对上了"照样打飞。
    // 现在要求预测命中点落在板矩形内并留 40% 余量；竖直方向最紧，正好对应
    // 题目要求的"实际打中装甲板"，而不是"云台角误差小于某个弧度"。
    const ArmorSize plate_size =
        lock_samples_.empty() ? ArmorSize::Small : lock_samples_.back().size;
    const double half_width_m = (plate_size == ArmorSize::Large) ? 0.115 : 0.0675;
    const double half_height_m = 0.0285;
    constexpr double kFireMargin = 0.6;
    const double miss_h_m = distance * std::tan(std::min(yaw_error, 1.5));
    const double miss_v_m = distance * std::tan(std::min(pitch_error, 1.5));
    const bool aligned = miss_h_m <= kFireMargin * half_width_m &&
                         miss_v_m <= kFireMargin * half_height_m;
    result.fire_miss_m = static_cast<float>(std::max(miss_h_m, miss_v_m));
    result.fire_budget_rad = static_cast<float>(
        std::min(config_.fire_angle_error_rad,
                 (kFireMargin * std::min(half_width_m, half_height_m)) /
                     std::max(distance, 1.0e-3)));
    const bool fresh = has_accept_ && missed_frames_ == 0 &&
                       (timestamp_s - last_accept_s_ <= config_.fire_max_age_s);
    result.fire_allowed = stable && aligned && fresh &&
                          distance >= config_.min_distance_m &&
                          distance <= config_.max_distance_m;

    if (!predictionFinite(result))
    {
        return PredictionResult{};
    }
    if (const char *force = std::getenv("FORCE_RATE"))
    {
        ekf_.setRate(std::atof(force));
    }
    return result;
}

PredictionResult RotatingArmorTracker::step(const std::vector<ArmorPose> &observations,
                                            const GimbalState &gimbal,
                                            double timestamp_seconds,
                                            double measured_latency_s)
{
    diagnostics_.updates_this_frame = 0;
    diagnostics_.rejected_this_frame = 0;
    measured_latency_s_ = measured_latency_s;
    diagnostics_.measured_latency_s = measured_latency_s;
    timestamp_s_ = timestamp_seconds;
    if (!std::isfinite(timestamp_seconds) || !std::isfinite(gimbal.yaw) ||
        !std::isfinite(gimbal.pitch) || !std::isfinite(gimbal.roll))
    {
        dropTarget("bad_input");
        return PredictionResult{};
    }
    if (has_timestamp_ && timestamp_seconds < last_timestamp_s_ - 1.0e-9)
    {
        dropTarget("clock_backwards");
        return PredictionResult{};
    }
    has_timestamp_ = true;
    last_timestamp_s_ = timestamp_seconds;

    std::vector<WorldObservation> world;
    collectWorldObservations(observations, gimbal, world);

    // 本帧是否还有"锁定编号"的观测，以及本帧出现最多的非零编号（用于换目标）。
    //
    // 这里曾经用"本帧观测最多的编号"直接判换目标，在旋转车体上会误判：一台车
    // 同时露出两块同编号装甲板时，票数会投给它，而另一台车只有一块——于是锁定
    // 的车还在视野里却被判成"换了目标"，实测每 30 帧就重锁一次（见
    // test_camera_playback 回放 tmp/v4.avi 的 target_switch 日志）。
    // 改为：只有锁定编号连续两帧都检测不到、且本帧有其他编号时，才换目标。
    bool locked_id_present = false;
    int frame_id = 0;
    {
        std::map<int, int> counts;
        for (const WorldObservation &observation : world)
        {
            if (observation.target_id == 0)
            {
                continue;
            }
            if (observation.target_id == target_id_)
            {
                locked_id_present = true;
            }
            ++counts[observation.target_id];
        }
        int best = 0;
        for (const auto &entry : counts)
        {
            if (entry.second > best)
            {
                best = entry.second;
                frame_id = entry.first;
            }
        }
    }

    // 目标选择优先级：正压在准星上的那头。
    //
    // 旧逻辑只在"锁定编号连续两帧消失"时才换目标，于是两台车同时在场时，
    // 谁先被检测列表排到前面就永远锁谁（实拍 v4：右边静止车锁了 1385 帧，
    // 左边自转车一次都没被瞄准）。这里改成：另一台车的板连续 relock_frames 帧
    // 都比当前目标更靠近准星（带 switch_margin 迟滞），就主动丢目标重新锁，
    // 下一帧的 tryLock 会选准星最近的那块。
    if (config_.prefer_centered_target && ekf_.initialized() && !world.empty() &&
        target_id_ != 0)
    {
        double ours = std::numeric_limits<double>::infinity();
        double other_best = std::numeric_limits<double>::infinity();
        for (const WorldObservation &observation : world)
        {
            if (observation.target_id == 0)
            {
                continue;
            }
            const double angle = boresightAngle(observation.position, gimbal);
            if (observation.target_id == target_id_)
            {
                ours = std::min(ours, angle);
            }
            else
            {
                other_best = std::min(other_best, angle);
            }
        }
        if (std::isfinite(ours) && std::isfinite(other_best) &&
            other_best + config_.target_switch_margin_rad < ours)
        {
            if (++centered_streak_ >= config_.relock_frames)
            {
                centered_streak_ = 0;
                dropTarget("target_switch_centered");
                return PredictionResult{};
            }
        }
        else
        {
            centered_streak_ = 0;
        }
    }

    if (!ekf_.initialized())
    {
        diagnostics_.phase = LockPhase::Collecting;
        if (tryRestoreSnapshot(world, timestamp_seconds) ||
            tryLock(world, timestamp_seconds, gimbal))
        {
            return buildResult(gimbal, timestamp_seconds);
        }
        return PredictionResult{};
    }
    diagnostics_.phase = LockPhase::Tracking;

    // 目标切换：锁定编号连续两帧消失、且画面里有别的编号，才换目标。
    // 多目标模式下这一条必须关掉：一辆车的 EKF 只认自己的板号，
    // "这块板不见了"就换到另一辆车上去，正是用户看到的串车。换车由
    // MultiTargetTracker 新建一辆车来处理，旧车按 prune 超时注销。
    if (!config_.target_switch_enabled)
    {
        other_id_streak_ = 0;
    }
    else if (locked_id_present)
    {
        other_id_streak_ = 0;
    }
    else if (frame_id != 0 && frame_id != target_id_)
    {
        if (++other_id_streak_ >= 2)
        {
            dropTarget("target_switch");
            return PredictionResult{};
        }
    }

    const double dt = timestamp_seconds - state_timestamp_s_;
    if (dt > 0.0 && dt < 1.0)
    {
        recent_dt_.push_back(dt);
        while (recent_dt_.size() > 9)
        {
            recent_dt_.pop_front();
        }
    }
    if (!(dt >= 0.0) || dt > config_.timeout_s)
    {
        dropTarget("state_stale");
        return PredictionResult{};
    }
    if (!ekf_.predict(dt))
    {
        dropTarget("predict_failed");
        return PredictionResult{};
    }
    state_timestamp_s_ = timestamp_seconds;

    // 先用窗口化的法向斜率把 omega 拉正，再做位置更新：顺序反过来会让位置
    // 更新先按错误的 omega 外推、把中心推歪，再靠斜率量测去救。
    updateYawRateFromWindow(world, timestamp_seconds);
    // 证据持续缺席时把 omega 慢慢拉回 0：转速由数据决定，数据说不转就不转。
    // 窗口很弱（sigma 2.0），压不过一次真实的斜率量测（sigma 0.15 量级）。
    if (config_.rate_update_enabled &&
        timestamp_seconds - last_rate_update_s_ > config_.rate_decay_after_s &&
        std::fabs(ekf_.state()(7)) > 0.05)
    {
        if (ekf_.updateYawRate(0.0, config_.rate_decay_sigma_rad_s))
        {
            // 衰减本身也算动过 omega：盖章后下一轮要再等 rate_decay_after_s，
            // 否则这里每帧都在拉 0，把刚吃进来的斜率量测对冲掉。
            last_rate_update_s_ = timestamp_seconds;
        }
    }
    updateBodyFromPair(world);
    updateRadiusFromPairs(world);
    noteChordRadius(world, timestamp_seconds);
    const int updates = associateAndUpdate(world, timestamp_seconds);
    // 整车重拟合：小陀螺车原地自转，板心轨迹就是一个圆。用最近一段（滑窗）
    // 的板心做圆拟合，把车心与半径整体校一次 —— 逐帧位置更新只能修相位，
    // 修不了"圆心整体偏了 0.2 m"这种系统误差；长窗口拟合能。
    if (config_.body_refit_enabled && ekf_.initialized() &&
        diagnostics_.phase == LockPhase::Tracking && refit_center_valid_)
    {
        // 车心钉回最近一次圆拟合的结果：逐帧位置更新只会把它拽来拽去
        // （实测整个片段漂 0.4 m），而圆拟合在 2 s 窗口上的残差只有 1.1 cm。
        const Eigen::Vector2d now_c(ekf_.state()(0), ekf_.state()(2));
        const Eigen::Vector2d d = refit_center_ - now_c;
        if (d.norm() > 1.0e-9)
        {
            ekf_.shiftCenter(Eigen::Vector3d(d.x(), d.y(), 0.0));
        }
    }
    if (config_.body_refit_enabled && ekf_.initialized() &&
        diagnostics_.phase == LockPhase::Tracking)
    {
        ++body_refit_countdown_;
        if (body_refit_countdown_ >= config_.body_refit_every)
        {
            body_refit_countdown_ = 0;
            const CircleFit fit = fitArmorCircle(body_samples_);
            const int count_now = std::max(config_.ekf.plate_count, 1);
            if (fit.ok && static_cast<int>(body_samples_.size()) >= config_.body_refit_min_samples &&
                fit.residual <= config_.body_refit_max_residual &&
                fit.radius >= config_.radius_min_m && fit.radius <= config_.radius_max_m)
            {
                const Eigen::Vector3d fitted_center(fit.cx, fit.cy, ekf_.state()(4));
                ekf_.shiftCenter(fitted_center - ekf_.state().head<3>().eval());
                applySlowRadius(fit.radius);
                refit_center_ = Eigen::Vector2d(fit.cx, fit.cy);
                refit_center_valid_ = true;
                ++diagnostics_.body_refits;
                (void)count_now;
            }
        }
    }
    // 多秒窗口联合批处理：把窗口内的板心一次性解成 (圆心, R, ω, φ)，
    // 直接把状态钉到这条轨迹上。逐帧位置更新做不到这一点：单板单帧欠定，
    // 会让相位和车心互相顶，整段下来车心漂 0.4 m。
    // 窗口转速：把最近一段时间里"同一颗板"的板心角度随时间做最小二乘。
    // 逐帧单步的 omega 量测在换板时是垃圾（实测 |ω| 中位 19 rad/s，p90 49），
    // 只保留了 22% 的量测；窗口内按位置连续性切段后，剩余段的角度是单调的。
    if (config_.rate_window_enabled && config_.rate_from_tracklet_enabled && ekf_.initialized() &&
        diagnostics_.phase == LockPhase::Tracking &&
        diagnostics_.detected_circle_samples >= 8)
    {
        const Eigen::Vector2d c(diagnostics_.detected_circle_center.x(),
                                diagnostics_.detected_circle_center.y());
        // 收集窗口内样本（按时间正序），按位置连续性切段。
        std::vector<const LockSample *> window;
        for (auto it = body_samples_.rbegin(); it != body_samples_.rend(); ++it)
        {
            if (timestamp_seconds - it->timestamp_s > config_.rate_window_s)
                continue;
            window.push_back(&(*it));
        }
        std::reverse(window.begin(), window.end());

        std::size_t best_begin = 0;
        std::size_t best_len = 0;
        std::size_t run_begin = 0;
        for (std::size_t i = 0; i < window.size(); ++i)
        {
            if (i > 0)
            {
                const Eigen::Vector2d a(window[i - 1]->position.x(), window[i - 1]->position.y());
                const Eigen::Vector2d b(window[i]->position.x(), window[i]->position.y());
                if ((b - a).norm() > config_.rate_segment_jump_m)
                {
                    if (i - run_begin > best_len)
                    {
                        best_len = i - run_begin;
                        best_begin = run_begin;
                    }
                    run_begin = i;
                }
            }
        }
        if (window.size() - run_begin > best_len)
        {
            best_len = window.size() - run_begin;
            best_begin = run_begin;
        }

        if (best_len >= 5)
        {
            // 角度关于窗口圆心；用增量展开，避免 4 板歧义。
            double theta = 0.0;
            double prev_angle = 0.0;
            std::vector<double> ts;
            std::vector<double> thetas;
            for (std::size_t k = best_begin; k < best_begin + best_len; ++k)
            {
                const Eigen::Vector2d p(window[k]->position.x(), window[k]->position.y());
                const double angle = std::atan2(p.y() - c.y(), p.x() - c.x());
                if (k == best_begin)
                {
                    theta = angle;
                }
                else
                {
                    theta += wrapToPi(angle - prev_angle);
                }
                prev_angle = angle;
                ts.push_back(window[k]->timestamp_s);
                thetas.push_back(theta);
            }
            const double t0 = ts.front();
            double stt = 0.0;
            double st = 0.0;
            double sth = 0.0;
            for (std::size_t k = 0; k < ts.size(); ++k)
            {
                const double t = ts[k] - t0;
                stt += t * t;
                st += t;
                sth += t * thetas[k];
            }
            const double n = static_cast<double>(ts.size());
            const double denom = n * stt - st * st;
            if (denom > 1e-9)
            {
                const double slope = (n * sth - st * std::accumulate(thetas.begin(), thetas.end(), 0.0)) /
                                     denom;
                double ss = 0.0;
                const double intercept =
                    (std::accumulate(thetas.begin(), thetas.end(), 0.0) - slope * st) / n;
                for (std::size_t k = 0; k < ts.size(); ++k)
                {
                    const double r = thetas[k] - (intercept + slope * (ts[k] - t0));
                    ss += r * r;
                }
                const double sigma_fit = std::sqrt(ss / n);
                const double span = ts.back() - ts.front();
                if (std::isfinite(slope) && std::fabs(slope) < 12.0 && span >= 0.12)
                {
                    // 拟合残差折成角速度不确定度；跨度越大越可信。
                    const double sigma_rate =
                        std::max(config_.rate_window_sigma_floor_rad_s, sigma_fit / std::max(span, 1e-3));
                    if (std::fabs(slope) >= config_.rate_window_min_rate_rad_s ||
                        std::fabs(ekf_.state()(7)) >= config_.rate_window_min_rate_rad_s)
                    {
                        if (ekf_.updateRate(slope, sigma_rate))
                        {
                            last_rate_update_s_ = timestamp_seconds;
                        }
                        diagnostics_.rate_measurements += 1;
                        static const bool win_dbg = std::getenv("RATE_DEBUG") != nullptr;
                        if (win_dbg)
                        {
                            std::fprintf(stderr,
                                         "[wrate] t=%.3f slope=%+.2f ekf=%+.2f n=%zu span=%.3f "
                                         "sigma=%.2f fit=%.3f\n",
                                         timestamp_seconds, slope, ekf_.state()(7), ts.size(), span,
                                         sigma_rate, sigma_fit);
                        }
                    }
                }
            }
        }
    }

    // 绿色（实测反推车体）的窗口圆：只写诊断，给俯视图用。
    // 单帧两块板的解会被 PnP 噪声带得抖 2-6 cm；同一窗口的板心其实落在一条
    // 半径稳定的圆上（残差 ~1 cm），所以显示层用窗口圆，单帧解只当参考。
    if (ekf_.initialized() && diagnostics_.phase == LockPhase::Tracking)
    {
        std::deque<LockSample> window;
        for (auto it = body_samples_.rbegin(); it != body_samples_.rend(); ++it)
        {
            if (timestamp_seconds - it->timestamp_s > config_.detected_circle_window_s)
            {
                continue;
            }
            window.push_front(*it);
        }
        if (static_cast<int>(window.size()) >= config_.detected_circle_min_samples)
        {
            const CircleFit fit = fitArmorCircle(window);
            if (fit.ok && fit.radius >= config_.radius_min_m &&
                fit.radius <= config_.radius_max_m)
            {
                diagnostics_.detected_circle_valid = true;
                diagnostics_.detected_circle_center = Eigen::Vector2d(fit.cx, fit.cy);
                diagnostics_.detected_circle_radius = fit.radius;
                diagnostics_.detected_circle_residual = fit.residual;
                diagnostics_.detected_circle_samples = static_cast<int>(window.size());
            }
        }
    }

    // 每帧用窗口板心圆覆盖 (c, R)：不碰相位与 ω
    if (config_.body_circle_enabled && ekf_.initialized() &&
        diagnostics_.phase == LockPhase::Tracking)
    {
        std::deque<LockSample> window;
        for (auto it = body_samples_.rbegin(); it != body_samples_.rend(); ++it)
        {
            if (timestamp_seconds - it->timestamp_s > config_.body_circle_window_s)
            {
                break;
            }
            window.push_front(*it);
        }
        if (static_cast<int>(window.size()) >= config_.body_circle_min_samples)
        {
            const CircleFit fit = fitArmorCircle(window);
            if (fit.ok && fit.residual <= config_.body_circle_max_residual_m &&
                fit.radius > config_.radius_min_m && fit.radius < config_.radius_max_m)
            {
                if (std::getenv("CIRCLE_DEBUG"))
                {
                    std::fprintf(stderr, "[circ] f=%.2f n=%d c=(%.3f,%.3f) R=%.3f res=%.4f\n",
                                 timestamp_seconds, static_cast<int>(window.size()), fit.cx, fit.cy,
                                 fit.radius, fit.residual);
                }
                ekf_.shiftCenter(Eigen::Vector3d(fit.cx - ekf_.state()(0),
                                                 fit.cy - ekf_.state()(2), 0.0));
                applySlowRadius(fit.radius);
                ++diagnostics_.body_circles;
            }
        }
    }
    // 同板轨迹段联合解 (c, ω)，见 TrackerConfig::tracklet_rate_enabled
    if (config_.tracklet_rate_enabled && ekf_.initialized() &&
        diagnostics_.phase == LockPhase::Tracking &&
        static_cast<int>(tracklet_samples_.size()) >= config_.tracklet_min_samples &&
        timestamp_seconds - tracklet_samples_.front().timestamp_s >= config_.tracklet_min_duration_s)
    {
        const double radius_now = ekf_.state()(8);
        CircleFit init = fitArmorCircle(tracklet_samples_);
        CircleFit centre_fit = fitCenterFixedRadius(tracklet_samples_, radius_now, init);
        double rate = 0.0;
        double arc = 0.0;
        double resid = 1.0e9;
        if (centre_fit.ok && estimateRateAlongTracklet(tracklet_samples_, centre_fit, &rate, &arc, &resid) &&
            std::fabs(arc) >= config_.tracklet_min_arc_rad && resid <= config_.tracklet_max_residual_m &&
            std::fabs(rate) < 12.0)
        {
            if (std::getenv("TRACKLET_DEBUG") != nullptr)
            {
                std::fprintf(stderr, "[tl] f=%.2f n=%d c=(%.3f,%.3f) R=%.3f om=%+.2f arc=%+.2f res=%.4f\n",
                             timestamp_seconds, static_cast<int>(tracklet_samples_.size()),
                             centre_fit.cx, centre_fit.cy, radius_now, rate, arc, resid);
            }
            // 只喂 ω，不动车心：轨迹段的车心精度远不如多秒窗口批处理，
            // 实测强行 shiftCenter 会把状态踹偏（v4 90.0 -> 98.4 px）。
            ekf_.setRate(rate);
            ++diagnostics_.tracklet_rates;
            tracklet_samples_.clear();
        }
    }
    if (config_.batch_fit_enabled && ekf_.initialized() &&
        diagnostics_.phase == LockPhase::Tracking)
    {
        ++batch_fit_countdown_;
        if (batch_fit_countdown_ >= config_.batch_fit_every)
        {
            batch_fit_countdown_ = 0;
            std::deque<LockSample> window;
            for (auto it = body_samples_.rbegin(); it != body_samples_.rend(); ++it)
            {
                if (timestamp_seconds - it->timestamp_s > config_.batch_fit_window_s)
                {
                    break;
                }
                window.push_front(*it);
            }
            if (static_cast<int>(window.size()) >= config_.batch_fit_min_samples)
            {
                const BodyBatchFit fit = fitBodyBatch(window, std::max(config_.ekf.plate_count, 1),
                                                 ekf_.state()(7), config_.batch_fit_min_coverage_rad,
                                                 config_.batch_fit_fold_phase, config_.batch_fit_alias_gate);
                const double radius_now = ekf_.state()(8);
                if (std::getenv("BATCH_DEBUG") != nullptr)
                {
                    std::fprintf(stderr,
                                 "[batch] f=%.2f ok=%d n=%d c=(%.3f,%.3f) R=%.3f om=%+.2f "
                                 "cres=%.4f pres=%.3f cov=%.2f\n",
                                 timestamp_seconds, fit.ok ? 1 : 0, fit.used, fit.center.x(),
                                 fit.center.y(), fit.radius, fit.rate_rad_s, fit.circle_residual_m,
                                 fit.phase_residual_rad, fit.coverage_rad);
                }
                if (fit.ok && fit.circle_residual_m <= config_.batch_fit_max_circle_residual_m &&
                    fit.phase_residual_rad <= config_.batch_fit_max_phase_residual_rad &&
                    // 自适应：合成数据（静止或干净自转）的相位残差很小，
                    // EKF 自己就够用，批处理反而会把 3cm 的射击精度拉到 4cm；
                    // 只有观测真的不干净（v4 这种手写板的抖动）才接管。
                    fit.phase_residual_rad >= config_.batch_fit_min_phase_residual_rad &&
                    fit.radius > config_.radius_min_m && fit.radius < config_.radius_max_m &&
                    std::fabs(fit.rate_rad_s) < 12.0 &&
                    std::fabs(fit.radius - radius_now) < 0.20 &&
                    (fit.center - Eigen::Vector2d(ekf_.state()(0), ekf_.state()(2))).norm() < 0.60)
                {
                    ekf_.shiftCenter(Eigen::Vector3d(fit.center.x() - ekf_.state()(0),
                                                     fit.center.y() - ekf_.state()(2), 0.0));
                    applySlowRadius(fit.radius);
                    if (std::getenv("BATCH_NO_RATE") == nullptr)
                    {
                        ekf_.setRate(fit.rate_rad_s);
                    }
                    if (std::getenv("BATCH_NO_PHASE") == nullptr)
                    {
                        const double dt = timestamp_seconds - window.front().timestamp_s;
                        const double phase_now = fit.phase_rad + fit.rate_rad_s * dt;
                        // 只转"不足一块板"的那部分：相位整体差整数个 2pi/N 时，
                        // 画出来的四块板完全一样，但槽位标号会整体错位，
                        // 射击点会跳到对面的板上（合成 G 就是这么挂的）。
                        const double period = 2.0 * kPi / static_cast<double>(std::max(config_.ekf.plate_count, 1));
                        double delta = wrapAngle(phase_now - ekf_.state()(6));
                        if (config_.batch_fit_slot_phase)
                        {
                            delta -= period * std::round(delta / period);
                        }
                        ekf_.rotatePhase(delta);
                    }
                    ++diagnostics_.batch_fits;
                }
            }
        }
    }
    diagnostics_.updates_this_frame = updates;
    last_updates_ = updates;
    if (updates > 0)
    {
        ++consecutive_updates_;
        missed_frames_ = 0;
        last_accept_s_ = timestamp_seconds;
        has_accept_ = true;
        diagnostics_.accepted_updates += static_cast<std::uint64_t>(updates);
    }
    else
    {
        ++missed_frames_;
        // 局部预测：已经连续多帧没有任何观测支持时，不要再让角速度自由滑行——
        // 蓝圈会继续转，而绿圈早就不在了（用户看到的"绿色走了蓝色还在转"）。
        if (missed_frames_ > config_.coast_decay_after_frames && ekf_.initialized())
        {
            const double damped = ekf_.state()(7) * config_.coast_omega_decay;
            ekf_.setRate(std::fabs(damped) < 1e-3 ? 0.0 : damped);
        }
        // 连续几帧都被拒、而当前帧恰好露出两块板：用双板几何重定中心与半径。
        // 这是"锁定后 0 次更新"的唯一出路——单板初值偏 R 量级，靠更新拉不回来。
        if (missed_frames_ >= 5 && reinitializeFromPair(world))
        {
            missed_frames_ = 0;
            has_accept_ = true;
            last_accept_s_ = timestamp_seconds;
        }
    }
    diagnostics_.consecutive_updates = consecutive_updates_;
    diagnostics_.missed_frames = missed_frames_;
    diagnostics_.observation_noise_m = observation_noise_m_;

    if (!has_accept_ || timestamp_seconds - last_accept_s_ > config_.timeout_s)
    {
        dropTarget("timeout");
        return PredictionResult{};
    }
    // 这个检查只为"声明了不转、其实在转"准备：转速明显不为 0 时它没有意义
    // （旋转目标本来就会让板面法向持续扫动），反而会把自己刚锁上的旋转目标打掉。
    if (config_.contradiction_check && std::fabs(ekf_.state()(7)) < 3.0 &&
        declarationContradicted(world, timestamp_seconds))
    {
        dropTarget("declaration_contradicted");
        return PredictionResult{};
    }
    return buildResult(gimbal, timestamp_seconds);
}

void RotatingArmorTracker::noteObservationForNoise(const Eigen::Vector3d &position,
                                                   double timestamp_s)
{
    if (!config_.adaptive_observation_noise)
    {
        return;
    }
    noise_samples_.push_back(NoiseSample{position, timestamp_s});
    const std::size_t keep = static_cast<std::size_t>(std::max(3, config_.adaptive_noise_window));
    while (noise_samples_.size() > keep)
    {
        noise_samples_.pop_front();
    }
    // 二阶差分：p(t) - 2p(t-dt) + p(t-2dt)。匀速/匀速自转下它≈0，
    // 只剩检测抖动（自转的向心项在 166fps 下 <1 mm）。换板造成的大跳变
    // 用中位数压掉。
    std::vector<double> squared;
    squared.reserve(noise_samples_.size());
    for (std::size_t i = 2; i < noise_samples_.size(); ++i)
    {
        const NoiseSample &a = noise_samples_[i - 2];
        const NoiseSample &b = noise_samples_[i - 1];
        const NoiseSample &c = noise_samples_[i];
        const double dt1 = b.timestamp_s - a.timestamp_s;
        const double dt2 = c.timestamp_s - b.timestamp_s;
        if (!(dt1 > 1e-4) || !(dt2 > 1e-4) || dt1 > 0.05 || dt2 > 0.05)
        {
            continue;
        }
        const Eigen::Vector3d d2 = c.position - 2.0 * b.position + a.position;
        squared.push_back(d2.squaredNorm() / 6.0);
    }
    if (squared.size() < 5)
    {
        return;
    }
    std::sort(squared.begin(), squared.end());
    const double median = squared[squared.size() / 2];
    const double sigma = std::sqrt(std::max(0.0, median));
    observation_noise_m_ =
        std::min(std::max(sigma, config_.adaptive_noise_min_sigma_m),
                 config_.adaptive_noise_max_sigma_m);
}

double RotatingArmorTracker::observationSigma(double detection_sigma_m) const
{
    if (!config_.adaptive_observation_noise)
    {
        return std::hypot(detection_sigma_m, config_.association_sigma_floor_m);
    }
    return std::hypot(detection_sigma_m, observation_noise_m_);
}

double RotatingArmorTracker::staleWindowSeconds() const
{
    double window = config_.stale_after_s;
    if (!recent_dt_.empty())
    {
        std::deque<double> sorted(recent_dt_);
        std::sort(sorted.begin(), sorted.end());
        const double dt_median = sorted[sorted.size() / 2];
        const double by_frames =
            std::max(config_.stale_min_s, config_.stale_min_frames * dt_median);
        window = std::min(window, by_frames);
    }
    return window;
}

} // namespace rotating_armor
