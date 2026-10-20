#include "RotatingArmorTracker.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
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
    config.process.angular_accel = 0.01 * scale;
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
        const double scale = clampd(1.0 + pose.reprojection_error / 6.0, 1.0, 3.0);
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
            const double candidate_radius = (pi2 - pj2).norm() / ndiff.norm();
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
    double size_prior = config_.radius_prior_small_m;
    if (config_.radius_prior_from_size && lock_samples_.back().size == ArmorSize::Large)
    {
        size_prior = config_.radius_prior_large_m;
    }
    size_prior = clampd(size_prior, config_.radius_min_m, config_.radius_max_m);
    CircleFit circle = fitArmorCircle(lock_samples_);
    bool used_fixed_radius = false;
    if (circle.ok && std::fabs(circle.radius - size_prior) > 0.3 * size_prior)
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
                slope_error = 0.0; // 由弦长直接解，没有"每步散布"这个自由度
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
                std::fabs(fitted) <= config_.radius_max_m && fitted_flipped == flipped)
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
        double size_prior = radius;
        if (config_.radius_prior_from_size)
        {
            size_prior = (last.size == ArmorSize::Large) ? config_.radius_prior_large_m
                                                         : config_.radius_prior_small_m;
        }
        const double tolerance = 0.3 * size_prior;
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
                }
            }
        }
        if (chosen == nullptr)
        {
            // 这块板这一帧没被看到（被挡/漏检）：窗口保留，等下一帧；
            // 间断太久由下面的 rate_max_gap_s 清空。
            if (timestamp_s - rate_samples_.back().timestamp_s > config_.rate_max_gap_s)
            {
                rate_samples_.clear();
                last_rate_update_s_ = -1.0e9;
            }
            return;
        }
    }
    if (chosen == nullptr)
    {
        diagnostics_.rate_window_samples = static_cast<int>(rate_samples_.size());
        return;
    }
    if (!rate_samples_.empty() &&
        timestamp_s - rate_samples_.back().timestamp_s > config_.rate_max_gap_s)
    {
        rate_samples_.clear();
        last_rate_update_s_ = -1.0e9;
    }

    // 2) 解缠：折叠掉换板造成的 2pi/N 整数倍。
    double unwrapped = chosen->yaw;
    if (!rate_samples_.empty())
    {
        const double delta =
            foldPeriod(chosen->yaw - rate_samples_.back().unwrapped_yaw, period);
        // 单帧 0.6 rad 在 166 fps 下相当于 100 rad/s，物理上不可能；出现就说明
        // 续接跳到了别的板/别的车，窗口重起，别把跳变当成旋转。
        if (std::fabs(delta) > 0.60)
        {
            rate_samples_.clear();
        }
        else
        {
            unwrapped = rate_samples_.back().unwrapped_yaw + delta;
        }
    }
    rate_samples_.push_back({timestamp_s, unwrapped, chosen->position,
                             Eigen::Vector2d(ekf_.state()(0), ekf_.state()(2))});
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
    double bearing_unwrapped = 0.0;
    for (int i = 0; i < count; ++i)
    {
        const RateSample &sample = rate_samples_[static_cast<std::size_t>(i)];
        times[static_cast<std::size_t>(i)] = sample.timestamp_s;
        yaws[static_cast<std::size_t>(i)] = sample.unwrapped_yaw;
        const double dx = sample.position.x() - sample.center.x();
        const double dy = sample.position.y() - sample.center.y();
        const double bearing = std::atan2(-dy, -dx);
        if (i == 0)
        {
            bearing_unwrapped = bearing;
        }
        else
        {
            bearing_unwrapped += foldPeriod(bearing - bearing_unwrapped, period);
        }
        bearings[static_cast<std::size_t>(i)] = bearing_unwrapped;
    }
    double mad = 0.0;
    const double slope = theilSenSlope(times, yaws, min_pair_dt, &mad);
    if (!std::isfinite(slope) || !std::isfinite(mad))
    {
        return;
    }
    const double sigma = std::max(1.4826 * mad, config_.rate_sigma_floor_rad_s);
    if (sigma > config_.rate_max_sigma_rad_s)
    {
        return;
    }

    // 3b) 第二路证据：板相对车心的方位角斜率。
    //
    //     法向在实拍小目标上会"整段地"漂（v4 的板心在 ±45° 折叠边界附近来回
    //     翻，解缠后是每帧 +5° 的锯齿，Theil-Sen 也救不了——实测斜率 -10 rad/s，
    //     而车其实没转）。位置没有这个问题：目标平移时板与车心一起动，方位角
    //     不变。两路证据打架时以"说不清"处理：不更新，让衰减项把 omega 停在 0。
    const double bearing_slope = theilSenSlope(times, bearings, min_pair_dt, nullptr);
    diagnostics_.rate_bearing_rad_s = std::isfinite(bearing_slope) ? bearing_slope : 0.0;
    const double tolerance = std::max(config_.rate_agreement_rad_s,
                                      0.5 * std::fabs(slope));
    if (std::isfinite(bearing_slope) && std::fabs(slope - bearing_slope) > tolerance)
    {
        return;
    }

    if (!ekf_.updateYawRate(slope, sigma))
    {
        return;
    }
    diagnostics_.rate_obs_used = true;
    diagnostics_.rate_obs_rad_s = slope;
    diagnostics_.rate_obs_sigma_rad_s = sigma;
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
void RotatingArmorTracker::updateRadiusFromPairs(const std::vector<WorldObservation> &world)
{
    if (!config_.pair_radius_update_enabled || target_id_ == 0 || !ekf_.initialized())
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
            const double dpsi = foldPeriod(world[j].yaw - world[i].yaw, 2.0 * kPi);
            const double half = 0.5 * std::fabs(dpsi);
            if (half < 0.35) // 同一块板被重复检出：弦长除噪会炸
            {
                continue;
            }
            const double chord = (world[j].position - world[i].position).norm();
            if (!std::isfinite(chord) || chord < config_.radius_update_min_chord_m)
            {
                continue;
            }
            const double radius = chord / (2.0 * std::sin(half));
            if (!std::isfinite(radius) || radius < config_.radius_min_m ||
                radius > config_.radius_max_m)
            {
                continue;
            }
            ekf_.updateRadius(radius, config_.radius_update_sigma_m);
            return;
        }
    }
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
    diagnostics_.rejected_this_frame = 0;
    diagnostics_.best_gate_distance_m = std::numeric_limits<double>::infinity();
    diagnostics_.best_gate_nis = std::numeric_limits<double>::infinity();

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
            for (int slot = 0; slot < plate_count; ++slot)
            {
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
                if (distance < best_distance)
                {
                    best_distance = distance;
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
        ArmorObservation observation;
        if (!smoothObservation(chosen, best_slot, timestamp_s, observation))
        {
            observation.position = chosen.position;
            observation.sigma_position_m = observationSigma(chosen.sigma_position_m);
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
                ekf_.rotatePhase(delta_theta);
                const Eigen::Vector3d plate_after =
                    platePosition(ekf_.state(), best_slot, count);
                ekf_.shiftCenter(chosen.position - plate_after);
            }
        }
        if (!ekf_.update(observation, best_slot))
        {
            break;
        }
        noteObservationForNoise(chosen.position, timestamp_s);
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
            if (phase_from_yaw)
            {
                // 反锚：先记下这块板在模型里的位置，改完 yaw 再把中心平移回去，
                // 让板留在观测到的位置。单板可见时 (center, yaw) 是规范自由度，
                // 不反锚就是"yaw 更新把板挪走、位置更新再拉回来"互相打架。
                const Eigen::Vector3d plate_before =
                    rotating_armor::platePosition(ekf_.state(), best_slot,
                                                  ekf_.config().plate_count);
                if (ekf_.updatePhase(chosen.yaw, config_.phase_update_sigma_rad))
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
    if (config_.pair_center_update_enabled && accepted >= 2)
    {
        int index[2] = {-1, -1};
        int found = 0;
        for (std::size_t i = 0; i < world.size() && found < 2; ++i)
        {
            if (!used[i] || world[i].target_id != target_id_ ||
                slot_of_observation[i] < 0)
            {
                continue;
            }
            index[found] = static_cast<int>(i);
            ++found;
        }
        const double radius = ekf_.state()(8);
        const int slots[2] = {found == 2 ? slot_of_observation[index[0]] : -1,
                              found == 2 ? slot_of_observation[index[1]] : -1};
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
                if (std::fabs(delta) > 0.5 * period)
                {
                    delta -= std::copysign(period, delta);
                }
                ekf_.rotatePhase(delta);
                // 相位定了以后中心由任一块板反推：c = p + R*u(theta + s*period)
                const double angle1 =
                    ekf_.state()(6) + static_cast<double>(slots[0]) * period;
                const Eigen::Vector3d center_measured =
                    p1 + radius * Eigen::Vector3d(std::cos(angle1), std::sin(angle1), 0.0);
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

    // 新鲜度按"距上次被接收的更新多少秒"判，不按丢帧数：相机帧率变化时
    // 帧数门限的含义会漂移（166fps 下 2 帧只有 12ms，实拍里会长期卡在 Unstable）。
    const double stale_s = has_accept_ ? (timestamp_s - last_accept_s_) : 1e9;
    const double stale_window_s = staleWindowSeconds();
    const bool fresh_enough = has_accept_ && stale_s <= stale_window_s;
    diagnostics_.stability = stable_reason(consecutive_updates_ >= config_.stable_updates,
                                             ekf_.phase_resolved(), fresh_enough,
                                             timestamp_s - last_accept_s_);
    const bool stable = consecutive_updates_ >= config_.stable_updates &&
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
    RotatingArmorEkf probe = ekf_;
    if (!probe.predict(bias + latency))
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
            best_slot = aim_slot_;
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
            if (error < best_error)
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
            const double horizon = bias + latency + (timestamp_s - seen);
            const Eigen::Vector3d center(probe.state()(0), probe.state()(2), probe.state()(4));
            const double omega = probe.state()(7);
            if (config_.aim_observation_extrapolate && horizon > 0.0 && std::isfinite(omega) &&
                std::fabs(omega) > 1.0e-3)
            {
                const Eigen::Vector3d relative = plate - center;
                const double angle = omega * horizon;
                const double c = std::cos(angle);
                const double s_ = std::sin(angle);
                plate = center + Eigen::Vector3d(c * relative.x() - s_ * relative.y(),
                                                 s_ * relative.x() + c * relative.y(),
                                                 relative.z());
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
    const bool aligned = yaw_error <= config_.fire_angle_error_rad &&
                         pitch_error <= config_.fire_angle_error_rad;
    const bool fresh = has_accept_ && missed_frames_ == 0 &&
                       (timestamp_s - last_accept_s_ <= config_.fire_max_age_s);
    result.fire_allowed = stable && aligned && fresh &&
                          distance >= config_.min_distance_m &&
                          distance <= config_.max_distance_m;

    if (!predictionFinite(result))
    {
        return PredictionResult{};
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
    if (locked_id_present)
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
        ekf_.updateYawRate(0.0, config_.rate_decay_sigma_rad_s);
    }
    updateRadiusFromPairs(world);
    const int updates = associateAndUpdate(world, timestamp_seconds);
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
