#include "RotatingArmorTracker.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace rotating_armor
{
namespace
{

constexpr double kPi = 3.141592653589793238462643383279502884;
// 角速度用前向差分算，步长 1ms：比手推 d(atan2)/dt 不容易错，代价可忽略。
constexpr double kVelocityDt = 1.0e-3;

double clampd(double value, double low, double high)
{
    return std::min(std::max(value, low), high);
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

void RotatingArmorTracker::reset()
{
    ekf_.reset();
    diagnostics_ = TrackerDiagnostics{};
    has_timestamp_ = false;
    last_timestamp_s_ = 0.0;
    state_timestamp_s_ = 0.0;
    initialized_ = false;
    target_id_ = 0;
    aim_slot_ = -1;
    last_accept_s_ = 0.0;
    has_accept_ = false;
    consecutive_updates_ = 0;
    missed_frames_ = 0;
    other_id_streak_ = 0;
    contradiction_streak_ = 0;
    relock_count_ = 0;
    lock_samples_.clear();
    for (auto &history : normal_history_per_slot_)
    {
        history.clear();
    }
    locked_radius_m_ = 0.0;
    locked_spin_rad_s_ = 0.0;
    lock_radius_from_data_ = false;
    lock_phase_flipped_ = false;
}

void RotatingArmorTracker::dropTarget(const char *reason)
{
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

bool RotatingArmorTracker::tryLock(const std::vector<WorldObservation> &world,
                                   double timestamp_s, const GimbalState &gimbal)
{
    (void)gimbal;

    // 1) 攒样本：只有"带板面法向 + 有编号 + 与上一帧连续"的观测才进缓冲区。
    //    连续性检查是必须的：同一辆车其它装甲板的法向相差 2*pi*k/N，
    //    混进来会让拟合出的角速度完全失真。
    // 连续性优先于"哪块更准"：同一辆车的不同装甲板法向相差 2*pi*k/N，
    // 一旦混进一个不连续的样本，后面解缠出来的角速度就完全失真。
    const WorldObservation *pick = nullptr;
    const bool have_last = !lock_samples_.empty();
    const double last_yaw = have_last ? lock_samples_.back().yaw : 0.0;
    for (const WorldObservation &observation : world)
    {
        if (observation.target_id == 0 || !observation.has_yaw)
        {
            continue;
        }
        if (have_last && std::fabs(wrapToPi(observation.yaw - last_yaw)) > 0.45)
        {
            continue;
        }
        if (pick == nullptr || observation.sigma_position_m < pick->sigma_position_m)
        {
            pick = &observation;
        }
    }
    if (pick == nullptr && have_last)
    {
        // 本帧没有和上一帧连续的板：这一帧不加样本，等缓冲区自然过期后重开一条链。
        diagnostics_.lock_samples = static_cast<int>(lock_samples_.size());
        diagnostics_.last_reject = "lock:waiting_continuity";
        return false;
    }
    if (pick != nullptr)
    {
        lock_samples_.push_back(
            LockSample{timestamp_s, pick->position, pick->yaw, pick->target_id});
    }
    while (!lock_samples_.empty() &&
           timestamp_s - lock_samples_.front().timestamp_s > config_.lock_max_age_s)
    {
        lock_samples_.pop_front();
    }
    diagnostics_.lock_samples = static_cast<int>(lock_samples_.size());
    if (static_cast<int>(lock_samples_.size()) < std::max(config_.lock_min_samples, 6))
    {
        diagnostics_.last_reject = "lock:waiting_samples";
        return false;
    }
    const double span =
        lock_samples_.back().timestamp_s - lock_samples_.front().timestamp_s;
    if (span < config_.lock_min_span_s)
    {
        diagnostics_.last_reject = "lock:waiting_span";
        return false;
    }

    // 2) 板面法向 yaw 的序列（解缠）+ 最小二乘斜率 = 角速度。
    const std::size_t count = lock_samples_.size();
    const double t_last = lock_samples_.back().timestamp_s;
    std::vector<double> times(count, 0.0);
    std::vector<double> phases(count, 0.0);
    for (std::size_t i = 0; i < count; ++i)
    {
        times[i] = lock_samples_[i].timestamp_s - t_last;
        phases[i] = (i == 0)
                        ? lock_samples_[i].yaw
                        : phases[i - 1] + wrapToPi(lock_samples_[i].yaw -
                                                   lock_samples_[i - 1].yaw);
    }
    const LinearFit fit = fitLinear(times, phases);
    if (!fit.ok)
    {
        diagnostics_.last_reject = "lock:fit_failed";
        return false;
    }
    const double omega = fit.slope;
    double phase_spread = 0.0;
    for (std::size_t i = 0; i < count; ++i)
    {
        phase_spread = std::max(phase_spread, std::fabs(phases[i] - fit.intercept));
    }

    // 3) 半径与"车体在板的哪一侧"：解线性最小二乘
    //      p_k = c + v*(t_k - t_last) - R*[cos(phi_k), sin(phi_k)]
    //    未知量 (cx, cy, vx, vy, R)。R 的符号直接回答法向是朝外还是朝内。
    //    目标不转（相位几乎不扫）时 R 与中心不可分辨，退回"车体在板背后"的
    //    几何先验 + 配置半径。
    const LockSample last = lock_samples_.back();
    double radius = clampd(config_.radius_prior_m, config_.radius_min_m, config_.radius_max_m);
    bool radius_from_data = false;

    // 相机只能看到装甲板正面，所以"车体中心在板背后"是可靠几何先验：
    // 板面外法向 yaw 应背离视线方向，即模型相位 = yaw + pi。
    const double ray = std::atan2(last.position.y(), last.position.x());
    const double inside = std::fabs(wrapToPi(wrapToPi(last.yaw) - ray));
    const double outside = std::fabs(wrapToPi(wrapToPi(last.yaw + kPi) - ray));
    bool flipped = outside < inside;

    // 只有相位确实扫过一段（>=30度）且样本足够多，才敢让数据决定半径：
    // 相位几乎不扫时 R 与车体中心在最小二乘里几乎共线，解出来的 R 符号会乱跳，
    // 一旦把相位判断翻错，车体中心就会落到相机这一侧（偏差 2R）。
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

    const double phase_last = wrapToPi(last.yaw + (flipped ? kPi : 0.0));
    const bool is_static =
        std::fabs(omega) <= std::max(config_.static_rate_rad_s, 2.5 * fit.slope_error);

    EkfInitPrior prior;
    prior.radius_m = radius;
    prior.plate_slot = 0; // 槽位基准任意：相对相位才是物理量
    prior.spin = is_static ? SpinPriorKind::Static : SpinPriorKind::Spinning;
    prior.yaw_rate_prior_rad_s = is_static ? 0.0 : omega;
    prior.position_sigma_m = 0.06;
    prior.velocity_sigma_m_s = 1.0;
    prior.yaw_sigma_rad = std::max(config_.normal_sigma_rad, 0.15);
    prior.yaw_rate_sigma_rad_s = is_static ? 0.0 : std::max(0.5, 2.0 * fit.slope_error);
    prior.radius_sigma_m = config_.radius_sigma_m;

    ArmorObservation observation;
    observation.position = last.position;
    observation.sigma_position_m = config_.position_sigma_m;
    observation.has_world_yaw = true;
    // initialize() 的约定：world_yaw_rad 是板面"外法向"（从车体中心指向板）。
    observation.world_yaw_rad = wrapToPi(phase_last + kPi);
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
    locked_spin_rad_s_ = is_static ? 0.0 : omega;
    lock_radius_from_data_ = radius_from_data;
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
    diagnostics_.rejected_this_frame = 0;

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
            ArmorObservation observation;
            observation.position = world[i].position;
            observation.sigma_position_m = world[i].sigma_position_m;
            for (int slot = 0; slot < plate_count; ++slot)
            {
                const Eigen::Vector3d predicted = platePosition(ekf_.state(), slot, plate_count);
                const double distance = (world[i].position - predicted).norm();
                if (!(distance <= config_.position_gate_m))
                {
                    continue;
                }
                double nis = 0.0;
                if (!ekf_.innovationNis(observation, slot, nis))
                {
                    continue;
                }
                if (nis > config_.nis_gate)
                {
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
        observation.position = chosen.position;
        observation.sigma_position_m = chosen.sigma_position_m;
        if (!ekf_.update(observation, best_slot))
        {
            break;
        }
        used[static_cast<std::size_t>(best_observation)] = true;
        ++accepted;
        diagnostics_.last_nis = best_nis;
        if (chosen.has_yaw)
        {
            pushNormalHistory(best_slot, timestamp_s, chosen.yaw);
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

    const bool stable = consecutive_updates_ >= config_.stable_updates &&
                        ekf_.phase_resolved() &&
                        missed_frames_ < config_.unstable_after_missed && has_accept_ &&
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

    // 预测到"子弹该对准的时刻"：控制器下发的预测时间偏置（秒）。
    const double bias = std::isfinite(gimbal.prediction_bias_s)
                            ? clampd(static_cast<double>(gimbal.prediction_bias_s), 0.0,
                                     config_.max_bias_s)
                            : 0.0;
    RotatingArmorEkf probe = ekf_;
    if (!probe.predict(bias))
    {
        return PredictionResult{};
    }
    const int plate_count = config_.ekf.plate_count;
    if (plate_count <= 0)
    {
        return PredictionResult{};
    }

    // 选瞄准的装甲板：预测时刻上"离当前云台指向最近"的那块，换板带迟滞。
    int best_slot = -1;
    double best_error = std::numeric_limits<double>::infinity();
    for (int slot = 0; slot < plate_count; ++slot)
    {
        const Eigen::Vector3d plate = platePosition(probe.state(), slot, plate_count);
        const double error =
            std::max(std::fabs(wrapToPi(vectorYaw(plate) - gimbal.yaw)),
                     std::fabs(vectorPitch(plate) - gimbal.pitch));
        const double biased = (slot == aim_slot_) ? error - config_.switch_margin_rad : error;
        if (biased < best_error)
        {
            best_error = biased;
            best_slot = slot;
        }
    }
    if (best_slot < 0)
    {
        return PredictionResult{};
    }
    aim_slot_ = best_slot;
    diagnostics_.aim_slot = best_slot;

    const Eigen::Vector3d plate = platePosition(probe.state(), best_slot, plate_count);
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
                                            double timestamp_seconds)
{
    diagnostics_.updates_this_frame = 0;
    diagnostics_.rejected_this_frame = 0;
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

    // 本帧观测里出现最多的非零编号：用于识别"换了目标车"。
    int frame_id = 0;
    {
        std::map<int, int> counts;
        for (const WorldObservation &observation : world)
        {
            if (observation.target_id != 0)
            {
                ++counts[observation.target_id];
            }
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

    if (!ekf_.initialized())
    {
        diagnostics_.phase = LockPhase::Collecting;
        if (tryLock(world, timestamp_seconds, gimbal))
        {
            return buildResult(gimbal, timestamp_seconds);
        }
        return PredictionResult{};
    }
    diagnostics_.phase = LockPhase::Tracking;

    // 目标切换：连续两帧都是别的编号才认，避免编号抖动导致频繁重锁。
    if (frame_id != 0 && frame_id != target_id_)
    {
        if (++other_id_streak_ >= 2)
        {
            dropTarget("target_switch");
            return PredictionResult{};
        }
    }
    else
    {
        other_id_streak_ = 0;
    }

    const double dt = timestamp_seconds - state_timestamp_s_;
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

    const int updates = associateAndUpdate(world, timestamp_seconds);
    diagnostics_.updates_this_frame = updates;
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
    }
    diagnostics_.consecutive_updates = consecutive_updates_;
    diagnostics_.missed_frames = missed_frames_;

    if (!has_accept_ || timestamp_seconds - last_accept_s_ > config_.timeout_s)
    {
        dropTarget("timeout");
        return PredictionResult{};
    }
    if (declarationContradicted(world, timestamp_seconds))
    {
        dropTarget("declaration_contradicted");
        return PredictionResult{};
    }
    return buildResult(gimbal, timestamp_seconds);
}

} // namespace rotating_armor
