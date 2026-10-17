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
    has_pair_relock_ = false;
    last_pair_relock_s_ = 0.0;
    timestamp_s_ = 0.0;
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

RotatingArmorTracker::CircleFit
RotatingArmorTracker::fitArmorCircle(const std::deque<LockSample> &samples)
{
    CircleFit fit;
    if (samples.size() < 4)
    {
        return fit;
    }
    // 初值：位置均值当圆心、平均距离当半径。
    double mx = 0.0;
    double my = 0.0;
    for (const LockSample &sample : samples)
    {
        mx += sample.position.x();
        my += sample.position.y();
    }
    mx /= static_cast<double>(samples.size());
    my /= static_cast<double>(samples.size());
    double r = 0.0;
    for (const LockSample &sample : samples)
    {
        r += std::hypot(sample.position.x() - mx, sample.position.y() - my);
    }
    r /= static_cast<double>(samples.size());
    // Gauss-Newton：残差 = |p - c| - R，未知量 (cx, cy, R)。
    for (int iter = 0; iter < 15; ++iter)
    {
        Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
        Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
        for (const LockSample &sample : samples)
        {
            const double dx = sample.position.x() - mx;
            const double dy = sample.position.y() - my;
            const double d = std::hypot(dx, dy);
            if (d < 1.0e-6)
            {
                continue;
            }
            const double residual = d - r;
            const Eigen::Vector3d jacobian(-dx / d, -dy / d, -1.0);
            normal += jacobian * jacobian.transpose();
            rhs += jacobian * (-residual);
        }
        const Eigen::Vector3d step = normal.ldlt().solve(rhs);
        if (!step.allFinite())
        {
            return fit;
        }
        mx += step(0);
        my += step(1);
        r += step(2);
        if (step.norm() < 1.0e-9)
        {
            break;
        }
    }
    double sum = 0.0;
    for (const LockSample &sample : samples)
    {
        const double residual =
            std::hypot(sample.position.x() - mx, sample.position.y() - my) - r;
        sum += residual * residual;
    }
    fit.cx = mx;
    fit.cy = my;
    fit.radius = r;
    fit.residual = std::sqrt(sum / static_cast<double>(samples.size()));
    fit.ok = std::isfinite(fit.cx) && std::isfinite(fit.cy) &&
             std::isfinite(fit.radius) && fit.radius > 0.0;
    return fit;
}

bool RotatingArmorTracker::tryLock(const std::vector<WorldObservation> &world,
                                   double timestamp_s, const GimbalState &gimbal)
{
    (void)gimbal;

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
        if (pick == nullptr ||
            (pick_handoff && !handoff) ||
            (!pick_handoff && !handoff && std::fabs(delta) < std::fabs(wrapToPi(pick->yaw - last_yaw))))
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
            LockSample{timestamp_s, pick->position, pick->yaw, pick->target_id, pick_handoff});
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
    diagnostics_.last_reject = "lock:rate_ambiguous";
    if (!fit.ok || count < 6)
    {
        return false;
    }
    const double omega = fit.slope;
    const double slope_error = fit.slope_error;

    double phase_spread = 0.0;
    for (std::size_t i = 0; i < count; ++i)
    {
        phase_spread = std::max(phase_spread, std::fabs(body[i] - body[count - 1]));
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
    const bool flipped = outside < inside;

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
    // 声明"不转"必须有据：速率接近 0 且拟合标准误小。速率本身测不准时
    // （噪声大、样本短）不能拿"static_rate_rad_s"兜底——那正是旧实现把
    // 2-8 rad/s 的旋转目标全判成 Static、然后硬钉住、整段跟踪 0 次更新的原因。
    // 转速接近 0 且满足下列任一条，才声明"不转"：
    //   - 拟合标准误小（速率测得准）；
    //   - 板面法向几乎没扫开（相位不动 ⇒ 车体没转，与速率估得准不准无关）。
    // 第二条是给"法向抖动比转速还大"的场合兜底的：抖动大时速率估不准，但
    // "压根没扫开"这个事实本身很强。
    const double static_confidence = std::max(1.0, 0.5 * config_.static_rate_rad_s);
    const bool is_static =
        std::fabs(omega) <= config_.static_rate_rad_s &&
        (slope_error <= static_confidence || phase_spread < 0.15);
    EkfInitPrior prior;
    prior.radius_m = radius;
    prior.plate_slot = 0; // 槽位基准任意：相对相位才是物理量
    prior.spin = is_static ? SpinPriorKind::Static : SpinPriorKind::Spinning;
    prior.yaw_rate_prior_rad_s = is_static ? 0.0 : omega;
    prior.position_sigma_m = 0.15;
    prior.velocity_sigma_m_s = 1.0;
    // 首帧的相位来自单帧板面法向（实拍里小目标法向抖动可达 10 度以上），
    // 初值给宽容一点，让随后的更新把它拉正；给太紧会把 0.2-0.3m 的初值偏差
    // 全部判成离群，于是整段跟踪 0 次更新。
    prior.yaw_sigma_rad = 0.80;
    prior.yaw_rate_sigma_rad_s =
        is_static ? 0.0 : clampd(2.0 * slope_error, 0.8, 6.0);
    prior.radius_sigma_m = config_.radius_sigma_m;

    ArmorObservation observation;
    observation.position = last.position;
    if (pair_ok)
    {
        // 双板解出的几何直接覆盖：半径来自两块板的弦长，中心来自射线交点。
        prior.radius_m = pair_radius;
        prior.radius_sigma_m = std::min(config_.radius_sigma_m, 0.05);
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
            prior.velocity_sigma_m_s = 1.0;
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
            ArmorObservation observation;
            observation.position = world[i].position;
            observation.sigma_position_m = world[i].sigma_position_m;
            for (int slot = 0; slot < plate_count; ++slot)
            {
                const Eigen::Vector3d predicted = platePosition(ekf_.state(), slot, plate_count);
                const double distance = (world[i].position - predicted).norm();
                // 换板（更新的不是当前瞄准槽位）时观测更不可信：板编号只到
                // 2*pi/N 的可辨识度，观测落到哪块板上本身带一个槽位的歧义。
                // 不放大 R 的话，静止时把协方差压得很小的滤波器会把换板观测
                // 全判成 NIS 离群（实拍 v4 旋转目标 416/718 帧就是这样丢的）。
                ArmorObservation slot_observation = observation;
                if (slot != aim_slot_)
                {
                    slot_observation.sigma_position_m =
                        observation.sigma_position_m * config_.handoff_sigma_scale;
                }
                diagnostics_.best_gate_distance_m =
                    std::min(diagnostics_.best_gate_distance_m, distance);
                if (!(distance <= config_.position_gate_m))
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
                if (!bootstrapping && nis > config_.nis_gate)
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
        observation.position = chosen.position;
        observation.sigma_position_m = chosen.sigma_position_m;
        if (best_slot != aim_slot_)
        {
            observation.sigma_position_m *= config_.handoff_sigma_scale;
        }
        if (!ekf_.update(observation, best_slot))
        {
            break;
        }
        used[static_cast<std::size_t>(best_observation)] = true;
        ++accepted;
        diagnostics_.last_nis = best_nis;
        if (chosen.has_yaw)
        {
            // 相位观测：板面法向直接约束车体 yaw（折叠掉 2pi/N 的换板歧义）。
            // 它和位置观测吃的是同一帧的同一块板，但因为折叠，换板不会把
            // 相位带歪——这正是位置关联最脆弱的地方。
            if (config_.phase_update_enabled)
            {
                ekf_.updatePhase(chosen.yaw, config_.phase_update_sigma_rad);
            }
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
    int best_slot = -1;
    double best_error = std::numeric_limits<double>::infinity();
    for (int slot = 0; slot < plate_count; ++slot)
    {
        // 世界系本身就是控制器用的绝对角参考系（观测在 collectWorldObservations
        // 里已按云台姿态转过一次），这里不能再转回去，否则会把云台角算两遍。
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
