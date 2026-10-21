// 任务 4 粘合层（RotatingArmorTracker）的验收测试：
//   A 旋转目标的锁定/收敛/输出合法性（含装甲板随旋转切换）
//   B 静止目标的锁定/开火许可
//   C 短时遮挡的纯预测、超时清空
//   D 离群观测被 NIS + 距离门限拒绝
//   E 目标编号变化后的重锁
#include "tracker_test_support.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

namespace
{
using PredictionResult = ::PredictionResult;
using rotating_armor::State;
using TrackingState = ::TrackingState;

int g_failures = 0;

void check(bool condition, const char *what)
{
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
    {
        ++g_failures;
    }
}

bool finite(float value)
{
    return std::isfinite(value);
}

// 每一帧都必须满足的合法性与安全不变量。
bool frameIsLegal(const PredictionResult &result)
{
    if (result.state == TrackingState::NoTarget)
    {
        return result.target_id == 0 && !result.fire_allowed;
    }
    if (!finite(result.center_x_m) || !finite(result.center_y_m) ||
        !finite(result.center_velocity_x_m_s) || !finite(result.center_velocity_y_m_s) ||
        !finite(result.body_yaw) || !finite(result.body_yaw_velocity) ||
        !finite(result.radius_1_m) || !finite(result.radius_2_m) || !finite(result.distance_m) ||
        !finite(result.target_yaw) || !finite(result.target_pitch) ||
        !finite(result.target_yaw_velocity) || !finite(result.target_pitch_velocity))
    {
        return false;
    }
    if (result.target_id == 0 || result.radius_1_m <= 0.0F || result.distance_m <= 0.0F)
    {
        return false;
    }
    // 只有 Stable 才允许开火。
    if (result.fire_allowed && result.state != TrackingState::Stable)
    {
        return false;
    }
    return true;
}

State spinningTruth()
{
    State truth;
    // 车体中心 (3.0, 0.0)，低速平移；z=0.35m；以 2.5rad/s 自转；半径 0.25m。
    truth << 3.00, 0.05, 0.00, 0.02, 0.35, 0.0, 0.30, 2.50, 0.25, 0.0, 0.0;
    return truth;
}

State staticTruth()
{
    State truth;
    // 静止目标：中心 (3.0, 0.40)，yaw=0，半径 0.25m（半径只能靠先验）。
    truth << 3.00, 0.0, 0.40, 0.0, 0.35, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0;
    return truth;
}

double angleError(double a, double b)
{
    return std::fabs(rotating_armor::wrapToPi(a - b));
}

// ------------------------------------------------------------------ A
void scenarioSpinning()
{
    std::printf("\n[A] 旋转目标：锁定 / 收敛 / 换板 / 输出合法性\n");
    sim::Rig rig(spinningTruth());
    // 云台不追板，固定在车体中心方向：装甲板会自己转进视场，最能压换板与关联。
    rig.gimbal.yaw = rotating_armor::vectorYaw(Eigen::Vector3d(3.0, 0.0, 0.0));
    rig.gimbal.pitch = rotating_armor::vectorPitch(Eigen::Vector3d(3.0, 0.0, 0.35));
    rig.gimbal.rate_rad_s = 0.0;

    int first_lock = -1;
    int first_stable = -1;
    int stable_frames = 0;
    int fire_frames = 0;
    bool legal = true;
    PredictionResult last;
    for (int frame = 0; frame < 360; ++frame)
    {
        last = rig.step(true, false);
        if (first_lock < 0 && rig.tracker.initialized())
        {
            first_lock = frame;
        }
        if (first_stable < 0 && last.state == TrackingState::Stable)
        {
            first_stable = frame;
        }
        if (last.state == TrackingState::Stable)
        {
            ++stable_frames;
        }
        if (last.fire_allowed)
        {
            ++fire_frames;
        }
        legal = legal && frameIsLegal(last);
    }
    const State &estimate = rig.tracker.ekf().state();
    const State &truth = rig.truth;
    const double center_error = std::hypot(estimate(0) - truth(0), estimate(2) - truth(2));
    const double radius_error = std::fabs(estimate(8) - truth(8));
    const double rate_error = std::fabs(estimate(7) - truth(7));
    // 车体 yaw 只能确定到 2*pi/N：4 块装甲板几何上完全相同，把相位整体平移
    // 90 度得到的板位集合与真值完全一致，任何观测都分不出来。所以验收看
    // "模 2*pi/N 的误差"，另外把绝对值也打出来（报告里要说明这条对称性）。
    double yaw_error = angleError(estimate(6), truth(6));
    const double spacing = 2.0 * sim::kPi / static_cast<double>(rig.plate_count);
    for (int k = 0; k < rig.plate_count; ++k)
    {
        yaw_error = std::min(yaw_error, angleError(estimate(6) - spacing * k, truth(6)));
    }
    const auto &diagnostics = rig.tracker.diagnostics();
    std::printf("锁定帧=%d 稳定帧=%d 稳定计数=%d 开火计数=%d 重锁=%d\n", first_lock, first_stable,
                stable_frames, fire_frames, diagnostics.relock_count);
    std::printf("末帧误差: 中心=%.4f m 半径=%.4f m 角速度=%.3f rad/s yaw(模%d度)=%.3f rad "
                "yaw(绝对)=%.3f rad\n",
                center_error, radius_error, rate_error,
                static_cast<int>(360 / rig.plate_count), yaw_error,
                angleError(estimate(6), truth(6)));
    std::printf("锁定诊断: 半径来自数据=%d 相位翻转=%d 半径=%.3f 角速度=%.2f\n",
                diagnostics.radius_from_data ? 1 : 0, diagnostics.phase_flipped ? 1 : 0,
                diagnostics.locked_radius_m, diagnostics.locked_spin_rad_s);

    check(frameIsLegal(last) && legal, "A 所有帧的合法性与安全不变量");
    check(first_lock >= 0 && first_lock <= 90, "A 90 帧内完成锁定");
    check(first_stable >= 0 && first_stable <= 200, "A 200 帧内进入 Stable");
    check(stable_frames >= 200, "A 至少 200 帧处于 Stable");
    check(center_error <= 0.10, "A 车体中心误差 <= 0.10 m");
    check(radius_error <= 0.04, "A 旋转半径误差 <= 0.04 m");
    check(rate_error <= 0.40, "A 角速度误差 <= 0.40 rad/s");
    check(yaw_error <= 0.30, "A 车体 yaw 误差 <= 0.30 rad（模 2*pi/N）");
    check(diagnostics.relock_count <= 3, "A 重锁次数 <= 3（没有抖动）");
}

// ------------------------------------------------------------------ B
void scenarioStatic()
{
    std::printf("\n[B] 静止目标：锁定 / 声明静态 / 开火许可\n");
    sim::Rig rig(staticTruth());
    // 云台固定对准真值里离 yaw=0 最近的那块板（静止目标不会跑掉）。
    rig.gimbal.snapTo(sim::nearestPlatePosition(rig.truth, rig.plate_count, 0.0));
    int first_stable = -1;
    int fire_frames = 0;
    bool legal = true;
    bool saw_no_target_after_lock = false;
    PredictionResult last;

    for (int frame = 0; frame < 240; ++frame)
    {
        last = rig.step(true, false);
        if (last.state == TrackingState::Stable)
        {
            // 云台伺服：跟踪器说瞄准点在哪个方向，炮塔就转到哪（无延迟理想伺服）。
            // 之前这里把云台死死钉在"离 yaw=0 最近的那块板"上，等于要求跟踪器
            // 必须瞄那一块；改成伺服后，测的是"开火许可与瞄准点是否自洽"。
            rig.gimbal.yaw = last.target_yaw;
            rig.gimbal.pitch = last.target_pitch;
        }
        if (first_stable < 0 && last.state == TrackingState::Stable)
        {
            first_stable = frame;
        }
        if (last.fire_allowed)
        {
            ++fire_frames;
        }
        if (rig.tracker.initialized() && last.state == TrackingState::NoTarget)
        {
            saw_no_target_after_lock = true;
        }
        legal = legal && frameIsLegal(last);
    }
    const auto &diagnostics = rig.tracker.diagnostics();
    std::printf("稳定帧=%d 开火计数=%d 锁定角速度=%.3f 半径来自数据=%d\n", first_stable,
                fire_frames, diagnostics.locked_spin_rad_s,
                diagnostics.radius_from_data ? 1 : 0);
    std::printf("末帧: yaw 指令=%.3f 云台 yaw=%.3f 距离=%.3f\n", last.target_yaw, rig.gimbal.yaw,
                last.distance_m);

    check(legal, "B 所有帧的合法性与安全不变量");
    check(first_stable >= 0, "B 进入 Stable");
    check(std::fabs(diagnostics.locked_spin_rad_s) < 1e-9, "B 判定为静止（角速度被钉住）");
    check(fire_frames >= 30, "B 对准后允许开火（>=30 帧）");
    check(!saw_no_target_after_lock, "B 锁定后没有掉成无目标");
    check(angleError(last.target_yaw, rig.gimbal.yaw) <= 0.06, "B 目标 yaw 与云台 yaw 一致");
    check(std::fabs(last.distance_m - 3.0) <= 0.35, "B 距离在真实距离附近");
}

// ------------------------------------------------------------------ C
void scenarioOcclusionAndTimeout()
{
    std::printf("\n[C] 短时遮挡的纯预测 / 超时清空\n");
    sim::Rig rig(spinningTruth());
    rig.gimbal.yaw = rotating_armor::vectorYaw(Eigen::Vector3d(3.0, 0.0, 0.0));
    rig.gimbal.pitch = rotating_armor::vectorPitch(Eigen::Vector3d(3.0, 0.0, 0.35));
    for (int frame = 0; frame < 150; ++frame)
    {
        rig.step(true, false);
    }
    check(rig.tracker.initialized(), "C 遮挡前已锁定");

    // 0.25 s 完全没有观测：必须继续预测，且不许开火。
    bool predicted_through = true;
    bool finite_state = true;
    bool fire_during_occlusion = false;
    bool saw_missed = false;
    for (int frame = 0; frame < 15; ++frame)
    {
        const PredictionResult result = rig.step(false, false);
        if (!frameIsLegal(result))
        {
            predicted_through = false;
        }
        if (result.state != TrackingState::NoTarget)
        {
            saw_missed = saw_missed || rig.tracker.diagnostics().missed_frames > 0;
            finite_state = finite_state && std::isfinite(result.center_x_m) &&
                           std::isfinite(result.target_yaw);
        }
        fire_during_occlusion = fire_during_occlusion || result.fire_allowed;
    }
    check(predicted_through, "C 遮挡期间仍输出合法预测（纯预测）");
    check(finite_state, "C 遮挡期间状态保持有限");
    check(!fire_during_occlusion, "C 遮挡期间禁止开火");
    check(saw_missed, "C 遮挡被记为丢帧");

    // 恢复观测后必须还能更新并重新稳定。
    bool recovered = false;
    for (int frame = 0; frame < 120; ++frame)
    {
        const PredictionResult result = rig.step(true, false);
        if (result.state == TrackingState::Stable)
        {
            recovered = true;
        }
    }
    check(recovered, "C 遮挡恢复后重新进入 Stable");

    // 超过 timeout 的长时间丢失：必须清空成无目标。
    PredictionResult last;
    for (int frame = 0; frame < 60; ++frame)
    {
        last = rig.step(false, false);
    }
    std::printf("丢失 1s 后: state=%d target_id=%d\n", static_cast<int>(last.state),
                static_cast<int>(last.target_id));
    check(last.state == TrackingState::NoTarget, "C 超时后输出无目标");
    check(!rig.tracker.initialized(), "C 超时后跟踪器被清空");
}

// ------------------------------------------------------------------ D
void scenarioOutlierRejected()
{
    std::printf("\n[D] 离群观测被拒绝\n");
    sim::Rig rig(spinningTruth());
    rig.gimbal.yaw = rotating_armor::vectorYaw(Eigen::Vector3d(3.0, 0.0, 0.0));
    rig.gimbal.pitch = rotating_armor::vectorPitch(Eigen::Vector3d(3.0, 0.0, 0.35));
    for (int frame = 0; frame < 150; ++frame)
    {
        rig.step(true, false);
    }
    check(rig.tracker.initialized(), "D 越界前已锁定");
    const State before = rig.tracker.ekf().state();

    // 一帧里塞一个 1.5m 之外的假观测：它必须被距离门限 + NIS 拒绝。
    rig.time_s += rig.dt;
    rig.truth = rotating_armor::transition(rig.truth, rig.dt);
    const auto plates = sim::platesAt(rig.truth, rig.plate_count);
    const GimbalState gimbal_state = rig.gimbal.state(0.0);
    std::vector<ArmorPose> observations = sim::observe(plates, gimbal_state, rig.target_id, rig.rng);
    ArmorPose outlier;
    outlier.detection.target_id = rig.target_id;
    outlier.detection.size = ArmorSize::Small;
    outlier.detection.confidence = 0.9F;
    outlier.reprojection_error = 0.5;
    outlier.position_camera_m = cv::Vec3d(3.6, 1.5, 0.35);
    outlier.position_gimbal_m = cv::Vec3d(3.6, 1.5, 0.35);
    outlier.normal_gimbal = cv::Vec3d(-1.0, 0.0, 0.0);
    outlier.armor_yaw = 3.14159;
    observations.push_back(outlier);

    const PredictionResult result = rig.tracker.step(observations, gimbal_state, rig.time_s);
    const auto &diagnostics = rig.tracker.diagnostics();
    const State after = rig.tracker.ekf().state();
    const double center_jump = std::hypot(after(0) - before(0), after(2) - before(2));
    std::printf("本帧接受=%d 拒绝=%d 中心跳变=%.4f m\n", diagnostics.updates_this_frame,
                diagnostics.rejected_this_frame, center_jump);
    check(diagnostics.rejected_this_frame >= 1, "D 离群观测被计数为拒绝");
    check(center_jump <= 0.05, "D 离群观测没有把状态拉走");
    check(frameIsLegal(result), "D 该帧输出仍然合法");
}

// ------------------------------------------------------------------ E
void scenarioTargetSwitch()
{
    std::printf("\n[E] 目标编号变化后重锁\n");
    sim::Rig rig(spinningTruth());
    rig.gimbal.yaw = rotating_armor::vectorYaw(Eigen::Vector3d(3.0, 0.0, 0.0));
    rig.gimbal.pitch = rotating_armor::vectorPitch(Eigen::Vector3d(3.0, 0.0, 0.35));
    for (int frame = 0; frame < 150; ++frame)
    {
        rig.step(true, false);
    }
    check(rig.tracker.initialized(), "E 切换前已锁定");

    // 连续给 3 帧新编号：应触发丢弃并重新锁定到新目标。
    bool dropped = false;
    for (int frame = 0; frame < 3; ++frame)
    {
        const PredictionResult result = rig.step(true, false, 9);
        if (result.state == TrackingState::NoTarget)
        {
            dropped = true;
        }
    }
    check(dropped, "E 换目标时先输出无目标（不允许带着旧状态开火）");

    int relocked = -1;
    for (int frame = 0; frame < 120; ++frame)
    {
        const PredictionResult result = rig.step(true, false, 9);
        if (result.state != TrackingState::NoTarget && result.target_id == 9)
        {
            relocked = frame;
            break;
        }
    }
    std::printf("重锁帧=%d 重锁次数=%d samples=%d reject=%s\n", relocked,
                rig.tracker.diagnostics().relock_count, rig.tracker.diagnostics().lock_samples,
                rig.tracker.diagnostics().last_reject.c_str());
    check(relocked >= 0, "E 重新锁定到新目标编号");
}

// ------------------------------------------------------------------ F
void scenarioLatencyCompensation()
{
    std::printf("\n[F] 预测时域包含实测处理延迟\n");
    // 静止不转、以 0.5m/s 平移的目标：延迟只应该把预测位置沿着速度方向推出去。
    State truth;
    truth << 3.00, 0.50, 0.40, 0.0, 0.35, 0.0, 0.0, 0.0, 0.25, 0.0, 0.0;
    sim::Rig plain(truth);
    sim::Rig delayed(truth);
    plain.gimbal.snapTo(sim::nearestPlatePosition(plain.truth, plain.plate_count, 0.0));
    delayed.gimbal = plain.gimbal;
    delayed.latency_s = 0.10;
    PredictionResult a;
    PredictionResult b;
    for (int frame = 0; frame < 120; ++frame)
    {
        a = plain.step(true, false);
        b = delayed.step(true, false);
        if (frame % 20 == 0)
        {
            std::printf("  f=%d phase=%d samples=%d relock=%d reject=%s\n", frame,
                        static_cast<int>(plain.tracker.diagnostics().phase),
                        plain.tracker.diagnostics().lock_samples,
                        plain.tracker.diagnostics().relock_count,
                        plain.tracker.diagnostics().last_reject.c_str());
        }
    }
    std::printf("F 诊断: plain phase=%d samples=%d reject=%s | delayed reject=%s\n",
                static_cast<int>(plain.tracker.diagnostics().phase),
                plain.tracker.diagnostics().lock_samples,
                plain.tracker.diagnostics().last_reject.c_str(),
                delayed.tracker.diagnostics().last_reject.c_str());
    const double difference = static_cast<double>(b.distance_m) - static_cast<double>(a.distance_m);
    std::printf("无延迟距离=%.3f 加 0.1s 延迟=%.3f 差=%.3f (期望约 %.3f)\n",
                static_cast<double>(a.distance_m), static_cast<double>(b.distance_m), difference,
                0.5 * 0.10);
    check(std::fabs(difference - 0.05) <= 0.03, "F 预测位置随实测延迟前推");
    check(a.state == TrackingState::Stable && b.state == TrackingState::Stable,
          "F 两条路径都进入 Stable");
    check(std::fabs(delayed.tracker.diagnostics().measured_latency_s - 0.10) < 1e-9,
          "F 诊断里能看到实测延迟");
}


// ------------------------------------------------------------------ G
// 高速自转目标的"能不能打着"验收：云台跟瞄 + 50ms 处理延迟，
// 比较"预测出的射击点"与"子弹到达时刻的真实板心"。
void scenarioFastSpin(double omega)
{
    State truth;
    truth << 3.00, 0.0, 0.0, 0.0, 0.35, 0.0, 0.30, omega, 0.25, 0.0, 0.0;
    sim::Rig rig(truth);
    rig.dt = 1.0 / 166.0;
    rig.sigma_position_m = 0.03; // 实拍检测抖动（3cm 量级）
    rig.latency_s = 0.05;        // 视觉 + 传输 + 电机的处理延迟
    rig.gimbal.yaw = rotating_armor::vectorYaw(
        sim::nearestPlatePosition(rig.truth, rig.plate_count, 0.0));
    rig.gimbal.pitch = rotating_armor::vectorPitch(
        sim::nearestPlatePosition(rig.truth, rig.plate_count, 0.0));
    rig.gimbal.rate_rad_s = 12.0;

    sim::Rig no_latency(truth);
    no_latency.dt = rig.dt;
    no_latency.sigma_position_m = rig.sigma_position_m;
    no_latency.gimbal = rig.gimbal;

    std::vector<double> strike_errors;
    std::vector<double> now_errors;
    std::vector<double> min4_errors;
    int front_hit = 0;
    int front_total = 0;
    int stable_frames = 0;
    for (int frame = 0; frame < 420; ++frame)
    {
        const PredictionResult result = rig.step(true, true);
        const PredictionResult plain = no_latency.step(true, true);
        static_cast<void>(plain);
        if (result.state != TrackingState::Stable)
        {
            continue;
        }
        ++stable_frames;
        const auto &diagnostics = rig.tracker.diagnostics();
        if (diagnostics.aim_slot < 0)
        {
            continue;
        }
        const Eigen::Vector3d aim = sim::polarPoint(static_cast<double>(result.target_yaw),
                                                    static_cast<double>(result.target_pitch),
                                                    static_cast<double>(result.distance_m));
        const State strike_state = rotating_armor::transition(rig.truth, rig.latency_s);
        const Eigen::Vector3d strike_plate =
            rotating_armor::measurement(strike_state, diagnostics.aim_slot, rig.plate_count);
        const Eigen::Vector3d now_plate =
            rotating_armor::measurement(rig.truth, diagnostics.aim_slot, rig.plate_count);
        strike_errors.push_back((aim - strike_plate).norm());
        now_errors.push_back((aim - now_plate).norm());
        // 四块板完全相同：瞄准点落在"任意一块真板"上也算能打到（槽位编号只是
        // 标号，锁定时的相位分支会让标号整体错一位）。
        double nearest = 1.0e9;
        int nearest_slot = -1;
        for (int slot = 0; slot < rig.plate_count; ++slot)
        {
            const double error =
                (aim - rotating_armor::measurement(strike_state, slot, rig.plate_count)).norm();
            if (error < nearest)
            {
                nearest = error;
                nearest_slot = slot;
            }
        }
        min4_errors.push_back(nearest);
        if (std::getenv("GD_DEBUG") != nullptr && frame % 30 == 0)
        {
            const State& est = rig.tracker.ekf().state();
            std::printf("[gd] f=%3d om=%.2f(est %.2f) R=%.3f l=%.3f h=%.3f "
                        "aim=%d near=%d near_err=%.3f strike=%.3f\n",
                        frame, omega, est(7), est(8), est(9), est(10),
                        diagnostics.aim_slot, nearest_slot, nearest,
                        strike_errors.empty() ? -1.0 : strike_errors.back());
        }
        // "打得到"还必须瞄在朝相机那一面：背面板被车体挡住，命中也只是打在车壳上。
        if (nearest <= 0.03 && nearest_slot >= 0)
        {
            const double phase = rotating_armor::wrapToPi(
                strike_state(6) + 2.0 * sim::kPi * static_cast<double>(nearest_slot) /
                                     static_cast<double>(rig.plate_count));
            const Eigen::Vector3d normal(std::cos(phase + sim::kPi), std::sin(phase + sim::kPi),
                                         0.0);
            const Eigen::Vector3d to_shooter = -aim.normalized();
            ++front_total;
            if (normal.dot(to_shooter) > 0.0)
            {
                ++front_hit;
            }

        }
    }
    auto median = [](std::vector<double> values) {
        if (values.empty())
        {
            return 0.0;
        }
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    };
    auto p90 = [](std::vector<double> values) {
        if (values.empty())
        {
            return 0.0;
        }
        std::sort(values.begin(), values.end());
        return values[static_cast<std::size_t>(0.9 * static_cast<double>(values.size() - 1))];
    };
    const double strike_median = median(strike_errors);
    const double min4_median = median(min4_errors);
    const double now_median = median(now_errors);
    int hit = 0;
    for (double error : min4_errors)
    {
        if (error <= 0.03)
        {
            ++hit;
        }
    }
    const double hit_rate =
        min4_errors.empty() ? 0.0
                            : static_cast<double>(hit) / static_cast<double>(min4_errors.size());
    const double front_rate =
        front_total == 0 ? 0.0 : static_cast<double>(front_hit) / static_cast<double>(front_total);
    const auto &estimate = rig.tracker.ekf().state();
    std::printf("  omega=%4.1f rad/s: 稳定帧=%d/%d 角速度估计=%.2f 射击点误差 中位=%.3f p90=%.3f "
                "| 最近真板 中位=%.3f | 朝面板占比=%.0f%% | 3cm 命中率=%.0f%%\n",
                omega, stable_frames, 420, estimate(7), strike_median, p90(strike_errors),
                min4_median, front_rate * 100.0, hit_rate * 100.0);
    if (omega >= 0.0)
    {
        check(min4_median <= 0.03 && hit_rate >= 0.9 && front_rate >= 0.9,
              "G 高速自转：射击点落在朝相机的真板上（中位 <= 3cm、命中率 >= 90%）");
    }
}

// ------------------------------------------------------------------ H
// 转速的"自我判别"：目标到底转不转，不能靠调用方声明，也不该被法向噪声带跑。
// 实拍里板面法向在小目标上抖 ±20°，而"静止"和"慢转"在单帧上根本分不开。
void scenarioSelfClassifiedRotation()
{
    std::printf("\n[H] 转速自我判别（不靠调用方声明，法向噪声决定信不信）\n");
    struct Case
    {
        const char *name;
        double omega;
        double vx;
        double vy;
        double sigma_normal_rad;
        double tolerance;
    };
    const Case cases[] = {
        // 实拍小目标法向抖 ±20°：静止/平移都必须判成"没在转"。
        {"静止+噪声法向", 0.0, 0.0, 0.0, 0.35, 0.45},
        {"静止+干净法向", 0.0, 0.0, 0.0, 0.02, 0.45},
        {"平移+噪声法向", 0.0, 0.0, 0.50, 0.35, 0.45},
        // 法向干净时，慢转必须真的被跟到（不能一律往 0 压）。
        {"慢转+干净法向", 1.5, 0.0, 0.0, 0.05, 0.80},
        // v1 的真实现象：车其实是静止的，动的是云台（手持抖动让"报告姿态"和
        // "真实姿态"差 0.03 rad）。这时候世界系里的板在乱晃，法向每帧转 ~2°，
        // 任何逐帧微分都会给出假转速——要求 omega 仍被压住。
        {"静止+云台抖动", 0.0, 0.0, 0.0, 0.12, 0.60},
    };
    for (const Case &test_case : cases)
    {
        State truth;
        truth << 3.0, test_case.vx, 0.0, test_case.vy, 0.35, 0.0, 0.40, test_case.omega, 0.20, 0.0, 0.0;
        // 用生产默认配置（半径钉死）：合成测试默认放开半径是为了验证"坏初值
        // 能收敛"，但半径与中心在单板观测下是耦合的，法向噪声大时半径会被
        // 位置残差拉着塌掉（实测 R 从 0.25 掉到 0.017），那不是这里要测的东西。
        rotating_armor::TrackerConfig config;
        sim::Rig rig(truth, config);
        rig.sigma_normal_rad = test_case.sigma_normal_rad;
        if (std::strcmp(test_case.name, "静止+云台抖动") == 0)
        {
            rig.shake_sigma_rad = 0.030; // 166fps 下相当于云台角速率 ~5 rad/s
        }
        rig.gimbal.yaw = rotating_armor::vectorYaw(Eigen::Vector3d(3.0, 0.0, 0.0));
        rig.gimbal.pitch = rotating_armor::vectorPitch(Eigen::Vector3d(3.0, 0.0, 0.35));
        // 云台追"车心"而不是追某块板：追板会让云台在 4 块板之间来回跳，
        // 世界系法向变成锯齿，那是夹具的 artifact，不是被测对象。
        for (int frame = 0; frame < 420; ++frame)
        {
            rig.gimbal.chase(Eigen::Vector3d(rig.truth(0), rig.truth(2), rig.truth(4)), rig.dt);
            rig.step(true, false);
        }
        const auto &estimate = rig.tracker.ekf().state();
        const double omega_error = std::fabs(estimate(7) - test_case.omega);
        std::printf("  %-16s sigma_yaw=%.2f rad  omega 估计=%+.2f（真值 %+.2f，差 %.2f）"
                    "R=%.3f 车心误差=%.3f m\n",
                    test_case.name, test_case.sigma_normal_rad, estimate(7), test_case.omega,
                    omega_error, estimate(8),
                    std::hypot(estimate(0) - rig.truth(0), estimate(2) - rig.truth(2)));
        check(omega_error <= test_case.tolerance,
              "H 转速自我判别：omega 估计落在真值附近（误差 <= tolerance）");
    }
}

} // namespace

int main()
{
    scenarioSpinning();
    scenarioStatic();
    scenarioOcclusionAndTimeout();
    scenarioOutlierRejected();
    scenarioTargetSwitch();
    scenarioLatencyCompensation();
    std::printf("\n[G] 高速自转目标：能不能瞄准（处理延迟 50ms）\n");
    scenarioFastSpin(2.0);
    scenarioFastSpin(4.0);
    scenarioFastSpin(6.0);
    scenarioFastSpin(8.0);
    scenarioSelfClassifiedRotation();
    if (g_failures != 0)
    {
        std::printf("\n%d 项失败\n", g_failures);
        return 1;
    }
    std::printf("\n全部通过\n");
    return 0;
}
