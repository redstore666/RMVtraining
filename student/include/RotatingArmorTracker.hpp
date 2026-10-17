#pragma once

// 旋转车体目标跟踪器（任务 4 的粘合层 + 关联 + 状态机 + 输出映射）。
//
// 分工：
//   RotatingArmorModel   —— 唯一一份运动模型 / 观测模型公式（自由函数）
//   RotatingArmorEkf     —— 单次 predict / update、初始化契约、NIS
//   RotatingArmorTracker —— 本文件：观测坐标系变换、装甲板槽位关联、
//                            离群点拒绝、锁定与重新锁定、超时注销、
//                            稳定/不稳定状态机、PredictionResult 输出
//
// 为什么锁定要攒多帧：单帧的 3D 板心 + 板面法向只给出"相位 mod pi"，
// "车体在板的哪一侧"是另一个自由度（等价于半径的符号）。所以锁定阶段用
// 最近若干帧做一次最小二乘：
//     p_k = c + v * (t_k - t_last) - R * [cos(phi_k), sin(phi_k)]
// 未知量 (cx, cy, vx, vy, R)，phi_k 由板面法向给出（mod pi 的两个候选取
// 拟合半径 > 0 的那个）。相位扫过角太小（目标不转）时 R 不可辨识，此时退回
// "车体在板背后"的几何先验（相机只能看到板正面），半径用配置先验。

#include "AimGeometry.hpp"
#include "AutoAimTypes.hpp"
#include "RotatingArmorEkf.hpp"

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace rotating_armor
{

// 跟踪系（世界系）里某块装甲板的中心位置。
// 薄封装 measurement()：让上层不必关心观测模型的公式，也保证只有一份真源。
Eigen::Vector3d platePosition(const State& state, int plate_slot, int plate_count);

struct TrackerConfig
{
    EkfConfig ekf{};

    // === 观测 ===
    // 位置噪声（米）。单帧 PnP 的板心抖动实测约 1cm，这里取 3cm 留出
    // "倾斜角/四角点抖动"的余量；重投影误差大的观测再按比例放大。
    double position_sigma_m{0.03};
    // 板面法向 yaw 的噪声（弧度）。IPPE 孪生解会让 yaw 差 10~30°，取 0.35。
    double normal_sigma_rad{0.35};

    // === 锁定 ===
    int lock_min_samples{8};         // 至少攒到的帧数（只收带法向的观测）
    double lock_min_span_s{0.18};     // 至少覆盖的时间跨度（短了圆拟合与转速都不可辨）
    double lock_max_age_s{0.50};      // 更早的样本丢掉（目标可能已经换了）
    double radius_prior_m{0.25};      // 无数据先验：实车按底盘实测
    double radius_sigma_m{0.10};
    double radius_min_m{0.05};
    double radius_max_m{0.60};
    double phase_spread_for_radius_rad{0.30}; // 相位扫过小于它 -> 半径不可辨识
    double static_rate_rad_s{1.2};            // |omega| 低于它判为"不转"（Static）

    // === 关联 / 门限 ===
    double position_gate_m{0.60}; // 单板位置门限（含中心先验不确定的余量）
    double nis_gate{11.34};      // chi2(3 dof) 99.5%
    double handoff_sigma_scale{2.5}; // 更新换板槽位时把观测标准差放大这么多倍
    int bootstrap_updates{10};   // 锁定期头几次更新先放宽 NIS 门（初值靠单帧法向，偏差可比预想大）
    // 声明反驳检查：法向历史里混入相邻板样本时斜率会失真，实拍里会误杀，
    // 默认关闭；需要时打开（合成夹具与 v4 的静止车都不需要它）。
    bool contradiction_check{false};
    int max_updates_per_frame{2}; // 同一帧最多用几块板更新（防止病态约束）

    // === 状态机 ===
    int stable_updates{12};          // 连续接受多少帧才允许报 Stable
    int unstable_after_missed{2};    // 仅供诊断：Stable 的降级判据已改为按秒（见 stale_after_s）
    double stale_after_s{0.10};      // 距上次被接收更新超过此秒数即判 Unstable
    double timeout_s{0.60};          // 超过多久没有任何有效观测 -> 注销

    // === 声明校验（Static/Spinning 被数据反驳时重锁）===
    int relock_frames{6};
    double relock_omega_diff_rad_s{1.0};
    double relock_min_span_s{0.18};

    // === 输出 ===
    double max_bias_s{0.80};
    // 本帧视觉侧处理延迟（取到帧 -> 算出结果）的允许上限。
    double max_latency_s{0.30};        // prediction_bias_s 的限幅
    double fire_angle_error_rad{0.05}; // 云台已对准的阈值（Planner 的 fire_thresh）
    double fire_max_age_s{0.20};    // 最近一次有效观测距今多久内允许开火
    double min_distance_m{0.40};
    double max_distance_m{12.0};
    double switch_margin_rad{0.06}; // 换板迟滞：别的板要明显更好才换
};

enum class LockPhase : std::uint8_t
{
    Idle = 0,     // 没有目标
    Collecting,   // 正在攒锁定样本
    Tracking,     // EKF 已初始化
};

// 每帧诊断量。只给测试与 Web Debug 用，不进通信协议。
struct TrackerDiagnostics
{
    LockPhase phase{LockPhase::Idle};
    int lock_samples{0};
    int last_lock_samples{0};
    double locked_radius_m{0.0};
    double locked_spin_rad_s{0.0};
    bool radius_from_data{false};
    bool phase_flipped{false};
    std::uint64_t accepted_updates{0};
    int consecutive_updates{0};
    int missed_frames{0};
    int updates_this_frame{0};
    double measured_latency_s{0.0};
    int rejected_this_frame{0};
    double last_nis{0.0};
    double best_gate_distance_m{0.0}; // 本帧所有(观测,槽位)对里最近的位置残差（诊断用）
    double best_gate_nis{0.0};       // 本帧最小 NIS（诊断用）
    int aim_slot{-1};
    int target_id{0};
    int relock_count{0};
    std::string last_reject;
    std::string stability; // 本帧 Stable/Unstable 的判定理由
    // 本帧预测出的瞄准点（世界系），离线回放评估用：与同帧检测到的板心比距离。
    Eigen::Vector3d predicted_aim_world{0.0, 0.0, 0.0};
};

class RotatingArmorTracker
{
public:
    explicit RotatingArmorTracker(TrackerConfig config = TrackerConfig{});

    void reset();

    // 与 ekf_predict 同签名：喂进本帧的全部装甲板位姿，返回本帧应该发出去的结果。
    // 没有目标、时间戳非法或跟踪失效时返回默认的 NoTarget 结果。
    // measured_latency_s：本帧从取到图到调用本函数已经花掉的时间（秒），
    // 会叠加到 prediction_bias_s 上一起作为预测时域。
    PredictionResult step(const std::vector<ArmorPose>& observations,
                          const GimbalState& gimbal, double timestamp_seconds,
                          double measured_latency_s = 0.0);

    bool initialized() const { return ekf_.initialized(); }
    const TrackerDiagnostics& diagnostics() const { return diagnostics_; }
    const TrackerConfig& config() const { return config_; }
    const RotatingArmorEkf& ekf() const { return ekf_; }

private:
    // 一帧观测转到跟踪系之后的形态。
    struct WorldObservation
    {
        Eigen::Vector3d position{0.0, 0.0, 0.0};
        bool has_yaw{false};
        double yaw{0.0};
        int target_id{0};
        double sigma_position_m{0.03};
    };

    struct LockSample
    {
        double timestamp_s{0.0};
        Eigen::Vector3d position{0.0, 0.0, 0.0};
        double yaw{0.0}; // 板面法向 yaw（世界系，mod pi 待定）
        int target_id{0};
        bool handoff{false}; // 与上一个样本之间换了板（相差 pi/2 的整数倍）
    };

    // 位置圆拟合（车体中心 + 半径），Gauss-Newton。
    struct CircleFit
    {
        double cx{0.0};
        double cy{0.0};
        double radius{0.0};
        double residual{0.0};
        bool ok{false};
    };
    static CircleFit fitArmorCircle(const std::deque<LockSample>& samples);

    // 观测 -> 跟踪系（含剔除明显非法的观测）。
    void collectWorldObservations(const std::vector<ArmorPose>& observations,
                                  const GimbalState& gimbal,
                                  std::vector<WorldObservation>& out) const;

    // 攒样本；够了就尝试锁定。成功返回 true。
    // 双板可见时用两块相邻板直接重定几何（p_i + R*n_i = p_j + R*n_j），
    // 用于"锁定后更新持续被拒"的场合：单板定出的中心偏 ~R，靠更新拉不回来。
    bool reinitializeFromPair(const std::vector<WorldObservation>& world);
    bool tryLock(const std::vector<WorldObservation>& world, double timestamp_s,
                 const GimbalState& gimbal);

    // 关联 + 更新。返回本帧接受的更新数。
    int associateAndUpdate(const std::vector<WorldObservation>& world, double timestamp_s);

    // 声明是否被观测反驳（Static/Spinning 判错、或目标转速突变）。
    bool declarationContradicted(const std::vector<WorldObservation>& world,
                                 double timestamp_s);

    // 填写输出（含瞄准板选择、开火许可、数值合法性检查）。
    PredictionResult buildResult(const GimbalState& gimbal, double timestamp_s);

    // 丢掉当前目标并转入重新收集（保留会话级计数与诊断）。
    void dropTarget(const char* reason);

    // 把某个 slot 的法向 yaw 记入历史（用于检查"转/不转"的声明是否被数据反驳）。
    void pushNormalHistory(int plate_slot, double timestamp_s, double world_yaw_rad);

    TrackerConfig config_{};
    RotatingArmorEkf ekf_{config_.ekf};
    TrackerDiagnostics diagnostics_{};

    bool has_timestamp_{false};
    double last_timestamp_s_{0.0};
    double state_timestamp_s_{0.0}; // EKF 状态对应的时刻

    bool initialized_{false};
    int target_id_{0};
    int aim_slot_{-1};
    double last_accept_s_{0.0};
    double measured_latency_s_{0.0};
    bool has_accept_{false};
    int consecutive_updates_{0};
    int missed_frames_{0};
    int other_id_streak_{0};

    std::deque<LockSample> lock_samples_{};
    // 每个 slot 一份：(timestamp, 解缠后的法向 yaw)，用于估计观测到的角速度。
    // 必须按 slot 分开：同一辆车不同装甲板的法向相差 2*pi*k/N，
    // 混在一起会让斜率完全失真。
    std::vector<std::deque<std::pair<double, double>>> normal_history_per_slot_{};
    int contradiction_streak_{0};
    int relock_count_{0};

    // 最近一次成功锁定用到的量（诊断/测试用）
    double locked_radius_m_{0.0};
    double locked_spin_rad_s_{0.0};
    bool has_pair_relock_{false};
    double last_pair_relock_s_{0.0};
    double timestamp_s_{0.0};
    bool lock_radius_from_data_{false};
    bool lock_phase_flipped_{false};
};

} // namespace rotating_armor
