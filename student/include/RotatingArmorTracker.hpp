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

#include <array>
#include <cstdint>
#include <deque>
#include <limits>
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
    // 车体平动速度的先验：单板可见时"板在动"和"车在动"不可分，把速度放开
    // 会让滤波器把旋转解释成平移、中心以 1 m/s 飞出去（实拍 v4 旋转目标
    // 中心跑 1.4 m 后位置门全拒）。所以速度先验压到很小、基本钉在 0。
    double velocity_sigma_m_s{0.05};
    // 过程噪声缩放：默认 1.0 = ProcessNoiseConfig 的原值。自转目标的板心每帧能走
    // 30-50 mm，滤波器 P 收得太紧时会"拒跟"观测（实测状态离刚更新的观测 87 mm）。
    double process_accel_scale{1.0};
    // 板面法向 yaw 的噪声（弧度）。IPPE 孪生解会让 yaw 差 10~30°，取 0.35。
    double normal_sigma_rad{0.35};

    // === 锁定 ===
    int lock_min_samples{8};         // 至少攒到的帧数（只收带法向的观测）
    double lock_min_span_s{0.18};     // 至少覆盖的时间跨度（短了圆拟合与转速都不可辨）
    double lock_max_age_s{0.50};      // 更早的样本丢掉（目标可能已经换了）
    double radius_prior_m{0.25};      // 无板型信息时的兜底先验：实车按底盘实测
    // 按板型给的半径先验（车体中心到装甲板面的距离）。实测 v4 的 3 号小装甲车
    // 是 0.197 m，用 0.20 比 0.25 的残差中位从 114 mm 降到 87 mm。
    double radius_prior_small_m{0.20};
    double radius_prior_large_m{0.25};
    // 是否按板型选半径先验。关掉（或用 --radius 显式给值）时一律用 radius_prior_m。
    bool radius_prior_from_size{true};
    double radius_sigma_m{0.10};
    // 锁定后是否把半径钉在锁定值：单板可见时 (中心, 半径) 退化，位置更新会
    // 把 R 一点点拉走（实测 0.25 -> 0.147，中心随之跑 1.4 m）。实时跟踪按
    // "半径由板型先验给出"处理，默认钉死；合成/离线场景想测半径收敛可以关掉。
    bool freeze_radius{true};
    double radius_min_m{0.05};
    double radius_max_m{0.60};
    double phase_spread_for_radius_rad{0.30}; // 相位扫过小于它 -> 半径不可辨识
    // 位置圆拟合锁定：高速自转时单帧板面 yaw 的抖动（实拍 ±0.3 rad）比单帧
    // 真实转角还大，yaw 序列拟合不出转速；板心位置却准（重投影 0.5 px），
    // 四块板共圆 => 圆拟合给 (c, R)，相邻板心的切向位移/半径给 omega。
    double lock_circle_residual_m{0.05};              // 圆拟合 RMS 残差超过它不可信
    double lock_circle_max_step_ratio{0.25};          // 单步位移 > 0.25R 视为换板，跳过
    double lock_circle_trusted_ratio{0.35};
    double lock_circle_min_span_rad{1.2};           // 有效步数占比低于它不采信位置速率
    double static_rate_rad_s{1.2};            // |omega| 低于它判为"不转"（Static）

    // === 关联 / 门限 ===
    double position_gate_m{0.60}; // 单板位置门限（含中心先验不确定的余量）
    double nis_gate{11.34};      // chi2(3 dof) 99.5%
    double handoff_sigma_scale{2.5}; // 已废弃（仅保留字段）：换板不再放大 sigma
    // 观测标准差的地板：单板预测的误差里有一块是"模型误差"（半径/相位/中心
    // 的系统性偏差），不体现在 P 里。地板太小 → NIS 门把有信息的帧全拒了
    // （实拍 v4 旋转目标 466/466 帧被拒）；地板太大 → 更新变钝。
    double association_sigma_floor_m{0.06};
    // 观测噪声自适应：用"连续三帧观测的二阶差分"估当前目标的检测抖动。
    // 固定地板（0.06 m）对静止车大了一个量级（实测帧间抖动 4 mm），滤波器
    // 于是几乎不信观测（NIS 中位 0.03，理论应 ~2.4）；对高速自转车 0.06 m
    // 又略小。改成按实测抖动给 sigma，静止车更贴合、自转车不被夸大。
    bool adaptive_observation_noise{true};
    int adaptive_noise_window{24};
    double adaptive_noise_min_sigma_m{0.004};
    double adaptive_noise_max_sigma_m{0.090};
    // 同帧两块板的中点直接给"车体中心"量测：对置板中点恰是圆心（误差 ~sigma/sqrt2），
    // 相邻板中点沿弦法向偏 R*cos45。半径已知时这块观测比单板位置强得多。
    bool pair_center_update_enabled{false};
    double pair_center_sigma_m{0.05};

    // 观测平滑：同一槽位连续命中时，对世界系位置做指数平均后再喂滤波器。
    // 单帧检测抖动（实测 v4 旋转目标相邻帧位移中位 52 mm、静止 3.8 mm）是
    // 白噪声，平均 N 帧能压到 1/sqrt(N)；换板/换 id/超时立即重置，避免把
    // 不同板的点混在一起（那会引入 0.4 m 假跳变）。
    bool observation_smoothing_enabled{false};
    int smoothing_window{4};            // 参与平均的最大帧数
    double smoothing_max_spread_m{0.12}; // 新观测离均值超过它 -> 重置而不是并入
    double smoothing_max_gap_s{0.10};    // 与上次命中的间隔超过它 -> 重置

    // 同帧两块板同时可见时，弦长能直接量出半径：R = chord / |n_i - n_j|
    // （相邻两板 |Δn| = √2）。静止目标只露一块板时 R 不可辨识，这条观测是
    // 唯一能让 R 脱离先验的来源。
    // 默认关闭：实测（多组 σ=0.02/0.06/50）对静止目标无改善（本来就 7.4 mm），
    // 对旋转目标反而变差（99 -> 125~140 mm），v1 只从 412 降到 380 mm。
    // 原因是锁定时已经用双板弦长定过 R（reinitializeFromPair），运行期再连续
    // 拉 R 会跟位置更新抢"中心该往哪挪"，旋转目标的 yaw 噪声把 R 拖偏。
    bool pair_radius_update_enabled{false};
    double radius_update_sigma_m{0.06};
    double radius_update_min_chord_m{0.08}; // 太近的两块板（法向几乎平行）弦长除噪会爆

    // 板面法向当"相位观测"塞进滤波器（折叠掉 2pi/N）。实拍小目标法向抖 0.05~0.15
    // rad，取 0.12；关掉它就退回"yaw 只靠位置间接约束"。
    // 默认关闭：合成 A 场景实测"打开后 yaw 抖动、半径被压到 0.18（真值 0.25）"，
    // 因为位置更新与相位更新在"哪个槽位"这件事上互相拉扯。能力保留、单测覆盖
    // RotatingArmorEkf::updatePhase 本身；等有"按槽位一致"的写法再默认打开。
    bool phase_update_enabled{false};
    // 静止目标（|omega| < static_rate_rad_s）且同一块板连续可见时，用板面法向
    // 约束车体 yaw。静止时法向没有旋转模糊，是干净量测；单板+钉死半径时
    // (center, yaw) 是规范自由度，不用法向相位就会自由漂移（实拍 v4 静止车
    // 1400 帧时相位偏 1.08 rad，另外三块模型板飞到车外）。
    bool phase_update_when_static{true};
    // 单板重锚：更新后状态里的板离观测还超过这个距离（相位漂了一整块板 / NIS
    // 门锁死）时，直接用这块板把相位+中心对齐。0 表示关闭。
    double single_plate_realign_m{0.0};
    // 重锚候选的最大距离：状态漂到位置门外时，只要还在这个范围内就当成"重新
    // 捕获同一块板"（跳过 NIS 门，先重锚再更新）。超过它说明观测根本不是这台车。
    double single_plate_realign_gate_m{1.5};
    double phase_update_sigma_rad{0.12};
    // 相位更新后是否"反锚"（把中心平移回去、让板留在原地）。单板可见时
    // (中心, 相位) 是规范自由度；但中心已经由位置更新约束住时，反锚会把
    // 相位修正变成中心漂移。两种做法都保留，A/B 用。
    bool phase_anti_anchor{true};
    int bootstrap_updates{10};   // 锁定期头几次更新先放宽 NIS 门（初值靠单帧法向，偏差可比预想大）
    // 声明反驳检查：法向历史里混入相邻板样本时斜率会失真，实拍里会误杀，
    // 默认关闭；需要时打开（合成夹具与 v4 的静止车都不需要它）。
    bool contradiction_check{false};
    int max_updates_per_frame{2}; // 同一帧最多用几块板更新（防止病态约束）

    // === 状态机 ===
    int stable_updates{12};          // 连续接受多少帧才允许报 Stable
    int unstable_after_missed{2};    // 仅供诊断：Stable 的降级判据已改为按秒（见 stale_after_s）
    double stale_after_s{0.10};      // 距上次被接收更新超过此秒数即判 Unstable（上限）
    // 自适应过期窗口：窗口 = min(stale_after_s, max(stale_min_s, stale_min_frames*dt))。
    // 166fps 下 0.10s 等于 17 帧，旋转目标早转过 90°；30fps 下又刚好 3 帧。
    // 用中位帧间隔把"秒"翻译成"帧"，让低帧率端不会过紧、高帧率端不会过松。
    double stale_min_frames{5.0};
    double stale_min_s{0.05};
    double timeout_s{0.60};          // 超过多久没有任何有效观测 -> 注销
    // 掉线快照续接：注销时把状态留一份，同一编号在 relock_reuse_s 内又出现，
    // 直接把旧状态装回去，省掉 10 个样本的重新收集。
    // 默认关闭：实测（v4 --only-id 3 / v1）续接能多救回 10% 的帧、把 Collecting
    // 从 319 帧压到 233 帧、p99 从 0.99 m 降到 0.58 m，但**中位残差变差**
    // （99 -> 115 mm），因为救回来的那批帧本身就是"拿 0.25 s 前的状态外推"，
    // 旋转目标在空窗里已经转过几十度。空窗越短越糟（0.10 s 时中位 1.35 m），
    // 说明这类帧不是"续一下就好"，而是必须重新锁定。
    bool relock_reuse_enabled{false};
    double relock_reuse_s{0.25};

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
    // 瞄准点只允许落在"最近真的被看见过"的板上：四块板里总有一两块朝向车体
    // 背面（几何上真实存在、但检测不到），按角度挑会挑到它们身上（实拍 v4 静止
    // 1384 帧全aim=3、看见的是 slot 0，瞄准点画到板外 0.35 m）。这个时间窗内
    // 见过的板才算候选，窗内的板里再挑离云台最近的。
    double aim_visible_hold_s{0.30};
    // 诊断用：强制锁定角速度（rad/s，NaN=不强制）。用来扫"真实角速度大概是多少"：
    // 在不同 forced omega 下看 30 ms 预测误差哪个最小，比从含跳变的序列里拟合更可靠。
    double forced_omega_rad_s{std::numeric_limits<double>::quiet_NaN()};
    // 瞄准点直接用"最近一次观测到的板心"的时间窗（秒）。0 = 关闭，退回 EKF。
    double aim_observation_hold_s{0.012};
    // 允许"新鲜观测外推"在任意预测时域下使用（默认只在零时域时用 EKF 四板模型）。
    // 实拍 v4 末段状态落后 0.19 m、而观测是 0~6 ms 前的真值，外推量只有 omega*30 ms。
    bool aim_observation_any_horizon{false};
    // 观测外推时是否按 omega 转动：omega 不可靠时（旋转模糊/换板）外推会把
    // 瞄准点推得比"原地不动"更远，关掉更稳。
    bool aim_observation_extrapolate{true};
    // 目标已经这么久没被观测到时，不再用状态外推（速度项会把瞄准点推着满场走），
    // 直接冻结在最后一次看到的那块板上。实拍 v4 有 100+ 帧的整段丢失，外推
    // 会让瞄准圈飘到另一台车上。
    double aim_freeze_after_s{0.30};
    // 瞄准点的一阶平滑系数（0 = 不平滑）。观测有 ~35 mm 的帧间抖动，直接送
    // 云台会抖；按 omega 做运动补偿后再做一阶平滑，不引入相位滞后。
    double aim_smoothing_alpha{0.0};
    // 瞄准点局部直线拟合的样本数与时间窗（0 = 关闭，直接用当帧观测）。
    int aim_fit_samples{0};
    double aim_fit_window_s{0.15};
    // 目标选择：自瞄应瞄"正压在准星上"的那台车。两台车同时在场时，另一台车的板
    // 连续 relock_frames 帧都比当前目标更靠近准星（多出 target_switch_margin_rad
    // 的迟滞），就丢目标重锁到它。
    bool prefer_centered_target{true};
    double target_switch_margin_rad{0.10};
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
    // 锁定时刻的速率估计细节（诊断"为什么判成不转/转多快"用）：
    double lock_rate_error_rad_s{0.0};
    bool lock_spin_from_positions{false}; // true = 速率来自板心位置圆拟合
    double lock_circle_radius_m{0.0};
    double lock_circle_residual_m{0.0};
    double lock_step_ratio{0.0};          // 有效（同板）步数占比
    bool radius_from_data{false};
    bool phase_flipped{false};
    std::uint64_t accepted_updates{0};
    int consecutive_updates{0};
    int missed_frames{0};
    double observation_noise_m{0.0}; // 本帧使用的观测噪声 sigma（m）
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
    void applyProcessNoise();

    // 独立相位量测（LED 灯条给出的板面法向）。内部按 phase_anti_anchor 决定
    // 是否把中心反锚、让瞄准板保持原位。
    bool applyPhaseMeasurement(double world_yaw_rad, double sigma_rad);


    // 与 ekf_predict 同签名：喂进本帧的全部装甲板位姿，返回本帧应该发出去的结果。
    // 没有目标、时间戳非法或跟踪失效时返回默认的 NoTarget 结果。
    // measured_latency_s：本帧从取到图到调用本函数已经花掉的时间（秒），
    // 会叠加到 prediction_bias_s 上一起作为预测时域。
    PredictionResult step(const std::vector<ArmorPose>& observations,
                          const GimbalState& gimbal, double timestamp_seconds,
                          double measured_latency_s = 0.0);

    bool initialized() const { return ekf_.initialized(); }

    // 自适应过期窗口：min(stale_after_s, max(stale_min_s, stale_min_frames * dt_median))。
    double staleWindowSeconds() const;
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
        // 板型（大/小装甲）。车体半径先验按它选：小装甲车 0.20 m、大装甲车
        // 0.25 m；没有板型信息时才退回 radius_prior_m。默认按小板处理（比赛里
        // 绝大多数是 3 号/4 号这种小装甲车）。
        ArmorSize size{ArmorSize::Small};
        double sigma_position_m{0.03};
    };

    struct LockSample
    {
        double timestamp_s{0.0};
        Eigen::Vector3d position{0.0, 0.0, 0.0};
        double yaw{0.0}; // 板面法向 yaw（世界系，mod pi 待定）
        int target_id{0};
        ArmorSize size{ArmorSize::Small}; // 板型（选半径先验用）
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
    static CircleFit fitCenterFixedRadius(const std::deque<LockSample>& samples, double radius,
                                          const CircleFit& initial);

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
    // 观测平滑（见 TrackerConfig::observation_smoothing_enabled）：返回 true 时
    // out 是平滑后的观测，否则调用方用原始观测。
    bool smoothObservation(const WorldObservation& observation, int slot,
                           double timestamp_s, ArmorObservation& out);
    struct ObservationSmoother
    {
        Eigen::Vector3d mean{Eigen::Vector3d::Zero()};
        int slot{-1};
        int target_id{-1};
        int count{0};
        double timestamp_s{0.0};
    };
    ObservationSmoother smoother_;
    // 观测噪声估计用的历史（只放被接受的观测，按时间升序）。
    struct NoiseSample { Eigen::Vector3d position; double timestamp_s; };
    std::deque<NoiseSample> noise_samples_;
    double observation_noise_m_{0.06};
    void noteObservationForNoise(const Eigen::Vector3d &position, double timestamp_s);
    double observationSigma(double detection_sigma_m) const;
    std::array<double, 8> last_seen_s_{{-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0}};
    // 每个槽位最近一次被接受到的世界系板心。零延迟时它就是"此刻板在哪"
    // 的最好答案；EKF 只负责给没有新鲜观测的槽位做外推。
    std::array<Eigen::Vector3d, 8> last_seen_position_{};
    // 每个槽位最近几次观测的(时间, 世界系板心)。瞄准点用"局部直线拟合"而不是
    // EKF 的 omega：实拍里 omega 常被判 0，靠它就等于纯滞后；直线拟合不需要
    // omega，30 ms 内圆周的曲率可以忽略。
    static constexpr int kAimHistory = 5;
    std::array<std::array<Eigen::Vector3d, kAimHistory>, 8> aim_history_{};
    std::array<std::array<double, kAimHistory>, 8> aim_history_s_{};
    std::array<int, 8> aim_history_count_{};
    Eigen::Vector3d aim_smoothed_world_{};
    double aim_smoothed_s_{-1.0};
    bool aim_smoothed_valid_{false};
    // 同帧双板弦长 -> 半径观测（见 .cpp 里的推导）。
    void updateRadiusFromPairs(const std::vector<WorldObservation> &world);
    // 掉线快照续接（见 config_.relock_reuse_enabled）。
    bool tryRestoreSnapshot(const std::vector<WorldObservation> &world, double timestamp_seconds);

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
    // 掉线快照（注销时留一份，短间隔内同编号再现就直接续上）。
    bool has_snapshot_{false};
    State snapshot_state_{State::Zero()};
    Covariance snapshot_covariance_{Covariance::Zero()};
    double snapshot_s_{0.0};
    int snapshot_id_{0};
    bool snapshot_phase_resolved_{false};
    bool snapshot_spin_pinned_{false};
    int consecutive_updates_{0};
    int missed_frames_{0};
    int other_id_streak_{0};
    int centered_streak_{0};
    // 上一次被接受更新的槽位：用来判断"同一块板连续可见"（换板时不能用板面法向
    // 约束车体 yaw）。
    int last_updated_slot_{-1};

    std::deque<LockSample> lock_samples_{};
    // 每个 slot 一份：(timestamp, 解缠后的法向 yaw)，用于估计观测到的角速度。
    // 必须按 slot 分开：同一辆车不同装甲板的法向相差 2*pi*k/N，
    // 混在一起会让斜率完全失真。
    std::vector<std::deque<std::pair<double, double>>> normal_history_per_slot_{};
    // 最近若干帧的间隔，只用来做"帧 <-> 秒"的换算（自适应过期窗口）。
    std::deque<double> recent_dt_{};
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
