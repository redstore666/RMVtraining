#pragma once

// 旋转车体目标的生产 EKF（任务 4 第 2 步）。
//
// 职责边界（刻意收窄）：
//   * 本类只做"给定观测 + 给定 slot"的 predict/update，以及首帧锁定所需的
//     最小初始化契约；
//   * 数据关联、NIS 门限、超时注销、状态机、PredictionResult 输出映射、
//     与 ekf_predict 的粘合都不在这里，留给下一步。
//
// 数学真源：全部复用 rotating_armor:: 的自由函数（transition /
// transitionJacobian / measurement / measurementJacobian / processNoise），
// 本类不复制任何一份公式，只负责状态、协方差、生命周期与输入校验。

#include "RotatingArmorModel.hpp"

#include <cstdint>

namespace rotating_armor {

// 一帧观测量：装甲板中心在跟踪系（前 x / 左 y / 上 z）的 3D 位置。
//
// world_yaw_rad 是这块板的板面法向在世界系里的 yaw（已由粘合层完成
// camera -> gimbal -> world 的变换）。它**只在 initialize() 里**用来定初相：
// 把法向也做成逐帧观测量等于新增第二套观测模型（另一份 H 与 R），
// 属于后续步骤，本步不做。
struct ArmorObservation {
    Measurement position{Measurement::Zero()};
    double sigma_position_m{0.01};
    bool has_world_yaw{false};
    double world_yaw_rad{0.0};
};

// 角速度先验的声明方式。
//
// 为什么必须显式声明、而不能让 yaw_rate_prior_rad_s 默认 0：
// 单块装甲板的 3D 位置观测无法把"车体中心 / 半径 / 相位"与"相位随时间线性
// 变化"分开，角速度只能从相位序列的斜率里学。若初值 omega=0 又只给一个
// 零均值、宽方差的先验，滤波器会把"相位在转"解释成"中心+半径偏了一点"，
// 锁进一个板残差只有 1.5~2cm、但车体状态全错的极小值（末帧中心差 0.3~0.43m、
// yaw 差 1.7~2.5rad）。8 个噪声种子里有 3 个会掉进去，所以它不是"偶发数值问题"，
// 而是"靠默认值启动"这件事本身不可靠。
//
// 因此 omega=0 只能作为调用方**显式承担**的假设出现（Static），并且这个假设
// 必须可被证伪：Static 会把 omega 的初始方差与过程噪声一起钉成 0（硬约束），
// 若目标其实在转，板残差会变大（而不是悄悄拟合出一个错误的中心）。
//
// 反过来也一样：**静止目标的 omega=0 无法从数据里学出来**。对静止的板，
// "yaw 缓慢漂移 + 中心同步平移"是模型的近零空间方向，两者的板位置几乎不变，
// 数据区分不开。所以"转不转"这个分叉只能由调用方声明，滤波器不会自己选。
enum class SpinPriorKind : std::uint8_t {
    Unspecified = 0, // 没表态 -> initialize 拒绝，禁止 omega=0 静默冷启动
    Static,          // 显式声明目标不转：要求 yaw_rate_prior_rad_s == 0（硬约束：
                     // omega 的初始方差与过程噪声均为 0，目标开始转时必须重新锁定）
    Spinning,        // 显式声明目标在转：要求 yaw_rate_prior_rad_s != 0（含符号）
};

// 初始化先验。
//
// radius_m 必须显式给出且 > 0：单帧观测只能给出装甲板中心，而中心 = 板中心
// + r * [cos(phase), sin(phase)]，相位、半径、车体中心三者互相耦合，
// 单帧无法同时定出，所以半径必须先验（由大小装甲板尺寸 + 车体结构给出），
// 否则初值只能靠猜。
struct EkfInitPrior {
    double radius_m{0.0};              // 必需，单位 m
    int plate_slot{0};                 // 本帧观测对应的几何 slot（关联由调用方保证）
    SpinPriorKind spin{SpinPriorKind::Unspecified};
    double yaw_rate_prior_rad_s{0.0};  // spin == Spinning 时必须非零（rad/s，含符号）
    double position_sigma_m{0.5};
    double velocity_sigma_m_s{1.0};
    double yaw_sigma_rad{0.2};         // 仅"有板面法向"时生效；无相位先验时强制为 pi
    double yaw_rate_sigma_rad_s{3.0};  // spin == Spinning 时生效；Static 时强制为 0
    double radius_sigma_m{0.05};
};

enum class EkfInitStatus : std::uint8_t {
    Ok = 0,
    InvalidRadiusPrior,  // radius_m <= 0 或非有限
    InvalidObservation,  // 观测/噪声参数非有限，或 sigma_position_m <= 0
    InvalidPlateSlot,    // slot 不在 [0, plate_count)
    // spin == Unspecified：调用方没声明目标转不转。拒绝，而不是替它假设 omega=0。
    SpinPriorUnspecified,
    // spin == Spinning 但 yaw_rate_prior_rad_s == 0 或非有限；
    // 或 spin == Static 却给了非零角速度（自相矛盾的声明）。
    InvalidSpinPrior,
};

const char* toString(EkfInitStatus status) noexcept;

struct EkfConfig {
    // 默认值即 RotatingArmorModel.hpp 里那组经过收敛测试的量级。
    ProcessNoiseConfig process{};
    int plate_count{4};
};

class RotatingArmorEkf {
public:
    RotatingArmorEkf() = default;
    explicit RotatingArmorEkf(const EkfConfig& config);

    // 回到"未初始化"：状态与协方差清零，需要重新 initialize()。
    void reset();

    // 掉线快照续接：把掉线前一刻的状态/协方差原样装回去（跳过重新收集样本）。
    // 只用于"同一编号、间隔很短又出现"的情形，由上层判定是否该续接。
    bool restore(const State &state, const Covariance &covariance, bool phase_resolved,
                 bool spin_pinned);

    bool initialized() const { return initialized_; }

    // true 表示初相由板面法向确定，此后 yaw 的含义立即可以解释；
    // false 表示走的是降级路径（相位靠 omega 先验 + 后续观测收敛），
    // 此时 yaw 需要多帧才可信，调用方不应据此直接开火。
    bool phase_resolved() const { return phase_resolved_; }
    bool spinPinned() const { return spin_pinned_; }

    // 首帧锁定。成功时状态正好与这一帧观测一致（残差为 0），
    // 因此不需要再补一次 update()。失败时不改变滤波器状态。
    EkfInitStatus initialize(const ArmorObservation& observation, const EkfInitPrior& prior);

    // 时间推进。未初始化、dt 非有限或 dt < 0 时返回 false 并保持状态不变
    // （模型层对非法 dt 抛异常，实时循环里不希望看到异常）；dt == 0 是
    // 合法的空推进（同一时间戳的重复帧），返回 true。
    bool predict(double dt);

    // 用一帧观测做校正。slot 由调用方给出（本步不做关联）。
    // 输入非法、未初始化、或内部出现非有限值时返回 false；后者会同时 reset()，
    // 让调用方重新锁定，而不是继续把 NaN 传播出去。
    bool update(const ArmorObservation& observation, int plate_slot);

    // 用"板面法向的世界朝角"直接观测车体 yaw 的一步更新。
    //
    // 四块板的法向相差 2pi/N，所以 yaw 只能确定到 2pi/N（4 板时 90°）；
    // 这条观测就是把该周期折叠掉的残差 h = fold(theta - psi, 2pi/N) 送进滤波器。
    // 好处：不依赖"这一帧看到的是哪一块板"——换板时折叠加起来永远是 0，
    // 所以位置关联出错也不会把相位带歪。sigma_rad 取值见上层配置。
    bool updatePhase(double measured_world_yaw, double sigma_rad);

    // 用"同一帧两块板的弦长"直接观测半径：相邻两板 |n_i-n_j| = sqrt(2)，
    // 弦长 = R*|n_i-n_j|，所以 R = chord / |n_i-n_j|，不需要圆拟合。
    // 静止目标只露一块板时 R 不可辨识（中心会跟着 R 平移，观测依然自洽），
    // 这条观测是"两块板同时可见"时唯一的 R 来源。
    bool updateRadius(double measured_radius, double sigma_radius);

    // 该观测在当前先验下的新息平方和（马氏距离）NIS = r' S^-1 r，S = H P H' + R。
    // 上层用它做离群点拒绝：3 维观测下 NIS ~ chi2(3)，99.5% 分位约 11.34。
    // 未初始化、slot 非法或输入非有限时返回 false。
    bool innovationNis(const ArmorObservation& observation, int plate_slot,
                       double& nis) const;

    const State& state() const { return state_; }

    // 直接平移车体中心：用于"相位更新后把中心反锚回去"。
    // 单块板可见时 (center, yaw) 有一个规范自由度（绕着板心转），
    // 只改 yaw 会把板挪走、再被位置更新拉回来，两个更新互相打架；
    // 改完 yaw 把中心平移回去、让那块板留在原地，等价于"沿可观方向更新"。
    void shiftCenter(const Eigen::Vector3d& delta) { state_.head<3>() += delta; }
    const Covariance& covariance() const { return covariance_; }
    const EkfConfig& config() const { return config_; }

private:
    EkfConfig config_{};
    State state_{State::Zero()};
    Covariance covariance_{Covariance::Zero()};
    bool initialized_{false};
    bool phase_resolved_{false};
    // spin == Static 时置位：角速度的初始方差与过程噪声都被钉成 0，使它成为硬约束。
    bool spin_pinned_{false};
};

} // namespace rotating_armor
