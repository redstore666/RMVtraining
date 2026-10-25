#pragma once

#include <Eigen/Dense>

namespace rotating_armor {

// State: [cx, vx, cy, vy, z, vz, yaw, yaw_rate, radius, radius_diff, height_diff]^T
//
// radius_diff / height_diff 借鉴 sp_vision_25 的 l / h：
// 四块装甲板并不共圆、也未必共高——前后两对板比左右两对离车心更远（车体前后长），
// 高度也常差一点。只用一个半径去画四块板，必然有一对系统性偏内/偏外，
// 这正是"绿板转心偏高、蓝板转半圈卡回去"的几何来源。
//
// 所以：偶数槽位用 radius，奇数槽位用 radius + radius_diff，高度同理。
// 只增加 2 个参数就表达了这个非对称性，而且它们是**刚体参数**（不随槽位震荡），
// 不会像 per-slot 自由偏移那样在两块同 id 板交替时来回抖。
// 板数不为 4（双板平衡车、三板前哨站）时不启用该修正。
constexpr int kStateDim = 11;
constexpr int kMeasurementDim = 3;

// 球坐标量测量 (yaw, pitch, distance) + 板面法向角，共 4 维。
// 与位置量测并存：位置量测给上层做几何/显示，球坐标量测供 update() 使用。
constexpr int kSphericalMeasurementDim = 4;

using State = Eigen::Matrix<double, kStateDim, 1>;
using StateJacobian = Eigen::Matrix<double, kStateDim, kStateDim>;
using Covariance = Eigen::Matrix<double, kStateDim, kStateDim>;
using Measurement = Eigen::Matrix<double, kMeasurementDim, 1>;
using MeasurementJacobian = Eigen::Matrix<double, kMeasurementDim, kStateDim>;
using SphericalMeasurement = Eigen::Matrix<double, kSphericalMeasurementDim, 1>;
using SphericalJacobian = Eigen::Matrix<double, kSphericalMeasurementDim, kStateDim>;

// Continuous-time white-noise spectral densities.
//
// 量纲与换算：白加速度模型下，一个步长 dt 内噪声注入的速度不确定度约为
//     sigma_v ≈ sqrt(q * dt)   ，角速度不确定度约为 sigma_omega ≈ sqrt(q * dt)。
// 所以同一组 q 在不同帧率下含义不同：60fps（dt ≈ 0.0167s）时
//     q = 0.01  ->  sigma_v ≈ 0.013 m/s、sigma_omega ≈ 0.013 rad/s（每帧），
//     1s 随机游走累积约 0.1 m/s、0.1 rad/s——与"匀速平移 + 匀角速度"的假设相称。
//
// 为什么默认值不能再大：历史默认值 accel=1.0 / angular=4.0 / radius_rw=1e-4 在
// 60fps 下等于每帧允许 sigma_v ≈ 0.13 m/s、sigma_omega ≈ 0.26 rad/s 的跳变，
// 合成真值测试里表现为"单帧拟合残差只有 8mm，但车体中心错 25cm、yaw 错 1.5rad"
// ——即滤波器用状态乱跳去换取残差，估计值不再可解释。回归护栏见
// tests/test_rotating_armor_convergence.cpp 的默认值断言与默认 Q 收敛断言。
//
// 当前默认值是合成真值测试验证过的量级，不是实车调参结果；实车应在
// "预测滞后 vs 估计抖动"之间重新权衡（本步不做扫参）。
struct ProcessNoiseConfig {
    double accel_x{0.01};      // (m/s^2)^2 / Hz
    double accel_y{0.01};
    double accel_z{0.01};
    double angular_accel{0.01}; // (rad/s^2)^2 / Hz
    // （移植实验记录 2026-10-25：直接对齐 sp_vision_25 v2=400 会破坏合成收敛
    // 契约 dyaw=0.58/d_omega=-3.08——历史默认 4.0 也同样失败并被锁进
    // ekf_model_convergence 断言。w 的拉拽强度需要与观测质量匹配，见
    // docs/spvision25转速跟随机制报告_2026-10-25.md §四的照抄陷阱。）
    double radius_random_walk{1e-6}; // m^2/s
    double pair_radius_random_walk{1e-6}; // (m^2/s) radius_diff 的随机游走
    double pair_height_random_walk{1e-6}; // (m^2/s) height_diff 的随机游走
};

// Wrap radians into [-pi, pi).
double wrapAngle(double angle);

// Constant-velocity translation + constant angular velocity model.
State transition(const State& x, double dt);
StateJacobian transitionJacobian(double dt);

// Predicted position of one armor plate, in the same tracking frame as cx/cy/z.
// The slot is a geometric slot index, NOT ArmorDetection::target_id.
Measurement measurement(const State& x, int plate_slot, int plate_count);
MeasurementJacobian measurementJacobian(
    const State& x, int plate_slot, int plate_count);

// 同一块板在同样的槽位约定下的球坐标量测 (yaw, pitch, distance, 板法向角)。
//
// 为什么用球坐标而不是直接量板心 xyz（借鉴 sp_vision_25 target.cpp:192-222）：
//   * 板心 xyz 的深度误差会沿着视线方向线性污染横向，误差结构不对；
//   * yaw/pitch 只和像素方向有关，深度误差基本不进这两维；
//   * distance 单独一维，可以把"测距不准"用一个大方差表达，而不是让它
//     同时污染 x、y、z 三个残差。
// 第 4 维是板面法向角（世界系 yaw）。四块板法向相差 90°，单看它是折叠的，
// 但配合槽位就唯一确定"这一帧看到的是哪块板"，正是关联与相位需要的量。
// has_yaw == false 时第 4 维整行置零（调用方负责把该维方差放大到几乎无穷）。
SphericalMeasurement sphericalMeasurement(const State& x, int plate_slot, int plate_count,
                                           bool has_yaw);
SphericalJacobian sphericalJacobian(const State& x, int plate_slot, int plate_count,
                                    bool has_yaw);

// Process covariance for white-acceleration noise on the (position, velocity)
// pairs and a random-walk model for the armor radius.
Covariance processNoise(double dt, const ProcessNoiseConfig& config = {});

// 板面法向与相位的关系（EKF 初始化用）。
//
// measurement() 把装甲板中心落在 center - r * [cos(phase), sin(phase)]：从车体
// 中心指向这块板的径向单位向量是 -[cos(phase), sin(phase)]，而装甲板朝外
// （法向就是该径向方向），于是
//     外法向角 = phase + pi
// 反过来由板面法向求相位：phase = worldYawToPlatePhase(world_yaw)。
//
// 注意：外法向依赖 PnP 物点里 (0,0,1) 的朝向。实拍粘合时若发现检测给出的
// 法向其实是朝内的，只需改这两个函数里 pi 的符号，EKF 的公式不用动。
double platePhaseToWorldYaw(double phase);
double worldYawToPlatePhase(double world_yaw);

} // namespace rotating_armor
