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

using State = Eigen::Matrix<double, kStateDim, 1>;
using StateJacobian = Eigen::Matrix<double, kStateDim, kStateDim>;
using Covariance = Eigen::Matrix<double, kStateDim, kStateDim>;
using Measurement = Eigen::Matrix<double, kMeasurementDim, 1>;
using MeasurementJacobian = Eigen::Matrix<double, kMeasurementDim, kStateDim>;

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
    // 【2026-10-28 从 1e-6 放开】l/h 的信息只在双板帧进来（单板帧 H 的 9/10 列
    // 为零，见 Ekf::update 的 plates_visible 分支），redrotate 这类旋转素材双板
    // 帧占比低，1e-6 的随机游走（1 s 累计 σ=1 mm）等于把 l/h 永久钉在 0。
    // 代价是实打实的：真实车体前后对与左右对的半径差 1~3 cm，l=0 时这部分的
    // 位置残差（实测 5~9 cm）没有 l 这个出口，只能被 yaw/center 吸收——双板帧
    // 的第一帧 yaw 单帧跳 0.17~0.19 rad，就是"尖峰"的来源。
    // 1e-4 对应 1 s σ=1 cm、10 s σ=3.2 cm，够在几秒内跟上装配量级的 l/h，
    // 又不会在长单板段里漂没谱（EKF update 侧另有 ±kPairOffsetMaxM 限幅）。
    double pair_radius_random_walk{1e-4}; // (m^2/s) radius_diff 的随机游走
    double pair_height_random_walk{1e-4}; // (m^2/s) height_diff 的随机游走
};

// Wrap radians into [-pi, pi).
double wrapAngle(double angle);

// 把角度折到一个周期内 [-period/2, period/2)。四板对称下 period = pi/2：
// 换到相邻那块板差一个整周期，折叠后残差为 0，这正是"板面法向能直接约束
// 车体 yaw"的数学形式。（2026-10-27 起这是唯一定义；tracker / ekf 各自的
// 匿名命名空间副本已并入这里。）
double foldPeriod(double value, double period);

// Constant-velocity translation + constant angular velocity model.
State transition(const State& x, double dt);
StateJacobian transitionJacobian(double dt);

// Predicted position of one armor plate, in the same tracking frame as cx/cy/z.
// The slot is a geometric slot index, NOT ArmorDetection::target_id.
Measurement measurement(const State& x, int plate_slot, int plate_count);
MeasurementJacobian measurementJacobian(
    const State& x, int plate_slot, int plate_count);

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
