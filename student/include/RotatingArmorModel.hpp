#pragma once

#include <Eigen/Dense>

namespace rotating_armor {

// State: [cx, vx, cy, vy, z, vz, yaw, yaw_rate, radius]^T
constexpr int kStateDim = 9;
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
    double radius_random_walk{1e-6}; // m^2/s
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
