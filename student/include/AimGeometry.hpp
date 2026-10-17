#pragma once

// 云台系 <-> 跟踪系（世界系）的坐标变换（任务 4）。
//
// 坐标约定（与 armor/include/BulletModel.hpp 的注释一致）：前 x / 左 y / 上 z。
//   * yaw 绕 z 轴，逆时针（向左转）为正；
//   * pitch > 0 表示抬头；
//   * roll 绕光轴（摄像机 x 轴）。
//
// 变换：p_world = Rz(yaw) * Ry(-pitch) * Rx(roll) * p_gimbal
//   yaw=pitch=roll=0 时两系重合；云台抬头 p 时，光轴 (1,0,0) 在世界系变成
//   (cos p, 0, sin p)。跟踪系的原点固定在云台初始位置（云台只转不移），
//   所以"世界系"其实就是"yaw=0 时的云台系"，不含机器人平移。
//
// 约定风险与校验方法：电控下发的 pitch 正方向若与我们相反（有的队 0=水平、
// 向下为正），只需把 kPitchPositiveIsUp 改成 false，其它代码不用动。实车上
// 用"抬头看静止目标，目标的 z 应变大（变高）"这一个用例即可确认。同一份约定
// 的单元测试在 tests/test_aim_geometry.cpp。

#include "AutoAimTypes.hpp"

#include <Eigen/Core>

namespace rotating_armor
{

constexpr bool kPitchPositiveIsUp = true;

// 云台系 -> 世界系的旋转矩阵。
Eigen::Matrix3d gimbalToWorld(const GimbalState &gimbal);
// 世界系 -> 云台系的旋转矩阵（即前者的转置）。
Eigen::Matrix3d worldToGimbal(const GimbalState &gimbal);

Eigen::Vector3d toWorld(const GimbalState &gimbal, const Eigen::Vector3d &p_gimbal);
Eigen::Vector3d toGimbal(const GimbalState &gimbal, const Eigen::Vector3d &p_world);

// 向量的方位角 atan2(y, x)，与云台 yaw 同一体系。
double vectorYaw(const Eigen::Vector3d &v);
// 向量的俯仰角 atan2(z, sqrt(x^2 + y^2))，与云台 pitch 同一体系。
double vectorPitch(const Eigen::Vector3d &v);

// 归一到 [-pi, pi)。
double wrapToPi(double angle);
bool isFiniteVector(const Eigen::Vector3d &v);

} // namespace rotating_armor
