#include "AimGeometry.hpp"

#include "RotatingArmorModel.hpp" // wrapAngle（wrapToPi 的唯一实现）

#include <Eigen/Geometry>

#include <cmath>
#include <limits>

namespace rotating_armor
{

Eigen::Matrix3d gimbalToWorld(const GimbalState &gimbal)
{
    const double pitch = kPitchPositiveIsUp ? static_cast<double>(gimbal.pitch)
                                            : -static_cast<double>(gimbal.pitch);
    const Eigen::AngleAxisd yaw_rotation(static_cast<double>(gimbal.yaw),
                                         Eigen::Vector3d::UnitZ());
    // 抬头时世界系 z 增大，对应 Ry(-pitch)，见头文件注释里的推导。
    const Eigen::AngleAxisd pitch_rotation(-pitch, Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd roll_rotation(static_cast<double>(gimbal.roll),
                                          Eigen::Vector3d::UnitX());
    return (yaw_rotation * pitch_rotation * roll_rotation).toRotationMatrix();
}

Eigen::Matrix3d worldToGimbal(const GimbalState &gimbal)
{
    return gimbalToWorld(gimbal).transpose();
}

Eigen::Vector3d toWorld(const GimbalState &gimbal, const Eigen::Vector3d &p_gimbal)
{
    return gimbalToWorld(gimbal) * p_gimbal;
}

Eigen::Vector3d toGimbal(const GimbalState &gimbal, const Eigen::Vector3d &p_world)
{
    return worldToGimbal(gimbal) * p_world;
}

double vectorYaw(const Eigen::Vector3d &v)
{
    return std::atan2(v.y(), v.x());
}

double vectorPitch(const Eigen::Vector3d &v)
{
    return std::atan2(v.z(), std::hypot(v.x(), v.y()));
}

// 与 RotatingArmorModel::wrapAngle 是同一个数学运算（2026-10-27 起直接委托，
// 消掉第二份实现；保留本名是因为 AimGeometry/StudentTasks 的调用点都用它）。
double wrapToPi(double angle)
{
    return wrapAngle(angle);
}

bool isFiniteVector(const Eigen::Vector3d &v)
{
    return v.allFinite();
}

} // namespace rotating_armor
