#include "AimGeometry.hpp"

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

double wrapToPi(double angle)
{
    if (!std::isfinite(angle))
    {
        return angle;
    }
    constexpr double kTwoPi = 6.283185307179586476925286766559;
    double wrapped = std::fmod(angle + 3.14159265358979323846, kTwoPi);
    if (wrapped < 0.0)
    {
        wrapped += kTwoPi;
    }
    return wrapped - 3.14159265358979323846;
}

bool isFiniteVector(const Eigen::Vector3d &v)
{
    return v.allFinite();
}

} // namespace rotating_armor
