// 任务 4 的云台/世界几何层（AimGeometry）单元测试：
// 坐标约定（前 x、左 y、上 z；yaw 逆时针；pitch 抬头为正）、正交性、往返一致性、
// 极角定义。这些约定一旦被改动，EKF 的输出语义就全错，必须钉住。
#include "AimGeometry.hpp"
#include "RotatingArmorModel.hpp"

#include <cmath>
#include <cstdio>

namespace
{
int g_failures = 0;

void check(bool condition, const char *what)
{
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
    {
        ++g_failures;
    }
}

bool near(double a, double b, double eps)
{
    return std::fabs(a - b) <= eps;
}

} // namespace

int main()
{
    using namespace rotating_armor;
    constexpr double kPi = 3.14159265358979323846;

    // wrapToPi
    check(near(wrapToPi(0.0), 0.0, 1e-12), "wrapToPi(0)");
    check(near(wrapToPi(4.0), 4.0 - 2.0 * kPi, 1e-12), "wrapToPi(4)");
    check(near(wrapToPi(-4.0), -4.0 + 2.0 * kPi, 1e-12), "wrapToPi(-4)");
    check(near(wrapToPi(3.0 * kPi), kPi, 1e-12) || near(wrapToPi(3.0 * kPi), -kPi, 1e-12),
          "wrapToPi(3pi) 落在 +-pi");

    // pitch 为正时抬头：云台系 (1,0,0) 应转到世界系 z>0。
    GimbalState gimbal{};
    gimbal.yaw = 0.0F;
    gimbal.pitch = 0.3F;
    gimbal.roll = 0.0F;
    Eigen::Vector3d forward = gimbalToWorld(gimbal) * Eigen::Vector3d(1.0, 0.0, 0.0);
    check(near(forward.x(), std::cos(0.3), 1e-6) && near(forward.z(), std::sin(0.3), 1e-6),
          "pitch>0 表示抬头");

    // yaw 为正时向左转：云台系 (1,0,0) 应转到世界系 y>0。
    gimbal.pitch = 0.0F;
    gimbal.yaw = 0.5F;
    forward = gimbalToWorld(gimbal) * Eigen::Vector3d(1.0, 0.0, 0.0);
    check(near(forward.x(), std::cos(0.5), 1e-6) && near(forward.y(), std::sin(0.5), 1e-6),
          "yaw>0 表示向左转");

    // 正交性与往返一致性
    gimbal.yaw = 0.7F;
    gimbal.pitch = -0.2F;
    gimbal.roll = 0.15F;
    const Eigen::Matrix3d rotation = gimbalToWorld(gimbal);
    check((rotation * rotation.transpose() - Eigen::Matrix3d::Identity()).norm() < 1e-12,
          "旋转矩阵正交");
    check(std::fabs(rotation.determinant() - 1.0) < 1e-12, "旋转矩阵行列式为 1");
    const Eigen::Vector3d point(2.0, -0.5, 0.3);
    check((toGimbal(gimbal, toWorld(gimbal, point)) - point).norm() < 1e-12,
          "云台系/世界系往返一致");

    // 极角定义：与 PredictionResult 的 target_yaw / target_pitch 同体系。
    check(near(vectorYaw(Eigen::Vector3d(1.0, 0.0, 0.0)), 0.0, 1e-12), "vectorYaw 正前方=0");
    check(near(vectorYaw(Eigen::Vector3d(0.0, 1.0, 0.0)), kPi / 2.0, 1e-12), "vectorYaw 左侧=+pi/2");
    check(near(vectorYaw(Eigen::Vector3d(0.0, -1.0, 0.0)), -kPi / 2.0, 1e-12),
          "vectorYaw 右侧=-pi/2");
    check(near(vectorPitch(Eigen::Vector3d(3.0, 0.0, 4.0)), std::atan2(4.0, 3.0), 1e-12),
          "vectorPitch 抬头为正");

    // 板面法向 yaw 与模型相位的关系：worldYawToPlatePhase 与 platePhaseToWorldYaw 互逆。
    for (double phase : {-3.0, -1.0, 0.0, 0.5, 2.0, 3.0})
    {
        const double yaw = platePhaseToWorldYaw(phase);
        check(near(worldYawToPlatePhase(yaw), wrapToPi(phase), 1e-12),
              "相位/法向 yaw 互逆");
    }

    if (g_failures != 0)
    {
        std::printf("\n%d 项失败\n", g_failures);
        return 1;
    }
    std::printf("\n全部通过\n");
    return 0;
}
