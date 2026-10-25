#pragma once

// 任务 4 粘合层的合成真值夹具：把"世界系里的旋转车体"变成一帧 ArmorPose 列表，
// 再喂给 rotating_armor::RotatingArmorTracker。测试与管线测试共用。
//
// 关键约定（与 AimGeometry.hpp 一致）：
//   世界系/云台系：前 x、左 y、上 z，yaw 逆时针为正，pitch 抬头为正。
//   板心 = 车体中心 - R*[cos(phase), sin(phase)]，板面外法向 = -[cos(phase), sin(phase)]，
//   其 yaw = phase + pi（与 worldYawToPlatePhase 互逆）。

#include "AimGeometry.hpp"
#include "AutoAimTypes.hpp"
#include "RotatingArmorModel.hpp"
#include "RotatingArmorTracker.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace sim
{

constexpr double kPi = 3.141592653589793238462643383279502884;

inline double gauss(std::mt19937 &rng, double sigma)
{
    if (sigma <= 0.0)
    {
        return 0.0;
    }
    std::normal_distribution<double> distribution(0.0, sigma);
    return distribution(rng);
}

struct Plate
{
    Eigen::Vector3d position{0.0, 0.0, 0.0};
    Eigen::Vector3d normal{0.0, 0.0, 0.0};
    double phase{0.0};
    int slot{0};
};

inline std::vector<Plate> platesAt(const rotating_armor::State &truth, int plate_count)
{
    std::vector<Plate> plates;
    if (plate_count <= 0)
    {
        return plates;
    }
    for (int slot = 0; slot < plate_count; ++slot)
    {
        Plate plate;
        plate.slot = slot;
        plate.phase =
            rotating_armor::wrapToPi(truth(6) + 2.0 * kPi * static_cast<double>(slot) /
                                                  static_cast<double>(plate_count));
        plate.position = rotating_armor::measurement(truth, slot, plate_count);
        plate.normal = Eigen::Vector3d(std::cos(plate.phase + kPi),
                                       std::sin(plate.phase + kPi), 0.0);
        plates.push_back(plate);
    }
    return plates;
}

// 用云台姿态把世界系里的板转回云台系，并按相机视场/正反面裁剪。
// 只保留真的可能被检测到的板，测试才算诚实。
inline std::vector<ArmorPose> observe(const std::vector<Plate> &plates,
                                      const GimbalState &gimbal, int target_id,
                                      std::mt19937 &rng, double sigma_position_m = 0.01,
                                      double sigma_normal_rad = 0.02,
                                      double horizontal_fov_rad = 0.55,
                                      double vertical_fov_rad = 0.45)
{
    const Eigen::Matrix3d world_to_gimbal = rotating_armor::worldToGimbal(gimbal);
    std::vector<ArmorPose> poses;
    for (const Plate &plate : plates)
    {
        const Eigen::Vector3d p_gimbal = world_to_gimbal * plate.position;
        if (!(p_gimbal.x() > 0.4))
        {
            continue;
        }
        if (std::fabs(std::atan2(p_gimbal.y(), p_gimbal.x())) > horizontal_fov_rad)
        {
            continue;
        }
        if (std::fabs(std::atan2(p_gimbal.z(), std::hypot(p_gimbal.x(), p_gimbal.y()))) >
            vertical_fov_rad)
        {
            continue;
        }
        const Eigen::Vector3d n_gimbal = world_to_gimbal * plate.normal;
        // 只有正面朝相机的板能被灯条检测到。
        if (n_gimbal.dot(-p_gimbal.normalized()) < 0.35)
        {
            continue;
        }
        const Eigen::Vector3d noisy =
            p_gimbal + Eigen::Vector3d(gauss(rng, sigma_position_m), gauss(rng, sigma_position_m),
                                       gauss(rng, sigma_position_m));
        const double normal_yaw =
            std::atan2(n_gimbal.y(), n_gimbal.x()) + gauss(rng, sigma_normal_rad);
        const double normal_pitch =
            std::atan2(n_gimbal.z(), std::hypot(n_gimbal.x(), n_gimbal.y())) +
            gauss(rng, sigma_normal_rad);

        ArmorPose pose;
        pose.detection.target_id = target_id;
        pose.detection.size = ArmorSize::Small;
        pose.detection.confidence = 0.9F;
        pose.reprojection_error = std::fabs(gauss(rng, 0.4)) + 1.2;
        // 测试里没有真实相机：把云台系直接当作相机系（外参为单位阵）。
        pose.position_camera_m = cv::Vec3d(noisy.x(), noisy.y(), noisy.z());
        pose.position_gimbal_m = cv::Vec3d(noisy.x(), noisy.y(), noisy.z());
        pose.normal_gimbal = cv::Vec3d(std::cos(normal_yaw) * std::cos(normal_pitch),
                                       std::sin(normal_yaw) * std::cos(normal_pitch),
                                       std::sin(normal_pitch));
        pose.armor_yaw = normal_yaw;
        poses.push_back(pose);
    }
    return poses;
}

// 世界系里的极坐标点（供"把云台指向某个板/某个预测角"使用）。
inline Eigen::Vector3d polarPoint(double yaw, double pitch, double distance)
{
    return Eigen::Vector3d(distance * std::cos(pitch) * std::cos(yaw),
                           distance * std::cos(pitch) * std::sin(yaw),
                           distance * std::sin(pitch));
}

struct GimbalAim
{
    double yaw{0.0};
    double pitch{0.0};
    double roll{0.0};
    double rate_rad_s{6.0};

    GimbalState state(double bias_s = 0.0) const
    {
        GimbalState gimbal;
        gimbal.yaw = static_cast<float>(yaw);
        gimbal.pitch = static_cast<float>(pitch);
        gimbal.roll = static_cast<float>(roll);
        gimbal.prediction_bias_s = static_cast<float>(bias_s);
        return gimbal;
    }

    void snapTo(const Eigen::Vector3d &world_point)
    {
        yaw = rotating_armor::vectorYaw(world_point);
        pitch = rotating_armor::vectorPitch(world_point);
    }

    void chase(const Eigen::Vector3d &world_point, double dt)
    {
        const double want_yaw = rotating_armor::vectorYaw(world_point);
        const double want_pitch = rotating_armor::vectorPitch(world_point);
        const double limit = rate_rad_s * dt;
        yaw = rotating_armor::wrapToPi(
            yaw + std::clamp(rotating_armor::wrapToPi(want_yaw - yaw), -limit, limit));
        pitch += std::clamp(want_pitch - pitch, -limit, limit);
    }
};

// 一帧的完整仿真：推进真值 -> （可选）云台追板 -> 生成观测 -> 调跟踪器。
struct Rig
{
    rotating_armor::State truth{};
    rotating_armor::RotatingArmorTracker tracker{};
    GimbalAim gimbal{};
    std::mt19937 rng{20240607U};
    double time_s{0.0};
    double dt{1.0 / 60.0};
    int plate_count{4};
    int target_id{7};
    double bias_s{0.0};
    double sigma_position_m{0.01};
    // 板面法向噪声：实拍小目标（v4 的 0.2 m 板、1.5 m 距离）实测抖到 ±20°，
    // 默认 0.02 rad 只适合合成场景。转速自我判别的回归测试会把它调大。
    double sigma_normal_rad{0.02};
    // 云台姿态"报告值"和"真实值"的差（rad，每帧独立抽样）。实拍里手持抖动就是
    // 这个：观测是在真实姿态下拍的，而 EKF 拿到的是云台报告姿态，两者差一个
    // 列向量旋转——v1 的"目标在世界系里乱晃"就是这么来的。0 表示理想 IMU。
    double shake_sigma_rad{0.0};
    // 传给跟踪器的"本帧视觉处理延迟"，用于验证预测时域包含它。
    double latency_s{0.0};
    // 最近一帧生成的观测（跟踪器已经消费过）。给俯视图之类的可视化用，
    // 这样它们不必重造一遍"真值 -> 观测"的流程。
    std::vector<ArmorPose> last_observations{};

    // 合成场景要验证"半径从坏初值收敛"，所以默认关掉实时跟踪用的半径钉死。
    static rotating_armor::TrackerConfig defaultConfig()
    {
        rotating_armor::TrackerConfig config;
        config.freeze_radius = false;
        return config;
    }

    explicit Rig(const rotating_armor::State &initial_truth,
                 rotating_armor::TrackerConfig config = defaultConfig())
        : truth(initial_truth), tracker(config)
    {
    }

    // aim_at_truth=true 时云台每帧朝向真值里离当前指向最近的那块板（模拟完美跟瞄）。
    PredictionResult step(bool provide_observations = true, bool aim_at_truth = false,
                          int observation_id = -1)
    {
        time_s += dt;
        truth = rotating_armor::transition(truth, dt);
        const std::vector<Plate> plates = platesAt(truth, plate_count);
        if (aim_at_truth && !plates.empty())
        {
            // 云台只能追"朝向自己这一面"的板：背面的板被车体挡住，拿去当云台指向
            // 会让测试夹具自己瞄错（实测 8rad/s 时一半帧云台在追背面那块）。
            std::vector<int> visible;
            for (int slot = 0; slot < static_cast<int>(plates.size()); ++slot)
            {
                const Eigen::Vector3d &position = plates[static_cast<std::size_t>(slot)].position;
                const Eigen::Vector3d &normal = plates[static_cast<std::size_t>(slot)].normal;
                if (normal.dot(-position.normalized()) > 0.3)
                {
                    visible.push_back(slot);
                }
            }
            if (visible.empty())
            {
                visible.push_back(0);
            }
            int best = visible[0];
            double best_error = 1.0e9;
            for (int slot : visible)
            {
                const double error = std::fabs(rotating_armor::wrapToPi(
                    rotating_armor::vectorYaw(plates[static_cast<std::size_t>(slot)].position) -
                    gimbal.yaw));
                if (error < best_error)
                {
                    best_error = error;
                    best = slot;
                }
            }
            gimbal.snapTo(plates[static_cast<std::size_t>(best)].position);
        }
        const GimbalState gimbal_state = gimbal.state(bias_s);
        // 观测用"真实"姿态生成，EKF 拿到的是报告姿态；两者之差就是抖动。
        GimbalState observing_gimbal = gimbal_state;
        if (shake_sigma_rad > 0.0)
        {
            observing_gimbal.yaw += static_cast<float>(gauss(rng, shake_sigma_rad));
            observing_gimbal.pitch += static_cast<float>(gauss(rng, shake_sigma_rad));
        }
        const int id = (observation_id >= 0) ? observation_id : target_id;
        std::vector<ArmorPose> observations;
        if (provide_observations)
        {
            observations = observe(plates, observing_gimbal, id, rng, sigma_position_m,
                                   sigma_normal_rad);
        }
        last_observations = observations;
        return tracker.step(observations, gimbal_state, time_s, latency_s);
    }
};

// 真值里"离某个绝对 yaw 最近"的板的世界坐标，测试里用来把云台摆到目标上。
inline Eigen::Vector3d nearestPlatePosition(const rotating_armor::State &truth, int plate_count,
                                            double yaw)
{
    const std::vector<Plate> plates = platesAt(truth, plate_count);
    Eigen::Vector3d best = plates.empty() ? Eigen::Vector3d(0.0, 0.0, 0.0) : plates[0].position;
    double best_error = 1.0e9;
    for (const Plate &plate : plates)
    {
        const double error =
            std::fabs(rotating_armor::wrapToPi(rotating_armor::vectorYaw(plate.position) - yaw));
        if (error < best_error)
        {
            best_error = error;
            best = plate.position;
        }
    }
    return best;
}

} // namespace sim
