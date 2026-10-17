// 任务 4 的管线测试：观测 -> ekf_predict -> applyPrediction -> 通信结构体。
// 验收重点是"安全输出"不变量（任务书 §4/§7、验收与评分"安全约束"）：
//   * 无目标 / 未稳定 / 结果非法时，armor_flag、distance、coo_x、coo_y、
//     yaw_vel、pitch_vel、fire_allowance 必须全零，crc=0；
//   * 只有 Stable 才可能出现 fire_allowance=1；
//   * 任何 float 字段都不得出现 NaN/Inf；
//   * center_x/y 以米给出、通信协议里是毫米。
#include "AppSupport.hpp"
#include "StudentTasks.hpp"
#include "tracker_test_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

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

bool frameFieldsFinite(const MessData_AutoAim &message)
{
    return std::isfinite(message.yaw) && std::isfinite(message.pitch) &&
           std::isfinite(message.roll) && std::isfinite(message.bias) &&
           std::isfinite(message.distance) && std::isfinite(message.coo_x) &&
           std::isfinite(message.coo_y) && std::isfinite(message.target_yaw) &&
           std::isfinite(message.target_pitch) && std::isfinite(message.yaw_vel) &&
           std::isfinite(message.pitch_vel);
}

bool isSafeFrame(const MessData_AutoAim &message)
{
    return message.head == kFrameHead && message.tail == kFrameTail && message.crc == 0U &&
           message.armor_flag == 0U && message.fire_allowance == 0U &&
           message.distance == 0.0F && message.coo_x == 0.0F && message.coo_y == 0.0F &&
           message.yaw_vel == 0.0F && message.pitch_vel == 0.0F;
}

} // namespace

int main()
{
    using rotating_armor::State;
    using TrackingState = ::TrackingState;
    using PredictionResult = ::PredictionResult;

    Translator translator{};
    translator.message.head = kFrameHead;
    translator.message.tail = kFrameTail;
    translator.message.status = 0; // 我方红：敌方蓝
    translator.message.bias = 0.0F;

    State truth;
    truth << 3.00, 0.05, 0.00, 0.02, 0.35, 0.0, 0.30, 2.50, 0.25;

    sim::GimbalAim gimbal;
    gimbal.yaw = rotating_armor::vectorYaw(Eigen::Vector3d(3.0, 0.0, 0.0));
    gimbal.pitch = rotating_armor::vectorPitch(Eigen::Vector3d(3.0, 0.0, 0.35));
    gimbal.rate_rad_s = 0.0;

    std::mt19937 rng(20240607U);
    const double dt = 1.0 / 60.0;
    double time_s = 0.0;

    int crc_counts[3] = {0, 0, 0};
    int stable_frames = 0;
    int fire_frames = 0;
    int first_stable = -1;
    bool finite_always = true;
    bool fire_only_when_stable = true;
    bool safe_when_invalid = true;
    bool coo_matches_prediction = true;
    bool armor_flag_matches = true;

    for (int frame = 0; frame < 330; ++frame)
    {
        time_s += dt;
        truth = rotating_armor::transition(truth, dt);
        const std::vector<sim::Plate> plates = sim::platesAt(truth, 4);

        // main.cpp 的顺序：先把手里的云台数据放进消息，再取云台状态。
        const GimbalState gimbal_state = gimbal.state(0.0);
        translator.message.yaw = gimbal_state.yaw;
        translator.message.pitch = gimbal_state.pitch;
        translator.message.roll = gimbal_state.roll;
        translator.message.bias = gimbal_state.prediction_bias_s;

        // 200~240 帧模拟短时遮挡：观测为空，但 ekf_predict 仍必须被调用。
        const bool occluded = (frame >= 200 && frame < 240);
        std::vector<ArmorPose> observations;
        if (!occluded)
        {
            observations = sim::observe(plates, gimbal_state, 7, rng);
        }

        const PredictionResult prediction =
            ekf_predict(observations, gimbalStateFromMessage(translator.message), time_s);
        const bool applied = applyPrediction(translator, prediction);
        const MessData_AutoAim &message = translator.message;

        finite_always = finite_always && frameFieldsFinite(message);
        fire_only_when_stable = fire_only_when_stable && !(message.fire_allowance != 0U && message.crc != 1U);
        if (!applied)
        {
            safe_when_invalid = safe_when_invalid && isSafeFrame(message);
        }
        else
        {
            coo_matches_prediction =
                coo_matches_prediction &&
                std::fabs(message.coo_x - prediction.center_x_m * 1000.0F) <= 1.0F &&
                std::fabs(message.coo_y - prediction.center_y_m * 1000.0F) <= 1.0F;
            armor_flag_matches = armor_flag_matches && message.armor_flag == prediction.target_id;
        }
        if (message.crc <= 2U)
        {
            ++crc_counts[message.crc];
        }
        if (message.crc == 1U)
        {
            ++stable_frames;
            if (first_stable < 0)
            {
                first_stable = frame;
            }
        }
        if (message.fire_allowance != 0U)
        {
            ++fire_frames;
        }
        if (frame == 0 || frame == 5 || frame == 60 || frame == 199 || frame == 239 ||
            frame == 300)
        {
            std::printf("帧%3d crc=%u fire=%u armor_flag=%u coo=(%.3f, %.3f) 距离=%.3f "
                        "target_yaw=%.3f\n",
                        frame, static_cast<unsigned>(message.crc),
                        static_cast<unsigned>(message.fire_allowance),
                        static_cast<unsigned>(message.armor_flag),
                        static_cast<double>(message.coo_x), static_cast<double>(message.coo_y),
                        static_cast<double>(message.distance), static_cast<double>(message.target_yaw));
        }
    }

    std::printf("\ncrc 计数: 0=%d 1=%d 2=%d 稳定帧=%d 开火帧=%d 首次稳定帧=%d\n", crc_counts[0],
                crc_counts[1], crc_counts[2], stable_frames, fire_frames, first_stable);

    check(finite_always, "所有帧的 float 字段有限（无 NaN/Inf 进入协议）");
    check(safe_when_invalid, "预测非法时的帧是完整的安全输出");
    check(fire_only_when_stable, "fire_allowance=1 只出现在 crc=1（Stable）");
    check(coo_matches_prediction, "center_x/y 米 -> 毫米 的换算一致");
    check(armor_flag_matches, "armor_flag 等于预测的目标编号");
    check(first_stable > 0, "链路最终进入 crc=1");
    check(stable_frames >= 150, "稳定（crc=1）帧数 >= 150");
    check(crc_counts[0] >= 5, "早期未锁定时输出 crc=0");

    // ---- 坐标系回归：占位外参必须补上"光学系 -> 云台系"的固定旋转 ----
    // 2026-10-17 实跑 CSV 的根因：config/camera.yaml 的外参是单位阵占位值，
    // 旧代码直接 p_gimbal = R*p_cam + t = p_cam，于是把 OpenCV 的"前(z)"当成云台的
    // "上(z)"，EKF 世界模型整体错位（target_pitch≈1.15rad、半径塌到 0.09m、锁不住）。
    {
        CameraParameters camera;
        camera.calibrated = true;
        camera.image_width = 1440;
        camera.image_height = 1080;
        camera.camera_matrix = (cv::Mat_<double>(3, 3) << 1806.2, 0.0, 738.8, 0.0, 1804.7,
                                526.3, 0.0, 0.0, 1.0);
        camera.distortion_coefficients = cv::Mat::zeros(1, 5, CV_64F);
        camera.rotation_camera_to_gimbal = cv::Mat::eye(3, 3, CV_64F);
        camera.translation_camera_to_gimbal = cv::Mat::zeros(3, 1, CV_64F);
        camera.mean_reprojection_error_px = 0.25;

        // 合成一块小装甲板：物点顺序 TL/TR/BR/BL，135x57mm，板面 z=0
        const double half_w = 0.135 / 2.0;
        const double half_h = 0.057 / 2.0;
        std::vector<cv::Point3f> object_points{
            {-half_w, -half_h, 0.0F}, {half_w, -half_h, 0.0F}, {half_w, half_h, 0.0F}, {-half_w, half_h, 0.0F}};

        // 真值位姿：板在相机前方 1.5m、右 0.3m、上 0.1m，绕竖直轴偏 20°
        cv::Matx33d rotation_yaw(std::cos(0.35), 0.0, std::sin(0.35), 0.0, 1.0, 0.0,
                                 -std::sin(0.35), 0.0, std::cos(0.35));
        cv::Matx33d flip(1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0);
        const cv::Matx33d truth_rotation = rotation_yaw * flip;
        const cv::Vec3d truth_translation(0.30, -0.10, 1.50);
        cv::Mat rvec_truth;
        cv::Rodrigues(cv::Mat(truth_rotation), rvec_truth);

        std::vector<cv::Point2f> corners;
        cv::projectPoints(object_points, rvec_truth, cv::Mat(truth_translation), camera.camera_matrix,
                          camera.distortion_coefficients, corners);
        std::vector<ArmorDetection> detections(1);
        detections[0].size = ArmorSize::Small;
        detections[0].target_id = 7;
        detections[0].confidence = 1.0F;
        for (std::size_t i = 0; i < 4; ++i)
        {
            detections[0].corners[i] = corners[i];
        }

        const std::vector<ArmorPose> placeholder_poses = armor_solve(detections, camera, GimbalState{});
        check(placeholder_poses.size() == 1, "占位外参下 armor_solve 仍输出观测");
        if (placeholder_poses.size() == 1)
        {
            const cv::Vec3d &p_camera = placeholder_poses[0].position_camera_m;
            const cv::Vec3d &p_gimbal = placeholder_poses[0].position_gimbal_m;
            // 相机光学系(右/下/前) -> 云台系(前/左/上)：x_g=z_c, y_g=-x_c, z_g=-y_c
            const double mapping_error = std::max({std::fabs(p_gimbal[0] - p_camera[2]),
                                                   std::fabs(p_gimbal[1] + p_camera[0]),
                                                   std::fabs(p_gimbal[2] + p_camera[1])});
            check(mapping_error < 1e-6, "占位外参按光学系->云台系重排坐标轴（x 前 / y 左 / z 上）");
            check(mapping_error < 1e-6 && p_gimbal[2] > 0.0,
                  "占位外参下云台系的 z 是真实高度（不是相机的前向距离）");
            check(std::fabs(cv::norm(p_gimbal) - cv::norm(p_camera)) < 1e-9,
                  "坐标重排不改变距离（旋转保模长）");
            check(std::isfinite(placeholder_poses[0].armor_yaw) &&
                      std::fabs(cv::norm(placeholder_poses[0].normal_gimbal) - 1.0) < 1e-6,
                  "板面法向已归一化且 yaw 有限");
        }

        // 真外参必须原样使用：p_gimbal = R * p_camera + t
        camera.rotation_camera_to_gimbal = (cv::Mat_<double>(3, 3) << 0.98, 0.0, 0.17, 0.0, 1.0, 0.0,
                                            -0.17, 0.0, 0.98);
        camera.translation_camera_to_gimbal = (cv::Mat_<double>(3, 1) << 0.02, 0.0, 0.05);
        const std::vector<ArmorPose> calibrated_poses = armor_solve(detections, camera, GimbalState{});
        check(calibrated_poses.size() == 1, "真外参下 armor_solve 仍输出观测");
        if (calibrated_poses.size() == 1)
        {
            const cv::Vec3d &p_camera = calibrated_poses[0].position_camera_m;
            const cv::Mat expected = camera.rotation_camera_to_gimbal * cv::Mat(cv::Vec3d(p_camera)) +
                                     camera.translation_camera_to_gimbal;
            const cv::Vec3d p_gimbal = calibrated_poses[0].position_gimbal_m;
            const double error = std::max({std::fabs(p_gimbal[0] - expected.at<double>(0)),
                                           std::fabs(p_gimbal[1] - expected.at<double>(1)),
                                           std::fabs(p_gimbal[2] - expected.at<double>(2))});
            check(error < 1e-9, "非占位外参原样参与 p_gimbal = R*p_camera + t");
        }
    }

    if (g_failures != 0)
    {
        std::printf("\n%d 项失败\n", g_failures);
        return 1;
    }
    std::printf("\n全部通过\n");
    return 0;
}
