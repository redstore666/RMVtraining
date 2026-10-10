// armor_solve 离线测试工具：合成图 → armor_detect → armor_solve，打印位姿观测。
//
// 用法: ./build/test_solve <图片> [--red|--blue]
// 示例:
//   ./build/test_solve tests/data/synth_small_1.png --red
//   ./build/test_solve tests/data/synth_dual_3_7_blue.png --blue
//
// 相机参数说明：
//   内参来自 tools/intrinsics_v2.yaml（三轮标定结果）；
//   外参用单位变换（R=I, t=0）——真外参到位后应改读 config/camera.yaml。
//   注意：单位外参仅用于离线开发，严禁作为提交参数（camera.yaml 的 calibrated
//   保持 0，直到真外参填入）。
//
// 期望结果（由几何推算，±5% 内算对）：
//   synth_small_1 --red    : 1 块, 距离 ≈0.80m, yaw≈0, 重投影误差≈0
//   synth_large_5 --red    : 1 块, 距离 ≈0.73m, yaw≈0
//   synth_tilted_2 --red   : 1 块, 距离 ≈0.80m, yaw≈0（滚转不改变法向与光轴夹角）
//   synth_dual_3_7_blue    : 2 块, ≈0.85m(大) / ≈0.92m(小)
#include "StudentTasks.hpp"

#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>

namespace {

// 开发用相机参数：真内参 + 单位外参（仅离线测试！）
// 内参来源：新相机（换机后）MATLAB Camera Calibrator 标定结果，
// 与 config/camera.yaml 保持一致。
CameraParameters makeDevCamera()
{
    CameraParameters cam;
    cam.calibrated = true;
    cam.image_width = 1440;
    cam.image_height = 1080;
    cam.camera_matrix = (cv::Mat_<double>(3, 3) << 1806.2, 0, 738.8,
                         0, 1804.7, 526.3,
                         0, 0, 1);
    cam.distortion_coefficients =
        (cv::Mat_<double>(1, 5) << -0.0761, 0.1522, 0, 0, 0);
    cam.rotation_camera_to_gimbal = cv::Mat::eye(3, 3, CV_64F);
    cam.translation_camera_to_gimbal = cv::Mat::zeros(3, 1, CV_64F);
    cam.mean_reprojection_error_px = 0.25;
    return cam;
}

} // namespace

int main(int argc, char **argv)
{
    std::string image_path;
    std::string save_path;
    TeamColor enemy = TeamColor::Red;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--red")
        {
            enemy = TeamColor::Red;
        }
        else if (arg == "--blue")
        {
            enemy = TeamColor::Blue;
        }
        else if (!arg.empty() && arg[0] != '-')
        {
            image_path = arg;
        }
        else
        {
            std::cerr << "未知参数: " << arg << "\n";
            return 2;
        }
    }
    if (image_path.empty())
    {
        std::cerr << "用法: " << argv[0] << " <图片> [--red|--blue]\n";
        return 2;
    }

    const cv::Mat image = cv::imread(image_path);
    if (image.empty())
    {
        std::cerr << "图片读取失败: " << image_path << "\n";
        return 2;
    }

    const CameraParameters camera = makeDevCamera();
    const GimbalState gimbal{}; // 云台 0 位；单位外参下对结果无影响

    const std::vector<ArmorDetection> detections = armor_detect(image, enemy);
    std::cout << "检测到 " << detections.size() << " 块装甲板\n";
    if (detections.empty())
    {
        return 0;
    }

    const std::vector<ArmorPose> poses = armor_solve(detections, camera, gimbal);
    if (poses.empty())
    {
        std::cout << "解算 0 个观测——若上方检测数非 0，多半是 buildObjectPoints "
                     "未实现或门禁全部拒绝\n";
        return 0;
    }

    for (const ArmorPose &p : poses)
    {
        const double dist_cam = cv::norm(p.position_camera_m);
        const double dist_gim = cv::norm(p.position_gimbal_m);
        std::cout << "id=" << p.detection.target_id
                  << " size=" << (p.detection.size == ArmorSize::Large ? "大" : "小")
                  << " 位置(相机系 m): (" << p.position_camera_m[0] << ", "
                  << p.position_camera_m[1] << ", " << p.position_camera_m[2] << ")"
                  << " 距离=" << dist_cam << "m"
                  << "  云台系距离=" << dist_gim << "m"
                  << "  yaw=" << p.armor_yaw * 180.0 / CV_PI << "°"
                  << "  重投影=" << p.reprojection_error << "px\n";
    }
    return 0;
}
