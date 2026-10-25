#pragma once

// 相机画面上的调试叠加层（检测框 / 装甲板坐标轴 / PnP 面板 / EKF 面板）。
//
// test_camera_live（第一阶段验收工具）与 test_camera_system（系统视图工具）
// 共用这一份实现：两个工具看到的框、轴、字完全一致，改了这里两边一起变。
// 只做画图，不含任何跟踪/检测逻辑。

#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "AutoAimTypes.hpp"
#include "RotatingArmorTracker.hpp"
#include "StudentTasks.hpp"

namespace rotating_armor
{

struct CameraOverlayFrame
{
    const std::vector<ArmorDetection> *detections{nullptr};
    const std::vector<ArmorPose> *poses{nullptr};
    const CameraParameters *camera{nullptr};

    GimbalState gimbal{};
    // EKF 状态与诊断：has_state=false 时只画检测/PnP，不画车心与瞄准点。
    State state{State::Zero()};
    bool has_state{false};
    // 槽位高度偏移均值：state(4) 会被"最近更新的那块板"带得上下跳，显示时补掉。
    double slot_z_offset{0.0};
    TrackerDiagnostics diagnostics{};
    PredictionResult prediction{};
    bool ekf_enabled{true};

    // 左上状态栏两段文字（由调用方拼好，避免把工具各自的计数逻辑搬进来）。
    std::string status_text;
    std::string fps_text;

    // 图像坐标缩放：1.0 = 原分辨率。面板位置与字号不随它变（"界面像素"）。
    double scale{1.0};
};

// 按键 'm' 用的最近一帧 PnP 位姿缓存（调用方负责 clear + 打印）。
struct CameraOverlayDebug
{
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    std::vector<int> ids;
};

// 就地画到 canvas 上。canvas 至少要有 detections/poses 的角点坐标（按 scale 换算）。
void drawCameraOverlay(cv::Mat &canvas, const CameraOverlayFrame &frame,
                       CameraOverlayDebug *debug = nullptr);

} // namespace rotating_armor
