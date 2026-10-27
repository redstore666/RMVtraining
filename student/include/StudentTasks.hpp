#pragma once

#include "AutoAimTypes.hpp"

#include <vector>

#include <opencv2/core.hpp>

extern std::vector<double> g_reproj_errors; // test（armor_solve 重投影误差样本）
// ===== test计数器（armor_solve 的拒绝原因统计，供 test_camera_live 打印）=====
extern int g_rej_reproj;
extern int g_rej_depth;
extern int g_rej_finite;
extern int g_pnp_fail;

// 任务 1：管理相机 SDK，返回一张非空的 CV_8UC3 BGR 图像。
// 超时或断开时返回 false；不要在此函数中终止进程。
bool init_camera(void **handle, bool verbose = false);
void close_camera(void *handle);
void release_camera(); // 释放相机资源（程序退出前调用）
// 相机固件自算的实际出帧率（MVS "ResultingFrameRate" 节点），与主机处理/显示
// 速度无关；未取到返回 -1。get_pic 内每 30 帧轮询刷新，供调试工具显示。
float cameraResultingFps();
// 画框的辅助函数：
void drawDetections(cv::Mat &image, const std::vector<ArmorDetection> &detections);
bool get_pic(cv::Mat &pic);

// 任务 2：神经网络四点检测（唯一实现）。YOLO 关键点检测模型直接输出
// 装甲板四角点，并解出颜色（敌方过滤）、编号与大小板类型；每个结果包含
// 四个按顺时针排列的图像角点、装甲板尺寸、目标编号和置信度。
// 模型：models/yolov5.xml（OpenVINO，RMV_DET_MODEL=yolo11 切备选）。
std::vector<ArmorDetection> armor_detect(const cv::Mat &pic, TeamColor enemy_color);

// 装甲板两侧 LED 灯条：比板面花纹亮且大，可以独立于神经网络给出"板心 + 板面方向"。
// 两灯条质心中点 = 板心（实测与神经网络板心差 2.5 px 中位）；两灯条连线在图像里的
// 方向 = 板面内水平轴，与竖直轴叉乘即板面法向 → 一条独立的 yaw（相位）量测。
struct LedArmorMeasurement
{
    cv::Point2f left_bar{};
    cv::Point2f right_bar{};
    cv::Point2f center{};
    double baseline_px{0.0};
    double height_px{0.0};
    int target_id{0};
};

std::vector<LedArmorMeasurement> detect_led_armors(const cv::Mat &bgr, TeamColor color);

// 覆盖神经网络检测的置信度阈值（默认 0.7）。离线回放用来量化"降阈值能换多少召回"。
void setArmorScoreThreshold(float threshold);

// 任务 3：使用已标定的相机内参进行位姿解算，再将观测从相机坐标系变换到云台坐标系。
// 输出位置单位为米。
// 云台系 -> 相机光学系（右/下/前），cameraToGimbalPoint 的逆变换。
// 只给可视化用：把 EKF 状态推出的 3D 板位投回图像，眼睛直接对比
// "模型板有没有压住真实板"。
cv::Vec3d gimbalToCameraPoint(const CameraParameters &camera, const cv::Vec3d &p_gimbal);

// 亚像素角点细化（可选）：把检测网络给出的四角在灰度图上细化到亚像素。
// 只在板够大（边长 >= min_edge_px）时做，避免小板上细化窗口跨到背景里把角点拉走。
// 返回被细化的板数。
// 实测（v4 --only-id 3）：win=3 / min_edge=60 比不细化好 5%（87.1 -> 82.5 mm 中位），
// 但 v1（遮挡重锁多）会差一倍，所以只作为可选项、默认关闭。
std::size_t refineArmorCorners(const cv::Mat &gray, std::vector<ArmorDetection> &detections,
                               double min_edge_px = 60.0, int win = 3);

std::vector<ArmorPose> armor_solve(const std::vector<ArmorDetection> &detections,
                                   const CameraParameters &camera,
                                   const GimbalState &gimbal);

// 任务 4：完成观测关联，更新旋转目标 EKF，并预测云台指令。
// timestamp_seconds 单调递增；prediction_bias_s 来自控制器。
PredictionResult ekf_predict(const std::vector<ArmorPose> &observations,
                             const GimbalState &gimbal, double timestamp_seconds);

// ===== 俯视图快照（上车程序 mini top-down 用；2026-10-25）=====
// 纯读取跟踪器内部状态打包给显示层，不改变任何跟踪/控制行为。
// seat_* 的几何与 test_camera_system 的 plateCentreOnScreen 完全一致：
//   phase  = body_yaw + slot·2π/4 + slot_angle_offset
//   centre = platePosition(state, slot)，z 加 slot_z_offset，再沿径向内收
//            slot_radius_offset
//   face_yaw = phase + π（外法向）
struct TopDownSnapshot {
    bool valid{false}; // 跟踪器已初始化；false 时其余字段无意义
    double center_x_m{0.0}, center_y_m{0.0}, center_z_m{0.0};
    double body_yaw_rad{0.0};
    double radius_m{0.0};
    // 车体中心速度（EKF state(1)/state(3)，世界系 m/s）。俯视图的速度箭头和
    // caption 分量都读这两个值；valid=false 时不使用。
    // 作业任务书 3.7 要求 webdebug 可见"EKF 车体中心、速度、yaw、角速度、
    // 半径及状态"，速度这一项原本漏在俯视图之外（只有 8080 的曲线图有）。
    double velocity_x_m_s{0.0}, velocity_y_m_s{0.0};
    double seat_centre_m[4][3]{};
    double seat_face_yaw_rad[4]{};
    int aim_slot{-1};
    bool has_predicted_aim{false};
    double predicted_aim_m[3]{};
};
TopDownSnapshot ekf_topdown_snapshot();
