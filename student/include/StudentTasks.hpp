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

// 显示用位姿解算（含平面目标孪生解消歧）：平面装甲板的 IPPE 存在"孪生对"
// （互为镜像倾角/朝向、重投影差常 <0.5px），逐帧独立选择会让坐标轴与 yaw 来回翻。
// 实测（7 帧）：孪生两解位置差 ≤8.5mm、装甲板 yaw 差 5~50°（典型 10~30°）。
// 消歧规则：误差差 >3px 以误差为准；否则取法向与上一帧最接近的解（时域一致）；
// 首帧取更接近竖直者。armor_solve 未使用本函数（其 yaw 观测在 EKF 阶段处理）。
bool solveArmorPoseDisambiguated(const std::vector<cv::Point3f> &object_points,
                                 const std::vector<cv::Point2f> &image_points,
                                 const cv::Mat &camera_matrix,
                                 const cv::Mat &distortion_coefficients,
                                 cv::Mat &rvec, cv::Mat &tvec);

// 任务 2：神经网络四点检测（唯一实现）。YOLO 关键点检测模型直接输出
// 装甲板四角点，并解出颜色（敌方过滤）、编号与大小板类型；每个结果包含
// 四个按顺时针排列的图像角点、装甲板尺寸、目标编号和置信度。
// 模型：models/yolov5.xml（OpenVINO，RMV_DET_MODEL=yolo11 切备选）。
std::vector<ArmorDetection> armor_detect(const cv::Mat &pic, TeamColor enemy_color);

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
