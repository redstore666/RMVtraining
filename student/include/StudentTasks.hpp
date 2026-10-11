#pragma once

#include "AutoAimTypes.hpp"

#include <vector>

#include <opencv2/core.hpp>

extern std::vector<double> g_reproj_errors; // test
// ===== test计数器（armor_solve 的拒绝原因统计）=====
extern int g_rej_reproj;
extern int g_rej_depth;
extern int g_rej_finite;
extern int g_pnp_fail;

// 任务 1：管理相机 SDK，返回一张非空的 CV_8UC3 BGR 图像。
// 超时或断开时返回 false；不要在此函数中终止进程。
bool init_camera(void **handle, bool verbose = false);
void close_camera(void *handle);
void release_camera(); // 释放相机资源（程序退出前调用）
// 画框的辅助函数：
void drawDetections(cv::Mat &image, const std::vector<ArmorDetection> &detections);
// 调试辅助：返回与 armor_detect 内部完全相同的敌方掩码（仅供测试工具可视化，
// 不参与主流程；实现为内部 buildEnemyMask 的薄包装）。
cv::Mat enemyMaskForDebug(const cv::Mat &bgr, TeamColor enemy_color);
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
bool get_pic(cv::Mat &pic);

// 任务 2：检测敌方装甲板并识别其编号。每个结果必须包含四个按顺时针排列的
// 图像角点、装甲板尺寸、目标编号和置信度。
std::vector<ArmorDetection> armor_detect(const cv::Mat &pic, TeamColor enemy_color);

// 任务 3：使用已标定的相机内参进行位姿解算，再将观测从相机坐标系变换到云台坐标系。
// 输出位置单位为米。
std::vector<ArmorPose> armor_solve(const std::vector<ArmorDetection> &detections,
                                   const CameraParameters &camera,
                                   const GimbalState &gimbal);

// 任务 4：完成观测关联，更新旋转目标 EKF，并预测云台指令。
// timestamp_seconds 单调递增；prediction_bias_s 来自控制器。
PredictionResult ekf_predict(const std::vector<ArmorPose> &observations,
                             const GimbalState &gimbal, double timestamp_seconds);
