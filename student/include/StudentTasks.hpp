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

// ===== test钩子（需求③：灯条约束拟合）=====
// 开关/变体：0=关闭（基线 minAreaRect）；1=矩形拟合 v1（PCA+分位）；2=四边拟合 v2（默认，
// 两条长边 Huber 直线 × 两条端线求交取四角）。环境变量 RMV_BAR_FIT=0/off 关闭、=1/v1 用 v1。
// 供离线 A/B 工具在进程内切换（同一帧同一掩码各跑一次，对照严格同源）。
extern int g_bar_fit_mode;
// 拟合统计（仅统计开启状态下的运行）：total=进入拟合的轮廓数；accept=采纳拟合数；
// rej_* 为各回退原因计数：pts=点数不足/退化；size=接受门；g1=支撑率；g2=轴偏差；g3=面积比；
// edges=边带点不足/线拟合退化；par=两长边夹角过大；width=宽度比/梯形发散过大。
extern int g_bar_fit_total;
extern int g_bar_fit_accept;
extern int g_bar_fit_rej_pts;
extern int g_bar_fit_rej_size;
extern int g_bar_fit_rej_g1;
extern int g_bar_fit_rej_g2;
extern int g_bar_fit_rej_g3;
extern int g_bar_fit_rej_edges;
extern int g_bar_fit_rej_par;
extern int g_bar_fit_rej_width;
extern double g_bar_fit_ms_total; // 累计拟合耗时（毫秒），除以帧数得每帧近似开销

// ===== test钩子（需求③ P3：宽度短时域稳健化）=====
// 灯条"宽度"参数的中位数窗长度（0=关闭；默认 5）。环境变量 RMV_BAR_WIDTH_WIN 可覆盖。
// 只平滑宽度（沿中轴的垂直分量），中心/方向/端点不参与，避免引入跟踪滞后。
extern int g_bar_width_window;
extern int g_bar_width_smooth_count; // 累计"应用了宽度平滑"的灯条次数
extern int g_bar_width_reset_count;  // 累计轨迹重置次数（新建/间隔过长/中心跳变）

// 调试钩子：最近一次 armor_detect 提取出的灯条最终几何（拟合采纳或回退后的结果），
// 供离线度量工具统计灯条级稳定性；主流程不读取。
struct DebugBar
{
    cv::Point2f center;
    cv::Point2f axis;   // 长轴单位方向（模 180° 唯一）
    float half_l{0.0F}; // 半长（像素）
    float half_w{0.0F}; // 半宽（像素）
    bool fitted{false}; // true=约束拟合结果（v1/v2）；false=回退 minAreaRect
    std::array<cv::Point2f, 4> corners{}; // 灯条四角（拟合采纳时为四边形交点）
};
std::vector<DebugBar> debugBars();

// 调试辅助：返回"正则化"掩码——原始掩码压暗（保留对照）+ 灯条拟合四边形画满（亮），
// 供回放/实时工具的 mask 窗口查看"剔除四周毛糙、只保留拟合几何"的效果；主流程不使用。
cv::Mat regularizedMaskForDebug(const cv::Mat &bgr, TeamColor enemy_color);

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
