#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

enum class TeamColor : std::uint8_t
{
    Red = 0,
    Blue = 1,
};

enum class ArmorSize : std::uint8_t
{
    Small = 0,
    Large = 1,
};

enum class TrackingState : std::uint16_t
{
    // 数值直接对应历史通信字段 crc。
    NoTarget = 0,
    Stable = 1,
    Unstable = 2,
};

struct GimbalState
{
    // 控制器快照：角度单位为弧度，prediction_bias_s 单位为秒。
    float yaw{0.0F};
    float pitch{0.0F};
    float roll{0.0F};
    float prediction_bias_s{0.0F};
};

struct CameraParameters
{
    // 所有矩阵均为单通道 CV_64F。刚体变换为
    // p_gimbal = R_camera_to_gimbal * p_camera + t_camera_to_gimbal。
    bool calibrated{false};
    int image_width{0};
    int image_height{0};
    cv::Mat camera_matrix;                   // 内参
    cv::Mat distortion_coefficients;         // 畸变系数
    cv::Mat rotation_camera_to_gimbal;       // 外参R
    cv::Mat translation_camera_to_gimbal;    // 外参T 3x1，单位为米，表示在云台坐标系中的平移。
    double mean_reprojection_error_px{-1.0}; // 平均重投影误差

    bool valid() const; // 检查可用性
};

struct ArmorDetection
{
    // 图像角点按顺时针排列：左上、右上、右下、左下。
    std::array<cv::Point2f, 4> corners{};
    ArmorSize size{ArmorSize::Small};
    int target_id{0};       // 0 保留给未知目标/无目标。
    float confidence{0.0F}; // 分类器置信度，范围为 [0, 1]。
};

struct ArmorPose
{
    // 平移向量单位为米。armor_yaw 是板面法向在云台坐标系中的偏航角（弧度，
    // atan2(n_y, n_x)，坐标约定前 x / 左 y / 上 z），reprojection_error 是单个
    // 观测的重投影误差（像素）。
    //
    // 注意分工：armor_solve 只做 camera -> gimbal 这一层旋转；云台姿态补偿
    // （gimbal -> 跟踪系/世界系）是 EKF 阶段的事，所以这两个字段都还是
    // "随云台转"的量。normal_gimbal 是同一法向的完整单位向量（未解算时为
    // (0,0,0)），EKF 用它加上云台姿态定初相——只靠 armor_yaw 这一个标量会丢掉
    // 法向的俯仰分量，云台有 pitch 时算不准。
    ArmorDetection detection;
    cv::Vec3d position_camera_m{0.0, 0.0, 0.0};
    cv::Vec3d position_gimbal_m{0.0, 0.0, 0.0};
    cv::Vec3d normal_gimbal{0.0, 0.0, 0.0};
    double armor_yaw{0.0};
    double reprojection_error{0.0};
};

struct PredictionResult
{
    // EKF 在约定跟踪坐标系中的车体状态：位置/半径单位为米，
    // 速度单位为米每秒，角度单位为弧度，角速度单位为弧度每秒。
    TrackingState state{TrackingState::NoTarget};
    std::uint8_t target_id{0};
    float center_x_m{0.0F};
    float center_y_m{0.0F};
    float center_velocity_x_m_s{0.0F};
    float center_velocity_y_m_s{0.0F};
    float body_yaw{0.0F};
    float body_yaw_velocity{0.0F};
    float radius_1_m{0.0F};
    float radius_2_m{0.0F};
    // 两个装甲板高度（米）：height_1 = EKF 的 z（偶数槽板面），height_2 =
    // z + height_diff（奇数槽板面）。四块板不共高时两者不同；单板可见时
    // height_diff 不可观测（更新里自动冻结），height_2 会跟着 height_1 走。
    float height_1_m{0.0F};
    float height_2_m{0.0F};
    // 最终云台指令：距离单位为米，角度单位为弧度，角速度单位为弧度每秒。
    float distance_m{0.0F};
    float target_yaw{0.0F};
    float target_pitch{0.0F};
    float target_yaw_velocity{0.0F};
    float target_pitch_velocity{0.0F};
    bool fire_allowed{false};
    // 开火诊断：板面内最大线性脱靶量（米）与实际生效的角度预算（弧度）。
    // 角度预算被"板半宽/半高 ÷ 距离 × 0.6"压住时，说明限制命中的是板尺寸而不是云台。
    float fire_miss_m{0.0F};
    float fire_budget_rad{0.0F};
};

// 读取 OpenCV YAML，将矩阵统一为 CV_64F，并拒绝占位数据或几何上无效的标定结果。
// 失败时会在 error 中写入错误信息。
bool loadCameraParameters(const std::string &path, CameraParameters &parameters,
                          std::string &error);
