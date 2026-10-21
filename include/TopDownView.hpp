#pragma once

// EKF 俯视图（Top-Down）渲染器：把旋转车体的 EKF 结果画成一张"上帝视角"调试图。
//
// 画面元素（与 tmp/example.jpg 同风格：白底 + 红点 + 四根蓝条）：
//   * 红实心圆 = 车体中心 (cx, cy)；
//   * 蓝粗线段 = 4 块装甲板：板心 = 中心 - R*[cos phase, sin phase]
//                （与 RotatingArmorModel::measurement() 是同一个公式），
//                线段方向 = 该处切线（垂直于法向），长度 plate_width_m；
//   * 黄描边   = 当前瞄准的那块板（diagnostics.aim_slot）；
//   * 绿点/绿线 = 本帧 PnP 实测板心（世界系）+ 板面切线，和蓝条错开就是抖；
//   * 品红十字 = 预测瞄准点，另有一条从云台原点射来的虚线；
//   * 灰折线   = 最近 trail_length 帧的车体中心轨迹；
//   * 左上文本 = 状态机 / 速度 / 角速度 / 半径 / NIS / 观测数量。
//
// 坐标约定与世界系一致（见 AimGeometry.hpp）：前 x / 左 y / 上 z。
// 画布"上"= +x（前方），画布"左"= +y（左方），因此车体自转在图上呈逆时针，
// 与俯视图直觉一致；这一点和"相机图像坐标系"完全不同，别混。
//
// 缩放两种模式：
//   follow_target = true （默认）—— 画布中心跟着车体中心走，按半径自适应缩放，
//                                  适合盯着一台车调参（tmp/example.jpg 就是这种构图）；
//   follow_target = false        —— 画布中心 = 世界原点（云台），即
//                                  px = w/2 - (wy-cy)*ppm、py = h/2 - (wx-cx)*ppm，
//                                  ppm 默认 200（±2 m 视野）；目标太远时按 fit_to_target
//                                  自动缩小到仍能看见，适合看"整台车相对云台在哪"。
//
// 这个渲染器只读 EKF 结果、不碰跟踪逻辑，也不参与主程序：test_ekf_viewer 用它做
// 合成真值演示，test_camera_playback 用它开 --topdown 实时显示。

#include "AutoAimTypes.hpp"
#include "RotatingArmorTracker.hpp"

#include <Eigen/Core>

#include <limits>
#include <vector>

#include <opencv2/core.hpp>

namespace rotating_armor
{

// 一帧的实测装甲板（世界系，米）。
// normal_yaw 是板面外法向在世界系里的偏航角；只解出位置时给 NaN，视图就只画点。
struct TopDownObservation
{
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};
    double normal_yaw{std::numeric_limits<double>::quiet_NaN()};
    int target_id{0};
};

struct TopDownOptions
{
    int width{800};
    int height{800};
    bool white_background{true};

    // true：画布中心跟车体中心，缩放按半径自适应；false：中心固定为世界原点。
    bool follow_target{true};
    double pixels_per_meter{200.0};
    double follow_half_extent_m{1.80}; // follow 模式下画布半宽对应的米数（固定比例尺，不再自动缩放）
    double fallback_radius_m{0.25};    // 状态未初始化/半径非法时画板用的半径

    int trail_length{50};         // 轨迹保留帧数（0 = 不画也不存）
    double plate_width_m{0.135};  // 装甲板横向宽度（画成垂直于法向的线段）
    double velocity_arrow_s{0.5}; // 速度箭头代表多少秒的位移
    bool show_grid{true};
    bool show_hud{true};
    // 固定原点（非 follow）视图里，目标超出标称缩放范围时自动缩小，保证它还在画面内。
    bool fit_to_target{true};
    bool show_legend{true};
    // 本轮（2026-10-21）：先只做"实测板 -> 整车"，蓝板（EKF 四块模型板）不再画。
    // 打开后 drawModel 全部跳过，只留实测板 + 法向线 + 反推出来的圆心/圆。
    bool detected_body_only{false};
};

// 视图一帧的输入：EKF 状态 + 诊断 + 预测输出 + 实测板。
// has_state 为 false 时只画观测、网格和 HUD（锁定期没有可信状态）。
struct TopDownFrame
{
    bool has_state{false};
    // 本帧这一路是否真有检测。false 的"其它车"不再画模型——
    // 否则一台早就没有观测的旧轨迹会一直挂在俯视图上（用户反馈的"没有车却还有建模"）。
    bool fresh{true};
    State state{State::Zero()};
    TrackerDiagnostics diagnostics{};
    PredictionResult result{};
    GimbalState gimbal{};
    std::vector<TopDownObservation> observations{};
};

class TopDownView
{
public:
    explicit TopDownView(TopDownOptions options = TopDownOptions{});

    // 清掉历史轨迹（换目标/重开视频时调用）。
    void reset();

    // 渲染一帧，返回内部画布（复用同一块 cv::Mat，别长期持有）。
    // others：同帧里的其它车（多目标）。先画它们（暗一档），再把 frame 当主目标
    // 画在最上面；follow 模式会把视野放大到能同时容下所有车。
    const cv::Mat &render(const TopDownFrame &frame,
                          const std::vector<TopDownFrame> &others = {});

    const cv::Mat &canvas() const noexcept { return canvas_; }
    const TopDownOptions &options() const noexcept { return options_; }
    TopDownOptions &options() noexcept { return options_; }

    // 最近一帧实际使用的缩放与画布中心（世界系，米）。
    double pixelsPerMeter() const noexcept { return pixels_per_meter_; }
    cv::Point2d worldCenter() const noexcept { return cv::Point2d(center_x_m_, center_y_m_); }

    // 世界系 (x, y) -> 画布像素。方向约定与 render() 内部完全一致，测试靠它验证。
    cv::Point2d worldToCanvas(double world_x, double world_y) const noexcept;

private:
    void ensureCanvas();
    void updateTrail(const TopDownFrame &frame);

    TopDownOptions options_;
    cv::Mat canvas_;
    std::vector<Eigen::Vector2d> trail_;
    int trail_target_id_{0};
    double pixels_per_meter_{200.0};
    double center_x_m_{0.0};
    double center_y_m_{0.0};
};

} // namespace rotating_armor
