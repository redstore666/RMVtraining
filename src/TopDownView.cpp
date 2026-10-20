#include "TopDownView.hpp"

#include "AimGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <string>

#include <opencv2/imgproc.hpp>

namespace rotating_armor
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

// 调色板（BGR）。白底/黑底共用，深色底上这些颜色同样可读。
const cv::Scalar kPlateColor(235, 45, 20);    // 蓝：模型装甲板
const cv::Scalar kAimPlateColor(0, 200, 255); // 黄：瞄准板描边
const cv::Scalar kCenterColor(25, 25, 235);   // 红：车体中心
const cv::Scalar kObsColor(40, 175, 60);      // 绿：本帧实测板
const cv::Scalar kOtherColor(150, 120, 40);   // 暗棕：别的编号的观测
const cv::Scalar kAimColor(215, 40, 205);     // 品红：预测瞄准点
const cv::Scalar kVelocityColor(0, 130, 255); // 橙：速度箭头
const cv::Scalar kTrailColor(150, 150, 150);  // 灰：历史轨迹
const cv::Scalar kCamColor(120, 120, 120);    // 深灰：相机光轴

const char *phaseName(LockPhase phase)
{
    switch (phase)
    {
    case LockPhase::Idle:
        return "idle";
    case LockPhase::Collecting:
        return "collecting";
    case LockPhase::Tracking:
        return "tracking";
    }
    return "?";
}

const char *stateName(TrackingState state)
{
    switch (state)
    {
    case TrackingState::NoTarget:
        return "NoTarget";
    case TrackingState::Unstable:
        return "Unstable";
    case TrackingState::Stable:
        return "Stable";
    }
    return "?";
}

std::string fmt(const char *format, ...)
{
    char buffer[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    return std::string(buffer);
}

cv::Scalar mix(const cv::Scalar &a, const cv::Scalar &b, double t)
{
    const double s = std::clamp(t, 0.0, 1.0);
    return cv::Scalar(a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s,
                      a[2] + (b[2] - a[2]) * s);
}

cv::Point ip(const cv::Point2d &point)
{
    return cv::Point(static_cast<int>(std::lround(point.x)),
                     static_cast<int>(std::lround(point.y)));
}

void textAt(cv::Mat &canvas, const std::string &text, const cv::Point &origin, double scale,
            const cv::Scalar &color, int thickness = 1)
{
    cv::putText(canvas, text, origin, cv::FONT_HERSHEY_SIMPLEX, scale, color, thickness,
                cv::LINE_AA);
}

// 半透明填充：只混合目标矩形，不整图 addWeighted。
void fillAlpha(cv::Mat &canvas, const cv::Rect &rect, const cv::Scalar &color, double alpha)
{
    const cv::Rect clipped = rect & cv::Rect(0, 0, canvas.cols, canvas.rows);
    if (clipped.width <= 0 || clipped.height <= 0)
        return;
    cv::Mat roi = canvas(clipped);
    const cv::Mat overlay(roi.size(), roi.type(), color);
    const double a = std::clamp(alpha, 0.0, 1.0);
    cv::addWeighted(overlay, a, roi, 1.0 - a, 0.0, roi);
}

void dashed(cv::Mat &canvas, const cv::Point2d &from, const cv::Point2d &to,
            const cv::Scalar &color, int thickness, double dash)
{
    const cv::Point2d delta = to - from;
    const double length = std::hypot(delta.x, delta.y);
    if (!(length > 1e-6))
        return;
    const cv::Point2d unit = delta / length;
    for (double start = 0.0; start < length; start += 2.0 * dash)
    {
        const double end = std::min(start + dash, length);
        cv::line(canvas, ip(from + unit * start), ip(from + unit * end), color, thickness,
                 cv::LINE_AA);
    }
}

// 世界系 -> 画布像素。画布"上"= +x（前方），画布"左"= +y（左方），所以车体
// 自转在图上呈逆时针——和相机图像坐标系正好相反，这是全局唯一的一处约定，
// 所有绘制都必须走这一个函数。
cv::Point2d worldToPixel(int width, int height, double ppm, double center_x, double center_y,
                         double world_x, double world_y)
{
    return cv::Point2d(0.5 * width - (world_y - center_y) * ppm,
                       0.5 * height - (world_x - center_x) * ppm);
}

struct Scene
{
    cv::Mat *canvas{nullptr};
    double ppm{200.0};
    double center_x{0.0};
    double center_y{0.0};
    double scale{1.0};
    cv::Scalar bg;
    cv::Scalar ink;
    double bottom_inset{0.0}; // 底部图例面板占掉的高度，边缘标记要避开

    cv::Point2d toPx(double x, double y) const
    {
        return worldToPixel(canvas->cols, canvas->rows, ppm, center_x, center_y, x, y);
    }
    bool inside(const cv::Point2d &point) const
    {
        return point.x >= 0.0 && point.y >= 0.0 && point.x < canvas->cols && point.y < canvas->rows;
    }
};

void drawGrid(const Scene &scene)
{
    cv::Mat &canvas = *scene.canvas;
    const double half_x = 0.5 * canvas.rows / scene.ppm;
    const double half_y = 0.5 * canvas.cols / scene.ppm;
    const double x_lo = scene.center_x - half_x;
    const double x_hi = scene.center_x + half_x;
    const double y_lo = scene.center_y - half_y;
    const double y_hi = scene.center_y + half_y;
    // 网格步长跟缩放走：缩得很小时还用 0.5 m 会糊成一片，缩得很大时又会只剩几条线。
    // 目标是最细的线之间留 ~40 px，粗线每 5 格一条。
    double step = 10.0;
    const double candidates[] = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0};
    for (double candidate : candidates)
    {
        if (candidate * scene.ppm >= 40.0)
        {
            step = candidate;
            break;
        }
    }
    const cv::Scalar minor = mix(scene.bg, scene.ink, 0.08);
    const cv::Scalar major = mix(scene.bg, scene.ink, 0.20);
    for (long i = static_cast<long>(std::ceil(x_lo / step)); i * step <= x_hi; ++i)
    {
        const double x = static_cast<double>(i) * step;
        cv::line(canvas, ip(scene.toPx(x, y_lo)), ip(scene.toPx(x, y_hi)),
                 (i % 5 == 0) ? major : minor, 1, cv::LINE_AA);
    }
    for (long i = static_cast<long>(std::ceil(y_lo / step)); i * step <= y_hi; ++i)
    {
        const double y = static_cast<double>(i) * step;
        cv::line(canvas, ip(scene.toPx(x_lo, y)), ip(scene.toPx(x_hi, y)),
                 (i % 5 == 0) ? major : minor, 1, cv::LINE_AA);
    }
}

void drawTrail(const Scene &scene, const std::vector<Eigen::Vector2d> &trail)
{
    cv::Mat &canvas = *scene.canvas;
    if (trail.size() < 2)
        return;
    for (std::size_t i = 1; i < trail.size(); ++i)
    {
        // 越新越深，旧段淡到接近背景。
        const double age = static_cast<double>(trail.size() - 1 - i) /
                           static_cast<double>(trail.size());
        const cv::Scalar color = mix(scene.bg, kTrailColor, 0.15 + 0.55 * (1.0 - age));
        cv::line(canvas, ip(scene.toPx(trail[i - 1].x(), trail[i - 1].y())),
                 ip(scene.toPx(trail[i].x(), trail[i].y())), color, 2, cv::LINE_AA);
    }
}

void drawOrigin(const Scene &scene, const cv::Point2d &origin_px)
{
    cv::Mat &canvas = *scene.canvas;
    if (!scene.inside(origin_px))
    {
        // follow 模式下云台原点常常在视野外：在画布边缘画一个方向标记，
        // 否则画面上那条瞄准射线就成了"从边框飞出来的无源之线"。
        const cv::Point2d center(0.5 * canvas.cols, 0.5 * canvas.rows);
        const cv::Point2d delta = origin_px - center;
        const double length = std::hypot(delta.x, delta.y);
        if (!(length > 1e-6))
            return;
        // 沿 center->origin 方向，找它穿出"内缩画布"的那个点当边缘标记位置。
        const double mx = 32.0 * scene.scale;
        const double my_top = 32.0 * scene.scale;
        const double my_bottom = 32.0 * scene.scale + scene.bottom_inset;
        double t = 1.0e9;
        const double right = canvas.cols - mx;
        const double left = mx;
        const double top = my_top;
        const double bottom = canvas.rows - my_bottom;
        if (delta.x > 1e-9)
            t = std::min(t, (right - center.x) / delta.x);
        else if (delta.x < -1e-9)
            t = std::min(t, (left - center.x) / delta.x);
        if (delta.y > 1e-9)
            t = std::min(t, (bottom - center.y) / delta.y);
        else if (delta.y < -1e-9)
            t = std::min(t, (top - center.y) / delta.y);
        if (!(t > 0.0 && t < 1.0e8))
            return;
        const cv::Point2d edge = center + delta * t;
        const cv::Point marker = ip(edge);
        cv::circle(canvas, marker, 5, kCamColor, -1, cv::LINE_AA);
        cv::circle(canvas, marker, 10, kCamColor, 1, cv::LINE_AA);
        const std::string label = fmt("gimbal %.1f m", length / scene.ppm);
        const double text_scale = 0.42 * scene.scale;
        int baseline = 0;
        const cv::Size size =
            cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, text_scale, 1, &baseline);
        cv::Point text_pos = marker + cv::Point(14, 5);
        if (text_pos.x + size.width > canvas.cols - 6)
            text_pos.x = marker.x - 14 - size.width;
        text_pos.y = std::clamp(text_pos.y, size.height + 4, canvas.rows - 6);
        textAt(canvas, label, text_pos, text_scale, kCamColor);
        return;
    }
    const int r = std::max(5, static_cast<int>(std::lround(7.0 * scene.scale)));
    cv::circle(canvas, ip(origin_px), r, kCamColor, 2, cv::LINE_AA);
    cv::line(canvas, ip(origin_px) + cv::Point(-r - 5, 0), ip(origin_px) + cv::Point(r + 5, 0),
             kCamColor, 1, cv::LINE_AA);
    cv::line(canvas, ip(origin_px) + cv::Point(0, -r - 5), ip(origin_px) + cv::Point(0, r + 5),
             kCamColor, 1, cv::LINE_AA);
    textAt(canvas, "gimbal (0,0)", ip(origin_px) + cv::Point(r + 9, 5),
           0.42 * scene.scale, kCamColor);
}

void drawModel(const Scene &scene, const TopDownFrame &frame, const TopDownOptions &options,
               double radius)
{
    cv::Mat &canvas = *scene.canvas;
    const State &state = frame.state;
    const cv::Point2d center_px = scene.toPx(state(0), state(2));

    // 半径参考圆：看一眼就知道 R 有没有被锁对。
    cv::circle(canvas, ip(center_px), static_cast<int>(std::lround(radius * scene.ppm)),
               mix(scene.bg, kPlateColor, 0.30), 1, cv::LINE_AA);

    const int thickness = std::clamp(static_cast<int>(std::lround(0.018 * scene.ppm)), 3, 18);
    const double half_width = 0.5 * options.plate_width_m;
    const int aim_slot = frame.diagnostics.aim_slot;
    for (int slot = 0; slot < 4; ++slot)
    {
        // 与 RotatingArmorModel::measurement() 完全同一个公式：
        //   板心 = 中心 - R * [cos phase, sin phase]，phase = yaw + k*pi/2。
        // 这里直接按几何算，不复用 EKF 内部（避免把"渲染"和"状态"绑死）。
        const double phase = state(6) + kPi * 0.5 * slot;
        const double cos_phase = std::cos(phase);
        const double sin_phase = std::sin(phase);
        const Eigen::Vector2d plate(state(0) - radius * cos_phase, state(2) - radius * sin_phase);
        // 板面方向 = 径向的切线（垂直于法向），所以是 (-sin, cos)。
        const Eigen::Vector2d tangent(-sin_phase, cos_phase);
        const cv::Point2d p0 =
            scene.toPx(plate.x() - half_width * tangent.x(), plate.y() - half_width * tangent.y());
        const cv::Point2d p1 =
            scene.toPx(plate.x() + half_width * tangent.x(), plate.y() + half_width * tangent.y());
        const bool is_aim = slot == aim_slot;
        if (is_aim)
        {
            // 瞄准板先画一圈更粗的黄描边，再盖蓝色。
            cv::line(canvas, ip(p0), ip(p1), kAimPlateColor, thickness + 8, cv::LINE_AA);
            cv::circle(canvas, ip(p0), (thickness + 8) / 2, kAimPlateColor, -1, cv::LINE_AA);
            cv::circle(canvas, ip(p1), (thickness + 8) / 2, kAimPlateColor, -1, cv::LINE_AA);
        }
        cv::line(canvas, ip(p0), ip(p1), kPlateColor, thickness, cv::LINE_AA);
        cv::circle(canvas, ip(p0), thickness / 2, kPlateColor, -1, cv::LINE_AA);
        cv::circle(canvas, ip(p1), thickness / 2, kPlateColor, -1, cv::LINE_AA);
    }

    // 车体中心：红实心圆（和 tmp/example.jpg 一致）。
    cv::circle(canvas, ip(center_px),
               std::clamp(static_cast<int>(std::lround(0.011 * scene.ppm)), 4, 9), kCenterColor,
               -1, cv::LINE_AA);

    // 速度箭头：长度 = velocity_arrow_s 秒的位移，短到看不见时给最小像素长度。
    const double vx = state(1);
    const double vy = state(3);
    const double speed = std::hypot(vx, vy);
    if (std::isfinite(speed) && speed > 0.02)
    {
        const double arrow_s = std::max(options.velocity_arrow_s, 0.02 / speed);
        cv::Point2d tip = scene.toPx(state(0) + vx * arrow_s, state(2) + vy * arrow_s);
        const double length_px = cv::norm(tip - center_px);
        if (length_px < 18.0)
        {
            const cv::Point2d dir = (tip - center_px) / std::max(length_px, 1e-6);
            tip = center_px + dir * 18.0;
        }
        // 箭头长度跟着线段走：线短的时候别画出一个占满整条线的大箭头。
        const double tip_length = std::clamp(16.0 / std::max(length_px, 1.0), 0.12, 0.40);
        cv::arrowedLine(canvas, ip(center_px), ip(tip), kVelocityColor, 2, cv::LINE_AA, 0,
                        tip_length);
        textAt(canvas, fmt("v %.2f", speed), ip(tip) + cv::Point(6, -6), 0.40 * scene.scale,
               kVelocityColor);
    }
}

void drawObservations(const Scene &scene, const TopDownFrame &frame, const TopDownOptions &options)
{
    cv::Mat &canvas = *scene.canvas;
    const int target_id = frame.diagnostics.target_id;
    for (const TopDownObservation &observation : frame.observations)
    {
        if (!observation.position.allFinite())
            continue;
        // 还没锁定（target_id==0）时所有观测都是"当前帧的板"，别画成灰色让人以为无关。
        const bool current = target_id == 0 || observation.target_id == target_id;
        const cv::Scalar color = current ? kObsColor : kOtherColor;
        const cv::Point2d center_px =
            scene.toPx(observation.position.x(), observation.position.y());
        cv::circle(canvas, ip(center_px), std::max(3, static_cast<int>(std::lround(5.0 * scene.scale))),
                   color, -1, cv::LINE_AA);
        if (std::isfinite(observation.normal_yaw))
        {
            // 板面内水平轴 = 法向转 90°；只当参考，PnP 的孪生解可能把法向翻 180°。
            const Eigen::Vector2d tangent(-std::sin(observation.normal_yaw),
                                          std::cos(observation.normal_yaw));
            const double half_width = 0.5 * options.plate_width_m;
            cv::line(canvas,
                     ip(scene.toPx(observation.position.x() - half_width * tangent.x(),
                                   observation.position.y() - half_width * tangent.y())),
                     ip(scene.toPx(observation.position.x() + half_width * tangent.x(),
                                   observation.position.y() + half_width * tangent.y())),
                     mix(scene.bg, color, 0.75), 3, cv::LINE_AA);
        }
    }
}

void drawAim(const Scene &scene, const TopDownFrame &frame, const cv::Point2d &origin_px)
{
    cv::Mat &canvas = *scene.canvas;
    const PredictionResult &result = frame.result;
    const bool has_aim = result.state != TrackingState::NoTarget &&
                         std::isfinite(result.distance_m) && result.distance_m > 1e-3F &&
                         std::isfinite(result.target_yaw) && std::isfinite(result.target_pitch);
    if (!has_aim)
        return;

    const double yaw = result.target_yaw;
    const double pitch = result.target_pitch;
    const double horizontal = result.distance_m * std::cos(pitch);
    const Eigen::Vector2d aim(horizontal * std::cos(yaw), horizontal * std::sin(yaw));

    // 云台光轴：PredictionResult 的 target_yaw 是世界系偏航，gimbal.yaw 也是世界系
    // 偏航，两者之差就是"云台还差多少度对上"，开火判据里的 yaw 误差就是它。
    if (std::isfinite(frame.gimbal.yaw))
    {
        const double offset = wrapToPi(yaw - frame.gimbal.yaw);
        if (std::fabs(offset) > 0.005)
        {
            const cv::Point2d cam_tip = origin_px + cv::Point2d(
                                                        std::cos(frame.gimbal.yaw),
                                                        std::sin(frame.gimbal.yaw)) *
                                                        40.0 * scene.ppm;
            cv::Point cam_a = ip(origin_px);
            cv::Point cam_b = ip(cam_tip);
            if (cv::clipLine(cv::Size(canvas.cols, canvas.rows), cam_a, cam_b))
                dashed(canvas, cv::Point2d(cam_a), cv::Point2d(cam_b), kCamColor, 1, 6.0);
        }
    }

    const cv::Point2d aim_px = scene.toPx(aim.x(), aim.y());
    // 瞄准射线：从云台原点指向预测瞄准点（裁剪到画布内，原点可能在视野外）。
    cv::Point ray_a = ip(origin_px);
    cv::Point ray_b = ip(aim_px);
    if (cv::clipLine(cv::Size(canvas.cols, canvas.rows), ray_a, ray_b))
        dashed(canvas, cv::Point2d(ray_a), cv::Point2d(ray_b), kAimColor, 1, 7.0);

    // 准星 + 距离标注。
    const int r = std::max(6, static_cast<int>(std::lround(10.0 * scene.scale)));
    const cv::Point center = ip(aim_px);
    cv::line(canvas, center + cv::Point(-r - 3, 0), center + cv::Point(r + 3, 0), kAimColor, 2,
             cv::LINE_AA);
    cv::line(canvas, center + cv::Point(0, -r - 3), center + cv::Point(0, r + 3), kAimColor, 2,
             cv::LINE_AA);
    cv::circle(canvas, center, r, kAimColor, 1, cv::LINE_AA);
    if (scene.inside(aim_px))
    {
        // 靠上时标签放到准星下面，免得压到左上角的 HUD 面板。
        const int dy = (center.y < canvas.rows / 2) ? (r + 16) : -r;
        textAt(canvas, fmt("AIM %.2f m", result.distance_m), center + cv::Point(r + 6, dy),
               0.44 * scene.scale, kAimColor);
    }
}

} // namespace

TopDownView::TopDownView(TopDownOptions options) : options_(options)
{
    reset();
}

void TopDownView::reset()
{
    trail_.clear();
    trail_target_id_ = 0;
    center_x_m_ = 0.0;
    center_y_m_ = 0.0;
    pixels_per_meter_ = options_.pixels_per_meter;
    canvas_.release();
}

void TopDownView::ensureCanvas()
{
    if (canvas_.rows != options_.height || canvas_.cols != options_.width ||
        canvas_.type() != CV_8UC3)
    {
        canvas_.create(options_.height, options_.width, CV_8UC3);
    }
}

cv::Point2d TopDownView::worldToCanvas(double world_x, double world_y) const noexcept
{
    return worldToPixel(options_.width, options_.height, pixels_per_meter_, center_x_m_,
                        center_y_m_, world_x, world_y);
}

void TopDownView::updateTrail(const TopDownFrame &frame)
{
    if (options_.trail_length <= 0)
    {
        trail_.clear();
        return;
    }
    if (!frame.has_state)
        return;
    const int target_id = frame.diagnostics.target_id;
    if (target_id != trail_target_id_)
    {
        trail_.clear();
        trail_target_id_ = target_id;
    }
    trail_.emplace_back(frame.state(0), frame.state(2));
    while (static_cast<int>(trail_.size()) > options_.trail_length)
        trail_.erase(trail_.begin());
}

const cv::Mat &TopDownView::render(const TopDownFrame &frame)
{
    ensureCanvas();
    const int width = options_.width;
    const int height = options_.height;

    Scene scene;
    scene.canvas = &canvas_;
    scene.scale = static_cast<double>(std::min(width, height)) / 800.0;
    scene.bg = options_.white_background ? cv::Scalar(255, 255, 255) : cv::Scalar(30, 30, 34);
    scene.ink = options_.white_background ? cv::Scalar(55, 55, 60) : cv::Scalar(228, 228, 232);
    canvas_.setTo(scene.bg);

    // 字号/面板尺寸提前算：world 模式的 auto-fit 要知道 HUD 占多高，才能把目标
    // 留在面板下方，而不是画到面板底下去。
    const double font_scale = 0.48 * scene.scale;
    const int line_height = static_cast<int>(std::lround(23.0 * scene.scale));
    const int pad = static_cast<int>(std::lround(10.0 * scene.scale));
    const int hud_height = 5 * line_height + 2 * pad;

    // ---- 1. 视图变换：follow 模式跟车心，world 模式固定原点 ----
    const double radius = (frame.has_state && std::isfinite(frame.state(8)) &&
                           frame.state(8) > 0.02)
                              ? frame.state(8)
                              : options_.fallback_radius_m;
    if (options_.follow_target)
    {
        // 丢状态时保持上一帧的中心，画面不要"跳回原点"。刚上电还没有 EKF 状态时，
        // 退一步用本帧实测板的质心当临时中心——否则 collecting 阶段画面是空的。
        if (frame.has_state)
        {
            center_x_m_ = frame.state(0);
            center_y_m_ = frame.state(2);
        }
        else
        {
            Eigen::Vector2d sum(0.0, 0.0);
            int count = 0;
            for (const TopDownObservation &observation : frame.observations)
            {
                if (!observation.position.allFinite())
                    continue;
                sum += observation.position.head<2>();
                ++count;
            }
            if (count > 0)
            {
                center_x_m_ = sum.x() / count;
                center_y_m_ = sum.y() / count;
            }
        }
        const double extent = std::max(options_.follow_half_extent_m, 2.2 * radius);
        pixels_per_meter_ = 0.5 * std::min(width, height) * 0.88 / extent;
    }
    else
    {
        center_x_m_ = 0.0;
        center_y_m_ = 0.0;
        double ppm = options_.pixels_per_meter;
        if (options_.fit_to_target && frame.has_state)
        {
            // 固定原点视图里，目标一远就会被画到画布外。按"原点 + 车体"整体能放下
            // 来定一个上限，近目标时仍然是用户给的 pixels_per_meter。
            // 竖直方向额外扣掉 HUD 那条带，否则目标会藏到面板底下。
            const double reach = std::hypot(frame.state(0), frame.state(2)) + radius + 0.15;
            const double v_room =
                0.5 * height - (options_.show_hud ? hud_height + 28.0 * scene.scale : 12.0);
            const double h_room = 0.5 * width - 12.0;
            const double fit = std::min(v_room, h_room) / std::max(reach, 0.60);
            ppm = std::min(ppm, fit);
        }
        pixels_per_meter_ = ppm;
    }
    scene.center_x = center_x_m_;
    scene.center_y = center_y_m_;
    scene.ppm = pixels_per_meter_;
    const int footer_height = static_cast<int>(std::lround(58.0 * scene.scale));
    scene.bottom_inset =
        options_.show_legend ? footer_height + static_cast<double>(15.0 * scene.scale) : 0.0;

    // ---- 2. 背景网格、轨迹 ----
    if (options_.show_grid)
        drawGrid(scene);
    updateTrail(frame);
    drawTrail(scene, trail_);

    const cv::Point2d origin_px = scene.toPx(0.0, 0.0);

    // ---- 3. 模型：四块板 + 车体中心 + 速度 ----
    if (frame.has_state &&
        frame.state.allFinite())
        drawModel(scene, frame, options_, radius);

    // ---- 4. 实测板（绿）与云台原点 ----
    drawObservations(scene, frame, options_);
    drawOrigin(scene, origin_px);

    // ---- 5. 瞄准射线与准星 ----
    drawAim(scene, frame, origin_px);

    // ---- 6. HUD（固定 5 行，面板高度恒定，方便 world 模式预留空间） ----
    const TrackerDiagnostics &diagnostics = frame.diagnostics;
    const PredictionResult &result = frame.result;
    std::vector<std::string> lines;
    lines.reserve(5);
    lines.push_back(fmt("EKF TOP-DOWN   phase %s   id %d   %s", phaseName(diagnostics.phase),
                        diagnostics.target_id,
                        frame.has_state ? stateName(result.state) : "no state"));
    if (frame.has_state)
    {
        const State &state = frame.state;
        lines.push_back(fmt("center x %+7.3f  y %+7.3f   vx %+6.2f  vy %+6.2f m/s", state(0),
                            state(2), state(1), state(3)));
        lines.push_back(fmt("spin   omega %+7.3f rad/s   R %.3f m   |v| %.2f m/s", state(7),
                            state(8), std::hypot(state(1), state(3))));
        if (result.state != TrackingState::NoTarget)
        {
            lines.push_back(fmt("aim    slot %d  yaw %+7.1f  pitch %+6.1f  d %.2f m   fire %s "
                                "(%.2f deg)",
                                diagnostics.aim_slot, result.target_yaw * 180.0 / kPi,
                                result.target_pitch * 180.0 / kPi, result.distance_m,
                                result.fire_allowed ? "YES" : "no",
                                std::fabs(result.target_yaw - frame.gimbal.yaw) * 180.0 / kPi));
        }
        else
        {
            lines.push_back(fmt("aim    none   %s", diagnostics.stability.c_str()));
        }
    }
    else
    {
        lines.push_back("center --");
        lines.push_back("spin   --");
        lines.push_back(fmt("waiting for lock   %d samples collected", diagnostics.lock_samples));
    }
    lines.push_back(fmt("obs    %zu seen   upd %d  rej %d  miss %d   NIS %.1f   sigma %.1f mm",
                        frame.observations.size(), diagnostics.updates_this_frame,
                        diagnostics.rejected_this_frame, diagnostics.missed_frames,
                        diagnostics.last_nis, diagnostics.observation_noise_m * 1000.0));

    if (options_.show_hud)
    {
        int panel_width = 0;
        for (const std::string &line : lines)
        {
            int baseline = 0;
            const cv::Size size =
                cv::getTextSize(line, cv::FONT_HERSHEY_SIMPLEX, font_scale, 1, &baseline);
            panel_width = std::max(panel_width, size.width);
        }
        const cv::Rect panel(pad, pad, panel_width + 2 * pad,
                             static_cast<int>(lines.size()) * line_height + 2 * pad);
        fillAlpha(canvas_, panel, scene.bg, 0.80);
        cv::rectangle(canvas_, panel, mix(scene.bg, scene.ink, 0.22), 1, cv::LINE_AA);
        for (std::size_t i = 0; i < lines.size(); ++i)
        {
            textAt(canvas_, lines[i],
                   cv::Point(panel.x + pad,
                             panel.y + pad + static_cast<int>(i) * line_height + line_height - pad),
                   font_scale, scene.ink, i == 0 ? 2 : 1);
        }
    }

    // ---- 7. 底部图例 + 比例尺 ----
    const cv::Rect footer(static_cast<int>(std::lround(10.0 * scene.scale)),
                          height - footer_height - static_cast<int>(std::lround(10.0 * scene.scale)),
                          width - static_cast<int>(std::lround(20.0 * scene.scale)), footer_height);
    fillAlpha(canvas_, footer, scene.bg, 0.78);
    cv::rectangle(canvas_, footer, mix(scene.bg, scene.ink, 0.18), 1, cv::LINE_AA);

    const double legend_scale = 0.42 * scene.scale;
    const int legend_start_x = footer.x + pad;
    int legend_x = legend_start_x;
    const int row1 = footer.y + static_cast<int>(std::lround(20.0 * scene.scale));
    const int row2 = footer.y + static_cast<int>(std::lround(42.0 * scene.scale));
    auto legendItem = [&](int y, const cv::Scalar &color, int kind, const char *label) {
        const int swatch = static_cast<int>(std::lround(16.0 * scene.scale));
        if (kind == 0)
        {
            cv::circle(canvas_, cv::Point(legend_x + swatch / 2, y), 5, color, -1, cv::LINE_AA);
        }
        else if (kind == 1)
        {
            cv::line(canvas_, cv::Point(legend_x, y), cv::Point(legend_x + swatch, y), color, 6,
                     cv::LINE_AA);
        }
        else if (kind == 2)
        {
            const cv::Point c(legend_x + swatch / 2, y);
            cv::line(canvas_, c + cv::Point(-5, 0), c + cv::Point(5, 0), color, 2, cv::LINE_AA);
            cv::line(canvas_, c + cv::Point(0, -5), c + cv::Point(0, 5), color, 2, cv::LINE_AA);
        }
        else
        {
            cv::line(canvas_, cv::Point(legend_x, y), cv::Point(legend_x + swatch, y), color, 2,
                     cv::LINE_AA);
        }
        legend_x += swatch + static_cast<int>(std::lround(6.0 * scene.scale));
        textAt(canvas_, label, cv::Point(legend_x, y + static_cast<int>(std::lround(5.0 * scene.scale))),
               legend_scale, scene.ink);
        legend_x += cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, legend_scale, 1, nullptr).width +
                    static_cast<int>(std::lround(18.0 * scene.scale));
    };
    if (options_.show_legend)
    {
        legendItem(row1, kCenterColor, 0, "body center");
        legendItem(row1, kPlateColor, 1, "armor (EKF model)");
        legendItem(row1, kAimPlateColor, 1, "aim plate");
        legend_x = legend_start_x; // 第二行从左边重新排，别接着第一行往后挤
        legendItem(row2, kObsColor, 0, "observed (PnP)");
        legendItem(row2, kAimColor, 2, "aim point");
        legendItem(row2, kTrailColor, 3, "trail / velocity");
    }

    // 比例尺：从 0.1/0.2/0.5/1/2/5 m 里挑一个画出来约 120 px 的长度。
    {
        const double candidates[] = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0};
        double bar_m = candidates[0];
        double best_error = std::numeric_limits<double>::infinity();
        for (double candidate : candidates)
        {
            const double error = std::fabs(candidate * scene.ppm - 120.0);
            if (error < best_error)
            {
                best_error = error;
                bar_m = candidate;
            }
        }
        const int bar_px = static_cast<int>(std::lround(bar_m * scene.ppm));
        // 比例尺放在图例面板的右端、和两行图例同一水平带内，避免和图例文字打架。
        const int bar_y = footer.y + footer.height / 2 + 1;
        const int bar_x1 = footer.x + footer.width - pad;
        const int bar_x0 = bar_x1 - bar_px;
        if (bar_x0 - legend_start_x > 220 * scene.scale)
        {
            cv::line(canvas_, cv::Point(bar_x0, bar_y), cv::Point(bar_x1, bar_y), scene.ink, 2,
                     cv::LINE_AA);
            cv::line(canvas_, cv::Point(bar_x0, bar_y - 5), cv::Point(bar_x0, bar_y + 5), scene.ink,
                     2, cv::LINE_AA);
            cv::line(canvas_, cv::Point(bar_x1, bar_y - 5), cv::Point(bar_x1, bar_y + 5), scene.ink,
                     2, cv::LINE_AA);
            const std::string label = fmt("%.4g m", bar_m);
            const cv::Size size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, legend_scale, 1,
                                                  nullptr);
            textAt(canvas_, label,
                   cv::Point(bar_x0 - size.width - static_cast<int>(std::lround(8.0 * scene.scale)),
                             bar_y + static_cast<int>(std::lround(4.0 * scene.scale))),
                   legend_scale, scene.ink);
        }
    }

    // ---- 8. 方向标：+x 朝上、+y 朝左（和世界系一致） ----
    {
        const int ax = width - static_cast<int>(std::lround(46.0 * scene.scale));
        const int ay = static_cast<int>(std::lround(96.0 * scene.scale));
        const int arm = static_cast<int>(std::lround(30.0 * scene.scale));
        cv::arrowedLine(canvas_, cv::Point(ax, ay), cv::Point(ax, ay - arm), scene.ink, 2,
                        cv::LINE_AA, 0, 0.35);
        cv::arrowedLine(canvas_, cv::Point(ax, ay), cv::Point(ax - arm, ay), scene.ink, 2,
                        cv::LINE_AA, 0, 0.35);
        textAt(canvas_, "+x", cv::Point(ax + 5, ay - arm + 4), 0.42 * scene.scale, scene.ink);
        textAt(canvas_, "+y", cv::Point(ax - arm - 20, ay + 4), 0.42 * scene.scale, scene.ink);
    }

    return canvas_;
}

} // namespace rotating_armor
