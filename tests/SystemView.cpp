#include "SystemView.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>

#include <opencv2/imgproc.hpp>

namespace rotating_armor
{
namespace
{

// 卡片右上角显示的数字格式：按量级选小数位，避免 "0.000" 与 "1234567.891" 都难看。
std::string formatValue(double value, const std::string &unit)
{
    if (!std::isfinite(value))
    {
        return "--";
    }
    const double magnitude = std::fabs(value);
    const int decimals = magnitude >= 100.0 ? 0 : magnitude >= 10.0 ? 1 : magnitude >= 1.0 ? 2 : 3;
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.*f", decimals, value);
    std::string text(buffer);
    if (!unit.empty())
    {
        text += " ";
        text += unit;
    }
    return text;
}

// 一条曲线的颜色（BGR）。下标越小越靠前的卡片，颜色越深。
cv::Scalar traceColour(std::size_t index)
{
    static const cv::Scalar palette[] = {
        cv::Scalar(200, 70, 0),   // 蓝
        cv::Scalar(0, 110, 200),  // 橙
        cv::Scalar(40, 140, 40),  // 绿
        cv::Scalar(180, 40, 160), // 紫
        cv::Scalar(160, 160, 0),  // 青
        cv::Scalar(30, 30, 190),  // 红
    };
    return palette[index % (sizeof(palette) / sizeof(palette[0]))];
}

// 取窗口内的样本：返回 [first, last) 区间，保证至少包含一个点。
void windowRange(const TelemetryTrace &trace, double now_s, std::size_t &first, std::size_t &last,
                 double &vmin, double &vmax)
{
    const double from_s = now_s - trace.window_s;
    first = trace.samples.size();
    for (std::size_t i = 0; i < trace.samples.size(); ++i)
    {
        if (trace.samples[i].first >= from_s)
        {
            first = i;
            break;
        }
    }
    // 全部都太旧时，至少保留最后一个点（否则折线会突然消失）。
    if (first >= trace.samples.size() && !trace.samples.empty())
    {
        first = trace.samples.size() - 1;
    }
    last = trace.samples.size();
    if (first >= last)
    {
        first = 0;
        last = 0;
        vmin = 0.0;
        vmax = 1.0;
        return;
    }

    vmin = std::numeric_limits<double>::infinity();
    vmax = -std::numeric_limits<double>::infinity();
    for (std::size_t i = first; i < last; ++i)
    {
        const double value = trace.samples[i].second;
        if (!std::isfinite(value))
        {
            continue;
        }
        vmin = std::min(vmin, value);
        vmax = std::max(vmax, value);
    }
    if (!std::isfinite(vmin) || !std::isfinite(vmax))
    {
        vmin = 0.0;
        vmax = 1.0;
        return;
    }
    if (vmax - vmin < 1e-9)
    {
        // 常数序列：上下各留一点，线画在中间而不是贴边。
        const double margin = std::max(1e-6, std::fabs(vmax) * 0.1);
        vmin -= margin;
        vmax += margin;
    }
}

void drawSparkline(cv::Mat &canvas, const cv::Rect &box, const TelemetryTrace &trace,
                   double now_s, const cv::Scalar &colour)
{
    cv::rectangle(canvas, box, cv::Scalar(234, 234, 234), cv::FILLED);
    if (box.height < 6 || box.width < 6)
    {
        return;
    }

    std::size_t first = 0;
    std::size_t last = 0;
    double vmin = 0.0;
    double vmax = 1.0;
    windowRange(trace, now_s, first, last, vmin, vmax);

    const double from_s = now_s - trace.window_s;
    const double span_s = std::max(1e-6, trace.window_s);
    auto x_of = [&](double t)
    {
        const double alpha = std::clamp((t - from_s) / span_s, 0.0, 1.0);
        return box.x + static_cast<int>(std::lround(alpha * (box.width - 1)));
    };
    auto y_of = [&](double v)
    {
        const double alpha = std::clamp((v - vmin) / (vmax - vmin), 0.0, 1.0);
        return box.y + box.height - 1 - static_cast<int>(std::lround(alpha * (box.height - 1)));
    };

    // 竖向秒刻度（每 window/4 一格），让"抖不抖"有个时间参照。
    for (int i = 1; i < 4; ++i)
    {
        const int gx = box.x + box.width * i / 4;
        cv::line(canvas, cv::Point(gx, box.y), cv::Point(gx, box.y + box.height - 1),
                 cv::Scalar(222, 222, 222), 1, cv::LINE_AA);
    }

    // 0 线：量纲上"过零"的曲线（速度/偏差）需要一条基准。
    if (vmin < 0.0 && vmax > 0.0)
    {
        const int zy = y_of(0.0);
        cv::line(canvas, cv::Point(box.x, zy), cv::Point(box.x + box.width - 1, zy),
                 cv::Scalar(205, 205, 205), 1, cv::LINE_AA);
    }

    if (last - first == 1)
    {
        cv::circle(canvas, cv::Point(x_of(trace.samples[first].first),
                                     y_of(trace.samples[first].second)),
                   2, colour, cv::FILLED, cv::LINE_AA);
        return;
    }

    for (std::size_t i = first + 1; i < last; ++i)
    {
        const cv::Point p0(x_of(trace.samples[i - 1].first), y_of(trace.samples[i - 1].second));
        const cv::Point p1(x_of(trace.samples[i].first), y_of(trace.samples[i].second));
        cv::line(canvas, p0, p1, colour, 1, cv::LINE_AA);
    }

    // 右上/右下角标出窗口内的极值，省得为了读数去量像素。
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.*f", std::fabs(vmax) >= 10.0 ? 1 : 2, vmax);
    cv::putText(canvas, buffer, cv::Point(box.x + box.width - 64, box.y + 11),
                cv::FONT_HERSHEY_SIMPLEX, 0.36, cv::Scalar(140, 140, 140), 1, cv::LINE_AA);
    std::snprintf(buffer, sizeof(buffer), "%.*f", std::fabs(vmin) >= 10.0 ? 1 : 2, vmin);
    cv::putText(canvas, buffer, cv::Point(box.x + box.width - 64, box.y + box.height - 3),
                cv::FONT_HERSHEY_SIMPLEX, 0.36, cv::Scalar(140, 140, 140), 1, cv::LINE_AA);
}

void drawCard(cv::Mat &canvas, cv::Point origin, int width, int height, const TelemetryTrace &trace,
              double now_s, const cv::Scalar &colour)
{
    const cv::Rect card(origin.x, origin.y, width, height);
    cv::rectangle(canvas, card, cv::Scalar(247, 247, 247), cv::FILLED);
    cv::rectangle(canvas, card, cv::Scalar(212, 212, 212), 1, cv::LINE_AA);

    // 字号是"最终显示像素"，卡片按显示尺寸直接画（不再整体缩图），所以这里放大到
    // 0.5 左右：1080p 上一个字符约 15 px，隔着屏幕也能读。
    const double name_font = 0.50;
    const double value_font = 0.55;
    cv::putText(canvas, trace.name, cv::Point(card.x + 9, card.y + 19),
                cv::FONT_HERSHEY_SIMPLEX, name_font, cv::Scalar(80, 80, 80), 1, cv::LINE_AA);
    cv::rectangle(canvas, cv::Rect(card.x + 8, card.y + 6, 4, 15), colour, cv::FILLED);

    const std::string value =
        !trace.text.empty() ? trace.text : formatValue(trace.latest(), trace.unit);
    const cv::Size value_size =
        cv::getTextSize(value, cv::FONT_HERSHEY_SIMPLEX, value_font, 1, nullptr);
    cv::putText(canvas, value,
                cv::Point(card.x + card.width - 9 - value_size.width, card.y + 20),
                cv::FONT_HERSHEY_SIMPLEX, value_font, cv::Scalar(20, 20, 20), 1, cv::LINE_AA);

    const int spark_y = card.y + 27;
    const int spark_h = card.height - 34;
    if (spark_h >= 6)
    {
        drawSparkline(canvas,
                      cv::Rect(card.x + 8, spark_y, card.width - 16, spark_h),
                      trace, now_s, colour);
    }
}

// 画一列卡片，返回已用掉的高度（相对列顶）。
int drawColumn(cv::Mat &canvas, int x, int width, int top, int avail_height, int card_height,
               const std::deque<TelemetryTrace> &traces, const std::string &title,
               const SystemViewOptions &options, double now_s)
{
    cv::rectangle(canvas, cv::Rect(x, top, width, options.title_height), cv::Scalar(236, 236, 236),
                  cv::FILLED);
    cv::putText(canvas, title, cv::Point(x + 9, top + 20), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(60, 60, 60), 1, cv::LINE_AA);

    int y = top + options.title_height + options.card_gap / 2;
    for (std::size_t i = 0; i < traces.size(); ++i)
    {
        // 放不下就不再画（宁可少画，也不把卡片压扁成看不清）。
        if (y + card_height > top + avail_height)
        {
            break;
        }
        drawCard(canvas, cv::Point(x, y), width, card_height, traces[i], now_s,
                 traceColour(i));
        y += card_height + options.card_gap;
    }
    return y - top;
}

double latestTimestamp(const std::deque<TelemetryTrace> &traces, double fallback)
{
    double now_s = fallback;
    for (const auto &trace : traces)
    {
        if (!trace.samples.empty())
        {
            now_s = std::max(now_s, trace.samples.back().first);
        }
    }
    return now_s;
}

} // namespace

void TelemetryTrace::push(double timestamp_s, double value)
{
    // 时间回退（换了时钟/重置过）时直接清空，免得折线来回折。
    if (!samples.empty() && timestamp_s < samples.back().first)
    {
        samples.clear();
    }
    samples.emplace_back(timestamp_s, value);
    const double cutoff_s = timestamp_s - std::max(window_s, 1e-3);
    while (!samples.empty() && samples.front().first < cutoff_s)
    {
        samples.pop_front();
    }
}

void TelemetryTrace::clear()
{
    samples.clear();
}

double TelemetryTrace::latest() const noexcept
{
    return samples.empty() ? std::numeric_limits<double>::quiet_NaN() : samples.back().second;
}

double TelemetryTrace::minimum() const noexcept
{
    double value = std::numeric_limits<double>::infinity();
    for (const auto &sample : samples)
    {
        value = std::min(value, sample.second);
    }
    return std::isfinite(value) ? value : std::numeric_limits<double>::quiet_NaN();
}

double TelemetryTrace::maximum() const noexcept
{
    double value = -std::numeric_limits<double>::infinity();
    for (const auto &sample : samples)
    {
        value = std::max(value, sample.second);
    }
    return std::isfinite(value) ? value : std::numeric_limits<double>::quiet_NaN();
}

SystemView::SystemView(SystemViewOptions options) : options_(std::move(options)) {}

TelemetryTrace &SystemView::trace(std::deque<TelemetryTrace> &traces, const std::string &name,
                                  const std::string &unit)
{
    for (auto &existing : traces)
    {
        if (existing.name == name)
        {
            if (existing.unit.empty() && !unit.empty())
            {
                existing.unit = unit;
            }
            return existing;
        }
    }
    traces.push_back(TelemetryTrace{});
    traces.back().name = name;
    traces.back().unit = unit;
    traces.back().window_s = 10.0;
    return traces.back();
}

TelemetryTrace &SystemView::leftTrace(const std::string &name, const std::string &unit)
{
    return trace(left_, name, unit);
}

TelemetryTrace &SystemView::rightTrace(const std::string &name, const std::string &unit)
{
    return trace(right_, name, unit);
}

void SystemView::reset()
{
    for (auto &trace : left_)
    {
        trace.clear();
    }
    for (auto &trace : right_)
    {
        trace.clear();
    }
}

cv::Mat SystemView::compose(const cv::Mat &camera, const cv::Mat &overview) const
{
    if (camera.empty() || camera.type() != CV_8UC3)
    {
        return cv::Mat{};
    }

    const int left_w = options_.show_left ? options_.left_width : 0;
    const int right_w = options_.show_right ? options_.right_width : 0;
    const int left_n = static_cast<int>(left_.size());
    const int right_n = static_cast<int>(right_.size());

    // 一列需要的高度：标题 + n 张卡 + 卡片间隙。
    const auto column_height = [&](int count, int card_h)
    {
        if (count <= 0)
        {
            return 0;
        }
        return options_.title_height + count * card_h + (count - 1) * options_.card_gap;
    };

    // 画布高度先由"相机画面"和"右列卡片 + 俯视图最小高度"里的大者决定，
    // 相机画面在中间垂直居中；否则俯视图会被右列卡片挤成一个小方块。
    int content_h = camera.rows;
    if (right_w > 0)
    {
        int right_need = column_height(right_n, options_.card_height);
        if (!overview.empty())
        {
            right_need += options_.card_gap + options_.overview_min_height;
        }
        content_h = std::max(content_h, right_need);
    }
    if (left_w > 0)
    {
        content_h = std::max(content_h, column_height(left_n, options_.card_height));
    }

    int x = options_.pad;
    if (left_w > 0)
    {
        x += left_w + options_.pad;
    }
    const int camera_x = x;
    x += camera.cols + options_.pad;
    const int right_x = x;
    if (right_w > 0)
    {
        x += right_w + options_.pad;
    }
    const int width = x;
    const int height = options_.pad + content_h + options_.pad;

    const cv::Scalar background =
        options_.white_background ? cv::Scalar(255, 255, 255) : cv::Scalar(28, 28, 28);
    cv::Mat canvas(height, width, CV_8UC3, background);

    // 相机画面已由调用方缩到显示尺寸，这里只做平移，绝不缩放——文字才是清晰的。
    const int camera_y = options_.pad + (content_h - camera.rows) / 2;
    camera.copyTo(canvas(cv::Rect(camera_x, camera_y, camera.cols, camera.rows)));
    cv::rectangle(canvas, cv::Rect(camera_x - 1, camera_y - 1, camera.cols + 2, camera.rows + 2),
                  cv::Scalar(190, 190, 190), 1, cv::LINE_AA);

    const double now_s = latestTimestamp(left_, latestTimestamp(right_, 0.0));

    if (left_w > 0 && left_n > 0)
    {
        int left_card_h = options_.left_card_height;
        if (left_card_h <= 0)
        {
            // 自动摊开：把整列高度平分给 n 张卡，至少 card_height、至多上限。
            const int avail =
                content_h - options_.title_height - (left_n - 1) * options_.card_gap;
            left_card_h = std::clamp(avail / left_n, options_.card_height,
                                     options_.left_card_max_height);
        }
        drawColumn(canvas, options_.pad, left_w, options_.pad, content_h, left_card_h, left_,
                   options_.left_title, options_, now_s);
    }

    if (right_w > 0)
    {
        const int right_used =
            drawColumn(canvas, right_x, right_w, options_.pad, content_h, options_.card_height,
                       right_, options_.right_title, options_, now_s);
        // 俯视图塞进右列剩下的空间（等比缩成正方形，居中）。
        if (!overview.empty())
        {
            const int avail = content_h - right_used - options_.card_gap;
            const int natural = std::max(overview.cols, overview.rows);
            if (avail > 80)
            {
                const int side = std::min({right_w, avail, natural});
                const double scale = static_cast<double>(side) / natural;
                const int w = std::max(1, static_cast<int>(std::lround(overview.cols * scale)));
                const int h = std::max(1, static_cast<int>(std::lround(overview.rows * scale)));
                cv::Mat scaled = overview;
                if (w != overview.cols || h != overview.rows)
                {
                    cv::resize(overview, scaled, cv::Size(w, h), 0, 0, cv::INTER_AREA);
                }
                const int ox = right_x + (right_w - w) / 2;
                const int oy = options_.pad + right_used + (avail - h) / 2;
                if (ox >= 0 && oy >= 0 && ox + w <= canvas.cols && oy + h <= canvas.rows)
                {
                    scaled.copyTo(canvas(cv::Rect(ox, oy, w, h)));
                }
            }
        }
    }
    return canvas;
}


// ===== 迷你俯视图 =====
//
// 只画"车在哪里、板朝哪边、瞄哪一点"，不画相机画面、不画网格/轨迹/图例：
// 右下角那一小块地方要一眼看懂，而不是塞满信息。
cv::Mat renderMiniTopDown(const MiniTopDownFrame &frame, const MiniTopDownOptions &options)
{
    const int width = std::max(80, options.width);
    const int height = std::max(60, options.height);
    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    // 画布坐标：右 = +x，上 = +y（与俯视图 HUD 的 +x/+y 一致）。
    //
    // 缩放口径：**以车为准**（4 块板 + 车心 + 瞄准点），按包围盒自动定尺度。
    // 相机在 2 m 外，若把它也算进包围盒，车就只有几十像素、板看不清——所以
    // 默认只用车定尺度，相机落在画面外时贴边画一个箭头 + 距离，"往哪边、多远"
    // 照样一眼看到。options.fit_camera=true 才是"相机与车一起装进画面"的字面口径。
    std::vector<Eigen::Vector2d> vehicle;
    if (frame.has_body)
        vehicle.emplace_back(frame.body_centre.x(), frame.body_centre.y());
    for (const Eigen::Vector3d &centre : frame.plate_centres)
        vehicle.emplace_back(centre.x(), centre.y());
    if (frame.has_valid_aim)
        vehicle.emplace_back(frame.valid_aim.x(), frame.valid_aim.y());
    if (frame.has_predicted_aim)
        vehicle.emplace_back(frame.predicted_aim.x(), frame.predicted_aim.y());

    // 只有相机、没有任何车体信息时，退回"以原点为中心"的默认视野。
    if (vehicle.empty())
    {
        vehicle.emplace_back(0.0, 0.0);
        vehicle.emplace_back(frame.camera_position.x(), frame.camera_position.y());
    }

    double min_x = vehicle.front().x(), max_x = min_x;
    double min_y = vehicle.front().y(), max_y = min_y;
    for (const Eigen::Vector2d &p : vehicle)
    {
        min_x = std::min(min_x, p.x());
        max_x = std::max(max_x, p.x());
        min_y = std::min(min_y, p.y());
        max_y = std::max(max_y, p.y());
    }
    double centre_x = 0.5 * (min_x + max_x);
    double centre_y = 0.5 * (min_y + max_y);
    double half_x = 0.5 * (max_x - min_x);
    double half_y = 0.5 * (max_y - min_y);

    if (options.fit_camera)
    {
        // 字面口径：相机与车一起装进画面。
        const double cam_x = frame.camera_position.x();
        const double cam_y = frame.camera_position.y();
        min_x = std::min(min_x, cam_x);
        max_x = std::max(max_x, cam_x);
        min_y = std::min(min_y, cam_y);
        max_y = std::max(max_y, cam_y);
        centre_x = 0.5 * (min_x + max_x);
        centre_y = 0.5 * (min_y + max_y);
        half_x = 0.5 * (max_x - min_x);
        half_y = 0.5 * (max_y - min_y);
    }

    // 视野半径 = 内容半宽 + 外扩；再兜一个下限，避免"全在原点"时放大到无穷。
    double half_extent = std::max(half_x, half_y) + options.margin_m;
    half_extent = std::max(half_extent, options.min_half_extent_m);
    const double ppm = 0.5 * static_cast<double>(std::min(width, height)) / half_extent;

    const auto to_px = [&](double x, double y)
    {
        return cv::Point(cvRound(0.5 * width + (x - centre_x) * ppm),
                         cvRound(0.5 * height - (y - centre_y) * ppm));
    };
    const auto to_px3 = [&](const Eigen::Vector3d &p) { return to_px(p.x(), p.y()); };
    const cv::Rect canvas_rect(0, 0, width, height);

    // 板：一段粗线（长度 = 整板宽），aim 板多画一圈白描边。
    // 整车建模后四块板都由 (车心, R, φ) 的刚体给出、一起随 φ 转，
    // 半宽用真实装甲板尺寸（小 0.0675 / 大 0.115），所以画出来是四段
    // 互不相连的短线段，而不是一个"方框"。
    for (std::size_t i = 0; i < frame.plate_centres.size(); ++i)
    {
        const double yaw = i < frame.plate_yaws.size() ? frame.plate_yaws[i] : 0.0;
        const double half_w = i < frame.plate_half_widths.size() ? frame.plate_half_widths[i]
                                                                : 0.115;
        if (!std::isfinite(yaw))
            continue;
        // 与 TopDownView 的板绘制同一定义：tangent = (-sin(yaw), cos(yaw))。
        const double tx = -std::sin(yaw);
        const double ty = std::cos(yaw);
        const cv::Point a = to_px(frame.plate_centres[i].x() - half_w * tx,
                                  frame.plate_centres[i].y() - half_w * ty);
        const cv::Point b = to_px(frame.plate_centres[i].x() + half_w * tx,
                                  frame.plate_centres[i].y() + half_w * ty);
        const bool is_aim = (frame.aim_slot >= 0 && static_cast<int>(i) == frame.aim_slot);
        if (is_aim)
        {
            cv::line(canvas, a, b, cv::Scalar(255, 255, 255), 6, cv::LINE_AA);
        }
        if (static_cast<int>(i) == frame.highlight_slot)
        {
            cv::line(canvas, a, b, cv::Scalar(0, 255, 255), 4, cv::LINE_AA); // 亮黄=落点
        }
        else
        {
            cv::line(canvas, a, b, cv::Scalar(150, 150, 150), 3, cv::LINE_AA); // 灰=模型板
        }
    }

    // 实测板（PnP 原始观测，不经过 EKF）：绿色细线 + 中心点。
    // 和灰色模型板并排看，一眼就能分辨"模型转慢了"和"观测在跳"：
    //   · 绿线整圈稳定、灰线落后一个角度 → 转速/相位滞后；
    //   · 绿线自己乱跳、灰线平稳      → 观测（法向/关联）有问题，不是滤波器。
    for (std::size_t i = 0; i < frame.detected_centres.size(); ++i)
    {
        const double yaw = i < frame.detected_yaws.size() ? frame.detected_yaws[i] : 0.0;
        const double half_w = i < frame.detected_half_widths.size()
                                  ? frame.detected_half_widths[i]
                                  : 0.0675;
        if (!std::isfinite(yaw))
            continue;
        const double tx = -std::sin(yaw);
        const double ty = std::cos(yaw);
        const cv::Point a = to_px(frame.detected_centres[i].x() - half_w * tx,
                                  frame.detected_centres[i].y() - half_w * ty);
        const cv::Point b = to_px(frame.detected_centres[i].x() + half_w * tx,
                                  frame.detected_centres[i].y() + half_w * ty);
        cv::line(canvas, a, b, cv::Scalar(80, 220, 80), 2, cv::LINE_AA);
        cv::circle(canvas, to_px3(frame.detected_centres[i]), 2, cv::Scalar(80, 220, 80), -1,
                   cv::LINE_AA);
    }

    // 云台原点：灰色空心小圆（"相机装在云台前面一点"，两者要能分开看）。
    const cv::Point origin_px = to_px(0.0, 0.0);
    if (canvas_rect.contains(origin_px))
        cv::circle(canvas, origin_px, 4, cv::Scalar(120, 120, 120), 1, cv::LINE_AA);

    // 相机位置：白色小方块 + 朝 +x 的短线。落在画面外时贴边画，指着真实方向，
    // 并把距离写出来——"车往哪边走、离相机多远"不用再回头看数字。
    const cv::Point cam_raw = to_px(frame.camera_position.x(), frame.camera_position.y());
    const int edge = 12;
    const bool cam_visible = cam_raw.x >= edge && cam_raw.x < width - edge &&
                             cam_raw.y >= edge && cam_raw.y < height - edge;
    cv::Point cam = cam_raw;
    if (!cam_visible)
    {
        const cv::Point2d mid(0.5 * width, 0.5 * height);
        cv::Point2d dir(cam_raw.x - mid.x, cam_raw.y - mid.y);
        const double len = std::hypot(dir.x, dir.y);
        if (len < 1e-6)
            dir = cv::Point2d(-1.0, 0.0), dir = dir / cv::norm(dir);
        else
            dir = dir / len;
        const double limit_x = (0.5 * width - edge) / std::max(std::fabs(dir.x), 1e-6);
        const double limit_y = (0.5 * height - edge) / std::max(std::fabs(dir.y), 1e-6);
        const double t = std::min(limit_x, limit_y);
        cam = cv::Point(cvRound(mid.x + dir.x * t), cvRound(mid.y + dir.y * t));
        // 指外的小三角：说明"相机在这个方向的画面外"。
        const cv::Point tip(cvRound(cam.x - dir.x * 8), cvRound(cam.y - dir.y * 8));
        const cv::Point side(dir.y * 5, -dir.x * 5);
        const std::vector<cv::Point> arrow = {tip, cv::Point(cvRound(cam.x + side.x),
                                                             cvRound(cam.y + side.y)),
                                              cv::Point(cvRound(cam.x - side.x),
                                                        cvRound(cam.y - side.y))};
        cv::fillConvexPoly(canvas, arrow, cv::Scalar(255, 255, 255), cv::LINE_AA);
    }
    else
    {
        cv::rectangle(canvas, cv::Rect(cam.x - 4, cam.y - 4, 8, 8), cv::Scalar(255, 255, 255),
                      cv::FILLED);
        cv::line(canvas, cam, cam + cv::Point(16, 0), cv::Scalar(255, 255, 255), 1,
                 cv::LINE_AA);
    }

    // 车心：红色实心圆 + 十字。
    if (frame.has_body)
    {
        const cv::Point body = to_px3(frame.body_centre);
        cv::circle(canvas, body, 5, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_AA);
        cv::drawMarker(canvas, body, cv::Scalar(0, 0, 255), cv::MARKER_CROSS, 14, 1);
    }

    // 车体速度箭头（橙）：从车心指向"velocity_arrow_s 秒后的位置"，尖端标 |v|。
    // 口径与 TopDownView 的 kVelocityColor 一致（BGR 0,130,255 = 橙）。
    //
    // 为什么必须画在俯视图上而不是只写进 caption：速度是一个**矢量**，矢量的
    // 方向（车在往哪走）和大小同样重要，纯数字看不出来。而且车体中心和速度
    // 落在同一张俯视图上，能直接看出"中心在漂但速度是 0"这种自相矛盾的状态
    // ——那正是 EKF 被单板观测带飞的典型表现。
    if (options.show_velocity && frame.has_velocity && frame.has_body)
    {
        const double vx = frame.velocity.x();
        const double vy = frame.velocity.y();
        const double speed = std::hypot(vx, vy);
        if (std::isfinite(speed) && speed > 0.02)
        {
            const cv::Point body = to_px3(frame.body_centre);
            // 箭头代表 velocity_arrow_s 秒位移；若太短（慢速目标），兜底成 0.02 m 的位移，
            // 再不够就由下面的最小像素长度兜住，保证方向永远看得见。
            const double arrow_s = std::max(options.velocity_arrow_s, 0.02 / speed);
            cv::Point tip = to_px(frame.body_centre.x() + vx * arrow_s,
                                  frame.body_centre.y() + vy * arrow_s);
            const double length_px = cv::norm(tip - body);
            constexpr double kMinPx = 16.0;
            // 方向退化（速度极小）时不画箭头；反过来画的会是一个随机朝向的箭头。
            // 注意这里不能 return——后面还有瞄准点/准星要画。
            const bool degenerate = length_px < kMinPx && length_px < 1e-6;
            if (!degenerate)
            {
                if (length_px < kMinPx)
                {
                    const cv::Point2d dir = (cv::Point2d(tip) - cv::Point2d(body)) / length_px;
                    tip = cv::Point(cvRound(body.x + dir.x * kMinPx),
                                    cvRound(body.y + dir.y * kMinPx));
                }
                const double tip_len = std::clamp(14.0 / std::max(length_px, 1.0), 0.12, 0.40);
                cv::arrowedLine(canvas, body, tip, cv::Scalar(0, 130, 255), 2, cv::LINE_AA, 0,
                                tip_len);
                // |v| 标在箭头尖端外侧，别压住箭身。
                const cv::Point2d d = cv::Point2d(tip) - cv::Point2d(body);
                const double dn = std::max(std::hypot(d.x, d.y), 1e-6);
                const cv::Point label(cvRound(tip.x + d.x / dn * 6.0 + 3),
                                      cvRound(tip.y + d.y / dn * 6.0 - 3));
                cv::putText(canvas, cv::format("%.2f", speed), label, cv::FONT_HERSHEY_SIMPLEX,
                            0.36, cv::Scalar(0, 130, 255), 1, cv::LINE_AA);
            }
        }
    }

    // 瞄准：青 = 有效（当前板位，不含提前量），黄 = 预测（含提前量/飞行时间）。
    // 允许开火时预测点变绿并画一条相机→瞄准点的虚线，一眼看出"现在能不能打"。
    if (frame.has_predicted_aim)
    {
        const cv::Point p = to_px3(frame.predicted_aim);
        if (frame.fire_allowed)
        {
            cv::circle(canvas, p, 6, cv::Scalar(0, 255, 0), cv::FILLED, cv::LINE_AA);
            // 虚线：32 位 alpha 通道画不了，这里用分段线模拟。
            const int segments = 12;
            for (int si = 0; si < segments; si += 2)
            {
                const double t0 = static_cast<double>(si) / segments;
                const double t1 = static_cast<double>(si + 1) / segments;
                const cv::Point a(cvRound(cam.x + (p.x - cam.x) * t0),
                                  cvRound(cam.y + (p.y - cam.y) * t0));
                const cv::Point b(cvRound(cam.x + (p.x - cam.x) * t1),
                                  cvRound(cam.y + (p.y - cam.y) * t1));
                cv::line(canvas, a, b, cv::Scalar(0, 160, 0), 1, cv::LINE_AA);
            }
        }
        else
        {
            cv::drawMarker(canvas, p, cv::Scalar(0, 200, 255), cv::MARKER_TILTED_CROSS, 18, 2);
        }
    }
    if (frame.has_valid_aim)
    {
        const cv::Point p = to_px3(frame.valid_aim);
        cv::circle(canvas, p, 7, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
        cv::drawMarker(canvas, p, cv::Scalar(255, 255, 0), cv::MARKER_CROSS, 12, 1);
    }

    // 左下角小字：距离/半径/角速度；没有目标时明确写 no target。
    std::string caption = frame.caption.empty()
        ? (frame.has_body ? std::string() : std::string("no target"))
        : frame.caption;
    // 速度分量追加在 caption 末尾（矢量的模长已在箭头尖端标过，这里补 vx/vy）。
    // 用 " | " 分隔而不是另起一行：caption 只有一行的高度预算，换行会压到画面里。
    if (options.show_velocity && frame.has_velocity && frame.has_body)
    {
        const double vx = frame.velocity.x();
        const double vy = frame.velocity.y();
        if (std::isfinite(vx) && std::isfinite(vy))
        {
            const std::string vel = cv::format("vx%+.2f vy%+.2f |v|%.2f m/s", vx, vy,
                                               std::hypot(vx, vy));
            caption = caption.empty() ? vel : caption + "  " + vel;
        }
    }
    if (!caption.empty())
    {
        cv::putText(canvas, caption, cv::Point(8, height - 8), cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
    }
    return canvas;
}

} // namespace rotating_armor
