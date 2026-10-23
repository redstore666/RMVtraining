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

} // namespace rotating_armor
