#pragma once

// 系统总览图：把"信息面板 + 相机画面 + 俯视图"拼成一张画布。
//
// 背景：web_debug（web_debug/ 目录，WebLogger + 网页折线图）能看遥测，但它是
// 另一个进程的浏览器页面，现场调参时"网页看曲线、窗口看画面"要来回切。这里把
// 同一批键（WEB_LOG 的键名）画成迷你卡片，与相机画面、EKF 俯视图并排拼成一张
// 图，一次 imshow 就能全看：左列 = 取图/检测耗时类，右列 = EKF/瞄准类，中间 =
// 相机画面，右下 = 俯视图。
//
// 本模块只做"画"，不含任何跟踪/检测逻辑，也不改 web_debug 的任何行为：
// 想同时看网页时仍然照常 START_WEB_SERVER（在 test_camera_live 里是另一个端口、
// 另一个进程，与主程序 rmv_training 的 web_debug 互不影响）。

#include <deque>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace rotating_armor
{

// 一条遥测曲线：保留最近 window_s 秒的样本，SystemView 把它画成一张卡片
// （键名 + 当前值 + 迷你折线）。
struct TelemetryTrace
{
    std::string name;  // 键名，建议与 web_debug 的 WEB_LOG 键同名
    std::string unit;  // 单位后缀（"ms"/"m"/"" ...），只用于显示
    double window_s{10.0};

    // 若非空则直接显示这个字符串（用于 Idle/Stable 这类非数值状态），
    // 数值折线仍然照画（由 push() 提供）。
    std::string text;

    std::deque<std::pair<double, double>> samples; // (时间戳 s, 值)

    void push(double timestamp_s, double value);
    void clear();
    bool empty() const noexcept { return samples.empty(); }
    double latest() const noexcept;
    double minimum() const noexcept;
    double maximum() const noexcept;
};

struct SystemViewOptions
{
    int left_width{340};
    int right_width{340};
    int pad{10};
    int title_height{26};
    int card_height{62};
    // 左列卡片高度：0 = 自动摊开填满整列（上限 left_card_max_height，至少
    // card_height）；>0 = 固定高度。左列不需要给俯视图留位置，不摊开会留白。
    int left_card_height{0};
    int left_card_max_height{210};
    // 俯视图至少占的高度：右列卡片排完后剩余空间不够时，画布变高、相机画面
    // 在中间垂直居中，而不是把俯视图压成一个小方块。
    int overview_min_height{300};
    int card_gap{6};
    bool white_background{true};
    bool show_left{true};
    bool show_right{true};
    std::string left_title{"detect / capture"};
    std::string right_title{"EKF / aim"};
};

// 存储用 std::deque：push_back 不会让已有元素的地址失效，所以下面
// `auto &t = view.leftTrace(...)` 这类引用在后续注册其它曲线后依然有效
// （用 vector 存就会在扩容时变成悬垂引用，写进去的样本会丢）。
//
// 布局: [pad][左列卡片][pad][相机画面][pad][右列卡片 + 下方俯视图][pad]
// 画布高度 = max(相机高度, 右列卡片 + overview_min_height)，相机画面垂直居中；
// 俯视图按剩余空间等比缩到方形区域内（不会放大到超过它自己的分辨率）。
//
// 关键：本模块**不做整图缩放**。调用方应把相机画面先缩到最终显示尺寸再传进来，
// 这样卡片文字与相机叠加文字都是按最终像素栅格化的，不会糊。
class SystemView
{
  public:
    explicit SystemView(SystemViewOptions options = SystemViewOptions{});

    // 注册并返回一条曲线。首次调用即注册，之后按注册顺序从上往下排；
    // 同名重复调用返回同一条（不会重复注册）。
    TelemetryTrace &leftTrace(const std::string &name, const std::string &unit = {});
    TelemetryTrace &rightTrace(const std::string &name, const std::string &unit = {});

    // 清空所有样本（保留注册顺序、单位与 text）。
    void reset();

    // camera 必须是 BGR 三通道且非空；overview 可为空（此时右列只有卡片）。
    // 返回新分配的画布（不修改入参）。
    cv::Mat compose(const cv::Mat &camera, const cv::Mat &overview = cv::Mat{}) const;

    const SystemViewOptions &options() const noexcept { return options_; }
    const std::deque<TelemetryTrace> &leftTraces() const noexcept { return left_; }
    const std::deque<TelemetryTrace> &rightTraces() const noexcept { return right_; }

  private:
    TelemetryTrace &trace(std::deque<TelemetryTrace> &traces, const std::string &name,
                          const std::string &unit);

    SystemViewOptions options_;
    std::deque<TelemetryTrace> left_;
    std::deque<TelemetryTrace> right_;
};

} // namespace rotating_armor
