#pragma once

// 实验性视图绘制（只给调试工具用，主程序 rmv_training 不依赖本模块）。
//
// 两次迭代留下的两样东西：
//  1) SystemView：imshow 版"卡片 + 相机画面 + 俯视图"拼图。现在两个工具都不再
//     调用它（系统视图改成了网页，见 tests/web_system/），保留作为参考实现，
//     由 test_system_view 的布局测试守着。
//  2) renderMiniTopDown()：网页右下角那块黑底俯视图——只画 4 块板 / 车心 /
//     相机 / 两个瞄准点，按内容自动缩放，不画相机画面，所以每帧很便宜。

#include <deque>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
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

// ===== 迷你俯视图（web 系统视图右下角那一块）=====
//
// 和上面的 compose() 是两回事：这里**不显示相机画面**，只有黑底 + 车体几何，
// 所以每帧只花零点几毫秒。坐标都是世界系（前 x / 左 y），画布按内容的包围盒
// 自动缩放（每帧重算），车走近走远都铺满画面。
struct MiniTopDownOptions
{
    int width{480};
    int height{360};
    // 包围盒外扩（米）：太小的话贴边的板会被画到边框上。
    double margin_m{0.35};
    // 视野半径下限（米）：只有相机、没有目标时也要有合理比例。
    double min_half_extent_m{0.75};
    // false（默认）= 只用车体定尺度，相机落在画面外时贴边画箭头 + 距离；
    // true = 相机与车一起装进画面（车会小很多，但距离关系是字面的）。
    bool fit_camera{false};

    // 速度箭头代表多少秒的位移（世界系米）。和 TopDownView::velocity_arrow_s 同一口径：
    // 箭头长度 = |v| × 这个值，所以 2 m/s 时箭头 = 0.5×2 = 1 m 长。
    // 太短会缩成一点看不见，渲染时再兜一个最小像素长度。
    double velocity_arrow_s{0.5};
    // false = 完全不画速度（箭头 + caption 里的分量都不写），给不想要这块信息的场景留退路。
    bool show_velocity{true};
};

struct MiniTopDownFrame
{
    // 车心（EKF 估计）。has_body=false 时只画相机与（可能存在的）瞄准点。
    Eigen::Vector3d body_centre{Eigen::Vector3d::Zero()};
    bool has_body{false};
    // 相机在云台系里的平移（顶视只取 x/y）——图上用来区分"云台原点"和"相机"。
    Eigen::Vector3d camera_position{Eigen::Vector3d::Zero()};

    // 4 块板的板心与法向 yaw（世界系，弧度）。两个数组等长；
    // 切线方向与 TopDownView 一致：tangent = (-sin(yaw), cos(yaw))。
    //
    // 【2026-10-25 整车建模】这四块板现在由 (车心, R, φ) 决定的**刚体**给出：
    // 板心 = platePosition(state, slot, N)、外法向 = φ + slot·90° + π，
    // 四块一起随 φ 转动。以前那套"四块灰板钉死在四个座位上、只有一格亮黄"
    // 的写法已废弃——它把"转"这件事压缩成"每 90° 跳一格"，看不出连续转速，
    // 也看不出模型和实测差多少。
    std::vector<Eigen::Vector3d> plate_centres;
    std::vector<double> plate_yaws;
    // 每块板的半宽（米）：按真实装甲板尺寸给（小 0.0675 / 大 0.115）。
    // 千万别再传 R——那会让四段线首尾相连成一个"方框"，看起来像四个矩形，
    // 也正是本轮要替换掉的画法。
    std::vector<double> plate_half_widths;

    // 实测板（这一帧 PnP 直接给出的世界系板心/法向，**不经过 EKF**）。
    // 和模型板分开画：模型板是"我们认为车在哪"，实测板是"传感器看到的在哪"，
    // 两条线的夹角就是相位误差、间距就是位置误差——调转速时直接看这个，
    // 不用再猜是模型慢还是观测跳。
    std::vector<Eigen::Vector3d> detected_centres;
    std::vector<double> detected_yaws;
    std::vector<double> detected_half_widths;

    // 车体速度（EKF 状态 vx/vy，世界系 m/s）。has_velocity=false 时不画箭头，
    // 也不在 caption 里报速度——锁定期没有可信速度，不能凭空写个 0。
    //
    // 画法：从车心出发的橙色箭头 = velocity_arrow_s 秒内的位移（和 TopDownView
    // 同一口径），箭头尖端标 |v|；左下角 caption 追加 vx/vy 分量。
    // 作业任务书 3.7 要求 webdebug 能看到"EKF 车体中心、速度、yaw、角速度、半径及状态"，
    // 前几项在俯视图上，速度就是这一项。
    Eigen::Vector2d velocity{Eigen::Vector2d::Zero()};
    bool has_velocity{false};

    // 高亮座位：**默认 -1（不画）**。整车建模后四块板本来就在转，
    // 不需要再用"某一格变黄"来表达相位；哪块板在被瞄由 aim_slot 的白色描边表达。
    // 保留这个字段只是为了兼容旧的诊断截图。
    int highlight_slot{-1};
    int aim_slot{-1};

    // 有效瞄准 = 当前时刻那块板的位置（弹丸瞬间到达会打中的点）；
    // 预测瞄准 = 含提前量/飞行时间的下发点。两者都画，颜色不同。
    Eigen::Vector3d valid_aim{Eigen::Vector3d::Zero()};
    bool has_valid_aim{false};
    Eigen::Vector3d predicted_aim{Eigen::Vector3d::Zero()};
    bool has_predicted_aim{false};
    bool fire_allowed{false};

    // 左下角小字（距离/半径/角速度）。空字符串就不画。
    std::string caption;
};

// 返回 width×height 的 BGR 黑底画布。
cv::Mat renderMiniTopDown(const MiniTopDownFrame &frame,
                          const MiniTopDownOptions &options = MiniTopDownOptions{});

} // namespace rotating_armor
