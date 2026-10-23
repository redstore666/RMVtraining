// SystemView 的布局/绘制冒烟测试：不需要相机、不开窗口。
//
// 验证三件事：
//  1) 画布尺寸 = pad + 左列 + pad + 相机 + pad + 右列 + pad，高度 = 相机高 + 2*pad；
//  2) 相机画面被原样搬到中间（不被卡片覆盖）；
//  3) 左右两列真的画了东西（不是空白），且空曲线/单点/时间回退都不崩。

#include "SystemView.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

#include <opencv2/imgproc.hpp>

namespace
{

int g_failures = 0;

void expect(bool condition, const char *what)
{
    if (!condition)
    {
        std::printf("[FAIL] %s\n", what);
        ++g_failures;
    }
    else
    {
        std::printf("[ ok ] %s\n", what);
    }
}

int nonWhitePixelCount(const cv::Mat &patch)
{
    int count = 0;
    for (int y = 0; y < patch.rows; ++y)
    {
        const cv::Vec3b *row = patch.ptr<cv::Vec3b>(y);
        for (int x = 0; x < patch.cols; ++x)
        {
            if (row[x] != cv::Vec3b(255, 255, 255))
            {
                ++count;
            }
        }
    }
    return count;
}

} // namespace

int main()
{
    rotating_armor::SystemViewOptions options;
    options.left_width = 220;
    options.right_width = 200;
    options.pad = 8;
    options.card_height = 50;
    options.title_height = 20;
    options.card_gap = 4;
    rotating_armor::SystemView view(options);

    // 左列：普通的耗时曲线；右列：一条状态（text 覆盖显示）+ 一条过零曲线。
    auto &grab = view.leftTrace("grab_ms", "ms");
    auto &detect = view.leftTrace("detect_ms", "ms");
    auto &phase = view.rightTrace("state");
    auto &omega = view.rightTrace("omega", "rad/s");

    for (int i = 0; i < 60; ++i)
    {
        const double t = i * 0.033;
        grab.push(t, 1.0 + 0.1 * std::sin(t));
        detect.push(t, 6.0 + 0.5 * std::cos(t));
        phase.push(t, 1.0);
        omega.push(t, 2.0 * std::sin(3.0 * t));
    }
    phase.text = "Stable";

    // 引用稳定性：上面先拿到 grab/phase 的引用、之后又注册了别的曲线，
    // 若内部用 vector 存就会扩容失效（往已释放内存里写，样本直接丢）。
    {
        bool stable = grab.samples.size() == 60 && phase.samples.size() == 60;
        expect(stable, "先取的引用在后续注册曲线后仍然有效（内部用 deque）");
    }

    cv::Mat camera(120, 300, CV_8UC3, cv::Scalar(10, 120, 200));
    cv::Mat overview(90, 90, CV_8UC3, cv::Scalar(250, 250, 250));
    cv::circle(overview, cv::Point(45, 45), 20, cv::Scalar(0, 0, 200), 2, cv::LINE_AA);

    const cv::Mat canvas = view.compose(camera, overview);
    expect(!canvas.empty(), "compose 返回非空画布");
    expect(canvas.type() == CV_8UC3, "画布是 BGR 三通道");

    const int expected_width =
        options.pad + options.left_width + options.pad + camera.cols + options.pad +
        options.right_width + options.pad;
    const int expected_height = options.pad + camera.rows + options.pad;
    expect(canvas.cols == expected_width,
           "画布宽度 = pad + 左列 + pad + 相机 + pad + 右列 + pad");
    expect(canvas.rows == expected_height, "画布高度 = 相机高 + 2*pad");
    if (canvas.cols != expected_width || canvas.rows != expected_height)
    {
        std::printf("      实际 %dx%d, 期望 %dx%d\n", canvas.cols, canvas.rows, expected_width,
                    expected_height);
    }

    // 相机画面原样搬进中间（抽查中心像素）。
    const int camera_x = options.pad + options.left_width + options.pad;
    const cv::Vec3b centre = canvas.at<cv::Vec3b>(options.pad + camera.rows / 2,
                                                  camera_x + camera.cols / 2);
    expect(centre == cv::Vec3b(10, 120, 200), "相机画面像素被原样拷贝到中间");

    const cv::Mat left_column =
        canvas(cv::Rect(options.pad, options.pad, options.left_width, camera.rows));
    const cv::Mat right_column = canvas(cv::Rect(options.pad + options.left_width + options.pad +
                                                     camera.cols + options.pad,
                                                 options.pad, options.right_width, camera.rows));
    expect(nonWhitePixelCount(left_column) > 2000, "左列画出卡片（非空白）");
    expect(nonWhitePixelCount(right_column) > 2000, "右列画出卡片（非空白）");

    // 空曲线 / 单点 / 时间回退都不该崩，也不该把画布搞坏。
    rotating_armor::SystemView empty_view(options);
    empty_view.leftTrace("no_samples", "ms");
    auto &single = empty_view.rightTrace("single");
    single.push(0.0, 42.0);
    const cv::Mat empty_canvas = empty_view.compose(camera, cv::Mat{});
    expect(!empty_canvas.empty(), "空曲线 + 无俯视图时仍能 compose");
    expect(empty_canvas.cols == canvas.cols && empty_canvas.rows == canvas.rows,
           "无俯视图时画布尺寸不变");

    auto &rewound = empty_view.leftTrace("rewound", "ms");
    rewound.push(10.0, 1.0);
    rewound.push(5.0, 2.0); // 时间回退：应当清空重建
    expect(rewound.samples.size() == 1 && std::fabs(rewound.latest() - 2.0) < 1e-9,
           "时间回退时曲线被清空重建");

    // 关掉右列后宽度应当少一个 right_width + pad。
    rotating_armor::SystemViewOptions left_only = options;
    left_only.show_right = false;
    rotating_armor::SystemView left_view(left_only);
    left_view.leftTrace("a", "ms").push(0.0, 1.0);
    const cv::Mat narrow = left_view.compose(camera, overview);
    expect(narrow.cols == canvas.cols - (options.right_width + options.pad),
           "关掉右列后宽度只少 右列宽 + pad");

    // 空相机应当返回空画布（调用方据此跳过 imshow）。
    expect(view.compose(cv::Mat{}, overview).empty(), "空相机返回空画布");

    if (g_failures != 0)
    {
        std::printf("%d 项失败\n", g_failures);
        return 1;
    }
    std::printf("system_view: 全部通过\n");
    return 0;
}
