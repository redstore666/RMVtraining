// SystemView / renderMiniTopDown 的绘制冒烟测试：不需要相机、不开窗口。
//
// 迷你俯视图部分验证：黑底、自动缩放（远近目标在画面里的占比接近）、
// 两个瞄准点颜色不同（青 = 有效、黄/绿 = 预测）、空态不崩且写 no target。
//
// 布局部分验证三件事：
//  1) 画布宽度 = pad + 左列 + pad + 相机 + pad + 右列 + pad，
//     高度 = pad + max(相机高, 右列卡片 + 俯视图最小高度) + pad；
//  2) 相机画面被原样搬到中间（不被卡片覆盖，列比画面高时垂直居中）；
//  3) 左右两列真的画了东西（不是空白），且空曲线/单点/时间回退都不崩。

#include "SystemView.hpp"

#include <algorithm>
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

int nonBlackPixelCount(const cv::Mat &canvas)
{
    int count = 0;
    for (int y = 0; y < canvas.rows; ++y)
    {
        const cv::Vec3b *row = canvas.ptr<cv::Vec3b>(y);
        for (int x = 0; x < canvas.cols; ++x)
        {
            if (row[x][0] > 24 || row[x][1] > 24 || row[x][2] > 24)
                ++count;
        }
    }
    return count;
}

bool hasColour(const cv::Mat &canvas, const cv::Scalar &target, int tolerance = 40)
{
    for (int y = 0; y < canvas.rows; ++y)
    {
        const cv::Vec3b *row = canvas.ptr<cv::Vec3b>(y);
        for (int x = 0; x < canvas.cols; ++x)
        {
            if (std::abs(row[x][0] - target[0]) <= tolerance &&
                std::abs(row[x][1] - target[1]) <= tolerance &&
                std::abs(row[x][2] - target[2]) <= tolerance)
                return true;
        }
    }
    return false;
}

// 车心红点（BGR 0,0,255）的质心。相机固定取景的位移映射断言要用它反算米数。
// 阈值收紧到 200/60/60，避免把灰板、瞄准点（青/黄）算进来。
bool findRedCentroid(const cv::Mat &canvas, cv::Point &centroid)
{
    long sum_x = 0;
    long sum_y = 0;
    long count = 0;
    for (int y = 0; y < canvas.rows; ++y)
    {
        const cv::Vec3b *row = canvas.ptr<cv::Vec3b>(y);
        for (int x = 0; x < canvas.cols; ++x)
        {
            if (row[x][2] > 200 && row[x][1] < 60 && row[x][0] < 60)
            {
                sum_x += x;
                sum_y += y;
                ++count;
            }
        }
    }
    if (count == 0)
        return false;
    centroid = cv::Point(static_cast<int>(sum_x / count), static_cast<int>(sum_y / count));
    return true;
}

// 灰板像素数（模型板画的是 BGR 150,150,150、粗 3）。容差压到 8，避免把
// 云台原点的 120 灰、相机白、车心红一起数进来。
int greyPixelCount(const cv::Mat &canvas)
{
    int count = 0;
    for (int y = 0; y < canvas.rows; ++y)
    {
        const cv::Vec3b *row = canvas.ptr<cv::Vec3b>(y);
        for (int x = 0; x < canvas.cols; ++x)
        {
            if (std::abs(row[x][0] - 150) <= 8 && std::abs(row[x][1] - 150) <= 8 &&
                std::abs(row[x][2] - 150) <= 8)
                ++count;
        }
    }
    return count;
}

// 造一台"车心在 (distance, 0)、半径 0.25、四块板"的标准车。
//
// phase     —— 车体相位 φ；四块板 = platePosition 的刚体解，随 φ 一起转。
// half_width—— 每块板的半宽。0.0675 = 真实小板；传 0.25（= 半径）就复刻了
//              以前那套"四段线首尾相连成方框"的画法，测试拿它当对照。
rotating_armor::MiniTopDownFrame makeVehicle(double distance, bool fire_allowed,
                                            double phase = 0.0, double half_width = 0.0675)
{
    rotating_armor::MiniTopDownFrame frame;
    frame.has_body = true;
    frame.body_centre = Eigen::Vector3d(distance, 0.0, 0.0);
    frame.camera_position = Eigen::Vector3d(0.0, 0.0, 0.0);
    for (int slot = 0; slot < 4; ++slot)
    {
        const double plate_phase = phase + slot * M_PI * 0.5;
        frame.plate_centres.emplace_back(distance - 0.25 * std::cos(plate_phase),
                                         -0.25 * std::sin(plate_phase), 0.30);
        // 外法向 = 相位 + π（与 platePhaseToWorldYaw 一致）；画线时用的是切线。
        frame.plate_yaws.push_back(plate_phase);
        frame.plate_half_widths.push_back(half_width);
        // 实测板：故意比模型晚 0.25 rad、并往外偏 1.5 cm —— 就是"转速跟不上"
        // 时模型板与 PnP 板之间的那种错位，用来验证绿线真的画出来了。
        const double obs_phase = plate_phase + 0.25;
        frame.detected_centres.emplace_back(distance - 0.265 * std::cos(obs_phase),
                                            -0.265 * std::sin(obs_phase), 0.30);
        frame.detected_yaws.push_back(obs_phase);
        frame.detected_half_widths.push_back(0.0675);
    }
    frame.aim_slot = 0;
    frame.has_valid_aim = true;
    frame.valid_aim = frame.plate_centres[0];
    frame.has_predicted_aim = true;
    frame.predicted_aim = frame.plate_centres[0] + Eigen::Vector3d(0.03, 0.02, 0.0);
    frame.fire_allowed = fire_allowed;
    frame.caption = "d=2.00m R=0.25 w=-3.0";
    return frame;
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
    // 高度由"相机画面"和"右列（卡片 + 俯视图最小高度）"里的大者决定。
    const auto column_height = [&options](int count)
    {
        if (count <= 0)
            return 0;
        return options.title_height + count * options.card_height + (count - 1) * options.card_gap;
    };
    const int right_need =
        column_height(2) + options.card_gap + options.overview_min_height;
    const int content_h = std::max({camera.rows, column_height(2), right_need});
    const int expected_height = options.pad + content_h + options.pad;
    expect(canvas.cols == expected_width,
           "画布宽度 = pad + 左列 + pad + 相机 + pad + 右列 + pad");
    expect(canvas.rows == expected_height, "画布高度 = pad + max(相机高, 右列需求) + pad");
    if (canvas.cols != expected_width || canvas.rows != expected_height)
    {
        std::printf("      实际 %dx%d, 期望 %dx%d\n", canvas.cols, canvas.rows, expected_width,
                    expected_height);
    }

    // 相机画面原样搬进中间，列更高时垂直居中（抽查中心像素）。
    const int camera_x = options.pad + options.left_width + options.pad;
    const int camera_y = options.pad + (content_h - camera.rows) / 2;
    const cv::Vec3b centre = canvas.at<cv::Vec3b>(camera_y + camera.rows / 2,
                                                  camera_x + camera.cols / 2);
    expect(centre == cv::Vec3b(10, 120, 200), "相机画面像素被原样拷贝到中间（垂直居中）");

    const cv::Mat left_column =
        canvas(cv::Rect(options.pad, options.pad, options.left_width, content_h));
    const cv::Mat right_patch = canvas(cv::Rect(
        options.pad + options.left_width + options.pad + camera.cols + options.pad,
        options.pad, options.right_width, content_h));
    expect(nonWhitePixelCount(left_column) > 2000, "左列画出卡片（非空白）");
    expect(nonWhitePixelCount(right_patch) > 2000, "右列画出卡片（非空白）");

    // 空曲线 / 单点 / 时间回退都不该崩，也不该把画布搞坏。
    rotating_armor::SystemView empty_view(options);
    empty_view.leftTrace("no_samples", "ms");
    auto &single = empty_view.rightTrace("single");
    single.push(0.0, 42.0);
    const cv::Mat empty_canvas = empty_view.compose(camera, cv::Mat{});
    expect(!empty_canvas.empty(), "空曲线 + 无俯视图时仍能 compose");
    // 只有各 1 条曲线、且没有俯视图 → 高度退回"相机画面对齐"，宽度不变。
    const int one_card_h = options.title_height + options.card_height;
    expect(empty_canvas.cols == expected_width &&
               empty_canvas.rows ==
                   options.pad + std::max(camera.rows, one_card_h) + options.pad,
           "无俯视图 + 卡片不多时，画布高度回到相机画面高度");

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

    // ===== 迷你俯视图（网页右下角那一块）=====
    {
        const cv::Mat near = rotating_armor::renderMiniTopDown(makeVehicle(2.0, false));
        const cv::Mat far = rotating_armor::renderMiniTopDown(makeVehicle(6.0, false));
        expect(near.cols == 480 && near.rows == 360 && near.type() == CV_8UC3,
               "迷你俯视图是 480x360 的 BGR 画布");
        expect(near.at<cv::Vec3b>(2, 2) == cv::Vec3b(0, 0, 0), "迷你俯视图是黑底");
        expect(nonBlackPixelCount(near) > 200, "迷你俯视图画了东西（不是全黑）");

        // 【2026-10-27 相机贴边】默认改成"相机钉在画布左边、覆盖前方 4 m"，
        // 车按真实米数移动。这里验证位移映射：2 m 与 4 m 的车心像素差应当
        // 正好等于 2 m × ppm，而不是"都铺满画面"。
        //
        // ppm 口径：相机边 → 对边的跨度 = width - 2*inset = 480 - 28 = 452 px，
        // 对应 4 m ⇒ 113 px/m（旧的"相机在正中"是 45 px/m，视野大了 2.5 倍）。
        {
            const cv::Mat at2 = rotating_armor::renderMiniTopDown(makeVehicle(2.0, false));
            const cv::Mat at4 = rotating_armor::renderMiniTopDown(makeVehicle(4.0, false));
            cv::Point c2, c4;
            const bool got2 = findRedCentroid(at2, c2);
            const bool got4 = findRedCentroid(at4, c4);
            expect(got2 && got4, "贴边取景：2 m 与 4 m 都能找到车心红点");
            if (got2 && got4)
            {
                // 默认 fixed_view_distance_m=4.0、inset=14、画布 480x360。
                const double ppm_expected = (480.0 - 2.0 * 14.0) / 4.0;
                const double dx = static_cast<double>(c4.x - c2.x);
                expect(std::abs(dx - 2.0 * ppm_expected) <= 3.0,
                       "贴边取景：车从 2 m 到 4 m 位移按真实米数映射（Δx 应 = 2 m × 113 px/m）");
                std::printf("      2m→(%d,%d)  4m→(%d,%d)  Δx=%.0fpx（应为 %.0fpx）\n", c2.x,
                            c2.y, c4.x, c4.y, dx, 2.0 * ppm_expected);
                // 相机贴左边 → 车沿 +x 往右走，纵坐标不漂；且 2 m 处 x 应约等于
                // 14 + 2×113 = 240（正好画面中间，纯属巧合，但可以锁住绝对位置）。
                const int x_at_2m = cvRound(14.0 + 2.0 * ppm_expected);
                expect(c4.x > c2.x && std::abs(c2.y - 180) <= 2 && std::abs(c4.y - 180) <= 2,
                       "贴边取景：车沿 +x 移动时只改变横坐标、纵坐标不漂");
                expect(std::abs(c2.x - x_at_2m) <= 3,
                       "贴边取景：2 m 处的绝对横坐标 = inset + 2 m × ppm");
                std::printf("      2m.x=%d（应为 %d）\n", c2.x, x_at_2m);
            }
        }

        // 相机图标必须贴在左边（白方块中心 x ≈ inset），不能跑到画面中间。
        {
            const cv::Mat at2 = rotating_armor::renderMiniTopDown(makeVehicle(2.0, false));
            int white_min_x = 480;
            for (int y = 0; y < at2.rows; ++y)
            {
                const cv::Vec3b *row = at2.ptr<cv::Vec3b>(y);
                for (int x = 0; x < at2.cols; ++x)
                {
                    if (row[x][0] > 240 && row[x][1] > 240 && row[x][2] > 240)
                        white_min_x = std::min(white_min_x, x);
                }
            }
            expect(white_min_x >= 6 && white_min_x <= 14,
                   "贴边取景：相机白方块贴在画布左边（x ≈ inset）");
            std::printf("      白色最小 x = %d\n", white_min_x);
        }

        // 两个瞄准点颜色不同：有效 = 青（BGR 255,255,0），预测 = 黄（0,200,255）。
        expect(hasColour(near, cv::Scalar(255, 255, 0), 10), "有效瞄准点是青色");
        expect(hasColour(near, cv::Scalar(0, 200, 255), 10), "预测瞄准点是黄色");
        // 允许开火时预测点变绿（0,255,0）。
        const cv::Mat hot = rotating_armor::renderMiniTopDown(makeVehicle(2.0, true));
        expect(hasColour(hot, cv::Scalar(0, 255, 0), 10), "fire_allowed 时预测点变绿");

        // 没有目标：只画相机与 no target，不该出现车心的红点。
        rotating_armor::MiniTopDownFrame empty;
        empty.camera_position = Eigen::Vector3d(0.0, 0.0, 0.0);
        const cv::Mat no_target = rotating_armor::renderMiniTopDown(empty);
        expect(!no_target.empty() && nonBlackPixelCount(no_target) < 2000,
               "无目标时迷你俯视图不崩、内容很少");
        expect(!hasColour(no_target, cv::Scalar(0, 0, 255), 20), "无目标时不画车心红点");

        // ===== 整车刚体建模（2026-10-25）=====
        // 实测板（PnP 原始观测，不经 EKF）画成绿色细线 + 中心点。
        expect(hasColour(near, cv::Scalar(80, 220, 80), 30), "实测板（PnP 世界系）画成绿线");

        // 不再有"某一格亮黄"：highlight_slot 默认 -1，四块板本来就随 φ 转，
        // 相位由板的位置表达，不需要一个会跳的色块。
        expect(!hasColour(near, cv::Scalar(0, 255, 255), 30),
               "默认不画黄色高亮格（highlight_slot = -1）");

        // 不是方框：半宽 = 半径时，四段线正好首尾相连围成闭合方框（旧画法）；
        // 半宽 = 真实板宽时只有四段短线段。同一个缩放口径下灰像素正比于线长，
        // 期望比例 0.135/0.5 = 0.27，取 0.5 留足抗锯齿余量。
        // 这条断言就是"别再画四个矩形"的回归测试。
        const cv::Mat short_plates = rotating_armor::renderMiniTopDown(makeVehicle(2.0, false));
        const cv::Mat square_plates =
            rotating_armor::renderMiniTopDown(makeVehicle(2.0, false, 0.0, 0.25));
        const int short_px = greyPixelCount(short_plates);
        const int square_px = greyPixelCount(square_plates);
        expect(short_px > 0 && short_px < square_px / 2,
               "四块板是四段短线、不再首尾相连成方框");
        if (short_px > 0 && square_px > 0)
            std::printf("      灰板像素：短线 %d / 方框 %d（比值 %.2f）\n", short_px, square_px,
                        static_cast<double>(short_px) / square_px);

        // 刚体：四块板整体转 90° 之后画面必须与原来逐像素重合（四板车的对称性）。
        // 这一条同时证明"没有任何元素钉在固定的绝对角度上"——旧画法里座位是死的、
        // 只有黄格会跳到下一格，转 90° 后黄格位置就变了，这里会把差值测出来。
        const auto rigid = [](double phase)
        {
            rotating_armor::MiniTopDownFrame f = makeVehicle(2.0, false, phase);
            f.aim_slot = -1; // 去掉瞄准板的白描边：它会跟着"槽位 0"转，不是几何差异
            f.has_valid_aim = false;
            f.has_predicted_aim = false;
            return rotating_armor::renderMiniTopDown(f);
        };
        const cv::Mat phase0 = rigid(0.0);
        const cv::Mat phase90 = rigid(M_PI / 2);
        const double diff = cv::norm(phase0, phase90, cv::NORM_L1) /
                            (255.0 * phase0.total() * phase0.channels());
        expect(diff < 0.002, "四块板整体转 90° 后画面重合（刚体、无固定座位）");
        if (diff >= 0.002)
            std::printf("      90° 相位差的画面差异 %.5f\n", diff);
    }

    if (g_failures != 0)
    {
        std::printf("%d 项失败\n", g_failures);
        return 1;
    }
    std::printf("system_view: 全部通过\n");
    return 0;
}
