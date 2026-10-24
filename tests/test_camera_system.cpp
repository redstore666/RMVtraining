// 系统视图（网页版）：取流 + 装甲板检测 + PnP + 旋转车体 EKF，结果不画在窗口里，
// 而是喂给浏览器——
//
//   左/右两列 = 与 web_debug 完全同款的卡片折线（SSE /data，键名与生产程序一致）
//   右下角    = 迷你俯视图（GET /frame.jpg，黑底只画 4 块板 / 车心 / 相机 / 瞄准点）
//
// 与 test_camera_live 的分工：
//   test_camera_live  —— 第一阶段验收工具：相机窗口 + 坐标轴 + PnP 面板 + 俯视图。
//   test_camera_system —— 现场看"曲线 + 俯视图"用。**不画相机画面**，省掉 draw 的
//                        11 ms（1440×1080 的 resize/压黑/文字/整图缩放），所以帧率
//                        比总览图那套高；需要看画面时加 --show-camera 另开一个窗口
//                        （用同一份 CameraOverlay 画法），窗口显示不影响网页侧。
//
// 用法: ./build/test_camera_system [--red|--blue] [--video FILE] [--web-port N]
//        [--show-camera] [--no-display] [--dump-every N]
//
//   --video FILE   不连相机、改读视频（没有相机的机器上验证整条链路）。
//                  按视频自己的时间轴喂 EKF（time = frame_index / fps），所以
//                  转速、提前量和实拍一致，与 test_camera_playback 同一口径。
//   --video-loop   视频放完从头再来（网页一直有数据，适合慢慢看）
//   --hold S       视频放完后继续服务 S 秒再退出（默认 0 = 立即退出）
//   --web-port N   网页端口（默认 8081；生产程序的 webdebug 是 8080，互不影响）
//   --id N         启动就只看 id=N（0 = 自动）。网页下拉菜单随时可以改；切 id 时跟踪器
//                  会 reset，不会把两台车的观测混进同一个状态。
//   --plate-width M  迷你俯视图里装甲板的长度（米，默认 0.135 = 小板）。
//                  固定长度：不随"这一帧检测到的是大板还是小板"变来变去。
//   --show-camera  额外开一个窗口显示相机画面 + 检测框 + 坐标轴 + EKF 叠加
//   --no-display   不开任何窗口（只推网页；纯吞吐/后台跑）
//   --dump-every N 每 N 帧把迷你俯视图存到 ./tmp/system_mini_XXXXXX.png
//                  （--show-camera 时同时存相机叠加图 ./tmp/system_cam_XXXXXX.png）
//
// 按键（有窗口时）: ESC/q 退出 | r 重置 EKF
//
// 选车：网页右下角的下拉菜单按 id 选目标（数据来自"这条视频/这场里累计检测到的 id"）。
// 选中后只把该 id 的观测喂给跟踪器：不会自动切到别的 id；该 id 丢失时俯视图直接空掉，
// 面板显示 `id=3 lost`，直到它重新出现。选 0（auto）恢复自动跟踪。
//
// 外参说明：本工具把相机→云台旋转当单位阵、平移当 0（外参未标定），所以
// "世界系 = 云台系"，俯视图里的相机与云台原点重合、gimbal 的 yaw/pitch 保持 0。
// 网页上的 fire/miss 因此是相对光轴的量，不是实车云台脱靶量。
#include "StudentTasks.hpp"
#include "RotatingArmorTracker.hpp"
#include "SystemView.hpp"
#include "SystemWebLogger.hpp"
#include "CameraOverlay.hpp"

#ifdef HAVE_CERES
#include <ceres/ceres.h>
// 正弦残差（局部类不允许成员模板，放到文件作用域；参考 task2_fit）
struct SinOmegaResidual
{
    double t, w;
    SinOmegaResidual(double tt, double ww) : t(tt), w(ww) {}
    template <typename T>
    bool operator()(const T *const p, T *r) const
    {
        r[0] = w - (p[1] + p[0] * ceres::sin(p[2] * t + p[3]));
        return true;
    }
};
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace
{

// 板数固定 4（旋转车体模型），phase = state(6) + k*pi/2 与
// RotatingArmorModel::measurement() 是同一个公式。
constexpr int kPlateCount = 4;

} // namespace

// ===== 斜视角旁路检测（--no-oblique 关闭）=====
// NN 只出正对板框（v4 实测降置信度无效），转速因此在观测流里不存在。
// 用传统 HSV 灯条配对的宽松几何版找斜视角板：配对门放宽（间距下限
// 0.35*h、错位上限 0.7*h——斜视角下投影宽度按 cos 收缩），角点取灯条
// 外缘四边形，target_id 标为被选车（数字斜视角不可读）。NN 已覆盖的
// 区域不重复；重投影门(6px)+NIS 过滤坏角点。
struct ObliqueBar
{
    cv::Point2f top;
    cv::Point2f bottom;
    double height;
};

std::vector<ArmorDetection> detectObliqueArmors(const cv::Mat &bgr, TeamColor color,
                                                const std::vector<ArmorDetection> &existing)
{
    std::vector<ArmorDetection> out;
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    if (color == TeamColor::Red)
    {
        cv::Mat low, high;
        cv::inRange(hsv, cv::Scalar(0, 110, 140), cv::Scalar(14, 255, 255), low);
        cv::inRange(hsv, cv::Scalar(166, 110, 140), cv::Scalar(180, 255, 255), high);
        cv::bitwise_or(low, high, mask);
    }
    else
    {
        cv::inRange(hsv, cv::Scalar(95, 110, 140), cv::Scalar(135, 255, 255), mask);
    }
    cv::Mat labels, stats, centroids;
    const int n = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    std::vector<ObliqueBar> bars;
    for (int i = 1; i < n; ++i)
    {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        if (area < 12 || h < 5 || w < 1)
            continue;
        const double long_side = std::max(w, h);
        if (long_side / std::max(1.0, static_cast<double>(std::min(w, h))) < 1.6)
            continue;
        ObliqueBar bar;
        const double cx = centroids.at<double>(i, 0);
        const double cy = centroids.at<double>(i, 1);
        bar.top = cv::Point2f(static_cast<float>(cx), static_cast<float>(cy - 0.5 * h));
        bar.bottom = cv::Point2f(static_cast<float>(cx), static_cast<float>(cy + 0.5 * h));
        bar.height = h;
        bars.push_back(bar);
    }
    for (std::size_t a = 0; a < bars.size(); ++a)
    {
        for (std::size_t b = a + 1; b < bars.size(); ++b)
        {
            const double hmax = std::max(bars[a].height, bars[b].height);
            if (std::abs(bars[a].height - bars[b].height) > 0.45 * hmax || hmax < 8.0)
                continue;
            const double gap = std::abs(bars[a].top.x - bars[b].top.x);
            if (gap < 0.35 * hmax || gap > 6.5 * hmax)
                continue;
            const double dy = std::abs((bars[a].top.y + bars[a].bottom.y) -
                                       (bars[b].top.y + bars[b].bottom.y)) / 2.0;
            if (dy > 0.7 * hmax)
                continue;
            const ObliqueBar &lb = bars[a].top.x <= bars[b].top.x ? bars[a] : bars[b];
            const ObliqueBar &rb = bars[a].top.x <= bars[b].top.x ? bars[b] : bars[a];
            ArmorDetection det;
            const float wob = static_cast<float>(0.5 * lb.height / 8.0);
            det.corners = {cv::Point2f(lb.top.x - wob, lb.top.y),
                           cv::Point2f(rb.top.x + wob, rb.top.y),
                           cv::Point2f(rb.bottom.x + wob, rb.bottom.y),
                           cv::Point2f(lb.bottom.x - wob, lb.bottom.y)};
            det.size = ArmorSize::Small;
            det.target_id = 0;
            det.confidence = 0.25F;
            const cv::Point2f c(0.25F * (det.corners[0].x + det.corners[1].x +
                                         det.corners[2].x + det.corners[3].x),
                                0.25F * (det.corners[0].y + det.corners[1].y +
                                         det.corners[2].y + det.corners[3].y));
            bool dup = false;
            for (const ArmorDetection &e : existing)
            {
                cv::Point2f ec(0, 0);
                for (const cv::Point2f &q : e.corners)
                    ec += q * 0.25F;
                if (cv::norm(c - ec) < 0.6 * gap)
                {
                    dup = true;
                    break;
                }
            }
            if (!dup)
                out.push_back(det);
        }
    }
    return out;
}


int main(int argc, char **argv)
{
    bool enemy_red = true;
    bool no_display = false;
    bool show_camera = false;
    std::string video_path;
    int web_port = 8081;
    int dump_every = 0;
    bool video_loop = false;
    double hold_s = 0.0;
    double plate_width_m = 0.135; // 固定长度，不随检测到的板型变
    int initial_pin = 0;          // --id N：启动就只看这个 id（网页下拉菜单仍可改）
    float conf_thres = -1.0F;     // --conf T：检测置信度门限（<0 = 用库默认 0.7）
    bool pair_body = false;       // --pair-body：打开"双板整车求解"（库默认关）
    bool rate_pll = false;        // --rate-pll：打开残差 PLL 的转速量测（库默认关）
    bool rate_tracklet = false;   // --rate-tracklet：打开 Tracklet 单板角速度（库默认关）
    bool pix_rate = true;         // 图像域转速通道（默认开；--no-pix-rate 关闭）。检测只出正对板框时转速在观测量里不存在，这是 v4 实拍 omega 稳定的唯一来源
    bool oblique_fallback = false; // 斜视角旁路检测（--oblique 开启）。默认关：斜板亮度会改变面积谱的周期性（实测锁 6.09Hz 错频），面积谱通道需重新设计后再默认开
    double pair_phase_err = -1.0; // --pair-phase-err RAD：覆盖双板解的相位差门限（默认 -1=用库值 0.35）

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--red")
            enemy_red = true;
        else if (arg == "--blue")
            enemy_red = false;
        else if (arg == "--no-display")
            no_display = true;
        else if (arg == "--show-camera")
            show_camera = true;
        else if (arg == "--video")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--video 后面要跟文件路径\n";
                return 2;
            }
            video_path = argv[++i];
        }
        else if (arg == "--help" || arg == "-h")
        {
            std::cout << "用法: test_camera_system [--red|--blue] [--video FILE] [--video-loop]\n"
                         "       [--hold S] [--web-port N] [--id N] [--plate-width M] [--conf T]\n"
                         "       [--show-camera] [--no-display] [--dump-every N]\n"
                         "  --id N           启动就只看 id=N（0=自动）；网页下拉菜单随时可改\n"
                         "  --plate-width M  迷你图里的板长（米，默认 0.135，四块板等长）\n"
                         "  --conf T         检测置信度门限（默认用库里的 0.7；调低会多出弱检\n"
                         "  --pair-body      打开双板整车求解（两块相邻板闭式解 c/R/yaw，库默认关）\n"
                         "  --rate-pll       打开残差 PLL 转速量测（库默认关；四板相位只有 mod 90 度可观测）\n"
                         "  --rate-tracklet  打开 Tracklet 单板角速度量测（库默认关）\n"
                         "  --pair-phase-err RAD  覆盖双板解的相位差门限（默认 0.35；实测法向有 45 度偏置）\n"
                         "  --show-camera    另开窗口显示相机画面（不影响网页帧率）\n"
                         "  --video-loop     视频放完从头循环\n"
                         "  浏览器: http://127.0.0.1:8081/  （右下角可切换跟踪的 id）\n";
            return 0;
        }
        else if (arg == "--pair-phase-err" && i + 1 < argc)
        {
            pair_phase_err = std::atof(argv[++i]);
        }
        else if (arg == "--rate-pll")
        {
            rate_pll = true;
        }
        else if (arg == "--rate-tracklet")
        {
            rate_tracklet = true;
        }
        else if (arg == "--pix-rate")
        {
            pix_rate = true;
        }
        else if (arg == "--no-pix-rate")
        {
            pix_rate = false;
        }
        else if (arg == "--oblique")
        {
            oblique_fallback = true;
        }
        else if (arg == "--pair-body")
        {
            pair_body = true;
        }
        else if (arg == "--conf")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--conf 后面要跟 0~1 的置信度门限\n";
                return 2;
            }
            conf_thres = static_cast<float>(std::atof(argv[++i]));
        }
        else if (arg == "--id")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--id 后面要跟目标 id（0 = 自动）\n";
                return 2;
            }
            initial_pin = std::atoi(argv[++i]);
        }
        else if (arg == "--plate-width")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--plate-width 后面要跟米数\n";
                return 2;
            }
            plate_width_m = std::atof(argv[++i]);
            if (!(plate_width_m > 0.02 && plate_width_m < 0.60))
            {
                std::cerr << "--plate-width 应在 0.02~0.60 m 之间\n";
                return 2;
            }
        }
        else if (arg == "--video-loop")
            video_loop = true;
        else if (arg == "--hold")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--hold 后面要跟秒数\n";
                return 2;
            }
            hold_s = std::atof(argv[++i]);
            if (hold_s < 0.0)
            {
                std::cerr << "--hold 不能是负数\n";
                return 2;
            }
        }
        else if (arg == "--web-port")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--web-port 后面要跟端口号\n";
                return 2;
            }
            web_port = std::atoi(argv[++i]);
            if (web_port <= 0 || web_port > 65535)
            {
                std::cerr << "--web-port 必须是 1~65535\n";
                return 2;
            }
        }
        else if (arg == "--dump-every")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--dump-every 后面要跟帧数\n";
                return 2;
            }
            dump_every = std::atoi(argv[++i]);
            if (dump_every < 0)
            {
                std::cerr << "--dump-every 不能是负数\n";
                return 2;
            }
        }
        else
        {
            std::cerr << "未知参数: " << arg << "\n";
            return 2;
        }
    }
    if (no_display)
        show_camera = false;

    // ===== 读相机内参 =====
    CameraParameters camera;
    {
        cv::FileStorage fs("config/camera.yaml", cv::FileStorage::READ);
        if (!fs.isOpened())
            fs.open("../config/camera.yaml", cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            std::cerr << "无法打开 config/camera.yaml（请从项目根目录运行）\n";
            return 2;
        }
        fs["camera_matrix"] >> camera.camera_matrix;
        fs["distortion_coefficients"] >> camera.distortion_coefficients;
        // 外参未标定：单位旋转 + 零平移（与 test_camera_live 一致）。
        camera.rotation_camera_to_gimbal = cv::Mat::eye(3, 3, CV_64F);
        camera.translation_camera_to_gimbal = cv::Mat::zeros(3, 1, CV_64F);
    }

    GimbalState gimbal{};

    rotating_armor::TrackerConfig tracker_config;
    // 双板整车求解：两相邻板的闭式解直接给出 c/R/yaw，是"单板位置几乎不含相位信息"
    // 的唯一补救；库里默认关着，这里留个开关做 A/B。
    tracker_config.pair_body_solve_enabled = pair_body;
    tracker_config.rate_from_residual_enabled = rate_pll;
    tracker_config.tracklet_rate_enabled = rate_tracklet;
    if (pair_phase_err >= 0.0)
    {
        tracker_config.pair_body_max_phase_error_rad = pair_phase_err;
    }
    rotating_armor::RotatingArmorTracker tracker(tracker_config);
    if (pair_body || rate_pll || rate_tracklet)
    {
        std::cout << cv::format("[tracker] pair_body=%d rate_pll=%d rate_tracklet=%d\n",
                                pair_body ? 1 : 0, rate_pll ? 1 : 0, rate_tracklet ? 1 : 0);
    }

    // ===== 图像域转速通道（--pix-rate）=====
    // 检测只出"正对板框"时（v4 实拍 8 rad/s 自转：位置/法向/弦向读出的全是
    // ≈0，转速在跟踪器观测量里不可观测），灯条图案在原始图像里仍以 4×转速
    // 的频率相干振荡（实测亮斑面积谱 79~91% 功率集中在 5.0~5.3Hz）。这里在
    // 系统视图层测图案频率：|ω| = π·f/2（四重对称板），符号取"面积-质心"
    // 互谱相位的 sin（图案扫过方向锁定旋转方向），以量测形式经
    // tracker.injectRateObservation 喂给 EKF——不是硬写状态，σ 由谱峰显著
    // 性决定，滤波器自己加权。

    struct PixRateChannel
    {
        struct Sample
        {
            double t;
            double area;
            double cx;
        };
        std::deque<Sample> buf;
        double last_est_s{-1.0e9};
        double last_lock_s{-1.0e9};
        double window_s{1.6};
        double omega{0.0};
        double sigma{0.0};
        double f_peak{0.0};
        bool valid{false};
        // ROI 平滑：检测框逐帧抖动会把亮斑序列弄脏（实测注入符号乱跳），
        // 用 EMA 跟住车的区域；检测缺失时冻结上一帧 ROI（车不会瞬移，
        // 灯条亮着就能测，检测丢了反而是自转板的常态）。
        cv::Rect roi_fixed;
        bool roi_valid{false};
        double last_bright_s{-1.0e9};
        // ===== 逐灯条扫动先验：消首锁延迟 =====
        // 面积谱的入场爬坡鬼峰（4.0~4.8Hz）与真峰在 1.6s 窗内不可分，首锁
        // 要等 ~4s 谱成熟。而每根灯条过境时的横向扫速 |vx| = f·R·ω/Z 直接
        // 给出 |ω|（帧间最近邻跟踪 2~6 帧即得）——作为 ±35% 先验帮 FFT 在
        // ~0.6s 直接选中真峰。scale = Z/(f·R) 由接线层每帧更新。
        struct PrevBar
        {
            double x;
            double y;
            double h;
            double t;
        };
        std::vector<PrevBar> prev_bars;
        std::deque<std::pair<double, double>> vx_win;  // (t, vx_px_s)
        double sweep_scale{0.0};  // Z/(f·R)；<=0 = 无效（先验停用）
        double sweep_omega{-1.0};  // 先验 |ω|（rad/s），vx 中值*scale

        void updateSweepPrior(double t)
        {
            if (sweep_scale <= 0.0 || gray_bars.empty())
            {
                prev_bars.clear();
                return;
            }
            if (!prev_bars.empty() && t - prev_bars.front().t < 0.05 &&
                t - prev_bars.front().t > 1e-4)
            {
                const double dt = t - prev_bars.front().t;
                for (const auto &cur : gray_bars)
                {
                    // 最近邻：x 差最小且在门内
                    const PrevBar *best = nullptr;
                    double best_d = 1e9;
                    for (const auto &pb : prev_bars)
                    {
                        const double d = std::hypot(cur.x - pb.x, cur.y - pb.y);
                        if (d < best_d)
                        {
                            best_d = d;
                            best = &pb;
                        }
                    }
                    if (best && best_d < 0.8 * std::max(cur.h, best->h))
                    {
                        vx_win.push_back({t, (cur.x - best->x) / dt});
                        if (vx_win.size() > 400)
                            vx_win.pop_front();
                    }
                }
            }
            prev_bars.clear();
            for (const auto &b : gray_bars)
            {
                prev_bars.push_back({b.x, b.y, b.h, t});
            }
            while (!vx_win.empty() && t - vx_win.front().first > 0.5)
                vx_win.pop_front();
            if (std::getenv("SWEEP_DEBUG") && (vx_win.size() % 400 == 0))
                std::fprintf(stderr, "[sweep] scale=%.2e bars=%zu prev=%zu vx_n=%zu om=%.2f\n",
                             sweep_scale, gray_bars.size(), prev_bars.size(), vx_win.size(),
                             sweep_omega);
            if (vx_win.size() >= 20)
            {
                std::vector<double> vs;
                vs.reserve(vx_win.size());
                for (const auto &q : vx_win)
                    vs.push_back(std::abs(q.second));
                std::sort(vs.begin(), vs.end());
                sweep_omega = vs[vs.size() / 2] * sweep_scale;
            }
        }
        struct GrayBar
        {
            double x;
            double y;
            double h;
        };
        std::vector<GrayBar> gray_bars;  // 本帧亮条（measure 里填）

        void measure(const cv::Mat &bgr, const cv::Rect &roi_raw, bool has_det, double t)
        {
            // 固定框（离线分析的几何）：首检锚定 720 宽、全高，之后**不动**——
            // 移动框（哪怕慢速 EMA）都会用车在框内的漂移调制面积序列，抬高
            // 谱底（实测 prom 从离线 4.3~6.4 掉到 1.5~2.5）。车出框且持续无亮
            // 斑 1.5s 后，在新检测处重锚。
            if (has_det && (!roi_valid || t - last_bright_s > 1.5))
            {
                const double cx = roi_raw.x + 0.5 * roi_raw.width;
                const int new_x = cvRound(cx - 360);
                const bool moved = roi_valid && std::abs(new_x - roi_fixed.x) > 100;
                roi_fixed = cv::Rect(0, 0, 720, bgr.rows);
                roi_fixed.x = new_x;
                roi_fixed &= cv::Rect(0, 0, bgr.cols, bgr.rows);
                roi_valid = true;
                if (moved && !buf.empty())
                {
                    buf.clear();  // 只有框真的挪了才作废旧序列（原地重锚不清，
                                  // 否则暗窗反复清缓冲、永远攒不够样本）
                    psi_init = false;
                }
            }
            if (!roi_valid)
            {
                return;
            }
            cv::Mat gray;
            cv::cvtColor(bgr(roi_fixed), gray, cv::COLOR_BGR2GRAY);
            // 全分辨率 + 阈值 170（按视频亮度直方图定标：两个可见窗 100%
            // 覆盖、中位 322/394 亮像素；150 会放进背景杂波压低 prom，
            // 200 只剩一半帧有信号，下采样则毁掉 2~3px 的细亮条）。
            const double thr = 170.0;
            double area = 0.0;
            double sx = 0.0;
            for (int y = 0; y < gray.rows; ++y)
            {
                const uint8_t *row = gray.ptr<uint8_t>(y);
                for (int x = 0; x < gray.cols; ++x)
                {
                    if (row[x] > thr)
                    {
                        area += 1.0;
                        sx += x;
                    }
                }
            }
            // 逐灯条提取（扫动先验用）：亮像素按行游程聚成条（列聚类简化版
            // ——对细长竖条足够：每行的亮段中心投票到 x 直方图再取峰）。
            gray_bars.clear();
            {
                std::vector<int> xhist(gray.cols, 0);
                for (int y = 0; y < gray.rows; ++y)
                {
                    const uint8_t *row = gray.ptr<uint8_t>(y);
                    for (int x = 0; x < gray.cols; ++x)
                    {
                        if (row[x] > thr)
                            xhist[x]++;
                    }
                }
                const int bar_min_votes = std::max(4, gray.rows / 200);  // 实测列票只有 15~21（远处小板），按 rows/40=27 会全拒
                int run = 0;
                long long run_sum = 0, run_w = 0;
                for (int x = 0; x <= gray.cols; ++x)
                {
                    const int v = x < gray.cols ? xhist[x] : 0;
                    if (v > 0)
                    {
                        run_sum += static_cast<long long>(x) * v;
                        run_w += v;
                        ++run;
                    }
                    else if (run > 0)
                    {
                        if (run_w >= static_cast<long long>(bar_min_votes) * run)
                        {
                            GrayBar bar;
                            bar.x = static_cast<double>(run_sum) / run_w;
                            bar.h = static_cast<double>(run_w) / run;
                            bar.y = 0.0;
                            gray_bars.push_back(bar);
                        }
                        run = 0;
                        run_sum = 0;
                        run_w = 0;
                    }
                }
            }
            if (area < 4.0)  // 入口段车小、亮斑少；门太高会让缓冲填充慢、
                            // 首锁推迟（实测可见 2.0s、锁到 4.0s）
            {
                return;
            }
            last_bright_s = t;
            updateSweepPrior(t);
            // 断档感知：时间缺口 >0.15s（车出画/丢检）后旧样本与新的不相干，
            // 拼进同一个 1.6s 谱窗会把频谱涂抹掉（实测 prom 从 5+ 掉到 1~2，
            // 这正是跨断档捕获全部失败的原因）。
            if (buf.empty() || (!buf.empty() && t - buf.back().t > 0.15))
            {
                if (buf.empty())
                {
                    buf_start_s = t;
                }
                else
                {
                    buf.clear();
                    buf_start_s = t;
                    psi_init = false;
                }
            }
            static FILE *pix_dump = std::getenv("PIX_DUMP") ? std::fopen("tmp/pixdump.csv", "w") : nullptr;
            if (pix_dump)
            {
                std::fprintf(pix_dump, "%.6f %.1f %.4f\n", t, area, sx / area / std::max(1, gray.cols));
                std::fflush(pix_dump);  // 无缓冲：kill 截断曾两次骗过诊断
            }
            buf.push_back({t, area, sx / area / std::max(1, gray.cols)});
            while (!buf.empty() && t - buf.front().t > window_s)
            {
                buf.pop_front();
            }
        }

        // ===== 状态机：ACQUIRING(严门搜索) -> LOCKED(f 永不再搜，解调承载) =====
        // 弱窗误锁/频率漂移的共同病根是"随时重搜频率"。锁定后 f 硬保持，
        // 速率与符号全部由相位解调给出；解调持续失败 >1.2s 才重新捕获。
        bool locked{false};
        double last_demod_ok_s{-1.0e9};
        double blacklist_f{0.0};
        double blacklist_until_s{-1.0e9};
        double buf_start_s{-1.0e9};
        double lock_started_s{-1.0e9};
        bool ladder_converged{false};
        double ladder_prev_f{-1.0e9};  // 当前连续缓冲段的起点（断档/清空后重置）
        double dbg_prom{0.0};
        double dbg_band{0.0};

        // 指定跨度窗上的 argmax + 亚 bin 抛物线顶点。返回 false = 样本不足。
        // 双窗一致性判据（1.2s 与 1.6s 窗的峰一致才采信）是防泄漏鬼峰的关键：
        // 1.2s Hann 主瓣 ±0.83Hz，真峰(5.2)低侧(4.4~4.8)被泄漏抬高，不成熟
        // 窗的 argmax 会落在鬼峰上；1.6s 窗分辨率更高、且两窗对鬼峰的响应
        // 不同——实测一致时给出的都是 5.0~5.3（离线验证 t>=2.15 全一致）。
        bool peak_on_span(double t, double span, double &f_out, double *amp_out = nullptr) const
        {
            const double t0 = std::max(buf.front().t, t - span);
            double st = 0.0, sa = 0.0, stt = 0.0, sta = 0.0, nn = 0.0;
            for (const Sample &sm : buf)
            {
                if (sm.t < t0) continue;
                const double x = sm.t - t0;
                st += x; sa += sm.area; stt += x * x; sta += x * sm.area; nn += 1.0;
            }
            // 窗必须真的有 >=90% 跨度：缓冲不足时 t0 会被钳到缓冲头，
            // "1.6s 窗"退化成短窗——双窗判据变成两个相似短窗，鬼峰轻松一致
            // （实测首锁 4.4 的直接原因）。
            if (nn < span * 110.0 || t - t0 < 0.9 * span)
            {
                return false;
            }
            const double det = nn * stt - st * st;
            const double beta = std::abs(det) > 1e-9 ? (nn * sta - st * sa) / det : 0.0;
            const double mean = (sa - beta * st) / nn;
            const double wspan = t - t0;
            double best = -1.0; std::size_t bk = 0;
            std::vector<double> sp_amps(static_cast<std::size_t>((15.0 - 4.0) / 0.1) + 1, 0.0);
            std::size_t nsp = 0;
            for (double f = 4.0; f <= 15.0; f += 0.1)
            {
                std::complex<double> sc(0.0, 0.0);
                for (const Sample &sm : buf)
                {
                    if (sm.t < t0) continue;
                    const double dt = sm.t - t0;
                    const double w = 0.5 - 0.5 * std::cos(2.0 * CV_PI * dt / wspan);
                    sc += (sm.area - mean - beta * dt) * w *
                          std::complex<double>(std::cos(-2.0 * CV_PI * f * dt),
                                               std::sin(-2.0 * CV_PI * f * dt));
                }
                sp_amps[nsp] = std::abs(sc);
                if (sp_amps[nsp] > best)
                {
                    best = sp_amps[nsp];
                    bk = nsp;
                }
                ++nsp;
            }
            double f = 4.0 + 0.1 * static_cast<double>(bk);
            if (bk >= 1 && bk + 1 < nsp)
            {
                const double al = sp_amps[bk - 1], am = sp_amps[bk], ar = sp_amps[bk + 1];
                const double dn = al - 2.0 * am + ar;
                if (std::abs(dn) > 1e-9)
                {
                    const double off = 0.5 * (al - ar) / dn;
                    if (std::abs(off) <= 0.6)
                    {
                        f += 0.1 * off;
                    }
                }
            }
            f_out = f;
            if (amp_out)
            {
                *amp_out = best;
            }
            return true;
        }

        void estimate(double t)
        {
            valid = false;
            if (!locked)
            {
                // 快速临时锁：0.9s 窗单窗 argmax（需 ~0.8s 连续数据）。用严门
                // （1.44s 连续 + 双窗一致）时第一段可见期（实测只有 1.8s）等
                // 凑齐数据车就出画了——整段被当成静止，用户看得见旋转而系统
                // 报 0。先锁个大概（可能落在泄漏鬼峰 4.5~4.8 -> 7.2 rad/s，
                // 板立刻转起来），梯式（双窗一致 + 谱成熟收敛）在 ~1s 内修正
                // 到真值；错锁由黑名单 + 断档解锁兜底。
                if (buf.size() < 70 || t - last_est_s < 0.15)
                {
                    return;
                }
                last_est_s = t;
                double f12 = 0.0;
                if (!peak_on_span(t, 0.6, f12))
                {
                    return;
                }
                // 扫速先验：|ω_sw| 与峰的 πf/2 差在 35% 内 → 采纳（否则维持
                // 原峰）。入场鬼峰(4.0~4.8Hz->6.3~7.5)与真峰(5.2->8.17)差
                // ~23%，35% 窗只收真峰……实际鬼峰差 (8.17-6.9)/8.17=15.6%，
                // 窗收到 12% 只收真峰。
                static const bool sweep_prior_on = std::getenv("SWEEP_PRIOR") != nullptr;
                if (sweep_prior_on && sweep_omega > 2.0)
                {
                    const double omega_f = CV_PI * f12 / 2.0;
                    const double rel = std::abs(omega_f - sweep_omega) / sweep_omega;
                    if (rel > 0.22)
                    {
                        return;  // 峰与扫速先验矛盾：等下一个候选（梯式会再评）
                    }
                }
                const double f16 = f12;
                // 低频候选必须带二倍频伴随（真图案经整流必有 2f；杂波没有）
                double dummy = 0.0, a2f = 0.0;
                // 用 1.6s 窗在 2f 处的幅值近似：直接再算一次峰幅对照
                const std::size_t k2b = static_cast<std::size_t>(std::lround((2.0 * f16 - 4.0) / 0.1));
                (void)k2b; (void)dummy; (void)a2f;
                const bool blacklisted = std::abs(f16 - blacklist_f) < 0.6 && t < blacklist_until_s;
                dbg_band = blacklisted ? 4.0 : 9.0;
                if (!blacklisted)
                {
                    f_peak = f16;
                    omega = CV_PI * f_peak / 2.0;
                    rate_lp_ = omega;
                    rate_lp_init_ = true;
                    sigma = 0.5;
                    locked = true;
                    lock_started_s = t;
                    ladder_converged = false;
                    ladder_prev_f = -1.0e9;
                    last_lock_s = t;
                    last_demod_ok_s = t;
                    valid = true;
                }
                return;
            }
            // LOCKED：不重搜频率；demodulate() 负责逐帧速率。解调持续失败
            // 超过 1.2s（车出画/真停转）才放弃锁定重新捕获。
            // 例外：锁定头 3s 内每 0.4s 复评（早锁可能锁到入场鬼峰）。
            if (std::getenv("LADDER_DEBUG") && t - last_est_s >= 0.4)
            {
                double d12 = 0.0, d16 = 0.0;
                const bool ok12 = peak_on_span(t, 1.2, d12);
                const bool ok16 = peak_on_span(t, 1.6, d16);
                std::fprintf(stderr, "[ladder] t=%.2f n=%zu f_peak=%.2f conv=%d w12=%s%.2f w16=%s%.2f\n",
                             t, buf.size(), f_peak, ladder_converged ? 1 : 0,
                             ok12 ? "" : "X", d12, ok16 ? "" : "X", d16);
            }
            if (t - lock_started_s < 8.0 && !ladder_converged && t - last_est_s >= 0.4 &&
                buf.size() >= 140)
            {
                last_est_s = t;
                double f12 = 0.0, f16 = 0.0;
                // 一致容差 0.6：1.2s 窗在谱未熟时读低（泄漏），0.3 的容差让
                // 双窗常年不一致、只能靠 ±0.06Hz/步的抛物线步行——4.0->5.2
                // 要 8 秒（用户'转速跟不上'的机制）。放宽到 0.6 并跳向长窗值
                // （更接近真值），成熟后两窗自然收敛。
                if (peak_on_span(t, 1.2, f12) && peak_on_span(t, 1.6, f16) &&
                    std::abs(f12 - f16) <= 0.6)
                {
                    const bool bl = std::abs(f16 - blacklist_f) < 0.6 && t < blacklist_until_s;
                    static const bool sweep_prior_on2 = std::getenv("SWEEP_PRIOR") != nullptr;
                    const bool prior_ok =
                        !sweep_prior_on2 || sweep_omega <= 2.0 ||
                        std::abs(CV_PI * f16 / 2.0 - sweep_omega) / sweep_omega <= 0.22;
                    if (!bl && prior_ok)
                    {
                        if (std::abs(f16 - f_peak) > 0.02)
                        {
                            f_peak = f16;
                            psi_init = false;
                            // 换频即换期望转速：rate_lp 从新频期望值出发，否则
                            // 揣着旧频的值要磨好几秒（实测 f 已 5.2、ω 还在
                            // 5.15 爬 5 秒）。
                            rate_lp_ = CV_PI * f16 / 2.0;
                            demod_rate_ema = rate_lp_;
                        }
                        // 收敛：谱成熟(>=2.0s)且连续两次双窗一致结果相同
                        // 收敛三条件：谱成熟(>=2s) + 连续两次一致 + 已离开杂波
                        // 地板带（f<=4.35 是入场爬坡鬼峰/主瓣泄漏的地盘，在那里
                        // "一致"只是不成熟谱的自洽——实测 3.5~5.5s 卡 6.28 平台
                        // 就是提前收敛冻在 4.0）。
                        if (ladder_prev_f > 0.0 && std::abs(f16 - ladder_prev_f) < 0.1 &&
                            t - buf_start_s >= 2.0 && f16 > 4.35)
                        {
                            ladder_converged = true;
                        }
                        ladder_prev_f = f16;
                    }
                }
            }
            // 解调持续失败但亮斑仍在（视频回卷/遮挡闪断）：图案相位跳了，
            // 频率没变——只重锚解调链（psi 重新起算），不解锁、不重捕。
            if (t - last_demod_ok_s > 0.6 && t - last_bright_s < 0.2)
            {
                psi_init = false;
            }
            // 区间跟随可见性：车还亮着（last_bright 新鲜）就保持区间——
            // stableOmega 只看 locked，恒定注入不中断；只有"解调失败且亮斑
            // 消失 >0.5s"（车真出画）才无责解锁；亮着但解调连续失败 >6s
            // （f 真的错了）才带责解锁+拉黑换频。
            const double demod_age = t - last_demod_ok_s;
            const double bright_age = t - last_bright_s;
            const bool gone = demod_age > 2.5 && bright_age > 0.5;
            const bool wrong_f = demod_age > 6.0 && bright_age <= 0.5;
            if (gone || wrong_f)
            {
                locked = false;
                lock_started_s = -1.0e9;
                if (wrong_f)
                {
                    blacklist_f = f_peak;
                    blacklist_until_s = t + 2.5;
                }
                f_peak = 0.0;
                psi_init = false;
                demod_rate_ema = 0.0;
                rate_lp_init_ = false;
                rate_lp_ = 0.0;
                dir_conviction_ = 0.0;
                rate_hist_.clear();
                sin_valid_ = false;
                sin_last_fit_s_ = -1.0e9;
            }
        }

        // ===== 相位解调：锁定 f 后逐帧跟图案相位（"完全同步"的关键）=====
        // ψ = arg(Σ (area-mean)·w·e^{-i2πf(t_s-t)})（0.45s 滑窗）以 ±2πf 推进，
        // 符号=转向，板转速 = ψ̇/4。逐帧连续、换向 ~0.5s 内跟随、弱信号窗也有
        // 相位（不再停顿），优于"每 0.15s 阶跃一次的频率估计"。
        double demod_window_s{0.55};
        double psi_prev{0.0};
        double psi_t_prev{0.0};
        bool psi_init{false};
        double demod_rate_ema{0.0};
        double demod_mag_ema{0.0};
        bool demod_ok{false};
        double demod_rate() const { return demod_rate_ema; }
        // 旋转区间稳定转速：区间内恒定 ±π·f/2（平顶），符号来自解调方向的
        // 多数表决；解调还没给出过方向前退回 EMA 值。区间外（未锁定）无效。
        // ===== 方向迟滞 + 正弦转速 =====
        // 方向一旦确立（连续同号证据累积够），要翻转需要压倒性的反证——
        // 错频拍频/内容跳变会让单帧解调符号抖动，用户实测看到"方向转反"
        // 的一段。换向（司机反转）应只发生在重锁/强反证时。
        double dir_conviction_{0.0};   // 方向确信度：±，|v|>=1 才允许翻转
        // 正弦包络：低通后的解调速率（真转速波动），时间常数 0.4s
        double rate_lp_{0.0};
        bool rate_lp_init_{false};

        // ===== 正弦小陀螺结构化拟合（Ceres，参考 task2_fit）=====
        // ω(t) = A·sin(Ωt+φ)+b。锁定区间内积累解调速率样本（≤6s），每 0.5s
        // 重拟合一次；有效判定：A<=0.95b（波谷非负——单向自转）、Ω∈[0.2,6.28]
        // （周期 1~31s）、b∈[2,12]、RMS<0.8。有效时注入结构化 ω(t) 并输出
        // 预测 ω(t+0.3s)；无效（如 v4 的缓变加速——粗扫退化到 A>b）回退包络。
        std::deque<std::pair<double, double>> rate_hist_;
        double sin_last_fit_s_{-1.0e9};
        double sin_A_{0.0}, sin_b_{0.0}, sin_om_{0.0}, sin_phi_{0.0};
        bool sin_valid_{false};
        double sin_rms_{1.0e9};
        double t_now_{0.0};

        // 粗扫 Ω + （有 Ceres 时）AutoDiff 精修。返回是否得到有效拟合。
        bool refitSin(double t)
        {
            if (rate_hist_.size() < 120 || t - sin_last_fit_s_ < 0.5)
                return sin_valid_;
            sin_last_fit_s_ = t;
            const double t0 = rate_hist_.front().first;

            // —— 第一步：粗扫（线性 LS 解 sin/cos 基底，等价 A/φ 线性化）——
            double best_cost = 1e30, best_om = 0.0, c_s = 0.0, c_c = 0.0, c_b = 0.0;
            for (double om = 0.2; om <= 6.281; om += 0.02)
            {
                double ss = 0, sc = 0, s0 = 0, ssq = 0, scq = 0, s0q = 0, sw = 0;
                double sws = 0, swc = 0;
                for (const auto &q : rate_hist_)
                {
                    const double x = q.first - t0;
                    const double S = std::sin(om * x), C = std::cos(om * x), w = q.second;
                    ss += S; sc += C; s0 += 1; sw += w;
                    ssq += S * S; scq += C * C; s0q += 1;
                    sws += S * w; swc += C * w;
                }
                // 3x3 正规方程 [S*C 0 求解]（sc=0：sin/cos 正交但采样不等距，
                // 老老实实解 3x3）
                Eigen::Matrix3d M;
                M << ssq, 0.0, ss,
                     0.0, scq, sc,
                     ss, sc, s0q;
                const Eigen::Vector3d r(sws, swc, sw);
                const Eigen::Vector3d sol = M.colPivHouseholderQr().solve(r);
                double cost = 0;
                for (const auto &q : rate_hist_)
                {
                    const double x = q.first - t0;
                    const double e = q.second - (sol[0] * std::sin(om * x) + sol[1] * std::cos(om * x) + sol[2]);
                    cost += e * e;
                }
                if (cost < best_cost)
                {
                    best_cost = cost;
                    best_om = om;
                    c_s = sol[0]; c_c = sol[1]; c_b = sol[2];
                }
            }
            sin_A_ = std::hypot(c_s, c_c);
            sin_phi_ = std::atan2(c_c, c_s);
            sin_b_ = c_b;
            sin_om_ = best_om;

#ifdef HAVE_CERES
            // —— 第二步：Ceres AutoDiff 精修 [A, b, Ω, φ] ——
            {
                double params[4] = {sin_A_, sin_b_, sin_om_, sin_phi_};
                ceres::Problem problem;
                for (const auto &q : rate_hist_)
                {
                    problem.AddResidualBlock(
                        new ceres::AutoDiffCostFunction<SinOmegaResidual, 1, 4>(
                            new SinOmegaResidual(q.first - t0, q.second)),
                        nullptr, params);
                }
                ceres::Solver::Options opts;
                opts.linear_solver_type = ceres::DENSE_QR;
                opts.max_num_iterations = 20;
                opts.minimizer_progress_to_stdout = false;
                ceres::Solver::Summary summ;
                ceres::Solve(opts, &problem, &summ);
                if (summ.IsSolutionUsable() && params[1] > 0.5 && params[2] > 0.15)
                {
                    sin_A_ = params[0]; sin_b_ = params[1];
                    sin_om_ = params[2]; sin_phi_ = params[3];
                }
            }
#endif
            sin_rms_ = std::sqrt(best_cost / static_cast<double>(rate_hist_.size()));
            sin_valid_ = sin_A_ <= 0.95 * sin_b_ && sin_om_ >= 0.2 && sin_om_ <= 6.281 &&
                         sin_b_ >= 3.0 && sin_b_ <= 11.0 && sin_rms_ < 0.8;
            return sin_valid_;
        }

        // 结构化 ω(t)（拟合有效时）
        double sinOmegaAt(double t) const
        {
            const double t0 = rate_hist_.empty() ? 0.0 : rate_hist_.front().first;
            return sin_b_ + sin_A_ * std::sin(sin_om_ * (t - t0) + sin_phi_);
        }

        bool stableOmega(double &omega_out)
        {
            if (!locked || f_peak <= 0.0)
            {
                return false;
            }
            const double mag = CV_PI * f_peak / 2.0;
            double sign = 0.0;
            if (demod_rate_ema > 0.5)
                sign = 1.0;
            else if (demod_rate_ema < -0.5)
                sign = -1.0;
            // 方向确信度积累：同号 +1、反号 -1，EMA 半衰 0.5s
            if (sign != 0.0)
            {
                const double a = 0.06;
                dir_conviction_ = (1.0 - a) * dir_conviction_ + a * sign * 3.0;
                dir_conviction_ = std::clamp(dir_conviction_, -2.0, 2.0);
            }
            const double dir = dir_conviction_ >= 0.0 ? 1.0 : -1.0;
            // ===== 正弦转速：优先用低通后的解调速率（含真实波动）， =====
            // 恒定 πf/2 只作兜底（解调从未成功/偏离锁频太远时）。
            double mag_used = mag;
            if (rate_lp_init_ && std::abs(rate_lp_) > 0.4 * mag && std::abs(rate_lp_) < 1.6 * mag)
            {
                mag_used = std::abs(rate_lp_);
            }
            // 正弦拟合有效：结构化 ω(t) 优先（波峰波谷由模型给出，且可预测）
            if (sin_valid_)
            {
                const double w_fit = sinOmegaAt(t_now_);
                if (w_fit > 0.55 * mag && w_fit < 1.35 * mag)  // 波峰波谷带宽：b±A 物理上不可能超出 mean±35% 太多
                    mag_used = w_fit;
            }
            omega_out = dir * mag_used;
            return demod_rate_ema != 0.0 || rate_lp_init_;
        }

        int last_fail{0};
        void demodulate(double t)
        {
            demod_ok = false;
            last_fail = 0;
            if (!locked || buf.size() < 30)
            {
                last_fail = locked ? 2 : 1;
                return;
            }
            const double t_lo = t - demod_window_s;
            double mean_area = 0.0;
            int n = 0;
            for (const Sample &sm : buf)
            {
                if (sm.t >= t_lo)
                {
                    mean_area += sm.area;
                    ++n;
                }
            }
            if (n < 20)
            {
                last_fail = 3;
                return;
            }
            mean_area /= static_cast<double>(n);
            std::complex<double> c(0.0, 0.0);
            for (const Sample &sm : buf)
            {
                if (sm.t < t_lo)
                {
                    continue;
                }
                const double w = 0.5 - 0.5 * std::cos(2.0 * CV_PI * (sm.t - t_lo) / demod_window_s);
                const double ph = -2.0 * CV_PI * f_peak * (sm.t - t);
                c += (sm.area - mean_area) * w * std::complex<double>(std::cos(ph), std::sin(ph));
            }
            const double mag = std::abs(c);
            demod_mag_ema = demod_mag_ema > 0.0 ? 0.95 * demod_mag_ema + 0.05 * mag : mag;
            // 置信门：|C| 明显低于近期水平时不算（背景杂波假相位）
            if (demod_mag_ema > 0.0 && mag < 0.25 * demod_mag_ema)
            {
                last_fail = 4;
                return;
            }
            const double psi = std::arg(c);
            if (!psi_init)
            {
                psi_prev = psi;
                psi_t_prev = t;
                psi_init = true;
                return;
            }
            const double dt = t - psi_t_prev;
            if (dt <= 1e-4 || dt > 0.1)
            {
                psi_prev = psi;
                psi_t_prev = t;
                return;
            }
            double dpsi = psi - psi_prev;
            while (dpsi > CV_PI) dpsi -= 2.0 * CV_PI;
            while (dpsi < -CV_PI) dpsi += 2.0 * CV_PI;
            psi_prev = psi;
            psi_t_prev = t;
            // 相位差分噪声大（单帧 0.19 rad 信号 vs 噪声同量级），EMA 0.12s 平滑
            const double rate = dpsi / dt / 4.0;
            // 一致性门：锁对 f 时 ψ̇/4 = ±π·f/2。偏离 30% 以上说明解调已失锁
            // （f_peak 被弱窗拖走/图案消失），此时不更新、不置 ok。
            const double expect = CV_PI * f_peak / 2.0;
            if (std::abs(std::abs(rate) - expect) > 0.45 * expect)
            {
                return;
            }
            const double a = std::min(1.0, dt / 0.12);
            demod_rate_ema += a * (rate - demod_rate_ema);
            // 正弦包络：0.4s 低通保留真实转速波动（RM 正弦小陀螺 ~1-2Hz）
            const double alp = std::min(1.0, dt / 0.2);
            if (!rate_lp_init_)
            {
                rate_lp_ = rate;
                rate_lp_init_ = true;
            }
            else
            {
                rate_lp_ += alp * (rate - rate_lp_);
            }
            demod_ok = true;
            last_demod_ok_s = t;
            rate_hist_.push_back({t, rate});
            while (!rate_hist_.empty() && t - rate_hist_.front().first > 6.0)
                rate_hist_.pop_front();
            refitSin(t);
        }

        // 惯性滑行：谱估计断续（车出画/谱弱）时，锁定后 1s 内按上次 ω 注入，
        // σ 随龄期增大，滤波器自行折权。自转是持续运动，短暂盲区不该把
        // omega 打回 0。
        bool coast(double t, double &omega_out, double &sigma_out) const
        {
            const double age = t - last_demod_ok_s;
            if (!(age >= 0.0) || age > 1.0 || f_peak <= 0.0 || demod_rate_ema == 0.0)
            {
                return false;
            }
            omega_out = demod_rate_ema;
            sigma_out = 0.6 + 0.9 * age;
            return true;
        }
    } pix_rate_channel;
    // 俯视图板相位的独立积分器（与 EKF 解耦，由检测转速驱动）。
    double topdown_phase{0.0};
    double topdown_last_t{0.0};
    double topdown_driving_rate{0.0};
    bool topdown_phase_init{false};
    // ===== 观测驱动的显示相位（零锁频延迟的真实相位）=====
    // 检测到的板的法向本身就是真实板相位（白十字标记转得和实际一致——
    // 用户的发现）。连续解缠观测法向序列：同板帧间 Δ=ω·dt 很小；换板
    // 精确跳 ±90°（四板对称），fold±π 下照样推进——净推进=真实转速、
    // 方向正确（离线验证该序列净转速 +8.5 rad/s）。显示优先用它；
    // 观测缺失（出画）回退通道积分相位。
    // 观测驱动的显示相位（零锁频延迟的真实相位）：0.3s 窗口内所有
    // (t, 法向) 样本跨帧全配对 fold±π 取中位速率，逐帧积分。
    double obs_disp_phase{0.0};
    bool obs_disp_init{false};
    if (pix_rate)
    {
        std::cout << "[pix-rate] 图像域转速通道开启（亮斑图案频率 -> injectRateObservation）\n";
    }

    if (conf_thres > 0.0F)
    {
        setArmorScoreThreshold(conf_thres);
        std::cout << cv::format("[det] 置信度门限 %.2f\n", conf_thres);
    }
    START_SYSTEM_WEB(web_port);
    std::cout << "[web] 系统视图: http://127.0.0.1:" << web_port << "/\n";
    std::cout << "[web] 迷你俯视图原图: http://127.0.0.1:" << web_port << "/frame.jpg\n";

    // 只有真的要显示相机画面时才建窗口：不建空窗口，纯网页模式不会弹出黑框。
    if (show_camera)
    {
        cv::namedWindow("camera_system", cv::WINDOW_NORMAL);
        std::cout << "ESC/q 退出, r 重置 EKF\n";
    }

    cv::Mat frame;
    int frame_cnt = 0;
    int stat_cnt = 0;
    int64_t last_show_tick = 0;
    int64_t last_mini_tick = 0;
    bool window_sized = false;
    double fps_loop_ema = 0.0;
    double last_frame_time_s = 0.0;
    // 网页选车：pinned=0 表示自动；切换时要 reset，免得新车的观测喂进旧车的状态。
    int pinned_last = 0;
    double pinned_last_seen_s = -1.0e9;
    int targets_tick = 0;
    std::set<int> seen_ids;
    std::set<int> visible_ids;

    auto tick_ms = [](int64_t a, int64_t b)
    { return static_cast<double>(b - a) * 1000.0 / cv::getTickFrequency(); };

    const auto t_start = std::chrono::steady_clock::now();

    cv::VideoCapture video_capture;
    double video_fps = 30.0;
    double video_duration_s = 0.0;
    double video_time_offset_s = 0.0; // 循环播放时累计的时间轴偏移
    long long video_frame_index = 0;  // 当前循环内的帧号（从 0 起）
    int video_loop_count = 0;
    if (!video_path.empty())
    {
        video_capture.open(video_path);
        if (!video_capture.isOpened())
        {
            std::cerr << "无法打开视频: " << video_path << "\n";
            return 2;
        }
        video_fps = video_capture.get(cv::CAP_PROP_FPS);
        if (!(video_fps > 1.0))
            video_fps = 30.0; // 容器里没写帧率时按 30 处理，别让 dt 变成 0
        const double frames = video_capture.get(cv::CAP_PROP_FRAME_COUNT);
        video_duration_s = frames > 0.0 ? frames / video_fps : 0.0;
        std::cout << "[--video] " << video_path << "  " << video_capture.get(cv::CAP_PROP_FRAME_WIDTH)
                  << "x" << video_capture.get(cv::CAP_PROP_FRAME_HEIGHT) << "  " << video_fps
                  << " fps  " << static_cast<long long>(frames) << " 帧  "
                  << cv::format("%.1f s", video_duration_s) << "\n";
        std::cout << "[--video] 时间轴按视频自己的帧率走（frame_index / fps），"
                     "转速与实拍一致\n";
        if (video_loop)
            std::cout << "[--video] 循环播放中（--video-loop）\n";
    }

    // 迷你俯视图 30 Hz 渲染门（检测/滤波照常全速跑）。
    const double mini_interval_ticks = 1.0 / 30.0 * cv::getTickFrequency();
    const double show_interval_ticks = 1.0 / 30.0 * cv::getTickFrequency();

    if (initial_pin > 0)
        SYSTEM_TARGET_SET(initial_pin);

    while (true)
    {
        const int64_t t0 = cv::getTickCount();
        if (!video_path.empty())
        {
            bool got = video_capture.read(frame) && !frame.empty();
            if (!got && video_loop)
            {
                // 回到第一帧接着放；时间轴不回退（否则 EKF 会看到负的 dt）。
                video_time_offset_s += video_duration_s > 0.0
                    ? video_duration_s
                    : static_cast<double>(video_frame_index) / video_fps;
                video_frame_index = 0;
                video_capture.set(cv::CAP_PROP_POS_FRAMES, 0.0);
                ++video_loop_count;
                got = video_capture.read(frame) && !frame.empty();
            }
            if (!got)
            {
                std::cout << "[--video] 播放结束\n";
                break;
            }
            ++video_frame_index;
        }
        else if (!get_pic(frame))
        {
            continue;
        }
        const int64_t t1 = cv::getTickCount();

        auto dets = armor_detect(frame, enemy_red ? TeamColor::Red : TeamColor::Blue);
        int oblique_added = 0;
        if (oblique_fallback)
        {
            const int pin = SYSTEM_TARGET();
            if (pin > 0)
            {
                for (ArmorDetection &fb : detectObliqueArmors(
                         frame, enemy_red ? TeamColor::Red : TeamColor::Blue, dets))
                {
                    fb.target_id = pin;
                    dets.push_back(fb);
                    ++oblique_added;
                }
            }
            SYSTEM_LOG("oblique_added", static_cast<double>(oblique_added));
        }
        const int64_t t2 = cv::getTickCount();
        const auto poses = armor_solve(dets, camera, gimbal);
        const int64_t t3 = cv::getTickCount();

        // 视频回放用视频自己的时间轴（与 test_camera_playback 一致）：按墙钟走的话，
        // 1440×1080@166fps 的片子在本机只跑到 ~110fps，EKF 看到的转速会偏慢 0.66 倍。
        // 实车（get_pic）仍用 steady_clock。
        const double time_s = video_path.empty()
            ? std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count()
            : video_time_offset_s + static_cast<double>(video_frame_index - 1) / video_fps;
        const double latency_s = tick_ms(t0, t3) / 1000.0;

        // ===== 网页选车：只把选中的 id 喂给跟踪器 =====
        // 这样"选了 id=3 却跟到 id=1"就不会发生：id=3 丢了就没人更新，跟踪器自己衰减。
        const int pinned = SYSTEM_TARGET();
        if (pinned != pinned_last)
        {
            tracker.reset();
            topdown_phase_init = false;
            obs_disp_init = false;
            pinned_last = pinned;
            pinned_last_seen_s = -1.0e9;
            std::cout << (pinned > 0 ? cv::format("[pin] 只看 id=%d（丢失也不自动切换）\n", pinned)
                                     : std::string("[pin] 自动选择目标\n"));
        }
        std::vector<ArmorPose> tracked_poses;
        tracked_poses.reserve(poses.size());
        visible_ids.clear();
        for (const ArmorPose &pose : poses)
        {
            const int id = pose.detection.target_id;
            if (id > 0)
            {
                visible_ids.insert(id);
                seen_ids.insert(id);
            }
            if (pinned <= 0 || id == pinned)
                tracked_poses.push_back(pose);
        }
        if (pinned > 0 && visible_ids.count(pinned) != 0)
            pinned_last_seen_s = time_s;

        // POSE_DEBUG=1：逐帧打印每块板的"位置 bearing vs 法向"，用来判断
        // 模型的"外法向 = 相位 + π"约定和实拍法向到底差多少（45 度偏置的来源）。
        static const bool pose_debug = std::getenv("POSE_DEBUG") != nullptr;
        // 观测转速诊断（obs_rate 键）：窗口内存所有 (t, 法向) 原始样本，跨帧
        // 全配对 fold±π/dt 取中位——同板对给真实 ω、换板/闪烁对(±90°)
        // 是离群对，中值免疫；链式"最近邻选择"会被闪烁系统性带偏（实测
        // ±24 rad/s 伪转速）。逐帧把中值速率积分成显示相位。
        {
            static std::deque<std::pair<double, double>> obs_yaw_win;  // (t, yaw)
            static double obs_int_t = 0.0;
            if (!tracked_poses.empty())
            {
                for (const ArmorPose &pose : tracked_poses)
                {
                    obs_yaw_win.push_back({time_s, pose.armor_yaw});
                }
                while (!obs_yaw_win.empty() && time_s - obs_yaw_win.front().first > 0.30)
                {
                    obs_yaw_win.pop_front();
                }
                if (obs_yaw_win.size() >= 16)
                {
                    std::vector<double> pair_rates;
                    for (std::size_t i = 0; i + 1 < obs_yaw_win.size(); ++i)
                    {
                        for (std::size_t j = i + 1; j < obs_yaw_win.size(); ++j)
                        {
                            const double dt = obs_yaw_win[j].first - obs_yaw_win[i].first;
                            if (dt < 0.03 || dt > 0.15)
                                continue;
                            double d = obs_yaw_win[j].second - obs_yaw_win[i].second;
                            while (d > CV_PI) d -= 2.0 * CV_PI;
                            while (d < -CV_PI) d += 2.0 * CV_PI;
                            pair_rates.push_back(d / dt);
                        }
                    }
                    if (pair_rates.size() >= 12)
                    {
                        std::sort(pair_rates.begin(), pair_rates.end());
                        const double rate = pair_rates[pair_rates.size() / 2];
                        // 物理量程 + 噪声门（|rate| 也要信得过）
                        if (std::abs(rate) < 12.0 && std::abs(rate) > 0.3)
                        {
                            if (obs_disp_init && obs_int_t > 0.0 &&
                                time_s - obs_int_t < 0.05)
                            {
                                obs_disp_phase += rate * (time_s - obs_int_t);
                            }
                            obs_disp_init = true;
                            obs_int_t = time_s;
                        }
                        else if (!obs_disp_init)
                        {
                            obs_disp_init = false;
                        }
                    }
                }
                if (!obs_disp_init && !obs_yaw_win.empty())
                {
                    // 未起转时的初始化锚（相位随便取一块板的法向即可，四重
                    // 对称下显示相位差 90° 不可分辨）
                    obs_disp_phase = obs_yaw_win.back().second;
                    obs_disp_init = true;
                    obs_int_t = time_s;
                }
            }
            else if (time_s - (obs_yaw_win.empty() ? 0.0 : obs_yaw_win.back().first) > 0.3)
            {
                obs_yaw_win.clear();
                obs_int_t = 0.0;
            }
        }

        if (pose_debug && tracker.initialized() && !tracked_poses.empty())
        {
            const Eigen::VectorXd st = tracker.ekf().state();
            // ts 供 gt_omega.py 做时间轴（[pose] 行没有自己的时间戳，解析时
            // 继承所属 [state] 的 ts）。
            std::printf("[state] ts=%.3f c=(%+.3f,%+.3f) z=%+.3f R=%.3f yaw=%+7.1f w=%+.2f\n",
                        time_s, st(0), st(2), st(4), st(8), st(6) * 180.0 / M_PI, st(7));
            const double cx = tracker.ekf().state()(0);
            const double cy = tracker.ekf().state()(2);
            const double body_yaw = tracker.ekf().state()(6);
            for (const ArmorPose &pose : tracked_poses)
            {
                const double px = pose.position_gimbal_m[0];
                const double py = pose.position_gimbal_m[1];
                const double bearing = std::atan2(py - cy, px - cx);
                const double normal = pose.armor_yaw;
                const double slot_phase = body_yaw + 0.0;
                std::printf("[pose] id=%d x=%+.3f y=%+.3f bear=%+7.1f normal=%+7.1f "
                            "normal-bear=%+7.1f model_phase=%+7.1f yaw=%+7.1f\n",
                            pose.detection.target_id, px, py,
                            bearing * 180.0 / M_PI, normal * 180.0 / M_PI,
                            (normal - bearing) * 180.0 / M_PI,
                            slot_phase * 180.0 / M_PI, body_yaw * 180.0 / M_PI);
                (void)slot_phase;
            }
        }

        // 图像域转速通道：ROI = 被选 id 检测框的外接矩形放大 2.5 倍（通道内
        // 再做 EMA 平滑；检测缺失时冻结 ROI 继续测）。
        if (pix_rate && !frame.empty())
        {
            cv::Rect roi;
            bool has_roi = false;
            for (const ArmorPose &pose : poses)
            {
                if (pose.detection.target_id != pinned || pinned <= 0)
                {
                    continue;
                }
                for (const cv::Point2f &pt : pose.detection.corners)
                {
                    if (!has_roi)
                    {
                        roi = cv::Rect(cv::Point(cvRound(pt.x), cvRound(pt.y)), cv::Size(1, 1));
                        has_roi = true;
                    }
                    else
                    {
                        roi |= cv::Rect(cv::Point(cvRound(pt.x), cvRound(pt.y)), cv::Size(1, 1));
                    }
                }
            }
            pix_rate_channel.measure(frame, roi, has_roi, time_s);
            pix_rate_channel.estimate(time_s);
            pix_rate_channel.demodulate(time_s);
            {
                // 扫速先验的几何比例 Z/(f·R)：EKF 的距离与半径 + 相机焦距
                double z_m = 0.0, r_m = 0.0;
                if (tracker.initialized())
                {
                    z_m = std::hypot(tracker.ekf().state()(0), tracker.ekf().state()(2));
                    r_m = tracker.ekf().state()(8);
                }
                const double f_px = camera.camera_matrix.empty()
                                        ? 0.0
                                        : camera.camera_matrix.at<double>(0, 0);
                pix_rate_channel.sweep_scale =
                    (z_m > 0.3 && z_m < 12.0 && r_m > 0.15 && f_px > 1.0)
                        ? z_m / (f_px * r_m)
                        : 0.0;
            }
            pix_rate_channel.t_now_ = time_s;
            SYSTEM_LOG("sweep_prior", pix_rate_channel.sweep_omega);
            SYSTEM_LOG("sin_valid", pix_rate_channel.sin_valid_ ? 1.0 : 0.0);
            SYSTEM_LOG("sin_A", pix_rate_channel.sin_A_);
            SYSTEM_LOG("sin_b", pix_rate_channel.sin_b_);
            SYSTEM_LOG("sin_f", pix_rate_channel.sin_om_);
            SYSTEM_LOG("sin_rms", pix_rate_channel.sin_rms_);
            if (pix_rate_channel.sin_valid_)
            {
                SYSTEM_LOG("omega_pred",
                           pix_rate_channel.sinOmegaAt(time_s + 0.3));  // 弹丸到达预测
            }
            double inject_omega = 0.0, inject_sigma = 0.0;
            bool inject = false;
            double driving_omega = 0.0; // 俯视图板相位用的"实际检测角速度"
            bool have_driving = false;
            double stable_omega = 0.0;
            if (pix_rate_channel.stableOmega(stable_omega))
            {
                // 旋转区间：恒定 ±πf/2 平顶注入（先稳定，正弦波动后续再做）。
                // 符号随解调方向多数表决走，区间随锁定状态走（解锁即结束）。
                inject_omega = stable_omega;
                inject_sigma = 0.5;
                inject = true;
                driving_omega = stable_omega;
                have_driving = true;
            }
            else if (pix_rate_channel.demod_ok)
            {
                inject_omega = pix_rate_channel.demod_rate();
                inject_sigma = 0.5;
                inject = true;
                driving_omega = pix_rate_channel.demod_rate();
                have_driving = true;
            }
            else if (pix_rate_channel.valid)
            {
                inject_omega = pix_rate_channel.omega;
                inject_sigma = pix_rate_channel.sigma;
                inject = true;
                driving_omega = pix_rate_channel.omega;
                have_driving = true;
            }
            else if (pix_rate_channel.coast(time_s, inject_omega, inject_sigma))
            {
                inject = true;
                driving_omega = inject_omega;
                have_driving = true;
            }
            if (inject && tracker.injectRateObservation(inject_omega, inject_sigma, time_s))
            {
                SYSTEM_LOG("pix_rate", inject_omega);
                SYSTEM_LOG("pix_f", pix_rate_channel.f_peak);
                SYSTEM_LOG("pix_lock", pix_rate_channel.locked ? 1.0 : 0.0);
            }
            // ===== 俯视图板相位：由"实际检测的角速度"独立积分（与 EKF 解耦）。
            // 验收口径：俯视图四块板的转速与实拍视频转速一致（学长：目前只要求
            // 这个）。EKF 的效果只由准心评判（青环=锁定板，黄叉=预测瞄准点）。
            // 通道盲区时相位保持（没有观测就没有转速）；车心/半径仍用 EKF
            // （与转速无关、且稳定）。
            if (have_driving)
            {
                if (!topdown_phase_init)
                {
                    topdown_phase = tracker.initialized() ? tracker.ekf().state()(6) : 0.0;
                    topdown_phase_init = true;
                    topdown_last_t = time_s;
                }
                const double dts = std::max(0.0, std::min(0.05, time_s - topdown_last_t));
                topdown_phase += driving_omega * dts;
                topdown_last_t = time_s;
            }
            topdown_driving_rate = have_driving ? driving_omega : 0.0;
            {
                static double s_prev_obs_phase = 0.0;
                static double s_prev_t = 0.0;
                if (obs_disp_init && s_prev_t > 0.0 && time_s > s_prev_t &&
                    time_s - s_prev_t < 0.05)
                {
                    SYSTEM_LOG("obs_rate",
                               (obs_disp_phase - s_prev_obs_phase) / (time_s - s_prev_t));
                }
                if (obs_disp_init)
                {
                    s_prev_obs_phase = obs_disp_phase;
                    s_prev_t = time_s;
                }
                else
                {
                    s_prev_t = 0.0;
                }
            }
            SYSTEM_LOG("td_phase", topdown_phase);
            SYSTEM_LOG("td_rate", topdown_driving_rate);
            SYSTEM_LOG("demod_ok", pix_rate_channel.last_fail == 0 ? 1.0 : 0.0);
        }

        const PredictionResult prediction =
            tracker.step(tracked_poses, gimbal, time_s, latency_s);
        const rotating_armor::TrackerDiagnostics &diagnostics = tracker.diagnostics();
        const int64_t t3b = cv::getTickCount();

        // ===== 相位差（"板转得跟不跟得上"的量化）=====
        // 把每个观测按最近的模型槽位配对，比较观测法向与模型法向：
        //   |lag| 长期接近 90° → 槽位归属错了一格（板"跳半圈"的典型来源）；
        //   |lag| 一路变大     → ω 偏小，相位在慢慢滑（窗口估计的固有滞后）。
        double phase_lag_deg = 0.0;
        bool has_phase_lag = false;
        if (tracker.initialized())
        {
            const rotating_armor::State &st = tracker.ekf().state();
            const int want_id = pinned > 0 ? pinned : diagnostics.target_id;
            double best_d = 0.10; // 只认 10 cm 内的板：够近才说明配对可信
            for (const ArmorPose &pose : poses)
            {
                if (pose.detection.target_id != want_id)
                    continue;
                const Eigen::Vector3d obs(pose.position_gimbal_m[0], pose.position_gimbal_m[1],
                                          pose.position_gimbal_m[2]);
                for (int slot = 0; slot < kPlateCount; ++slot)
                {
                    const Eigen::Vector3d model =
                        rotating_armor::platePosition(st, slot, kPlateCount);
                    const double d = (model - obs).norm();
                    if (d < best_d)
                    {
                        best_d = d;
                        // 模型里板的外法向 = 相位 + π（RotatingArmorModel.cpp 的
                        // platePhaseToWorldYaw），直接用相位比会差 180°。
                        const double model_phase = st(6) + M_PI * 0.5 * slot;
                        const double model_yaw = rotating_armor::wrapToPi(model_phase + M_PI);
                        phase_lag_deg = rotating_armor::wrapToPi(pose.armor_yaw - model_yaw) *
                                        180.0 / M_PI;
                        has_phase_lag = true;
                        static const bool lag_debug = std::getenv("LAG_DEBUG") != nullptr;
                        if (lag_debug)
                            std::printf("[lag] id=%d slot=%d d=%.3f yaw=%+.0f model=%+.0f "
                                        "phase6=%+.0f R=%.3f w=%+.2f\n",
                                        pose.detection.target_id, slot, d,
                                        pose.armor_yaw * 180.0 / M_PI,
                                        model_yaw * 180.0 / M_PI, st(6) * 180.0 / M_PI, st(8),
                                        st(7));
                    }
                }
            }
        }

        const double frame_dt_ms =
            last_frame_time_s > 0.0 ? (time_s - last_frame_time_s) * 1000.0 : 0.0;
        last_frame_time_s = time_s;

        // ===== 迷你俯视图：30 Hz 渲染 + JPEG + 推给网页 =====
        // 网页通道独立于窗口：--no-display 也照推（只是不开本地窗口）。
        cv::Mat mini_canvas;
        if (cv::getTickCount() - last_mini_tick >= mini_interval_ticks)
        {
            last_mini_tick = cv::getTickCount();
            rotating_armor::MiniTopDownFrame mini;
            mini.camera_position = Eigen::Vector3d(0.0, 0.0, 0.0); // 外参未标定，与云台重合
            // 选中的 id 丢了就整块空掉（不自动换成别的车），最多容忍 0.25 s 的检测抖动。
            const bool pinned_visible =
                pinned <= 0 || (time_s - pinned_last_seen_s) < 0.25;
            if (tracker.initialized() && pinned_visible)
            {
                const rotating_armor::State &state = tracker.ekf().state();
                mini.has_body = true;
                mini.body_centre = Eigen::Vector3d(
                    state(0), state(2), state(4) + tracker.slotZOffsetMean());
                const double radius = state(8);
                // 板相位：优先用"实际检测角速度"独立积分的相位（与 EKF 解耦，
                // 验收=俯视图转速与实拍视频一致）；通道从未锁定时退回 EKF yaw。
                // 板相位优先级：观测相位（零延迟真实相位）> 通道积分相位
                // > EKF yaw。观测在，板就跟着真值走；出画回退通道。
                const double body_phase_for_plates =
                    topdown_phase_init ? topdown_phase : state(6);
                // 固定半宽：四块板画一样长。以前取"本帧检测到的板型"，两台车大小板
                // 交替出现时长度会一格一格跳，看着很难受。
                const double half_w = 0.5 * plate_width_m;
                for (int slot = 0; slot < kPlateCount; ++slot)
                {
                    const double phase = body_phase_for_plates + M_PI * 0.5 * slot;
                    mini.plate_centres.push_back(
                        Eigen::Vector3d(state(0) - radius * std::cos(phase),
                                        state(2) - radius * std::sin(phase),
                                        state(4) + tracker.slotZOffsetMean()));
                    mini.plate_yaws.push_back(phase);
                    mini.plate_half_widths.push_back(half_w);
                }
                // 观测叠加：白十字 + 法向短线（只看选中的那条轨迹）。
                for (const ArmorPose &pose : tracked_poses)
                {
                    mini.observed_centres.push_back(
                        Eigen::Vector3d(pose.position_gimbal_m[0], pose.position_gimbal_m[1],
                                        pose.position_gimbal_m[2]));
                    mini.observed_yaws.push_back(pose.armor_yaw);
                }
                mini.aim_slot = diagnostics.aim_slot;
                if (diagnostics.aim_slot >= 0 && diagnostics.aim_slot < kPlateCount)
                {
                    mini.has_valid_aim = true;
                    mini.valid_aim = rotating_armor::platePosition(state, diagnostics.aim_slot,
                                                                  kPlateCount);
                }
                if (diagnostics.predicted_aim_world.allFinite())
                {
                    mini.has_predicted_aim = true;
                    mini.predicted_aim = diagnostics.predicted_aim_world;
                }
                mini.fire_allowed = prediction.fire_allowed;
                mini.caption = cv::format("d=%.2fm R=%.2fm w=%+.1f%s", prediction.distance_m, radius,
                                          topdown_phase_init ? topdown_driving_rate
                                                             : static_cast<double>(prediction.body_yaw_velocity),
                                          topdown_phase_init ? "(det)" : "");
                if (has_phase_lag)
                    mini.caption += cv::format("  lag=%+.0f", phase_lag_deg);
            }
            else if (pinned > 0)
            {
                mini.caption = cv::format("id=%d lost", pinned);
            }
            mini_canvas = rotating_armor::renderMiniTopDown(mini);
            std::vector<unsigned char> jpeg;
            const std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 80};
            if (cv::imencode(".jpg", mini_canvas, jpeg, params))
                SYSTEM_FRAME(jpeg);
            if (dump_every > 0 && frame_cnt % dump_every == 0)
                cv::imwrite(cv::format("./tmp/system_mini_%06d.png", frame_cnt), mini_canvas);
        }

        // ===== 可选：独立的相机画面窗口（含 CameraOverlay 的完整叠加）=====
        if (show_camera)
        {
            cv::Mat canvas = frame.clone();
            rotating_armor::CameraOverlayFrame overlay;
            overlay.detections = &dets;
            overlay.poses = &poses;
            overlay.camera = &camera;
            overlay.gimbal = gimbal;
            overlay.has_state = tracker.initialized();
            if (overlay.has_state)
                overlay.state = tracker.ekf().state();
            overlay.slot_z_offset = tracker.slotZOffsetMean();
            overlay.diagnostics = diagnostics;
            overlay.prediction = prediction;
            overlay.ekf_enabled = true;
            overlay.scale = 1.0;
            overlay.status_text =
                cv::format("Det=%zu Poses=%zu", dets.size(), poses.size());
            overlay.fps_text = cv::format("fps_loop=%.1f", fps_loop_ema);
            rotating_armor::drawCameraOverlay(canvas, overlay);
            if (dump_every > 0 && frame_cnt % dump_every == 0)
                cv::imwrite(cv::format("./tmp/system_cam_%06d.png", frame_cnt), canvas);
            if (!no_display && cv::getTickCount() - last_show_tick >= show_interval_ticks)
            {
                if (!window_sized)
                {
                    cv::resizeWindow("camera_system", canvas.cols, canvas.rows);
                    window_sized = true;
                }
                cv::imshow("camera_system", canvas);
                last_show_tick = cv::getTickCount();
            }
        }

        // ===== 遥测：与生产程序（main.cpp）/ 回放同名的键，喂网页折线 =====
        {
            // 0 = 没锁到（Idle），1 = 跟踪中（Unstable），2 = 稳定（Stable）。
            const double state_code = !tracker.initialized()
                ? 0.0
                : (prediction.state == TrackingState::Stable ? 2.0 : 1.0);
            const double reproj = poses.empty() ? 0.0 : poses.front().reprojection_error;
            SYSTEM_LOG("frame_dt_ms", frame_dt_ms);
            SYSTEM_LOG("grab_ms", tick_ms(t0, t1));
            SYSTEM_LOG("detect_ms", tick_ms(t1, t2));
            SYSTEM_LOG("pnp_ms", tick_ms(t2, t3));
            SYSTEM_LOG("ekf_ms", tick_ms(t3, t3b));
            SYSTEM_LOG("loop_ms", tick_ms(t0, cv::getTickCount()));
            SYSTEM_LOG("detect_count", static_cast<double>(dets.size()));
            SYSTEM_LOG("armor_count", static_cast<double>(dets.size()));
            SYSTEM_LOG("reproj_px", reproj);
            SYSTEM_LOG("distance_m", prediction.distance_m);
            SYSTEM_LOG("tracking", tracker.initialized() ? 1.0 : 0.0);
            SYSTEM_LOG("ekf_state",
                       static_cast<double>(static_cast<int>(prediction.state)));
            SYSTEM_LOG("target_id", static_cast<double>(prediction.target_id));
            SYSTEM_LOG("omega", prediction.body_yaw_velocity);
            SYSTEM_LOG("radius", prediction.radius_1_m);
            SYSTEM_LOG("nis", diagnostics.last_nis);
            // 更新被拒的诊断：best_gate_distance_m 是"走到 NIS 门那一步的候选"
            // 里最近的板距（inf = 一个都没走到，位置门/连续性门就先拒了）；
            // cont_rej 是连续性门累计拒掉的候选数。用来看 60% 空帧到底卡在哪。
            // 没锁上时 associateAndUpdate 根本不跑，diagnostics 里是上一轮的旧值，
            // 直接按 99 记（= 没走到门那一步），免得把陈旧值读成"候选就在门口"。
            const bool linked = tracker.initialized();
            SYSTEM_LOG("gate_dist",
                       linked && std::isfinite(diagnostics.best_gate_distance_m)
                           ? diagnostics.best_gate_distance_m
                           : 99.0);
            SYSTEM_LOG("gate_nis",
                       linked && std::isfinite(diagnostics.best_gate_nis)
                           ? diagnostics.best_gate_nis
                           : 99.0);
            SYSTEM_LOG("pin_poses", static_cast<double>(tracked_poses.size()));
            // 这一帧检出了、但不是被选中 id 的观测数——用来区分"压根没检到这台车"
            // 和"检到了但 id 分成了别的号"（四板车的 id 是靠四块板推出来的）。
            SYSTEM_LOG("other_poses",
                       static_cast<double>(poses.size() - tracked_poses.size()));
            // 其中有多少块"别的 id"的板其实就在当前跟踪中心 0.8 m 以内——即
            // "同一台车被分成了另一个 id"。这部分是可以救回来的观测。
            double near_other = 0.0;
            if (tracker.initialized())
            {
                const double cx = prediction.center_x_m;
                const double cy = prediction.center_y_m;
                for (const ArmorPose &pose : poses)
                {
                    if (pinned > 0 && pose.detection.target_id == pinned)
                        continue;
                    const double dx = pose.position_gimbal_m[0] - cx;
                    const double dy = pose.position_gimbal_m[1] - cy;
                    if (std::hypot(dx, dy) < 0.8)
                        ++near_other;
                }
            }
            SYSTEM_LOG("near_other", near_other);
            // "四块板只振荡不转"的直接嫌疑人：单板重锚是硬对齐（rotatePhase +
            // shiftCenter），每触发一次就把相位瞬间拽到观测上，观感就是来回跳。
            // 双板整车解要求观测带法向（has_yaw）；法向缺失时它永远不会触发。
            double yaw_poses = 0.0;
            for (const ArmorPose &pose : tracked_poses)
            {
                const double n = std::sqrt(pose.normal_gimbal[0] * pose.normal_gimbal[0] +
                                           pose.normal_gimbal[1] * pose.normal_gimbal[1] +
                                           pose.normal_gimbal[2] * pose.normal_gimbal[2]);
                if (n > 0.5)
                    ++yaw_poses;
            }
            SYSTEM_LOG("yaw_poses", yaw_poses);
            SYSTEM_LOG("realigns", static_cast<double>(diagnostics.single_plate_realigns));
            SYSTEM_LOG("pair_solves", static_cast<double>(diagnostics.pair_body_solves));
            SYSTEM_LOG("body_refits", static_cast<double>(diagnostics.body_refits));
            SYSTEM_LOG("yaw_deg", prediction.body_yaw * 180.0 / M_PI);
            SYSTEM_LOG("cont_rej", static_cast<double>(diagnostics.continuity_rejects));
            SYSTEM_LOG("rej_frame", static_cast<double>(diagnostics.rejected_this_frame));
            SYSTEM_LOG("updates", diagnostics.updates_this_frame);
            SYSTEM_LOG("missed", diagnostics.missed_frames);
            SYSTEM_LOG("aim_miss", prediction.fire_miss_m * 1000.0);
            SYSTEM_LOG("fire", prediction.fire_allowed ? 1.0 : 0.0);
            SYSTEM_LOG("latency_ms", diagnostics.measured_latency_s * 1000.0);
            if (has_phase_lag)
                SYSTEM_LOG("phase_lag_deg", phase_lag_deg);
            SYSTEM_LOG("pinned_id", pinned);
            SYSTEM_LOG("state_code", state_code);
        }

        // ===== 把"可选 id"发给网页（10 Hz 足够，下拉菜单不用每帧重建）=====
        if (++targets_tick >= 10)
        {
            targets_tick = 0;
            SYSTEM_TARGETS(std::vector<int>(seen_ids.begin(), seen_ids.end()),
                           std::vector<int>(visible_ids.begin(), visible_ids.end()));
        }

        // ===== 吞吐统计（每 150 帧一行）=====
        const double loop_ms = tick_ms(t0, cv::getTickCount());
        if (loop_ms > 0.1)
        {
            const double f = 1000.0 / loop_ms;
            fps_loop_ema = (fps_loop_ema <= 0.0) ? f : 0.9 * fps_loop_ema + 0.1 * f;
        }
        ++frame_cnt;
        if (++stat_cnt >= 150)
        {
            stat_cnt = 0;
            const std::string lag_text = has_phase_lag
                ? cv::format(" lag=%+.0f", phase_lag_deg)
                : std::string();
            std::printf("[%d帧] fps_loop=%.1f | grab=%.1f detect=%.1f pnp=%.1f ekf=%.1f ms | "
                        "Det=%zu Poses=%zu | EKF id=%d R=%.2f w=%+.2f%s\n",
                        frame_cnt, fps_loop_ema, tick_ms(t0, t1), tick_ms(t1, t2),
                        tick_ms(t2, t3), tick_ms(t3, t3b), dets.size(), poses.size(),
                        static_cast<int>(prediction.target_id), prediction.radius_1_m,
                        prediction.body_yaw_velocity, lag_text.c_str());
        }

        if (!no_display && show_camera)
        {
            const int key = cv::pollKey();
            if (key == 27 || key == 'q')
                break;
            if (key == 'r')
            {
                tracker.reset();
                std::cout << "已重置 EKF\n";
            }
        }
    }

    // 视频放完后可以让网页多活一会儿：最后一帧 JPEG 和曲线都还在，方便对着看。
    if (hold_s > 0.0)
    {
        std::cout << "[hold] 网页保持 " << hold_s << " 秒（Ctrl+C 立即退出）\n";
        const auto until = std::chrono::steady_clock::now() +
                           std::chrono::milliseconds(static_cast<long long>(hold_s * 1000.0));
        while (std::chrono::steady_clock::now() < until)
        {
            if (show_camera)
            {
                const int key = cv::pollKey();
                if (key == 27 || key == 'q')
                    break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    STOP_SYSTEM_WEB();
    std::cout << "[web] 已停止\n";
    return 0;
}
