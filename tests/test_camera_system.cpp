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
    bool pix_rate = false;        // --pix-rate：图像域转速通道（亮斑图案频率）
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
        cv::Rect2d roi_ema;
        bool roi_valid{false};

        void measure(const cv::Mat &bgr, const cv::Rect &roi_raw, bool has_det, double t)
        {
            if (has_det)
            {
                if (!roi_valid)
                {
                    roi_ema = roi_raw;
                    roi_valid = true;
                }
                else
                {
                    const double a = 0.15;
                    roi_ema.x += a * (roi_raw.x - roi_ema.x);
                    roi_ema.y += a * (roi_raw.y - roi_ema.y);
                    roi_ema.width += a * (roi_raw.width - roi_ema.width);
                    roi_ema.height += a * (roi_raw.height - roi_ema.height);
                }
                // 自转图案横扫范围远大于单板检测框：ROI 太小会把扫出去的灯条
                // 截掉、图案频率变弱（实测注入密度不足）。给个下限，围绕中心生长。
                const double min_w = 420.0, min_h = 320.0;
                if (roi_ema.width < min_w)
                {
                    roi_ema.x -= 0.5 * (min_w - roi_ema.width);
                    roi_ema.width = min_w;
                }
                if (roi_ema.height < min_h)
                {
                    roi_ema.y -= 0.5 * (min_h - roi_ema.height);
                    roi_ema.height = min_h;
                }
            }
            if (!roi_valid)
            {
                return;
            }
            cv::Rect roi(cvRound(roi_ema.x), cvRound(roi_ema.y),
                         cvRound(roi_ema.width), cvRound(roi_ema.height));
            roi &= cv::Rect(0, 0, bgr.cols, bgr.rows);
            if (roi.width < 8 || roi.height < 8)
            {
                return;
            }
            cv::Mat gray;
            cv::cvtColor(bgr(roi), gray, cv::COLOR_BGR2GRAY);
            cv::Scalar mean, sd;
            cv::meanStdDev(gray, mean, sd);
            const double thr = std::max(150.0, mean[0] + 3.0 * sd[0]);
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
            if (area < 20.0)
            {
                return;
            }
            buf.push_back({t, area, sx / area / std::max(1, gray.cols)});
            while (!buf.empty() && t - buf.front().t > window_s)
            {
                buf.pop_front();
            }
        }

        void estimate(double t)
        {
            valid = false;
            if (buf.size() < 60 || t - last_est_s < 0.15)
            {
                return;
            }
            last_est_s = t;
            const double t0 = buf.front().t;
            double mean_area = 0.0;
            for (const Sample &s : buf)
            {
                mean_area += s.area;
            }
            mean_area /= static_cast<double>(buf.size());
            double best_f = 0.0, best_amp = 0.0, median_amp = 0.0, band_share = 0.0;
            double best_phase = 0.0;
            std::vector<double> amps;
            amps.reserve(140);
            double total_power = 0.0;
            for (double f = 2.0; f <= 15.0; f += 0.1)
            {
                std::complex<double> sa(0.0, 0.0), sx(0.0, 0.0);
                for (const Sample &s : buf)
                {
                    const double dt = s.t - t0;
                    const double w = 0.5 - 0.5 * std::cos(2.0 * CV_PI * dt / window_s);
                    const std::complex<double> e{std::cos(-2.0 * CV_PI * f * dt),
                                                 std::sin(-2.0 * CV_PI * f * dt)};
                    sa += (s.area - mean_area) * w * e;
                    sx += (s.cx - 0.5) * w * e;
                }
                amps.push_back(std::abs(sa));
                if (f > 0.5)
                {
                    total_power += std::norm(sa);
                }
                if (f >= 2.0 && std::abs(sa) > best_amp)
                {
                    best_amp = std::abs(sa);
                    best_f = f;
                    best_phase = std::arg(sx * std::conj(sa));
                }
            }
            {
                std::vector<double> sorted = amps;
                std::sort(sorted.begin(), sorted.end());
                median_amp = sorted[sorted.size() / 2];
            }
            // 基频选择用"谐波家族评分"：score(f) = A(f) + 0.6*A(2f)。整流会把
            // 能量推到二倍频，单看最大幅值容易锁到次谐波（实测 5.1Hz 真图案
            // 被锁成 2.5Hz -> 板半速转）；家族评分让基频（连同其倍频）总体
            // 最强的候选胜出：真 5.1Hz 时 score(5.1)=A5.1+0.6*A10.2 压过
            // score(2.5)=A2.5+0.6*A5.1，真慢转（2.5Hz 强、5.1 是其倍频）时
            // 2.5 仍胜。
            // 频率连续性：转速不会瞬间跳 2 倍。刚锁定过（1.5s 内）时只在
            // f_peak ± 0.6Hz 邻域内选峰——弱信号窗里次谐波（实测 2.4~2.8Hz）
            // 的家族评分偶尔会压过真基频，保持期不换挡。
            const bool freq_hold = (t - last_lock_s) < 1.5 && f_peak > 0.0;
            {
                double best_score = -1.0;
                for (std::size_t k = 0; k < amps.size(); ++k)
                {
                    const double f = 2.0 + 0.1 * static_cast<double>(k);
                    if (freq_hold && std::abs(f - f_peak) > 0.6)
                    {
                        continue;
                    }
                    const double f2 = 2.0 * f;
                    double score = amps[k];
                    if (f2 <= 15.0)
                    {
                        const std::size_t k2 = static_cast<std::size_t>(std::lround((f2 - 2.0) / 0.1));
                        if (k2 < amps.size())
                        {
                            score += 0.6 * amps[k2];
                        }
                    }
                    if (score > best_score)
                    {
                        best_score = score;
                        // 记住家族头部的幅值/相位（用头部 f 的谱值）
                    }
                }
                // 用家族头部的 f 重算 best_amp/best_phase（保持后续门槛语义：
                // prominence/band 相对本头部频率的谱线）
                for (std::size_t k = 0; k < amps.size(); ++k)
                {
                    const double f = 2.0 + 0.1 * static_cast<double>(k);
                    const double f2 = 2.0 * f;
                    double score = amps[k];
                    if (f2 <= 15.0)
                    {
                        const std::size_t k2 = static_cast<std::size_t>(std::lround((f2 - 2.0) / 0.1));
                        if (k2 < amps.size())
                        {
                            score += 0.6 * amps[k2];
                        }
                    }
                    if (score >= best_score - 1e-12)
                    {
                        // 重新投影复谱取该 f 的幅值与相位
                        std::complex<double> sa2(0.0, 0.0), sx2(0.0, 0.0);
                        for (const Sample &s : buf)
                        {
                            const double dt = s.t - t0;
                            const double w = 0.5 - 0.5 * std::cos(2.0 * CV_PI * dt / window_s);
                            const std::complex<double> e{std::cos(-2.0 * CV_PI * f * dt),
                                                         std::sin(-2.0 * CV_PI * f * dt)};
                            sa2 += (s.area - mean_area) * w * e;
                            sx2 += (s.cx - 0.5) * w * e;
                        }
                        best_amp = std::abs(sa2);
                        best_f = f;
                        best_phase = std::arg(sx2 * std::conj(sa2));
                        break;
                    }
                }
            }
            for (std::size_t k = 0; k < amps.size(); ++k)
            {
                const double f = 2.0 + 0.1 * static_cast<double>(k);
                band_share += amps[k] * amps[k];
            }
            band_share /= std::max(total_power, 1e-9);
            if (best_amp < 1e-9 || median_amp <= 0.0)
            {
                return;
            }
            const double prominence = best_amp / median_amp;
            const bool strict = buf.size() >= 110 && prominence >= 4.0 && band_share >= 0.50;
            // 迟滞：刚锁定过（1.5s 内）且谱峰还钉在同一频率上（±0.3Hz）时，
            // 放宽门槛维持注入——自转在估计窗之间是持续的，间隙里衰减项会
            // 把 omega 拖回 0。
            const double age = t - last_lock_s;
            const bool hold = age < 1.5 && std::abs(best_f - f_peak) <= 0.3 &&
                              prominence >= 2.0 && band_share >= 0.35;
            if (!strict && !hold)
            {
                return;
            }
            if (strict)
            {
                f_peak = best_f;
                omega = CV_PI * best_f / 2.0;
                omega *= std::sin(best_phase) >= 0.0 ? 1.0 : -1.0;
                last_lock_s = t;
            }
            sigma = 0.4 + 0.8 / std::max(prominence, 1.0) + (hold && !strict ? 0.3 : 0.0);
            valid = true;
        }
        // 惯性滑行：谱估计断续（车出画/谱弱）时，锁定后 1s 内按上次 ω 注入，
        // σ 随龄期增大，滤波器自行折权。自转是持续运动，短暂盲区不该把
        // omega 打回 0。
        bool coast(double t, double &omega_out, double &sigma_out) const
        {
            const double age = t - last_lock_s;
            if (!(age >= 0.0) || age > 1.0 || f_peak <= 0.0)
            {
                return false;
            }
            omega_out = omega;
            sigma_out = 0.6 + 0.9 * age;
            return true;
        }
    } pix_rate_channel;
    // 俯视图板相位的独立积分器（与 EKF 解耦，由检测转速驱动）。
    double topdown_phase{0.0};
    double topdown_last_t{0.0};
    double topdown_driving_rate{0.0};
    bool topdown_phase_init{false};
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

        const auto dets = armor_detect(frame, enemy_red ? TeamColor::Red : TeamColor::Blue);
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
            if (has_roi)
            {
                roi.x -= roi.width * 3 / 2;
                roi.y -= roi.height * 3 / 2;
                roi.width *= 4;
                roi.height *= 4;
                roi &= cv::Rect(0, 0, frame.cols, frame.rows);
            }
            pix_rate_channel.measure(frame, roi, has_roi, time_s);
            pix_rate_channel.estimate(time_s);
            double inject_omega = 0.0, inject_sigma = 0.0;
            bool inject = false;
            double driving_omega = 0.0; // 俯视图板相位用的"实际检测角速度"
            bool have_driving = false;
            if (pix_rate_channel.valid)
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
            SYSTEM_LOG("td_phase", topdown_phase);
            SYSTEM_LOG("td_rate", topdown_driving_rate);
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
                const double body_phase_for_plates = topdown_phase_init ? topdown_phase : state(6);
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
