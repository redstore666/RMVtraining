// 实时相机测试：取流 + 装甲板检测 + PnP 位姿显示（需连接相机）。
//
// 用法: ./build/test_camera_live [--red|--blue] [--no-display]
//        [--ekf|--no-ekf] [--topdown|--no-topdown] [--topdown-ppm P] [--topdown-follow]
//   --no-display  纯吞吐模式：无窗口/无绘制/无按键（Ctrl+C 退出），
//                 用于测处理吞吐上限与相机真实帧率
//   --ekf         跑旋转车体 EKF 并把结果画到画面上（默认开；--no-ekf 关掉，
//                 回到"只有 PnP 单帧位姿"的旧行为）
//   --topdown     额外开一个"EKF 俯视图"窗口（默认开，需要 --ekf）
//   --topdown-ppm P  俯视图像素/米（默认 200，即 ±2 m 视野）
//   --topdown-follow 俯视图跟着车体中心走（默认固定原点=云台）
//   --video FILE  不连相机、改读视频（上车前在没相机的机器上验收 EKF 显示用）
//   --dump-every N 每 N 帧把带标注画面存到 ./tmp/live_dump_XXXXXX.png
//   --topdown-out DIR 俯视图每 50 帧导出 DIR/t%06d.png
//
// 这是第一阶段验收用的工具："相机窗口 + 可选俯视图窗口"，不含网页/卡片。
// 想在网页上看曲线 + 右下角迷你俯视图（不显示相机画面、帧率更高），用
// ./build/test_camera_system。
//
// 按键:  ESC/q 退出 | s 保存当前帧(./tmp/) | m 打印各观测 PnP 的 R/t 矩阵（全精度）
//        | r 重置 EKF（换车/换场地后手动清状态）
// 画面:  检测框 + 装甲板坐标系(x红/y绿/z蓝，板中心为原点)
//        + 左上角面板：每个观测的 编号/距离/重投影误差/角度 + PnP 的 R/t 矩阵
//        + EKF 叠加：品红十字 = 车体中心，黄十字 = EKF 预测瞄准点（含提前量），
//          右上角文本框 = 锁定状态/半径/角速度/距离/云台指令/开火判据
//        + 状态栏含两个帧率：fps_cam（相机固件帧率）与 fps_nodisp
//          （"不带显示窗口"的吞吐——窗口照常显示，只是不计 GUI 时间）
// 计时:  全部用 OpenCV 库函数 cv::getTickCount()/cv::getTickFrequency()
// 控制台: 每 150 帧一行分段耗时（grab/detect/pnp/draw/show/key）与拒绝统计；
//         每 300 帧一次重投影误差分布（均不刷屏）
//
// 帧率口径说明（答疑"去掉 imshow 为何从 70 上 100+"）：
//   fps_cam 取相机固件 "ResultingFrameRate" 节点（MV_CC_GetFloatValue），
//   相机自己算、不看主机行为——纯硬件层帧率，加不加窗口都不变。
//   fps_nodisp = 1/(grab+detect+pnp+draw 耗时)，再受 fps_cam 封顶，
//   即"窗口显示但 GUI 不占计时"时的吞吐，等于去掉显示窗口后的帧率。
//   带窗口实测吞吐 ≈ 1/(处理 + show + key)，与 fps_nodisp 的差距就是
//   GUI（imshow/pollKey，1080p 常被合成器/vsync 限制）的代价。
//   imshow 已限频 30fps + pollKey 不等待；要测纯吞吐用 --no-display。
#include "StudentTasks.hpp"
#include "RotatingArmorTracker.hpp"
#include "TopDownView.hpp"
// 检测框/坐标轴/PnP 面板/EKF 面板的画法（与 test_camera_system 共用一份）。
#include "CameraOverlay.hpp"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp> // solvePnP / Rodrigues / drawFrameAxes
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <utility>

int main(int argc, char **argv)
{
    TeamColor enemy = TeamColor::Red;
    // --no-display：纯吞吐模式。不建窗口/不 imshow/不 waitKey/不绘制——
    // GUI（imshow+waitKey+合成器）是 1080p 下最大的非算法开销，去掉后
    // fps_loop 才是处理吞吐上限，fps_grab 也会回落为相机真实出帧率
    // （处理快于相机时 GetOneFrameTimeout 阻塞等真下一帧）。
    // 注意：该模式无按键可退，用 Ctrl+C。
    bool no_display = false;
    // EKF 默认开：这个工具本来就是看"检测 + 位姿"的，加上 EKF 后
    // 车体中心/预测瞄准点与俯视图一起看，才能判断"跟踪到底锁没锁住"。
    bool ekf_enabled = true;
    bool topdown_enabled = true;
    double topdown_ppm = 200.0;
    bool topdown_follow = false;
    // --video FILE：不连相机，改读视频（上车前在没有相机的机器上验收 EKF 显示用）。
    std::string video_path;
    // --dump-every N：每 N 帧把带标注的画面写到 ./tmp/live_dump_XXXXXX.png（0=不写）。
    int dump_every = 0;
    // --topdown-out DIR：把俯视图逐帧写成 DIR/t%06d.png（无需看窗口也能留证据）。
    std::string topdown_out;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--red")
            enemy = TeamColor::Red;
        else if (arg == "--blue")
            enemy = TeamColor::Blue;
        else if (arg == "--no-display")
            no_display = true;
        else if (arg == "--ekf")
            ekf_enabled = true;
        else if (arg == "--no-ekf")
            ekf_enabled = false;
        else if (arg == "--topdown")
            topdown_enabled = true;
        else if (arg == "--no-topdown")
            topdown_enabled = false;
        else if (arg == "--topdown-follow")
            topdown_follow = true;
        else if (arg == "--topdown-out")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--topdown-out 后面要跟目录\n";
                return 2;
            }
            topdown_out = argv[++i];
            topdown_enabled = true;
        }
        else if (arg == "--dump-every")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--dump-every 后面要跟一个整数\n";
                return 2;
            }
            dump_every = std::atoi(argv[++i]);
        }
        else if (arg == "--video")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--video 后面要跟视频路径\n";
                return 2;
            }
            video_path = argv[++i];
        }
        else if (arg == "--topdown-ppm")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--topdown-ppm 后面要跟一个数（像素/米）\n";
                return 2;
            }
            topdown_ppm = std::atof(argv[++i]);
            if (!(topdown_ppm > 0.0))
            {
                std::cerr << "--topdown-ppm 必须是正数\n";
                return 2;
            }
        }
        else
        {
            std::cerr << "未知参数: " << arg << "\n";
            return 2;
        }
    }

    // --no-display 是纯吞吐模式，两个窗口都不建；--topdown 依赖 --ekf。
    if (no_display)
        topdown_enabled = false; // 纯吞吐模式不建俯视图窗口；EKF 本身照跑
    if (topdown_enabled && !ekf_enabled)
    {
        std::cerr << "[俯视图] 需要同时开 EKF，已忽略 --topdown\n";
        topdown_enabled = false;
    }

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
        camera.rotation_camera_to_gimbal = cv::Mat::eye(3, 3, CV_64F);
        camera.translation_camera_to_gimbal = cv::Mat::zeros(3, 1, CV_64F);
    }

    GimbalState gimbal{};

    // ===== EKF：实时跟踪旋转车体（与 test_camera_playback 的 --ekf 同一套）=====
    // 实跑时 yaw/pitch 来自电控；这里没有 IMU，保持 0，于是"世界系 = 云台静止系"，
    // 俯视图里云台原点就是相机光轴方向，够用来判断跟踪是否锁住。
    rotating_armor::TrackerConfig tracker_config;
    rotating_armor::RotatingArmorTracker tracker(tracker_config);
    rotating_armor::TopDownOptions topdown_options;
    topdown_options.follow_target = topdown_follow;
    topdown_options.pixels_per_meter = topdown_ppm;
    rotating_armor::TopDownView topdown_view(topdown_options);
    const std::string topdown_window = "EKF top-down";

    if (!no_display)
    {
        cv::namedWindow("camera_live", cv::WINDOW_NORMAL);
        if (topdown_enabled)
            cv::namedWindow(topdown_window, cv::WINDOW_AUTOSIZE);
    }

    if (topdown_enabled && !topdown_out.empty())
    {
        std::error_code mkdir_error;
        std::filesystem::create_directories(topdown_out, mkdir_error);
    }

    if (no_display)
    {
        std::cout << "[--no-display] 纯吞吐模式：无窗口/无按键，Ctrl+C 退出\n";
    }
    else
    {
        std::cout << "ESC/q 退出, s 保存帧, m 打印 PnP 的 R/t 矩阵\n";
    }
    std::cout << "[extr not calibrated] yaw/pitch 相对相机光轴\n";

    cv::Mat frame;
    int save_idx = 0;
    int stat_cnt = 0;
    int frame_cnt = 0;
    // ===== 计时统一用 OpenCV 库函数：cv::getTickCount() / cv::getTickFrequency() =====
    // 帧率只显示一个：相机固件自算的 ResultingFrameRate（cameraResultingFps()），
    // 由相机固件根据实际出帧计算，与主机处理/显示速度无关——纯硬件层帧率。
    // 主机侧开销看控制台分段耗时：处理慢于相机时 MVS 缓冲积压、取帧立即返回，
    // "取帧间隔"会被拖成处理速度；GUI（imshow/pollKey）的代价看 show/key 两段。
    auto tick_ms = [](int64_t a, int64_t b)
    { return static_cast<double>(b - a) * 1000.0 / cv::getTickFrequency(); };

    double last_wait_ms = 0.0;
    double fps_nodisp_ema = 0.0; // "不带显示窗口"的吞吐（处理段耗时倒数，受相机帧率封顶）
    int64_t last_show_tick = 0; // imshow 限频用：上次真正显示的 tick
    bool window_sized = false; // 首帧把窗口调到图像原始尺寸，避免 imshow 缩放糊字
    std::vector<cv::Mat> dbg_rvecs, dbg_tvecs;
    std::vector<int> dbg_ids;

    // EKF 的时间戳用 steady_clock（单调、不受系统校时影响），起点为程序启动。
    const auto t_start = std::chrono::steady_clock::now();

    // 视频源（--video）与相机源（get_pic）只影响"取帧"，后面整条链路完全一致。
    cv::VideoCapture video_capture;
    if (!video_path.empty())
    {
        video_capture.open(video_path);
        if (!video_capture.isOpened())
        {
            std::cerr << "无法打开视频: " << video_path << "\n";
            return 2;
        }
        std::cout << "[--video] " << video_path << "  " << video_capture.get(cv::CAP_PROP_FRAME_WIDTH)
                  << "x" << video_capture.get(cv::CAP_PROP_FRAME_HEIGHT) << "  "
                  << video_capture.get(cv::CAP_PROP_FPS) << " fps\n";
    }

    while (true)
    {
        const int64_t t0 = cv::getTickCount();
        if (!video_path.empty())
        {
            if (!video_capture.read(frame) || frame.empty())
            {
                std::cout << "[--video] 播放结束\n";
                break;
            }
        }
        else if (!get_pic(frame))
        {
            continue;
        }
        const int64_t t1 = cv::getTickCount();

        // ===== 检测 + PnP =====
        const auto dets = armor_detect(frame, enemy);
        const int64_t t2 = cv::getTickCount();
        const auto poses = armor_solve(dets, camera, gimbal);
        const int64_t t3 = cv::getTickCount();

        // ===== EKF 一步 =====
        // measured latency 取"开始取图 → 位姿算完"这段（t0→t3），也就是这一帧的
        // 观测相对真实世界已经过时多久；EKF 用它把预测时域往后推。
        PredictionResult prediction{};
        rotating_armor::TrackerDiagnostics diagnostics{};
        const double time_s =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
        if (ekf_enabled)
        {
            const double latency_s = tick_ms(t0, t3) / 1000.0;
            prediction = tracker.step(poses, gimbal, time_s, latency_s);
            diagnostics = tracker.diagnostics();
        }
        const int64_t t3b = cv::getTickCount();

        // ===== 绘制 + 显示 + 按键（--no-display 纯吞吐模式整体跳过）=====
        // no_display 时 t4=t5=t35=t3：draw/show/key 三段耗时报 0。
        // t35 = 绘制完成时刻（GUI 开始前）：grab→detect→pnp→draw 的耗时即
        // "不带显示窗口"时的循环时间，其倒数（受相机帧率封顶）即 fps_nodisp。
        int64_t t4 = t3b;
        int64_t t5 = t3b;
        int64_t t35 = t3b;
        if (!no_display)
        {
            // ===== 叠加层：检测框 / 坐标轴 / PnP 面板 / EKF 面板 =====
            // 画法在 tests/CameraOverlay.cpp 里，test_camera_system 用的是同一份实现。
            cv::Mat canvas = frame.clone();
            {
                const float cam_fps = cameraResultingFps();
                const double nodisp_show =
                    cam_fps > 0.0F ? std::min(fps_nodisp_ema, static_cast<double>(cam_fps))
                                   : fps_nodisp_ema;
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
                overlay.ekf_enabled = ekf_enabled;
                overlay.status_text = cv::format("Det=%zu Poses=%zu save=%d", dets.size(),
                                                 poses.size(), save_idx);
                overlay.fps_text = cam_fps >= 0.0F
                    ? cv::format("fps_cam=%.1f fps_nodisp=%.1f", cam_fps, nodisp_show)
                    : cv::format("fps_cam=n/a fps_nodisp=%.1f", nodisp_show);
                overlay.scale = 1.0;
                rotating_armor::CameraOverlayDebug overlay_debug;
                rotating_armor::drawCameraOverlay(canvas, overlay, &overlay_debug);
                dbg_rvecs = overlay_debug.rvecs;
                dbg_tvecs = overlay_debug.tvecs;
                dbg_ids = overlay_debug.ids;
            }

            cv::Mat topdown_canvas; // 本帧俯视图（30fps 门内才真正 imshow）
            // ===== 俯视图：EKF 状态/诊断 + 本帧实测板，画成上帝视角 =====
            if (topdown_enabled)
            {
                rotating_armor::TopDownFrame topdown_frame;
                topdown_frame.timestamp_s = time_s;
                topdown_frame.has_state = tracker.initialized();
                if (topdown_frame.has_state)
                    topdown_frame.state = tracker.ekf().state();
                topdown_frame.diagnostics = diagnostics;
                topdown_frame.result = prediction;
                topdown_frame.gimbal = gimbal;
                const Eigen::Matrix3d gimbal_to_world = rotating_armor::gimbalToWorld(gimbal);
                for (const ArmorPose &pose : poses)
                {
                    if (pose.detection.target_id == 0)
                        continue;
                    rotating_armor::TopDownObservation observation;
                    observation.position =
                        gimbal_to_world * Eigen::Vector3d(pose.position_gimbal_m[0],
                                                          pose.position_gimbal_m[1],
                                                          pose.position_gimbal_m[2]);
                    observation.normal_yaw = rotating_armor::wrapToPi(pose.armor_yaw + gimbal.yaw);
                    observation.target_id = pose.detection.target_id;
                    topdown_frame.observations.push_back(observation);
                }
                topdown_canvas = topdown_view.render(topdown_frame).clone();
                if (!topdown_out.empty() && frame_cnt % 50 == 0)
                {
                    cv::imwrite(cv::format("%s/t%06d.png", topdown_out.c_str(), frame_cnt),
                                topdown_canvas);
                }
            }

            if (dump_every > 0 && frame_cnt % dump_every == 0)
            {
                cv::imwrite(cv::format("./tmp/live_dump_%06d.png", frame_cnt), canvas);
            }

            t35 = cv::getTickCount(); // 绘制完成（GUI 开始前的分界）

            // ===== imshow 降频到约 30fps =====
            // 处理全速跑；只有距上次真正显示 ≥ 33ms 才 imshow 一次，
            // 避免 GUI 队列积压拖慢循环。
            const double show_interval_ticks =
                33.0 / 1000.0 * cv::getTickFrequency();
            if (cv::getTickCount() - last_show_tick >= show_interval_ticks)
            {
                if (!window_sized)
                {
                    // 窗口必须和图像一样大：WINDOW_NORMAL 下 imshow 会把图像缩放进
                    // 当前窗口，窗口比图小就又糊了。
                    cv::resizeWindow("camera_live", canvas.cols, canvas.rows);
                    window_sized = true;
                }
                cv::imshow("camera_live", canvas);
                if (!topdown_canvas.empty())
                    cv::imshow(topdown_window, topdown_canvas);
                last_show_tick = cv::getTickCount();
            }
            t4 = cv::getTickCount();
            // ===== 按键（pollKey：无键返回 -1，不 sleep 不等待）=====
            const int key = cv::pollKey();
            if (key >= 0)
            {
                if (key == 27 || key == 'q')
                    break;
                if (key == 's')
                {
                    const std::string fname = cv::format("./tmp/live_%03d.png", save_idx);
                    cv::imwrite(fname, frame);
                    std::cout << "已保存: " << fname << "\n";
                    ++save_idx;
                }
                else if (key == 'r')
                {
                    tracker.reset();
                    topdown_view.reset();
                    std::cout << "已重置 EKF / 俯视图轨迹\n";
                }
                else if (key == 'm')
                {
                    // 打印当前帧各观测的 PnP 位姿矩阵（R: 3x3 旋转矩阵，t: 3x1 平移，单位 m）
                    std::cout << "===== 当前帧 PnP 位姿（" << dbg_tvecs.size() << " 个观测）=====\n";
                    for (std::size_t di = 0; di < dbg_tvecs.size(); ++di)
                    {
                        cv::Mat R;
                        cv::Rodrigues(dbg_rvecs[di], R);
                        std::cout << "观测 " << di << "（ID=" << dbg_ids[di] << "）:\nR =\n"
                                  << R << "\nt =\n"
                                  << dbg_tvecs[di].t() << "   (单位: m)\n";
                    }
                }
            }
            t5 = cv::getTickCount();
            last_wait_ms = tick_ms(t4, t5);
        }

        // ===== "不带显示窗口"的吞吐：grab+detect+pnp+draw 耗时（EMA）=====
        // 这段就是去掉 GUI（show/key）后的循环时间；其倒数再受相机帧率封顶，
        // 即"窗口照常显示、但显示不参与计时"时的无显示吞吐。
        const double proc_ms_now = tick_ms(t0, t35);
        if (proc_ms_now > 0.1)
        {
            const double f = 1000.0 / proc_ms_now;
            fps_nodisp_ema = (fps_nodisp_ema <= 0.0) ? f : 0.9 * fps_nodisp_ema + 0.1 * f;
        }
        const float cam_fps_now = cameraResultingFps();
        const double fps_nodisp =
            cam_fps_now > 0.0F ? std::min(fps_nodisp_ema, static_cast<double>(cam_fps_now))
                               : fps_nodisp_ema;

        // ===== 统计（精简输出：每 150 帧一行；err 分布每 300 帧）=====
        ++frame_cnt;
        if (frame_cnt % 150 == 0)
        {
            printf("[%d帧] fps_cam=%.1f fps_nodisp=%.1f | grab=%.1f detect=%.1f "
                   "pnp=%.1f ekf=%.1f draw=%.1f show=%.1f key=%.1f ms | "
                   "Det=%zu Poses=%zu | 拒绝 reproj=%d depth=%d fin=%d pnpFail=%d\n",
                   frame_cnt, cam_fps_now, fps_nodisp,
                   tick_ms(t0, t1), tick_ms(t1, t2), tick_ms(t2, t3), tick_ms(t3, t3b),
                   tick_ms(t3b, t4), last_wait_ms, tick_ms(t4, t5),
                   dets.size(), poses.size(),
                   g_rej_reproj, g_rej_depth, g_rej_finite, g_pnp_fail);
            g_rej_reproj = g_rej_depth = g_rej_finite = g_pnp_fail = 0;
        }
        if (++stat_cnt % 300 == 0 && !g_reproj_errors.empty())
        {
            std::vector<double> tmp = g_reproj_errors;
            std::sort(tmp.begin(), tmp.end());
            auto pct = [&](double p)
            {
                std::size_t i = static_cast<std::size_t>(p * tmp.size());
                if (i >= tmp.size())
                    i = tmp.size() - 1;
                return tmp[i];
            };
            printf("err分布: P50=%.1f  P90=%.1f  P95=%.1f  P99=%.1f  max=%.1f  (n=%zu)\n",
                   pct(0.50), pct(0.90), pct(0.95), pct(0.99), tmp.back(), tmp.size());
            g_reproj_errors.clear();
        }
    }

    if (video_path.empty())
        release_camera();
    cv::destroyAllWindows();
    return 0;
}
