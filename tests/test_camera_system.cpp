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
//   --show-camera  额外开一个窗口显示相机画面 + 检测框 + 坐标轴 + EKF 叠加
//   --no-display   不开任何窗口（只推网页；纯吞吐/后台跑）
//   --dump-every N 每 N 帧把迷你俯视图存到 ./tmp/system_mini_XXXXXX.png
//                  （--show-camera 时同时存相机叠加图 ./tmp/system_cam_XXXXXX.png）
//
// 按键（有窗口时）: ESC/q 退出 | r 重置 EKF
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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace
{

double plateHalfWidth(ArmorSize size)
{
    return size == ArmorSize::Large ? 0.115 : 0.0675;
}

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
    rotating_armor::RotatingArmorTracker tracker(tracker_config);

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
    ArmorSize last_size = ArmorSize::Small;

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

        // 板型（决定迷你俯视图里板画的长度）：取本帧置信度最高的观测。
        {
            float best = -1.0F;
            for (const ArmorPose &pose : poses)
            {
                if (pose.detection.confidence > best)
                {
                    best = pose.detection.confidence;
                    last_size = pose.detection.size;
                }
            }
        }

        // 视频回放用视频自己的时间轴（与 test_camera_playback 一致）：按墙钟走的话，
        // 1440×1080@166fps 的片子在本机只跑到 ~110fps，EKF 看到的转速会偏慢 0.66 倍。
        // 实车（get_pic）仍用 steady_clock。
        const double time_s = video_path.empty()
            ? std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count()
            : video_time_offset_s + static_cast<double>(video_frame_index - 1) / video_fps;
        const double latency_s = tick_ms(t0, t3) / 1000.0;
        const PredictionResult prediction = tracker.step(poses, gimbal, time_s, latency_s);
        const rotating_armor::TrackerDiagnostics &diagnostics = tracker.diagnostics();
        const int64_t t3b = cv::getTickCount();

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
            if (tracker.initialized())
            {
                const rotating_armor::State &state = tracker.ekf().state();
                mini.has_body = true;
                mini.body_centre = Eigen::Vector3d(
                    state(0), state(2), state(4) + tracker.slotZOffsetMean());
                const double radius = state(8);
                const double half_w = plateHalfWidth(last_size);
                for (int slot = 0; slot < kPlateCount; ++slot)
                {
                    const double phase = state(6) + M_PI * 0.5 * slot;
                    mini.plate_centres.push_back(
                        Eigen::Vector3d(state(0) - radius * std::cos(phase),
                                        state(2) - radius * std::sin(phase),
                                        state(4) + tracker.slotZOffsetMean()));
                    mini.plate_yaws.push_back(phase);
                    mini.plate_half_widths.push_back(half_w);
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
                mini.caption = cv::format("d=%.2fm R=%.2fm w=%+.1f", prediction.distance_m, radius,
                                          prediction.body_yaw_velocity);
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
            SYSTEM_LOG("updates", diagnostics.updates_this_frame);
            SYSTEM_LOG("missed", diagnostics.missed_frames);
            SYSTEM_LOG("aim_miss", prediction.fire_miss_m * 1000.0);
            SYSTEM_LOG("fire", prediction.fire_allowed ? 1.0 : 0.0);
            SYSTEM_LOG("latency_ms", diagnostics.measured_latency_s * 1000.0);
            SYSTEM_LOG("state_code", state_code);
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
            std::printf("[%d帧] fps_loop=%.1f | grab=%.1f detect=%.1f pnp=%.1f ekf=%.1f ms | "
                        "Det=%zu Poses=%zu | EKF id=%d R=%.2f\n",
                        frame_cnt, fps_loop_ema, tick_ms(t0, t1), tick_ms(t1, t2),
                        tick_ms(t2, t3), tick_ms(t3, t3b), dets.size(), poses.size(),
                        static_cast<int>(prediction.target_id), prediction.radius_1_m);
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
