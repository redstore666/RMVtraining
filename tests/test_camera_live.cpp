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

namespace
{
    // 工具内自建物点（与 StudentTasks.cpp 的 buildObjectPoints 相同模型），
    // 仅用于绘制装甲板坐标系：重解一次 PnP 拿 rvec/tvec，不影响 armor_solve 流程。
    std::vector<cv::Point3f> makeObjectPoints(ArmorSize size)
    {
        const float half_w = (size == ArmorSize::Small ? 0.135F : 0.230F) / 2.0F;
        const float half_h = 0.057F / 2.0F;
        return {{-half_w, -half_h, 0.0F}, {half_w, -half_h, 0.0F}, {half_w, half_h, 0.0F}, {-half_w, half_h, 0.0F}};
    }

    // 仅用于调试显示：按目标 ID + 装甲板尺寸保存历史法向，避免不同目标共用上一帧状态。
    // 蓝轴端点做轻微像素域平滑；这不会反馈到 armor_solve() 的实际位姿输出。
    struct AxisDisplayHistory
    {
        cv::Vec3d normal{0.0, 0.0, 1.0};
        bool has_normal{false};
        cv::Point2f blue_end{};
        bool has_blue_end{false};
        cv::Point2f image_center{};
        bool has_center{false};
        std::chrono::steady_clock::time_point last_seen{};
        bool has_last_seen{false};
    };

    bool solveAxisPoseForDisplay(const std::vector<cv::Point3f> &object_points,
                                 const std::vector<cv::Point2f> &image_points,
                                 const cv::Mat &camera_matrix,
                                 const cv::Mat &distortion_coefficients,
                                 int target_id, ArmorSize size,
                                 cv::Mat &rvec, cv::Mat &tvec,
                                 cv::Point2f &origin, cv::Point2f &x_end,
                                 cv::Point2f &y_end, cv::Point2f &blue_end)
    {
        using TrackKey = std::pair<int, int>;
        static std::map<TrackKey, AxisDisplayHistory> history_by_track;
        const TrackKey key{target_id, static_cast<int>(size)};
        AxisDisplayHistory &history = history_by_track[key];
        const auto now = std::chrono::steady_clock::now();
        cv::Point2f current_center(0.0F, 0.0F);
        for (const cv::Point2f &point : image_points)
            current_center += point * (1.0F / static_cast<float>(image_points.size()));

        // 目标消失、或同 ID/尺寸突然跳到画面另一处时，丢弃旧姿态历史。
        if ((history.has_last_seen &&
             now - history.last_seen > std::chrono::milliseconds(250)) ||
            (history.has_center && cv::norm(current_center - history.image_center) > 120.0))
        {
            history.has_normal = false;
            history.has_blue_end = false;
        }

        std::vector<cv::Mat> rvecs, tvecs;
        const int count = cv::solvePnPGeneric(object_points, image_points,
                                              camera_matrix, distortion_coefficients,
                                              rvecs, tvecs, false, cv::SOLVEPNP_IPPE);
        if (count <= 0)
            return false;

        const double inf = std::numeric_limits<double>::infinity();
        std::vector<double> errors(static_cast<std::size_t>(count), inf);
        std::vector<cv::Vec3d> normals(static_cast<std::size_t>(count), cv::Vec3d(0, 0, 1));
        int best_error_index = -1;
        for (int i = 0; i < count; ++i)
        {
            if (tvecs[i].at<double>(2) <= 0.0)
                continue; // 忽略落在相机后方的解

            std::vector<cv::Point2f> projected;
            cv::projectPoints(object_points, rvecs[i], tvecs[i], camera_matrix,
                              distortion_coefficients, projected);
            double sum = 0.0;
            for (std::size_t k = 0; k < projected.size(); ++k)
                sum += cv::norm(projected[k] - image_points[k]);
            errors[static_cast<std::size_t>(i)] = sum / std::max<std::size_t>(projected.size(), 1);

            cv::Mat rotation;
            cv::Rodrigues(rvecs[i], rotation);
            normals[static_cast<std::size_t>(i)] = cv::Vec3d(
                rotation.at<double>(0, 2), rotation.at<double>(1, 2), rotation.at<double>(2, 2));
            if (best_error_index < 0 || errors[static_cast<std::size_t>(i)] <
                                            errors[static_cast<std::size_t>(best_error_index)])
                best_error_index = i;
        }
        if (best_error_index < 0)
            return false;

        int pick = best_error_index;
        // 两个平面解的误差接近时，按该目标自己的历史法向选择连续解。
        if (count >= 2 && std::isfinite(errors[0]) && std::isfinite(errors[1]) &&
            std::fabs(errors[0] - errors[1]) <= 3.0)
        {
            if (history.has_normal)
                pick = normals[0].dot(history.normal) >= normals[1].dot(history.normal) ? 0 : 1;
            else
                pick = std::fabs(normals[0][1]) <= std::fabs(normals[1][1]) ? 0 : 1;
            if (!std::isfinite(errors[static_cast<std::size_t>(pick)]))
                pick = best_error_index;
        }

        rvec = rvecs[static_cast<std::size_t>(pick)].clone();
        tvec = tvecs[static_cast<std::size_t>(pick)].clone();

        // 将原点和三条轴端点投影到图像；坐标轴长度 8 cm，与旧版 drawFrameAxes 一致。
        const std::vector<cv::Point3f> axis_points = {
            {0.0F, 0.0F, 0.0F}, {0.08F, 0.0F, 0.0F}, {0.0F, 0.08F, 0.0F}, {0.0F, 0.0F, 0.08F}};
        std::vector<cv::Point2f> projected_axes;
        cv::projectPoints(axis_points, rvec, tvec, camera_matrix,
                          distortion_coefficients, projected_axes);
        if (projected_axes.size() != 4)
            return false;

        origin = projected_axes[0];
        x_end = projected_axes[1];
        y_end = projected_axes[2];
        cv::Point2f current_blue_end = projected_axes[3];

        // 只平滑调试画面上的蓝轴端点，不平滑/修改控制链路中的实际位姿。
        if (history.has_blue_end && history.has_last_seen &&
            now - history.last_seen <= std::chrono::milliseconds(250) &&
            cv::norm(current_blue_end - history.blue_end) < 80.0)
        {
            constexpr float current_weight = 0.35F;
            current_blue_end = history.blue_end * (1.0F - current_weight) +
                               current_blue_end * current_weight;
        }
        blue_end = current_blue_end;
        history.normal = normals[static_cast<std::size_t>(pick)];
        history.has_normal = true;
        history.blue_end = blue_end;
        history.has_blue_end = true;
        history.image_center = current_center;
        history.has_center = true;
        history.last_seen = now;
        history.has_last_seen = true;
        return true;
    }
} // namespace

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
    if (topdown_enabled)
        cv::namedWindow(topdown_window, cv::WINDOW_AUTOSIZE);
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
            cv::Mat topdown_canvas; // 本帧俯视图（30fps 门内才真正 imshow）

            // ===== 画检测框 + 角点标签 =====
            drawDetections(frame, dets);
            for (const auto &det : dets)
            {
                const char *labels[4] = {"TL", "TR", "BR", "BL"};
                const cv::Scalar colors[4] = {
                    cv::Scalar(255, 0, 0),  // 蓝 = TL
                    cv::Scalar(0, 255, 0),  // 绿 = TR
                    cv::Scalar(0, 0, 255),  // 红 = BR
                    cv::Scalar(0, 255, 255) // 黄 = BL
                };
                for (int k = 0; k < 4; ++k)
                {
                    const cv::Point pt(cvRound(det.corners[k].x),
                                       cvRound(det.corners[k].y));
                    cv::circle(frame, pt, 4, colors[k], -1);
                    cv::putText(frame, labels[k],
                                pt + cv::Point(6, -6),
                                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                                colors[k], 1);
                }
            }

            // ===== 每个 pose 画信息 + 装甲板坐标系（x 红 / y 绿 / z 蓝）=====
            dbg_rvecs.clear();
            dbg_tvecs.clear();
            dbg_ids.clear();
            for (std::size_t pi = 0; pi < poses.size(); ++pi)
            {
                const auto &pose = poses[pi];
                const char size_ch = (pose.detection.size == ArmorSize::Small) ? 'S' : 'L';

                const std::vector<cv::Point3f> obj = makeObjectPoints(pose.detection.size);
                const std::vector<cv::Point2f> img(pose.detection.corners.begin(),
                                                   pose.detection.corners.end());
                cv::Mat rvec, tvec;
                cv::Point2f axis_origin, x_end, y_end, blue_end;
                const bool axis_ok = solveAxisPoseForDisplay(
                    obj, img, camera.camera_matrix, camera.distortion_coefficients,
                    pose.detection.target_id, pose.detection.size,
                    rvec, tvec, axis_origin, x_end, y_end, blue_end);

                // 显示面板尽可能使用与当前可视坐标轴相同的解，避免 R/t 和轴来自不同的 IPPE 解。
                cv::Vec3d p = pose.position_camera_m;
                double display_reproj_error = pose.reprojection_error;
                double armor_yaw = pose.armor_yaw;
                cv::Mat R;
                if (axis_ok)
                {
                    p = cv::Vec3d(tvec);
                    cv::Rodrigues(rvec, R);
                    const cv::Vec3d normal(R.at<double>(0, 2), R.at<double>(1, 2),
                                           R.at<double>(2, 2));
                    armor_yaw = std::atan2(normal[0], normal[2]);

                    std::vector<cv::Point2f> projected;
                    cv::projectPoints(obj, rvec, tvec, camera.camera_matrix,
                                      camera.distortion_coefficients, projected);
                    double err_sq = 0.0;
                    for (std::size_t k = 0; k < projected.size(); ++k)
                    {
                        const double e = cv::norm(projected[k] - img[k]);
                        err_sq += e * e;
                    }
                    if (!projected.empty())
                        display_reproj_error = std::sqrt(err_sq / projected.size());

                    // BGR：x 轴红、y 轴绿、z 轴蓝。只有蓝轴端点做调试显示平滑。
                    const auto pixel = [](const cv::Point2f &point)
                    { return cv::Point(cvRound(point.x), cvRound(point.y)); };
                    cv::line(frame, pixel(axis_origin), pixel(x_end), cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
                    cv::line(frame, pixel(axis_origin), pixel(y_end), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                    cv::line(frame, pixel(axis_origin), pixel(blue_end), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);

                    dbg_rvecs.push_back(rvec.clone());
                    dbg_tvecs.push_back(tvec.clone());
                    dbg_ids.push_back(pose.detection.target_id);
                }

                // OpenCV 相机系: x 右、y 下、z 前
                const double dist = cv::norm(p);
                const double pos_yaw = std::atan2(p[0], p[2]) * 180.0 / CV_PI;
                const double pitch = std::atan2(-p[1], std::hypot(p[0], p[2])) * 180.0 / CV_PI;
                armor_yaw *= 180.0 / CV_PI;

                char line1[128], line2[160], line3[160];
                std::snprintf(line1, sizeof(line1),
                              "ID=%d %c  dist=%.2fm  err=%.1fpx",
                              pose.detection.target_id, size_ch, dist, display_reproj_error);
                std::snprintf(line2, sizeof(line2),
                              "pos=(%+.3f, %+.3f, %+.3f)m",
                              p[0], p[1], p[2]);
                std::snprintf(line3, sizeof(line3),
                              "posYaw=%+.1f  pitch=%+.1f  armYaw=%+.1f  [deg, cam frame]",
                              pos_yaw, pitch, armor_yaw);

                const cv::Point base(20, 80 + static_cast<int>(pi) * 150);
                const cv::Scalar color(0, 255, 255); // 黄
                cv::putText(frame, line1, base, cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1);
                cv::putText(frame, line2, base + cv::Point(0, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1);
                cv::putText(frame, line3, base + cv::Point(0, 44), cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1);

                if (axis_ok)
                {
                    cv::putText(frame,
                                cv::format("R= [%+.3f %+.3f %+.3f]", R.at<double>(0, 0),
                                           R.at<double>(0, 1), R.at<double>(0, 2)),
                                base + cv::Point(0, 64), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                    cv::putText(frame,
                                cv::format("    [%+.3f %+.3f %+.3f]", R.at<double>(1, 0),
                                           R.at<double>(1, 1), R.at<double>(1, 2)),
                                base + cv::Point(0, 82), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                    cv::putText(frame,
                                cv::format("    [%+.3f %+.3f %+.3f]", R.at<double>(2, 0),
                                           R.at<double>(2, 1), R.at<double>(2, 2)),
                                base + cv::Point(0, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                    cv::putText(frame,
                                cv::format("t= [%+.3f %+.3f %+.3f] m", tvec.at<double>(0),
                                           tvec.at<double>(1), tvec.at<double>(2)),
                                base + cv::Point(0, 120), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                }
            }

            // ===== EKF 叠加：车心 + 预测瞄准点 + 状态文本 =====
            // 与下面俯视图用的是同一个 prediction / diagnostics，两个窗口不会互相矛盾。
            if (ekf_enabled)
            {
                const auto project_gimbal_point =
                    [&](const cv::Vec3d &p_gimbal, cv::Point2f &pixel) -> bool
                {
                    const cv::Vec3d p_camera = gimbalToCameraPoint(camera, p_gimbal);
                    if (!(p_camera[2] > 1e-3))
                        return false; // 落在相机后方，投影没有意义
                    std::vector<cv::Point2d> projected;
                    cv::projectPoints(
                        std::vector<cv::Point3d>{cv::Point3d(p_camera[0], p_camera[1], p_camera[2])},
                        cv::Vec3d(0.0, 0.0, 0.0), cv::Vec3d(0.0, 0.0, 0.0),
                        camera.camera_matrix, camera.distortion_coefficients, projected);
                    if (projected.size() != 1 || !std::isfinite(projected[0].x) ||
                        !std::isfinite(projected[0].y))
                        return false;
                    pixel = cv::Point2f(static_cast<float>(projected[0].x),
                                        static_cast<float>(projected[0].y));
                    return true;
                };

                // 车体中心：状态排布 [x, vx, y, vy, z, vz, theta, omega, R]，中心取 (0, 2, 4)。
                // z 再补槽位高度偏移均值——state(4) 会被"最近更新的那块板"带得上下跳。
                if (tracker.initialized())
                {
                    const auto s = tracker.ekf().state();
                    const Eigen::Vector3d c_world(s(0), s(2), s(4) + tracker.slotZOffsetMean());
                    const Eigen::Vector3d c_gimbal = rotating_armor::toGimbal(gimbal, c_world);
                    cv::Point2f center_px;
                    if (project_gimbal_point(
                            cv::Vec3d(c_gimbal.x(), c_gimbal.y(), c_gimbal.z()), center_px))
                    {
                        const cv::Point p(cvRound(center_px.x), cvRound(center_px.y));
                        cv::drawMarker(frame, p, cv::Scalar(255, 0, 255), cv::MARKER_CROSS, 26, 2);
                        cv::putText(frame, cv::format("body R=%.2f m", s(8)),
                                    p + cv::Point(16, -16), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                                    cv::Scalar(255, 0, 255), 1);
                    }
                }

                // 预测瞄准点（含提前量与飞行时间）：直接用 EKF 给出的世界系瞄准点，
                // 不再由 yaw/pitch/distance 反算，避免两处公式不一致。
                if (prediction.state != TrackingState::NoTarget)
                {
                    const Eigen::Vector3d aim_gimbal =
                        rotating_armor::toGimbal(gimbal, diagnostics.predicted_aim_world);
                    cv::Point2f aim_px;
                    if (project_gimbal_point(
                            cv::Vec3d(aim_gimbal.x(), aim_gimbal.y(), aim_gimbal.z()), aim_px))
                    {
                        const cv::Point p(cvRound(aim_px.x), cvRound(aim_px.y));
                        cv::drawMarker(frame, p, cv::Scalar(0, 255, 255), cv::MARKER_TILTED_CROSS,
                                       22, 2);
                        cv::putText(frame, cv::format("AIM %.2f m", prediction.distance_m),
                                    p + cv::Point(16, -16), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                                    cv::Scalar(0, 255, 255), 1);
                    }
                }

                // 右上角文本：锁定状态 / 车体参数 / 云台还差多少 / 开火判据
                const char *state_text = prediction.state == TrackingState::Stable ? "Stable"
                                       : prediction.state == TrackingState::Unstable ? "Unstable"
                                                                                     : "NoTarget";
                const int panel_x = std::max(20, frame.cols - 540);
                const int panel_y = 80;
                const cv::Scalar ekf_color =
                    prediction.fire_allowed ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 200, 255);
                const std::string ekf_lines[] = {
                    cv::format("EKF %s  id=%d  R=%.2fm  w=%+.1f rad/s", state_text,
                               static_cast<int>(prediction.target_id), prediction.radius_1_m,
                               prediction.body_yaw_velocity),
                    cv::format("aim slot=%d  d=%.2fm  yaw=%+.1f pitch=%+.1f deg",
                               diagnostics.aim_slot, prediction.distance_m,
                               prediction.target_yaw * 180.0 / CV_PI,
                               prediction.target_pitch * 180.0 / CV_PI),
                    cv::format("NIS=%.1f  upd=%d  miss=%d  lat=%.0fms  obs=%.0fmm",
                               diagnostics.last_nis, diagnostics.updates_this_frame,
                               diagnostics.missed_frames,
                               diagnostics.measured_latency_s * 1000.0,
                               diagnostics.observation_noise_m * 1000.0),
                    // 注意：本机没有 IMU，gimbal.yaw/pitch 恒为 0，所以这里的
                    // fire/miss 是"相对相机光轴"的量，不是实车上的云台脱靶量。
                    cv::format("fire=%s  miss=%.1fcm  budget=%.2fdeg",
                               prediction.fire_allowed ? "YES" : "no",
                               prediction.fire_miss_m * 100.0,
                               prediction.fire_budget_rad * 180.0 / CV_PI),
                    // putText 只画 ASCII，中文注释留在源码里。
                    std::string("(no IMU: fire/miss vs boresight)"),
                };
                for (int li = 0; li < 5; ++li)
                {
                    cv::putText(frame, ekf_lines[li], cv::Point(panel_x, panel_y + li * 22),
                                cv::FONT_HERSHEY_SIMPLEX, 0.55, ekf_color, 1);
                }
            }

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

        // ===== 状态栏：相机固件帧率 + 无显示吞吐（EMA，滞后一帧无碍）=====
        const float cam_fps = cameraResultingFps();
        const double nodisp_show =
            cam_fps > 0.0F ? std::min(fps_nodisp_ema, static_cast<double>(cam_fps))
                           : fps_nodisp_ema;
        const std::string fps_text = cam_fps >= 0.0F
            ? cv::format("fps_cam=%.1f fps_nodisp=%.1f", cam_fps, nodisp_show)
            : cv::format("fps_cam=n/a fps_nodisp=%.1f", nodisp_show);
        cv::putText(frame,
                    cv::format("Det=%zu Poses=%zu save=%d  %s",
                               dets.size(), poses.size(), save_idx, fps_text.c_str()),
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 255), 2);

            // ===== 显示 =====
            cv::Mat display;
            if (frame.cols > 1920)
                cv::resize(frame, display, cv::Size(), 0.5, 0.5);
            else
                display = frame;

            if (dump_every > 0 && frame_cnt % dump_every == 0)
            {
                cv::imwrite(cv::format("./tmp/live_dump_%06d.png", frame_cnt), frame);
            }

            t35 = cv::getTickCount(); // 绘制完成（GUI 开始前的分界）

            // ===== imshow 降频到约 30fps =====
            // 处理全速跑；只有距上次真正显示 ≥ 33ms 才 imshow 一次，
            // 避免 GUI 队列积压拖慢循环。显示分辨率判断保持原样。
            const double show_interval_ticks =
                33.0 / 1000.0 * cv::getTickFrequency();
            if (cv::getTickCount() - last_show_tick >= show_interval_ticks)
            {
                cv::imshow("camera_live", display);
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
