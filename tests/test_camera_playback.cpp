// 离线回放测试：把录制的视频喂进完整链路（检测 -> PnP -> 可选 EKF）。
//
// 用途有两个：
//   1) 没有人/没有相机时评估**视觉识别质量**：检测率、空帧率、编号/大小板分布、
//      距离与重投影误差分布、同一目标连续帧的位置抖动（给 EKF 的 R 定初值）；
//   2) 用**真实序列**做 EKF 关联/锁定/预测的离线评估与调参（之前只有合成真值）。
//
// 用法：
//   ./build/test_camera_playback tmp/v4.avi --red --headless
//   ./build/test_camera_playback tmp/v4.avi --red --ekf --headless --csv build/replay.csv
//   ./build/test_camera_playback tmp/v4.avi --red                  # 带窗口回放
//
// 参数：
//   --red | --blue   敌方颜色（默认 red）
//   --ekf            启用旋转目标 EKF（关联 + 锁定 + 预测）
//   --headless       无窗口（只统计，适合批量跑）
//   --stride N       每 N 帧处理 1 帧（默认 1）
//   --start N --end N  只处理 [start, end) 帧（挑片段用）
//   --max-frames N   最多处理 N 帧
//   --wait MS        窗口模式每帧等待毫秒（默认 0，按视频帧率跑）
//   --csv PATH       逐帧导出 CSV（表头见文件第一行）
//   --bias S         模拟控制器前馈补偿（秒），写入云台快照的 prediction_bias_s
//   --config PATH    相机参数文件（默认 config/camera.yaml，或 RMV_CAMERA_CONFIG）
//
// 按键（窗口模式）：ESC/q 退出 | 空格 暂停/继续 | s 保存当前帧 | m 打印 EKF 状态
//
// 坐标系：回放没有云台角，约定"相机与云台同轴、视频中云台角为零"，于是云台系 =
// 世界系；再用固定的光学系 -> 水平系旋转把坐标摆正：
//   x_世界 = z_光学（前），y_世界 = -x_光学（左），z_世界 = -y_光学（上）。
// 这样 EKF 里"车体绕 z 轴转"才和画面上的旋转对得上。
// 注意：视频若是手持/云台晃动拍的，世界系会跟着晃，EKF 只能用在"相机基本静止"
// 的片段——所以每帧都统计画面运动量（CSV 的 motion 列），用它挑片段。
#include "AutoAimTypes.hpp"
#include "RotatingArmorTracker.hpp"
#include "StudentTasks.hpp"
#include <opencv2/calib3d.hpp>
#include <filesystem>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace
{

    // 光学系 -> 水平系（前 x、左 y、上 z）的固定旋转：
    //   x_w =  z_c , y_w = -x_c , z_w = -y_c
    cv::Mat opticalToWorldRotation()
    {
        cv::Mat rotation = (cv::Mat_<double>(3, 3) << 0.0, 0.0, 1.0,
                            -1.0, 0.0, 0.0,
                            0.0, -1.0, 0.0);
        return rotation;
    }

    // 视频分辨率与标定分辨率不一致时按比例缩放内参（假设画面未被裁剪）。
    void scaleIntrinsics(CameraParameters &camera, double sx, double sy)
    {
        if (!camera.camera_matrix.empty() && camera.camera_matrix.type() == CV_64FC1)
        {
            camera.camera_matrix.at<double>(0, 0) *= sx;
            camera.camera_matrix.at<double>(1, 1) *= sy;
            camera.camera_matrix.at<double>(0, 2) *= sx;
            camera.camera_matrix.at<double>(1, 2) *= sy;
        }
        camera.image_width = static_cast<int>(std::lround(camera.image_width * sx));
        camera.image_height = static_cast<int>(std::lround(camera.image_height * sy));
    }

    double fractionBelow(const std::vector<double> &values, double threshold)
    {
        if (values.empty())
            return 0.0;
        std::size_t count = 0;
        for (double v : values)
        {
            if (v <= threshold)
                ++count;
        }
        return static_cast<double>(count) / static_cast<double>(values.size());
    }

    double percentile(std::vector<double> values, double ratio)
    {
        if (values.empty())
            return 0.0;
        std::sort(values.begin(), values.end());
        const std::size_t index = std::min(values.size() - 1,
                                           static_cast<std::size_t>(ratio * static_cast<double>(values.size())));
        return values[index];
    }

    double mean(const std::vector<double> &values)
    {
        if (values.empty())
            return 0.0;
        double sum = 0.0;
        for (double value : values)
            sum += value;
        return sum / static_cast<double>(values.size());
    }

    struct Args
    {
        std::string video;
        std::string csv_path;
        std::string config_path;
        TeamColor enemy{TeamColor::Red};
        bool ekf{false};
        bool headless{false};
        int stride{1};
        int start{0};
        int end{-1};
        int max_frames{0};
        int wait_ms{0};
        double bias_s{0.0};
        int dump_every{0};
        int only_id{0};
        double radius_prior{0.25};
        double v_sigma{0.05};
        double lock_span{0.18};
        double lock_age{0.50};
        bool radius_given{false};
        bool smooth{false};
        int smooth_window{4};
        bool trace{false};
    bool phase_yaw{false};
    double sigma_pos{0.03};
    double handoff_scale{2.5};
    double phase_sigma{0.12};
    std::string dump_csv;
    };

    bool parseArgs(int argc, char **argv, Args &args)
    {
        if (argc < 2)
            return false;
        args.video = argv[1];
        for (int i = 2; i < argc; ++i)
        {
            const std::string arg = argv[i];
            const auto next = [&](const char *name) -> std::string {
                if (i + 1 >= argc)
                {
                    std::cerr << "参数 " << name << " 缺少取值\n";
                    std::exit(2);
                }
                return argv[++i];
            };
            if (arg == "--red")
                args.enemy = TeamColor::Red;
            else if (arg == "--blue")
                args.enemy = TeamColor::Blue;
            else if (arg == "--ekf")
                args.ekf = true;
            else if (arg == "--headless")
                args.headless = true;
            else if (arg == "--stride")
                args.stride = std::max(1, std::atoi(next("--stride").c_str()));
            else if (arg == "--start")
                args.start = std::max(0, std::atoi(next("--start").c_str()));
            else if (arg == "--end")
                args.end = std::atoi(next("--end").c_str());
            else if (arg == "--max-frames")
                args.max_frames = std::max(0, std::atoi(next("--max-frames").c_str()));
            else if (arg == "--wait")
                args.wait_ms = std::max(0, std::atoi(next("--wait").c_str()));
            else if (arg == "--bias")
                args.bias_s = std::atof(next("--bias").c_str());
            else if (arg == "--csv")
                args.csv_path = next("--csv");
            else if (arg == "--dump-every")
                args.dump_every = std::max(0, std::atoi(next("--dump-every").c_str()));
            else if (arg == "--trace")
                args.trace = true;
            else if (arg == "--sigma-pos")
                args.sigma_pos = std::stod(next("--sigma-pos"));
            else if (arg == "--handoff-scale")
                args.handoff_scale = std::stod(next("--handoff-scale"));
            else if (arg == "--phase-yaw")
                args.phase_yaw = true;
            else if (arg == "--phase-sigma")
                args.phase_sigma = std::stod(next("--phase-sigma"));
            else if (arg == "--dump")
                args.dump_csv = next("--dump");
            else if (arg == "--only-id")
                args.only_id = std::atoi(next("--only-id").c_str());
            else if (arg == "--smooth")
                args.smooth = true;
            else if (arg == "--smooth-window")
                args.smooth_window = std::max(1, std::atoi(next("--smooth-window").c_str()));
            else if (arg == "--radius")
            {
                args.radius_prior = std::atof(next("--radius").c_str());
            }
            else if (arg == "--v-sigma")
            {
                args.v_sigma = std::atof(next("--v-sigma").c_str());
            }
            else if (arg == "--lock-span")
            {
                args.lock_span = std::atof(next("--lock-span").c_str());
            }
            else if (arg == "--lock-age")
            {
                args.lock_age = std::atof(next("--lock-age").c_str());
                args.radius_given = true;
            }
            else if (arg == "--config")
                args.config_path = next("--config");
            else
            {
                std::cerr << "未知参数: " << arg << "\n";
                return false;
            }
        }
        return true;
    }

    void printUsage(const char *program)
    {
        std::cerr << "用法: " << program << " <video> [--red|--blue] [--ekf] [--headless]\n"
                  << "        [--stride N] [--start N] [--end N] [--max-frames N]\n"
                  << "        [--wait MS] [--csv PATH] [--bias S] [--config PATH] [--dump-every N]\n";
    }

    const char *phaseName(rotating_armor::LockPhase phase)
    {
        switch (phase)
        {
        case rotating_armor::LockPhase::Idle:
            return "idle";
        case rotating_armor::LockPhase::Collecting:
            return "collecting";
        case rotating_armor::LockPhase::Tracking:
            return "tracking";
        }
        return "?";
    }

    const char *stateName(TrackingState state)
    {
        switch (state)
        {
        case TrackingState::NoTarget:
            return "NoTarget";
        case TrackingState::Unstable:
            return "Unstable";
        case TrackingState::Stable:
            return "Stable";
        }
        return "?";
    }

} // namespace

int main(int argc, char **argv)
{
    Args args;
    if (!parseArgs(argc, argv, args))
    {
        printUsage(argv[0]);
        return 2;
    }

    // 1. 相机参数：内参来自标定文件；外参换成"光学系 -> 水平系"的固定旋转，
    //    平移置零（回放不关心装车位置，只关心相对运动）。
    CameraParameters camera;
    {
        std::string config_path = args.config_path;
        if (config_path.empty())
        {
            const char *env = std::getenv("RMV_CAMERA_CONFIG");
            config_path = (env != nullptr && env[0] != '\0') ? env : "config/camera.yaml";
        }
        std::string error;
        if (!loadCameraParameters(config_path, camera, error))
        {
            std::cerr << "加载相机参数失败: " << error << "\n";
            return 1;
        }
        camera.rotation_camera_to_gimbal = opticalToWorldRotation();
        camera.translation_camera_to_gimbal = cv::Mat::zeros(3, 1, CV_64F);
    }

    // 2. 打开视频。
    cv::VideoCapture capture(args.video);
    if (!capture.isOpened())
    {
        std::cerr << "无法打开视频: " << args.video << "\n";
        return 1;
    }
    const double fps = capture.get(cv::CAP_PROP_FPS) > 1.0 ? capture.get(cv::CAP_PROP_FPS) : 30.0;
    const int total_frames = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    const int video_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int video_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (camera.image_width > 0 && camera.image_height > 0 &&
        (video_width != camera.image_width || video_height != camera.image_height))
    {
        std::cout << "[回放] 视频分辨率 " << video_width << "x" << video_height
                  << " 与标定分辨率 " << camera.image_width << "x" << camera.image_height
                  << " 不一致，按比例缩放内参\n";
        scaleIntrinsics(camera,
                        static_cast<double>(video_width) / camera.image_width,
                        static_cast<double>(video_height) / camera.image_height);
    }
    std::cout << "[回放] 视频: " << args.video << "  " << video_width << "x" << video_height
              << "  " << fps << " fps  " << total_frames << " 帧  "
              << "敌方颜色: " << (args.enemy == TeamColor::Red ? "red" : "blue") << "\n";
    std::cout << "[回放] EKF: " << (args.ekf ? "启用" : "关闭")
              << "  模式: " << (args.headless ? "无窗口" : "窗口") << "\n";

    const int range_start = std::max(0, args.start);
    const int range_end = (args.end > 0) ? std::min(args.end, total_frames > 0 ? total_frames : args.end)
                                         : (total_frames > 0 ? total_frames : -1);
    if (range_start > 0)
        capture.set(cv::CAP_PROP_POS_FRAMES, range_start);

    // 3. 统计容器。
    struct TargetStats
    {
        long observations{0};
        std::vector<double> distances;
        std::vector<double> reprojections;
        std::vector<double> yaws;
        long big{0};
        long small{0};
        double confidence_sum{0.0};
        double confidence_min{1.0};
        double confidence_max{0.0};
        // 连续帧位置差分（同一 id+size），用于估观测噪声量级。
        bool has_last{false};
        cv::Vec3d last_position{};
        std::vector<double> step_distances;
    };
    std::map<std::pair<int, int>, TargetStats> targets; // (id, size) -> 统计
    std::map<int, long> frame_histogram;                // 每帧检测数直方图
    std::vector<double> frame_motions;
    long frames_processed{0};
    long frames_with_target{0};
    long frames_empty{0};
    double detect_ms_sum{0.0};
    double solve_ms_sum{0.0};
    double motion_sum{0.0};
    long motion_count{0};
    int detection_id_switches{0};
    int last_frame_target_id{0};
    cv::Mat previous_gray_small;

    // EKF 侧统计。
    rotating_armor::TrackerConfig tracker_config;
    tracker_config.position_sigma_m = args.sigma_pos;
    tracker_config.handoff_sigma_scale = args.handoff_scale;
    tracker_config.radius_prior_m = args.radius_prior;
    tracker_config.velocity_sigma_m_s = args.v_sigma;
    tracker_config.lock_min_span_s = args.lock_span;
    tracker_config.lock_max_age_s = args.lock_age;
    tracker_config.observation_smoothing_enabled = args.smooth;
    tracker_config.smoothing_window = args.smooth_window;
    // 显式 --radius 时以它为准（关掉按板型自动选），方便做 A/B。
    if (args.radius_given)
    {
        tracker_config.radius_prior_from_size = false;
    }
    if (args.phase_yaw)
    {
        tracker_config.phase_update_enabled = true;
        tracker_config.phase_update_sigma_rad = args.phase_sigma;
    }
    rotating_armor::RotatingArmorTracker tracker(tracker_config);
    long ekf_first_stable_frame{-1};
    long ekf_stable_frames{0};
    long ekf_accepted{0};
    long ekf_rejected{0};
    int ekf_relock_count{0};
    int ekf_target_switches{0};
    int last_ekf_id{0};
    std::string last_phase{"idle"};
    std::string last_reject;
    rotating_armor::TrackerDiagnostics final_diagnostics{};
    PredictionResult final_prediction{};
    std::vector<double> ekf_predicted_errors; // 仅记录"预测后到下一帧同 id 观测"的差（粗略）

    // 预测精度评估：把"预测时刻 + 预测瞄准点（世界系）"存起来，等视频推进到
    // 那个时刻，再和该帧实测板心比距离。这是"预测准不准"的客观指标——
    // 比 Stable 帧数更接近验收口径（板真的在预测位置上吗）。
    struct AimTrace
    {
        double t_cmd_s{0.0};
        Eigen::Vector3d aim_world{0.0, 0.0, 0.0};
        int target_id{0};
    };
    std::vector<AimTrace> pending_aims;
    std::vector<double> aim_errors_m;

    std::ofstream csv;
    if (!args.csv_path.empty())
    {
        csv.open(args.csv_path);
        if (!csv.is_open())
        {
            std::cerr << "无法写入 CSV: " << args.csv_path << "\n";
            return 1;
        }
        csv << "frame,time_s,id,size,conf,x,y,z,dist_m,yaw_deg,reproj_px,motion,"
               "ekf_phase,ekf_state,ekf_aim,ekf_id,ekf_yaw_deg,ekf_pitch_deg,ekf_dist_m,ekf_fire\n";
    }

    std::ofstream dump_stream;
    if (!args.dump_csv.empty())
    {
        dump_stream.open(args.dump_csv);
        dump_stream << "frame,t,det,target_id,aim_slot,obs_id,x,y,z,best_slot,best_dist,"
                       "second_dist,cx,cy,cyaw,R\n";
    }

    const bool show = !args.headless;
    const std::string window = "camera playback";
    if (show)
        cv::namedWindow(window, cv::WINDOW_NORMAL);

    bool paused = false;
    int frame_index = range_start;
    cv::Mat frame;
    for (; ; )
    {
        if (!paused && !capture.read(frame))
            break;
        if (frame.empty())
            break;
        if (range_end > 0 && frame_index >= range_end)
            break;
        if (args.max_frames > 0 && frames_processed >= args.max_frames)
            break;

        const double time_s = static_cast<double>(frame_index) / fps;

        // 画面运动量：缩到 1/8 灰度后与上一帧的平均绝对差（0~255）。
        double motion = 0.0;
        {
            cv::Mat gray_small;
            cv::cvtColor(frame, gray_small, cv::COLOR_BGR2GRAY);
            cv::resize(gray_small, gray_small, cv::Size(180, 135), 0, 0, cv::INTER_AREA);
            if (!previous_gray_small.empty())
                motion = cv::norm(gray_small, previous_gray_small, cv::NORM_L1) /
                         static_cast<double>(gray_small.total());
            gray_small.copyTo(previous_gray_small);
            frame_motions.push_back(motion);
            motion_sum += motion;
            ++motion_count;
        }

        const auto t0 = std::chrono::steady_clock::now();
        const std::vector<ArmorDetection> detections = armor_detect(frame, args.enemy);
        const auto t1 = std::chrono::steady_clock::now();
        GimbalState gimbal{};
        gimbal.prediction_bias_s = static_cast<float>(args.bias_s);
        std::vector<ArmorPose> poses = armor_solve(detections, camera, gimbal);
        if (args.only_id != 0)
        {
            // 调参用：只保留指定编号的观测，避免两台车的装甲板混进同一个跟踪器。
            poses.erase(std::remove_if(poses.begin(), poses.end(),
                                       [&](const ArmorPose &pose) {
                                           return pose.detection.target_id != args.only_id;
                                       }),
                        poses.end());
        }
        const auto t2 = std::chrono::steady_clock::now();
        const double detect_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double solve_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        detect_ms_sum += detect_ms;
        solve_ms_sum += solve_ms;

        // ---- 帧级统计 ----
        ++frames_processed;
        ++frame_histogram[static_cast<int>(detections.size())];
        if (detections.empty())
            ++frames_empty;
        else
            ++frames_with_target;
        int current_target_id = 0;
        if (!poses.empty())
            current_target_id = poses.front().detection.target_id;
        if (detection_id_switches == 0)
            last_frame_target_id = current_target_id;
        if (current_target_id != 0 && last_frame_target_id != 0 &&
            current_target_id != last_frame_target_id)
            ++detection_id_switches;
        if (current_target_id != 0)
            last_frame_target_id = current_target_id;

        // ---- 目标级统计 ----
        const ArmorPose *nearest = nullptr;
        double nearest_distance = 1e18;
        for (const ArmorPose &pose : poses)
        {
            const cv::Vec3d &p = pose.position_camera_m;
            const double distance = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
            if (distance < nearest_distance)
            {
                nearest_distance = distance;
                nearest = &pose;
            }
            const std::pair<int, int> key{pose.detection.target_id,
                                          static_cast<int>(pose.detection.size)};
            TargetStats &stats = targets[key];
            ++stats.observations;
            stats.distances.push_back(distance);
            stats.reprojections.push_back(pose.reprojection_error);
            stats.yaws.push_back(pose.armor_yaw * 180.0 / 3.141592653589793);
            if (pose.detection.size == ArmorSize::Large)
                ++stats.big;
            else
                ++stats.small;
            stats.confidence_sum += pose.detection.confidence;
            stats.confidence_min = std::min(stats.confidence_min,
                                            static_cast<double>(pose.detection.confidence));
            stats.confidence_max = std::max(stats.confidence_max,
                                            static_cast<double>(pose.detection.confidence));
            if (stats.has_last)
            {
                const cv::Vec3d &a = stats.last_position;
                const cv::Vec3d &b = pose.position_camera_m;
                stats.step_distances.push_back(std::sqrt((a[0] - b[0]) * (a[0] - b[0]) +
                                                         (a[1] - b[1]) * (a[1] - b[1]) +
                                                         (a[2] - b[2]) * (a[2] - b[2])));
            }
            stats.last_position = pose.position_camera_m;
            stats.has_last = true;
        }

        // 模型解释度：观测到的板心到 4 个"模型板位"里最近的那个的距离。
        // 单看某一块板没意义（4 板对称，槽位标签本来就只能确定到 2pi/N），
        // 但"每一块观测都能被模型解释"是状态对不对的硬指标：状态错 2R 时
        // 这个残差就是 0.3 m 量级，状态对时是几厘米。
        std::vector<double> frame_model_residuals;
        if (args.ekf && tracker.initialized())
        {
            const rotating_armor::State &st = tracker.ekf().state();
            for (const ArmorPose &pose : poses)
            {
                if (pose.detection.target_id == 0 ||
                    pose.detection.target_id != tracker.diagnostics().target_id)
                {
                    continue; // 只评估当前跟踪的那辆车
                }
                // 观测是云台系的，模型板位是世界系的：必须先转过去再比，
                // 否则残差里混进了"云台朝向"这一项（v4 里云台几乎朝正前，
                // 只差 7 mm；换到侧向目标就会凭空多出几厘米）。
                const Eigen::Vector3d observed =
                    rotating_armor::gimbalToWorld(gimbal) *
                    Eigen::Vector3d(pose.position_gimbal_m[0], pose.position_gimbal_m[1],
                                    pose.position_gimbal_m[2]);
                double best = std::numeric_limits<double>::infinity();
                double second = std::numeric_limits<double>::infinity();
                int best_slot = -1;
                for (int slot = 0; slot < 4; ++slot)
                {
                    const double d =
                        (observed - rotating_armor::platePosition(st, slot, 4)).norm();
                    if (d < best)
                    {
                        second = best;
                        best = d;
                        best_slot = slot;
                    }
                    else if (d < second)
                    {
                        second = d;
                    }
                }
                if (dump_stream.is_open())
                {
                    dump_stream << frame_index << ',' << time_s << ',' << poses.size() << ','
                                << tracker.diagnostics().target_id << ','
                                << tracker.diagnostics().aim_slot << ',' << pose.detection.target_id
                                << ',' << observed.x() << ',' << observed.y() << ',' << observed.z()
                                << ',' << best_slot << ',' << best << ',' << second << ','
                                << st(0) << ',' << st(2) << ',' << st(6) << ',' << st(8) << '\n';
                }
                if (std::isfinite(best))
                {
                    frame_model_residuals.push_back(best);
                }
            }
        }

        // ---- EKF ----
        rotating_armor::TrackerDiagnostics diagnostics{};
        PredictionResult prediction{};
        if (args.ekf)
        {
            prediction = tracker.step(poses, gimbal, time_s, 0.0);
            diagnostics = tracker.diagnostics();
            final_diagnostics = diagnostics;
            if (args.trace)
            {
                const auto &st = tracker.ekf().state();
                std::printf("[trace] f=%d t=%.3f det=%zu init=%d phase=%d id=%d ok=%llu rej=%d miss=%d "
                            "relock=%d ls=%d omega=%.2f R=%.3f yaw=%.2f nis=%.1f why=%s stab=%s gd=%.2f gn=%.0f\n",
                            frame_index, time_s, poses.size(), static_cast<int>(tracker.initialized()),
                            static_cast<int>(diagnostics.phase), diagnostics.target_id,
                            static_cast<unsigned long long>(diagnostics.accepted_updates),
                            diagnostics.rejected_this_frame, diagnostics.missed_frames,
                            diagnostics.relock_count, diagnostics.lock_samples, st(7), st(8), st(6), diagnostics.last_nis,
                            diagnostics.last_reject.c_str(), diagnostics.stability.c_str(),
                            diagnostics.best_gate_distance_m, diagnostics.best_gate_nis);
                        if (diagnostics.phase == rotating_armor::LockPhase::Tracking)
                        {
                            const auto &st = tracker.ekf().state();
                            std::printf("   v=(%.2f,%.2f) c=(%.2f,%.2f)\n", st(1), st(3),
                                        st(0), st(2));
                        }
            }
            final_prediction = prediction;
            ekf_accepted = static_cast<long>(diagnostics.accepted_updates);
            ekf_rejected += diagnostics.rejected_this_frame;
            if (diagnostics.phase == rotating_armor::LockPhase::Tracking)
            {
                if (ekf_first_stable_frame < 0)
                    ekf_first_stable_frame = frame_index;
                if (prediction.state == TrackingState::Stable)
                    ++ekf_stable_frames;
            }
            if (diagnostics.relock_count > ekf_relock_count)
                ekf_relock_count = diagnostics.relock_count;
            for (double r : frame_model_residuals)
            {
                aim_errors_m.push_back(r);
            }
            if (prediction.state != TrackingState::NoTarget &&
                std::isfinite(prediction.distance_m) && prediction.distance_m > 0.0f)
            {
                const double yaw = prediction.target_yaw;
                const double pitch = prediction.target_pitch;
                const double range = prediction.distance_m;
                AimTrace trace;
                trace.t_cmd_s = time_s + static_cast<double>(args.bias_s) + 0.0;
                trace.aim_world = Eigen::Vector3d(range * std::cos(pitch) * std::cos(yaw),
                                                  range * std::cos(pitch) * std::sin(yaw),
                                                  range * std::sin(pitch));
                trace.target_id = prediction.target_id;
                pending_aims.push_back(trace);
            }
            const int ekf_id = prediction.target_id;
            if (last_ekf_id != 0 && ekf_id != 0 && ekf_id != last_ekf_id)
            {
                ++ekf_target_switches;
                std::cout << "[EKF] 帧 " << frame_index << " 目标 " << last_ekf_id
                          << " -> " << ekf_id << "\n";
            }
            if (ekf_id != 0)
                last_ekf_id = ekf_id;
            const std::string phase = phaseName(diagnostics.phase);
            if (phase != last_phase)
            {
                std::cout << "[EKF] 帧 " << frame_index << " 阶段 " << last_phase << " -> "
                          << phase << "（样本 " << diagnostics.lock_samples
                          << "，半径 " << (diagnostics.radius_from_data ? "估" : "先验")
                          << "，omega " << diagnostics.locked_spin_rad_s << "）\n";
                last_phase = phase;
            }
            if (diagnostics.last_reject != last_reject)
            {
                std::cout << "[EKF] 帧 " << frame_index << " 拒绝原因: "
                          << diagnostics.last_reject << "\n";
                last_reject = diagnostics.last_reject;
            }
        }

        // ---- CSV：一行一个观测（多目标时同一帧多行）----
        if (csv.is_open())
        {
            for (const ArmorPose &pose : poses)
            {
                const cv::Vec3d &p = pose.position_camera_m;
                const double distance = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
                csv << frame_index << ',' << time_s << ',' << pose.detection.target_id << ','
                    << (pose.detection.size == ArmorSize::Large ? "L" : "S") << ','
                    << pose.detection.confidence << ',' << p[0] << ',' << p[1] << ',' << p[2]
                    << ',' << distance << ','
                    << pose.armor_yaw * 180.0 / 3.141592653589793 << ','
                    << pose.reprojection_error << ',' << motion << ',';
                if (args.ekf)
                {
                    csv << phaseName(diagnostics.phase) << ',' << stateName(prediction.state) << ','
                        << diagnostics.aim_slot << ',' << static_cast<int>(prediction.target_id) << ','
                        << prediction.target_yaw * 180.0 / 3.141592653589793 << ','
                        << prediction.target_pitch * 180.0 / 3.141592653589793 << ','
                        << prediction.distance_m << ',' << prediction.fire_allowed << '\n';
                }
                else
                {
                    csv << ",,,,,,,\n";
                }
            }
        }

            // ---- 模型板投影：可视化验收的核心 ----
            // 把 EKF 状态推出的 4 块板（世界系）转回相机系再投影到图像：
            // 模型板压在真实板上 = 状态对；偏出去多少，眼睛直接量。
            if (args.ekf && diagnostics.phase == rotating_armor::LockPhase::Tracking)
            {
                const auto &ekf_state = tracker.ekf().state();
                std::vector<cv::Point3d> model_points;
                for (int slot = 0; slot < 4; ++slot)
                {
                    const Eigen::Vector3d p_world = rotating_armor::platePosition(ekf_state, slot, 4);
                    const Eigen::Vector3d p_gimbal = rotating_armor::toGimbal(gimbal, p_world);
                    const cv::Vec3d p_camera = gimbalToCameraPoint(
                        camera, cv::Vec3d(p_gimbal.x(), p_gimbal.y(), p_gimbal.z()));
                    model_points.emplace_back(p_camera[0], p_camera[1], p_camera[2]);
                }
                std::vector<cv::Point2d> projected;
                cv::projectPoints(model_points, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0),
                                  camera.camera_matrix, cv::noArray(), projected);
                // 只把"跟踪中的那块板"(aim slot) 画实：另外 3 块是模型外推到车体
                // 背面的，本来就看不见，画出来会像"满天飞的点"，容易误判成状态错了。
                for (int slot = 0; slot < 4; ++slot)
                {
                    if (slot == diagnostics.aim_slot)
                    {
                        continue;
                    }
                    cv::drawMarker(frame, projected[slot], cv::Scalar(120, 120, 120),
                                   cv::MARKER_TILTED_CROSS, 8, 1);
                }
                if (diagnostics.aim_slot >= 0 && diagnostics.aim_slot < 4)
                {
                    const cv::Point2d p = projected[diagnostics.aim_slot];
                    cv::circle(frame, p, 9, cv::Scalar(255, 0, 255), 2);
                    cv::putText(frame, "AIM", p + cv::Point2d(12, -12), cv::FONT_HERSHEY_SIMPLEX,
                                0.6, cv::Scalar(255, 0, 255), 2);
                }
                // 车体中心
                const Eigen::Vector3d c_world = ekf_state.head<3>();
                const Eigen::Vector3d c_gimbal = rotating_armor::toGimbal(gimbal, c_world);
                const cv::Vec3d c_camera =
                    gimbalToCameraPoint(camera, cv::Vec3d(c_gimbal.x(), c_gimbal.y(), c_gimbal.z()));
                std::vector<cv::Point2d> c_proj;
                cv::projectPoints(std::vector<cv::Point3d>{cv::Point3d(c_camera[0], c_camera[1],
                                                                      c_camera[2])},
                                  cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), camera.camera_matrix,
                                  cv::noArray(), c_proj);
                cv::drawMarker(frame, c_proj[0], cv::Scalar(255, 0, 255), cv::MARKER_CROSS, 20, 2);
            }
        // ---- 落盘标注帧（给报告/答辩留证据）----
        if (args.dump_every > 0 && frame_index % args.dump_every == 0)
        {
            std::error_code mkdir_error;
            std::filesystem::create_directories("build/replay_marks", mkdir_error);
            cv::Mat marked = frame.clone();
            drawDetections(marked, detections);
            char line[256];
            std::snprintf(line, sizeof(line), "frame %d  t=%.2fs  det=%d  motion=%.1f",
                          frame_index, time_s, static_cast<int>(detections.size()), motion);
            cv::putText(marked, line, cv::Point(12, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                        cv::Scalar(0, 255, 255), 2);
            char path[96];
            std::snprintf(path, sizeof(path), "build/replay_marks/f%06d.png", frame_index);
            cv::imwrite(path, marked);
        }

        // ---- 显示 ----
        if (show)
        {
            drawDetections(frame, detections);
            char line[256];
            std::snprintf(line, sizeof(line), "frame %d  t=%.2fs  det=%d  motion=%.1f  %.1f+%.1fms",
                          frame_index, time_s, static_cast<int>(detections.size()), motion,
                          detect_ms, solve_ms);
            cv::putText(frame, line, cv::Point(12, 34), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 255, 255), 2);
            int y = 70;
            for (const ArmorPose &pose : poses)
            {
                const double distance = std::sqrt(pose.position_camera_m[0] * pose.position_camera_m[0] +
                                                  pose.position_camera_m[1] * pose.position_camera_m[1] +
                                                  pose.position_camera_m[2] * pose.position_camera_m[2]);
                std::snprintf(line, sizeof(line), "id=%d %s  d=%.2fm  yaw=%.1f  reproj=%.2fpx  conf=%.2f",
                              pose.detection.target_id,
                              pose.detection.size == ArmorSize::Large ? "B" : "S", distance,
                              pose.armor_yaw * 180.0 / 3.141592653589793,
                              pose.reprojection_error, pose.detection.confidence);
                cv::putText(frame, line, cv::Point(12, y), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                            cv::Scalar(0, 255, 0), 2);
                y += 30;
            }
            if (args.ekf)
            {
                std::snprintf(line, sizeof(line), "EKF: %s %s aim=%d fire=%d yaw=%.1f pitch=%.1f d=%.2f",
                              phaseName(diagnostics.phase), stateName(prediction.state),
                              prediction.target_id, static_cast<int>(prediction.fire_allowed),
                              prediction.target_yaw * 180.0 / 3.141592653589793,
                              prediction.target_pitch * 180.0 / 3.141592653589793,
                              prediction.distance_m);
                cv::putText(frame, line, cv::Point(12, y + 10), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                            cv::Scalar(255, 200, 0), 2);
            }
            if (frame.cols > 1920)
                cv::resize(frame, frame, cv::Size(), 0.5, 0.5);
            cv::imshow(window, frame);
            const int key = cv::pollKey();
            if (key == 27 || key == 'q')
                break;
            if (key == ' ')
                paused = !paused;
            if (key == 's')
            {
                char path[64];
                std::snprintf(path, sizeof(path), "./tmp/replay_%06d.png", frame_index);
                cv::imwrite(path, frame);
                std::cout << "已保存 " << path << "\n";
            }
            if (key == 'm' && args.ekf)
            {
                const rotating_armor::State &state = tracker.ekf().state();
                std::cout << "EKF state: cx=" << state(0) << " vx=" << state(1)
                          << " cy=" << state(2) << " vy=" << state(3)
                          << " z=" << state(4) << " vz=" << state(5)
                          << " yaw=" << state(6) << " omega=" << state(7)
                          << " R=" << state(8) << "\n";
            }
            if (args.wait_ms > 0)
                cv::waitKey(args.wait_ms);
        }
        else if (frames_processed % 50 == 0)
        {
            std::cout << "[回放] " << frames_processed << " 帧  det=" << detections.size()
                      << "  detect " << detect_ms << "ms  solve " << solve_ms << "ms\n";
        }

        frame_index += args.stride;
        if (args.stride > 1)
            capture.set(cv::CAP_PROP_POS_FRAMES, frame_index);
    }

    // 4. 汇总。
    std::cout << "\n================ 回放汇总 ================\n";
    std::cout << "处理帧数 " << frames_processed << "（有目标 " << frames_with_target
              << "，空帧 " << frames_empty << "）\n";
    std::cout << "耗时: detect " << (frames_processed > 0 ? detect_ms_sum / frames_processed : 0.0)
              << " ms/帧，solve " << (frames_processed > 0 ? solve_ms_sum / frames_processed : 0.0)
              << " ms/帧\n";
    std::cout << "画面运动量 motion: 平均 " << (motion_count > 0 ? motion_sum / motion_count : 0.0)
              << "（越小越接近静止；CSV 的 motion 列可挑片段）\n";

    std::cout << "\n-- 每帧检测数分布 --\n";
    for (const auto &entry : frame_histogram)
        std::cout << "  " << entry.first << " 块: " << entry.second << " 帧 ("
                  << (frames_processed > 0 ? 100.0 * entry.second / frames_processed : 0.0) << "%)\n";

    std::cout << "\n-- 各目标观测统计（id=8 表示掉编号，单独一行）--\n";
    std::cout << "  id size 观测数  big  small  距离(m) min/中位/max   重投影(px) 中位/最大   "
                 "yaw(deg) 中位  置信 中位/最小  帧间位移(m) 中位\n";
    for (const auto &entry : targets)
    {
        const TargetStats &stats = entry.second;
        const double conf_mean = stats.observations > 0
                                     ? stats.confidence_sum / static_cast<double>(stats.observations)
                                     : 0.0;
        std::cout << "  " << entry.first.first << "  " << (entry.first.second == 1 ? "L" : "S")
                  << "  " << stats.observations << "  " << stats.big << "  " << stats.small
                  << "  " << percentile(stats.distances, 0.0) << "/"
                  << percentile(stats.distances, 0.5) << "/"
                  << percentile(stats.distances, 0.999) << "  "
                  << percentile(stats.reprojections, 0.5) << "/"
                  << percentile(stats.reprojections, 0.999) << "  "
                  << percentile(stats.yaws, 0.5) << "  "
                  << conf_mean << "/" << stats.confidence_min << "  "
                  << percentile(stats.step_distances, 0.5) << "\n";
    }

    std::cout << "\n-- armor_solve 拒绝计数 --\n"
              << "  重投影超限 " << g_rej_reproj << "，深度越界 " << g_rej_depth
              << "，解非有限 " << g_rej_finite << "，PnP 失败 " << g_pnp_fail << "\n";
    std::cout << "检测编号跳变（相邻有目标帧 id 不同）次数: " << detection_id_switches << "\n";

    if (args.ekf)
    {
        std::cout << "\n-- EKF 汇总 --\n";
        std::cout << "  首次 Tracking 帧: " << ekf_first_stable_frame
                  << "，Stable 帧数: " << ekf_stable_frames << "/" << frames_processed << "\n";
        std::cout << "  接收更新 " << ekf_accepted << "，被拒 " << ekf_rejected
                  << "，重锁 " << ekf_relock_count << "，目标切换 " << ekf_target_switches << "\n";
        std::cout << "  末态阶段: " << phaseName(final_diagnostics.phase)
                  << "，锁定半径 " << final_diagnostics.locked_radius_m
                  << "（" << (final_diagnostics.radius_from_data ? "数据估计" : "先验") << "）"
                  << "，omega " << final_diagnostics.locked_spin_rad_s << "\n";
        if (args.ekf && !aim_errors_m.empty())
        {
            std::cout << "  预测命中评估: " << aim_errors_m.size() << " 帧，"
                      << "|误差| p25 " << percentile(aim_errors_m, 0.25) << " 中位 "
                      << percentile(aim_errors_m, 0.50) << " p75 "
                      << percentile(aim_errors_m, 0.75) << " m，"
                      << "p90 " << percentile(aim_errors_m, 0.90) << " m，"
                      << "p99 " << percentile(aim_errors_m, 0.99) << " m，"
                      << "≤2cm " << 100.0 * fractionBelow(aim_errors_m, 0.02) << "% ，"
                      << "≤5cm " << 100.0 * fractionBelow(aim_errors_m, 0.05) << "%\n";
        }
        std::cout << "  末帧预测: " << stateName(final_prediction.state)
                  << " id " << static_cast<int>(final_prediction.target_id)
                  << " yaw " << final_prediction.target_yaw * 180.0 / 3.141592653589793
                  << " pitch " << final_prediction.target_pitch * 180.0 / 3.141592653589793
                  << " 距离 " << final_prediction.distance_m
                  << " 发射 " << final_prediction.fire_allowed << "\n";
        const rotating_armor::State &state = tracker.ekf().state();
        std::cout << "  末帧状态: cx=" << state(0) << " vx=" << state(1)
                  << " cy=" << state(2) << " vy=" << state(3)
                  << " z=" << state(4) << " yaw=" << state(6)
                  << " omega=" << state(7) << " R=" << state(8) << "\n";
    }
    if (csv.is_open())
    {
        csv.close();
        std::cout << "\nCSV 已写入 " << args.csv_path << "\n";
    }
    return 0;
}
