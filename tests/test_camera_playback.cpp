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
//   --topdown        额外开一个"EKF 俯视图"窗口（红点车心 + 四块模型板 + 绿点实测板，
//                    风格见 tmp/example.jpg）；跟随车心、自动缩放
//   --topdown-world  俯视图改成固定原点（云台）视图，按 --topdown-ppm 缩放（默认 200）
//   --topdown-ppm P  固定原点视图的像素/米（默认 200）
//   --topdown-out D  把俯视图逐帧写成 D/t%06d.png（无需 --headless，也可不显示）
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
#include "TopDownView.hpp"
#include "StudentTasks.hpp"
#include "WebLogger.hpp"
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
        bool adaptive_noise{true};
        double noise_min{0.004};
        double omega{0.0};
        bool omega_given{false};
        double q_scale{1.0};
    bool web{false};
    int web_port{8080};
    double web_hold_s{0.0};
        bool led_yaw{false};
        bool roi_retry{false};
        int roi_size{240};
        double led_sigma{0.08};
        double score{0.7};
        bool score_given{false};
        bool centered_switch{true};
        double sigma_floor{0.06};
        bool pair_center{false};
        bool aim_obs_horizon{false};
        bool phase_yaw_static{false};
        bool no_aim_extrapolate{false};
        double realign_m{0.0};
        bool phase_anchor{true};
        double aim_obs{0.012};
        double aim_alpha{0.0};
        int aim_fit{0};
        double switch_margin{0.10};
        double lock_span{0.18};
        double lock_age{0.50};
        bool radius_given{false};
        bool refine_corners{false};
        double nis_gate{11.34};
        int max_updates{2};
        bool nis_gate_given{false};
        bool smooth{false};
        int smooth_window{4};
        bool trace{false};
    bool phase_yaw{false};
    double sigma_pos{0.03};
    double handoff_scale{2.5};
    double phase_sigma{0.12};
    std::string dump_csv;
    bool topdown{false};
    bool topdown_world{false};
    double topdown_ppm{200.0};
    std::string topdown_out;
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
            else if (arg == "--refine-corners")
                args.refine_corners = true;
            else if (arg == "--max-updates" && i + 1 < argc)
                args.max_updates = std::atoi(argv[++i]);
            else if (arg == "--nis-gate" && i + 1 < argc)
                args.nis_gate = std::atof(argv[++i]), args.nis_gate_given = true;
            else if (arg == "--smooth-window")
                args.smooth_window = std::max(1, std::atoi(next("--smooth-window").c_str()));
            else if (arg == "--radius")
            {
                args.radius_prior = std::atof(next("--radius").c_str());
                args.radius_given = true;
            }
            else if (arg == "--aim-fit")
            {
                args.aim_fit = std::atoi(next("--aim-fit").c_str());
            }
            else if (arg == "--aim-alpha")
            {
                args.aim_alpha = std::atof(next("--aim-alpha").c_str());
            }
            else if (arg == "--aim-obs")
            {
                args.aim_obs = std::atof(next("--aim-obs").c_str());
            }
            else if (arg == "--no-phase-anchor")
                args.phase_anchor = false;
            else if (arg == "--sigma-floor")
                args.sigma_floor = std::atof(next("--sigma-floor").c_str());
            else if (arg == "--realign")
            {
                args.realign_m = std::atof(next("--realign").c_str());
            }
            else if (arg == "--no-aim-extrapolate")
            {
                args.no_aim_extrapolate = true;
            }
            else if (arg == "--phase-yaw-static")
            {
                args.phase_yaw_static = true;
            }
            else if (arg == "--aim-obs-horizon")
            {
                args.aim_obs_horizon = true;
            }
            else if (arg == "--pair-center")
                args.pair_center = true;
            else if (arg == "--no-centered-switch")
                args.centered_switch = false;
            else if (arg == "--switch-margin")
                args.switch_margin = std::atof(next("--switch-margin").c_str());
            else if (arg == "--roi-retry")
            {
                args.roi_retry = true;
            }
            else if (arg == "--roi-size")
            {
                args.roi_size = std::atoi(next("--roi-size").c_str());
            }
            else if (arg == "--led-yaw")
            {
                args.led_yaw = true;
            }
            else if (arg == "--led-sigma")
            {
                args.led_sigma = std::atof(next("--led-sigma").c_str());
            }
            else if (arg == "--q-scale")
            {
                args.q_scale = std::atof(next("--q-scale").c_str());
            }
            else if (arg == "--web")
            {
                args.web = true;
            }
            else if (arg == "--web-port")
            {
                args.web_port = std::atoi(next("--web-port").c_str());
            }
            else if (arg == "--web-hold")
            {
                args.web_hold_s = std::atof(next("--web-hold").c_str());
            }
            else if (arg == "--score")
            {
                args.score = std::atof(next("--score").c_str());
                args.score_given = true;
            }
            else if (arg == "--omega")
            {
                args.omega = std::atof(next("--omega").c_str());
                args.omega_given = true;
            }
            else if (arg == "--noise-min")
            {
                args.noise_min = std::atof(next("--noise-min").c_str());
            }
            else if (arg == "--no-adaptive-noise")
            {
                args.adaptive_noise = false;
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
            }
            else if (arg == "--topdown")
                args.topdown = true;
            else if (arg == "--topdown-world")
            {
                args.topdown = true;
                args.topdown_world = true;
            }
            else if (arg == "--topdown-ppm")
            {
                args.topdown_ppm = std::atof(next("--topdown-ppm").c_str());
                args.topdown = true;
            }
            else if (arg == "--topdown-out")
            {
                args.topdown_out = next("--topdown-out");
                args.topdown = true;
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
                  << "        [--wait MS] [--csv PATH] [--bias S] [--config PATH] [--dump-every N]\n"
                  << "        [--topdown] [--topdown-world] [--topdown-ppm P] [--topdown-out DIR]\n";
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

    if (args.web)
    {
        START_WEB_SERVER(args.web_port);
        std::cout << "[web] 曲线页面 http://127.0.0.1:" << args.web_port
                  << "/  （回放结束后保持 " << args.web_hold_s << " s）\n";
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
    if (args.score_given)
    {
        setArmorScoreThreshold(static_cast<float>(args.score));
    }
    tracker_config.velocity_sigma_m_s = args.v_sigma;
    tracker_config.adaptive_observation_noise = args.adaptive_noise;
    tracker_config.adaptive_noise_min_sigma_m = args.noise_min;
    tracker_config.process_accel_scale = args.q_scale;
    if (args.omega_given)
    {
        tracker_config.forced_omega_rad_s = args.omega;
    }
    tracker_config.association_sigma_floor_m = args.sigma_floor;
    tracker_config.pair_center_update_enabled = args.pair_center;
    tracker_config.prefer_centered_target = args.centered_switch;
    tracker_config.target_switch_margin_rad = args.switch_margin;
    tracker_config.max_updates_per_frame = args.max_updates;
    if (args.nis_gate_given)
        tracker_config.nis_gate = args.nis_gate;
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
        tracker_config.phase_anti_anchor = args.phase_anchor;
        tracker_config.phase_update_sigma_rad = args.phase_sigma;
    }
    tracker_config.aim_observation_hold_s = args.aim_obs;
    tracker_config.aim_smoothing_alpha = args.aim_alpha;
    tracker_config.aim_fit_samples = args.aim_fit;
    tracker_config.aim_observation_any_horizon = args.aim_obs_horizon;
    tracker_config.phase_update_when_static = args.phase_yaw_static;
    tracker_config.aim_observation_extrapolate = !args.no_aim_extrapolate;
    tracker_config.single_plate_realign_m = args.realign_m;
    rotating_armor::RotatingArmorTracker tracker(tracker_config);
    long ekf_first_stable_frame{-1};
    long ekf_stable_frames{0};
    long ekf_accepted{0};
    long refine_count{0};
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
    // 真正的"预测准不准"：aim 在 t_cmd_s 时刻打到哪，就和那一刻实测板心比。
    // （之前 aim_errors_m 装的是"模型解释度"，不是预测误差，两者被混在一起了。）
    std::vector<double> prediction_errors_m;
    std::vector<double> aim_slot_errors_m;

    // 瞄准残差按"跟踪状态"分列：Idle / Collecting / Tracking。用户验收时最关心的
    // 是"没锁上时的偏离"和"锁上后的偏离"是两个量，混在一起看不出改动的效果。
    std::array<std::vector<double>, 3> aim_slot_by_phase;
    // 预测质量单独看：没有观测的帧只能靠 EKF 外推（"瞄准偏离"主要发生在这里）。
    std::array<std::vector<double>, 2> aim_slot_by_measurement; // 0=有观测 1=无观测
    std::array<long, 2> aim_frames_by_measurement{{0, 0}};
    std::array<long, 3> frames_by_phase{{0, 0, 0}};
    std::array<long, 3> aim_frames_by_phase{{0, 0, 0}};
    // 按 |omega| 分桶（<1 / 1-5 / 5-15 / >=15 rad/s）：高速自转单独看，
    // 否则被大量静止帧的中位数掩盖。
    std::array<std::vector<double>, 4> aim_slot_by_omega;
    std::array<long, 4> aim_frames_by_omega{{0, 0, 0, 0}};

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
                       "second_dist,cx,cy,cz,cyaw,R,oyaw,aimx,aimy,aimz,u,v,stab,upd,missed\n";
    }

    const bool show = !args.headless;
    const std::string window = "camera playback";
    if (show)
        cv::namedWindow(window, cv::WINDOW_NORMAL);

    // ---- 俯视图（Top-Down）：只是"把 EKF 结果画出来"，不参与任何估计逻辑 ----
    const bool topdown_enabled = args.topdown && args.ekf;
    if (args.topdown && !args.ekf)
    {
        std::cerr << "[俯视图] 需要同时给 --ekf，已忽略 --topdown\n";
    }
    rotating_armor::TopDownOptions topdown_options;
    topdown_options.follow_target = !args.topdown_world;
    topdown_options.pixels_per_meter = args.topdown_ppm;
    rotating_armor::TopDownView topdown_view(topdown_options);
    const std::string topdown_window = "EKF top-down";
    if (topdown_enabled && show)
        cv::namedWindow(topdown_window, cv::WINDOW_AUTOSIZE);
    if (topdown_enabled && !args.topdown_out.empty())
    {
        std::error_code mkdir_error;
        std::filesystem::create_directories(args.topdown_out, mkdir_error);
        if (mkdir_error)
        {
            std::cerr << "无法创建俯视图目录 " << args.topdown_out << ": "
                      << mkdir_error.message() << "\n";
            return 1;
        }
    }

    int led_yaw_updates = 0;
    int roi_recovered = 0;
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
        std::vector<ArmorDetection> detections = armor_detect(frame, args.enemy);

        const auto t1 = std::chrono::steady_clock::now();
        GimbalState gimbal{};
        gimbal.prediction_bias_s = static_cast<float>(args.bias_s);
        // ---- ROI 跟踪式复检 ----
        // 网络整帧没出框时（小板侧视/贴画面边缘），用上一帧的预测板位裁一小块、
        // 放大后再过一遍同一个网络。复检到的角点只当"普通检测"用，后面走同一条
        // armor_solve -> WorldObservation -> tracker.step 流水线。
        if (args.roi_retry && tracker.initialized() && tracker.diagnostics().phase == rotating_armor::LockPhase::Tracking &&
            tracker.diagnostics().target_id != 0)
        {
            bool already = false;
            for (const ArmorDetection &det : detections)
            {
                if (det.target_id == tracker.diagnostics().target_id)
                {
                    already = true;
                    break;
                }
            }
            if (!already && camera.valid() && !camera.camera_matrix.empty())
            {
                const auto &st = tracker.ekf().state();
                const int slot = tracker.diagnostics().aim_slot >= 0 ? tracker.diagnostics().aim_slot : 0;
                const Eigen::Vector3d p_world = rotating_armor::platePosition(st, slot, 4);
                const Eigen::Vector3d p_gimbal = rotating_armor::toGimbal(gimbal, p_world);
                const cv::Vec3d p_camera = gimbalToCameraPoint(camera, cv::Vec3d(p_gimbal.x(), p_gimbal.y(), p_gimbal.z()));
                if (p_camera[2] > 0.4)
                {
                    std::vector<cv::Point3d> one{cv::Point3d(p_camera[0], p_camera[1], p_camera[2])};
                    std::vector<cv::Point2d> px;
                    cv::projectPoints(one, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0),
                                      camera.camera_matrix, cv::noArray(), px);
                    const int half = std::max(40, args.roi_size / 2);
                    int x0 = static_cast<int>(px[0].x) - half;
                    int y0 = static_cast<int>(px[0].y) - half;
                    x0 = std::max(0, std::min(x0, frame.cols - 2 * half));
                    y0 = std::max(0, std::min(y0, frame.rows - 2 * half));
                    if (x0 + 2 * half <= frame.cols && y0 + 2 * half <= frame.rows)
                    {
                        const cv::Rect roi_rect(x0, y0, 2 * half, 2 * half);
                        cv::Mat crop = frame(roi_rect).clone();
                        cv::Mat upscaled;
                        cv::resize(crop, upscaled, cv::Size(), 2.0, 2.0, cv::INTER_LINEAR);
                        const std::vector<ArmorDetection> roi_detections = armor_detect(upscaled, args.enemy);
                        if (std::getenv("RMV_ROI_DBG") != nullptr)
                        {
                            std::fprintf(stderr, "[roi] f=%d rect=%d,%d px=%.0f,%.0f dets=%zu\n",
                                         frame_index, x0, y0, px[0].x, px[0].y, roi_detections.size());
                            for (const ArmorDetection &d : roi_detections)
                            {
                                std::fprintf(stderr, "      id=%d conf=%.2f\n", d.target_id, d.confidence);
                            }
                        }
                        for (ArmorDetection det : roi_detections)
                        {
                            if (det.target_id != tracker.diagnostics().target_id)
                            {
                                continue;
                            }
                            for (cv::Point2f &corner : det.corners)
                            {
                                corner.x = corner.x * 0.5F + static_cast<float>(x0);
                                corner.y = corner.y * 0.5F + static_cast<float>(y0);
                            }
                            detections.push_back(det);
                            ++roi_recovered;
                        }
                    }
                }
            }
        }
        if (args.refine_corners && !detections.empty())
        {
            cv::Mat gray;
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            refine_count += refineArmorCorners(gray, detections);
        }
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

        // ---- EKF ----
        rotating_armor::TrackerDiagnostics diagnostics{};
        PredictionResult prediction{};
        if (args.ekf)
        {
            prediction = tracker.step(poses, gimbal, time_s, 0.0);
            diagnostics = tracker.diagnostics();
            final_diagnostics = diagnostics;
            if (args.web)
            {
                const auto &st = tracker.ekf().state();
                WEB_LOG("frame", static_cast<double>(frame_index));
                WEB_LOG("phase", static_cast<double>(static_cast<int>(diagnostics.phase)));
                WEB_LOG("tracking", diagnostics.phase == rotating_armor::LockPhase::Tracking ? 1.0 : 0.0);
                WEB_LOG("omega", st(7));
                WEB_LOG("radius", st(8));
                WEB_LOG("nis", diagnostics.last_nis);
                WEB_LOG("updates", static_cast<double>(diagnostics.updates_this_frame));
                WEB_LOG("missed", static_cast<double>(diagnostics.missed_frames));
                WEB_LOG("cx", st(0));
                WEB_LOG("cy", st(2));
            }
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
        // 模型解释度：观测到的板心到 4 个"模型板位"里最近的那个的距离。
        // 单看某一块板没意义（4 板对称，槽位标签本来就只能确定到 2pi/N），
        // 但"每一块观测都能被模型解释"是状态对不对的硬指标：状态错 2R 时
        // 这个残差就是 0.3 m 量级，状态对时是几厘米。
        std::vector<double> frame_model_residuals;
        std::vector<double> frame_aim_candidates;
        std::vector<double> frame_aim_residuals;
        if (args.ekf && tracker.initialized())
        {
            const rotating_armor::State &st = tracker.ekf().state();
            // 先把本帧所有观测转到世界系，供"到期 aim 对账"使用。
            std::vector<std::pair<int, Eigen::Vector3d>> frame_obs_world;
            for (const ArmorPose &pose : poses)
            {
                if (pose.detection.target_id == 0)
                {
                    continue;
                }
                frame_obs_world.emplace_back(
                    pose.detection.target_id,
                    rotating_armor::gimbalToWorld(gimbal) *
                        Eigen::Vector3d(pose.position_gimbal_m[0], pose.position_gimbal_m[1],
                                        pose.position_gimbal_m[2]));
            }
            for (auto it = pending_aims.begin(); it != pending_aims.end();)
            {
                if (it->t_cmd_s > time_s)
                {
                    ++it;
                    continue;
                }
                double best = std::numeric_limits<double>::infinity();
                for (const auto &obs : frame_obs_world)
                {
                    if (it->target_id != 0 && obs.first != it->target_id)
                    {
                        continue;
                    }
                    best = std::min(best, (obs.second - it->aim_world).norm());
                }
                if (std::isfinite(best))
                {
                    prediction_errors_m.push_back(best);
                }
                it = pending_aims.erase(it);
            }
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
                                << st(0) << ',' << st(2) << ',' << st(4) << ','
                                << st(6) << ',' << st(8) << ','
                                << rotating_armor::wrapToPi(static_cast<double>(pose.armor_yaw) +
                                                            static_cast<double>(gimbal.yaw))
                                << ',' << tracker.diagnostics().predicted_aim_world.x()
                                << ',' << tracker.diagnostics().predicted_aim_world.y()
                                << ',' << tracker.diagnostics().predicted_aim_world.z();
                                {
                                    // 观测板心投影回像素：用来和 LED 灯条质心对比精度
                                    // （世界 -> 云台 -> 相机 -> 像素）。
                                    const Eigen::Vector3d p_gimbal = rotating_armor::toGimbal(gimbal, observed);
                                    const cv::Vec3d p_camera = gimbalToCameraPoint(
                                        camera, cv::Vec3d(p_gimbal.x(), p_gimbal.y(), p_gimbal.z()));
                                    std::vector<cv::Point3d> one{cv::Point3d(p_camera[0], p_camera[1], p_camera[2])};
                                    std::vector<cv::Point2d> px;
                                    cv::projectPoints(one, cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0),
                                                      camera.camera_matrix, cv::noArray(), px);
                                    dump_stream << ',' << px[0].x << ',' << px[0].y;
                                }
                                dump_stream << ',' << tracker.diagnostics().stability
                                            << ',' << tracker.diagnostics().updates_this_frame
                                            << ',' << tracker.diagnostics().missed_frames
                                            << '\n';
                }
                if (std::isfinite(best))
                {
                    frame_model_residuals.push_back(best);
                }
                // 只对"当前瞄准的那块板"算残差。min-over-4 对相位转 90 度完全
                // 不敏感（四块板转到另一块板的位置上，集合不变），所以它看不出
                // "瞄准点跑到别的板上去了"；要验收瞄准，必须单独看 aim 板。
                // 用"实际发布出去的瞄准点"（diagnostics().predicted_aim_world），
                // 而不是再用 EKF 状态重算一遍：后者会把 buildResult 里的
                // 瞄准点后处理（观测直用 / 可见性窗口）全部绕过。
                // 同帧可能有两块板同时可见（实拍 v4 自转车 209/742 帧）：瞄准点只
                // 可能落在其中一块上，所以一帧只取"离瞄准点最近的那个观测"作为
                // 该帧的瞄准残差；否则另一块板（离 0.35 m）会把中位数抬到无意义。
                const Eigen::Vector3d &aim_world = tracker.diagnostics().predicted_aim_world;
                if (aim_world.allFinite() && aim_world.squaredNorm() > 1.0e-6)
                {
                    frame_aim_candidates.push_back((observed - aim_world).norm());
                }
            }
            if (!frame_aim_candidates.empty())
            {
                const double best_aim =
                    *std::min_element(frame_aim_candidates.begin(), frame_aim_candidates.end());
                frame_aim_residuals.push_back(best_aim);
                if (args.web)
                {
                    WEB_LOG("aim_err_mm", best_aim * 1000.0);
                }
                const std::size_t measured = tracker.diagnostics().updates_this_frame > 0 ? 0 : 1;
                aim_slot_by_measurement[measured].push_back(best_aim);
                ++aim_frames_by_measurement[measured];
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
            for (double r : frame_aim_residuals)
            {
                aim_slot_errors_m.push_back(r);
            }
            {
                const int phase_index = std::clamp(static_cast<int>(diagnostics.phase), 0, 2);
                ++frames_by_phase[static_cast<std::size_t>(phase_index)];
                for (double r : frame_aim_residuals)
                {
                    aim_slot_by_phase[static_cast<std::size_t>(phase_index)].push_back(r);
                    ++aim_frames_by_phase[static_cast<std::size_t>(phase_index)];
                }
                const double omega_abs = std::fabs(tracker.ekf().state()(7));
                const int omega_bucket = omega_abs < 1.0 ? 0 : (omega_abs < 5.0 ? 1 : (omega_abs < 15.0 ? 2 : 3));
                for (double r : frame_aim_residuals)
                {
                    aim_slot_by_omega[static_cast<std::size_t>(omega_bucket)].push_back(r);
                    ++aim_frames_by_omega[static_cast<std::size_t>(omega_bucket)];
                }
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
                          << " " << diagnostics.locked_radius_m
                          << "，omega " << diagnostics.locked_spin_rad_s
                          << " 误差 " << diagnostics.lock_rate_error_rad_s
                          << "，来源 " << (diagnostics.lock_spin_from_positions ? "位置圆拟合" : "法向序列")
                          << "，圆R " << diagnostics.lock_circle_radius_m
                          << " 残差 " << diagnostics.lock_circle_residual_m
                          << " 有效步 " << diagnostics.lock_step_ratio << "）\n";
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
                // LED 灯条独立相位量测：两灯条连线 = 板面内水平轴，与竖直轴叉乘得法向。
                if (args.led_yaw && diagnostics.aim_slot >= 0 && diagnostics.aim_slot < 4)
                {
                    const cv::Point2d aim_px = projected[diagnostics.aim_slot];
                    const double depth = model_points[diagnostics.aim_slot].z;
                    if (aim_px.x > 0.0 && aim_px.y > 0.0 && depth > 0.3 && depth < 20.0 &&
                        !camera.camera_matrix.empty())
                    {
                        const std::vector<LedArmorMeasurement> leds = detect_led_armors(frame, args.enemy);
                        const LedArmorMeasurement *best_led = nullptr;
                        double best_px = 70.0;
                        for (const LedArmorMeasurement &led : leds)
                        {
                            const double d = cv::norm(led.center - cv::Point2f(static_cast<float>(aim_px.x),
                                                                              static_cast<float>(aim_px.y)));
                            if (d < best_px)
                            {
                                best_px = d;
                                best_led = &led;
                            }
                        }
                        if (best_led != nullptr)
                        {
                            const cv::Mat k_inv = camera.camera_matrix.inv();
                            auto back_project = [&](const cv::Point2f &px) {
                                cv::Mat ray = k_inv * cv::Mat(cv::Vec3d(px.x, px.y, 1.0));
                                const double scale = depth / ray.at<double>(2);
                                const cv::Vec3d p_camera(ray.at<double>(0) * scale,
                                                         ray.at<double>(1) * scale, depth);
                                // 相机 -> 云台：p_gimbal = R * p_camera + t
                                cv::Vec3d p_gimbal = p_camera;
                                if (!camera.translation_camera_to_gimbal.empty())
                                {
                                    for (int i = 0; i < 3; ++i)
                                    {
                                        p_gimbal[i] += camera.translation_camera_to_gimbal.at<double>(i);
                                    }
                                }
                                const cv::Mat p_g = camera.rotation_camera_to_gimbal * cv::Mat(p_gimbal);
                                return rotating_armor::toWorld(
                                    gimbal, Eigen::Vector3d(p_g.at<double>(0), p_g.at<double>(1),
                                                            p_g.at<double>(2)));
                            };
                            const Eigen::Vector3d bar_l = back_project(best_led->left_bar);
                            const Eigen::Vector3d bar_r = back_project(best_led->right_bar);
                            Eigen::Vector3d baseline = bar_r - bar_l;
                            baseline.z() = 0.0;
                            if (baseline.norm() > 1e-6)
                            {
                                baseline.normalize();
                                Eigen::Vector3d normal =
                                    baseline.cross(Eigen::Vector3d(0.0, 0.0, 1.0));
                                const Eigen::Vector3d plate_center =
                                    rotating_armor::platePosition(tracker.ekf().state(),
                                                                  diagnostics.aim_slot, 4);
                                const Eigen::Vector3d camera_world =
                                    rotating_armor::toWorld(gimbal, Eigen::Vector3d::Zero());
                                if (normal.dot(camera_world - plate_center) < 0.0)
                                {
                                    normal = -normal;
                                }
                                const double led_yaw_world = std::atan2(normal.y(), normal.x());
                                if (tracker.applyPhaseMeasurement(led_yaw_world, args.led_sigma))
                                {
                                    ++led_yaw_updates;
                                }
                            }
                        }
                    }
                }

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
                    // 瞄准点用颜色区分可信度：
                    //   绿实心 = 锁定且本帧刚用真实观测更新过（稳）；
                    //   橙空心 = 锁定但本帧没有观测，靠模型外推（预测）；
                    //   灰     = 未锁定/刚起步，只有先验，别当真。
                    const bool tracking = diagnostics.phase == rotating_armor::LockPhase::Tracking;
                    const bool measured = tracking && diagnostics.updates_this_frame > 0;
                    cv::Scalar color(150, 150, 150);
                    const char *label = "PRED";
                    if (measured)
                    {
                        color = cv::Scalar(0, 255, 0);
                        label = "AIM";
                    }
                    else if (tracking)
                    {
                        color = cv::Scalar(0, 170, 255);
                        label = "AIM-P";
                    }
                    const cv::Point2d p = projected[diagnostics.aim_slot];
                    // 提前量可视化：白色细十字 = 模型此刻的板心（不前推），
                    // 连线指向 AIM 圈 = 外推到开火时刻的瞄准点。高速自转时
                    // 两者本来就会差出 ω·(bias+latency)·R，不是画错。
                    const Eigen::Vector3d plate_now =
                        rotating_armor::platePosition(ekf_state, diagnostics.aim_slot, 4);
                    const Eigen::Vector3d g_now = rotating_armor::toGimbal(gimbal, plate_now);
                    const cv::Vec3d c_now = gimbalToCameraPoint(
                        camera, cv::Vec3d(g_now.x(), g_now.y(), g_now.z()));
                    std::vector<cv::Point2d> now_proj;
                    cv::projectPoints(std::vector<cv::Point3d>{cv::Point3d(
                                          c_now[0], c_now[1], c_now[2])},
                                      cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0),
                                      camera.camera_matrix, cv::noArray(), now_proj);
                    const cv::Point2d q = now_proj[0];
                    const double lead_px = cv::norm(p - q);
                    if (lead_px > 1.5)
                    {
                        cv::drawMarker(frame, q, cv::Scalar(255, 255, 255),
                                       cv::MARKER_TILTED_CROSS, 10, 1);
                        cv::line(frame, q, p, cv::Scalar(200, 200, 200), 1);
                    }
                    cv::circle(frame, p, 10, color, measured ? -1 : 2);
                    char aim_text[64];
                    std::snprintf(aim_text, sizeof(aim_text), "%s lead %.0fpx", label, lead_px);
                    cv::putText(frame, aim_text, p + cv::Point2d(13, -13),
                                cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2);
                }
                // 车体中心
                // 状态排布是 [x, vx, y, vy, z, vz, theta, omega, R]，中心是 (0,2,4)。
                // 之前取 head<3>() 等于拿 (x, vx, y) 当中心，十字会画到图像下方去。
                const Eigen::Vector3d c_world(ekf_state(0), ekf_state(2), ekf_state(4));
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
        // ---- 俯视图：把 EKF 状态/诊断 + 本帧实测板画成一张上帝视角图 ----
        if (topdown_enabled)
        {
            rotating_armor::TopDownFrame topdown_frame;
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
                observation.position = gimbal_to_world *
                                       Eigen::Vector3d(pose.position_gimbal_m[0],
                                                       pose.position_gimbal_m[1],
                                                       pose.position_gimbal_m[2]);
                observation.normal_yaw =
                    rotating_armor::wrapToPi(pose.armor_yaw + gimbal.yaw);
                observation.target_id = pose.detection.target_id;
                topdown_frame.observations.push_back(observation);
            }
            const cv::Mat canvas = topdown_view.render(topdown_frame).clone();
            if (!args.topdown_out.empty())
            {
                char path[512];
                std::snprintf(path, sizeof(path), "%s/t%06d.png", args.topdown_out.c_str(),
                              frame_index);
                cv::imwrite(path, canvas);
            }
            if (show)
                cv::imshow(topdown_window, canvas);
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
    if (topdown_enabled && !args.topdown_out.empty())
        std::cout << "俯视图已逐帧写入 " << args.topdown_out << "/t%06d.png\n";
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
        if (args.ekf && !aim_slot_errors_m.empty())
        {
            if (!prediction_errors_m.empty())
            {
                std::cout << "  预测瞄准误差(到期对账): " << prediction_errors_m.size()
                          << " 帧，中位 " << percentile(prediction_errors_m, 0.50) << " m，p90 "
                          << percentile(prediction_errors_m, 0.90) << " m，≤5cm "
                          << 100.0 * fractionBelow(prediction_errors_m, 0.05) << "%\n";
            }
            std::cout << "  瞄准板残差: " << aim_slot_errors_m.size() << " 帧，中位 "
                      << percentile(aim_slot_errors_m, 0.50) << " m，p90 "
                      << percentile(aim_slot_errors_m, 0.90) << " m，≤2cm "
                      << 100.0 * fractionBelow(aim_slot_errors_m, 0.02) << "% ，≤5cm "
                      << 100.0 * fractionBelow(aim_slot_errors_m, 0.05) << "%\n";
            const char *phase_names[3] = {"Idle(未追踪)", "Collecting(攒样本)", "Tracking(已锁定)"};
            std::cout << "  瞄准板残差分状态:\n";
            for (std::size_t i = 0; i < 3; ++i)
            {
                if (aim_slot_by_phase[i].empty())
                {
                    std::cout << "    " << phase_names[i] << ": 帧数 " << frames_by_phase[i]
                              << "，无瞄准点\n";
                    continue;
                }
                std::cout << "    " << phase_names[i] << ": 帧数 " << frames_by_phase[i]
                          << "，瞄准样本 " << aim_slot_by_phase[i].size() << "，中位 "
                          << percentile(aim_slot_by_phase[i], 0.50) << " m，p90 "
                          << percentile(aim_slot_by_phase[i], 0.90) << " m，≤5cm "
                          << 100.0 * fractionBelow(aim_slot_by_phase[i], 0.05) << "%\n";
            }
            {
                static const char *const names[2] = {"本帧有观测", "本帧无观测(纯预测)"};
                std::cout << "  瞄准板残差分测量状态:\n";
                for (std::size_t i = 0; i < 2; ++i)
                {
                    if (aim_slot_by_measurement[i].empty())
                    {
                        std::cout << "    " << names[i] << ": 帧数 "
                                  << aim_frames_by_measurement[i] << "，无瞄准点\n";
                        continue;
                    }
                    std::cout << "    " << names[i] << ": 帧数 " << aim_frames_by_measurement[i]
                              << "，中位 " << percentile(aim_slot_by_measurement[i], 0.50)
                              << " m，p90 " << percentile(aim_slot_by_measurement[i], 0.90)
                              << " m，≤5cm "
                              << 100.0 * fractionBelow(aim_slot_by_measurement[i], 0.05) << "%\n";
                }
            }
            const char *omega_names[4] = {"|w|<1", "|w|1-5", "|w|5-15", "|w|>=15"};
            std::cout << "  瞄准板残差按角速度:\n";
            for (std::size_t i = 0; i < 4; ++i)
            {
                if (aim_slot_by_omega[i].empty())
                {
                    std::cout << "    " << omega_names[i] << ": 样本 0\n";
                    continue;
                }
                std::cout << "    " << omega_names[i] << ": 样本 " << aim_slot_by_omega[i].size()
                          << "，中位 " << percentile(aim_slot_by_omega[i], 0.50) << " m，p90 "
                          << percentile(aim_slot_by_omega[i], 0.90) << " m\n";
            }
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
    if (args.web)
    {
        if (args.web_hold_s > 0.0)
        {
            std::cout << "\n[web] 回放结束，服务器保持 " << args.web_hold_s
                      << " s：http://127.0.0.1:" << args.web_port << "/\n" << std::flush;
            std::this_thread::sleep_for(std::chrono::duration<double>(args.web_hold_s));
        }
        STOP_WEB_SERVER();
    }
    return 0;
}
