// EKF 俯视图（Top-Down）演示与调参工具。
//
// 合成一台绕云台旋转的车体，把 rotating_armor::TopDownView 的每一帧画出来：
//   * 红点 = EKF 车体中心，蓝条 = 模型推出的 4 块装甲板，黄边 = 当前瞄准板；
//   * 绿点 = 本帧 PnP 实测板心（和蓝条错开就是抖动或状态偏差）；
//   * 品红十字 = 预测瞄准点（PredictionResult::target_yaw/pitch/distance 的方向），
//     灰虚线 = 云台当前光轴，两根线的夹角就是开火判据里的 yaw 误差；
//   * 左上文本 = 状态机 / 速度 / 角速度 / R / NIS / 观测数；底部 = 图例与比例尺。
//
// 用法：
//   ./build/test_ekf_viewer                              # 窗口实时看（跟随车心）
//   ./build/test_ekf_viewer --world --ppm 200            # 固定世界原点视图（近目标 ±2m，远目标自动缩小）
//   ./build/test_ekf_viewer --no-show --out build/topdown_demo   # 无显示环境导出 PNG
//   ./build/test_ekf_viewer --omega 0 --range 2.5        # 静止目标（测开火与中心稳定性）
//
// 参数：
//   --range D   车体中心到云台的水平距离，米（默认 3.0）
//   --omega W   车体自转角速度，rad/s（默认 6.0，0 = 不转）
//   --radius R  装甲半径，米（默认 0.25）
//   --speed V   车体平移速度，米/秒（沿 +y，默认 0）
//   --frames N  跑多少帧，60fps（默认 240）
//   --world     固定世界原点视图（默认跟随车体中心）
//   --ppm P     世界模式的缩放，像素/米（默认 200）
//   --trail N   轨迹保留帧数（默认 50，0 = 不画）
//   --out DIR   每帧保存 PNG（目录自动创建）
//   --no-show   不开窗口
//   --wait MS   每帧等待毫秒（默认 25）
//   --seed S    随机种子（默认 20240607）
//   --help      打印本说明
//
// 这个工具只读 EKF 结果，不参与主程序；同样的渲染器也被
// test_camera_playback --topdown 用来显示真视频回放。

#include "TopDownView.hpp"
#include "tracker_test_support.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

namespace
{

struct Options
{
    double range_m{3.0};
    double omega{6.0};
    double radius{0.25};
    double speed{0.0};
    int frames{240};
    bool world{false};
    double pixels_per_meter{200.0};
    int trail{50};
    std::string out_dir;
    bool show{true};
    int wait_ms{25};
    unsigned int seed{20240607U};
};

void printUsage(const char *program)
{
    std::printf(
        "用法: %s [--range D] [--omega W] [--radius R] [--speed V] [--frames N]\n"
        "          [--world] [--ppm P] [--trail N] [--out DIR] [--no-show]\n"
        "          [--wait MS] [--seed S]\n"
        "默认: range=3.0m omega=6.0rad/s radius=0.25m frames=240 跟随车心视图\n",
        program);
}

bool parseArgs(int argc, char **argv, Options &options)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const auto value = [&](const char *name) -> const char * {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "参数 %s 缺少取值\n", name);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--range")
            options.range_m = std::atof(value("--range"));
        else if (arg == "--omega")
            options.omega = std::atof(value("--omega"));
        else if (arg == "--radius")
            options.radius = std::atof(value("--radius"));
        else if (arg == "--speed")
            options.speed = std::atof(value("--speed"));
        else if (arg == "--frames")
            options.frames = std::max(1, std::atoi(value("--frames")));
        else if (arg == "--world")
            options.world = true;
        else if (arg == "--ppm")
            options.pixels_per_meter = std::atof(value("--ppm"));
        else if (arg == "--trail")
            options.trail = std::max(0, std::atoi(value("--trail")));
        else if (arg == "--out")
            options.out_dir = value("--out");
        else if (arg == "--no-show")
            options.show = false;
        else if (arg == "--wait")
            options.wait_ms = std::max(1, std::atoi(value("--wait")));
        else if (arg == "--seed")
            options.seed = static_cast<unsigned int>(std::strtoul(value("--seed"), nullptr, 10));
        else if (arg == "--help" || arg == "-h")
        {
            printUsage(argv[0]);
            std::exit(0);
        }
        else
        {
            std::fprintf(stderr, "未知参数: %s\n", arg.c_str());
            printUsage(argv[0]);
            return false;
        }
    }
    return true;
}

// 合成"完美跟瞄"的云台：指向真值里离当前指向最近的那块可见板。
void aimGimbalAtVisiblePlate(sim::Rig &rig)
{
    const std::vector<sim::Plate> plates = sim::platesAt(rig.truth, rig.plate_count);
    const sim::Plate *best = nullptr;
    double best_error = 1.0e9;
    for (const sim::Plate &plate : plates)
    {
        // 背面的板被车体挡住，不能让云台去追（和 Rig::step 的可见性判据一致）。
        if (plate.normal.dot(-plate.position.normalized()) <= 0.3)
            continue;
        const double error = std::fabs(rotating_armor::wrapToPi(
            rotating_armor::vectorYaw(plate.position) - rig.gimbal.yaw));
        if (error < best_error)
        {
            best_error = error;
            best = &plate;
        }
    }
    if (best != nullptr)
        rig.gimbal.snapTo(best->position);
}

} // namespace

int main(int argc, char **argv)
{
    Options options;
    if (!parseArgs(argc, argv, options))
        return 1;

    // 没有显示环境时自动退化成导出图片，避免"跑完什么都没看见"。
    const bool has_display = std::getenv("DISPLAY") != nullptr ||
                             std::getenv("WAYLAND_DISPLAY") != nullptr;
    if (options.show && !has_display)
    {
        std::cout << "[俯视图] 未检测到显示环境（DISPLAY/WAYLAND_DISPLAY 为空），"
                     "自动关闭窗口\n";
        options.show = false;
    }
    if (!options.show && options.out_dir.empty())
    {
        options.out_dir = "build/topdown_demo";
        std::cout << "[俯视图] 没有窗口也没有 --out，默认导出到 " << options.out_dir << "\n";
    }

    rotating_armor::TopDownOptions view_options;
    view_options.width = 800;
    view_options.height = 800;
    view_options.follow_target = !options.world;
    view_options.pixels_per_meter = options.pixels_per_meter;
    view_options.trail_length = options.trail;
    rotating_armor::TopDownView view(view_options);

    // 初值：车体中心在 +x 方向 range 米处，沿 +y 平移，绕 z 轴以 omega 自转。
    rotating_armor::State truth = rotating_armor::State::Zero();
    truth(0) = options.range_m;
    truth(3) = options.speed;
    truth(7) = options.omega;
    truth(8) = options.radius;
    sim::Rig rig(truth);
    rig.rng.seed(options.seed);

    if (!options.out_dir.empty())
    {
        std::error_code error;
        std::filesystem::create_directories(options.out_dir, error);
        if (error)
        {
            std::fprintf(stderr, "建目录失败 %s: %s\n", options.out_dir.c_str(),
                         error.message().c_str());
            return 1;
        }
    }

    if (options.show)
        cv::namedWindow("EKF top-down", cv::WINDOW_AUTOSIZE);

    int saved = 0;
    int tracked_frames = 0;
    bool quit = false;
    for (int frame_index = 0; frame_index < options.frames && !quit; ++frame_index)
    {
        // 云台先对准（模拟完美跟瞄），再按真值推进并用同一姿态生成观测。
        aimGimbalAtVisiblePlate(rig);
        const PredictionResult prediction = rig.step(true, false);
        const GimbalState gimbal = rig.gimbal.state(rig.bias_s);

        rotating_armor::TopDownFrame frame;
        frame.has_state = rig.tracker.initialized();
        if (frame.has_state)
            frame.state = rig.tracker.ekf().state();
        frame.diagnostics = rig.tracker.diagnostics();
        frame.result = prediction;
        frame.gimbal = gimbal;
        const Eigen::Matrix3d gimbal_to_world = rotating_armor::gimbalToWorld(gimbal);
        for (const ArmorPose &pose : rig.last_observations)
        {
            rotating_armor::TopDownObservation observation;
            observation.position =
                gimbal_to_world * Eigen::Vector3d(pose.position_gimbal_m[0],
                                                  pose.position_gimbal_m[1],
                                                  pose.position_gimbal_m[2]);
            observation.normal_yaw = rotating_armor::wrapToPi(pose.armor_yaw + gimbal.yaw);
            observation.target_id = pose.detection.target_id;
            frame.observations.push_back(observation);
        }

        const cv::Mat &canvas = view.render(frame);
        if (!options.out_dir.empty())
        {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/f%05d.png", options.out_dir.c_str(),
                          frame_index);
            if (cv::imwrite(path, canvas))
                ++saved;
        }
        if (frame.diagnostics.phase == rotating_armor::LockPhase::Tracking)
            ++tracked_frames;

        if (options.show)
        {
            cv::imshow("EKF top-down", canvas);
            const int key = cv::waitKey(options.wait_ms) & 0xFF;
            if (key == 27 || key == 'q')
                quit = true;
        }
    }

    // 结果自检：跟踪上以后状态应该贴着真值。
    std::printf("\n================ 俯视图演示 ================\n");
    std::printf("帧数 %d，其中 Tracking %d 帧（%d fps）\n", options.frames, tracked_frames,
                options.frames > 0 ? 60 : 0);
    if (saved > 0)
        std::printf("已保存 %d 张 PNG 到 %s\n", saved, options.out_dir.c_str());
    if (rig.tracker.initialized())
    {
        const rotating_armor::State &state = rig.tracker.ekf().state();
        const double center_error = std::hypot(state(0) - rig.truth(0), state(2) - rig.truth(2));
        // 四块板 90° 对称，yaw 只能识别到 mod 90°，所以误差按对称性折算。
        const double yaw_error = std::fabs(rotating_armor::wrapToPi(
            state(6) - rig.truth(6) - std::round((state(6) - rig.truth(6)) / (0.5 * sim::kPi)) *
                                        0.5 * sim::kPi));
        const double radius_error = std::fabs(state(8) - rig.truth(8));
        const char *phase = "?";
        switch (rig.tracker.diagnostics().phase)
        {
        case rotating_armor::LockPhase::Idle:
            phase = "idle";
            break;
        case rotating_armor::LockPhase::Collecting:
            phase = "collecting";
            break;
        case rotating_armor::LockPhase::Tracking:
            phase = "tracking";
            break;
        }
        std::printf("末帧 phase=%s  车心误差 %.3f m  yaw 误差 %.1f deg  R 误差 %.3f m  "
                    "omega %.2f (真值 %.2f) rad/s\n",
                    phase, center_error, yaw_error * 180.0 / sim::kPi, radius_error, state(7),
                    rig.truth(7));
        std::printf("末帧 state: cx=%.3f cy=%.3f yaw=%.1f deg R=%.3f\n", state(0), state(2),
                    state(6) * 180.0 / sim::kPi, state(8));
    }
    else
    {
        std::printf("末帧仍无 EKF 状态（还没锁上）\n");
    }
    return 0;
}
