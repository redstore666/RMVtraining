#include "AppSupport.hpp"
#include "SerialPort.hpp"
#include "StudentTasks.hpp"
#include "WebLogger.hpp"

#ifdef RMV_SYSTEM_WEB
// 迷你俯视图（上车调试显示）：渲染/推流全部被动，串口控制路径零改动。
// 2026-10-27 起不再单开 8081 端口：俯视图 JPEG 直接并入 webdebug（8080）的
// WEB_TOPDOWN_FRAME 通道，网页右下角 <img src="/frame.jpg"> 显示——上车实测
// 独立的 SystemWebLogger(8081) 拿不到数据，而 8080 是确认可达的，只保一条通道。
#include "AimGeometry.hpp"
#include "SystemView.hpp"

#include <opencv2/imgcodecs.hpp>
#endif

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <string>
#include <thread>

#include <opencv2/highgui.hpp> // --show-camera: imshow/waitKey
#include <opencv2/imgproc.hpp> // --show-camera: polylines/putText
#include <vector>

// 这个文件有意保留 aimbot_26/main.cpp 的组织方式：
// main 负责初始化，ReadFunction 负责收串口，OperationFunction 负责整条视觉流水线。
// 学生只需要替换 student/src/StudentTasks.cpp 中的四个任务函数。

namespace {

constexpr int kDefaultWebPort = 8080;
constexpr int kInitialReadAttempts = 5;
constexpr auto kIdleDelay = std::chrono::milliseconds(10);
// 控制器数据过期后，结果必须降级为无目标，不能继续允许开火。
constexpr auto kMessageMaxAge = std::chrono::milliseconds(500);

// 程序退出标志。原子变量同时供信号处理函数、读线程和操作线程使用。
static_assert(std::atomic<bool>::is_always_lock_free,
              "退出标志必须支持无锁原子操作");
std::atomic<bool> shutdown_requested{false};

// 下面几个全局变量对应参考工程中的 camera、temp、translator、pic。
// 相机 SDK 的对象和生命周期由学生在 get_pic() 中自行管理，这里保存已标定参数。
CameraParameters camera;
Translator temp;       // 读线程写入的最新控制器帧
Translator translator; // 操作线程使用并回写的当前帧
cv::Mat pic;           // 操作线程当前取得的图像

// temp 及其时间戳由同一把锁保护，避免读线程写入时操作线程同时复制。
std::mutex message_mutex;
bool message_available = false;
std::chrono::steady_clock::time_point message_received_at{};

int web_port = kDefaultWebPort;

// 命令行选项的全局副本（与 web_port 同一模式：pthread 线程不传参，直接读全局）。
// forced_enemy_color: 0 = 敌方颜色跟串口状态（默认，原行为）；
//                     1 = 强制敌方为红方（--red），2 = 强制敌方为蓝方（--blue）。
// show_camera_window: --show-camera 时在本地窗口显示相机画面 + 检测框。
int forced_enemy_color = 0;
bool show_camera_window = false;

// --show-camera 的跨线程交接：操作线程只把画好的帧放进缓冲，GUI 调用
// （imshow/waitKey）全部在主线程做。OpenCV 的 Qt 后端禁止非主线程 GUI，
// 否则直接 abort（不可捕获）——这是"桌面环境下 --show-camera 运行失败"
// 的最可能根因。GTK/headless 等其他后端同样由此一并覆盖。
std::mutex display_frame_mutex;
cv::Mat display_frame;
std::atomic<bool> operation_running{false};

struct Options {
    std::string serial_device;
    int baud_rate{115200};
    std::string camera_config;
    int forced_color{0}; // 0=跟串口, 1=--red, 2=--blue
    bool show_camera{false};
};

void request_shutdown(int) {
    shutdown_requested.store(true, std::memory_order_relaxed);
}

void printUsage(const char* program) {
    std::cout << "用法: " << program
              << " <串口设备> <波特率> <相机标定文件> [Web端口] [选项]\n"
              << "选项:\n"
              << "  --red           强制敌方为红方（默认跟串口 status）\n"
              << "  --blue          强制敌方为蓝方（默认跟串口 status）\n"
              << "  --show-camera   本地窗口显示相机画面 + 检测框（调车用）\n"
              << "示例: " << program
              << " /dev/ttyACM0 115200 config/camera.yaml 8080 --blue --show-camera\n";
}

bool parsePositiveInt(const char* text, int& value) {
    try {
        std::size_t parsed = 0;
        const int candidate = std::stoi(text, &parsed);
        if (text[parsed] != '\0' || candidate <= 0) {
            return false;
        }
        value = candidate;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

bool parseOptions(int argc, char** argv, Options& options) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        return false;
    }

    // 位置参数与标志混排：标志可放在任意位置，其余按顺序是
    // <串口设备> <波特率> <相机标定文件> [Web端口]。
    std::vector<const char*> positional;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            printUsage(argv[0]);
            return false;
        }
        if (arg == "--red") {
            options.forced_color = 1;
        } else if (arg == "--blue") {
            options.forced_color = 2;
        } else if (arg == "--show-camera") {
            options.show_camera = true;
        } else {
            positional.push_back(argv[i]);
        }
    }

    if (positional.size() < 3 || positional.size() > 4) {
        printUsage(argv[0]);
        return false;
    }

    options.serial_device = positional[0];
    options.camera_config = positional[2];
    if (!parsePositiveInt(positional[1], options.baud_rate)) {
        std::cerr << "波特率无效: " << positional[1] << '\n';
        return false;
    }
    if (positional.size() == 4 && !parsePositiveInt(positional[3], web_port)) {
        std::cerr << "Web端口无效: " << positional[3] << '\n';
        return false;
    }
    return true;
}

bool hasFrameMarkers(const char* frame) noexcept {
    // 帧头和帧尾只用于确认边界；浮点数是否有效在操作线程中检查。
    return static_cast<std::uint8_t>(frame[0]) == kFrameHead &&
           static_cast<std::uint8_t>(frame[kCommunicationFrameSize - 1]) == kFrameTail;
}

bool hasFiniteControlInput(const MessData_AutoAim& message) noexcept {
    return std::isfinite(message.yaw) && std::isfinite(message.pitch) &&
           std::isfinite(message.roll) && std::isfinite(message.bias);
}

void publishFrame(const Translator& incoming) {
    std::lock_guard<std::mutex> lock(message_mutex);
    temp = incoming;
    message_received_at = std::chrono::steady_clock::now();
    message_available = true;
}

bool copyLatestFrame(Translator& destination,
                     std::chrono::steady_clock::time_point& received_at) {
    std::lock_guard<std::mutex> lock(message_mutex);
    if (!message_available) {
        return false;
    }
    destination = temp;
    received_at = message_received_at;
    return true;
}

bool frameIsFresh(const std::chrono::steady_clock::time_point& received_at) {
    return std::chrono::steady_clock::now() - received_at <= kMessageMaxAge;
}

bool writeFrame(SerialPort& serialPort, Translator& frame) {
    // 统一设置协议边界，并保证一次写出完整的 64 字节帧。
    frame.message.head = kFrameHead;
    frame.message.tail = kFrameTail;
    const int written =
        serialPort.Write(frame.data, static_cast<int>(kCommunicationFrameSize));
    if (written != static_cast<int>(kCommunicationFrameSize)) {
        std::cerr << "串口写入失败\n";
        shutdown_requested.store(true, std::memory_order_relaxed);
        return false;
    }
    return true;
}

void logPrediction(const PredictionResult& result) {
    WEB_LOG("target_found", result.state == TrackingState::NoTarget ? 0.0 : 1.0);
    WEB_LOG("ekf_state", static_cast<double>(result.state));
    if (result.state == TrackingState::NoTarget) {
        return;
    }
    WEB_LOG("target_id", result.target_id);
    WEB_LOG("target_yaw", result.target_yaw);
    WEB_LOG("target_pitch", result.target_pitch);
    WEB_LOG("target_distance", result.distance_m);
    WEB_LOG("ekf_x_m", result.center_x_m);
    WEB_LOG("ekf_y_m", result.center_y_m);
    WEB_LOG("ekf_vx_m_s", result.center_velocity_x_m_s);
    WEB_LOG("ekf_vy_m_s", result.center_velocity_y_m_s);
    WEB_LOG("ekf_yaw_rad", result.body_yaw);
    WEB_LOG("ekf_yaw_rate_rad_s", result.body_yaw_velocity);
    WEB_LOG("ekf_radius_1_m", result.radius_1_m);
    WEB_LOG("ekf_radius_2_m", result.radius_2_m);
}

// main 中启动线程前先同步读一帧，保持参考工程的初始化顺序。
// 返回 1 表示读到合法帧，0 表示暂时超时，-1 表示串口出错。
int readInitialFrame(SerialPort& serialPort, Translator& frame) {
    for (int attempt = 0; attempt < kInitialReadAttempts &&
                            !shutdown_requested.load(std::memory_order_relaxed);
         ++attempt) {
        Translator incoming{};
        const int length =
            serialPort.Read(incoming.data, static_cast<int>(kCommunicationFrameSize));
        if (length < 0) {
            return -1;
        }
        if (length == static_cast<int>(kCommunicationFrameSize) &&
            hasFrameMarkers(incoming.data)) {
            frame = incoming;
            return 1;
        }
    }
    return 0;
}

}  // 匿名命名空间

// 串口读线程：只负责读取完整帧并更新共享的 temp。
void *ReadFunction(void *arg);
// 操作线程：按照“复制 -> 取流 -> 检测 -> PnP -> EKF -> 写回”的顺序处理。
void *OperationFunction(void *arg);

int main(int argc, char** argv) {
    std::signal(SIGINT, request_shutdown);
    std::signal(SIGTERM, request_shutdown);

    Options options;
    if (!parseOptions(argc, argv, options)) {
        return argc == 2 && std::string(argv[1]) == "--help" ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    // 选项落全局（线程不传参，直接读全局，与 web_port 同一模式）。
    forced_enemy_color = options.forced_color;
    show_camera_window = options.show_camera;
    if (show_camera_window && std::getenv("DISPLAY") == nullptr) {
        // 车上一般 SSH 无图形环境，cv::imshow 会直接异常退出——提前拦下。
        std::cerr << "警告: 当前无图形环境（DISPLAY 未设置），--show-camera 已禁用。"
                     "需要画面时请在带桌面的会话里运行。\n";
        show_camera_window = false;
    }
    if (forced_enemy_color == 1) {
        std::cout << "敌方颜色: 强制红方（--red，忽略串口 status）\n";
    } else if (forced_enemy_color == 2) {
        std::cout << "敌方颜色: 强制蓝方（--blue，忽略串口 status）\n";
    } else {
        std::cout << "敌方颜色: 跟随串口 status\n";
    }

    std::string camera_error;
    if (!loadCameraParameters(options.camera_config, camera, camera_error)) {
        std::cerr << camera_error << '\n';
        return EXIT_FAILURE;
    }

    // 与参考工程一样，串口对象在 main 中创建，再把指针传给两个线程。
    SerialPort* serialPort = new SerialPort(options.serial_device);
    if (!serialPort->isOpen()) {
        std::cerr << "无法打开串口设备: " << options.serial_device << '\n';
        delete serialPort;
        return EXIT_FAILURE;
    }
    if (!serialPort->InitSerialPort(options.baud_rate, 8, 1, 'N')) {
        std::cerr << "串口配置失败\n";
        delete serialPort;
        return EXIT_FAILURE;
    }

    // 先同步读取首帧并缓存控制器状态；后续每帧仍会重新读取 status 来确定敌方颜色。
    Translator first_frame{};
    const int initial_result = readInitialFrame(*serialPort, first_frame);
    if (initial_result < 0) {
        std::cerr << "读取串口首帧失败\n";
        delete serialPort;
        return EXIT_FAILURE;
    }
    if (initial_result > 0) {
        publishFrame(first_frame);
        std::cout << "收到首帧，status = "
                  << static_cast<unsigned int>(first_frame.message.status) << '\n';
    } else {
        // 首帧超时不直接退出，读线程启动后仍会继续等待控制器数据。
        std::cerr << "未在启动阶段收到合法首帧，继续等待串口数据\n";
    }

    pthread_t readThread{};
    pthread_t operationThread{};
    bool read_started = false;
    bool operation_started = false;

    // 保留参考工程的 pthread 双线程结构：先创建读线程，再创建操作线程。
    if (pthread_create(&readThread, nullptr, ReadFunction, serialPort) == 0) {
        read_started = true;
    } else {
        std::cerr << "创建串口读线程失败\n";
    }
    if (read_started &&
        pthread_create(&operationThread, nullptr, OperationFunction, serialPort) == 0) {
        operation_started = true;
    } else {
        std::cerr << "创建视觉操作线程失败\n";
        shutdown_requested.store(true, std::memory_order_relaxed);
    }

    // --show-camera 的显示泵：GUI 必须在主线程（OpenCV Qt 后端禁止非主线程
    // GUI 调用，会直接 abort）。操作线程只往共享缓冲放画好的帧，这里以
    // ~60fps 取出显示；q / ESC 请求退出。显示失败不拖垮控制程序。
    if (operation_started && show_camera_window) {
        try {
            cv::namedWindow("camera (--show-camera)", cv::WINDOW_AUTOSIZE);
        } catch (const cv::Exception& e) {
            std::cerr << "show-camera 无法创建窗口，已禁用: " << e.what() << '\n';
            show_camera_window = false;
        }
    }
    while (operation_started && show_camera_window &&
           operation_running.load(std::memory_order_acquire) &&
           !shutdown_requested.load(std::memory_order_relaxed)) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(display_frame_mutex);
            if (!display_frame.empty()) {
                frame = display_frame; // 浅拷贝引用计数，锁内极短
            }
        }
        if (!frame.empty()) {
            cv::imshow("camera (--show-camera)", frame);
        }
        // waitKey 驱动窗口事件循环；q / ESC 请求退出。
        const int key = cv::waitKey(16);
        if (key == 'q' || key == 27) {
            shutdown_requested.store(true, std::memory_order_relaxed);
            break;
        }
    }
    if (show_camera_window) {
        cv::destroyAllWindows();
    }

    // 先等待操作线程结束，再通知读线程退出，和原 main 的生命周期一致。
    if (operation_started) {
        pthread_join(operationThread, nullptr);
    }
    shutdown_requested.store(true, std::memory_order_relaxed);
    if (read_started) {
        pthread_join(readThread, nullptr);
    }

    STOP_WEB_SERVER();
    delete serialPort;
    return operation_started ? EXIT_SUCCESS : EXIT_FAILURE;
}

void *ReadFunction(void *arg) {
    SerialPort* serialPort = static_cast<SerialPort*>(arg);
    if (serialPort == nullptr) {
        shutdown_requested.store(true, std::memory_order_relaxed);
        return nullptr;
    }

    // SerialPort::Read 已经负责短读缓存；这里额外保留滑动缓冲，
    // 使程序从半帧启动或遇到干扰字节时仍能重新找到 0x71 ... 0x4C。
    std::vector<char> stream_buffer;
    stream_buffer.reserve(kCommunicationFrameSize * 2);

    while (!shutdown_requested.load(std::memory_order_relaxed)) {
        std::array<char, kCommunicationFrameSize> chunk{};
        const int length =
            serialPort->Read(chunk.data(), static_cast<int>(chunk.size()));
        if (length < 0) {
            std::cerr << "串口读取失败\n";
            shutdown_requested.store(true, std::memory_order_relaxed);
            break;
        }
        if (length == 0) {
            // 超时不是错误；短暂让出 CPU 后继续等待下一帧。
            std::this_thread::sleep_for(kIdleDelay);
            continue;
        }
        if (length > static_cast<int>(chunk.size())) {
            continue;
        }

        stream_buffer.insert(stream_buffer.end(), chunk.begin(), chunk.begin() + length);
        std::size_t consumed = 0;
        while (stream_buffer.size() - consumed >= kCommunicationFrameSize) {
            const char* candidate = stream_buffer.data() + consumed;
            if (!hasFrameMarkers(candidate)) {
                ++consumed;
                continue;
            }

            Translator incoming{};
            std::memcpy(incoming.data, candidate, kCommunicationFrameSize);
            consumed += kCommunicationFrameSize;
            publishFrame(incoming);
        }
        if (consumed != 0U) {
            stream_buffer.erase(stream_buffer.begin(),
                                stream_buffer.begin() + static_cast<std::ptrdiff_t>(consumed));
        }
    }
    return nullptr;
}

void *OperationFunction(void *arg) {
    SerialPort* serialPort = static_cast<SerialPort*>(arg);
    if (serialPort == nullptr) {
        shutdown_requested.store(true, std::memory_order_relaxed);
        operation_running.store(false, std::memory_order_release);
        return nullptr;
    }
    operation_running.store(true, std::memory_order_release);

    // Web 服务和相机 SDK 都属于操作线程的资源，生命周期与原工程一致。
    START_WEB_SERVER(web_port);

    double dt = 0.0;
    double last_time_stamp = 0.0;
    auto last_frame_time = std::chrono::steady_clock::time_point{};
    int empty_frame_count = 0;

#ifdef RMV_SYSTEM_WEB
    // 迷你俯视图渲染门。实测车上操作循环 ~10ms/帧（约 100fps），原 60Hz 门限
    // 会隔帧丢弃、实际只有 ~45Hz；画面又在网页侧被 30fps 轮询再压一档，视觉上
    // 明显卡顿。画布仅 480x360，JPEG 编码 ~1ms，每帧发布成本可忽略——门限放宽
    // 到 120Hz，让发布跟随循环逐帧更新；流畅度上限由网页轮询决定。
    const int64_t mini_interval_ticks = cv::getTickFrequency() / 120;
    int64_t last_mini_tick = 0;
#endif

    while (!shutdown_requested.load(std::memory_order_relaxed)) {
        const auto process_start = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point received_at{};

        // 对应 aimbot_26 中的 MManager.copy(temp, translator)。
        if (!copyLatestFrame(translator, received_at)) {
            std::this_thread::sleep_for(kIdleDelay);
            continue;
        }

        // 不支持的状态或异常姿态不进入学生算法，直接回传无目标帧。
        if (!frameIsFresh(received_at) || !isAutoAimStatus(translator.message.status) ||
            !hasFiniteControlInput(translator.message)) {
            setSafeOutput(translator);
            writeFrame(*serialPort, translator);
            // 同一份过期快照可能会被重复看到，避免无休止地高速回包。
            std::this_thread::sleep_for(kIdleDelay);
            continue;
        }

        const GimbalState gimbal = gimbalStateFromMessage(translator.message);

        // 对应参考工程中的 camera.get_pic(&pic, ...)。具体 SDK 初始化和取流由学生完成。
        pic.release();
        const bool got_pic = get_pic(pic);
        bool using_fallback_frame = false;
        if (!got_pic || pic.empty()) {
            ++empty_frame_count;
            WEB_LOG("empty_frame", static_cast<double>(empty_frame_count));
            if (empty_frame_count > 3) {
                // 与参考工程一致，连续多次取流失败后结束操作线程。
                setSafeOutput(translator);
                translator.message.latency = static_cast<float>(
                    std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - process_start)
                        .count());
                writeFrame(*serialPort, translator);
                std::cerr << "连续取流失败，操作线程退出\n";
                shutdown_requested.store(true, std::memory_order_relaxed);
                break;
            }
            // 临时补一张黑帧，使后面的检测、PnP 和 EKF 阶段仍按原顺序执行。
            // 该帧不是相机观测，最终输出会被强制设为无目标。
            pic = cv::Mat(camera.image_height, camera.image_width, CV_8UC3,
                          cv::Scalar(0, 0, 0));
            using_fallback_frame = true;
        } else {
            empty_frame_count = 0;
        }
        if (!using_fallback_frame) {
            WEB_LOG("empty_frame", 0.0);
        }

        const auto frame_time = std::chrono::steady_clock::now();
        double frame_seconds = 0.0;
        if (last_frame_time.time_since_epoch().count() != 0) {
            frame_seconds =
                std::chrono::duration<double>(frame_time - last_frame_time).count();
            if (frame_seconds > 0.0) {
                WEB_LOG("camera_fps", 1.0 / frame_seconds);
            }
        }
        last_frame_time = frame_time;

        // 1. 装甲板检测。检测结果为空时不提前结束，继续向下传递空观测。
        // 敌方颜色默认跟串口 status；--red/--blue 强制覆盖（调车时串口状态
        // 可能没接对，颜色错了检测器就全程空转）。
        TeamColor enemy_color = enemyColorFromStatus(translator.message.status);
        if (forced_enemy_color == 1) {
            enemy_color = TeamColor::Red;
        } else if (forced_enemy_color == 2) {
            enemy_color = TeamColor::Blue;
        }
        const std::vector<ArmorDetection> detections = armor_detect(pic, enemy_color);
        WEB_LOG("armor_count", static_cast<double>(detections.size()));
        if (!detections.empty()) {
            // 记录第一块候选装甲板，便于在网页中快速判断检测器是否工作。
            const ArmorDetection& first_detection = detections.front();
            WEB_LOG("detection_id", first_detection.target_id);
            WEB_LOG("detection_confidence", first_detection.confidence);
            const cv::Point2f center =
                (first_detection.corners[0] + first_detection.corners[1] +
                 first_detection.corners[2] + first_detection.corners[3]) * 0.25F;
            WEB_LOG("target_x_px", center.x);
            WEB_LOG("target_y_px", center.y);
        }

        // 2. PnP 位姿解算。PnP 失败时同样保留空 observations，交给 EKF 做纯预测。
        const std::vector<ArmorPose> observations =
            armor_solve(detections, camera, gimbal);
        if (!observations.empty()) {
            WEB_LOG("pnp_reprojection_error", observations.front().reprojection_error);
            WEB_LOG("pnp_distance_m", cv::norm(observations.front().position_gimbal_m));
        } else {
            WEB_LOG("pnp_reprojection_error", -1.0);
        }

        // --show-camera：画好带检测框的画面放进共享缓冲，由主线程的显示泵
        // 负责 imshow/waitKey（GUI 必须在主线程，见全局变量处的说明）。
        if (show_camera_window) {
            cv::Mat display = pic.clone();
            for (const ArmorDetection& detection : detections) {
                const cv::Point2f* pts = detection.corners.data();
                cv::polylines(display,
                              std::vector<cv::Point2f>(pts, pts + 4), true,
                              cv::Scalar(0, 255, 0), 2);
                const cv::Point2f center =
                    (detection.corners[0] + detection.corners[1] +
                     detection.corners[2] + detection.corners[3]) * 0.25F;
                cv::putText(display,
                            cv::format("id=%d %.2f", detection.target_id,
                                       static_cast<double>(detection.confidence)),
                            center + cv::Point2f(0.0F, -12.0F),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(0, 255, 0), 1);
            }
            cv::putText(display,
                        cv::format("enemy=%s fps=%.1f",
                                   enemy_color == TeamColor::Red ? "RED" : "BLUE",
                                   1.0 / std::max(frame_seconds, 1.0e-6)),
                        cv::Point(10, 24), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(255, 255, 255), 2);
            {
                std::lock_guard<std::mutex> lock(display_frame_mutex);
                display_frame = std::move(display);
            }
        }

        // 3. 计算帧间隔和通信延迟，再调用 EKF。即使 observations 为空也必须调用。
        const double time_stamp =
            std::chrono::duration<double>(frame_time.time_since_epoch()).count();
        dt = last_time_stamp > 0.0 ? time_stamp - last_time_stamp : 0.0;
        if (dt < 0.0 || !std::isfinite(dt)) {
            dt = 0.0;
        }
        last_time_stamp = time_stamp;
        translator.message.latency = static_cast<float>(
            std::chrono::duration<double, std::milli>(frame_time - process_start).count());
        WEB_LOG("frame_dt_ms", dt * 1000.0);

        const PredictionResult prediction = ekf_predict(observations, gimbal, time_stamp);
        bool prediction_valid = applyPrediction(translator, prediction);

        // 学生算法耗时过长时，丢弃本次结果，避免旧控制器数据触发动作。
        if (using_fallback_frame || !frameIsFresh(received_at)) {
            setSafeOutput(translator);
            prediction_valid = false;
        }

        const auto process_end = std::chrono::steady_clock::now();
        translator.message.latency = static_cast<float>(
            std::chrono::duration<double, std::milli>(process_end - process_start).count());
        WEB_LOG("process_time_ms", translator.message.latency);
        logPrediction(prediction_valid ? prediction : PredictionResult{});

        // 4. 将结果写回原 64 字节结构体。
        writeFrame(*serialPort, translator);

#ifdef RMV_SYSTEM_WEB
        // ===== 迷你俯视图（纯被动显示：渲染/推流在串口写出之后，零控制延迟）=====
        // 渲染推帧并入 webdebug 的 8080 通道（WEB_TOPDOWN_FRAME / WEB_LOG），
        // 网页右下角显示。几何与 test_camera_system 的刚体版一致：座位由
        // ekf_topdown_snapshot() 提供（EKF 刚体 + 每槽位显示偏移），绿色实测板 =
        // 本帧 PnP 观测（gimbal->world + 180° 消歧，与 collectWorldObservations
        // 同一变换）。
        {
            const int64_t now_tick = cv::getTickCount();
            if (now_tick - last_mini_tick >= mini_interval_ticks)
            {
                last_mini_tick = now_tick;
                rotating_armor::MiniTopDownFrame mini;
                mini.camera_position = Eigen::Vector3d(0.0, 0.0, 0.0);
                const TopDownSnapshot snap = ekf_topdown_snapshot();
                if (snap.valid)
                {
                    mini.has_body = true;
                    mini.body_centre = Eigen::Vector3d(snap.center_x_m, snap.center_y_m,
                                                       snap.center_z_m);
                    // 车体速度（作业 3.7 要求俯视图能看到 EKF 速度）：橙色箭头 +
                    // caption 里的 vx/vy 分量。锁定期之外才有意义，snap.valid 已经保证。
                    mini.velocity = Eigen::Vector2d(snap.velocity_x_m_s, snap.velocity_y_m_s);
                    mini.has_velocity = true;
                    constexpr int kPlateCount = 4;
                    for (int slot = 0; slot < kPlateCount; ++slot)
                    {
                        mini.plate_centres.push_back(Eigen::Vector3d(
                            snap.seat_centre_m[slot][0], snap.seat_centre_m[slot][1],
                            snap.seat_centre_m[slot][2]));
                        mini.plate_yaws.push_back(snap.seat_face_yaw_rad[slot]);
                        mini.plate_half_widths.push_back(0.0675); // 小板；宽学习留给工具版
                    }
                    mini.aim_slot = snap.aim_slot;
                    if (snap.has_predicted_aim)
                    {
                        mini.has_predicted_aim = true;
                        mini.predicted_aim = Eigen::Vector3d(
                            snap.predicted_aim_m[0], snap.predicted_aim_m[1],
                            snap.predicted_aim_m[2]);
                    }
                    // 实测叠加：绿线整圈稳定、灰线慢半拍 → EKF 相位滞后；
                    // 绿线乱跳、灰线平稳 → 观测/关联侧问题，不是滤波器。
                    const std::uint8_t want_id = prediction.target_id;
                    if (want_id != 0)
                    {
                        const Eigen::Matrix3d rotation =
                            rotating_armor::gimbalToWorld(gimbal);
                        for (const ArmorPose &pose : observations)
                        {
                            if (pose.detection.target_id != want_id)
                                continue;
                            const Eigen::Vector3d p_gimbal(pose.position_gimbal_m[0],
                                                           pose.position_gimbal_m[1],
                                                           pose.position_gimbal_m[2]);
                            if (!p_gimbal.allFinite() || p_gimbal.norm() < 1.0e-3)
                                continue;
                            const Eigen::Vector3d p_world = rotation * p_gimbal;
                            const Eigen::Vector3d n_gimbal(pose.normal_gimbal[0],
                                                           pose.normal_gimbal[1],
                                                           pose.normal_gimbal[2]);
                            Eigen::Vector3d normal_world = Eigen::Vector3d::Zero();
                            if (n_gimbal.allFinite() && n_gimbal.norm() > 0.5)
                                normal_world = rotation * (n_gimbal / n_gimbal.norm());
                            else
                                normal_world =
                                    rotation * Eigen::Vector3d(std::cos(pose.armor_yaw),
                                                               std::sin(pose.armor_yaw), 0.0);
                            if (normal_world.dot(p_world) > 0.0)
                                normal_world = -normal_world;
                            const double yaw =
                                (std::hypot(normal_world.x(), normal_world.y()) > 1.0e-3)
                                    ? rotating_armor::vectorYaw(normal_world)
                                    : 0.0;
                            mini.detected_centres.push_back(p_world);
                            mini.detected_yaws.push_back(yaw);
                            mini.detected_half_widths.push_back(
                                pose.detection.size == ArmorSize::Large ? 0.115 : 0.0675);
                        }
                    }
                    mini.caption =
                        cv::format("d=%.2fm R=%.2fm w=%+.1f", prediction.distance_m,
                                   prediction.radius_1_m,
                                   static_cast<double>(prediction.body_yaw_velocity));
                }
                else
                {
                    mini.caption = "no target";
                }
                const cv::Mat mini_canvas = rotating_armor::renderMiniTopDown(mini);
                std::vector<unsigned char> jpeg;
                const std::vector<int> jpeg_params = {cv::IMWRITE_JPEG_QUALITY, 80};
                if (cv::imencode(".jpg", mini_canvas, jpeg, jpeg_params))
                    WEB_TOPDOWN_FRAME(jpeg);
                WEB_LOG("topdown_omega", static_cast<double>(prediction.body_yaw_velocity));
                WEB_LOG("topdown_distance_m", static_cast<double>(prediction.distance_m));
                // 俯视图上的速度（世界系 vx/vy + 模长），与图上橙色箭头同一数据源，
                // 便于事后用 CSV 复核"图上箭头长度对不对"。
                WEB_LOG("topdown_vx_m_s", snap.valid ? snap.velocity_x_m_s : 0.0);
                WEB_LOG("topdown_vy_m_s", snap.valid ? snap.velocity_y_m_s : 0.0);
                WEB_LOG("topdown_speed_m_s",
                        snap.valid ? std::hypot(snap.velocity_x_m_s, snap.velocity_y_m_s) : 0.0);
            }
        }
#endif
    }

    STOP_WEB_SERVER();
    operation_running.store(false, std::memory_order_release);
    return nullptr;
}
