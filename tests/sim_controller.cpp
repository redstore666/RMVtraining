// 虚拟电控：用一对伪终端（PTY）代替真实主控板，让 rmv_training 在没有电控硬件时
// 也能跑完整链路（串口 → 64 字节帧 → 视觉 → 回传帧）。
//
// 用法：
//   ./build/sim_controller                       # 默认我方红(status=0)、云台角 0、100Hz
//   ./build/sim_controller --status 5            # 我方蓝
//   ./build/sim_controller --bias 0.05           # 下发预测时间偏置(秒)
//   ./build/sim_controller --follow              # 用视觉回传的 target_yaw/pitch 当下一次
//                                                # 的云台角（相机真装在云台上、云台会动时用）
//   ./build/sim_controller --rate 50             # 帧率(Hz)
//
// 启动后它会打印从端路径，例如：
//   [sim] 视觉端请运行: ./build/rmv_training /dev/pts/7 115200 config/camera.yaml 8080
// 然后另开一个终端执行那行命令，Web Debug 在 http://127.0.0.1:8080。
//
// 它每帧校验视觉回传的 head/tail，并按 crc 变化打印状态，是"通信协议正确性"的
// 现场证据：armor_flag/coo_x/coo_y/fire_allowance/target_yaw 等字段都能看到。
#include "Communication.hpp"

#include <pty.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <sys/select.h>

namespace
{

volatile std::sig_atomic_t g_stop = 0;

void onSignal(int) { g_stop = 1; }

struct Options
{
    std::uint8_t status = 0; // 0=我方红，5=我方蓝
    float bias_s = 0.0F;     // MCU -> 视觉：预测时间偏置
    float yaw = 0.0F;
    float pitch = 0.0F;
    float roll = 0.0F;
    int rate_hz = 100;
    bool follow = false;
};

void printUsage()
{
    std::printf("用法: sim_controller [--status 0|5] [--bias 秒] [--yaw rad] [--pitch rad]\n"
                "                     [--rate Hz] [--follow]\n"
                "  --follow  把视觉回传的 target_yaw/target_pitch 当作下一帧的云台角\n"
                "            （只在相机真的随云台转动时使用；固定相机请保持关闭）\n");
}

bool parseOptions(int argc, char **argv, Options &options)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        const auto needValue = [&](const char *name) -> const char *
        {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "[sim] %s 需要一个参数\n", name);
                return nullptr;
            }
            return argv[++i];
        };
        if (arg == "--help" || arg == "-h")
        {
            printUsage();
            return false;
        }
        if (arg == "--follow")
        {
            options.follow = true;
            continue;
        }
        if (arg == "--status")
        {
            const char *value = needValue("--status");
            if (value == nullptr)
            {
                return false;
            }
            options.status = static_cast<std::uint8_t>(std::atoi(value));
            continue;
        }
        if (arg == "--bias")
        {
            const char *value = needValue("--bias");
            if (value == nullptr)
            {
                return false;
            }
            options.bias_s = static_cast<float>(std::atof(value));
            continue;
        }
        if (arg == "--yaw")
        {
            const char *value = needValue("--yaw");
            if (value == nullptr)
            {
                return false;
            }
            options.yaw = static_cast<float>(std::atof(value));
            continue;
        }
        if (arg == "--pitch")
        {
            const char *value = needValue("--pitch");
            if (value == nullptr)
            {
                return false;
            }
            options.pitch = static_cast<float>(std::atof(value));
            continue;
        }
        if (arg == "--rate")
        {
            const char *value = needValue("--rate");
            if (value == nullptr)
            {
                return false;
            }
            options.rate_hz = std::atoi(value);
            continue;
        }
        std::fprintf(stderr, "[sim] 未知参数: %s\n", arg.c_str());
        printUsage();
        return false;
    }
    if (options.rate_hz <= 0 || options.rate_hz > 1000)
    {
        std::fprintf(stderr, "[sim] --rate 需要在 1..1000 之间\n");
        return false;
    }
    return true;
}

// 把从端配置成裸模式：不要回显、不要行规程转换，否则二进制帧会被终端层改写。
void configureSlave(int slave_fd)
{
    termios settings{};
    if (::tcgetattr(slave_fd, &settings) != 0)
    {
        return;
    }
    ::cfmakeraw(&settings);
    settings.c_cc[VMIN] = 0;
    settings.c_cc[VTIME] = 0;
    ::tcsetattr(slave_fd, TCSANOW, &settings);
}

} // namespace

int main(int argc, char **argv)
{
    Options options;
    if (!parseOptions(argc, argv, options))
    {
        return 1;
    }

    int master_fd = -1;
    int slave_fd = -1;
    char slave_name[256] = {0};
    if (::openpty(&master_fd, &slave_fd, slave_name, nullptr, nullptr) != 0)
    {
        std::perror("[sim] openpty 失败");
        return 1;
    }
    configureSlave(slave_fd);

    std::signal(SIGINT, onSignal);
    std::signal(SIGTERM, onSignal);

    std::printf("[sim] 虚拟电控已就绪：我方%s status=%u bias=%.3fs 云台角 yaw=%.3f pitch=%.3f 帧率 %dHz\n",
                options.status == 5 ? "蓝" : "红", static_cast<unsigned>(options.status),
                static_cast<double>(options.bias_s), static_cast<double>(options.yaw),
                static_cast<double>(options.pitch), options.rate_hz);
    std::printf("[sim] 视觉端请运行: ./build/rmv_training %s 115200 config/camera.yaml 8080\n",
                slave_name);
    std::printf("[sim] Web Debug: http://127.0.0.1:8080  （Ctrl+C 退出）\n");

    const auto period = std::chrono::microseconds(1000000 / options.rate_hz);
    auto next_tick = std::chrono::steady_clock::now();

    Translator frame{};
    std::vector<char> rx;

    std::uint64_t sent = 0;
    std::uint64_t received = 0;
    std::uint64_t bad_frames = 0;
    std::uint16_t last_crc = 0xFFFF;
    std::uint8_t last_flag = 0xFF;
    std::uint8_t last_fire = 0xFF;
    auto last_print = std::chrono::steady_clock::now();

    // 视觉端打开从端之前，写入的数据可能丢失；这里不做重试，直接按帧率持续发送，
    // 视觉端一打开就能收到完整的帧流（它会自己等 head/tail 对齐）。
    while (!g_stop)
    {
        // ---------------- 发送一帧到视觉端 ----------------
        std::memset(&frame, 0, sizeof(frame));
        frame.message.head = kFrameHead;
        frame.message.tail = kFrameTail;
        frame.message.yaw = options.yaw;
        frame.message.pitch = options.pitch;
        frame.message.roll = options.roll;
        frame.message.status = options.status;
        frame.message.is_far = 0;
        frame.message.bias = options.bias_s;
        frame.message.pitch_offset = 0.0F;
        frame.message.wheel_w = 0.0F;

        // 写主端 -> 从端（视觉端）读到；读主端 <- 视觉端写到从端的数据。
        // slave_fd 只用来占住伪终端并配置裸模式，不读它，否则会和视觉端抢数据。
        const ssize_t written = ::write(master_fd, frame.data, kCommunicationFrameSize);
        if (written != static_cast<ssize_t>(kCommunicationFrameSize))
        {
            // 视觉端还没打开从端时主端写会失败（EIO/缓冲满），继续等它连上即可。
            if (written < 0 && (errno == EIO || errno == EAGAIN))
            {
                std::this_thread::sleep_until(next_tick);
                next_tick += period;
                continue;
            }
            std::fprintf(stderr, "\n[sim] 写入失败（视觉端可能已退出），停止。\n");
            break;
        }
        ++sent;

        // ---------------- 读取视觉端回传的帧 ----------------
        fd_set read_set;
        FD_ZERO(&read_set);
        FD_SET(master_fd, &read_set);
        timeval timeout{0, 0};
        if (::select(master_fd + 1, &read_set, nullptr, nullptr, &timeout) > 0)
        {
            char buffer[512];
            const ssize_t got = ::read(master_fd, buffer, sizeof(buffer));
            for (ssize_t i = 0; i < got; ++i)
            {
                rx.push_back(buffer[i]);
            }
        }
        while (rx.size() >= kCommunicationFrameSize)
        {
            const auto *raw = reinterpret_cast<const std::uint8_t *>(rx.data());
            if (raw[0] == kFrameHead && raw[kCommunicationFrameSize - 1] == kFrameTail)
            {
                Translator received_frame{};
                std::memcpy(received_frame.data, rx.data(), kCommunicationFrameSize);
                ++received;
                const MessData_AutoAim &message = received_frame.message;
                const bool changed = message.crc != last_crc || message.armor_flag != last_flag ||
                                     message.fire_allowance != last_fire;
                const auto now = std::chrono::steady_clock::now();
                const bool periodic =
                    std::chrono::duration_cast<std::chrono::milliseconds>(now - last_print).count() >= 500;
                if (changed || periodic)
                {
                    std::printf("\r[帧 %6llu] crc=%u armor_flag=%3u fire=%u 距离=%6.3fm "
                                "目标yaw=%7.3f pitch=%7.4f v=%.2f/%.2f 处理=%.1fms "
                                "中心=(%.0f,%.0f)mm   ",
                                static_cast<unsigned long long>(received),
                                static_cast<unsigned>(message.crc),
                                static_cast<unsigned>(message.armor_flag),
                                static_cast<unsigned>(message.fire_allowance),
                                static_cast<double>(message.distance),
                                static_cast<double>(message.target_yaw),
                                static_cast<double>(message.target_pitch),
                                static_cast<double>(message.yaw_vel),
                                static_cast<double>(message.pitch_vel),
                                static_cast<double>(message.latency),
                                static_cast<double>(message.coo_x),
                                static_cast<double>(message.coo_y));
                    std::fflush(stdout);
                    last_print = now;
                }
                if (options.follow && std::isfinite(message.target_yaw) &&
                    std::isfinite(message.target_pitch) && message.crc != 0)
                {
                    options.yaw = message.target_yaw;
                    options.pitch = message.target_pitch;
                }
                last_crc = message.crc;
                last_flag = message.armor_flag;
                last_fire = message.fire_allowance;
                rx.erase(rx.begin(), rx.begin() + static_cast<std::ptrdiff_t>(kCommunicationFrameSize));
            }
            else
            {
                ++bad_frames;
                rx.erase(rx.begin());
            }
        }

        next_tick += period;
        std::this_thread::sleep_until(next_tick);
    }

    std::printf("\n[sim] 结束：发出 %llu 帧，收到 %llu 帧，丢弃错位字节块 %llu 次\n",
                static_cast<unsigned long long>(sent), static_cast<unsigned long long>(received),
                static_cast<unsigned long long>(bad_frames));
    ::close(slave_fd);
    ::close(master_fd);
    return 0;
}
