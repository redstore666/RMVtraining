#pragma once

#include <string>

// web_debug/src/WebLogger.cpp 的副本，只给 test_camera_system 用（WEB_SYSTEM 目标）。
//
// 为什么要复制而不是直接用 web_debug：
//  * 两套端口各跑一个进程，生产程序（8080）那条 webdebug 一个字符都不改；
//  * 这里多一条"迷你俯视图"通道（GET /frame.jpg），按 30 Hz 覆盖写最新一帧 JPEG，
//    网页右下角用 <img> 直接取——不走 SSE（base64 塞进事件流会把 10 s 的历史缓冲
//    塞爆），也不轮询 JSON。
// 数值通道（SSE /data、/health、卡片折线）的行为与 web_debug 完全一致。
#ifdef WEB_SYSTEM

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "json.hpp"

namespace httplib {
class Server;
}

class SystemWebLogger {
public:
    static SystemWebLogger& getInstance();

    // 在后台线程启动 HTTP 服务。服务运行期间重复调用不会产生任何操作；
    // 默认监听本机地址。
    void start(int port = 8080, const std::string& host = "127.0.0.1");

    // 停止监听，并等待服务器线程和正在进行的 SSE 流退出。
    // 可以重复调用，之后也可以再次调用 start() 启动服务。
    void stop();

    // 向有容量上限的广播缓冲区加入一个数值样本。记录日志不会等待网页读取样本，
    // 内存占用始终受到限制。
    void log(const std::string& key, double value);

    bool isRunning() const noexcept;

    // 发布一帧 JPEG（覆盖上一帧）。服务没启动时直接丢弃，不占内存。
    void publishFrame(const std::vector<unsigned char>& jpeg);

    // 取最近一帧 JPEG 与其序号（序号用于网页侧判断"是不是同一帧"）。
    std::vector<unsigned char> latestFrame(std::uint64_t* sequence = nullptr) const;

    // ---- 网页端"选车"（本副本独有，调试特化）----
    // 网页 GET /select?id=N 写入；0 = 自动（让跟踪器自己挑目标）。
    // 工具每帧读 selectedTarget()，只把该 id 的观测喂给跟踪器，所以选了谁就一直看谁，
    // 丢了也不自动切到别的 id。
    void selectTarget(int id) noexcept;
    int selectedTarget() const noexcept;

    // 工具定期发布"累计见过的 id / 本帧可见的 id"，网页上的下拉菜单用它。
    // pinned / lost 由本类自己拼（pinned 来自 selectTarget），所以 /select 的响应
    // 立刻就带上新的选择，不用等工具下一轮发布。
    void publishTargets(const std::vector<int>& seen, const std::vector<int>& visible);
    std::string targetsJson() const;

    // 返回不会改变内部状态的快照，供诊断和测试使用。
    nlohmann::json getBufferJSON() const;

private:
    struct LogData {
        std::string key;
        double value;
        std::int64_t timestamp;
        std::uint64_t sequence;
    };

    // 所有客户端共享这段历史记录，但通过各自独立的序号读取。
    static constexpr std::size_t kBufferCapacity = 10000;

    SystemWebLogger();
    ~SystemWebLogger();
    SystemWebLogger(const SystemWebLogger&) = delete;
    SystemWebLogger& operator=(const SystemWebLogger&) = delete;

    void configureServer(httplib::Server& server);
    void serverLoop();

    std::atomic<bool> running_{false};
    mutable std::mutex lifecycle_mutex_;
    std::thread server_thread_;
    std::unique_ptr<httplib::Server> server_;

    mutable std::mutex buffer_mutex_;
    std::condition_variable data_cv_;
    std::deque<LogData> buffer_;
    std::uint64_t next_sequence_{1};

    // ---- 迷你俯视图通道（本副本独有）----
    // 只在服务启动后才覆盖写；网页取到的是"最近一帧"，不做历史。
    mutable std::mutex frame_mutex_;
    std::vector<unsigned char> frame_jpeg_;
    std::uint64_t frame_sequence_{0};

    std::atomic<int> selected_target_{0};
    mutable std::mutex targets_mutex_;
    std::vector<int> seen_ids_;
    std::vector<int> visible_ids_;
};

#define START_SYSTEM_WEB(port) SystemWebLogger::getInstance().start(port)
#define STOP_SYSTEM_WEB() SystemWebLogger::getInstance().stop()
#define SYSTEM_LOG(key, value) SystemWebLogger::getInstance().log((key), (value))
#define SYSTEM_FRAME(jpeg) SystemWebLogger::getInstance().publishFrame((jpeg))
// 网页上选了哪个 id（0 = 自动）。工具每帧读它来决定把哪些观测喂给跟踪器。
#define SYSTEM_TARGET() SystemWebLogger::getInstance().selectedTarget()
#define SYSTEM_TARGETS(...) SystemWebLogger::getInstance().publishTargets(__VA_ARGS__)
#define SYSTEM_TARGET_SET(id) SystemWebLogger::getInstance().selectTarget((id))

#else

// 这些宏会有意忽略传入参数。例如关闭调试时，SYSTEM_LOG("fps", expensive())
// 不会调用 expensive()。
#define START_SYSTEM_WEB(port) ((void)0)
#define STOP_SYSTEM_WEB() ((void)0)
#define SYSTEM_LOG(key, value) ((void)0)
#define SYSTEM_FRAME(jpeg) ((void)0)
#define SYSTEM_TARGET() 0
#define SYSTEM_TARGETS(...) ((void)0)
#define SYSTEM_TARGET_SET(id) ((void)0)

#endif // WEB_SYSTEM
