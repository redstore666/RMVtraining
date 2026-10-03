// 独立相机取流测试：不依赖串口，直接循环调用 get_pic() 验证 MVS SDK 链路。
#include "StudentTasks.hpp"

#include <chrono>
#include <iostream>
#include <thread>

#include <opencv2/imgcodecs.hpp>

int main()
{
    cv::Mat pic;
    int ok = 0, fail = 0;
    auto start = std::chrono::steady_clock::now();
    // 最多尝试 30 帧
    for (int i = 0; i < 30; ++i)
    {
        if (get_pic(pic) && !pic.empty())
        {
            ++ok;
            std::cout << "帧" << i << ": " << pic.cols << "x" << pic.rows
                      << " CV_8UC3=" << (pic.type() == CV_8UC3) << "\n";
            if (ok == 1)
                cv::imwrite("/tmp/first_frame.png", pic); // 保存首帧人工检查
        }
        else
        {
            ++fail;
            std::cout << "帧" << i << ": 取流失败\n";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - start)
                  .count();
    std::cout << "成功 " << ok << " / 失败 " << fail << "，耗时 " << ms << "ms\n";
    return ok > 0 ? 0 : 1; // 至少成功一帧才算通过
}
