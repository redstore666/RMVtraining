// FFV1 无损视频录制
// 用法: ./build/test_camera_record --red
// 按键: s 开始/停止录制, q/ESC 退出
// 输出: ./tmp/record_YYYYMMDD_HHMMSS/video.avi
#include "StudentTasks.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp> // VideoWriter

int main(int argc, char **argv)
{
    TeamColor enemy = TeamColor::Red;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--red")
            enemy = TeamColor::Red;
        else if (arg == "--blue")
            enemy = TeamColor::Blue;
        else
        {
            std::cerr << "未知参数: " << arg << "\n";
            return 2;
        }
    }

    std::cout << "按 s 开始/停止录制, 按 q/ESC 退出\n";

    cv::Mat frame;
    cv::VideoWriter writer;
    bool recording = false;
    int frame_idx = 0;
    std::string out_dir;
    std::string video_path;

    while (true)
    {
        if (!get_pic(frame))
            continue;

        if (recording && writer.isOpened())
        {
            writer.write(frame);
            ++frame_idx;
        }

        // 显示
        cv::Mat display = frame.clone();
        if (recording)
        {
            cv::circle(display, cv::Point(40, 40), 12, cv::Scalar(0, 0, 255), -1);
            cv::putText(display, "FFV1 REC  " + std::to_string(frame_idx) + " frames",
                        cv::Point(70, 50), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                        cv::Scalar(0, 0, 255), 2);
        }
        cv::imshow("record_ffv1", display);

        const int key = cv::waitKey(30) & 0xFF;
        if (key == 27 || key == 'q')
            break;

        if (key == 's')
        {
            if (!recording)
            {
                // 生成输出目录
                const auto now = std::chrono::system_clock::now();
                const auto t = std::chrono::system_clock::to_time_t(now);
                std::tm tm = *std::localtime(&t);
                std::ostringstream oss;
                oss << "./tmp/record_" << std::put_time(&tm, "%Y%m%d_%H%M%S");
                out_dir = oss.str();
                system(("mkdir -p " + out_dir).c_str());
                video_path = out_dir + "/video.avi";

                // ===== 打开 FFV1 编码器 =====
                const int fourcc = cv::VideoWriter::fourcc('F', 'F', 'V', '1');

                writer.open(video_path, cv::CAP_FFMPEG, fourcc,
                            30, // fps（与相机一致）
                            cv::Size(frame.cols, frame.rows),
                            true); // isColor = true

                if (!writer.isOpened())
                {
                    std::cerr << "FFV1 编码器打开失败（OpenCV 可能没编 FFmpeg）\n";
                    writer.release();
                    return 2;
                }

                frame_idx = 0;
                recording = true;
                std::cout << "开始录制 (FFV1) → " << video_path << "\n";
            }
            else
            {
                recording = false;
                writer.release();
                std::cout << "录制结束: " << frame_idx << " 帧 → " << video_path << "\n";
            }
        }
    }

    if (recording)
        writer.release();
    release_camera();
    cv::destroyAllWindows();
    return 0;
}