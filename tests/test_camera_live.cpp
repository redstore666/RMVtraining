#include "StudentTasks.hpp"
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

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

    std::cout << "按 ESC 或 q 退出，按 s 保存当前帧 + mask\n";

    cv::Mat frame;
    int save_idx = 0;

    while (true)
    {
        if (!get_pic(frame))
            continue;

        // ===== 检测 + 画框 =====
        auto dets = armor_detect(frame, enemy);
        drawDetections(frame, dets);

        // ===== 手工算 mask（调试用，逻辑和 buildEnemyMask 对齐）=====
        cv::Mat hsv_dbg, mask_low, mask_high, mask_dbg;
        cv::cvtColor(frame, hsv_dbg, cv::COLOR_BGR2HSV);
        cv::inRange(hsv_dbg, cv::Scalar(0, 100, 80), cv::Scalar(10, 255, 255), mask_low);
        cv::inRange(hsv_dbg, cv::Scalar(160, 100, 80), cv::Scalar(180, 255, 255), mask_high);
        cv::bitwise_or(mask_low, mask_high, mask_dbg);

        // ===== 左上角叠加信息 =====
        cv::putText(frame,
                    cv::format("Detections: %zu  |  save_idx: %d", dets.size(), save_idx),
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(0, 255, 255), 2);

        // ===== 显示：原图 + mask 并排 =====
        cv::Mat display;
        if (frame.cols > 1920)
            cv::resize(frame, display, cv::Size(), 0.5, 0.5);
        else
            display = frame;

        cv::imshow("camera_live", display);
        cv::imshow("mask", mask_dbg); // 单独一个窗口显示掩码

        // ===== 按键处理 =====
        const int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q')
            break;
        if (key == 's')
        {
            const std::string fname = cv::format("./tmp/live_%03d.png", save_idx);
            const std::string mask_fname = cv::format("./tmp/mask_%03d.png", save_idx);
            cv::imwrite(fname, frame);
            cv::imwrite(mask_fname, mask_dbg);
            std::cout << "已保存: " << fname << " 和 " << mask_fname << "\n";
            ++save_idx;
        }
    }

    release_camera();
    cv::destroyAllWindows();
    return 0;
}