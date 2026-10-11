#include "StudentTasks.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

/*
# 默认速度
./build/test_camera_playback tmp/red/video.avi --red

# 放慢 2 倍
./build/test_camera_playback tmp/red/video.avi --red --wait 66

# 放慢 3 倍
./build/test_camera_playback tmp/red/video.avi --red --wait 100

# 蓝色
./build/test_camera_playback tmp/blue/video.avi --blue

q / ESC：退出
空格：暂停/继续
← →：后退/前进 1 帧（自动暂停）
s：保存当前帧 + mask（保存到 ./tmp/）
*/
int main(int argc, char **argv)
{
    std::string video_path;
    int wait_ms = 33;

    TeamColor enemy = TeamColor::Red;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--red")
            enemy = TeamColor::Red;
        else if (arg == "--blue")
            enemy = TeamColor::Blue;
        else if (arg == "--wait" && i + 1 < argc)
            wait_ms = std::atoi(argv[++i]);
        else if (!arg.empty() && arg[0] != '-')
            video_path = arg;
        else
        {
            std::cerr << "未知参数: " << arg << "\n";
            return 2;
        }
    }

    if (video_path.empty())
    {
        std::cerr << "用法: " << argv[0]
                  << " <video.avi> [--red|--blue] [--wait N]\n";
        return 2;
    }
    std::cout << "视频: " << video_path << "  wait=" << wait_ms << " ms/帧\n";

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

    // ===== 打开视频 =====
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cerr << "无法打开视频: " << video_path << "\n";
        return 2;
    }
    std::cout << "视频总帧数: " << cap.get(cv::CAP_PROP_FRAME_COUNT)
              << "  FPS(元数据): " << cap.get(cv::CAP_PROP_FPS) << "\n";
    std::cout << "ESC/q quit, 空格 暂停/继续, ←→ 逐帧前后, s 保存帧\n";

    cv::Mat frame;
    int save_idx = 0;
    int frame_no = 0;
    bool paused = false;

    // ===== 逐帧 seek 相关 =====
    int seek_to = 0; // 0 = 不 seek；>0 = 目标帧号（1 计数）
    bool exit_flag = false;
    bool need_new_frame = true;

    // 方向键码——不同平台/后端不一样，全都认一下
    auto is_left_arrow = [](int k)
    {
        return k == 65361 || k == 16777234 || k == 2424832;
    };
    auto is_right_arrow = [](int k)
    {
        return k == 65363 || k == 16777236 || k == 2555904;
    };

    while (true)
    {
        // ===== 读取新帧（只在需要时）=====
        if (need_new_frame)
        {
            // 处理挂起的 seek
            if (seek_to > 0)
            {
                cap.set(cv::CAP_PROP_POS_FRAMES, seek_to - 1); // 转 0 索引
                frame_no = seek_to - 1;                        // 读取后会 ++ 到 seek_to
                seek_to = 0;
            }

            if (!cap.read(frame) || frame.empty())
            {
                std::cout << "视频播放结束（共 " << frame_no << " 帧）\n";
                break;
            }
            ++frame_no;
            need_new_frame = false;
        }

        // ===== 检测 + PnP =====
        // 注意：paused 且只有按键不 seek 时，会重复检测同一帧——
        // 对调试影响不大；如果嫌慢，可以用缓存优化。
        const auto dets = armor_detect(frame, enemy);
        const auto poses = armor_solve(dets, camera, gimbal);

        // ===== 画框 + 角点标签 =====
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

        // ===== PnP 信息（固定左上角）=====
        for (std::size_t pi = 0; pi < poses.size(); ++pi)
        {
            const auto &pose = poses[pi];
            const cv::Vec3d p = pose.position_camera_m;

            const double dist = cv::norm(p);
            const double pos_yaw = std::atan2(p[0], p[2]) * 180.0 / CV_PI;
            const double pitch = std::atan2(-p[1], std::hypot(p[0], p[2])) * 180.0 / CV_PI;
            const double armor_yaw = pose.armor_yaw * 180.0 / CV_PI;
            const char size_ch = (pose.detection.size == ArmorSize::Small) ? 'S' : 'L';

            char line1[128], line2[160], line3[160];
            std::snprintf(line1, sizeof(line1),
                          "ID=%d %c  dist=%.2fm  err=%.1fpx",
                          pose.detection.target_id, size_ch, dist,
                          pose.reprojection_error);
            std::snprintf(line2, sizeof(line2),
                          "pos=(%+.3f, %+.3f, %+.3f)m",
                          p[0], p[1], p[2]);
            std::snprintf(line3, sizeof(line3),
                          "posYaw=%+.1f  pitch=%+.1f  armYaw=%+.1f  [deg, cam frame]",
                          pos_yaw, pitch, armor_yaw);

            const cv::Point base(20, 80 + static_cast<int>(pi) * 90);
            const cv::Scalar color(0, 255, 255);
            cv::putText(frame, line1, base, cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1);
            cv::putText(frame, line2, base + cv::Point(0, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1);
            cv::putText(frame, line3, base + cv::Point(0, 44), cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1);
        }

        // ===== 状态栏 =====
        cv::putText(frame,
                    cv::format("Frame=%d/%d  Det=%zu  Poses=%zu  save=%d  [extr=ID]",
                               frame_no,
                               static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT)),
                               dets.size(), poses.size(), save_idx),
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 255), 2);

        // ===== mask 调试 =====
        cv::Mat hsv_dbg, mask_low, mask_high, mask_dbg;
        cv::cvtColor(frame, hsv_dbg, cv::COLOR_BGR2HSV);
        if (enemy == TeamColor::Red)
        {
            cv::inRange(hsv_dbg, cv::Scalar(0, 100, 80), cv::Scalar(10, 255, 255), mask_low);
            cv::inRange(hsv_dbg, cv::Scalar(160, 100, 80), cv::Scalar(180, 255, 255), mask_high);
            cv::bitwise_or(mask_low, mask_high, mask_dbg);
        }
        else
        {
            cv::inRange(hsv_dbg, cv::Scalar(100, 100, 80), cv::Scalar(130, 255, 255), mask_dbg);
        }

        // ===== 显示 =====
        cv::Mat display;
        if (frame.cols > 1920)
            cv::resize(frame, display, cv::Size(), 0.5, 0.5);
        else
            display = frame;
        if (paused)
        {
            cv::putText(display, "PAUSED",
                        cv::Point(display.cols / 2 - 100, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 1.5,
                        cv::Scalar(0, 0, 255), 3);
        }
        cv::imshow("playback", display);
        cv::imshow("mask", mask_dbg);

        // ===== 按键处理 =====
        // 用 waitKeyEx 拿到完整键码——方向键的高位不会丢
        const int raw_key = cv::waitKeyEx(paused ? 30 : wait_ms);

        // ① 方向键——必须在 & 0xFF 之前判断
        if (is_left_arrow(raw_key))
        {
            seek_to = std::max(1, frame_no - 1);
            paused = true;
            need_new_frame = true;
            continue;
        }
        if (is_right_arrow(raw_key))
        {
            seek_to = frame_no + 1;
            paused = true;
            need_new_frame = true;
            continue;
        }

        // ② 普通按键
        const int key = raw_key & 0xFF;
        if (key == 27 || key == 'q')
        {
            exit_flag = true;
        }
        else if (key == ' ')
        {
            paused = !paused;
        }
        else if (key == 's')
        {
            const std::string fname = cv::format("./tmp/playback_%03d.png", save_idx);
            const std::string mask_fname = cv::format("./tmp/playback_mask_%03d.png", save_idx);
            cv::imwrite(fname, frame);
            cv::imwrite(mask_fname, mask_dbg);
            std::cout << "已保存: " << fname << " 和 " << mask_fname << "\n";
            ++save_idx;
        }

        // ③ 决定下一轮是否读新帧
        if (exit_flag)
            break;
        if (!paused)
            need_new_frame = true;
    }

    cv::destroyAllWindows();
    return 0;
}