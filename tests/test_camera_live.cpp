#include "StudentTasks.hpp"
#include <iostream>
#include <cmath>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>    // test
#include <algorithm> //test

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

    std::cout << "ESC/q quit, s save frame\n";
    std::cout << "[extr not calibrated] yaw/pitch are relative to camera optical axis\n";

    cv::Mat frame;
    int save_idx = 0;
    int stat_cnt = 0;                        // test the average err
    using Clock = std::chrono::steady_clock; // test the average err
    auto ms = [](auto a, auto b)             // test the average err
    {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    while (true)
    {
        auto t0 = Clock::now(); // test
        if (!get_pic(frame))
            continue;
        auto t1 = Clock::now(); // test
        // ===== 检测 + PnP =====
        const auto dets = armor_detect(frame, enemy);
        auto t2 = Clock::now(); // test
        const auto poses = armor_solve(dets, camera, gimbal);
        auto t3 = Clock::now(); // test
        drawDetections(frame, dets);
        // test 每个检测画 4 个带标签的角点
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

        // ===== 每个 pose 画多行完整信息 =====
        for (const auto &pose : poses)
        {
            const cv::Point2f tl = pose.detection.corners[0];
            const cv::Vec3d p = pose.position_camera_m;

            // OpenCV 相机系: x 右、y 下、z 前
            const double dist = cv::norm(p);
            const double pos_yaw = std::atan2(p[0], p[2]) * 180.0 / CV_PI;
            const double pitch = std::atan2(-p[1], std::hypot(p[0], p[2])) * 180.0 / CV_PI;
            const double armor_yaw = pose.armor_yaw * 180.0 / CV_PI;

            const char size_ch = (pose.detection.size == ArmorSize::Small) ? 'S' : 'L';

            // 第 1 行：ID / 大小 / 距离 / 重投影误差
            char line1[128];
            std::snprintf(line1, sizeof(line1),
                          "ID=%d %c  dist=%.2fm  err=%.1fpx",
                          pose.detection.target_id, size_ch, dist,
                          pose.reprojection_error);

            // 第 2 行：相机系位置三分量
            char line2[160];
            std::snprintf(line2, sizeof(line2),
                          "pos=(%+.3f, %+.3f, %+.3f)m",
                          p[0], p[1], p[2]);

            // 第 3 行：角度
            char line3[160];
            std::snprintf(line3, sizeof(line3),
                          "posYaw=%+.1f  pitch=%+.1f  armYaw=%+.1f  [deg, cam frame]",
                          pos_yaw, pitch, armor_yaw);

            const cv::Point base(static_cast<int>(tl.x) - 10,
                                 static_cast<int>(tl.y) - 60);
            const cv::Scalar color(0, 255, 255); // 黄
            const double font = 0.55;
            const int thick = 1;

            cv::putText(frame, line1, base, cv::FONT_HERSHEY_SIMPLEX, font, color, thick);
            cv::putText(frame, line2, base + cv::Point(0, 22), cv::FONT_HERSHEY_SIMPLEX, font, color, thick);
            cv::putText(frame, line3, base + cv::Point(0, 44), cv::FONT_HERSHEY_SIMPLEX, font, color, thick);
        }

        // ===== 左上角状态栏（英文）=====
        cv::putText(frame,
                    cv::format("Det=%zu  Poses=%zu  save=%d  [extr=ID]",
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

        cv::imshow("camera_live", display);
        cv::imshow("mask", mask_dbg);
        auto t4 = Clock::now(); // test

        // 每 30 帧打印一次，避免刷屏
        static int cnt = 0;  // test
        if (++cnt % 30 == 0) // test
        {
            printf("grab=%.1f detect=%.1f pnp=%.1f draw=%.1f ms | Det=%zu Poses=%zu | "
                   "rejErr=%d rejDep=%d rejFin=%d pnpFail=%d\n",
                   ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4),
                   dets.size(), poses.size(),
                   g_rej_reproj, g_rej_depth, g_rej_finite, g_pnp_fail);

            g_rej_reproj = g_rej_depth = g_rej_finite = g_pnp_fail = 0;
        }
        if (++stat_cnt % 100 == 0 && !g_reproj_errors.empty()) // test
        {
            std::vector<double> tmp = g_reproj_errors;
            std::sort(tmp.begin(), tmp.end());
            auto pct = [&](double p)
            {
                std::size_t i = static_cast<std::size_t>(p * tmp.size());
                if (i >= tmp.size())
                    i = tmp.size() - 1;
                return tmp[i];
            };

            printf("err分布: P50=%.1f  P90=%.1f  P95=%.1f  P99=%.1f  max=%.1f  (n=%zu)\n",
                   pct(0.50), pct(0.90), pct(0.95), pct(0.99), tmp.back(), tmp.size());
            g_reproj_errors.clear();
        }

        // ===== 按键 =====
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