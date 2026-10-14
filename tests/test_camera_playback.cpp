#include "StudentTasks.hpp"
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <opencv2/calib3d.hpp> // solvePnP / Rodrigues / drawFrameAxes（坐标系显示用）
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
m：打印当前帧各观测 PnP 的 R / t 矩阵（全精度；左上角面板常显 %.3f 版本）
画面：检测框 + 装甲板中心 xyz 坐标系（x红 y绿 z蓝）+ 左上角面板（坐标轴/矩阵/帧号）
*/

namespace {
// 工具内自建物点（与 StudentTasks.cpp 的 buildObjectPoints 相同模型），
// 仅用于在画面上绘制装甲板坐标系：重解一次 PnP 拿 rvec/tvec，不影响 armor_solve。
std::vector<cv::Point3f> makeObjectPoints(ArmorSize size)
{
    const float half_w = (size == ArmorSize::Small ? 0.135F : 0.230F) / 2.0F;
    const float half_h = 0.057F / 2.0F;
    return {{-half_w, -half_h, 0.0F}, {half_w, -half_h, 0.0F},
            {half_w, half_h, 0.0F}, {-half_w, half_h, 0.0F}};
}
} // namespace
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
    std::cout << "ESC/q quit, 空格 暂停/继续, ←→ 逐帧前后, s 保存帧, k 掩码窗口 原始/正则化\n";

    cv::Mat frame;
    int save_idx = 0;
    int frame_no = 0;
    bool paused = false;

    // ===== 逐帧 seek 相关 =====
    int seek_to = 0; // 0 = 不 seek；>0 = 目标帧号（1 计数）
    bool exit_flag = false;
    bool need_new_frame = true;

    // 方向键码——不同平台/后端不一样，全都认一下；还认不出来时按 ←/→ 看终端
    // 打印的 "key = ..." 键码，把值补进下面的列表（或补进低位兜底分支）。
    auto is_left_arrow = [](int k)
    {
        return k == 65361 || k == 16777234 || k == 2424832 ||
               ((k & 0xFF) == 0x51 && k != 0x51) ||        // GTK 高位变体
               ((k & 0xFF) == 0x12 && k > 0x10000);        // Qt 0x1000012 低位兜底
    };
    auto is_right_arrow = [](int k)
    {
        return k == 65363 || k == 16777236 || k == 2555904 ||
               ((k & 0xFF) == 0x53 && k != 0x53) ||        // GTK 高位变体
               ((k & 0xFF) == 0x14 && k > 0x10000);        // Qt 0x1000014 低位兜底
    };

    // 当前帧的 PnP 位姿缓存（供 m 键打印矩阵用）
    std::vector<cv::Mat> dbg_rvecs, dbg_tvecs;
    std::vector<int> dbg_ids;

    // 每帧的检测结果缓存（只在读入新帧时刷新一次）
    std::vector<ArmorDetection> dets;
    std::vector<ArmorPose> poses;
    bool need_recompute = true;

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
            need_recompute = true;
        }

        // ===== 检测 + PnP + 掩码：只在新帧时计算一次 =====
        // 重要（修复"回放误识别"）：绘制只发生在克隆图上（display_canvas），
        // 绝不回写原始 frame——否则暂停/逐帧时，下一轮循环会把上一轮画上去的
        // 检测框、角点、坐标轴、PAUSED 字样当成图像内容重新检测，
        // 产生大量假灯条/假 ID/跨条配对（历史 bug：屏幕上"全是误识别"）。
        if (need_recompute)
        {
            dets = armor_detect(frame, enemy);
            poses = armor_solve(dets, camera, gimbal);

            // PnP 位姿（供坐标轴绘制与 m 键矩阵打印）
            dbg_rvecs.clear();
            dbg_tvecs.clear();
            dbg_ids.clear();
            for (const auto &pose : poses)
            {
                const std::vector<cv::Point3f> obj = makeObjectPoints(pose.detection.size);
                const std::vector<cv::Point2f> img(pose.detection.corners.begin(),
                                                   pose.detection.corners.end());
                cv::Mat rvec, tvec;
                if (solveArmorPoseDisambiguated(obj, img, camera.camera_matrix,
                                                camera.distortion_coefficients, rvec, tvec))
                {
                    dbg_rvecs.push_back(rvec.clone());
                    dbg_tvecs.push_back(tvec.clone());
                    dbg_ids.push_back(pose.detection.target_id);
                }
            }
            need_recompute = false;
        }

        // ===== 绘制（全部画在克隆图上，frame 保持原始像素）=====
        cv::Mat display_canvas = frame.clone();

        // 检测框 + 角点标签
        drawDetections(display_canvas, dets);
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
                cv::circle(display_canvas, pt, 4, colors[k], -1);
                cv::putText(display_canvas, labels[k],
                            pt + cv::Point(6, -6),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            colors[k], 1);
            }
        }

        // PnP 信息（固定左上角）
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

            const cv::Point base(20, 80 + static_cast<int>(pi) * 150);
            const cv::Scalar color(0, 255, 255);
            cv::putText(display_canvas, line1, base, cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1);
            cv::putText(display_canvas, line2, base + cv::Point(0, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1);
            cv::putText(display_canvas, line3, base + cv::Point(0, 44), cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1);

            // ===== 左上角面板：PnP 的 R/t 矩阵（rvec/tvec 为同一帧的缓存解算结果）=====
            if (pi < dbg_rvecs.size())
            {
                cv::Mat R;
                cv::Rodrigues(dbg_rvecs[pi], R);
                const cv::Mat &tvec = dbg_tvecs[pi];
                cv::putText(display_canvas,
                            cv::format("R= [%+.3f %+.3f %+.3f]", R.at<double>(0, 0),
                                       R.at<double>(0, 1), R.at<double>(0, 2)),
                            base + cv::Point(0, 64), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                cv::putText(display_canvas,
                            cv::format("    [%+.3f %+.3f %+.3f]", R.at<double>(1, 0),
                                       R.at<double>(1, 1), R.at<double>(1, 2)),
                            base + cv::Point(0, 82), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                cv::putText(display_canvas,
                            cv::format("    [%+.3f %+.3f %+.3f]", R.at<double>(2, 0),
                                       R.at<double>(2, 1), R.at<double>(2, 2)),
                            base + cv::Point(0, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                cv::putText(display_canvas,
                            cv::format("t= [%+.3f %+.3f %+.3f] m", tvec.at<double>(0),
                                       tvec.at<double>(1), tvec.at<double>(2)),
                            base + cv::Point(0, 120), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            }
        }

        // 装甲板坐标系（板中心为原点，学长要求）：x 红 / y 绿 / z 蓝
        for (std::size_t di = 0; di < dbg_tvecs.size(); ++di)
        {
            cv::drawFrameAxes(display_canvas, camera.camera_matrix,
                              camera.distortion_coefficients,
                              dbg_rvecs[di], dbg_tvecs[di], 0.08F, 2);
        }

        // 状态栏
        cv::putText(display_canvas,
                    cv::format("Frame=%d/%d  Det=%zu  Poses=%zu  save=%d  [extr=ID]",
                               frame_no,
                               static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT)),
                               dets.size(), poses.size(), save_idx),
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 255), 2);

        // ===== 显示 =====
        cv::Mat display;
        if (display_canvas.cols > 1920)
            cv::resize(display_canvas, display, cv::Size(), 0.5, 0.5);
        else
            display = display_canvas;
        if (paused)
        {
            cv::putText(display, "PAUSED",
                        cv::Point(display.cols / 2 - 100, 60),
                        cv::FONT_HERSHEY_SIMPLEX, 1.5,
                        cv::Scalar(0, 0, 255), 3);
        }
        cv::imshow("playback", display);

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
        else if (key == 'm')
        {
            // 打印当前帧各观测的 PnP 位姿矩阵（R: 3x3 旋转矩阵，t: 3x1 平移，单位 m）
            std::cout << "===== Frame " << frame_no << "  PnP 位姿（"
                      << dbg_tvecs.size() << " 个观测）=====\n";
            for (std::size_t di = 0; di < dbg_tvecs.size(); ++di)
            {
                cv::Mat R;
                cv::Rodrigues(dbg_rvecs[di], R);
                std::cout << "观测 " << di << "（ID=" << dbg_ids[di] << "）:\nR =\n"
                          << R << "\nt =\n" << dbg_tvecs[di].t() << "   (单位: m)\n";
            }
        }
        else if (key == 's')
        {
            const std::string fname = cv::format("./tmp/playback_%03d.png", save_idx);
            cv::imwrite(fname, display_canvas); // 保存带标注的克隆图（原始帧不含标注）
            std::cout << "已保存: " << fname << "\n";
            ++save_idx;
        }
        else if (raw_key > 0 && raw_key != -1 && raw_key != 255)
        {
            // 未识别的键码打印出来（帮助补全方向键列表）
            printf("key = %d (0x%x)\n", raw_key, raw_key);
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