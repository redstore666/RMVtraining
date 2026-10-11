// 实时相机测试：取流 + 装甲板检测 + PnP 位姿显示（需连接相机）。
//
// 用法: ./build/test_camera_live [--red|--blue]
//
// 按键:  ESC/q 退出 | s 保存当前帧+掩码(./tmp/) | m 打印各观测 PnP 的 R/t 矩阵（全精度）
// 画面:  检测框 + 装甲板坐标系(x红/y绿/z蓝，板中心为原点)
//        + 左上角面板：每个观测的 编号/距离/重投影误差/角度 + PnP 的 R/t 矩阵
//        + 状态栏含【实时相机帧率 fps】（区别于视频回放的"视频帧号"）
// 控制台: 每 150 帧一行耗时/拒绝统计；每 300 帧一次重投影误差分布（均不刷屏）
#include "StudentTasks.hpp"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <opencv2/calib3d.hpp> // solvePnP / Rodrigues / drawFrameAxes
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>

namespace {
// 工具内自建物点（与 StudentTasks.cpp 的 buildObjectPoints 相同模型），
// 仅用于绘制装甲板坐标系：重解一次 PnP 拿 rvec/tvec，不影响 armor_solve 流程。
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

    std::cout << "ESC/q 退出, s 保存帧, m 打印 PnP 的 R/t 矩阵\n";
    std::cout << "[extr not calibrated] yaw/pitch 相对相机光轴\n";

    cv::Mat frame;
    int save_idx = 0;
    int stat_cnt = 0;
    int frame_cnt = 0;
    using Clock = std::chrono::steady_clock;
    auto ms = [](auto a, auto b)
    { return std::chrono::duration<double, std::milli>(b - a).count(); };

    // 实时相机帧率（EMA 平滑）与当前帧 PnP 位姿缓存（m 键用）
    double fps_ema = 0.0;
    Clock::time_point prev_grab_done{};
    std::vector<cv::Mat> dbg_rvecs, dbg_tvecs;
    std::vector<int> dbg_ids;

    while (true)
    {
        const auto t0 = Clock::now();
        if (!get_pic(frame))
        {
            continue;
        }
        const auto t1 = Clock::now();

        // 实时相机帧率 = 相邻两次成功取帧的间隔倒数（EMA 平滑）
        if (prev_grab_done.time_since_epoch().count() != 0)
        {
            const double dt_ms = ms(prev_grab_done, t1);
            if (dt_ms > 0.1)
            {
                const double fps_now = 1000.0 / dt_ms;
                fps_ema = (fps_ema <= 0.0) ? fps_now : 0.9 * fps_ema + 0.1 * fps_now;
            }
        }
        prev_grab_done = t1;

        // ===== 检测 + PnP =====
        const auto dets = armor_detect(frame, enemy);
        const auto t2 = Clock::now();
        const auto poses = armor_solve(dets, camera, gimbal);
        const auto t3 = Clock::now();

        // ===== 画检测框 + 角点标签 =====
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

        // ===== 每个 pose 画信息 + 装甲板坐标系（x 红 / y 绿 / z 蓝）=====
        dbg_rvecs.clear();
        dbg_tvecs.clear();
        dbg_ids.clear();
        for (std::size_t pi = 0; pi < poses.size(); ++pi)
        {
            const auto &pose = poses[pi];
            const cv::Vec3d p = pose.position_camera_m;

            // OpenCV 相机系: x 右、y 下、z 前
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
            const cv::Scalar color(0, 255, 255); // 黄
            cv::putText(frame, line1, base, cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1);
            cv::putText(frame, line2, base + cv::Point(0, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1);
            cv::putText(frame, line3, base + cv::Point(0, 44), cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 1);

            // 坐标系（重解 PnP 拿 rvec/tvec，仅供显示）
            const std::vector<cv::Point3f> obj = makeObjectPoints(pose.detection.size);
            const std::vector<cv::Point2f> img(pose.detection.corners.begin(),
                                               pose.detection.corners.end());
            cv::Mat rvec, tvec;
            if (cv::solvePnP(obj, img, camera.camera_matrix,
                             camera.distortion_coefficients, rvec, tvec, false,
                             cv::SOLVEPNP_IPPE))
            {
                cv::drawFrameAxes(frame, camera.camera_matrix,
                                  camera.distortion_coefficients, rvec, tvec, 0.08F, 2);
                dbg_rvecs.push_back(rvec.clone());
                dbg_tvecs.push_back(tvec.clone());
                dbg_ids.push_back(pose.detection.target_id);

                // ===== 左上角面板：PnP 的 R/t 矩阵 =====
                cv::Mat R;
                cv::Rodrigues(rvec, R);
                cv::putText(frame,
                            cv::format("R= [%+.3f %+.3f %+.3f]", R.at<double>(0, 0),
                                       R.at<double>(0, 1), R.at<double>(0, 2)),
                            base + cv::Point(0, 64), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                cv::putText(frame,
                            cv::format("    [%+.3f %+.3f %+.3f]", R.at<double>(1, 0),
                                       R.at<double>(1, 1), R.at<double>(1, 2)),
                            base + cv::Point(0, 82), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                cv::putText(frame,
                            cv::format("    [%+.3f %+.3f %+.3f]", R.at<double>(2, 0),
                                       R.at<double>(2, 1), R.at<double>(2, 2)),
                            base + cv::Point(0, 100), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
                cv::putText(frame,
                            cv::format("t= [%+.3f %+.3f %+.3f] m", tvec.at<double>(0),
                                       tvec.at<double>(1), tvec.at<double>(2)),
                            base + cv::Point(0, 120), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
            }
        }

        // ===== 状态栏（含实时相机帧率）=====
        cv::putText(frame,
                    cv::format("Det=%zu  Poses=%zu  save=%d  fps=%.1f  [extr=ID]",
                               dets.size(), poses.size(), save_idx, fps_ema),
                    cv::Point(20, 40),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 255), 2);

        // ===== mask 调试（真实流水线掩码：HSV+RGB补充+闭运算+填洞）=====
        const cv::Mat mask_dbg = enemyMaskForDebug(frame, enemy);

        // ===== 显示 =====
        cv::Mat display;
        if (frame.cols > 1920)
            cv::resize(frame, display, cv::Size(), 0.5, 0.5);
        else
            display = frame;

        cv::imshow("camera_live", display);
        cv::imshow("mask", mask_dbg);
        const auto t4 = Clock::now();

        // ===== 统计（精简输出：每 150 帧一行；err 分布每 300 帧）=====
        ++frame_cnt;
        if (frame_cnt % 150 == 0)
        {
            printf("[%d帧] fps=%.1f | grab=%.1f detect=%.1f pnp=%.1f draw=%.1f ms | "
                   "Det=%zu Poses=%zu | 拒绝 reproj=%d depth=%d fin=%d pnpFail=%d\n",
                   frame_cnt, fps_ema, ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4),
                   dets.size(), poses.size(),
                   g_rej_reproj, g_rej_depth, g_rej_finite, g_pnp_fail);
            g_rej_reproj = g_rej_depth = g_rej_finite = g_pnp_fail = 0;
        }
        if (++stat_cnt % 300 == 0 && !g_reproj_errors.empty())
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
        else if (key == 'm')
        {
            // 打印当前帧各观测的 PnP 位姿矩阵（R: 3x3 旋转矩阵，t: 3x1 平移，单位 m）
            std::cout << "===== 当前帧 PnP 位姿（" << dbg_tvecs.size() << " 个观测）=====\n";
            for (std::size_t di = 0; di < dbg_tvecs.size(); ++di)
            {
                cv::Mat R;
                cv::Rodrigues(dbg_rvecs[di], R);
                std::cout << "观测 " << di << "（ID=" << dbg_ids[di] << "）:\nR =\n"
                          << R << "\nt =\n" << dbg_tvecs[di].t() << "   (单位: m)\n";
            }
        }
    }

    release_camera();
    cv::destroyAllWindows();
    return 0;
}
