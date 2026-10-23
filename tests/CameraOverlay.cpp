#include "CameraOverlay.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp> // solvePnPGeneric / Rodrigues / projectPoints
#include <opencv2/imgproc.hpp>

#include "AimGeometry.hpp"

namespace rotating_armor
{
namespace
{

// 工具内自建物点（与 StudentTasks.cpp 的 buildObjectPoints 相同模型），
// 仅用于绘制装甲板坐标系：重解一次 PnP 拿 rvec/tvec，不影响 armor_solve 流程。
std::vector<cv::Point3f> makeObjectPoints(ArmorSize size)
{
    const float half_w = (size == ArmorSize::Small ? 0.135F : 0.230F) / 2.0F;
    const float half_h = 0.057F / 2.0F;
    return {{-half_w, -half_h, 0.0F},
            {half_w, -half_h, 0.0F},
            {half_w, half_h, 0.0F},
            {-half_w, half_h, 0.0F}};
}

// 仅用于调试显示：按目标 ID + 装甲板尺寸保存历史法向，避免不同目标共用上一帧状态。
// 蓝轴端点做轻微像素域平滑；这不会反馈到 armor_solve() 的实际位姿输出。
struct AxisDisplayHistory
{
    cv::Vec3d normal{0.0, 0.0, 1.0};
    bool has_normal{false};
    cv::Point2f blue_end{};
    bool has_blue_end{false};
    cv::Point2f image_center{};
    bool has_center{false};
    std::chrono::steady_clock::time_point last_seen{};
    bool has_last_seen{false};
};

bool solveAxisPoseForDisplay(const std::vector<cv::Point3f> &object_points,
                             const std::vector<cv::Point2f> &image_points,
                             const cv::Mat &camera_matrix,
                             const cv::Mat &distortion_coefficients,
                             int target_id, ArmorSize size,
                             cv::Mat &rvec, cv::Mat &tvec,
                             cv::Point2f &origin, cv::Point2f &x_end,
                             cv::Point2f &y_end, cv::Point2f &blue_end)
{
    using TrackKey = std::pair<int, int>;
    static std::map<TrackKey, AxisDisplayHistory> history_by_track;
    const TrackKey key{target_id, static_cast<int>(size)};
    AxisDisplayHistory &history = history_by_track[key];
    const auto now = std::chrono::steady_clock::now();
    cv::Point2f current_center(0.0F, 0.0F);
    for (const cv::Point2f &point : image_points)
        current_center += point * (1.0F / static_cast<float>(image_points.size()));

    // 目标消失、或同 ID/尺寸突然跳到画面另一处时，丢弃旧姿态历史。
    if ((history.has_last_seen &&
         now - history.last_seen > std::chrono::milliseconds(250)) ||
        (history.has_center && cv::norm(current_center - history.image_center) > 120.0))
    {
        history.has_normal = false;
        history.has_blue_end = false;
    }

    std::vector<cv::Mat> rvecs, tvecs;
    const int count = cv::solvePnPGeneric(object_points, image_points,
                                          camera_matrix, distortion_coefficients,
                                          rvecs, tvecs, false, cv::SOLVEPNP_IPPE);
    if (count <= 0)
        return false;

    const double inf = std::numeric_limits<double>::infinity();
    std::vector<double> errors(static_cast<std::size_t>(count), inf);
    std::vector<cv::Vec3d> normals(static_cast<std::size_t>(count), cv::Vec3d(0, 0, 1));
    int best_error_index = -1;
    for (int i = 0; i < count; ++i)
    {
        if (tvecs[i].at<double>(2) <= 0.0)
            continue; // 忽略落在相机后方的解

        std::vector<cv::Point2f> projected;
        cv::projectPoints(object_points, rvecs[i], tvecs[i], camera_matrix,
                          distortion_coefficients, projected);
        double sum = 0.0;
        for (std::size_t k = 0; k < projected.size(); ++k)
            sum += cv::norm(projected[k] - image_points[k]);
        errors[static_cast<std::size_t>(i)] = sum / std::max<std::size_t>(projected.size(), 1);

        cv::Mat rotation;
        cv::Rodrigues(rvecs[i], rotation);
        normals[static_cast<std::size_t>(i)] = cv::Vec3d(
            rotation.at<double>(0, 2), rotation.at<double>(1, 2), rotation.at<double>(2, 2));
        if (best_error_index < 0 || errors[static_cast<std::size_t>(i)] <
                                        errors[static_cast<std::size_t>(best_error_index)])
            best_error_index = i;
    }
    if (best_error_index < 0)
        return false;

    int pick = best_error_index;
    // 两个平面解的误差接近时，按该目标自己的历史法向选择连续解。
    if (count >= 2 && std::isfinite(errors[0]) && std::isfinite(errors[1]) &&
        std::fabs(errors[0] - errors[1]) <= 3.0)
    {
        if (history.has_normal)
            pick = normals[0].dot(history.normal) >= normals[1].dot(history.normal) ? 0 : 1;
        else
            pick = std::fabs(normals[0][1]) <= std::fabs(normals[1][1]) ? 0 : 1;
        if (!std::isfinite(errors[static_cast<std::size_t>(pick)]))
            pick = best_error_index;
    }

    rvec = rvecs[static_cast<std::size_t>(pick)].clone();
    tvec = tvecs[static_cast<std::size_t>(pick)].clone();

    // 将原点和三条轴端点投影到图像；坐标轴长度 8 cm，与旧版 drawFrameAxes 一致。
    const std::vector<cv::Point3f> axis_points = {
        {0.0F, 0.0F, 0.0F}, {0.08F, 0.0F, 0.0F}, {0.0F, 0.08F, 0.0F}, {0.0F, 0.0F, 0.08F}};
    std::vector<cv::Point2f> projected_axes;
    cv::projectPoints(axis_points, rvec, tvec, camera_matrix,
                      distortion_coefficients, projected_axes);
    if (projected_axes.size() != 4)
        return false;

    origin = projected_axes[0];
    x_end = projected_axes[1];
    y_end = projected_axes[2];
    cv::Point2f current_blue_end = projected_axes[3];

    // 只平滑调试画面上的蓝轴端点，不平滑/修改控制链路中的实际位姿。
    if (history.has_blue_end && history.has_last_seen &&
        now - history.last_seen <= std::chrono::milliseconds(250) &&
        cv::norm(current_blue_end - history.blue_end) < 80.0)
    {
        constexpr float current_weight = 0.35F;
        current_blue_end = history.blue_end * (1.0F - current_weight) +
                           current_blue_end * current_weight;
    }
    blue_end = current_blue_end;
    history.normal = normals[static_cast<std::size_t>(pick)];
    history.has_normal = true;
    history.blue_end = blue_end;
    history.has_blue_end = true;
    history.image_center = current_center;
    history.has_center = true;
    history.last_seen = now;
    history.has_last_seen = true;
    return true;
}

} // namespace

void drawCameraOverlay(cv::Mat &canvas, const CameraOverlayFrame &frame,
                       CameraOverlayDebug *debug)
{
    if (canvas.empty() || frame.camera == nullptr)
        return;

    static const std::vector<ArmorDetection> kNoDetections;
    static const std::vector<ArmorPose> kNoPoses;
    const std::vector<ArmorDetection> &dets =
        frame.detections != nullptr ? *frame.detections : kNoDetections;
    const std::vector<ArmorPose> &poses = frame.poses != nullptr ? *frame.poses : kNoPoses;
    const CameraParameters &camera = *frame.camera;
    const PredictionResult &prediction = frame.prediction;
    const TrackerDiagnostics &diagnostics = frame.diagnostics;

    // 图像坐标（角点、PnP 投影点）按 k 换算；面板位置与字号是"界面像素"，
    // 不随 k 变——画面缩小了，字不会跟着缩成蚂蚁。
    const double k = frame.scale > 1e-6 ? frame.scale : 1.0;
    const auto px = [k](const cv::Point2f &p)
    { return cv::Point(cvRound(p.x * k), cvRound(p.y * k)); };
    const auto wd = [k](int w) { return std::max(1, cvRound(w * k)); };
    const auto text = [&canvas](const std::string &str, cv::Point at,
                                const cv::Scalar &colour, double font, int thick)
    { cv::putText(canvas, str, at, cv::FONT_HERSHEY_SIMPLEX, font, colour, thick,
                  cv::LINE_AA); };

    // 半透明底板：画面暗/亮区域会让叠加字看不清，先压一层黑底再写字。
    const auto backdrop = [&canvas](cv::Rect rect)
    {
        rect &= cv::Rect(0, 0, canvas.cols, canvas.rows);
        if (rect.width <= 0 || rect.height <= 0)
            return;
        cv::Mat layer = canvas.clone();
        cv::rectangle(layer, rect, cv::Scalar(0, 0, 0), cv::FILLED);
        cv::addWeighted(layer, 0.62, canvas, 0.38, 0.0, canvas);
    };
    // PnP 面板贴着左下角往上排：画面中间的装甲板和角点标签在别处，
    // 这样多块板时也不会把检测框压住。
    const int pose_line_h = 20;
    const int pose_panel_h = 6 * pose_line_h + 24;
    const auto posePanelTop = [&](std::size_t index, std::size_t count)
    {
        const int rows = static_cast<int>(count);
        return canvas.rows - 12 - pose_panel_h * (rows - static_cast<int>(index));
    };

    backdrop(cv::Rect(6, 10, 620, 34)); // 状态栏
    for (std::size_t bi = 0; bi < poses.size(); ++bi) // 每块板的 PnP 面板
        backdrop(cv::Rect(6, posePanelTop(bi, poses.size()), 404, pose_panel_h));
    if (frame.ekf_enabled)
        backdrop(cv::Rect(std::max(8, canvas.cols - 462), 44, 456, 124)); // 右上 EKF 面板

    // ===== 画检测框 + 角点标签 =====
    for (const auto &det : dets)
    {
        std::vector<cv::Point> box;
        box.reserve(4);
        for (int ci = 0; ci < 4; ++ci)
            box.push_back(px(det.corners[ci]));
        cv::polylines(canvas, box, true, cv::Scalar(0, 255, 0), wd(2), cv::LINE_AA);
        cv::circle(canvas, box[0], wd(5), cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
        text(cv::format("ID=%d %c %.2f", det.target_id,
                        det.size == ArmorSize::Small ? 'S' : 'L', det.confidence),
             box[0] + cv::Point(0, -24), cv::Scalar(0, 255, 0), 0.6, 2);

        const char *labels[4] = {"TL", "TR", "BR", "BL"};
        const cv::Scalar colours[4] = {
            cv::Scalar(255, 0, 0),  // 蓝 = TL
            cv::Scalar(0, 255, 0),  // 绿 = TR
            cv::Scalar(0, 0, 255),  // 红 = BR
            cv::Scalar(0, 255, 255) // 黄 = BL
        };
        for (int ci = 0; ci < 4; ++ci)
        {
            cv::circle(canvas, box[ci], wd(4), colours[ci], -1, cv::LINE_AA);
            text(labels[ci], box[ci] + cv::Point(7, -7), colours[ci], 0.5, 1);
        }
    }

    // ===== 每个 pose 画信息 + 装甲板坐标系（x 红 / y 绿 / z 蓝）=====
    if (debug != nullptr)
    {
        debug->rvecs.clear();
        debug->tvecs.clear();
        debug->ids.clear();
    }
    for (std::size_t pi = 0; pi < poses.size(); ++pi)
    {
        const auto &pose = poses[pi];
        const char size_ch = (pose.detection.size == ArmorSize::Small) ? 'S' : 'L';

        const std::vector<cv::Point3f> obj = makeObjectPoints(pose.detection.size);
        const std::vector<cv::Point2f> img(pose.detection.corners.begin(),
                                           pose.detection.corners.end());
        cv::Mat rvec, tvec;
        cv::Point2f axis_origin, x_end, y_end, blue_end;
        const bool axis_ok = solveAxisPoseForDisplay(
            obj, img, camera.camera_matrix, camera.distortion_coefficients,
            pose.detection.target_id, pose.detection.size,
            rvec, tvec, axis_origin, x_end, y_end, blue_end);

        // 显示面板尽可能使用与当前可视坐标轴相同的解，避免 R/t 和轴来自不同的 IPPE 解。
        cv::Vec3d p = pose.position_camera_m;
        double display_reproj_error = pose.reprojection_error;
        double armor_yaw = pose.armor_yaw;
        cv::Mat R;
        if (axis_ok)
        {
            p = cv::Vec3d(tvec);
            cv::Rodrigues(rvec, R);
            const cv::Vec3d normal(R.at<double>(0, 2), R.at<double>(1, 2),
                                   R.at<double>(2, 2));
            armor_yaw = std::atan2(normal[0], normal[2]);

            std::vector<cv::Point2f> projected;
            cv::projectPoints(obj, rvec, tvec, camera.camera_matrix,
                              camera.distortion_coefficients, projected);
            double err_sq = 0.0;
            for (std::size_t ci = 0; ci < projected.size(); ++ci)
            {
                const double e = cv::norm(projected[ci] - img[ci]);
                err_sq += e * e;
            }
            if (!projected.empty())
                display_reproj_error = std::sqrt(err_sq / projected.size());

            // BGR：x 轴红、y 轴绿、z 轴蓝。
            cv::line(canvas, px(axis_origin), px(x_end), cv::Scalar(0, 0, 255), wd(2),
                     cv::LINE_AA);
            cv::line(canvas, px(axis_origin), px(y_end), cv::Scalar(0, 255, 0), wd(2),
                     cv::LINE_AA);
            cv::line(canvas, px(axis_origin), px(blue_end), cv::Scalar(255, 0, 0), wd(2),
                     cv::LINE_AA);

            if (debug != nullptr)
            {
                debug->rvecs.push_back(rvec.clone());
                debug->tvecs.push_back(tvec.clone());
                debug->ids.push_back(pose.detection.target_id);
            }
        }

        // OpenCV 相机系: x 右、y 下、z 前
        const double dist = cv::norm(p);
        const double pos_yaw = std::atan2(p[0], p[2]) * 180.0 / CV_PI;
        const double pitch = std::atan2(-p[1], std::hypot(p[0], p[2])) * 180.0 / CV_PI;
        armor_yaw *= 180.0 / CV_PI;

        char line1[128], line2[160], line3[160];
        std::snprintf(line1, sizeof(line1),
                      "ID=%d %c  dist=%.2fm  err=%.1fpx",
                      pose.detection.target_id, size_ch, dist, display_reproj_error);
        std::snprintf(line2, sizeof(line2),
                      "pos=(%+.3f, %+.3f, %+.3f)m",
                      p[0], p[1], p[2]);
        std::snprintf(line3, sizeof(line3),
                      "posYaw=%+.1f  pitch=%+.1f  armYaw=%+.1f",
                      pos_yaw, pitch, armor_yaw);

        const int line_h = pose_line_h;
        const cv::Point base(14, posePanelTop(pi, poses.size()) + 26);
        const cv::Scalar color(0, 255, 255); // 黄
        text(line1, base, color, 0.58, 1);
        text(line2, base + cv::Point(0, line_h), color, 0.58, 1);
        text(line3, base + cv::Point(0, 2 * line_h), color, 0.58, 1);

        if (axis_ok)
        {
            text(cv::format("R= [%+.3f %+.3f %+.3f]", R.at<double>(0, 0),
                            R.at<double>(0, 1), R.at<double>(0, 2)),
                 base + cv::Point(0, 3 * line_h), color, 0.52, 1);
            text(cv::format("    [%+.3f %+.3f %+.3f]", R.at<double>(1, 0),
                            R.at<double>(1, 1), R.at<double>(1, 2)),
                 base + cv::Point(0, 4 * line_h), color, 0.52, 1);
            text(cv::format("    [%+.3f %+.3f %+.3f]", R.at<double>(2, 0),
                            R.at<double>(2, 1), R.at<double>(2, 2)),
                 base + cv::Point(0, 5 * line_h), color, 0.52, 1);
            text(cv::format("t= [%+.3f %+.3f %+.3f] m", tvec.at<double>(0),
                            tvec.at<double>(1), tvec.at<double>(2)),
                 base + cv::Point(0, 6 * line_h), color, 0.52, 1);
        }
    }

    // ===== EKF 叠加：车心 + 预测瞄准点 + 状态文本 =====
    if (frame.ekf_enabled)
    {
        const auto project_gimbal_point =
            [&](const Eigen::Vector3d &p_gimbal, cv::Point2f &pixel) -> bool
        {
            const cv::Vec3d p_camera = gimbalToCameraPoint(
                camera, cv::Vec3d(p_gimbal.x(), p_gimbal.y(), p_gimbal.z()));
            if (!(p_camera[2] > 1e-3))
                return false; // 落在相机后方，投影没有意义
            std::vector<cv::Point2d> projected;
            cv::projectPoints(
                std::vector<cv::Point3d>{cv::Point3d(p_camera[0], p_camera[1], p_camera[2])},
                cv::Vec3d(0.0, 0.0, 0.0), cv::Vec3d(0.0, 0.0, 0.0),
                camera.camera_matrix, camera.distortion_coefficients, projected);
            if (projected.size() != 1 || !std::isfinite(projected[0].x) ||
                !std::isfinite(projected[0].y))
                return false;
            pixel = cv::Point2f(static_cast<float>(projected[0].x),
                                static_cast<float>(projected[0].y));
            return true;
        };

        // 车体中心：状态排布 [x, vx, y, vy, z, vz, theta, omega, R]，中心取 (0, 2, 4)。
        // z 再补槽位高度偏移均值——state(4) 会被"最近更新的那块板"带得上下跳。
        if (frame.has_state)
        {
            const State &s = frame.state;
            const Eigen::Vector3d c_world(s(0), s(2), s(4) + frame.slot_z_offset);
            const Eigen::Vector3d c_gimbal = toGimbal(frame.gimbal, c_world);
            cv::Point2f center_px;
            if (project_gimbal_point(c_gimbal, center_px))
            {
                const cv::Point p = px(center_px);
                cv::drawMarker(canvas, p, cv::Scalar(255, 0, 255), cv::MARKER_CROSS, 24, 2);
                text(cv::format("body R=%.2f m", s(8)), p + cv::Point(16, -16),
                     cv::Scalar(255, 0, 255), 0.58, 1);
            }
        }

        // 预测瞄准点（含提前量与飞行时间）：直接用 EKF 给出的世界系瞄准点，
        // 不再由 yaw/pitch/distance 反算，避免两处公式不一致。
        if (prediction.state != TrackingState::NoTarget)
        {
            const Eigen::Vector3d aim_gimbal =
                toGimbal(frame.gimbal, diagnostics.predicted_aim_world);
            cv::Point2f aim_px;
            if (project_gimbal_point(aim_gimbal, aim_px))
            {
                const cv::Point p = px(aim_px);
                cv::drawMarker(canvas, p, cv::Scalar(0, 255, 255), cv::MARKER_TILTED_CROSS,
                               22, 2);
                text(cv::format("AIM %.2f m", prediction.distance_m),
                     p + cv::Point(16, -16), cv::Scalar(0, 255, 255), 0.58, 1);
            }
        }

        // 右上角文本：锁定状态 / 车体参数 / 云台还差多少 / 开火判据
        const char *state_text = prediction.state == TrackingState::Stable ? "Stable"
                               : prediction.state == TrackingState::Unstable ? "Unstable"
                                                                             : "NoTarget";
        const int panel_x = std::max(14, canvas.cols - 452);
        const int panel_y = 62;
        const cv::Scalar ekf_color =
            prediction.fire_allowed ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 200, 255);
        const std::string ekf_lines[] = {
            cv::format("EKF %s  id=%d  R=%.2fm  w=%+.1f rad/s", state_text,
                       static_cast<int>(prediction.target_id), prediction.radius_1_m,
                       prediction.body_yaw_velocity),
            // 分成两行：合成一行会顶到画面右缘被截断（putText 不换行）。
            cv::format("aim slot=%d  d=%.2fm", diagnostics.aim_slot, prediction.distance_m),
            cv::format("yaw=%+.1f  pitch=%+.1f deg",
                       prediction.target_yaw * 180.0 / CV_PI,
                       prediction.target_pitch * 180.0 / CV_PI),
            cv::format("NIS=%.1f  upd=%d  miss=%d  lat=%.0fms  obs=%.0fmm",
                       diagnostics.last_nis, diagnostics.updates_this_frame,
                       diagnostics.missed_frames,
                       diagnostics.measured_latency_s * 1000.0,
                       diagnostics.observation_noise_m * 1000.0),
            // 注意：本机没有 IMU，gimbal.yaw/pitch 恒为 0，所以这里的
            // fire/miss 是"相对相机光轴"的量，不是实车上的云台脱靶量。
            cv::format("fire=%s  miss=%.1fcm  budget=%.2fdeg",
                       prediction.fire_allowed ? "YES" : "no",
                       prediction.fire_miss_m * 100.0,
                       prediction.fire_budget_rad * 180.0 / CV_PI),
            // putText 只画 ASCII，中文注释留在源码里。
            std::string("(no IMU: fire/miss vs boresight)"),
        };
        for (int li = 0; li < 6; ++li)
            text(ekf_lines[li], cv::Point(panel_x, panel_y + li * 21), ekf_color, 0.56, 1);
    }

    // ===== 左上状态栏（最后画，保证压在所有东西上面）=====
    std::string status = frame.status_text;
    if (!frame.fps_text.empty())
    {
        if (!status.empty())
            status += "  ";
        status += frame.fps_text;
    }
    if (!status.empty())
        text(status, cv::Point(14, 32), cv::Scalar(0, 255, 255), 0.75, 2);
}

} // namespace rotating_armor
