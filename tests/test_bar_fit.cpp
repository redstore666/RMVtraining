// 灯条约束拟合（fitBarConstrained，需求③）离线验证工具。
//
// 用法:
//   ./build/test_bar_fit <视频> [--red|--blue] [--max-frames N] [--range A B]
//       进程内 A/B：同一帧、同一掩码下，几何估计"关（基线 minAreaRect）/ 开（约束拟合）"
//       各跑一次 armor_detect，统计灯条级与角点级的帧间抖动、检出与耗时。
//       "on" 槽 = 拟合 v3（四边约束，默认）/ --fit-v1、--fit-v2 可切 v1（矩形）、v2（四边独立求交）。
//       --range A B：只统计 [A, B] 帧区间（与既有实验的静态段口径对齐）。
//   ./build/test_bar_fit <视频> [--red|--blue] --pose [--max-frames N] [--range A B]
//       单路位姿统计（指标D）：平移/法向/坐标轴端点帧间变化、重投影误差分布、
//       双解跳变、门禁拒绝。变体由环境变量 RMV_BAR_FIT 选择（不设=拟合），
//       跑两次即可对比。
//   ./build/test_bar_fit --synth
//       合成真值对照（指标C）：已知几何的灯条（干净/末端拖尾/侧边光晕/边缘锯齿/
//       长边缺口），比较"拟合"与"minAreaRect"相对真值的误差；锯齿样本多随机种子，
//       额外给出跨样本离散度（小=对边缘噪声不敏感）。
//
// 说明：视频/位姿模式从仓库根目录运行（config/camera.yaml 按相对路径加载）。
#include "StudentTasks.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

namespace
{

    struct Stats
    {
        std::vector<double> v;

        void add(double x)
        {
            v.push_back(x);
        }

        void report(const char *name, const char *unit) const
        {
            if (v.empty())
            {
                std::printf("%-30s 无数据\n", name);
                return;
            }
            std::vector<double> s = v;
            std::sort(s.begin(), s.end());
            auto pct = [&](double p)
            {
                std::size_t i = static_cast<std::size_t>(p * s.size());
                if (i >= s.size())
                    i = s.size() - 1;
                return s[i];
            };
            std::printf("%-30s P50=%7.3f P90=%7.3f P95=%7.3f max=%7.3f %s (n=%zu)\n", name,
                        pct(0.50), pct(0.90), pct(0.95), s.back(), unit, s.size());
        }
    };

    // 角度差按 180° 周期折算到 [0, 90°]
    double angleDiff180(double a, double b)
    {
        double d = std::fmod(std::fabs(a - b), 180.0);
        return d > 90.0 ? 180.0 - d : d;
    }

    double barAngleDeg(const DebugBar &b)
    {
        return std::atan2(b.axis.y, b.axis.x) * 180.0 / CV_PI;
    }

    // 在 pts 里找离 q 最近的点（>max_dist 视为未匹配）
    int nearestIndex(const std::vector<cv::Point2f> &pts, const cv::Point2f &q, double max_dist)
    {
        int best = -1;
        double bd = max_dist;
        for (std::size_t i = 0; i < pts.size(); ++i)
        {
            const double d = cv::norm(pts[i] - q);
            if (d < bd)
            {
                bd = d;
                best = static_cast<int>(i);
            }
        }
        return best;
    }

    std::vector<cv::Point2f> barCenters(const std::vector<DebugBar> &bars)
    {
        std::vector<cv::Point2f> pts;
        pts.reserve(bars.size());
        for (const DebugBar &b : bars)
        {
            pts.push_back(b.center);
        }
        return pts;
    }

    // 灯条级帧间变化：只统计"两变体共有的同一物理灯条"（同一帧内两变体都提取到、
    // 且相邻帧都存在的灯条），避免拟合变体多提取的碎条污染尾部指标；每帧灯条数另计。
    struct CommonBarJitter
    {
        Stats center_px[2], angle_deg[2], dhalf_l_px[2], dhalf_w_px[2];
        Stats corner_px[2];       // 灯条四角帧间位移（与上一帧 4 角最近匹配）
        Stats corner_worst[2];    // 每根灯条"最差角点"位移（不掩盖个别角点）
        std::vector<std::pair<DebugBar, DebugBar>> prev; // 上一帧共有灯条(基线, 拟合)
        long pairs_total{0}, matched[2] = {0, 0}, new_bars{0};
        long bars_sum[2] = {0, 0};
        long frames_seen{0};
        long count_diff_frames{0};
        std::vector<long> count_diff_list;

        void update(const std::vector<DebugBar> &bars_a, const std::vector<DebugBar> &bars_b,
                    long frame_idx, double common_dist = 5.0)
        {
            ++frames_seen;
            bars_sum[0] += static_cast<long>(bars_a.size());
            bars_sum[1] += static_cast<long>(bars_b.size());
            if (bars_a.size() != bars_b.size())
            {
                ++count_diff_frames;
                if (count_diff_list.size() < 20)
                {
                    count_diff_list.push_back(frame_idx);
                }
            }
            // 1) 本帧共有灯条：基线条 ↔ 拟合条（同一物条，中心 5px 内）
            const std::vector<cv::Point2f> cb = barCenters(bars_b);
            std::vector<std::pair<DebugBar, DebugBar>> cur;
            for (const DebugBar &ba : bars_a)
            {
                const int j = nearestIndex(cb, ba.center, common_dist);
                if (j >= 0)
                {
                    cur.emplace_back(ba, bars_b[static_cast<std::size_t>(j)]);
                }
            }
            // 2) 与上一帧共有灯条按基线中心匹配（"上一帧存在且本帧仍在"才算帧间样本）
            std::vector<cv::Point2f> pc;
            pc.reserve(prev.size());
            for (const auto &p : prev)
            {
                pc.push_back(p.first.center);
            }
            for (const auto &pair : cur)
            {
                ++pairs_total;
                const int k = nearestIndex(pc, pair.first.center, 120.0);
                if (k < 0)
                {
                    ++new_bars;
                    continue;
                }
                const DebugBar *now[2] = {&pair.first, &pair.second};
                const DebugBar *was[2] = {&prev[k].first, &prev[k].second};
                for (int var = 0; var < 2; ++var)
                {
                    center_px[var].add(cv::norm(now[var]->center - was[var]->center));
                    angle_deg[var].add(
                        angleDiff180(barAngleDeg(*now[var]), barAngleDeg(*was[var])));
                    dhalf_l_px[var].add(std::fabs(now[var]->half_l - was[var]->half_l));
                    dhalf_w_px[var].add(std::fabs(now[var]->half_w - was[var]->half_w));
                    // 四角：与上一帧 4 个角最近匹配（PCA 轴符号可使角点顺序整体翻转）
                    double worst = 0.0;
                    for (int k = 0; k < 4; ++k)
                    {
                        double best = 1e18;
                        for (int j = 0; j < 4; ++j)
                        {
                            best = std::min(best, cv::norm(now[var]->corners[k] -
                                                           was[var]->corners[j]));
                        }
                        corner_px[var].add(best);
                        worst = std::max(worst, best);
                    }
                    corner_worst[var].add(worst);
                    ++matched[var];
                }
            }
            prev = std::move(cur);
        }

        void report() const
        {
            static const char *kTags[2] = {"基线", "拟合"};
            char name[128];
            for (int var = 0; var < 2; ++var)
            {
                std::snprintf(name, sizeof(name), "%s 共有灯条中心位移", kTags[var]);
                center_px[var].report(name, "px");
                std::snprintf(name, sizeof(name), "%s 共有灯条方向变化", kTags[var]);
                angle_deg[var].report(name, "deg");
                std::snprintf(name, sizeof(name), "%s 共有灯条 Δ半长", kTags[var]);
                dhalf_l_px[var].report(name, "px");
                std::snprintf(name, sizeof(name), "%s 共有灯条 Δ半宽", kTags[var]);
                dhalf_w_px[var].report(name, "px");
                std::snprintf(name, sizeof(name), "%s 灯条四角位移", kTags[var]);
                corner_px[var].report(name, "px");
                std::snprintf(name, sizeof(name), "%s 灯条最差角点位移", kTags[var]);
                corner_worst[var].report(name, "px");
            }
            const double denom = frames_seen > 0 ? static_cast<double>(frames_seen) : 1.0;
            std::printf("共有灯条样本 %ld 对 | 新出现(上一帧无) %ld 对 | 每帧灯条数: 基线 %.2f | 拟合 "
                        "%.2f | 灯条数不同帧 %ld",
                        pairs_total, new_bars, static_cast<double>(bars_sum[0]) / denom,
                        static_cast<double>(bars_sum[1]) / denom, count_diff_frames);
            if (!count_diff_list.empty())
            {
                std::printf("（前几帧: ");
                for (long f : count_diff_list)
                {
                    std::printf("%ld ", f);
                }
                std::printf("）");
            }
            std::printf("\n");
        }
    };

    // 角点帧间位移（单检出帧，角点顺序固定 TL,TR,BR,BL）
    struct CornerJitter
    {
        Stats per_corner[4];
        Stats worst_per_frame; // 每帧四个角点位移的最大值（不掩盖个别角点）
        std::array<cv::Point2f, 4> prev{};
        bool has_prev{false};
        long prev_frame{-1};

        void update(const ArmorDetection &d, long frame_idx)
        {
            if (has_prev && frame_idx == prev_frame + 1) // 严格相邻帧（蓝方素材常有断检）
            {
                double worst = 0.0;
                for (int k = 0; k < 4; ++k)
                {
                    const double dist = cv::norm(d.corners[k] - prev[k]);
                    per_corner[k].add(dist);
                    worst = std::max(worst, dist);
                }
                worst_per_frame.add(worst);
            }
            prev = d.corners;
            has_prev = true;
            prev_frame = frame_idx;
        }

        void reset()
        {
            has_prev = false;
        }

        void report(const char *tag) const
        {
            static const char *kNames[4] = {"TL", "TR", "BR", "BL"};
            char name[128];
            for (int k = 0; k < 4; ++k)
            {
                std::snprintf(name, sizeof(name), "%s 角点%s 帧间位移", tag, kNames[k]);
                per_corner[k].report(name, "px");
            }
            std::snprintf(name, sizeof(name), "%s 每帧最大角点位移", tag);
            worst_per_frame.report(name, "px");
        }
    };

    // 装甲板物点模型（与 armor_solve 的 buildObjectPoints 一致：外缘 135/230mm × 57mm）
    std::vector<cv::Point3f> objectPoints(ArmorSize size)
    {
        const float half_w = (size == ArmorSize::Small ? 0.135F : 0.230F) / 2.0F;
        const float half_h = 0.057F / 2.0F;
        return {{-half_w, -half_h, 0.0F},
                {half_w, -half_h, 0.0F},
                {half_w, half_h, 0.0F},
                {-half_w, half_h, 0.0F}};
    }

    cv::Vec3d normalOf(const cv::Mat &rvec)
    {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        cv::Vec3d n(R.at<double>(0, 2), R.at<double>(1, 2), R.at<double>(2, 2));
        const double nrm = cv::norm(n);
        return nrm > 1e-12 ? n * (1.0 / nrm) : cv::Vec3d(0.0, 0.0, 1.0);
    }

    double angleBetweenDeg(const cv::Vec3d &a, const cv::Vec3d &b)
    {
        double d = a.dot(b);
        d = std::max(-1.0, std::min(1.0, d));
        return std::acos(d) * 180.0 / CV_PI;
    }

    bool loadCamera(CameraParameters &camera)
    {
        cv::FileStorage fs("config/camera.yaml", cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            fs.open("../config/camera.yaml", cv::FileStorage::READ);
        }
        if (!fs.isOpened())
        {
            std::fprintf(stderr, "无法打开 config/camera.yaml（请从项目根目录运行）\n");
            return false;
        }
        fs["camera_matrix"] >> camera.camera_matrix;
        fs["distortion_coefficients"] >> camera.distortion_coefficients;
        camera.rotation_camera_to_gimbal = cv::Mat::eye(3, 3, CV_64F);
        camera.translation_camera_to_gimbal = cv::Mat::zeros(3, 1, CV_64F);
        return !camera.camera_matrix.empty();
    }

    void printFitStats(long frames)
    {
        std::printf("\n===== 拟合门控统计（拟合变体，累计）=====\n");
        std::printf("进入拟合 %d 条 | 采纳 %d (%.1f%%) | 回退: 点数/退化 %d, 接受门 %d, G1支撑率 %d, "
                    "G2轴偏差 %d, G3面积比 %d, E边带 %d, E平行 %d, E宽度 %d\n",
                    g_bar_fit_total, g_bar_fit_accept,
                    g_bar_fit_total > 0 ? 100.0 * g_bar_fit_accept / g_bar_fit_total : 0.0,
                    g_bar_fit_rej_pts, g_bar_fit_rej_size, g_bar_fit_rej_g1, g_bar_fit_rej_g2,
                    g_bar_fit_rej_g3, g_bar_fit_rej_edges, g_bar_fit_rej_par,
                    g_bar_fit_rej_width);
        if (frames > 0)
        {
            std::printf("拟合总耗时 %.2f ms → 每帧 ≈ %.3f ms（%.1f 条/帧）\n", g_bar_fit_ms_total,
                        g_bar_fit_ms_total / static_cast<double>(frames),
                        static_cast<double>(g_bar_fit_total) / static_cast<double>(frames));
        }
        std::printf("宽度平滑(P3): 窗口=%d 帧 | 应用 %d 次 | 轨迹重置 %d 次\n", g_bar_width_window,
                    g_bar_width_smooth_count, g_bar_width_reset_count);
    }

    // ================= 模式 1：进程内 A/B（几何/角点/检出/耗时）=================
    int run_ab(const std::string &video, TeamColor enemy, long max_frames, long range_lo,
               long range_hi, bool same_variant = false, int fit_variant = 3)
    {
        cv::VideoCapture cap(video);
        if (!cap.isOpened())
        {
            std::fprintf(stderr, "无法打开视频: %s\n", video.c_str());
            return 2;
        }
        char mode_desc[96];
        std::snprintf(mode_desc, sizeof(mode_desc), "进程内 A/B（基线 minAreaRect vs 拟合 v%d）",
                      fit_variant);
        std::printf("视频: %s | 敌方: %s | %s | 统计窗口 [%ld, %ld]\n", video.c_str(),
                    enemy == TeamColor::Red ? "红" : "蓝",
                    same_variant ? "同变体对照（两槽都跑基线 → 量化顺序/缓存偏差）" : mode_desc,
                    range_lo, range_hi);

        long frames = 0;
        long kept = 0;
        long det_eq = 0, det_ne = 0, id_eq = 0;
        long det0[2] = {0, 0}, det1[2] = {0, 0}, det2[2] = {0, 0};
        std::map<int, long> id_hist[2];
        std::vector<long> mismatch_frames;

        CommonBarJitter bar_jitter;
        CornerJitter corner_a, corner_b;
        Stats gd_center, gd_angle, gd_hl, gd_hw; // 同帧（拟合-基线）几何差（仅共有灯条）
        Stats det_ms[2];
        long corner_pairs[2] = {0, 0};

        cv::Mat frame;
        while (cap.read(frame))
        {
            ++frames;
            if (frames < range_lo)
            {
                continue; // 窗口前：只解码不处理（不产生统计与状态）
            }
            if (frames > range_hi || (max_frames > 0 && frames - range_lo + 1 > max_frames))
            {
                break;
            }
            ++kept;

            std::vector<ArmorDetection> dets[2];
            std::vector<DebugBar> bars[2];
            for (int var = 0; var < 2; ++var)
            {
                g_bar_fit_mode = ((var == 1) && !same_variant) ? fit_variant : 0;
                const auto t0 = std::chrono::steady_clock::now();
                dets[var] = armor_detect(frame, enemy);
                det_ms[var].add(std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t0)
                                    .count());
                bars[var] = debugBars();
                const long n = static_cast<long>(dets[var].size());
                if (n == 0)
                    ++det0[var];
                else if (n == 1)
                    ++det1[var];
                else
                    ++det2[var];
                for (const ArmorDetection &d : dets[var])
                    ++id_hist[var][d.target_id];
            }

            // 检出对照
            if (dets[0].size() == dets[1].size())
            {
                ++det_eq;
            }
            else
            {
                ++det_ne;
                if (mismatch_frames.size() < 20)
                    mismatch_frames.push_back(frames);
            }
            if (dets[0].size() == 1 && dets[1].size() == 1 &&
                dets[0][0].target_id == dets[1][0].target_id)
            {
                ++id_eq;
            }

            // 灯条级帧间抖动（仅两变体共有灯条）
            bar_jitter.update(bars[0], bars[1], frames);

            // 同帧几何差（拟合 vs 基线，仅共有灯条：同一物条）
            const std::vector<cv::Point2f> cb = barCenters(bars[1]);
            for (const DebugBar &ba : bars[0])
            {
                const int i = nearestIndex(cb, ba.center, 5.0);
                if (i < 0)
                {
                    continue;
                }
                gd_center.add(cv::norm(ba.center - bars[1][i].center));
                gd_angle.add(angleDiff180(barAngleDeg(ba), barAngleDeg(bars[1][i])));
                gd_hl.add(std::fabs(ba.half_l - bars[1][i].half_l));
                gd_hw.add(std::fabs(ba.half_w - bars[1][i].half_w));
            }

            // 角点级帧间抖动（各变体独立：只有单检出帧参与）
            for (int var = 0; var < 2; ++var)
            {
                CornerJitter &cj = (var == 0) ? corner_a : corner_b;
                if (dets[var].size() == 1)
                {
                    cj.update(dets[var][0], frames);
                    ++corner_pairs[var];
                }
                else
                {
                    cj.reset();
                }
            }

            if (frames % 500 == 0)
            {
                std::printf("... %ld 帧\n", frames);
            }
        }

        std::printf("\n统计帧数 %ld（区间 [%ld, %ld]，共解码 %ld 帧）\n", kept, range_lo,
                    range_hi, frames);
        for (int var = 0; var < 2; ++var)
        {
            std::printf("[%s] 检出 det=0: %ld | det=1: %ld (%.1f%%) | det>=2: %ld | 编号: ",
                        var == 0 ? "基线" : "拟合", det0[var], det1[var],
                        kept > 0 ? 100.0 * static_cast<double>(det1[var]) / kept : 0.0,
                        det2[var]);
            for (const auto &kv : id_hist[var])
            {
                std::printf("id=%d:%ld ", kv.first, kv.second);
            }
            std::printf("\n");
        }
        std::printf("检出规模/编号一致的帧 %ld/%ld（不一致 %ld", det_eq, kept, det_ne);
        if (!mismatch_frames.empty())
        {
            std::printf("，前几帧: ");
            for (long f : mismatch_frames)
                std::printf("%ld ", f);
        }
        std::printf("）\n");

        std::printf("\n===== 指标A：灯条几何帧间变化（仅两变体共有灯条，严格相邻帧）=====\n");
        bar_jitter.report();

        std::printf("\n===== 同帧几何差（拟合 - 基线，同一灯条）=====\n");
        gd_center.report("Δ中心", "px");
        gd_angle.report("Δ方向", "deg");
        gd_hl.report("Δ半长", "px");
        gd_hw.report("Δ半宽", "px");

        std::printf("\n===== 指标B：角点帧间位移（单检出帧）=====\n");
        corner_a.report("基线");
        corner_b.report("拟合");
        std::printf("单检出帧计: 基线 %ld | 拟合 %ld\n", corner_pairs[0], corner_pairs[1]);

        std::printf("\n===== 指标E：耗时（含掩码/几何/裁剪/分类）=====\n");
        det_ms[0].report("基线 armor_detect", "ms");
        det_ms[1].report("拟合 armor_detect", "ms");

        printFitStats(kept);
        return 0;
    }

    // ================= 模式 2：单路位姿统计（指标D）=================
    int run_pose(const std::string &video, TeamColor enemy, long max_frames, long range_lo,
                 long range_hi)
    {
        CameraParameters camera;
        if (!loadCamera(camera))
        {
            return 2;
        }
        GimbalState gimbal{};

        cv::VideoCapture cap(video);
        if (!cap.isOpened())
        {
            std::fprintf(stderr, "无法打开视频: %s\n", video.c_str());
            return 2;
        }
        std::printf("视频: %s | 敌方: %s | 位姿模式，变体 = %s（RMV_BAR_FIT=%s）| 统计窗口 [%ld, %ld]\n",
                    video.c_str(), enemy == TeamColor::Red ? "红" : "蓝",
                    g_bar_fit_mode > 0 ? "约束拟合" : "基线 minAreaRect",
                    g_bar_fit_mode > 0 ? (g_bar_fit_mode == 1 ? "1/v1" : "2/v2 默认") : "0/off",
                    range_lo, range_hi);

        long frames = 0, kept = 0, det0 = 0, det1 = 0, det2 = 0;
        long poses_total = 0, poses_eq1 = 0;
        std::map<int, long> id_hist;
        Stats dpos_mm, dyaw_deg, derr_px;
        Stats dist_mm; // 距离（相机系平移模长，检查是否有系统性偏移）
        Stats dnorm_dis_deg, dnorm_base_deg;         // 相邻帧法向夹角（消歧路 / 独立最优路）
        Stats axis_origin_px, axis_end_px;           // 坐标轴端点帧间位移
        long jump_dis = 0, jump_base = 0, yaw_jumps = 0;
        std::vector<long> jump_dis_frames, yaw_jump_frames;

        struct Prev
        {
            bool valid{false};
            cv::Vec3d pos_mm{0, 0, 0};
            double yaw_deg{0.0};
            cv::Vec3d n{0, 0, 1};      // 消歧路法向
            cv::Vec3d n_base{0, 0, 1}; // 独立最优路法向
            cv::Point2f ax0, ax1;      // 轴端点（图像）
            bool has_axes{false};
        };
        std::map<std::pair<int, int>, Prev> prev; // 键：(id, size)

        cv::Mat frame;
        while (cap.read(frame))
        {
            ++frames;
            if (frames < range_lo)
            {
                continue; // 窗口前：只解码不处理
            }
            if (frames > range_hi || (max_frames > 0 && frames - range_lo + 1 > max_frames))
            {
                break;
            }
            ++kept;
            const auto dets = armor_detect(frame, enemy);
            const long n = static_cast<long>(dets.size());
            if (n == 0)
                ++det0;
            else if (n == 1)
                ++det1;
            else
                ++det2;
            for (const ArmorDetection &d : dets)
                ++id_hist[d.target_id];

            const auto poses = armor_solve(dets, camera, gimbal);
            poses_total += static_cast<long>(poses.size());
            if (dets.size() == 1 && poses.size() == 1)
            {
                ++poses_eq1;
            }
            if (dets.size() != 1)
            {
                continue; // 多板同框时 (id,size) 键会混板，只统计单检出帧
            }

            const ArmorDetection &d = dets[0];
            const std::pair<int, int> key{d.target_id, static_cast<int>(d.size)};
            const std::vector<cv::Point3f> obj = objectPoints(d.size);
            const std::vector<cv::Point2f> img(d.corners.begin(), d.corners.end());

            // ---- 两路法向：消歧路（公开 API）/ 独立取误差最小解 ----
            cv::Mat r_dis, t_dis;
            const bool ok_dis =
                solveArmorPoseDisambiguated(obj, img, camera.camera_matrix,
                                            camera.distortion_coefficients, r_dis, t_dis);
            std::vector<cv::Mat> rvecs, tvecs;
            const int nsol = cv::solvePnPGeneric(obj, img, camera.camera_matrix,
                                                 camera.distortion_coefficients, rvecs, tvecs,
                                                 false, cv::SOLVEPNP_IPPE);
            int best = -1;
            double best_err = std::numeric_limits<double>::infinity();
            for (int i = 0; i < nsol; ++i)
            {
                if (tvecs[i].at<double>(2) <= 0.0)
                {
                    continue;
                }
                std::vector<cv::Point2f> proj;
                cv::projectPoints(obj, rvecs[i], tvecs[i], camera.camera_matrix,
                                  camera.distortion_coefficients, proj);
                double sum = 0.0;
                for (std::size_t k = 0; k < proj.size(); ++k)
                {
                    sum += cv::norm(proj[k] - img[k]);
                }
                const double err = sum / static_cast<double>(proj.size());
                if (err < best_err)
                {
                    best_err = err;
                    best = i;
                }
            }
            if (!ok_dis || best < 0)
            {
                prev.erase(key);
                continue;
            }

            Prev &pv = prev[key];
            if (pv.valid && poses.size() == 1)
            {
                const cv::Vec3d pos_now = poses[0].position_camera_m * 1000.0;
                dpos_mm.add(cv::norm(pos_now - pv.pos_mm));
                double dy = std::fabs(poses[0].armor_yaw * 180.0 / CV_PI - pv.yaw_deg);
                if (dy > 180.0)
                    dy = 360.0 - dy;
                dyaw_deg.add(dy);
                if (dy > 25.0)
                {
                    ++yaw_jumps;
                    if (yaw_jump_frames.size() < 40)
                        yaw_jump_frames.push_back(frames);
                }
                derr_px.add(poses[0].reprojection_error);
                dist_mm.add(cv::norm(poses[0].position_camera_m) * 1000.0);

                const double jd = angleBetweenDeg(normalOf(r_dis), pv.n);
                const double jb = angleBetweenDeg(normalOf(rvecs[best]), pv.n_base);
                dnorm_dis_deg.add(jd);
                dnorm_base_deg.add(jb);
                if (jd > 25.0)
                {
                    ++jump_dis;
                    if (jump_dis_frames.size() < 40)
                        jump_dis_frames.push_back(frames);
                }
                if (jb > 25.0)
                {
                    ++jump_base;
                }
            }
            pv.valid = true;
            pv.pos_mm = poses.size() == 1 ? poses[0].position_camera_m * 1000.0
                                          : cv::Vec3d(pv.pos_mm);
            pv.yaw_deg = poses.size() == 1 ? poses[0].armor_yaw * 180.0 / CV_PI : pv.yaw_deg;
            pv.n = normalOf(r_dis);
            pv.n_base = normalOf(rvecs[best]);

            // 坐标轴端点（8cm x 轴）：原点与 (0.08,0,0) 投到图像
            std::vector<cv::Point3f> ax_obj = {{0.0F, 0.0F, 0.0F}, {0.08F, 0.0F, 0.0F}};
            std::vector<cv::Point2f> ax_img;
            cv::projectPoints(ax_obj, r_dis, t_dis, camera.camera_matrix,
                              camera.distortion_coefficients, ax_img);
            if (pv.has_axes)
            {
                axis_origin_px.add(cv::norm(ax_img[0] - pv.ax0));
                axis_end_px.add(cv::norm(ax_img[1] - pv.ax1));
            }
            pv.ax0 = ax_img[0];
            pv.ax1 = ax_img[1];
            pv.has_axes = true;

            if (frames % 500 == 0)
            {
                std::printf("... %ld 帧\n", frames);
            }
        }

        std::printf("\n统计帧数 %ld | 检出 det=0: %ld | det=1: %ld | det>=2: %ld | 编号: ", kept,
                    det0, det1, det2);
        for (const auto &kv : id_hist)
        {
            std::printf("id=%d:%ld ", kv.first, kv.second);
        }
        std::printf("\nposposes 总数 %ld | 单帧 det=1 且 poses=1: %ld | 门禁拒绝: reproj=%d depth=%d "
                    "finite=%d pnpFail=%d\n",
                    poses_total, poses_eq1, g_rej_reproj, g_rej_depth, g_rej_finite, g_pnp_fail);
        std::printf("\n===== 指标D：位姿帧间变化（单检出帧）=====\n");
        dist_mm.report("距离(相机系平移模长)", "mm");
        dpos_mm.report("PnP 平移帧间变化", "mm");
        dyaw_deg.report("armor_yaw 帧间变化", "deg");
        derr_px.report("重投影误差", "px");
        std::printf("armor_yaw >25° 台阶: %ld", yaw_jumps);
        if (!yaw_jump_frames.empty())
        {
            std::printf("（帧: ");
            for (long f : yaw_jump_frames)
                std::printf("%ld ", f);
            std::printf("）");
        }
        std::printf("\n");
        std::printf("\n--- 法向跳变对照（相邻帧夹角）---\n");
        dnorm_dis_deg.report("消歧路 法向帧间夹角", "deg");
        std::printf("消歧路 >25° 跳变: %ld", jump_dis);
        if (!jump_dis_frames.empty())
        {
            std::printf("（帧: ");
            for (long f : jump_dis_frames)
                std::printf("%ld ", f);
            std::printf("）");
        }
        std::printf("\n");
        dnorm_base_deg.report("基准(独立最优解) 法向夹角", "deg");
        std::printf("基准 >25° 跳变: %ld\n", jump_base);
        axis_origin_px.report("轴原点帧间位移", "px");
        axis_end_px.report("轴端点(8cm)帧间位移", "px");

        printFitStats(kept);
        return 0;
    }

    // ================= 模式 3：合成真值对照（指标C）=================
    // 画布上画一根已知几何的灯条（可叠加缺陷），跑真实检测链路，
    // 比较"拟合/minAreaRect"两条几何估计相对真值的误差。
    enum class SynthKind
    {
        Clean = 0,   // 干净矩形
        EndTail,     // 末端拖尾（8x3 凸起 → 撑长 minAreaRect）
        SideGlow,    // 侧边光晕（9x9 圆斑 → 撑宽）
        Jagged,      // 长边随机锯齿（±2px，多随机种子）
        Notch,       // 长边缺口（凹 4px → 内凹不影响凸包，负对照）
    };

    const char *synthKindAscii(SynthKind k)
    {
        switch (k)
        {
        case SynthKind::Clean:
            return "clean";
        case SynthKind::EndTail:
            return "endtail";
        case SynthKind::SideGlow:
            return "sideglow";
        case SynthKind::Jagged:
            return "jagged";
        default:
            return "notch";
        }
    }

    const char *synthKindName(SynthKind k)
    {
        switch (k)
        {
        case SynthKind::Clean:
            return "干净矩形";
        case SynthKind::EndTail:
            return "末端拖尾";
        case SynthKind::SideGlow:
            return "侧边光晕";
        case SynthKind::Jagged:
            return "长边锯齿";
        default:
            return "长边缺口";
        }
    }

    cv::Mat makeSynthBar(SynthKind kind, double angle_deg, unsigned seed, cv::Point2f &gt_center)
    {
        const int W = 640, H = 480;
        cv::Mat img(H, W, CV_8UC3, cv::Scalar(30, 30, 30));
        const cv::Point2f center(320.0F, 240.0F);
        gt_center = center;
        const double L = 90.0, BW = 9.0;
        const double th = angle_deg * CV_PI / 180.0;
        const cv::Point2f ax(static_cast<float>(std::cos(th)), static_cast<float>(std::sin(th)));
        const cv::Point2f pp(-ax.y, ax.x);

        std::mt19937 rng(seed);
        std::uniform_int_distribution<int> jag(0, 2);

        // 灯条主体多边形：两条长边各分 15 段，可叠加随机外凸（Jagged）
        std::vector<cv::Point> poly;
        const int seg = 15;
        auto edgePoint = [&](double t, double offset) // t∈[-1,1] 沿轴，offset 沿宽向
        {
            const cv::Point2f p = center + ax * static_cast<float>(t * L / 2.0) +
                                  pp * static_cast<float>(offset);
            return cv::Point(cvRound(p.x), cvRound(p.y));
        };
        for (int i = 0; i <= seg; ++i) // 一侧长边
        {
            const double t = -1.0 + 2.0 * i / seg;
            double off = BW / 2.0;
            if (kind == SynthKind::Jagged)
            {
                off += jag(rng);
            }
            poly.push_back(edgePoint(t, off));
        }
        for (int i = seg; i >= 0; --i) // 另一侧长边（回程）
        {
            const double t = -1.0 + 2.0 * i / seg;
            double off = -BW / 2.0;
            if (kind == SynthKind::Jagged)
            {
                off -= jag(rng);
            }
            poly.push_back(edgePoint(t, off));
        }
        cv::fillPoly(img, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(0, 0, 255));

        if (kind == SynthKind::EndTail)
        {
            // 末端 8x3 拖尾（同色）：凸出 → minAreaRect 长度/中心被拉偏
            std::vector<cv::Point> tail;
            const double t0 = L / 2.0, t1 = L / 2.0 + 8.0;
            tail.push_back(edgePoint(t0 * 2.0 / L, 1.5));
            tail.push_back(edgePoint(t1 * 2.0 / L, 1.5));
            tail.push_back(edgePoint(t1 * 2.0 / L, -1.5));
            tail.push_back(edgePoint(t0 * 2.0 / L, -1.5));
            cv::fillPoly(img, std::vector<std::vector<cv::Point>>{tail}, cv::Scalar(0, 0, 255));
        }
        else if (kind == SynthKind::SideGlow)
        {
            // 长边中段外凸圆斑（半径 4.5）→ 撑宽
            const cv::Point2f p = center + pp * static_cast<float>(BW / 2.0);
            cv::circle(img, cv::Point(cvRound(p.x), cvRound(p.y)), 5, cv::Scalar(0, 0, 255), -1);
        }
        else if (kind == SynthKind::Notch)
        {
            // 长边中段凹口（背景色 6x4，凹 4px）
            const cv::Point2f p = center - pp * static_cast<float>(BW / 2.0);
            std::vector<cv::Point> notch;
            for (double t = -3; t <= 3; t += 1.5)
            {
                const cv::Point2f q = p + ax * static_cast<float>(t) - pp * 3.0F;
                notch.push_back(cv::Point(cvRound(q.x), cvRound(q.y)));
            }
            notch.push_back(cv::Point(cvRound(p.x + 3 * ax.x), cvRound(p.y + 3 * ax.y)));
            notch.push_back(cv::Point(cvRound(p.x - 3 * ax.x), cvRound(p.y - 3 * ax.y)));
            cv::fillPoly(img, std::vector<std::vector<cv::Point>>{notch}, cv::Scalar(30, 30, 30));
        }
        return img;
    }

    struct SynthErr
    {
        Stats dc, dang, dL, dW;
        void add(const DebugBar &b, const cv::Point2f &gt, double gt_angle)
        {
            dc.add(cv::norm(b.center - gt));
            dang.add(angleDiff180(barAngleDeg(b), gt_angle));
            dL.add(std::fabs(2.0 * b.half_l - 90.0));
            dW.add(std::fabs(2.0 * b.half_w - 9.0));
        }
    };

    // 按 (中心, 长轴, 半长, 半宽) 画一个矩形轮廓（用于合成对照图）
    void drawBarRect(cv::Mat &img, const cv::Point2f &center, const cv::Point2f &axis,
                     float half_l, float half_w, const cv::Scalar &color, int thickness)
    {
        const cv::Point2f perp(-axis.y, axis.x);
        std::vector<cv::Point> pts;
        const cv::Point2f corners[4] = {center + axis * half_l + perp * half_w,
                                       center + axis * half_l - perp * half_w,
                                       center - axis * half_l - perp * half_w,
                                       center - axis * half_l + perp * half_w};
        for (const cv::Point2f &p : corners)
        {
            pts.emplace_back(cvRound(p.x), cvRound(p.y));
        }
        cv::polylines(img, std::vector<std::vector<cv::Point>>{pts}, true, color, thickness,
                      cv::LINE_AA);
    }

    int run_synth(const std::string &save_prefix = "")
    {
        std::printf("合成真值对照（灯条 L=90px, W=9px；真值 = 绘制几何；倾角 18° 相对竖直，"
                    "以通过 |tilt|<=40 灯条过滤）\n");
        const double gt_angle = 72.0; // 长轴方向角：与竖直方向夹角 18°
        struct Case
        {
            SynthKind kind;
            int samples;
        };
        const std::array<Case, 5> cases = {
            Case{SynthKind::Clean, 1},   Case{SynthKind::EndTail, 3},
            Case{SynthKind::SideGlow, 3}, Case{SynthKind::Jagged, 12},
            Case{SynthKind::Notch, 3}};

        for (const Case &c : cases)
        {
            SynthErr err_base, err_fit;
            int used = 0;
            for (int s = 0; s < c.samples; ++s)
            {
                cv::Point2f gt;
                const cv::Mat img = makeSynthBar(c.kind, gt_angle, 1234 + 77 * s, gt);
                std::vector<DebugBar> bars[2];
                for (int var = 0; var < 2; ++var)
                {
                    g_bar_fit_mode = (var == 1) ? 3 : 0;
                    armor_detect(img, TeamColor::Red);
                    bars[var] = debugBars();
                }
                const std::vector<cv::Point2f> bc = barCenters(bars[0]);
                const std::vector<cv::Point2f> bf = barCenters(bars[1]);
                const int ib = nearestIndex(bc, gt, 40.0);
                const int iff = nearestIndex(bf, gt, 40.0);
                if (ib >= 0 && iff >= 0)
                {
                    err_base.add(bars[0][ib], gt, gt_angle);
                    err_fit.add(bars[1][iff], gt, gt_angle);
                    ++used;
                }
                if (!save_prefix.empty() && s == 0)
                {
                    // 对照图：以真值为中心裁剪 100x100 并 5x 最近邻放大；白=真值矩形、
                    // 红=minAreaRect、绿=约束拟合（三者的像素级偏移在该缩放下可见）
                    constexpr int kCrop = 100;
                    constexpr double kZoom = 5.0;
                    cv::Rect roi(cvRound(gt.x) - kCrop / 2, cvRound(gt.y) - kCrop / 2, kCrop,
                                 kCrop);
                    roi &= cv::Rect(0, 0, img.cols, img.rows);
                    cv::Mat canvas;
                    cv::resize(img(roi), canvas, cv::Size(), kZoom, kZoom, cv::INTER_NEAREST);
                    cv::Rect roi2(cvRound(gt.x) - kCrop / 2, cvRound(gt.y) - kCrop / 2, kCrop,
                                  kCrop);
                    roi2 &= cv::Rect(0, 0, img.cols, img.rows);
                    cv::Point2f off(static_cast<float>(-roi2.x * kZoom),
                                    static_cast<float>(-roi2.y * kZoom));
                    auto to_canvas = [&](const cv::Point2f &p)
                    { return p * static_cast<float>(kZoom) + off; };
                    const double th = gt_angle * CV_PI / 180.0;
                    const cv::Point2f axis(static_cast<float>(std::cos(th)),
                                           static_cast<float>(std::sin(th)));
                    drawBarRect(canvas, to_canvas(gt), axis, static_cast<float>(45.0 * kZoom),
                                static_cast<float>(4.5 * kZoom), cv::Scalar(255, 255, 255), 2);
                    if (ib >= 0 && bc[static_cast<std::size_t>(ib)].x > 0.0F)
                    {
                        const DebugBar &b = bars[0][static_cast<std::size_t>(ib)];
                        drawBarRect(canvas, to_canvas(b.center), b.axis,
                                    static_cast<float>(b.half_l * kZoom),
                                    static_cast<float>(b.half_w * kZoom), cv::Scalar(0, 0, 255), 1);
                    }
                    if (iff >= 0 && bf[static_cast<std::size_t>(iff)].x > 0.0F)
                    {
                        const DebugBar &b = bars[1][static_cast<std::size_t>(iff)];
                        drawBarRect(canvas, to_canvas(b.center), b.axis,
                                    static_cast<float>(b.half_l * kZoom),
                                    static_cast<float>(b.half_w * kZoom), cv::Scalar(0, 255, 0), 1);
                    }
                    cv::putText(canvas, "white=truth  red=minAreaRect  green=fit",
                                cv::Point(8, 24), cv::FONT_HERSHEY_SIMPLEX, 0.55,
                                cv::Scalar(200, 200, 200), 1, cv::LINE_AA);
                    const std::string path = save_prefix + "_" + synthKindAscii(c.kind) + ".png";
                    cv::imwrite(path, canvas);
                    std::printf("  对照图（白=真值/红=minAreaRect/绿=拟合）: %s\n", path.c_str());
                }
            }
            std::printf("\n[%s] 有效样本 %d/%d\n", synthKindName(c.kind), used, c.samples);
            if (used == 0)
            {
                std::printf("  未检出灯条（样本未被链路保留）\n");
                continue;
            }
            char name[128];
            std::snprintf(name, sizeof(name), "  minAreaRect Δ中心");
            err_base.dc.report(name, "px");
            std::snprintf(name, sizeof(name), "  minAreaRect Δ方向");
            err_base.dang.report(name, "deg");
            std::snprintf(name, sizeof(name), "  minAreaRect Δ长边");
            err_base.dL.report(name, "px");
            std::snprintf(name, sizeof(name), "  minAreaRect Δ短边");
            err_base.dW.report(name, "px");
            std::snprintf(name, sizeof(name), "  约束拟合   Δ中心");
            err_fit.dc.report(name, "px");
            std::snprintf(name, sizeof(name), "  约束拟合   Δ方向");
            err_fit.dang.report(name, "deg");
            std::snprintf(name, sizeof(name), "  约束拟合   Δ长边");
            err_fit.dL.report(name, "px");
            std::snprintf(name, sizeof(name), "  约束拟合   Δ短边");
            err_fit.dW.report(name, "px");
        }
        printFitStats(0);
        return 0;
    }

    // ================= 模式 4：单帧转储（跨语言交叉验证用）=================
    // 输出第 N 帧（1 起）的敌方掩码 PNG 与本次提取的全部灯条几何（拟合开启），
    // 供 tools/verify_fit_constrained.py 用同一规格的 Python 复刻逐轮廓对比。
    int run_dump(const std::string &video, TeamColor enemy, long frame_no, const std::string &prefix)
    {
        cv::VideoCapture cap(video);
        if (!cap.isOpened())
        {
            std::fprintf(stderr, "无法打开视频: %s\n", video.c_str());
            return 2;
        }
        cv::Mat frame;
        long idx = 0;
        while (idx < frame_no && cap.read(frame))
        {
            ++idx;
        }
        if (idx < frame_no || frame.empty())
        {
            std::fprintf(stderr, "读取第 %ld 帧失败\n", frame_no);
            return 2;
        }
        g_bar_fit_mode = 3;
        const auto dets = armor_detect(frame, enemy);
        const auto bars = debugBars();
        // 计数快照：后面生成正则化掩码会再跑一次提取，避免重复计数
        const int c_total = g_bar_fit_total, c_accept = g_bar_fit_accept;
        const int c_pts = g_bar_fit_rej_pts, c_size = g_bar_fit_rej_size;
        const int c_g1 = g_bar_fit_rej_g1, c_g2 = g_bar_fit_rej_g2, c_g3 = g_bar_fit_rej_g3;
        const int c_edges = g_bar_fit_rej_edges, c_par = g_bar_fit_rej_par;
        const int c_width = g_bar_fit_rej_width;
        const cv::Mat mask = enemyMaskForDebug(frame, enemy);
        cv::imwrite(prefix + "_mask.png", mask);
        cv::imwrite(prefix + "_regmask.png", regularizedMaskForDebug(frame, enemy));

        {
            // 轮廓提取耗时（真实分辨率）：SIMPLE 一趟 vs 拟合额外需要的 NONE 一趟
            std::vector<std::vector<cv::Point>> c1, c2;
            const auto repeat = [&](int flag)
            {
                const auto t0 = std::chrono::steady_clock::now();
                for (int k = 0; k < 20; ++k)
                {
                    cv::findContours(mask, c1, cv::RETR_EXTERNAL, flag);
                }
                return std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - t0)
                           .count() / 20.0;
            };
            const double t_simple = repeat(cv::CHAIN_APPROX_SIMPLE);
            const double t_none = repeat(cv::CHAIN_APPROX_NONE);
            (void)c2;
            std::printf("轮廓提取耗时：SIMPLE %.3f ms/次 | NONE %.3f ms/次（%dx%d 掩码，各 20 次均值）\n",
                        t_simple, t_none, mask.cols, mask.rows);
        }

        FILE *fp = std::fopen((prefix + "_bars.txt").c_str(), "w");
        if (fp == nullptr)
        {
            std::fprintf(stderr, "无法写入 %s_bars.txt\n", prefix.c_str());
            return 2;
        }
        for (const DebugBar &b : bars)
        {
            std::fprintf(fp, "%.6f %.6f %.9f %.9f %.6f %.6f %d\n", b.center.x, b.center.y,
                         b.axis.x, b.axis.y, b.half_l, b.half_w, b.fitted ? 1 : 0);
        }
        std::fclose(fp);
        std::printf("帧 %ld：检出 %zu 条装甲板；灯条 %zu 条 → %s_mask.png / %s_bars.txt\n", frame_no,
                    dets.size(), bars.size(), prefix.c_str(), prefix.c_str());
        std::printf("本帧拟合计数: total=%d accept=%d rej: pts=%d size=%d g1=%d g2=%d g3=%d "
                    "edges=%d par=%d width=%d\n",
                    c_total, c_accept, c_pts, c_size, c_g1, c_g2, c_g3, c_edges, c_par, c_width);
        return 0;
    }

} // namespace

int main(int argc, char **argv)
{
    std::string video;
    TeamColor enemy = TeamColor::Red;
    long max_frames = 0;
    long range_lo = 1, range_hi = std::numeric_limits<long>::max();
    long dump_frame = 0;
    bool same_variant = false;
    int fit_variant = 3;
    std::string dump_prefix;
    std::string save_prefix;
    bool pose_mode = false;
    bool synth_mode = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--red")
        {
            enemy = TeamColor::Red;
        }
        else if (arg == "--blue")
        {
            enemy = TeamColor::Blue;
        }
        else if (arg == "--synth")
        {
            synth_mode = true;
        }
        else if (arg == "--pose")
        {
            pose_mode = true;
        }
        else if (arg == "--max-frames" && i + 1 < argc)
        {
            max_frames = std::stol(argv[++i]);
        }
        else if (arg == "--range" && i + 2 < argc)
        {
            range_lo = std::stol(argv[++i]);
            range_hi = std::stol(argv[++i]);
        }
        else if (arg == "--dump-frame" && i + 2 < argc)
        {
            dump_frame = std::stol(argv[++i]);
            dump_prefix = argv[++i];
        }
        else if (arg == "--save" && i + 1 < argc)
        {
            save_prefix = argv[++i];
        }
        else if (arg == "--same-variant")
        {
            same_variant = true;
        }
        else if (arg == "--fit-v1")
        {
            fit_variant = 1;
        }
        else if (arg == "--fit-v2")
        {
            fit_variant = 2;
        }
        else if (!arg.empty() && arg[0] != '-')
        {
            video = arg;
        }
        else
        {
            std::fprintf(stderr, "未知参数: %s\n", arg.c_str());
            return 2;
        }
    }

    if (synth_mode)
    {
        return run_synth(save_prefix);
    }
    if (dump_frame > 0 && !video.empty())
    {
        return run_dump(video, enemy, dump_frame, dump_prefix);
    }
    if (video.empty())
    {
        std::fprintf(stderr,
                     "用法: %s <视频> [--red|--blue] [--max-frames N] [--range A B] | <视频> --pose | "
                     "<视频> --dump-frame N 前缀 | --synth\n",
                     argv[0]);
        return 2;
    }
    return pose_mode ? run_pose(video, enemy, max_frames, range_lo, range_hi)
                     : run_ab(video, enemy, max_frames, range_lo, range_hi, same_variant,
                              fit_variant);
}
