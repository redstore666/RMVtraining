#include "StudentTasks.hpp"
#include "ArmorClassifier.hpp"
#include "MvCameraControl.h"

#include <array>
#include <cstddef>
#include <iostream>
#include <utility>
#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp> // solvePnP / projectPoints / Rodrigues
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

//============ 初始化 ============
static bool s_initialized = false; // 只初始化一次
static bool s_ready = false;
static void *s_handle = nullptr; // MV_CC handle
static int s_lost_count = 0;
static int s_init_attempt_count = 0;
// ============ 辅助函数 ============
bool init_camera(void **handle, bool verbose = false)
{

    if (*handle)
    {
        close_camera(*handle);
        *handle = nullptr;
    }

    MV_CC_DEVICE_INFO_LIST device_list;
    memset(&device_list, 0, sizeof(device_list));
    int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list);
    if (ret != MV_OK || device_list.nDeviceNum == 0)
    {
        if (verbose)
            printf("未发现相机设备\n");
        return false;
    }

    // 直接取第一个设备（如果需要按序列号筛选，可保留原来的serial匹配逻辑）
    ret = MV_CC_CreateHandle(handle, device_list.pDeviceInfo[0]);
    if (ret != MV_OK)
    {
        if (verbose)
            printf("创建句柄失败 0x%x\n", ret);
        return false;
    }

    ret = MV_CC_OpenDevice(*handle);
    if (ret != MV_OK)
    {
        if (verbose)
            printf("打开设备失败 0x%x\n", ret);
        MV_CC_DestroyHandle(*handle);
        *handle = nullptr;
        return false;
    }

    // 参数设置（可改成从config读取）-加入了对返回值的检查
    auto warn = [](int r, const char *name)
    {
        if (r != MV_OK)
            printf("警告：设置 %s 失败 0x%x\n", name, r);
    };
    warn(MV_CC_SetEnumValue(*handle, "TriggerMode", MV_TRIGGER_MODE_OFF), "TriggerMode");
    warn(MV_CC_SetEnumValue(*handle, "ExposureAuto", 0), "ExposureAuto");
    warn(MV_CC_SetEnumValue(*handle, "GainAuto", 0), "GainAuto");
    warn(MV_CC_SetFloatValue(*handle, "ExposureTime", 2000.0f), "ExposureTime");
    warn(MV_CC_SetFloatValue(*handle, "Gain", 10.0f), "Gain");
    warn(MV_CC_SetFloatValue(*handle, "AcquisitionFrameRate", 30.0f), "AcquisitionFrameRate");

    // PixelFormat 关键：失败就回滚整个 init
    if (MV_CC_SetEnumValue(*handle, "PixelFormat", PixelType_Gvsp_BayerRG8) != MV_OK)
    {
        if (verbose)
            printf("像素格式设置失败，回滚\n");
        MV_CC_CloseDevice(*handle);
        MV_CC_DestroyHandle(*handle);
        *handle = nullptr;
        return false;
    }

    ret = MV_CC_StartGrabbing(*handle);
    if (ret != MV_OK)
    {
        if (verbose)
            printf("开始取流失败 0x%x\n", ret);
        MV_CC_CloseDevice(*handle);
        MV_CC_DestroyHandle(*handle);
        *handle = nullptr;
        return false;
    }

    return true;
}

void close_camera(void *handle)
{
    if (handle)
    {
        MV_CC_StopGrabbing(handle);
        MV_CC_CloseDevice(handle);
        MV_CC_DestroyHandle(handle);
    }
}
void release_camera()
{
    if (s_handle)
    {
        close_camera(s_handle);
        s_handle = nullptr;
    }
    s_ready = false;
    s_initialized = false;
    s_lost_count = 0;
    s_init_attempt_count = 0;
}
// 画框函数
void drawDetections(cv::Mat &image, const std::vector<ArmorDetection> &detections)
{
    for (const ArmorDetection &det : detections)
    {
        std::vector<cv::Point> pts;
        for (const cv::Point2f &p : det.corners)
            pts.emplace_back(cvRound(p.x), cvRound(p.y));

        cv::polylines(image, pts, true, cv::Scalar(0, 255, 0), 2);
        cv::circle(image, pts[0], 5, cv::Scalar(0, 0, 255), -1); // 左上角红点

        std::string label = "ID=" + std::to_string(det.target_id) + (det.size == ArmorSize::Small ? " S" : " L") + " " + cv::format("%.2f", det.confidence);
        cv::putText(image, label,
                    pts[0] + cv::Point(0, -8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 0), 2);
    }
}
//============ 主逻辑 ============
bool get_pic(cv::Mat &pic)
{
    // tips：在这里完成相机的一次性初始化/打开/启动，随后获取一帧并转换为 BGR。
    // ============ 首次调用 → 初始化相机 ============
    if (!s_initialized)
    {
        ++s_init_attempt_count;
        // "第一次"才详细打印
        const bool first_try = (s_init_attempt_count == 1);
        if (first_try)
            printf("相机未连接，等待相机接入...\n");

        // 每 30 帧才真正尝试一次（约 1 秒，避免每帧枚举设备）
        if (s_init_attempt_count % 30 == 1 && init_camera(&s_handle, first_try))
        {
            s_initialized = true;
            s_ready = true;
            s_init_attempt_count = 0;
            printf("相机连接成功\n");
        }
    }
    // ============ 相机未就绪 → 返回 false ============
    if (!s_ready || !s_handle)
    {
        return false; // ← 对应 main.cpp 的失败计数逻辑
    }
    // ============ 取一帧 ============
    MV_FRAME_OUT_INFO_EX frame_info;
    memset(&frame_info, 0, sizeof(frame_info));

    // 动态分配足够大的缓冲区
    static std::vector<unsigned char> raw_buffer;
    raw_buffer.resize(4096 * 3000 * 3);

    int ret = MV_CC_GetOneFrameTimeout(
        s_handle,
        raw_buffer.data(),
        raw_buffer.size(),
        &frame_info,
        100); // 100ms 超时

    // 采集失败
    if (ret != MV_OK)
    {
        s_lost_count++;
        if (s_lost_count > 5)
        {
            close_camera(s_handle);
            s_handle = nullptr;
            s_ready = false;
            // 下次调用 get_pic 时会走重新初始化
            s_initialized = false; // ← 触发重新 init_camera()
            s_lost_count = 0;
        }
        pic.release();
        return false;
    }
    s_lost_count = 0;

    // ============ 像素格式转换：原始 Bayer → BGR ============
    static std::vector<unsigned char> bgr_buffer;
    bgr_buffer.resize(frame_info.nWidth * frame_info.nHeight * 3);

    MV_CC_PIXEL_CONVERT_PARAM cvt;
    memset(&cvt, 0, sizeof(cvt));
    cvt.nWidth = frame_info.nWidth;
    cvt.nHeight = frame_info.nHeight;
    cvt.pSrcData = raw_buffer.data();
    cvt.nSrcDataLen = frame_info.nFrameLen;
    cvt.enSrcPixelType = frame_info.enPixelType;
    cvt.enDstPixelType = PixelType_Gvsp_BGR8_Packed; // ← BGR，OpenCV默认
    cvt.pDstBuffer = bgr_buffer.data();
    cvt.nDstBufferSize = bgr_buffer.size();

    ret = MV_CC_ConvertPixelType(s_handle, &cvt);
    if (ret != MV_OK)
    {
        printf("像素转换失败 0x%x\n", ret);
        pic.release();
        return false;
    }
    // ============ 写入 cv::Mat ============
    pic = cv::Mat(
              frame_info.nHeight,
              frame_info.nWidth,
              CV_8UC3,
              bgr_buffer.data())
              .clone(); // clone：深拷贝，防止下次覆盖
    return true;
}

// ==================== 任务 2：装甲板识别 ====================
// 传统视觉路线：颜色掩码 -> 灯条提取 -> 灯条配对 -> 角点 -> 数字分类。
// 下面 6 个辅助函数是流水线的骨架（由 mentor 按已验证的方案搭好），
// 内部逻辑：每个 TODO 上方注释给出了过滤条件与依据。
// 完成后用 ./build/test_detect 在 tests/data/ 的合成图上验证，
// 再到相机实拍上调阈值。参考流程验证记录见 models/README.md。
namespace
{

    constexpr char kModelPath[] = "models/armor_cls_64.onnx";
    constexpr float kMinConfidence = 0.7F; // 现场按误检/漏检权衡调整

    struct LightBar
    {
        cv::RotatedRect rect;                 // minAreaRect 原始结果
        std::array<cv::Point2f, 4> corners{}; // rect 的四个角点
        float width{0.0F};                    // 归一化后：height 为长边
        float height{0.0F};
        float tilt_deg{0.0F}; // 相对竖直方向的夹角，竖直=0
    };

    // 步骤 1：敌方颜色掩码（HSV 路线）。
    // 流程：cv::cvtColor(pic, hsv, cv::COLOR_BGR2HSV) → 对敌方颜色做
    //   cv::inRange(hsv, 下界Scalar, 上界Scalar, mask) → 形态学清理（开 3x3 去噪、闭 5x5 补断缝）。
    // 关键点（HSV 经典坑，答辩可讲）：
    //   * OpenCV 的 H 范围是 0~179（不是 0~360）。红色横跨色相环两端：
    //     需要 [0,10] 与 [160,180) 两段 inRange 再 bitwise_or；蓝只需一段 ≈[100,130]。
    //     （实测合成图同一根红条上 H 同时出现 0 和 179。）
    //   * S 下界绝不能填 0：灰白像素 S≈0 且 H 记为 0，正好落进红色第一段区间。
    //     白光/数字区/灰背景全靠 S 下界排除（白色 R≈G≈B → S≈0）。
    //   * V 下界排除暗背景。
    //   * 阈值不是背出来的：用 tools/hsv_picker.py 在合成图上采样灯条/背景/数字
    //     的 H/S/V，按"采样 min/max ± 余量"定界；实拍后用同一工具重调。
    // 颜色掩码天然排除了白色的数字区域，因此数字不会被当成灯条。
    cv::Mat buildEnemyMask(const cv::Mat &bgr, TeamColor enemy)
    {
        // TODO
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        cv::Mat opened, closed;
        cv::Mat kernelopen = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat kernelclose = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 13));
        if (enemy == TeamColor::Red)
        {
            cv::Mat RMaskLow, RMaskHigh, RMask;
            cv::inRange(hsv, cv::Scalar(0, 100, 80), cv::Scalar(10, 255, 255), RMaskLow);
            cv::inRange(hsv, cv::Scalar(160, 100, 80), cv::Scalar(180, 255, 255), RMaskHigh);
            cv::bitwise_or(RMaskLow, RMaskHigh, RMask);
            cv::morphologyEx(RMask, opened, cv::MORPH_OPEN, kernelopen);
            cv::morphologyEx(RMask, closed, cv::MORPH_CLOSE, kernelclose);

            // ===== 新增：填孔洞 =====
            cv::Mat holes;
            cv::bitwise_not(closed, holes);
            cv::floodFill(holes, cv::Point(0, 0), cv::Scalar(0));
            cv::bitwise_or(closed, holes, closed);
            // =======================

            // cv::imwrite("./tmp/mask_debug.png", RMask);
            return closed;
        }
        else
        {
            cv::Mat BMask;
            cv::inRange(hsv, cv::Scalar(100, 100, 80), cv::Scalar(130, 255, 255), BMask);
            cv::morphologyEx(BMask, opened, cv::MORPH_OPEN, kernelopen);
            cv::morphologyEx(BMask, closed, cv::MORPH_CLOSE, kernelclose);

            // ===== 新增：填孔洞 =====
            cv::Mat holes;
            cv::bitwise_not(closed, holes);
            cv::floodFill(holes, cv::Point(0, 0), cv::Scalar(0));
            cv::bitwise_or(closed, holes, closed);
            // =======================

            // cv::imwrite("./tmp/mask_debug.png", BMask);
            return closed;
        }
    }

    // 把旋转矩形归一化并填入 LightBar：height=长边，tilt 折到 (-90, 90]。
    // （minAreaRect 角度约定随 OpenCV 版本变过，补偿方向已由实验校准。）
    void fillBarGeometry(LightBar &bar, const cv::RotatedRect &rect)
    {
        float width = rect.size.width;
        float height = rect.size.height;
        float angle = rect.angle;
        if (width > height)
        {
            std::swap(width, height);
            angle += 90.0F;
        }
        if (angle > 90.0F)
        {
            angle -= 180.0F;
        }
        else if (angle <= -90.0F)
        {
            angle += 180.0F;
        }
        bar.rect = rect;
        cv::Point2f pts[4];
        rect.points(pts);
        for (int k = 0; k < 4; ++k)
        {
            bar.corners[k] = pts[k];
        }
        bar.width = width;
        bar.height = height;
        bar.tilt_deg = angle;
    }

    // 灯条过滤：细长、够长、近似实心、不太斜。
    // 实拍调参：|tilt| 上限 30°→40°（实拍存在车体滚转约 40° 的大倾角场景）。
    bool passesBarFilter(const LightBar &bar, float fill)
    {
        return bar.height >= 20.0F && bar.width > 1e-3F && bar.height / bar.width >= 2.5F &&
               fill >= 0.45F && bar.tilt_deg <= 40.0F && bar.tilt_deg >= -40.0F;
    }

    // ===== 实拍调参（2026-10-09）：白芯辅助的灯条几何修复 =====
    // 实拍中 LED 光晕会把彩块撑宽、拉斜（minAreaRect 严重失真，实测可达 -65°）；
    // 过曝白芯（V 高、S 低）是发光条本体：窄、直、干净。对"不达标的彩块"用
    // 白芯修复几何：宽度/方向取白芯 minAreaRect，长度/端点取彩块像素在白芯轴线
    // 上的投影（两端各截尾 2%，裁掉光晕拖尾）。
    // 白芯阈值自适应降档：近条 V>=230 就够，远条白芯暗，需降到 V>=200/185。
    constexpr int kCoreLevels = 3;
    constexpr int kCoreVTh[kCoreLevels] = {230, 200, 185};
    constexpr int kCoreSTh[kCoreLevels] = {80, 110, 130};
    constexpr double kCoreMinArea = 100.0; // 白芯最小面积
    constexpr float kCoreMinLong = 25.0F;  // 白芯长边最小值
    constexpr float kCoreMinAspect = 2.0F; // 白芯长/短比最小值
    constexpr double kProjectTrim = 0.02;  // 投影截尾比例

    // 白芯掩码：V >= vth 且 S <= scap，小核闭运算补断缝。
    cv::Mat buildCoreMask(const cv::Mat &bgr, int vth, int scap)
    {
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        cv::Mat mask;
        cv::inRange(hsv, cv::Scalar(0, 0, vth), cv::Scalar(180, scap, 255), mask);
        const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        return mask;
    }

    // 白芯是否"像一根灯条芯"：够大、够长、够细长。
    bool usableCore(const std::vector<cv::Point> &core, float &long_side)
    {
        const cv::RotatedRect rect = cv::minAreaRect(core);
        long_side = std::max(rect.size.width, rect.size.height);
        const float short_side = std::min(rect.size.width, rect.size.height);
        return cv::contourArea(core) >= kCoreMinArea && long_side >= kCoreMinLong &&
               short_side > 1e-3F && long_side / short_side >= kCoreMinAspect;
    }

    // 用白芯修复彩块几何。color_contour：光晕彩块轮廓；core_contour：有效白芯轮廓。
    bool refineBarWithCore(const std::vector<cv::Point> &color_contour,
                           const std::vector<cv::Point> &core_contour, LightBar &bar)
    {
        const cv::RotatedRect core_rect = cv::minAreaRect(core_contour);
        float core_w = std::min(core_rect.size.width, core_rect.size.height);
        if (core_w < 1e-3F)
        {
            return false;
        }
        // 长边方向 = 矩形四条边中最长边的单位方向
        cv::Point2f cpts[4];
        core_rect.points(cpts);
        cv::Point2f axis(0.0F, 1.0F);
        cv::Point2f wvec(1.0F, 0.0F);
        float best_len = -1.0F;
        for (int k = 0; k < 4; ++k)
        {
            const cv::Point2f edge = cpts[(k + 1) % 4] - cpts[k];
            const float len = cv::norm(edge);
            if (len > best_len)
            {
                best_len = len;
                axis = edge * (1.0F / std::max(len, 1e-6F));
                wvec = cv::Point2f(-axis.y, axis.x);
            }
        }
        // 彩块像素投影到轴线，2%~98% 截尾得到长度与端点
        std::vector<float> t_values;
        t_values.reserve(color_contour.size());
        for (const cv::Point &p : color_contour)
        {
            t_values.push_back((cv::Point2f(p) - core_rect.center).dot(axis));
        }
        if (t_values.size() < 4)
        {
            return false;
        }
        std::sort(t_values.begin(), t_values.end());
        const std::size_t n = t_values.size();
        const std::size_t i0 = static_cast<std::size_t>(kProjectTrim * static_cast<double>(n));
        const float t0 = t_values[i0];
        const float t1 = t_values[n - 1 - i0];
        const float half = std::max((t1 - t0) / 2.0F, 1.0F);
        const cv::Point2f center = core_rect.center + axis * ((t0 + t1) / 2.0F);
        const float half_w = core_w / 2.0F;
        const std::array<cv::Point2f, 4> box = {
            center + axis * half + wvec * half_w, center - axis * half + wvec * half_w,
            center - axis * half - wvec * half_w, center + axis * half - wvec * half_w};
        fillBarGeometry(bar, cv::minAreaRect(box)); // 用同一约定重算，保证角度符号一致
        return true;
    }

    // 步骤 2：从掩码提取灯条。
    // 轮廓 -> minAreaRect -> 过滤。实拍调参后的策略：
    //   * 彩块本身达标（细长/实心/|tilt|<=40°）→ 直接采用彩块几何；
    //   * 不达标（被光晕撑宽的胖块/被拖尾拉斜的块）→ 用白芯做几何修复（见上方说明），
    //     修复结果仍不达标则丢弃。白芯缺失（如合成图纯色条）时自然走彩块路径。
    std::vector<LightBar> extractLightBars(const cv::Mat &bgr, const cv::Mat &mask)
    {
        std::vector<LightBar> bars;
        if (mask.empty() || mask.type() != CV_8UC1)
        {
            return bars;
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 白芯轮廓按档位懒加载（只有出现不达标彩块时才生成对应档位的掩码）
        std::array<std::vector<std::vector<cv::Point>>, kCoreLevels> core_contours;
        std::array<bool, kCoreLevels> core_ready{false, false, false};

        for (const std::vector<cv::Point> &contour : contours)
        {
            if (cv::contourArea(contour) < 20.0)
            {
                continue;
            }
            LightBar bar;
            fillBarGeometry(bar, cv::minAreaRect(contour));
            const float fill = static_cast<float>(cv::contourArea(contour)) /
                               (bar.width * bar.height);

            // 达标不修：避免短噪白芯把本就良好的几何带偏（实测教训）
            if (passesBarFilter(bar, fill))
            {
                bars.push_back(bar);
                continue;
            }
            if (bgr.empty())
            {
                continue;
            }

            for (int level = 0; level < kCoreLevels; ++level)
            {
                if (!core_ready[level])
                {
                    cv::Mat core_mask = buildCoreMask(bgr, kCoreVTh[level], kCoreSTh[level]);
                    cv::findContours(core_mask, core_contours[level], cv::RETR_EXTERNAL,
                                     cv::CHAIN_APPROX_SIMPLE);
                    core_ready[level] = true;
                }
                // 取与该彩块重叠的"有效白芯"中面积最大者
                const std::vector<cv::Point> *best_core = nullptr;
                double best_area = 0.0;
                for (const std::vector<cv::Point> &core : core_contours[level])
                {
                    float long_side = 0.0F;
                    if (!usableCore(core, long_side))
                    {
                        continue;
                    }
                    const cv::Point2f core_center = cv::minAreaRect(core).center;
                    if (cv::pointPolygonTest(contour, core_center, false) < 0)
                    {
                        continue;
                    }
                    const double core_area = cv::contourArea(core);
                    if (core_area > best_area)
                    {
                        best_area = core_area;
                        best_core = &core;
                    }
                }
                if (best_core != nullptr)
                {
                    LightBar refined;
                    if (refineBarWithCore(contour, *best_core, refined))
                    {
                        // 修复后 fill 用白芯自身实心度（白芯近似实心柱）
                        const cv::RotatedRect crect = cv::minAreaRect(*best_core);
                        const float cw = std::min(crect.size.width, crect.size.height);
                        const float ch = std::max(crect.size.width, crect.size.height);
                        const float rfill =
                            cw > 1e-3F && ch > 1e-3F
                                ? static_cast<float>(best_area) / (cw * ch)
                                : 0.0F;
                        if (passesBarFilter(refined, std::min(rfill, 1.0F)))
                        {
                            bars.push_back(refined);
                        }
                    }
                    break; // 只用最严档位里找到的有效白芯（与调参实验一致）
                }
            }
        }
        return bars;
    }

    // 返回灯条 4 个角点按 y 从小到大排序后的副本。
    // 角点的"上组/下组"划分在配对、角点生成、大小判定里都要用——排序逻辑只写一次。
    std::array<cv::Point2f, 4> cornersSortedByY(const LightBar &bar)
    {
        std::array<cv::Point2f, 4> pts = bar.corners;
        std::sort(pts.begin(), pts.end(),
                  [](const cv::Point2f &a, const cv::Point2f &b)
                  { return a.y < b.y; });
        return pts;
    }

    // 步骤 3：灯条配对，返回"左灯条下标, 右灯条下标"对。
    // 对每一对 (i, j)（i 的 x 更小），全部满足才配对：
    //   * 高度比 min/max >= ~0.75
    //   * 灯条自身倾斜角差 <= ~12°
    //   * 上端点连线与下端点连线近似平行（夹角差 <= ~10°）且长度比 >= ~0.7
    //     （"连线 ⊥ 灯条"的物理约束；用平行性表达可以在画面旋转时同样成立，
    //      也顺便排除了两根竖直灯条却上下错位很远的跨目标误配）
    //   * 两灯条之间没有其它灯条
    //   * 外缘间距 / 平均灯条高 在 [~1.5, ~5.5]（粗过滤异常间距）
    std::vector<std::pair<std::size_t, std::size_t>> matchLightBars(
        const std::vector<LightBar> &bars)
    {
        // tips
        std::vector<std::pair<std::size_t, std::size_t>> pairs;
        for (std::size_t i = 0; i < bars.size(); ++i)
        {
            for (std::size_t j = i + 1; j < bars.size(); ++j)
            {
                // 左右判定
                std::size_t li = i, ri = j;
                if (bars[li].rect.center.x > bars[ri].rect.center.x)
                {
                    std::swap(li, ri); // 保证 li 是左
                }
                const LightBar &L = bars[li];
                const LightBar &R = bars[ri];
                // 条件1（实拍调参：高度比 0.75→0.65，近距离强透视下近杆可比远杆大 30%+）
                if (std::min(L.height, R.height) / std::max(L.height, R.height) < 0.65)
                    continue;
                // 条件2（实拍调参：倾角差 12°→16°，白芯截断会让两杆倾角估计出现偏差）
                double tilt_diff = std::fabs(L.tilt_deg - R.tilt_deg);
                if (tilt_diff > 16)
                    continue;
                // 排序角点（公共助手，见 cornersSortedByY）
                const std::array<cv::Point2f, 4> lc = cornersSortedByY(L);
                const std::array<cv::Point2f, 4> rc = cornersSortedByY(R);

                cv::Point2f L_top = (lc[0] + lc[1]) * 0.5F;
                cv::Point2f L_bot = (lc[2] + lc[3]) * 0.5F;
                cv::Point2f R_top = (rc[0] + rc[1]) * 0.5F;
                cv::Point2f R_bot = (rc[2] + rc[3]) * 0.5F;
                // 条件3
                cv::Point2f vt = R_top - L_top;
                cv::Point2f vb = R_bot - L_bot;
                double angt = std::atan2(vt.y, vt.x) * 180.0 / CV_PI;
                double angb = std::atan2(vb.y, vb.x) * 180.0 / CV_PI;
                double d = std::fabs(angt - angb);
                if (d > 180.0)
                    d = 360.0 - d; // 现在才是真正的"两向量夹角"
                if (d > 15)        // 实拍调参：平行阈值 10°→15°（与倾角差放宽配套）
                    continue;
                double lt = cv::norm(vt);                               // 上端点连线的长度
                double lb = cv::norm(vb);                               // 下端点连线的长度
                double len_ratio = std::min(lt, lb) / std::max(lt, lb); // 恒在 (0, 1]
                if (len_ratio < 0.7)
                    continue;
                // 条件④：两灯条之间没有其它灯条
                bool inner_ok = true;
                for (std::size_t k = 0; k < bars.size(); ++k)
                {
                    if (k == li || k == ri)
                    {
                        continue; // 跳过这一对自身
                    }
                    const LightBar &M = bars[k];
                    const bool between_x = M.rect.center.x > L.rect.center.x &&
                                           M.rect.center.x < R.rect.center.x;
                    const bool between_y = M.rect.center.y > std::min(L_top.y, R_top.y) &&
                                           M.rect.center.y < std::max(L_bot.y, R_bot.y);
                    if (between_x && between_y)
                    {
                        inner_ok = false;
                        break; // 找到一个第三者就够了
                    }
                }
                if (!inner_ok)
                {
                    continue;
                }
                // 条件⑤：外缘间距（连线中值 + 两个半条宽）与灯条高之比
                const double outer_w = (lt + lb) / 2.0 + (L.width + R.width) / 2.0;
                const double h_avg = (L.height + R.height) / 2.0;
                const double dist_ratio = outer_w / h_avg;
                if (dist_ratio < 1.5 || dist_ratio > 5.5)
                {
                    continue;
                }

                // 验证通过
                // std::cout << "pair " << li << "-" << ri
                //<< " ratio=" << dist_ratio << "\n";
                pairs.emplace_back(li, ri);
            }
        }

        return pairs;
    }

    // 步骤 4：由一对灯条生成装甲板四角点，顺序：左上、右上、右下、左下（顺时针，与
    // ArmorDetection::corners 约定一致）。取"外缘角点"：
    //   * 每个灯条的 4 个角点按 y 排序：上两个一组、下两个一组；
    //   * 左灯条：上/下两组中各取 x 较小者为外角；右灯条取 x 较大者。
    std::array<cv::Point2f, 4> makeArmorCorners(const LightBar &left, const LightBar &right)
    {
        // 上组/下组划分交给公共助手，再在每组内取"外缘"角点：
        // 左灯条取 x 较小者（TL/BL），右灯条取 x 较大者（TR/BR）。
        const std::array<cv::Point2f, 4> lc = cornersSortedByY(left);
        const std::array<cv::Point2f, 4> rc = cornersSortedByY(right);

        const cv::Point2f TL = lc[0].x < lc[1].x ? lc[0] : lc[1]; // 上组，靠左外缘
        const cv::Point2f TR = rc[0].x > rc[1].x ? rc[0] : rc[1]; // 上组，靠右外缘
        const cv::Point2f BR = rc[2].x > rc[3].x ? rc[2] : rc[3]; // 下组，靠右外缘
        const cv::Point2f BL = lc[2].x < lc[3].x ? lc[2] : lc[3]; // 下组，靠左外缘

        return {TL, TR, BR, BL}; // 左上、右上、右下、左下（顺时针，与 PnP 物点对应）
    }

    // 步骤 5：大小装甲板判定。
    // 依据 装甲板尺寸.txt：小 135x57、大 230x57（mm），即
    // 外缘间距/灯条高：小 ≈ 2.37、大 ≈ 4.04。取更接近的档位。
    // 实现提示：先调 makeArmorCorners(left, right) 拿到四角点（复用，别重算），
    //   上外缘长 = |TL-TR|、下外缘长 = |BL-BR|（cv::norm），
    //   两者均值 / 平均灯条高 得比值，再与 2.37 / 4.04 择近。
    ArmorSize classifyArmorSize(const LightBar &left, const LightBar &right)
    {
        const std::array<cv::Point2f, 4> c = makeArmorCorners(left, right); // 复用四角点

        const double top_edge = cv::norm(c[0] - c[1]);    // 上外缘长 TL→TR
        const double bottom_edge = cv::norm(c[3] - c[2]); // 下外缘长 BL→BR
        const double outer_w = (top_edge + bottom_edge) / 2.0;
        const double h_avg = (left.height + right.height) / 2.0;
        const double ratio = outer_w / h_avg;

        // 标称比值来自 装甲板尺寸.txt；实拍因光晕/灯条倾角会略偏，必要时微调常数。
        const double dist_small = std::fabs(ratio - 2.37);
        const double dist_large = std::fabs(ratio - 4.04);
        return dist_small <= dist_large ? ArmorSize::Small : ArmorSize::Large;
    }

    // 步骤 6：数字区域裁剪（分类器输入）。
    //   * 用四角点做透视变换拉正（如 270x114，比例同装甲板），
    //     取中央约 70% 宽度，把两侧灯条切掉；
    //   * 灰度 + Otsu 二值化后做连通域，取面积最大者（以及面积 >= 最大者 10%
    //     的连通域并集）的外接矩形，外扩约 15%；
    //   * 这一步必须"紧贴数字"：训练裁片里数字几乎充满画面，若数字占比太小
    //     （实测 < ~10% 画面）分类器会误判为 9neg。
    cv::Mat cropNumberRegion(const cv::Mat &bgr, const std::array<cv::Point2f, 4> &corners)
    {
        // 1. 透视拉正：目标 270x114（与装甲板同比例）
        // 源点：corners 顺序是 TL,TR,BR,BL —— 用 C 数组最稳（getPerspectiveTransform 要 float 点）
        cv::Point2f src[4];
        for (int k = 0; k < 4; ++k)
        {
            src[k] = corners[k];
        }
        // 目标点：与源点一一对应，TL→左上角、TR→右上角、BR→右下角、BL→左下角
        const cv::Point2f dst[4] = {
            {0.0F, 0.0F}, {270.0F, 0.0F}, {270.0F, 114.0F}, {0.0F, 114.0F}};

        const cv::Mat h = cv::getPerspectiveTransform(src, dst); // 3x3 单应矩阵
        cv::Mat rectified;
        cv::warpPerspective(bgr, rectified, h, cv::Size(270, 114)); // 注意 Size 是(宽,高)
        // 2. 切掉两侧灯条：取中央 70% 宽
        const int left_cut = static_cast<int>(270 * 0.15);
        const int right_cut = static_cast<int>(270 * 0.85); // ≈229
        const cv::Mat center = rectified.colRange(left_cut, right_cut);
        // center：189 宽 × 114 高，正好是中央 ~70
        // 3. 灰度预处理。
        // 实拍调参：彩色光晕会污染普通灰度（红/蓝光在加权灰度里仍偏亮，Otsu 会
        // 割在光晕上导致数字残缺）；改用 min(B,G,R) 通道——任何饱和彩色（红/蓝
        // 光晕）至少有一个低通道，在 min 通道里必然变暗；灰白数字 R≈G≈B 不受损。
        cv::Mat channels[3];
        cv::split(center, channels);
        const cv::Mat channel_min = cv::min(channels[0], channels[1]);
        const cv::Mat gray = cv::min(channel_min, channels[2]);
        cv::Mat binary;
        cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
        //  4. 连通域：
        cv::Mat labels, stats, centroids;
        const int n = cv::connectedComponentsWithStats(binary, labels, stats, centroids);
        // 5. 找最大面积的前景块（第 0 行是背景，跳过）
        int max_area = 0;
        for (int i = 1; i < n; ++i)
        {
            max_area = std::max(max_area, stats.at<int>(i, 4)); // 4 = CC_STAT_AREA
        }
        if (n <= 1 || max_area < 20)
        {
            return {}; // 中间没有像样的前景，交给下游按"无效观测"处理
        }

        // 6. 并集所有面积 >= 最大块 10% 的块的外接框：
        //    数字笔画可能被二值化打散，只取最大块会切掉笔画；全取会混入噪声
        int x0 = center.cols, y0 = center.rows, x1 = -1, y1 = -1;
        for (int i = 1; i < n; ++i)
        {
            const int area = stats.at<int>(i, 4);
            const int left = stats.at<int>(i, 0);
            const int top = stats.at<int>(i, 1);
            const int w = stats.at<int>(i, 2);
            const int h = stats.at<int>(i, 3);
            if (area < 0.10 * max_area)
            {
                continue;
            }
            x0 = std::min(x0, left);
            y0 = std::min(y0, top);
            x1 = std::max(x1, left + w);
            y1 = std::max(y1, top + h);
        }
        if (x1 < 0)
        {
            return {}; // 防御：没有任何块入选
        }

        // 7. 外扩 15%：训练裁片里数字几乎充满画面，留一点呼吸边贴近训练分布；
        //    扩太多则数字占比掉到 10% 以下，分类器会判 9neg
        const int margin = static_cast<int>(0.15 * std::max(x1 - x0, y1 - y0));
        x0 -= margin;
        y0 -= margin;
        x1 += margin;
        y1 += margin;

        // 8. 裁界并返回（& 求交集，防止外扩越界；crop 的坐标系即 gray 的坐标系）。
        // 实拍调参：返回 min 通道裁剪（单通道），分类器对 1/3 通道输入都兼容——
        // 必须返回 min 通道本身，否则分类器内部又转回被光晕污染的灰度。
        const cv::Rect crop = cv::Rect(x0, y0, x1 - x0, y1 - y0) &
                              cv::Rect(0, 0, center.cols, center.rows);
        return gray(crop);
    }

} // namespace

std::vector<ArmorDetection> armor_detect(const cv::Mat &pic, TeamColor enemy_color)
{
    std::vector<ArmorDetection> detections;
    if (pic.empty() || pic.type() != CV_8UC3)
    {
        return detections;
    }

    // 分类器只加载一次（首次调用时）。
    static ArmorClassifier classifier;
    static bool load_attempted = false;
    if (!load_attempted)
    {
        load_attempted = true;
        std::string error;
        if (!classifier.load(kModelPath, error) &&
            !classifier.load(std::string("../") + kModelPath, error))
        {
            std::cerr << "数字分类器加载失败: " << error << "\n";
        }
    }

    const cv::Mat mask = buildEnemyMask(pic, enemy_color);
    const std::vector<LightBar> bars = extractLightBars(pic, mask);
    const std::vector<std::pair<std::size_t, std::size_t>> pairs = matchLightBars(bars);

    for (const std::pair<std::size_t, std::size_t> &pair : pairs)
    {
        const LightBar &left = bars[pair.first];
        const LightBar &right = bars[pair.second];
        const std::array<cv::Point2f, 4> corners = makeArmorCorners(left, right);

        // 拒绝超出图像范围的角点。
        bool inside = true;
        for (const cv::Point2f &p : corners)
        {
            if (p.x < 0.0F || p.y < 0.0F || p.x >= static_cast<float>(pic.cols) ||
                p.y >= static_cast<float>(pic.rows))
            {
                inside = false;
                break;
            }
        }
        if (!inside)
        {
            continue;
        }

        if (!classifier.ready())
        {
            continue; // 分类器不可用时不输出未经确认的装甲板
        }

        // 数字分类；负类（9neg）或置信度不足则丢弃。
        const cv::Mat number = cropNumberRegion(pic, corners);
        const ArmorClassifier::Result classified = classifier.infer(number);
        if (classified.target_id == 0 || classified.confidence < kMinConfidence)
        {
            continue;
        }

        ArmorDetection detection;
        detection.corners = corners;
        detection.size = classifyArmorSize(left, right);
        detection.target_id = classified.target_id;
        detection.confidence = classified.confidence;
        detections.push_back(detection);
    }

    // 可选：按置信度从高到低排序、限制返回数量。
    return detections;
}

// ==================== 任务 3：位姿解算 ====================
// 输入任务 2 的检测结果，输出每个装甲板的位姿观测（ArmorPose）。
// 数据流：物点建模（米）→ solvePnP → 重投影自检 → 门禁 → 外参变换到云台系。
// 四个输出含义：position_camera/gimbal_m（在哪）、armor_yaw（什么朝向）、
// reprojection_error（这次观测可信吗）——后面 EKF 全靠这四样。
namespace
{

    // 物点建模：某尺寸装甲板的 4 个三维角点，单位米，顺序必须与
    // ArmorDetection::corners 一致（TL, TR, BR, BL）。
    // 装甲板坐标系：原点在板面中心，x 向右，y 向下，z 沿板面外法向。
    // 尺寸来自 装甲板尺寸.txt：小 135x57、大 230x57（毫米→米，除以 1000！）。
    // 记住"图像点与物点同源"原则：物点是外缘尺寸，corners 也是外缘角点。
    std::vector<cv::Point3f> buildObjectPoints(ArmorSize size)
    {
        // tips：按 size 取半宽 half_w、半高 half_h（米），返回 4 个点：
        //   TL(-half_w, -half_h, 0)  TR(+half_w, -half_h, 0)
        //   BR(+half_w, +half_h, 0)  BL(-half_w, +half_h, 0)
        double half_w = 0.0, half_h = 0.057f / 2;
        if (size == ArmorSize::Small)
            half_w = 0.135f / 2;
        else
            half_w = 0.230f / 2;
        const cv::Point3f TL(-half_w, -half_h, 0.0F);
        const cv::Point3f TR(+half_w, -half_h, 0.0F);
        const cv::Point3f BR(+half_w, +half_h, 0.0F);
        const cv::Point3f BL(-half_w, +half_h, 0.0F);
        return {TL, TR, BR, BL};
    }

    // 门禁阈值（现场可调；任务书要求拒绝深度非法或误差异常的观测）
    constexpr double kMaxReprojErrorPx = 3.0; // 重投影误差上限,阈值按数据分布定，实拍再调
    constexpr double kMinDepthM = 0.05;       // 深度下限（近于 5cm 视为异常）
    constexpr double kMaxDepthM = 20.0;       // 深度上限

} // namespace

std::vector<ArmorPose> armor_solve(const std::vector<ArmorDetection> &detections,
                                   const CameraParameters &camera,
                                   const GimbalState &gimbal)
{
    (void)gimbal; // 云台姿态补偿属于 EKF/预测阶段，此处不用
    std::vector<ArmorPose> poses;

    for (const ArmorDetection &det : detections)
    {
        const std::vector<cv::Point3f> object_points = buildObjectPoints(det.size);
        if (object_points.empty())
        {
            continue; // 物点未建模（TODO 未完成）时静默跳过
        }

        // 图像点：corners 本身就是 TL,TR,BR,BL 顺序，直接搬进 vector
        const std::vector<cv::Point2f> image_points(det.corners.begin(),
                                                    det.corners.end());

        // PnP：4 个共面点。SOLVEPNP_IPPE 是专为共面目标设计的解法，
        // 自动在两组镜像解中取重投影更优者。
        cv::Mat rvec, tvec;
        const bool ok = cv::solvePnP(object_points, image_points,
                                     camera.camera_matrix,
                                     camera.distortion_coefficients, rvec, tvec,
                                     false, cv::SOLVEPNP_IPPE);
        if (!ok)
        {
            continue; // PnP 失败 → 空观测，交给 EKF 纯预测
        }

        // 重投影自检：用解出的位姿把物点投回图像，与实际角点比较
        std::vector<cv::Point2f> projected;
        cv::projectPoints(object_points, rvec, tvec, camera.camera_matrix,
                          camera.distortion_coefficients, projected);
        double err_sq = 0.0;
        for (std::size_t k = 0; k < projected.size(); ++k)
        {
            const double e = cv::norm(projected[k] - image_points[k]);
            err_sq += e * e;
        }
        const double reproj_error =
            std::sqrt(err_sq / static_cast<double>(projected.size()));

        // tips 门禁：以下任一不满足就 continue（拒绝该观测）——
        //   1) reproj_error > kMaxReprojErrorPx
        //   2) 深度 z = tvec.at<double>(2) 超出 [kMinDepthM, kMaxDepthM]
        //   3) 数值非有限（std::isfinite 检查 x/y/z 与 reproj_error）
        const bool all_finite = std::isfinite(reproj_error) &&
                                std::isfinite(tvec.at<double>(0)) &&
                                std::isfinite(tvec.at<double>(1)) &&
                                std::isfinite(tvec.at<double>(2));
        if (!all_finite)
            continue;
        if (reproj_error > kMaxReprojErrorPx ||
            tvec.at<double>(2) < kMinDepthM || tvec.at<double>(2) > kMaxDepthM)
            continue;

        ArmorPose pose;
        pose.detection = det;
        pose.reprojection_error = reproj_error;
        pose.position_camera_m = cv::Vec3d(tvec); // tvec 即板心在相机系坐标（米）

        // 外参变换：p_gimbal = R * p_camera + t（单位外参时两者相同）
        if (camera.rotation_camera_to_gimbal.empty())
        {
            pose.position_gimbal_m = pose.position_camera_m; // 未标定外参的防御分支
        }
        else
        {
            const cv::Mat p_cam_mat(pose.position_camera_m);
            const cv::Mat p_gim_mat = camera.rotation_camera_to_gimbal * p_cam_mat +
                                      camera.translation_camera_to_gimbal;
            pose.position_gimbal_m = cv::Vec3d(p_gim_mat);
        }

        // armor_yaw：板面法向相对光轴的水平偏角（弧度）。
        //   1) cv::Rodrigues(rvec, R_obj2cam) 把旋转向量变成 3x3 旋转矩阵
        //   2) 板面法向在相机系：n = R_obj2cam * (0, 0, 1)
        //   3) pose.armor_yaw = std::atan2(n.x, n.z)（合成图上应 ≈0；
        //      符号约定用实拍/带 yaw 的场景实证校准，方法同 tilt_deg）
        cv::Mat R_obj2cam;
        cv::Rodrigues(rvec, R_obj2cam);                                          // 旋转向量 → 3x3 矩阵
        const cv::Mat n = R_obj2cam * (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0); // 板面法向转到相机系
        // cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0是OpenCV 的逗号初始化语法,专门用来写小矩阵
        pose.armor_yaw = std::atan2(n.at<double>(0), n.at<double>(2)); // n.x, n.z

        poses.push_back(pose);
    }
    return poses;
}

PredictionResult ekf_predict(const std::vector<ArmorPose> &, const GimbalState &,
                             double)
{
    // tips：在未来状态预测前处理观测关联、初始化、角度归一化、离群点、
    // 装甲板切换、短时丢失以及超时重置。
    return {};
}
