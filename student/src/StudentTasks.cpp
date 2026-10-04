#include "StudentTasks.hpp"
#include "ArmorClassifier.hpp"
#include "MvCameraControl.h"

#include <array>
#include <cstddef>
#include <iostream>
#include <utility>
#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

//============ 初始化 ============
static bool s_initialized = false; // 只初始化一次
static bool s_ready = false;
static void *s_handle = nullptr; // MV_CC handle
static int s_lost_count = 0;
// ============ 辅助函数 ============
bool init_camera(void **handle)
{
    MV_CC_DEVICE_INFO_LIST device_list;
    memset(&device_list, 0, sizeof(device_list));
    int ret = MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list);
    if (ret != MV_OK || device_list.nDeviceNum == 0)
    {
        printf("未发现相机设备\n");
        return false;
    }

    // 直接取第一个设备（如果需要按序列号筛选，可保留原来的serial匹配逻辑）
    ret = MV_CC_CreateHandle(handle, device_list.pDeviceInfo[0]);
    if (ret != MV_OK)
    {
        printf("创建句柄失败 0x%x\n", ret);
        return false;
    }

    ret = MV_CC_OpenDevice(*handle);
    if (ret != MV_OK)
    {
        printf("打开设备失败 0x%x\n", ret);
        MV_CC_DestroyHandle(*handle);
        *handle = nullptr;
        return false;
    }

    // 参数设置（可改成从config读取）
    MV_CC_SetEnumValue(*handle, "TriggerMode", MV_TRIGGER_MODE_OFF);
    MV_CC_SetEnumValue(*handle, "ExposureAuto", 0);
    MV_CC_SetEnumValue(*handle, "GainAuto", 0);
    MV_CC_SetFloatValue(*handle, "ExposureTime", 10000.0f);
    MV_CC_SetFloatValue(*handle, "Gain", 10.0f);
    MV_CC_SetFloatValue(*handle, "AcquisitionFrameRate", 30.0f);
    MV_CC_SetEnumValue(*handle, "PixelFormat", PixelType_Gvsp_BayerRG8);

    ret = MV_CC_StartGrabbing(*handle);
    if (ret != MV_OK)
    {
        printf("开始取流失败 0x%x\n", ret);
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
//============ 主逻辑 ============
bool get_pic(cv::Mat &pic)
{
    // TODO(student)：在这里完成相机的一次性初始化/打开/启动，随后获取一帧并转换为 BGR。
    // ============ 首次调用 → 初始化相机 ============
    if (!s_initialized)
    {
        s_initialized = true;
        if (init_camera(&s_handle))
        { // ← 复用ros2代码中原来的 init_camera()
            s_ready = true;
            printf("相机连接成功\n");
        }
        else
        {
            printf("相机未连接，等待相机接入...\n");
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
        cv::Mat kernelclose = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        if (enemy == TeamColor::Red)
        {
            cv::Mat RMaskLow, RMaskHigh, RMask;
            cv::inRange(hsv, cv::Scalar(0, 100, 80), cv::Scalar(10, 255, 255), RMaskLow);
            cv::inRange(hsv, cv::Scalar(160, 100, 80), cv::Scalar(180, 255, 255), RMaskHigh);
            cv::bitwise_or(RMaskLow, RMaskHigh, RMask);
            cv::morphologyEx(RMask, opened, cv::MORPH_OPEN, kernelopen);
            cv::morphologyEx(RMask, closed, cv::MORPH_CLOSE, kernelclose);
            cv::imwrite("./tmp/mask_debug.png", RMask);
            return closed;
        }
        else
        {
            cv::Mat BMask;
            cv::inRange(hsv, cv::Scalar(100, 100, 80), cv::Scalar(130, 255, 255), BMask);
            cv::morphologyEx(BMask, opened, cv::MORPH_OPEN, kernelopen);
            cv::morphologyEx(BMask, closed, cv::MORPH_CLOSE, kernelclose);
            // cv::imwrite("./tmp/mask_debug.png", BMask);
            return closed;
        }
    }

    // 步骤 2：从掩码提取灯条。mask 为空时直接返回空。
    // 轮廓 -> minAreaRect -> 过滤条件（合成图验证过的起点值）：
    //   * 轮廓面积 >= ~20 px
    //   * 长边/短边 >= ~2.5          （灯条细长）
    //   * 轮廓面积 / 外接矩形面积 >= ~0.45（近似实心，排除零散亮斑）
    //   * 轴线与竖直方向夹角 <= ~30°
    //   * 长边 >= ~20 px
    // 把长短边统一为 height=长边，并算出 tilt_deg（注意 minAreaRect 的
    // 角度约定：宽>高时 angle 加 90 才是"离竖直的夹角"）。
    std::vector<LightBar> extractLightBars(const cv::Mat &mask)
    {
        // TODO
        std::vector<LightBar> bars;
        if (mask.empty() || mask.type() != CV_8UC1)
        {
            return bars;
        }

        // 1. 外轮廓：mask 是二值图，每个白色连通域对应一条轮廓。
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const std::vector<cv::Point> &contour : contours)
        {
            // 2. 面积粗滤：去掉孤立噪点。
            const double area = cv::contourArea(contour);
            if (area < 20.0)
            {
                continue;
            }

            // 3. 旋转外接矩形。不能用 boundingRect（轴对齐）：灯条倾斜时轴对齐框
            //    会变大变方，长宽比和角度全部失真。
            cv::RotatedRect rect = cv::minAreaRect(contour);
            float width = rect.size.width;
            float height = rect.size.height;
            float angle = rect.angle;

            // 4. 归一化：约定 height 为长边，并把角度折到 (-90, 90]。
            //    minAreaRect 的角度约定随 OpenCV 版本变过，论坛公式不可靠；
            //    补偿方向由实验校准：竖直灯条 tilt 应 ≈0，画面滚转 12° 的
            //    synth_tilted_2 应 ≈±12。若实测出现 90/180，调整补偿/折叠方向。
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
            const float tilt_deg = angle;

            // 5. 过滤：细长、实心、不太斜、够长（起点值，实拍时再调）。
            const float fill = static_cast<float>(area) /
                               (static_cast<float>(width) * static_cast<float>(height));
            if (height < 20.0F || height / width < 2.5F || fill < 0.45F ||
                tilt_deg > 30.0F || tilt_deg < -30.0F)
            {
                continue;
            }

            // 6. 填结构体。corners 为 4 个角点（顺序任意，配对与角点生成阶段
            //    再按 y 排序）。
            LightBar bar;
            bar.rect = rect;
            cv::Point2f pts[4];
            rect.points(pts);
            for (int k = 0; k < 4; ++k)
            {
                bar.corners[k] = pts[k];
            }
            bar.width = width;
            bar.height = height;
            bar.tilt_deg = tilt_deg;

            // std::cout << "bar: w=" << width << " h=" << height << " tilt=" << tilt_deg << "\n";

            bars.push_back(bar);
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
        // TODO(student)
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
                // 条件1
                if (std::min(L.height, R.height) / std::max(L.height, R.height) < 0.75)
                    continue;
                // 条件2
                double tilt_diff = std::fabs(L.tilt_deg - R.tilt_deg);
                if (tilt_diff > 12)
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
                if (d > 10)
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
        // 3. 灰度 + Otsu：cv::cvtColor → cv::threshold(..., THRESH_BINARY | THRESH_OTSU)
        cv::Mat gray, binary;
        cv::cvtColor(center, gray, cv::COLOR_BGR2GRAY);
        cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU); // 有 OTSU 标志时这个 0 只是占位，函数会自己算真阈值
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

        // 8. 裁界并返回（& 求交集，防止外扩越界；center 的坐标系即 binary 的坐标系）
        const cv::Rect crop = cv::Rect(x0, y0, x1 - x0, y1 - y0) &
                              cv::Rect(0, 0, center.cols, center.rows);
        return center(crop);
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
    const std::vector<LightBar> bars = extractLightBars(mask);
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

std::vector<ArmorPose> armor_solve(const std::vector<ArmorDetection> &,
                                   const CameraParameters &, const GimbalState &)
{
    // TODO(student)：按装甲板尺寸建立物点（毫米转换为米），调用 solvePnP，
    // 拒绝深度或重投影误差异常的结果，再应用已标定的刚体变换。
    return {};
}

PredictionResult ekf_predict(const std::vector<ArmorPose> &, const GimbalState &,
                             double)
{
    // TODO(student)：在未来状态预测前处理观测关联、初始化、角度归一化、离群点、
    // 装甲板切换、短时丢失以及超时重置。
    return {};
}
