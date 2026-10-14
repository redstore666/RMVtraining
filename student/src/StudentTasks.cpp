#include "StudentTasks.hpp"
#include "ArmorClassifier.hpp"
#include "MvCameraControl.h"

#include <array>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <vector> //test

#include <opencv2/calib3d.hpp> // solvePnP / projectPoints / Rodrigues
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

// test the average err
std::vector<double> g_reproj_errors;

//============ 初始化 ============
static bool s_initialized = false; // 只初始化一次
static bool s_ready = false;
static void *s_handle = nullptr; // MV_CC handle
static int s_lost_count = 0;
static int s_init_attempt_count = 0;

// ===== test计数器（供 test_camera_live 打印）=====
int g_rej_reproj = 0;
int g_rej_depth = 0;
int g_rej_finite = 0;
int g_pnp_fail = 0;

// ===== test钩子（需求③：灯条约束拟合；说明见 StudentTasks.hpp）=====
// 0=基线 minAreaRect；1=矩形拟合 v1；2=四边独立求交 v2；3=四边约束求交 v3（默认）。
// RMV_BAR_FIT=0/off、1/v1、2/v2 可切换（供离线 A/B）。
int g_bar_fit_mode = [] {
    const char *env = std::getenv("RMV_BAR_FIT");
    if (env == nullptr)
    {
        return 3;
    }
    if (std::strcmp(env, "0") == 0 || std::strcmp(env, "off") == 0)
    {
        return 0;
    }
    if (std::strcmp(env, "1") == 0 || std::strcmp(env, "v1") == 0)
    {
        return 1;
    }
    if (std::strcmp(env, "2") == 0 || std::strcmp(env, "v2") == 0)
    {
        return 2;
    }
    return 3;
}();
int g_bar_fit_total = 0;
int g_bar_fit_accept = 0;
int g_bar_fit_rej_pts = 0;
int g_bar_fit_rej_size = 0;
int g_bar_fit_rej_g1 = 0;
int g_bar_fit_rej_g2 = 0;
int g_bar_fit_rej_g3 = 0;
int g_bar_fit_rej_edges = 0;
int g_bar_fit_rej_par = 0;
int g_bar_fit_rej_width = 0;
double g_bar_fit_ms_total = 0.0;

// ===== test钩子（需求③ P3：宽度短时域稳健化；说明见 StudentTasks.hpp）=====
// 窗口长度上限 5（见 StudentTasks.cpp 内 P3 区块的机制说明）；RMV_BAR_WIDTH_WIN=0 关闭。
constexpr int kWidthWindowMax = 5;
int g_bar_width_window = [] {
    const char *env = std::getenv("RMV_BAR_WIDTH_WIN");
    if (env == nullptr)
    {
        return kWidthWindowMax;
    }
    const int v = std::atoi(env);
    return v <= 0 ? 0 : std::min(v, kWidthWindowMax);
}();
int g_bar_width_smooth_count = 0;
int g_bar_width_reset_count = 0;
// ============ 辅助函数 ============
bool init_camera(void **handle, bool verbose)
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
    warn(MV_CC_SetFloatValue(*handle, "ExposureTime", 3000.0f), "ExposureTime");
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
        // 画框内缩（学长要求"识别框放中间"）：左右边从灯条外缘收到灯条中线附近。
        // 内缩比例 ≈ 半条宽 / 外缘宽：小装甲板 ~4%、大装甲板 ~2%。
        // 注意：只影响显示；检测角点、PnP、数字裁剪仍使用外缘角点
        // （与物点模型 135/230mm 外缘尺寸一致，见 makeArmorCorners 注释）。
        const float inset = (det.size == ArmorSize::Small) ? 0.04F : 0.02F;
        std::vector<cv::Point> pts;
        for (int k = 0; k < 4; ++k)
        {
            const cv::Point2f a = det.corners[k];
            const int neighbor = (k % 2 == 0) ? k + 1 : k - 1; // TL↔TR、BR↔BL
            const cv::Point2f p = a + (det.corners[neighbor] - a) * inset;
            pts.emplace_back(cvRound(p.x), cvRound(p.y));
        }

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
    constexpr float kMinConfidence = 0.9F; // 现场按误检/漏检权衡调整

    struct LightBar
    {
        cv::RotatedRect rect;                 // minAreaRect 原始结果（宽度平滑后就地重算）
        std::array<cv::Point2f, 4> corners{}; // 四角（拟合 v2/v3 采纳时为四边形交点）
        bool fitted{false};                   // true=几何来自 fitBarConstrained 采纳
        float width{0.0F};                    // 归一化后：height 为长边
        float height{0.0F};
        float tilt_deg{0.0F}; // 相对竖直方向的夹角，竖直=0
    };

    // 步骤 1：敌方颜色掩码（HSV 路线）。
    // 流程：cv::cvtColor(pic, hsv, cv::COLOR_BGR2HSV) → 对敌方颜色做
    //   cv::inRange(hsv, 下界Scalar, 上界Scalar, mask) → 形态学清理（闭 13x13 补断缝）
    //   → RGB 通道差补充暗灯条 → floodFill 填孔 → 3x3 多数投票。
    // 关键点（HSV 经典坑，答辩可讲）：
    //   * OpenCV 的 H 范围是 0~179（不是 0~360）。红色横跨色相环两端：
    //     需要 [0,40] 与 [160,180) 两段 inRange 再 bitwise_or；蓝只需一段 [80,130]。
    //     （实测合成图同一根红条上 H 同时出现 0 和 179；新曝光下过曝核心呈黄橙/青白，
    //      窗宽按实测覆盖核心，见下方"色相窗重调"说明。）
    //   * S 下界绝不能填 0：灰白像素 S≈0 且 H 记为 0，正好落进红色第一段区间。
    //     白光/数字区/灰背景全靠 S 下界排除（白色 R≈G≈B → S≈0）。
    //   * V 下界排除暗背景。
    //   * 阈值不是背出来的：用 tools/hsv_picker.py 在合成图上采样灯条/背景/数字
    //     的 H/S/V，按"采样 min/max ± 余量"定界；实拍后用同一工具重调。
    // 颜色掩码天然排除了白色的数字区域，因此数字不会被当成灯条。
    cv::Mat buildEnemyMask(const cv::Mat &bgr, TeamColor enemy)
    {
        cv::Mat hsv;
        cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
        cv::Mat kernelopen = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat kernelclose = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(13, 13));

        // ---- HSV 彩色掩码 ----
        // 色相窗按新曝光素材（2500μs）实测重调（2026-10-14 恢复此前验证过的修正）：
        // 该曝光下灯条核心大面积过曝且**有色**——红条横跨 H 0~39（核心 R≈G≈254 呈
        // 黄橙 H 20~39，占条像素 ~29%）、蓝条横跨 H 80~119（核心 B≈G≈255 呈青白
        // H 80~99，占 ~57%）。核心若落在色相窗外：掩码空心/边界毛糙（floodFill 只能
        // 补闭合空洞），蓝条尤其严重——实测这些像素占条内 37%（≈1762px/帧）→ 条细/
        // 断 → 检出率低；扩窗后（历史数据）蓝色检出 78.6%→99.9%、红色 100% 保持、
        // 掩码宽度抖动 P90 3.50→0.21px（详见实验报告相关章节）。
        cv::Mat closed;
        if (enemy == TeamColor::Red)
        {
            cv::Mat RMaskLow, RMaskHigh;
            cv::inRange(hsv, cv::Scalar(0, 100, 80), cv::Scalar(40, 255, 255), RMaskLow);
            cv::inRange(hsv, cv::Scalar(160, 100, 80), cv::Scalar(180, 255, 255), RMaskHigh);
            cv::bitwise_or(RMaskLow, RMaskHigh, closed);
        }
        else
        {
            // 蓝色 V 下限 80→60、并在 RGB 补充里用 B−G>50：旧判据边界落在光晕
            // "平坡"（梯度 ~2 计数/px）上，压缩/噪声让边界游走 ±5px（闪烁）；
            // 移入陡坡后高频闪烁降 3~4 倍、检出提升（历史数据）。
            cv::inRange(hsv, cv::Scalar(80, 100, 60), cv::Scalar(130, 255, 255), closed);
        }
        // 注意（实测结论）：HSV 掩码上不做 3x3 开运算——开运算会抹掉白芯细段，
        // 导致白芯几何修复失效（曾实验引入过，识别变差后回滚）。
        cv::morphologyEx(closed, closed, cv::MORPH_CLOSE, kernelclose);

        // ===== 实拍调参：RGB 通道相减补充暗灯条 =====
        // 视频回放证据：暗灯条 V 值低于 HSV 下界，被滤掉大半，mask 中断成细线。
        // R-G（红）/B-G（蓝）反映"颜色程度"，与绝对亮度无关——暗灯条依然高响应，
        // 而灰白背景/数字响应低。
        // 注意：不要降 HSV 的 V 下界（会引入暗红背景，且灯条与背景连通后
        // floodFill 会填掉整条灯条，实测更差）。
        std::vector<cv::Mat> channels;
        cv::split(bgr, channels); // [0]=B [1]=G [2]=R
        cv::Mat diff;
        if (enemy == TeamColor::Red)
        {
            cv::subtract(channels[2], channels[1], diff); // R - G
        }
        else
        {
            cv::subtract(channels[0], channels[1], diff); // B - G
        }
        cv::Mat rgb_mask;
        // 蓝色判据 40→50（阈值落在边缘陡坡处，轮廓更平滑；红色保持 40）
        cv::threshold(diff, rgb_mask, enemy == TeamColor::Red ? 40 : 50, 255, cv::THRESH_BINARY);
        cv::Mat rgb_opened, rgb_closed;
        cv::morphologyEx(rgb_mask, rgb_opened, cv::MORPH_OPEN, kernelopen);
        cv::morphologyEx(rgb_opened, rgb_closed, cv::MORPH_CLOSE, kernelclose);
        cv::bitwise_or(closed, rgb_closed, closed);

        // ---- 填孔洞（过曝白芯在掩码里是"空心环"）----
        // 放在 RGB 补充之后：暗灯条的环状空洞也一并填充。
        // 注意：floodFill 只能补"被前景完全包围"的洞——核心丢在边界上时留下的
        // 开口空洞补不了，须靠上面的色相窗覆盖（见 buildEnemyMask 顶部说明）。
        {
            cv::Mat holes;
            cv::bitwise_not(closed, holes);
            cv::floodFill(holes, cv::Point(0, 0), cv::Scalar(0));
            cv::bitwise_or(closed, holes, closed);
        }

        // 边界正则化（3x3 多数投票）：抹掉 1px 级毛刺/凹口，让二值图边缘更平直
        // （逐行质心锯齿 1.00→0.50 px/行；对检出为净正贡献，历史实测）。
        cv::Mat smoothed;
        cv::medianBlur(closed, smoothed, 3);
        return smoothed;
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

    // ===== 需求③：灯条约束矩形拟合（fitBarConstrained）=====
    // 动机：minAreaRect 由掩码边界的少数极值点决定——边界 ±1px 噪声、末端拖尾
    // （光晕）或局部凸起都会整体平移/旋转/撑大矩形。本拟合改用"全部边界点"的
    // 统计量，抑制这类极值敏感：
    //   * 主轴：边界点集的 2x2 协方差主特征向量（闭式解），噪声按 ~1/sqrt(N) 平均；
    //   * 尺寸/端点：投影到局部坐标 (u=长向, v=宽向) 后取分位数定边——u 截 2%/98%
    //     去掉端点拖尾，v 截 10%/90% 去掉长边缺口/毛刺；分位用线性插值
    //     （与 numpy.percentile 默认口径一致）。
    // 输入必须是完整边界点集（CHAIN_APPROX_NONE）：SIMPLE 会把长直边压缩成两点，
    // 投影分布/协方差都会失真。
    // 质量门（任一触发即回退 minAreaRect，不改变检出；判据标定见设计报告 §7.3）：
    //   G2 轴偏差 |angle(e1) - angle(minAreaRect 主轴)| > 5° → 形状远离矩形；
    //   G3 面积比 (u98-u02)*(v90-v10) / minAreaRect 面积 ∉ [0.75, 1.15]；
    //   G1 支撑率 (u98-u02)/(u_max-u_min) < 0.85 → 端点严重拖尾。
    // 接受门（§3.4，与 passesBarFilter 的几何部分同语义）：L >= 20 且 half_l/half_w >= 2.5。
    // 尺度：分位数取自真实像素坐标，输出天然是像素尺度，无需归一化/反归一化。

    // 分位数（线性插值）：pos = q*(n-1)，lo/hi 两侧加权。sorted 必须已升序且非空。
    double quantileLinear(const std::vector<double> &sorted, double q)
    {
        const double pos = q * static_cast<double>(sorted.size() - 1);
        const std::size_t lo = static_cast<std::size_t>(std::floor(pos));
        const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
        const double frac = pos - static_cast<double>(lo);
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    }

    // 角度差按 180° 周期折算到 [0, 90°]（长轴方向模 180° 唯一）
    double angleDiff180(double a, double b)
    {
        double d = std::fmod(std::fabs(a - b), 180.0);
        if (d > 90.0)
        {
            d = 180.0 - d;
        }
        return d;
    }

    // 旋转矩形最长边的单位方向（长轴方向）
    cv::Point2f rectLongAxis(const cv::RotatedRect &rect)
    {
        cv::Point2f pts[4];
        rect.points(pts);
        cv::Point2f axis(0.0F, 1.0F);
        float best = -1.0F;
        for (int k = 0; k < 4; ++k)
        {
            const cv::Point2f e = pts[(k + 1) % 4] - pts[k];
            const float len = cv::norm(e);
            if (len > best)
            {
                best = len;
                axis = e * (1.0F / std::max(len, 1e-6F));
            }
        }
        return axis;
    }

    // 约束矩形拟合 v1（PCA 主轴 + 分位定边）。返回 true=采纳拟合几何（rect_out 为拟合结果）；
    // false=回退（rect_out 保持 rough，即原 minAreaRect 结果）。
    bool fitBarConstrainedRect(const std::vector<cv::Point> &contour_full,
                               const cv::RotatedRect &rough, double mask_area,
                               cv::RotatedRect &rect_out)
    {
        rect_out = rough;
        const std::size_t n = contour_full.size();
        if (n < 8) // 分位数需要足够样本
        {
            ++g_bar_fit_rej_pts;
            return false;
        }

        // ---- 1) 主轴：边界点集的 2x2 协方差主特征向量（闭式解）----
        double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
        for (const cv::Point &p : contour_full)
        {
            const double x = p.x, y = p.y;
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
        }
        const double inv_n = 1.0 / static_cast<double>(n);
        const double mx = sx * inv_n, my = sy * inv_n;
        const double cxx = sxx * inv_n - mx * mx; // Σ(x-mu)²/n
        const double cyy = syy * inv_n - my * my; // Σ(y-mu)²/n
        const double cxy = sxy * inv_n - mx * my; // Σ(x-mu)(y-mu)/n
        const double disc = std::sqrt((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy);
        const double lam1 = 0.5 * (cxx + cyy + disc); // 主特征值
        if (lam1 <= 1e-9)
        {
            ++g_bar_fit_rej_pts; // 退化（各向同性）：主轴不可定义
            return false;
        }
        double e1x = 0.0, e1y = 0.0;
        if (std::fabs(cxy) > 1e-12)
        {
            e1x = cxy;
            e1y = lam1 - cxx; // 主特征向量 ∝ (cxy, λ1-cxx)
        }
        else if (cxx >= cyy)
        {
            e1x = 1.0;
        }
        else
        {
            e1y = 1.0;
        }
        const double enorm = std::sqrt(e1x * e1x + e1y * e1y);
        if (enorm < 1e-12)
        {
            ++g_bar_fit_rej_pts;
            return false;
        }
        e1x /= enorm;
        e1y /= enorm;
        const double pvx = -e1y, pvy = e1x; // 宽向（垂直）

        // ---- 2) 全部点投影到局部坐标 ----
        std::vector<double> us(n), vs(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            const double dx = contour_full[i].x - mx;
            const double dy = contour_full[i].y - my;
            us[i] = dx * e1x + dy * e1y;
            vs[i] = dx * pvx + dy * pvy;
        }

        // ---- 3) 分位定边（离群处理：用分位替代 min/max 极值）----
        std::vector<double> u_sorted = us, v_sorted = vs;
        std::sort(u_sorted.begin(), u_sorted.end());
        std::sort(v_sorted.begin(), v_sorted.end());
        const double u_lo = quantileLinear(u_sorted, 0.02);
        const double u_hi = quantileLinear(u_sorted, 0.98);
        const double v_lo = quantileLinear(v_sorted, 0.10);
        const double v_hi = quantileLinear(v_sorted, 0.90);
        const double L = u_hi - u_lo;
        const double W = v_hi - v_lo;
        const double half_l = 0.5 * L;
        const double half_w = 0.5 * W;

        // ---- 4) 接受门（与 passesBarFilter 的几何部分同语义）----
        if (L < 20.0 || half_w <= 1e-6 || half_l / half_w < 2.5)
        {
            ++g_bar_fit_rej_size;
            return false;
        }

        // ---- 5) 质量门 G2/G3/G1（判据标定见设计报告 §7.3）----
        const cv::Point2f rough_axis = rectLongAxis(rough);
        const double fit_ang = std::atan2(e1y, e1x) * 180.0 / CV_PI;
        const double rough_ang = std::atan2(rough_axis.y, rough_axis.x) * 180.0 / CV_PI;
        if (angleDiff180(fit_ang, rough_ang) > 5.0) // G2 轴偏差
        {
            ++g_bar_fit_rej_g2;
            return false;
        }
        // G3 面积一致性（2026-10-14 修订，旧口径把"正确的修剪"拒掉）：
        //   * solidity：拟合矩形面积 / 掩码实心面积（contourArea）∈ [0.80, 1.30]——
        //     修剪凸起/光晕不改变它与 1 的接近程度；"把大片空白框进来"的形状会露馅；
        //   * 对 minAreaRect 只保留上界 1.15：拟合面积**小于** minAreaRect 面积是
        //     设计目的（后者含光晕/凸起），只有"变大"才是异常。
        //   （旧口径双向卡 [0.75,1.15]：合成"侧边光晕/锯齿"案例中把本来正确的修剪
        //    结果拒掉 → 回退到含光晕的 minAreaRect，改进被门吃掉——门控也被数据检验。）
        const double solidity = (L * W) / std::max(mask_area, 1.0);
        if (solidity < 0.80 || solidity > 1.30)
        {
            ++g_bar_fit_rej_g3;
            return false;
        }
        const double rough_area = static_cast<double>(rough.size.width) * rough.size.height;
        if ((L * W) / std::max(rough_area, 1e-9) > 1.15)
        {
            ++g_bar_fit_rej_g3;
            return false;
        }
        const double support = L / std::max(u_sorted.back() - u_sorted.front(), 1e-9);
        if (support < 0.85) // G1 支撑率（端点拖尾）
        {
            ++g_bar_fit_rej_g1;
            return false;
        }

        // ---- 6) 输出：center=mu、size=(L,W)、angle=长轴方向角 ----
        rect_out = cv::RotatedRect(cv::Point2f(static_cast<float>(mx), static_cast<float>(my)),
                                   cv::Size2f(static_cast<float>(L), static_cast<float>(W)),
                                   static_cast<float>(fit_ang));
        return true;
    }

    // ===== 四边拟合 v2（按"检测灯条的四个边"重构，2026-10-14）=====
    // 动机（v1 的遗留问题，均由数据暴露）：v1 的宽度/端点来自**分位数**——分位数对
    // 局部点密度敏感，单侧大凸起（>10% 周长，如侧向光晕）会让分位落进凸起分布，
    // 拟合退化到与 minAreaRect 同值；掩码轮廓在过曝/光晕下近似"椭圆"时，分位宽度
    // 也不稳定。v2 改为"四条边"模型：
    //   * 两条长边：把完整边界点按粗轴分成上下两条边带，各做 Huber 直线拟合——单侧
    //     少数异常点被降权，宽度 = 两条边的**距离**（用整条边的几十~上百个点，而非
    //     10%/90% 两个分位点），从机制上消除"分位掉进凸起"的失效模式；
    //   * 两条端边：端盖含圆角/光晕、独立拟合端边线不可靠（历史 v3 实验已证伪：
    //     蓝方检出 4227→1855），故端边取**⊥中轴**的约束形式，位置由沿轴稳健投影
    //     2%/98% 分位给出；
    //   * 四角 = 两条长边线 × 两条端线的**交点**——长边不平行时自然呈梯形，保留
    //     透视形变（外缘是真实三维矩形的投影，梯形比内接矩形更贴近四角真值）。
    // 质量门（任一触发回退 minAreaRect，与 v1 同风格；E=边模型，G=规格门）：
    //   E1 边带点数不足 / 线拟合退化；E2 两长边夹角 > 8°（形状可疑）；
    //   E3 拟合宽度 / minAreaRect 短边 ∉ [0.55, 1.35]；E4 两端的宽度比 > 1.4
    //     （长边发散过大 → 交点外推不可靠）；G1/G2/G3 同 v1（支撑率/轴偏差/面积比）。
    struct EdgeFit
    {
        bool ok{false};
        std::array<cv::Point2f, 4> corners{}; // 四角（沿轴 -端/+端 × ±宽侧）
        cv::RotatedRect rect;                 // 等价外接旋转矩形（下游兼容）
        cv::Point2f center;
        cv::Point2f axis;
        float half_l{0.0F};
        float half_w{0.0F};
    };

    bool fitBarConstrainedQuad(const std::vector<cv::Point> &contour_full,
                               const cv::RotatedRect &rough, double mask_area, EdgeFit &out)
    {
        const std::size_t n = contour_full.size();
        if (n < 12)
        {
            ++g_bar_fit_rej_pts;
            return false;
        }

        // ---- 1) 主轴（同 v1：协方差闭式主特征向量）----
        double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
        for (const cv::Point &p : contour_full)
        {
            const double x = p.x, y = p.y;
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
        }
        const double inv_n = 1.0 / static_cast<double>(n);
        const double mx = sx * inv_n, my = sy * inv_n;
        const double cxx = sxx * inv_n - mx * mx;
        const double cyy = syy * inv_n - my * my;
        const double cxy = sxy * inv_n - mx * my;
        const double disc = std::sqrt((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy);
        const double lam1 = 0.5 * (cxx + cyy + disc);
        if (lam1 <= 1e-9)
        {
            ++g_bar_fit_rej_pts;
            return false;
        }
        double e1x = 0.0, e1y = 0.0;
        if (std::fabs(cxy) > 1e-12)
        {
            e1x = cxy;
            e1y = lam1 - cxx;
        }
        else if (cxx >= cyy)
        {
            e1x = 1.0;
        }
        else
        {
            e1y = 1.0;
        }
        const double enorm = std::sqrt(e1x * e1x + e1y * e1y);
        if (enorm < 1e-12)
        {
            ++g_bar_fit_rej_pts;
            return false;
        }
        e1x /= enorm;
        e1y /= enorm;
        const double pvx = -e1y, pvy = e1x;

        // ---- 2) 投影 + 逐侧稳健初始宽度（中位数，抗单侧凸起）----
        std::vector<double> us(n), vs(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            const double dx = contour_full[i].x - mx;
            const double dy = contour_full[i].y - my;
            us[i] = dx * e1x + dy * e1y;
            vs[i] = dx * pvx + dy * pvy;
        }
        std::vector<double> v_pos, v_neg;
        v_pos.reserve(n / 2);
        v_neg.reserve(n / 2);
        for (double v : vs)
        {
            (v > 0.0 ? v_pos : v_neg).push_back(v);
        }
        if (v_pos.size() < 6 || v_neg.size() < 6)
        {
            ++g_bar_fit_rej_edges; // E1：单侧点太少
            return false;
        }
        auto median_of = [](std::vector<double> &v)
        {
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            return v[v.size() / 2];
        };
        const double vp0 = median_of(v_pos);  // 正侧长边的稳健起始位置
        const double vn0 = median_of(v_neg);  // 负侧
        if (vp0 - vn0 <= 1.0 || vp0 <= 0.0 || vn0 >= 0.0)
        {
            ++g_bar_fit_rej_edges; // E1：宽度初值异常
            return false;
        }

        // ---- 3) 两条长边：取每侧"外半带"的点做 Huber 直线拟合 ----
        std::vector<cv::Point2f> band_pos, band_neg;
        band_pos.reserve(n / 2);
        band_neg.reserve(n / 2);
        for (std::size_t i = 0; i < n; ++i)
        {
            const cv::Point2f p(contour_full[i]);
            if (vs[i] >= 0.5 * vp0)
            {
                band_pos.push_back(p);
            }
            if (vs[i] <= 0.5 * vn0)
            {
                band_neg.push_back(p);
            }
        }
        if (band_pos.size() < 6 || band_neg.size() < 6)
        {
            ++g_bar_fit_rej_edges; // E1
            return false;
        }
        cv::Vec4f la, lb;
        cv::fitLine(band_pos, la, cv::DIST_HUBER, 0, 0.01, 0.01);
        cv::fitLine(band_neg, lb, cv::DIST_HUBER, 0, 0.01, 0.01);
        cv::Point2f da(la[0], la[1]), db(lb[0], lb[1]);
        const float na = cv::norm(da), nb = cv::norm(db);
        if (na < 1e-6F || nb < 1e-6F)
        {
            ++g_bar_fit_rej_edges; // E1
            return false;
        }
        da *= 1.0F / na;
        db *= 1.0F / nb;
        if (da.dot(db) < 0.0F)
        {
            db = -db;
        }
        if (std::fabs(da.dot(db)) < 0.9903F) // E2：夹角 > 8°
        {
            ++g_bar_fit_rej_par;
            return false;
        }
        const cv::Point2f ca(la[2], la[3]), cb(lb[2], lb[3]); // 直线过点
        cv::Point2f dir = da + db;
        const float dnorm = cv::norm(dir);
        if (dnorm < 1e-3F)
        {
            ++g_bar_fit_rej_par;
            return false;
        }
        dir *= 1.0F / dnorm;
        const cv::Point2f nrm(-dir.y, dir.x);
        // 宽度 = 两条长边在 nrm 方向上的距离
        const float W = std::fabs((cb - ca).dot(nrm));
        const float rough_short = std::min(rough.size.width, rough.size.height);
        if (W <= 1.0F || rough_short <= 1e-3F || W / rough_short < 0.55F || W / rough_short > 1.35F)
        {
            ++g_bar_fit_rej_width; // E3
            return false;
        }
        const double fit_ang = std::atan2(dir.y, dir.x) * 180.0 / CV_PI;

        // ---- 4) 端点：沿轴稳健投影 2%/98% 分位（同 v1）----
        std::vector<double> u_sorted = us;
        std::sort(u_sorted.begin(), u_sorted.end());
        const double u_lo = quantileLinear(u_sorted, 0.02);
        const double u_hi = quantileLinear(u_sorted, 0.98);
        const double L = u_hi - u_lo;
        const double half_l = 0.5 * L;
        const double half_w = 0.5 * W;
        if (L < 20.0 || half_l / half_w < 2.5) // 接受门（同 v1）
        {
            ++g_bar_fit_rej_size;
            return false;
        }

        // ---- 5) 质量门 G2/G3/G1（同 v1）----
        const cv::Point2f rough_axis = rectLongAxis(rough);
        const double rough_ang = std::atan2(rough_axis.y, rough_axis.x) * 180.0 / CV_PI;
        if (angleDiff180(fit_ang, rough_ang) > 5.0) // G2 轴偏差
        {
            ++g_bar_fit_rej_g2;
            return false;
        }
        // G3 面积一致性（2026-10-14 修订，旧口径把"正确的修剪"拒掉）：
        //   * solidity：拟合矩形面积 / 掩码实心面积（contourArea）∈ [0.80, 1.30]——
        //     修剪凸起/光晕不改变它与 1 的接近程度；"把大片空白框进来"的形状会露馅；
        //   * 对 minAreaRect 只保留上界 1.15：拟合面积**小于** minAreaRect 面积是
        //     设计目的（后者含光晕/凸起），只有"变大"才是异常。
        //   （旧口径双向卡 [0.75,1.15]：合成"侧边光晕/锯齿"案例中把本来正确的修剪
        //    结果拒掉 → 回退到含光晕的 minAreaRect，改进被门吃掉——门控也被数据检验。）
        const double solidity = (L * W) / std::max(mask_area, 1.0);
        if (solidity < 0.80 || solidity > 1.30)
        {
            ++g_bar_fit_rej_g3;
            return false;
        }
        const double rough_area = static_cast<double>(rough.size.width) * rough.size.height;
        if ((L * W) / std::max(rough_area, 1e-9) > 1.15)
        {
            ++g_bar_fit_rej_g3;
            return false;
        }
        const double support = L / std::max(u_sorted.back() - u_sorted.front(), 1e-9);
        if (support < 0.85) // G1 支撑率
        {
            ++g_bar_fit_rej_g1;
            return false;
        }

        // ---- 6) 四角 = 两条长边线 × 两条端线（端线过中轴上的分位点、方向 ⊥ 中轴）----
        // 交点用 2x2 线性方程组闭式解；解出后按"沿轴 −/+ 端 × 宽侧 ±"整理顺序。
        auto intersect = [](const cv::Point2f &p1, const cv::Point2f &d1, const cv::Point2f &p2,
                            const cv::Point2f &d2, cv::Point2f &out) -> bool
        {
            const double det = static_cast<double>(d1.x) * d2.y - static_cast<double>(d1.y) * d2.x;
            if (std::fabs(det) < 1e-9)
            {
                return false;
            }
            const cv::Point2f r = p2 - p1;
            const double t = (static_cast<double>(r.x) * d2.y - static_cast<double>(r.y) * d2.x) / det;
            out = p1 + d1 * static_cast<float>(t);
            return true;
        };
        // 中心：沿轴 = 分位区间中点；法向 = 两条长边的中线（相对投影原点 mu 计算）
        const cv::Point2f mu_pt(static_cast<float>(mx), static_cast<float>(my));
        const double u_mid = 0.5 * (u_lo + u_hi);
        const float v_mid = (ca - mu_pt).dot(nrm) + 0.5F * (cb - ca).dot(nrm);
        const cv::Point2f center = mu_pt + dir * static_cast<float>(u_mid) + nrm * v_mid;
        // 两条端线：过中心、方向 ⊥ 中轴（nrm），分别落在 u_lo / u_hi 处
        const cv::Point2f end_lo = center + dir * static_cast<float>(u_lo - u_mid);
        const cv::Point2f end_hi = center + dir * static_cast<float>(u_hi - u_mid);
        cv::Point2f c_pos_lo, c_neg_lo, c_pos_hi, c_neg_hi;
        if (!intersect(ca, da, end_lo, nrm, c_pos_lo) || !intersect(cb, db, end_lo, nrm, c_neg_lo) ||
            !intersect(ca, da, end_hi, nrm, c_pos_hi) || !intersect(cb, db, end_hi, nrm, c_neg_hi))
        {
            ++g_bar_fit_rej_edges;
            return false;
        }
        // E4：两端宽度比（长边发散过大 → 透视外推不可靠）
        const float w_lo = cv::norm(c_pos_lo - c_neg_lo);
        const float w_hi = cv::norm(c_pos_hi - c_neg_hi);
        if (std::max(w_lo, w_hi) / std::max(std::min(w_lo, w_hi), 1e-3F) > 1.4F)
        {
            ++g_bar_fit_rej_width;
            return false;
        }

        out.ok = true;
        out.corners = {c_pos_lo, c_pos_hi, c_neg_hi, c_neg_lo}; // 沿轴顺序成环
        out.center = 0.25F * (c_pos_lo + c_pos_hi + c_neg_hi + c_neg_lo);
        out.axis = dir;
        out.half_l = 0.5F * ((cv::norm(c_pos_hi - c_pos_lo) + cv::norm(c_neg_hi - c_neg_lo)) * 0.5F);
        out.half_w = 0.5F * static_cast<float>(0.5 * (w_lo + w_hi));
        out.rect = cv::RotatedRect(out.center,
                                   cv::Size2f(2.0F * out.half_l, 2.0F * out.half_w), fit_ang);
        return true;
    }

    // ===== 四边拟合 v3（两条长边"偏移"估计 + 垂直端线；默认）=====
    // v2（两条长边各自独立拟合直线、再四条边求交）在本素材上的实测更抖：独立线方向
    // 经半长（~45px）力臂放大到角点，且宽度系统性偏小（红 G3 面积比拒 753 条、
    // 蓝拒 4003 条；红角点 P50 1.12→1.29px、Δ半宽 P95 0.12→1.10px）——四条独立边的
    // 自由度超出本素材信噪比能支撑的量（任务书：数据不支持四条独立边时，应采用约束
    // 更强的模型）。v3 保留"四条边"的**鲁棒性内核**、去掉多余自由度：
    //   * 两条长边：方向由两带 Huber 直线拟合**平均**（与 v1 同口径，实测方向最稳），
    //     各自的位置用"该侧全部点在法向上的**中位数**"——对单侧凸起/光晕鲁棒
    //     （中位数可容忍该侧 <50% 污染点；v1 的 10%/90% 分位只能容忍 <10%，
    //      合成"侧边光晕"案例中 v1 与 minAreaRect 同值、误差 5.1px）；
    //   * 两条端线：⊥ 方向、位于沿轴 2%/98% 分位处（端盖圆角/光晕不可独立拟合，
    //     历史 v3 实验已证伪）；
    //   * 四角 = 四条边求交（长边平行约束下即矩形——约束来自数据，不是假设）。
    bool fitBarConstrainedEdges(const std::vector<cv::Point> &contour_full,
                                const cv::RotatedRect &rough, double mask_area, EdgeFit &out)
    {
        const std::size_t n = contour_full.size();
        if (n < 12)
        {
            ++g_bar_fit_rej_pts;
            return false;
        }

        // ---- 1) 主轴（同 v1/v2）----
        double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
        for (const cv::Point &p : contour_full)
        {
            const double x = p.x, y = p.y;
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
        }
        const double inv_n = 1.0 / static_cast<double>(n);
        const double mx = sx * inv_n, my = sy * inv_n;
        const double cxx = sxx * inv_n - mx * mx;
        const double cyy = syy * inv_n - my * my;
        const double cxy = sxy * inv_n - mx * my;
        const double disc = std::sqrt((cxx - cyy) * (cxx - cyy) + 4.0 * cxy * cxy);
        const double lam1 = 0.5 * (cxx + cyy + disc);
        if (lam1 <= 1e-9)
        {
            ++g_bar_fit_rej_pts;
            return false;
        }
        double e1x = 0.0, e1y = 0.0;
        if (std::fabs(cxy) > 1e-12)
        {
            e1x = cxy;
            e1y = lam1 - cxx;
        }
        else if (cxx >= cyy)
        {
            e1x = 1.0;
        }
        else
        {
            e1y = 1.0;
        }
        const double enorm = std::sqrt(e1x * e1x + e1y * e1y);
        if (enorm < 1e-12)
        {
            ++g_bar_fit_rej_pts;
            return false;
        }
        e1x /= enorm;
        e1y /= enorm;
        const cv::Point2f mu_pt(static_cast<float>(mx), static_cast<float>(my));

        // ---- 2) 两条长边：外半带 Huber 直线定方向（v1 口径）----
        const cv::Point2f e1(static_cast<float>(e1x), static_cast<float>(e1y));
        const cv::Point2f e1n(-e1.y, e1.x);
        std::vector<double> vs_n(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            vs_n[i] = (cv::Point2f(contour_full[i]) - mu_pt).dot(e1n);
        }
        std::vector<double> vpos, vneg;
        for (double v : vs_n)
        {
            (v > 0.0 ? vpos : vneg).push_back(v);
        }
        if (vpos.size() < 6 || vneg.size() < 6)
        {
            ++g_bar_fit_rej_edges; // E1
            return false;
        }
        auto median_v = [](std::vector<double> &v)
        {
            std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
            return v[v.size() / 2];
        };
        const double vp0 = median_v(vpos);
        const double vn0 = median_v(vneg);
        if (vp0 - vn0 <= 1.0 || vp0 <= 0.0 || vn0 >= 0.0)
        {
            ++g_bar_fit_rej_edges; // E1：宽度初值异常
            return false;
        }
        std::vector<cv::Point2f> band_pos, band_neg;
        for (std::size_t i = 0; i < n; ++i)
        {
            const cv::Point2f p(contour_full[i]);
            if (vs_n[i] >= 0.5 * vp0)
            {
                band_pos.push_back(p);
            }
            if (vs_n[i] <= 0.5 * vn0)
            {
                band_neg.push_back(p);
            }
        }
        if (band_pos.size() < 6 || band_neg.size() < 6)
        {
            ++g_bar_fit_rej_edges; // E1
            return false;
        }
        cv::Vec4f la, lb;
        cv::fitLine(band_pos, la, cv::DIST_HUBER, 0, 0.01, 0.01);
        cv::fitLine(band_neg, lb, cv::DIST_HUBER, 0, 0.01, 0.01);
        cv::Point2f da(la[0], la[1]), db(lb[0], lb[1]);
        const float na = cv::norm(da), nb = cv::norm(db);
        if (na < 1e-6F || nb < 1e-6F)
        {
            ++g_bar_fit_rej_edges; // E1
            return false;
        }
        da *= 1.0F / na;
        db *= 1.0F / nb;
        if (da.dot(db) < 0.0F)
        {
            db = -db;
        }
        if (std::fabs(da.dot(db)) < 0.9903F) // E2：两长边夹角 > 8°
        {
            ++g_bar_fit_rej_par;
            return false;
        }
        cv::Point2f dir = da + db;
        const float dnorm = cv::norm(dir);
        if (dnorm < 1e-3F)
        {
            ++g_bar_fit_rej_par;
            return false;
        }
        dir *= 1.0F / dnorm;
        const cv::Point2f nrm(-dir.y, dir.x);
        const double fit_ang = std::atan2(dir.y, dir.x) * 180.0 / CV_PI;

        // ---- 3) 两条长边的"位置"：该侧全部点在 nrm 上的中位数（抗单侧凸起）----
        std::vector<double> off_pos, off_neg;
        off_pos.reserve(n / 2);
        off_neg.reserve(n / 2);
        for (std::size_t i = 0; i < n; ++i)
        {
            const double d = (cv::Point2f(contour_full[i]) - mu_pt).dot(nrm);
            (d > 0.0 ? off_pos : off_neg).push_back(d);
        }
        if (off_pos.size() < 6 || off_neg.size() < 6)
        {
            ++g_bar_fit_rej_edges; // E1
            return false;
        }
        const double v_pos = median_v(off_pos);
        const double v_neg = median_v(off_neg);
        const double W = v_pos - v_neg;
        const float rough_short = std::min(rough.size.width, rough.size.height);
        if (W <= 1.0 || rough_short <= 1e-3F || W / rough_short < 0.55 || W / rough_short > 1.35)
        {
            ++g_bar_fit_rej_width; // E3
            return false;
        }
        const double half_w = 0.5 * W;

        // ---- 4) 两条端线：沿轴 2%/98% 分位（同 v1）----
        std::vector<double> us(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            us[i] = (cv::Point2f(contour_full[i]) - mu_pt).dot(dir);
        }
        std::vector<double> u_sorted = us;
        std::sort(u_sorted.begin(), u_sorted.end());
        const double u_lo = quantileLinear(u_sorted, 0.02);
        const double u_hi = quantileLinear(u_sorted, 0.98);
        const double L = u_hi - u_lo;
        const double half_l = 0.5 * L;
        if (L < 20.0 || half_l / half_w < 2.5) // 接受门（同 v1）
        {
            ++g_bar_fit_rej_size;
            return false;
        }

        // ---- 5) 质量门 G2/G3/G1（同 v1）----
        const cv::Point2f rough_axis = rectLongAxis(rough);
        const double rough_ang = std::atan2(rough_axis.y, rough_axis.x) * 180.0 / CV_PI;
        if (angleDiff180(fit_ang, rough_ang) > 5.0) // G2 轴偏差
        {
            ++g_bar_fit_rej_g2;
            return false;
        }
        // G3 面积一致性（2026-10-14 修订，旧口径把"正确的修剪"拒掉）：
        //   * solidity：拟合矩形面积 / 掩码实心面积（contourArea）∈ [0.80, 1.30]——
        //     修剪凸起/光晕不改变它与 1 的接近程度；"把大片空白框进来"的形状会露馅；
        //   * 对 minAreaRect 只保留上界 1.15：拟合面积**小于** minAreaRect 面积是
        //     设计目的（后者含光晕/凸起），只有"变大"才是异常。
        //   （旧口径双向卡 [0.75,1.15]：合成"侧边光晕/锯齿"案例中把本来正确的修剪
        //    结果拒掉 → 回退到含光晕的 minAreaRect，改进被门吃掉——门控也被数据检验。）
        const double solidity = (L * W) / std::max(mask_area, 1.0);
        if (solidity < 0.80 || solidity > 1.30)
        {
            ++g_bar_fit_rej_g3;
            return false;
        }
        const double rough_area = static_cast<double>(rough.size.width) * rough.size.height;
        if ((L * W) / std::max(rough_area, 1e-9) > 1.15)
        {
            ++g_bar_fit_rej_g3;
            return false;
        }
        const double support = L / std::max(u_sorted.back() - u_sorted.front(), 1e-9);
        if (support < 0.85) // G1 支撑率
        {
            ++g_bar_fit_rej_g1;
            return false;
        }

        // ---- 6) 四条边求交（长边平行约束下即矩形）----
        const double u_mid = 0.5 * (u_lo + u_hi);
        const double v_mid = 0.5 * (v_pos + v_neg);
        const cv::Point2f center = mu_pt + dir * static_cast<float>(u_mid) + nrm * static_cast<float>(v_mid);
        auto intersect = [](const cv::Point2f &p1, const cv::Point2f &d1, const cv::Point2f &p2,
                            const cv::Point2f &d2, cv::Point2f &out) -> bool
        {
            const double det = static_cast<double>(d1.x) * d2.y - static_cast<double>(d1.y) * d2.x;
            if (std::fabs(det) < 1e-9)
            {
                return false;
            }
            const cv::Point2f r = p2 - p1;
            const double t = (static_cast<double>(r.x) * d2.y - static_cast<double>(r.y) * d2.x) / det;
            out = p1 + d1 * static_cast<float>(t);
            return true;
        };
        const cv::Point2f line_pos = mu_pt + nrm * static_cast<float>(v_pos); // 正侧长边上的点
        const cv::Point2f line_neg = mu_pt + nrm * static_cast<float>(v_neg);
        const cv::Point2f end_lo = center + dir * static_cast<float>(u_lo - u_mid);
        const cv::Point2f end_hi = center + dir * static_cast<float>(u_hi - u_mid);
        cv::Point2f c_pos_lo, c_neg_lo, c_pos_hi, c_neg_hi;
        if (!intersect(line_pos, dir, end_lo, nrm, c_pos_lo) ||
            !intersect(line_neg, dir, end_lo, nrm, c_neg_lo) ||
            !intersect(line_pos, dir, end_hi, nrm, c_pos_hi) ||
            !intersect(line_neg, dir, end_hi, nrm, c_neg_hi))
        {
            ++g_bar_fit_rej_edges;
            return false;
        }

        out.ok = true;
        out.corners = {c_pos_lo, c_pos_hi, c_neg_hi, c_neg_lo};
        out.center = center;
        out.axis = dir;
        out.half_l = static_cast<float>(half_l);
        out.half_w = static_cast<float>(half_w);
        out.rect = cv::RotatedRect(center, cv::Size2f(static_cast<float>(L), static_cast<float>(W)),
                                   static_cast<float>(fit_ang));
        return true;
    }

    // 调度：按 g_bar_fit_mode 选择拟合实现；返回 true 时 rect_out 为拟合几何，
    // quad_out.ok=true 时用四角覆盖灯条角点。
    bool fitBarConstrained(const std::vector<cv::Point> &contour_full,
                           const cv::RotatedRect &rough, double mask_area,
                           cv::RotatedRect &rect_out, EdgeFit &quad_out)
    {
        quad_out = EdgeFit{};
        if (g_bar_fit_mode == 1)
        {
            return fitBarConstrainedRect(contour_full, rough, mask_area, rect_out);
        }
        const bool ok = (g_bar_fit_mode == 2)
                            ? fitBarConstrainedQuad(contour_full, rough, mask_area, quad_out)
                            : fitBarConstrainedEdges(contour_full, rough, mask_area, quad_out);
        if (!ok)
        {
            rect_out = rough;
            return false;
        }
        rect_out = quad_out.rect;
        return true;
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

    // ===== 需求③ P3：灯条"宽度"的短时域稳健化（2026-10-14，用户验收反馈驱动）=====
    // 动机（蓝素材取证）：宽度是缓变量，但二值边界的栅格相位逐帧变化 → 单帧宽度按
    // ~1px 阶梯振荡（"移动后相对静止"最差段帧间 |Δ宽|>0.5px 占比 49%、P90 1.24px、
    // 灯条在画面里仍以 ~1.7px/帧 漂移），角点/检测框随之频繁跳动。
    // 处理：只对"宽度"这一个参数做"最近 ≤N 帧的中位数窗"——中位数无偏（不引入
    // 系统偏窄，均值类口径实测在合成真值上偏窄 0.4~0.9px，故不采用），窗口首帧用原始
    // 值；中心/方向/端点**不参与**平滑，避免跟踪滞后。
    // 关联与重置（任务书方案D 的要求）：
    //   * 关联：中心 ≤15px 且长轴方向 ≤20° 视为同一物理灯条（贪心最近，一对一）；
    //   * 重置：新建轨迹 / 与上次出现间隔 >10 帧（≈0.3s@30fps，覆盖短时丢检）/ 中心
    //     跳变 >40px（换目标或快速重捕）→ 窗口清空、本帧用原始宽度；
    //   * 只缩放角点在"垂直中轴"方向上的分量；轴向位置/中心/方向不动。
    // 开关：环境变量 RMV_BAR_WIDTH_WIN=0（或 --width-win 0）关闭；默认 5。
    // 按 g_bar_fit_mode 分表存储，保证离线 A/B（同进程交替两个变体）互不污染。
    constexpr long kWidthGapFrames = 10;
    constexpr float kWidthAssocDist = 15.0F;
    constexpr float kWidthAssocAngle = 20.0F;
    constexpr float kWidthResetDist = 40.0F;

    struct WidthTrack
    {
        cv::Point2f center;
        float angle_deg{0.0F};
        long last_frame{-1};
        std::vector<float> widths; // 最近 ≤N 帧的"原始"宽度
    };
    std::array<std::vector<WidthTrack>, 4> s_width_tracks; // 按拟合模式分表
    std::array<long, 4> s_width_frame{0, 0, 0, 0};

    // 只缩放角点在垂直中轴方向上的分量（中心/轴向/方向不动），并同步 width/rect。
    void applyWidthScale(LightBar &bar, float scale)
    {
        const cv::Point2f axis = rectLongAxis(bar.rect);
        if (cv::norm(axis) < 1e-6F)
        {
            return;
        }
        const cv::Point2f perp(-axis.y, axis.x);
        const cv::Point2f c = bar.rect.center;
        for (cv::Point2f &p : bar.corners)
        {
            const cv::Point2f d = p - c;
            const float axial = d.dot(axis);
            const float lateral = d.dot(perp);
            p = c + axis * axial + perp * (lateral * scale);
        }
        bar.width *= scale;
        bar.rect = cv::minAreaRect(std::vector<cv::Point2f>(bar.corners.begin(), bar.corners.end()));
        ++g_bar_width_smooth_count;
    }

    // 一帧的宽度平滑：原地修改 bars（宽度与角点），推进该模式的轨迹表。
    void applyWidthSmoothing(std::vector<LightBar> &bars, std::vector<WidthTrack> &tracks, long &frame)
    {
        ++frame;
        if (g_bar_width_window <= 1)
        {
            return; // 关闭（0 或 1 帧窗口 = 原始宽度）
        }
        // 1) 清理过期轨迹（间隔超过 kWidthGapFrames 视为目标消失）
        for (auto it = tracks.begin(); it != tracks.end();)
        {
            it = (frame - it->last_frame > kWidthGapFrames) ? tracks.erase(it) : it + 1;
        }
        // 2) 贪心最近中心一对一关联（同帧重复中心/方向门限）
        std::vector<int> match(bars.size(), -1);
        std::vector<bool> used(tracks.size(), false);
        for (std::size_t i = 0; i < bars.size(); ++i)
        {
            int best_j = -1;
            float best_d = kWidthAssocDist;
            for (std::size_t j = 0; j < tracks.size(); ++j)
            {
                if (used[j])
                {
                    continue;
                }
                const float d = cv::norm(bars[i].rect.center - tracks[j].center);
                if (d >= best_d ||
                    angleDiff180(bars[i].tilt_deg, tracks[j].angle_deg) > kWidthAssocAngle)
                {
                    continue;
                }
                best_d = d;
                best_j = static_cast<int>(j);
            }
            if (best_j >= 0)
            {
                used[static_cast<std::size_t>(best_j)] = true;
                match[i] = best_j;
            }
        }
        // 3) 更新轨迹并决定本帧使用的宽度
        for (std::size_t i = 0; i < bars.size(); ++i)
        {
            const float raw_w = bars[i].width;
            if (raw_w <= 1e-3F)
            {
                continue;
            }
            const int j = match[i];
            if (j < 0)
            {
                // 新轨迹：本帧用原始宽度（无偏），建立历史
                tracks.push_back(WidthTrack{bars[i].rect.center, bars[i].tilt_deg, frame, {raw_w}});
                ++g_bar_width_reset_count;
                continue;
            }
            WidthTrack &tr = tracks[static_cast<std::size_t>(j)];
            if (cv::norm(bars[i].rect.center - tr.center) > kWidthResetDist)
            {
                tr.widths.assign(1, raw_w); // 中心跳变：重置窗口
                ++g_bar_width_reset_count;
            }
            else
            {
                tr.widths.push_back(raw_w);
                if (static_cast<int>(tr.widths.size()) > g_bar_width_window)
                {
                    tr.widths.erase(tr.widths.begin());
                }
            }
            tr.center = bars[i].rect.center;
            tr.angle_deg = bars[i].tilt_deg;
            tr.last_frame = frame;

            std::vector<float> med = tr.widths;
            std::sort(med.begin(), med.end());
            const float w_med = med[med.size() / 2];
            const float scale = w_med / raw_w;
            if (std::fabs(scale - 1.0F) > 1e-3F && scale > 0.6F && scale < 1.6F)
            {
                applyWidthScale(bars[i], scale);
            }
        }
    }

    // 步骤 2：从掩码提取灯条。
    // 轮廓 -> minAreaRect -> 过滤。实拍调参后的策略：
    //   * 彩块本身达标（细长/实心/|tilt|<=40°）→ 直接采用彩块几何；
    //   * 不达标（被光晕撑宽的胖块/被拖尾拉斜的块）→ 用白芯做几何修复（见上方说明），
    //     修复结果仍不达标则丢弃。白芯缺失（如合成图纯色条）时自然走彩块路径。
    // 几何估计（需求③）：彩块轮廓先过 fitBarConstrained（约束矩形拟合），通过全部门控
    // 时采纳拟合几何，否则回退 minAreaRect（与旧行为逐位一致）。白芯修复路径不变。
    std::vector<DebugBar> s_debug_bars; // 供 debugBars()（见 header）

    std::vector<LightBar> extractLightBars(const cv::Mat &bgr, const cv::Mat &mask,
                                          bool update_state = true)
    {
        s_debug_bars.clear();
        std::vector<LightBar> bars;
        if (mask.empty() || mask.type() != CV_8UC1)
        {
            return bars;
        }

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        // 需求③：约束拟合需要完整边界点集（规格要求 CHAIN_APPROX_NONE；SIMPLE 会把
        // 长直边压缩成两点，分位数/协方差失真）。两次 findContours 对同一掩码的
        // 轮廓顺序一致（按行扫描、RETR_EXTERNAL），按下标配对；关闭拟合时不跑第二遍。
        std::vector<std::vector<cv::Point>> contours_full;
        if (g_bar_fit_mode > 0)
        {
            cv::findContours(mask, contours_full, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
            if (contours_full.size() != contours.size())
            {
                contours_full.clear(); // 顺序/数量异常：本次全部回退（防御性）
            }
        }

        // 白芯轮廓按档位懒加载（只有出现不达标彩块时才生成对应档位的掩码）
        std::array<std::vector<std::vector<cv::Point>>, kCoreLevels> core_contours;
        std::array<bool, kCoreLevels> core_ready{false, false, false};

        // 收集灯条（几何平滑在函数末尾统一做，debugBars 记录平滑后的最终几何）
        auto push_bar = [&bars](const LightBar &b, bool fitted)
        {
            LightBar out = b;
            out.fitted = fitted;
            bars.push_back(out);
        };

        for (std::size_t ci = 0; ci < contours.size(); ++ci)
        {
            const std::vector<cv::Point> &contour = contours[ci];
            if (cv::contourArea(contour) < 20.0)
            {
                continue;
            }
            const double contour_area = cv::contourArea(contour); // 只算一次
            const cv::RotatedRect rough = cv::minAreaRect(contour); // 原几何（回退值）
            cv::RotatedRect rect = rough;
            EdgeFit quad;
            bool fitted = false;
            if (ci < contours_full.size())
            {
                const auto fit_t0 = std::chrono::steady_clock::now();
                ++g_bar_fit_total;
                fitted = fitBarConstrained(contours_full[ci], rough, contour_area, rect, quad);
                if (fitted)
                {
                    ++g_bar_fit_accept;
                }
                g_bar_fit_ms_total += std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - fit_t0)
                                          .count();
            }
            LightBar bar;
            fillBarGeometry(bar, rect);
            if (quad.ok)
            {
                // v2/v3：四角用"四条边求交"的结果；rect 仍为等价外接（中心/倾角/配对用），
                // 二者共用同一套下游。
                bar.corners = quad.corners;
            }
            const float fill = static_cast<float>(contour_area) / (bar.width * bar.height);

            // 达标不修：避免短噪白芯把本就良好的几何带偏（实测教训）
            if (passesBarFilter(bar, fill))
            {
                push_bar(bar, fitted);
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
                            push_bar(refined, false); // 白芯修复几何（不参与拟合）
                        }
                    }
                    break; // 只用最严档位里找到的有效白芯（与调参实验一致）
                }
            }
        }

        if (update_state)
        {
            // 需求③ P3：宽度短时域稳健化（只动宽度参数；分模式存储）
            const std::size_t mode = static_cast<std::size_t>(std::min(std::max(g_bar_fit_mode, 0), 3));
            applyWidthSmoothing(bars, s_width_tracks[mode], s_width_frame[mode]);
        }
        // 记录灯条最终几何（供 debugBars()）：平滑后的宽度/角点即为下游实际使用值
        for (const LightBar &b : bars)
        {
            DebugBar db;
            db.center = b.rect.center;
            db.axis = rectLongAxis(b.rect);
            db.half_l = b.height * 0.5F;
            db.half_w = b.width * 0.5F;
            db.fitted = b.fitted;
            db.corners = b.corners;
            s_debug_bars.push_back(db);
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
    // 候选配对（ratio 供"一灯一配"贪心筛选排序用）
    struct CandidatePair
    {
        std::size_t left{0};
        std::size_t right{0};
        double ratio{0.0};
    };

    std::vector<CandidatePair> matchLightBars(
        const std::vector<LightBar> &bars)
    {
        // tips
        std::vector<CandidatePair> candidates;
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
                // 条件⑥（视频回放修复）：上下连线应近似垂直于灯条轴线（真实装甲板
                // 接近矩形）。条件③只保证"上下连线彼此平行"，无法排除"斜切平行四边形"
                // ——frame 118 的两块斜向碎片（机器人边缘反光）拼出的假装甲就是：
                // 两杆平行、连线平行，但连线偏离垂直达 35°。
                // 抗滚转写法：连线相对水平的角度 |angt| 应与灯条离竖直的倾角 |tilt| 接近。
                const double bar_tilt_avg = std::fabs((L.tilt_deg + R.tilt_deg) / 2.0);
                if (std::fabs(std::fabs(angt) - bar_tilt_avg) > 25.0)
                {
                    continue;
                }
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
                // 保留粗过滤区间 [1.5, 5.5]。
                // 实拍证据（f00104 等假框帧）：真对 ratio=2.84、假对 ratio=3.32——在
                // 该测试机器人上二者重叠，"按标准尺寸收紧区间"会连真对一起杀掉
                // （实测收紧后视频检出率 78.8%→28.9%）。假对的排除依靠下方"一灯一配"。
                if (dist_ratio < 1.5 || dist_ratio > 5.5)
                {
                    continue;
                }

                // ===== 通过几何检查：全部收集为候选 =====
                // "一灯一配"的冲突消解放在 armor_detect 的分类之后做（见那里注释）：
                // 伪候选（反光/灯条边缘凑出的对）大多会死在分类器，不该在几何阶段抢灯条。
                candidates.push_back(CandidatePair{li, ri, dist_ratio});
            }
        }

        return candidates;
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

        // 8. 裁界（& 求交集，防止外扩越界；crop 的坐标系即 gray 的坐标系）。
        const cv::Rect crop = cv::Rect(x0, y0, x1 - x0, y1 - y0) &
                              cv::Rect(0, 0, center.cols, center.rows);

        // ===== 实拍调参：极小裁剪拒绝 =====
        // 视频回放证据（f00456/f00477/f00742/f00743）：12~35px 的碎片裁剪被放大到
        // 64x64 后会"伪装成数字"——分类器在 22x12 的图上给出 "4"0.996、"8"0.955、
        // "2"0.994 的荒谬高置信度。真数字区域在实测有效距离内最小约 42x80（面积
        // 3360），假碎片则 ≤33x35（面积 1155）；按 面积>=1200 且 边长>=16 拒绝。
        if (crop.width < 16 || crop.height < 16 || crop.width * crop.height < 1200)
        {
            return {};
        }

        // ===== 实拍调参：数字区域形状检查 =====
        // 视频回放证据：假装甲板的中间可能是灯条边缘/背景纹理，裁出来后"确实像
        // 数字"，分类器会以高置信度（0.93~1.00）输出错误 ID。真数字的形状特征：
        // 白像素占比适中、外接框接近方形；灯条边缘是细长斜条——按形状拒掉。
        const double white_ratio =
            static_cast<double>(cv::countNonZero(binary(crop))) /
            static_cast<double>(crop.width * crop.height);
        if (white_ratio < 0.05 || white_ratio > 0.5)
        {
            return {};
        }
        {
            cv::Mat white_mask;
            cv::compare(binary(crop), 255, white_mask, cv::CMP_EQ);
            std::vector<cv::Point> white_pts;
            cv::findNonZero(white_mask, white_pts);
            if (!white_pts.empty())
            {
                const cv::Rect bbox = cv::boundingRect(white_pts);
                const double aspect =
                    static_cast<double>(bbox.width) / std::max(bbox.height, 1);
                if (aspect < 0.25 || aspect > 3.0)
                {
                    return {};
                }
            }
        }

        // 9. 返回 min 通道裁剪（单通道），分类器对 1/3 通道输入都兼容——
        // 必须返回 min 通道本身，否则分类器内部又转回被光晕污染的灰度。
        return gray(crop);
    }

} // namespace

// 调试辅助：暴露内部掩码（逻辑与 armor_detect 完全一致，只供测试工具显示用）
cv::Mat enemyMaskForDebug(const cv::Mat &bgr, TeamColor enemy_color)
{
    return buildEnemyMask(bgr, enemy_color);
}

// 调试钩子：最近一次 armor_detect 提取出的灯条最终几何（说明见 header）。
std::vector<DebugBar> debugBars()
{
    return s_debug_bars;
}

// 调试辅助：正则化掩码——原始掩码压暗（保留对照），灯条拟合四边形画满（亮）。
// 与 enemyMaskForDebug 一样只供测试工具显示；内部会重跑一次灯条提取（会覆盖
// debugBars() 的暂存，工具请在 armor_detect 之后、显示之前调用）。
cv::Mat regularizedMaskForDebug(const cv::Mat &bgr, TeamColor enemy_color)
{
    const cv::Mat mask = buildEnemyMask(bgr, enemy_color);
    const std::vector<LightBar> bars = extractLightBars(bgr, mask, /*update_state=*/false);
    cv::Mat canvas(mask.size(), CV_8UC1, cv::Scalar(0));
    canvas.setTo(70, mask > 0); // 原始掩码：暗灰底，能看清"毛糙"来源
    for (const LightBar &b : bars)
    {
        std::vector<cv::Point> poly;
        poly.reserve(4);
        for (const cv::Point2f &p : b.corners)
        {
            poly.emplace_back(cvRound(p.x), cvRound(p.y));
        }
        cv::fillPoly(canvas, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(255));
    }
    return canvas;
}

// 显示用位姿解算（含平面目标孪生解消歧），详见头文件注释。
// 实测数据（视频 frame 20/565 等 7 帧）：孪生两解重投影差 0.15~2.3px、位置差 ≤8.5mm、
// 装甲板 yaw 差 5~50°（典型 10~30°）——逐帧选"误差小者"会让坐标轴与 yaw 来回翻。
bool solveArmorPoseDisambiguated(const std::vector<cv::Point3f> &object_points,
                                 const std::vector<cv::Point2f> &image_points,
                                 const cv::Mat &camera_matrix,
                                 const cv::Mat &distortion_coefficients,
                                 cv::Mat &rvec, cv::Mat &tvec)
{
    static cv::Vec3d prev_n(0.0, 0.0, 0.0); // 上一帧所选解的法向
    static bool has_prev = false;
    std::vector<cv::Mat> rvecs, tvecs;
    const int n = cv::solvePnPGeneric(object_points, image_points, camera_matrix,
                                      distortion_coefficients, rvecs, tvecs, false,
                                      cv::SOLVEPNP_IPPE);
    if (n <= 0)
    {
        return false;
    }
    int pick = 0;
    if (n >= 2)
    {
        double err[2] = {0.0, 0.0};                  // 各解的重投影平均误差
        cv::Vec3d n_vec[2] = {cv::Vec3d(0, 0, 1),     // 各解法向 n = R·(0,0,1)
                              cv::Vec3d(0, 0, 1)};
        for (int i = 0; i < 2; ++i)
        {
            std::vector<cv::Point2f> proj;
            cv::projectPoints(object_points, rvecs[i], tvecs[i], camera_matrix,
                              distortion_coefficients, proj);
            double sum = 0.0;
            for (std::size_t k = 0; k < proj.size(); ++k)
            {
                sum += cv::norm(proj[k] - image_points[k]);
            }
            err[i] = sum / static_cast<double>(proj.size());
            cv::Mat R;
            cv::Rodrigues(rvecs[i], R);
            n_vec[i] = cv::Vec3d(R.at<double>(0, 2), R.at<double>(1, 2), R.at<double>(2, 2));
        }
        if (std::fabs(err[0] - err[1]) > 3.0)
        {
            pick = (err[0] <= err[1]) ? 0 : 1; // 误差差距明显：以误差为准
        }
        else if (has_prev)
        {
            // 时域一致：取法向与上一帧最接近的解（连续帧间真实姿态变化远小于孪生差）
            pick = (n_vec[0].dot(prev_n) >= n_vec[1].dot(prev_n)) ? 0 : 1;
        }
        else
        {
            pick = (std::fabs(n_vec[0][1]) <= std::fabs(n_vec[1][1])) ? 0 : 1; // 首帧：更接近竖直
        }
        prev_n = n_vec[pick];
        has_prev = true;
    }
    rvec = rvecs[pick].clone();
    tvec = tvecs[pick].clone();
    return true;
}

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
    const std::vector<CandidatePair> candidates = matchLightBars(bars);

    // 逐候选裁剪 + 分类；只有通过分类的候选进入随后的冲突消解。
    struct ScoredDetection
    {
        ArmorDetection detection;
        double score{0.0}; // 与标准尺寸的偏差（越小越可信）
        std::size_t left{0};
        std::size_t right{0};
    };
    std::vector<ScoredDetection> validated;

    for (const CandidatePair &candidate : candidates)
    {
        const LightBar &left = bars[candidate.left];
        const LightBar &right = bars[candidate.right];
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
        if (number.empty())
        {
            continue; // 裁剪阶段明确拒绝的候选不应继续送入分类器
        }
        const ArmorClassifier::Result classified = classifier.infer(number);
        if (classified.target_id == 0 || classified.confidence < kMinConfidence)
        {
            continue;
        }

        ScoredDetection scored;
        scored.detection.corners = corners;
        scored.detection.size = classifyArmorSize(left, right);
        scored.detection.target_id = classified.target_id;
        scored.detection.confidence = classified.confidence;
        scored.score = std::min(std::fabs(candidate.ratio - 2.37),
                                std::fabs(candidate.ratio - 4.04));
        scored.left = candidate.left;
        scored.right = candidate.right;
        validated.push_back(scored);
    }

    // ===== 一灯一配（实拍调参，视频回放证据）=====
    // 同一根灯条被两个装甲板共享会产生"假装甲板"（实测 f00104：真板 7 与假板 2
    // 共享中间灯条、两者都能过分类器）。冲突消解放在分类之后：
    //   * 伪候选（反光/灯条边缘凑出的对）大多在分类环节就被拒，不再来抢灯条；
    //   * 真冲突按【分类置信度】降序择优（实测 f00456/f00477/f00742/f00743：
    //     假碎片裁剪只有 12~35px、被放大后伪装成数字（"2"0.99/"4"0.996/"8"0.955），
    //     但它们与真"7"共享灯条时，真"7"的置信度总是更高（≈1.0）。
    //     ——切勿按 ratio 偏差排序：假配对的外缘/高之比常更接近标称值，
    //     曾导致"真框被丢、假框留下"。ratio 仅作置信度打平时的次序键。
    std::sort(validated.begin(), validated.end(),
              [](const ScoredDetection &a, const ScoredDetection &b)
              {
                  if (std::fabs(a.detection.confidence - b.detection.confidence) > 1e-4F)
                  {
                      return a.detection.confidence > b.detection.confidence;
                  }
                  return a.score < b.score;
              });
    std::vector<bool> used(bars.size(), false);
    for (const ScoredDetection &scored : validated)
    {
        if (used[scored.left] || used[scored.right])
        {
            continue; // 灯条已被更优的候选占用
        }
        used[scored.left] = true;
        used[scored.right] = true;
        detections.push_back(scored.detection);
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

    // 按目标编号和尺寸分别保存平面 PnP 的历史法向，避免不同观测互相污染。
    // 这只用于消解 IPPE 的双解跳变；连续帧的测量噪声仍应交给后续滤波器处理。
    struct TrackedPoseHistory
    {
        cv::Vec3d normal{0.0, 0.0, 1.0};
        cv::Point2f image_center{};
        bool has_normal{false};
        bool has_center{false};
        std::chrono::steady_clock::time_point last_seen{};
        bool has_last_seen{false};
    };

    bool solvePoseForTrack(const std::vector<cv::Point3f> &object_points,
                           const std::vector<cv::Point2f> &image_points,
                           const cv::Mat &camera_matrix,
                           const cv::Mat &distortion_coefficients,
                           int target_id, ArmorSize size,
                           cv::Mat &rvec, cv::Mat &tvec)
    {
        using TrackKey = std::pair<int, int>;
        static std::map<TrackKey, TrackedPoseHistory> history_by_track;
        TrackedPoseHistory &history = history_by_track[
            TrackKey{target_id, static_cast<int>(size)}];
        const auto now = std::chrono::steady_clock::now();

        cv::Point2f center(0.0F, 0.0F);
        for (const cv::Point2f &point : image_points)
            center += point * (1.0F / static_cast<float>(image_points.size()));
        // 同 ID/尺寸的观测若突然跳到画面另一处，避免沿用旧目标的朝向历史。
        if ((history.has_last_seen &&
             now - history.last_seen > std::chrono::milliseconds(250)) ||
            (history.has_center && cv::norm(center - history.image_center) > 120.0))
            history.has_normal = false;

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
                continue;
            std::vector<cv::Point2f> projected;
            cv::projectPoints(object_points, rvecs[i], tvecs[i], camera_matrix,
                              distortion_coefficients, projected);
            double sum = 0.0;
            for (std::size_t k = 0; k < projected.size(); ++k)
                sum += cv::norm(projected[k] - image_points[k]);
            errors[static_cast<std::size_t>(i)] =
                sum / std::max<std::size_t>(projected.size(), 1);

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
        history.normal = normals[static_cast<std::size_t>(pick)];
        history.has_normal = true;
        history.image_center = center;
        history.has_center = true;
        history.last_seen = now;
        history.has_last_seen = true;
        return true;
    }

    // 门禁阈值（现场可调；任务书要求拒绝深度非法或误差异常的观测）
    constexpr double kMaxReprojErrorPx = 6.0; // 重投影误差上限,阈值按数据分布定(6最合适)
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

        // PnP：4 个共面点。IPPE 可能返回两个相近重投影误差的姿态解；
        // 按目标编号/尺寸维护历史法向，避免逐帧在两个解之间跳变。
        cv::Mat rvec, tvec;
        const bool ok = solvePoseForTrack(object_points, image_points,
                                          camera.camera_matrix,
                                          camera.distortion_coefficients,
                                          det.target_id, det.size, rvec, tvec);
        if (!ok)
        {
            ++g_pnp_fail; // test
            continue;     // PnP 失败 → 空观测，交给 EKF 纯预测
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
        // 统计中排除 NaN/Inf，否则后续 std::sort 的比较关系不再满足严格弱序。
        if (std::isfinite(reproj_error))
        {
            g_reproj_errors.push_back(reproj_error);
        }

        // tips 门禁：以下任一不满足就 continue（拒绝该观测）——
        //   1) reproj_error > kMaxReprojErrorPx
        //   2) 深度 z = tvec.at<double>(2) 超出 [kMinDepthM, kMaxDepthM]
        //   3) 数值非有限（std::isfinite 检查 x/y/z 与 reproj_error）
        const bool all_finite = std::isfinite(reproj_error) &&
                                std::isfinite(tvec.at<double>(0)) &&
                                std::isfinite(tvec.at<double>(1)) &&
                                std::isfinite(tvec.at<double>(2));
        if (!all_finite)
        {
            ++g_rej_finite; // ← test
            continue;
        }
        if (tvec.at<double>(2) < kMinDepthM || tvec.at<double>(2) > kMaxDepthM)
        {
            ++g_rej_depth; // ← test
            continue;
        }
        if (reproj_error > kMaxReprojErrorPx)
        {
            ++g_rej_reproj; // ← test
            continue;
        }

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
