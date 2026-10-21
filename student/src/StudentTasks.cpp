#include "StudentTasks.hpp"
#include "ArmorDetectorNet.hpp"
#include "RotatingArmorTracker.hpp"
#include "MvCameraControl.h"

#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
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
// 相机固件自算帧率（MV_CC_GetFloatValue "ResultingFrameRate"），-1=未取到。
// 与主机处理速度无关，是纯硬件层实际出帧率；每 30 帧查询一次（GenICam 读节点）。
static float s_camera_fps = -1.0F;

// ===== test计数器（armor_solve 拒绝原因统计，供 test_camera_live 打印）=====
int g_rej_reproj = 0;
int g_rej_depth = 0;
int g_rej_finite = 0;
int g_pnp_fail = 0;

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
    warn(MV_CC_SetFloatValue(*handle, "ExposureTime", 1000.0f), "ExposureTime");
    warn(MV_CC_SetFloatValue(*handle, "Gain", 16.0f), "Gain");
    warn(MV_CC_SetFloatValue(*handle, "AcquisitionFrameRate", 160.0f), "AcquisitionFrameRate");

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
        // 神经网络路线角点即装甲板四角（与 PnP 物点同源），直接从角点画框，
        // 不再做"内收到灯条中线"的显示修正（那是传统路线灯条外缘口径的补偿）。
        std::vector<cv::Point> pts;
        pts.reserve(4);
        for (int k = 0; k < 4; ++k)
        {
            pts.emplace_back(cvRound(det.corners[k].x), cvRound(det.corners[k].y));
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

    // ===== 相机固件自算帧率（每 30 帧轮询一次，GenICam 节点读取）=====
    // "ResultingFrameRate" 由相机固件根据实际出帧计算，不受主机处理/显示速度
    // 影响，是纯硬件层帧率；调试工具以此为准判断"相机到底出了多少帧"。
    static int fps_poll_count = 0;
    if (++fps_poll_count >= 30)
    {
        fps_poll_count = 0;
        MVCC_FLOATVALUE fps_value{};
        if (MV_CC_GetFloatValue(s_handle, "ResultingFrameRate", &fps_value) == MV_OK)
        {
            s_camera_fps = fps_value.fCurValue;
        }
    }

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

// 相机固件自算的实际出帧率（"ResultingFrameRate"），未取到时返回 -1。
// 由相机固件计算、与主机处理/显示速度无关；get_pic 内每 30 帧轮询刷新。
float cameraResultingFps()
{
    return s_camera_fps;
}

// ==================== 任务 2：装甲板识别 ====================
// 神经网络四点检测路线（唯一实现）：OpenVINO 推理 + 关键点解码，
// 输出角点/大小/编号/置信度。接口契约见 StudentTasks.hpp。
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
        double err[2] = {0.0, 0.0};               // 各解的重投影平均误差
        cv::Vec3d n_vec[2] = {cv::Vec3d(0, 0, 1), // 各解法向 n = R·(0,0,1)
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

// 检测置信度硬门槛：默认 0.7，可由 setArmorScoreThreshold() 覆盖。
static float g_nn_min_confidence = 0.7F;

namespace
{

    // ===== 神经网络四点检测路线（默认）=====
    // 模型：models/yolov5.xml（sp_vision_25 上场默认）；RMV_DET_MODEL=yolo11 切备选模型。
    // 解码与语义映射见 ArmorDetectorNet.cpp 与 models/README.md。
    // 最终输出置信度门限：0.7（512 输入时远处小目标定位噪声大，用户实测 0.6/0.8
    // 都不影响"抖动只来自定位精度"这一结论；门限只过滤漏检不过滤抖动）。
    // 与解码阈值分开的硬过滤：历史上写死 0.7，导致 --score 降到 0.3 也一条都不过。

    ArmorDetectorNet &nnDetector(bool &ok)
    {
        static ArmorDetectorNet detector;
        static bool load_attempted = false;
        static bool load_ok = false;
        if (!load_attempted)
        {
            load_attempted = true;
            std::string error;
            const char *model_env = std::getenv("RMV_DET_MODEL");
            const bool use_yolo11 = model_env != nullptr && std::string(model_env) == "yolo11";
            // 推理输入尺寸：默认 640x480。相机是 4:3，640x640 输入会把 1440x1080
            // 等比缩到 640x480 后剩下的 160 行全是黑边——图像内容与缩放比例和
            // 640x640 完全相同，但那 25% 的卷积是白算的。改成 640x480 输入后
            // 计算量 -25%，角点分辨率一点没变（不像 RMV_DET_SIZE=512 那样整体降采样）。
            // RMV_DET_SIZE=512 或 640x480 可覆盖；缺对应模型文件时回退 640x640。
            int width = 640;
            int height = 480;
            if (const char *size_env = std::getenv("RMV_DET_SIZE"))
            {
                int w = 0;
                int h = 0;
                if (std::sscanf(size_env, "%dx%d", &w, &h) == 2 && h > 0)
                {
                    width = w;
                    height = h;
                }
                else if (std::sscanf(size_env, "%d", &w) == 1 && w > 0)
                {
                    width = w;
                    height = w;
                }
            }
            const auto size_valid = [](int value)
            { return value >= 320 && value % 32 == 0; };
            if (!size_valid(width) || !size_valid(height))
            {
                width = 640;
                height = 480;
            }
            // 推理设备：RMV_DET_DEVICE=GPU/AUTO 可切（有 iGPU 渲染节点的机器可用；
            // 编译失败自动回退 CPU）。
            std::string device = "CPU";
            if (const char *dev_env = std::getenv("RMV_DET_DEVICE"))
            {
                device = dev_env;
            }
            const std::string base = use_yolo11 ? "yolo11" : "yolov5";
            const ArmorDetectorNet::ModelKind kind =
                use_yolo11 ? ArmorDetectorNet::ModelKind::YOLO11
                           : ArmorDetectorNet::ModelKind::YOLOv5;
            const std::string default_path = "models/" + base + ".xml";
            std::string path = default_path;
            if (!(width == 640 && height == 640))
            {
                path = "models/" + base + "_" + std::to_string(width) + "x" +
                       std::to_string(height) + ".xml";
            }
            if (width == 640 && height == 640)
            {
                load_ok = detector.load(path, kind, width, height, device, error);
            }
            else
            {
                std::string sized_error;
                if (detector.load(path, kind, width, height, device, sized_error))
                {
                    load_ok = true;
                }
                else
                {
                    std::cerr << "未找到 " << path << "（" << sized_error.substr(0, 80)
                              << "），回退 640x640 输入\n";
                    detector = ArmorDetectorNet{};
                    width = 640;
                    height = 640;
                    path = default_path;
                    load_ok = detector.load(path, kind, width, height, device, error);
                }
            }
            if (load_ok)
            {
                std::cout << "神经网络检测器就绪: " << base << " " << detector.input_width()
                          << "x" << detector.input_height() << " device=" << detector.device()
                          << "\n";
            }
            else
            {
                std::cerr << "神经网络检测器加载失败: " << error << "\n";
            }
        }
        ok = load_ok;
        return detector;
    }

    std::vector<ArmorDetection> armorDetectNn(const cv::Mat &pic, TeamColor enemy_color)
    {
        std::vector<ArmorDetection> detections;
        if (pic.empty() || pic.type() != CV_8UC3)
        {
            return detections;
        }

        bool ok = false;
        ArmorDetectorNet &detector = nnDetector(ok);
        if (!ok)
        {
            return detections; // 模型加载失败已在加载时打印错误，按无目标处理
        }

        for (const ArmorDetectorNet::Detection &det : detector.detect(pic))
        {
            // 只保留敌方颜色；extinguish/purple 已在解码时丢弃。
            if (det.color != enemy_color)
            {
                continue;
            }
            if (det.confidence < g_nn_min_confidence)
            {
                continue;
            }
            // 拒绝超出图像范围的角点。
            bool inside = true;
            for (const cv::Point2f &p : det.corners)
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

            ArmorDetection out;
            out.corners = det.corners;
            out.size = det.size;
            out.target_id = det.target_id;
            out.confidence = det.confidence;
            detections.push_back(out);
        }
        return detections;
    }

} // namespace

// 任务 2 入口：神经网络四点检测（唯一实现）。
std::vector<LedArmorMeasurement> detect_led_armors(const cv::Mat &bgr, TeamColor color)
{
    std::vector<LedArmorMeasurement> out;
    if (bgr.empty() || bgr.type() != CV_8UC3)
    {
        return out;
    }

    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask;
    if (color == TeamColor::Red)
    {
        // 红色跨 0 度：两端各取一段再合并。
        cv::Mat low;
        cv::Mat high;
        cv::inRange(hsv, cv::Scalar(0, 110, 140), cv::Scalar(14, 255, 255), low);
        cv::inRange(hsv, cv::Scalar(166, 110, 140), cv::Scalar(180, 255, 255), high);
        cv::bitwise_or(low, high, mask);
    }
    else
    {
        cv::inRange(hsv, cv::Scalar(95, 110, 140), cv::Scalar(135, 255, 255), mask);
    }

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);
    struct Bar
    {
        cv::Point2f center;
        double height;
        double width;
    };
    std::vector<Bar> bars;
    for (int i = 1; i < count; ++i)
    {
        const int area = stats.at<int>(i, cv::CC_STAT_AREA);
        const int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        const int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        if (area < 24 || h < 5 || w < 2)
        {
            continue;
        }
        // 灯条是细长的：允许板面有侧倾，所以用长边/短边 >= 2 而不是只认竖直。
        const double long_side = std::max(w, h);
        const double short_side = std::max(1.0, static_cast<double>(std::min(w, h)));
        if (long_side / short_side < 2.0)
        {
            continue;
        }
        Bar bar;
        bar.center = cv::Point2f(static_cast<float>(centroids.at<double>(i, 0)),
                                 static_cast<float>(centroids.at<double>(i, 1)));
        bar.height = h;
        bar.width = w;
        bars.push_back(bar);
    }

    for (std::size_t a = 0; a < bars.size(); ++a)
    {
        for (std::size_t b = a + 1; b < bars.size(); ++b)
        {
            const double h_a = bars[a].height;
            const double h_b = bars[b].height;
            const double h_max = std::max(h_a, h_b);
            if (std::abs(h_a - h_b) > 0.4 * h_max || h_max < 9.0)
            {
                continue;
            }
            const double gap = std::abs(bars[a].center.x - bars[b].center.x);
            // 小板 0.135x0.057、大板 0.23x0.057：间距/高度大致 2.4 与 4.0。
            if (gap < 0.9 * h_max || gap > 6.0 * h_max)
            {
                continue;
            }
            const double dy = std::abs(bars[a].center.y - bars[b].center.y);
            if (dy > 0.5 * h_max)
            {
                continue;
            }
            LedArmorMeasurement m;
            const Bar &left = bars[a].center.x <= bars[b].center.x ? bars[a] : bars[b];
            const Bar &right = bars[a].center.x <= bars[b].center.x ? bars[b] : bars[a];
            m.left_bar = left.center;
            m.right_bar = right.center;
            m.center = cv::Point2f(0.5F * (left.center.x + right.center.x),
                                   0.5F * (left.center.y + right.center.y));
            m.baseline_px = gap;
            m.height_px = h_max;
            out.push_back(m);
        }
    }
    return out;
}

void setArmorScoreThreshold(float threshold)
{
    bool ok = false;
    ArmorDetectorNet &detector = nnDetector(ok);
    g_nn_min_confidence = threshold;
    if (ok)
    {
        detector.setScoreThreshold(threshold);
    }
}

std::vector<ArmorDetection> armor_detect(const cv::Mat &pic, TeamColor enemy_color)
{
    return armorDetectNn(pic, enemy_color);
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
        // 诊断开关：板上实际尺寸未知时，用环境变量扫一遍尺寸对 PnP 深度的影响
        // （v4 的手写"3"板明显比 135mm 大，框宽中位 156px 对应 ~0.25m）
        if (const char *ow = std::getenv("ARMOR_HALF_W"))
        {
            half_w = std::atof(ow);
        }
        if (const char *oh = std::getenv("ARMOR_HALF_H"))
        {
            half_h = std::atof(oh);
        }
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
        // 图像域时间平均：连续帧的角点做 EMA，再解一次 PnP。
        // 166 fps 下每帧只转 ~1.4°，相邻帧几乎是同一个姿态；逐帧解算的
        // 2~3 cm 抖动主要来自 4 点共面相位的病态，而不是物体真的动了。
        std::array<cv::Point2f, 4> smoothed_corners{};
        bool has_smoothed{false};
    };

    bool solvePoseForTrack(const std::vector<cv::Point3f> &object_points,
                           const std::vector<cv::Point2f> &image_points,
                           const cv::Mat &camera_matrix,
                           const cv::Mat &distortion_coefficients,
                           int target_id, ArmorSize size,
                           cv::Mat &rvec, cv::Mat &tvec,
                           double depth_prior = 0.0)
    {
        using TrackKey = std::pair<int, int>;
        static std::map<TrackKey, TrackedPoseHistory> history_by_track;
        TrackedPoseHistory &history = history_by_track[TrackKey{target_id, static_cast<int>(size)}];
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
        // 深度先验：用板的竖直边（灯条端点）长度给出的深度。俯仰不变，
        // 所以它不受绕竖轴旋转的 cos(yaw) 衰减影响，比横向宽度可靠。
        if (depth_prior > 0.0 && count >= 1)
        {
            int best_depth_index = -1;
            double best_dz = 0.0;
            for (int i = 0; i < count; ++i)
            {
                if (!std::isfinite(errors[static_cast<std::size_t>(i)]))
                    continue;
                const double dz = std::fabs(tvecs[static_cast<std::size_t>(i)].at<double>(2) -
                                            depth_prior);
                if (best_depth_index < 0 || dz < best_dz)
                {
                    best_depth_index = i;
                    best_dz = dz;
                }
            }
            if (best_depth_index >= 0)
                pick = best_depth_index;
            if (std::getenv("DEPTH_DEBUG") != nullptr)
                std::fprintf(stderr, "[depth] id=%d prior=%.3f pick=%.3f dz=%.3f\n", target_id,
                             depth_prior, tvecs[static_cast<std::size_t>(pick)].at<double>(2),
                             best_dz);
        }
        else if (count >= 2 && std::isfinite(errors[0]) && std::isfinite(errors[1]) &&
            std::fabs(errors[0] - errors[1]) <= 3.0)
        {
            const char *mode = std::getenv("IPPE_PICK");
            if (mode != nullptr && std::strcmp(mode, "0") == 0)
                pick = 0;
            else if (mode != nullptr && std::strcmp(mode, "1") == 0)
                pick = 1;
            else if (mode != nullptr && std::strcmp(mode, "nohist") == 0)
                pick = std::fabs(normals[0][1]) <= std::fabs(normals[1][1]) ? 0 : 1;
            else if (history.has_normal)
                pick = normals[0].dot(history.normal) >= normals[1].dot(history.normal) ? 0 : 1;
            else
                pick = std::fabs(normals[0][1]) <= std::fabs(normals[1][1]) ? 0 : 1;
            if (!std::isfinite(errors[static_cast<std::size_t>(pick)]))
                pick = best_error_index;
            if (std::getenv("IPPE_DEBUG") != nullptr)
            {
                std::fprintf(stderr,
                             "[ippe] id=%d e0=%.3f e1=%.3f z0=%.3f z1=%.3f dz=%.3f pick=%d\n",
                             target_id, errors[0], errors[1], tvecs[0].at<double>(2),
                             tvecs[1].at<double>(2),
                             std::fabs(tvecs[0].at<double>(2) - tvecs[1].at<double>(2)), pick);
            }
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

    // 相机光学系（x 右 / y 下 / z 前，OpenCV 约定）→ 云台系（x 前 / y 左 / z 上）。
    // 云台外参 R_gimbal_camera 本应是"光学系 → 云台系"的完整旋转，但仓库里的占位值是
    // 单位阵（config/camera.yaml 注释写着"无数据，暂时占位"）。单位阵在物理上不可能成立：
    // 相机光轴必然与云台前向对齐，二者至少差这 90° 的轴重排。占位值直接套用会把 OpenCV
    // 的"前"当成云台的"上"，EKF 的世界模型随即整体错位——2026-10-17 实跑 CSV 里
    // target_pitch≈1.15rad、半径塌到 0.09m、跟踪锁不住，根因就是这里。
    // 所以显式识别占位值并补上这层固定旋转；真正标定后（R 已含光学系→云台系旋转）走原分支。
    const cv::Matx33d kOpticalToGimbal(0.0, 0.0, 1.0,
                                       -1.0, 0.0, 0.0,
                                       0.0, -1.0, 0.0);

    bool extrinsicIsPlaceholder(const CameraParameters &camera)
    {
        if (camera.rotation_camera_to_gimbal.empty() ||
            camera.rotation_camera_to_gimbal.size() != cv::Size(3, 3))
        {
            return true; // 没有外参：同样只能按占位处理
        }
        const double rotation_deviation = cv::norm(
            camera.rotation_camera_to_gimbal - cv::Mat::eye(3, 3, CV_64F), cv::NORM_INF);
        double translation_norm = 0.0;
        if (!camera.translation_camera_to_gimbal.empty())
            translation_norm = cv::norm(camera.translation_camera_to_gimbal, cv::NORM_INF);
        return rotation_deviation < 1e-6 && translation_norm < 1e-9;
    }

    // 相机系 → 云台系的旋转（占位外参时补上光学系→云台系的固定旋转）
    cv::Mat rotationCameraToGimbal(const CameraParameters &camera)
    {
        if (extrinsicIsPlaceholder(camera))
        {
            static bool s_warned = false;
            if (!s_warned)
            {
                s_warned = true;
                std::fprintf(stderr,
                             "[warn] 云台外参仍是占位单位阵：已按"
                             "相机光学系(右/下/前)->云台系(前/左/上)的固定旋转处理。"
                             "完成手眼标定后请用真实 R/t 替换 config/camera.yaml。\n");
            }
            return cv::Mat(kOpticalToGimbal);
        }
        return camera.rotation_camera_to_gimbal;
    }

    cv::Vec3d cameraToGimbalPoint(const CameraParameters &camera, const cv::Vec3d &p_camera)
    {
        const cv::Mat p_gimbal = rotationCameraToGimbal(camera) * cv::Mat(cv::Vec3d(p_camera));
        cv::Vec3d result(p_gimbal.at<double>(0), p_gimbal.at<double>(1), p_gimbal.at<double>(2));
        if (!extrinsicIsPlaceholder(camera) && !camera.translation_camera_to_gimbal.empty())
        {
            result[0] += camera.translation_camera_to_gimbal.at<double>(0);
            result[1] += camera.translation_camera_to_gimbal.at<double>(1);
            result[2] += camera.translation_camera_to_gimbal.at<double>(2);
        }
        return result;
    }

} // namespace

cv::Vec3d gimbalToCameraPoint(const CameraParameters &camera, const cv::Vec3d &p_gimbal)
{
    // 与 cameraToGimbalPoint 严格互逆：R_c2g 的转置，再减掉平移。
    cv::Vec3d p = cv::Vec3d(p_gimbal);
    if (!extrinsicIsPlaceholder(camera) && !camera.translation_camera_to_gimbal.empty())
    {
        for (int i = 0; i < 3; ++i)
        {
            p[i] -= camera.translation_camera_to_gimbal.at<double>(i);
        }
    }
    const cv::Mat p_camera = rotationCameraToGimbal(camera).t() * cv::Mat(p);
    return cv::Vec3d(p_camera.at<double>(0), p_camera.at<double>(1), p_camera.at<double>(2));
}

std::size_t refineArmorCorners(const cv::Mat &gray, std::vector<ArmorDetection> &detections,
                               double min_edge_px, int win)
{
    if (gray.empty() || gray.type() != CV_8UC1)
    {
        return 0;
    }
    std::size_t refined = 0;
    for (ArmorDetection &det : detections)
    {
        const double edge = cv::norm(det.corners[1] - det.corners[0]);
        if (edge < min_edge_px)
        {
            continue;
        }
        std::vector<cv::Point2f> points(det.corners.begin(), det.corners.end());
        cv::cornerSubPix(gray, points, cv::Size(win, win), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.02));
        // 细化结果离原角点太远说明窗口里没有可用的强角点，保留原值。
        bool sane = true;
        for (int k = 0; k < 4; ++k)
        {
            if (cv::norm(points[k] - det.corners[k]) > 4.0)
            {
                sane = false;
                break;
            }
        }
        if (!sane)
        {
            continue;
        }
        for (int k = 0; k < 4; ++k)
        {
            det.corners[k] = points[k];
        }
        ++refined;
    }
    return refined;
}

std::vector<ArmorPose> armor_solve(const std::vector<ArmorDetection> &detections,
                                   const CameraParameters &camera,
                                   const GimbalState &gimbal)
{
    (void)gimbal; // 云台姿态补偿不在此层：armor_solve 输出云台系，
                  // gimbal->world 由 RotatingArmorTracker::collectWorldObservations 完成
    std::vector<ArmorPose> poses;

    for (const ArmorDetection &det : detections)
    {
        const std::vector<cv::Point3f> object_points = buildObjectPoints(det.size);
        if (object_points.empty())
        {
            continue; // 物点未建模（TODO 未完成）时静默跳过
        }

        // 图像点：corners 本身就是 TL,TR,BR,BL 顺序，直接搬进 vector
        std::vector<cv::Point2f> image_points(det.corners.begin(),
                                              det.corners.end());

        // ---- 图像域时间平均（CORNER_SMOOTH_ALPHA，0/未设 = 关闭）----
        // 在角点上平均而不是在三维位姿上平均：噪声本来就是角点层面的，
        // 先解 PnP 再平滑会把病态的深度歧义一起平均进来（试过，无效）。
        static const char *corner_smooth_env = std::getenv("CORNER_SMOOTH_ALPHA");
        static const double corner_smooth_alpha =
            corner_smooth_env != nullptr ? std::atof(corner_smooth_env) : 0.0;
        if (corner_smooth_alpha > 0.0 && corner_smooth_alpha <= 1.0)
        {
            // 与 solvePoseForTrack 共用同一份历史（同 id/尺寸），避免多目标互相污染。
            static std::map<std::pair<int, int>, std::array<cv::Point2f, 4>> smooth_history;
            auto &history = smooth_history[{det.target_id, static_cast<int>(det.size)}];
            static std::map<std::pair<int, int>, std::chrono::steady_clock::time_point>
                smooth_time;
            auto &last_time = smooth_time[{det.target_id, static_cast<int>(det.size)}];
            const auto now = std::chrono::steady_clock::now();
            bool reset = true;
            if (history[0].x != 0.0F || history[0].y != 0.0F)
            {
                double max_shift = 0.0;
                for (std::size_t k = 0; k < 4; ++k)
                    max_shift = std::max(max_shift,
                                         cv::norm(det.corners[k] - history[k]));
                const bool stale = last_time.time_since_epoch().count() != 0 &&
                                   now - last_time > std::chrono::milliseconds(120);
                reset = stale || max_shift > 28.0; // 换板/跳变则不跨帧平均
            }
            for (std::size_t k = 0; k < 4; ++k)
            {
                if (reset)
                    history[k] = det.corners[k];
                else
                    history[k] = history[k] * static_cast<float>(1.0 - corner_smooth_alpha) +
                                 det.corners[k] * static_cast<float>(corner_smooth_alpha);
                image_points[k] = history[k];
            }
            last_time = now;
        }

        // ---- 深度先验：用竖直边（灯条端点）长度反推深度（DEPTH_FROM_EDGES=1）----
        static const bool depth_from_edges = std::getenv("DEPTH_FROM_EDGES") != nullptr;
        double depth_prior = 0.0;
        if (depth_from_edges)
        {
            const double plate_h = 0.057;
            const double top_h = cv::norm(image_points[0] - image_points[3]);
            const double bottom_h = cv::norm(image_points[1] - image_points[2]);
            const double h_px = 0.5 * (top_h + bottom_h);
            const double fy = camera.camera_matrix.at<double>(1, 1);
            if (h_px > 4.0 && fy > 1.0)
                depth_prior = fy * plate_h / h_px;
        }

        // PnP：4 个共面点。IPPE 可能返回两个相近重投影误差的姿态解；
        // 按目标编号/尺寸维护历史法向，避免逐帧在两个解之间跳变。
        cv::Mat rvec, tvec;
        const bool ok = solvePoseForTrack(object_points, image_points,
                                          camera.camera_matrix,
                                          camera.distortion_coefficients,
                                          det.target_id, det.size, rvec, tvec,
                                          depth_prior);
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

        // 外参变换：p_gimbal = R * p_camera + t
        // 占位外参（单位阵）时按"相机光学系 → 云台系"的固定旋转处理，见
        // rotationCameraToGimbal()。这不改变协议与 PredictionResult 语义，
        // 只是把 p_camera 的 OpenCV 轴重排成云台系的前/左/上。
        pose.position_gimbal_m = cameraToGimbalPoint(camera, pose.position_camera_m);

        // 板面法向：armor_yaw 与 normal_gimbal（都在云台系）。
        //   1) cv::Rodrigues(rvec, R_obj2cam) 把旋转向量变成 3x3 旋转矩阵
        //   2) 物点建模时 z 轴是板面外法向，所以在相机系 n = R_obj2cam * (0,0,1)
        //   3) 再乘外参旋转到云台系：n_gimbal = R_cam2gimbal * n_cam
        //   4) armor_yaw = atan2(n_gimbal.y, n_gimbal.x)，与 position_gimbal_m 同一系
        // 云台姿态补偿（gimbal -> 世界系）在 EKF 阶段做，这里只用外参。
        cv::Mat R_obj2cam;
        cv::Rodrigues(rvec, R_obj2cam);
        const cv::Mat n_cam = R_obj2cam * (cv::Mat_<double>(3, 1) << 0.0, 0.0, 1.0);
        const cv::Mat n_gimbal = rotationCameraToGimbal(camera) * n_cam;
        const double nx = n_gimbal.at<double>(0);
        const double ny = n_gimbal.at<double>(1);
        const double nz = n_gimbal.at<double>(2);
        const double n_norm = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (n_norm > 1e-9)
        {
            cv::Vec3d n(nx / n_norm, ny / n_norm, nz / n_norm);
            // 法向必须"指向相机"（能看见的那一面）：在同一坐标系里 n·p < 0。
            // 实测 v4 全片段 896 帧，IPPE 给出的物体系 +Z 与 p 同向（n·p 中位 +1.22、
            // 100% 为正），即法向朝里。这个符号会一路影响：相位初值、按法向反推车心、
            // 法向关联——不统一就会把车心放到板的近侧，俯视图上模型板/aim 全落在远端。
            const cv::Vec3d p = pose.position_gimbal_m;
            if (n[0] * p[0] + n[1] * p[1] + n[2] * p[2] > 0.0)
                n = cv::Vec3d(-n[0], -n[1], -n[2]);
            pose.normal_gimbal = n;
            pose.armor_yaw = std::atan2(n[1], n[0]);
        }

        poses.push_back(pose);
    }
    return poses;
}

// ==================== 任务 4：EKF 跟踪与预测 ====================

namespace
{

    // 跟踪器持有跨帧状态：锁定样本、EKF、瞄准槽位、丢失计时。
    // 处理线程是单线程顺序调用，所以这里不需要加锁。
    rotating_armor::RotatingArmorTracker &trackerInstance()
    {
        static rotating_armor::RotatingArmorTracker instance;
        return instance;
    }

    // 锁定过程对现场调试是黑箱：只在"阶段 / 已攒样本 / 最近被拒原因 / 重锁次数"变化时
    // 打印一行，避免盯着 crc 从 0 跳到 1 干等（实机 CSV 里曾有连续 20s 一行输出都没有）。
    const char *lockPhaseName(rotating_armor::LockPhase phase)
    {
        switch (phase)
        {
        case rotating_armor::LockPhase::Idle:
            return "idle";
        case rotating_armor::LockPhase::Collecting:
            return "collecting";
        case rotating_armor::LockPhase::Tracking:
            return "tracking";
        }
        return "?";
    }

    void logTrackerDiagnostics(const rotating_armor::TrackerDiagnostics &diagnostics)
    {
        static std::string last_key;
        const std::string key = std::string(lockPhaseName(diagnostics.phase)) + "|" +
                                diagnostics.last_reject + "|" +
                                std::to_string(std::min(diagnostics.lock_samples, 6)) + "|" +
                                std::to_string(diagnostics.relock_count);
        if (key == last_key)
            return;
        last_key = key;
        std::fprintf(stderr, "[ekf] phase=%s lock_samples=%d/%d relock=%d reject=%s\n",
                     lockPhaseName(diagnostics.phase), diagnostics.lock_samples,
                     trackerInstance().config().lock_min_samples,
                     diagnostics.relock_count, diagnostics.last_reject.c_str());
    }

} // namespace

PredictionResult ekf_predict(const std::vector<ArmorPose> &observations, const GimbalState &gimbal,
                             double timestamp_seconds)
{
    // 本帧视觉侧处理延迟：帧时间戳是取流时刻，这里补上"到现在为止"已经花掉的
    // 时间（检测 + PnP + 调用开销）。电控下发的 prediction_bias_s 是另一个环节
    // 的补偿，两者相加才是子弹到达时的总预测时域。
    // 只有测到合理值（0 < delta < 0.3s）才采用，避免时钟口径不一致时误补偿。
    double measured_latency_s = 0.0;
    if (std::isfinite(timestamp_seconds))
    {
        const double now_s = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now().time_since_epoch())
                                 .count();
        const double elapsed = now_s - timestamp_seconds;
        if (elapsed > 0.0 && elapsed < 0.3)
        {
            measured_latency_s = elapsed;
        }
    }

    // 观测关联/门限/状态机/输出映射都在 rotating_armor::RotatingArmorTracker 里：
    // 它负责把云台系观测转到世界系、攒样本锁定初相与角速度、逐帧预测与 NIS 门限更新、
    // 换板与瞄准板选择、开火许可，并在超时/声明被反驳时重锁。
    rotating_armor::RotatingArmorTracker &tracker = trackerInstance();
    const PredictionResult prediction =
        tracker.step(observations, gimbal, timestamp_seconds, measured_latency_s);
    logTrackerDiagnostics(tracker.diagnostics());
    return prediction;
}
