#include "ArmorDetectorNet.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

// 解码布局与 sp_vision_25 tasks/auto_aim/yolos/{yolov5,yolo11}.cpp 严格一致；
// 语义映射表见 models/README.md 与本文件注释。
namespace
{

// yolo11 的 38 类属性表（sp_vision_25 tasks/auto_aim/armor.hpp armor_properties）。
// 颜色枚举：0=red 1=blue 2=extinguish 3=purple；编号为本工程 target_id（0=丢弃）；
// 大小：true=Large。
struct ArmorProp
{
    int color_id;   // 0=red 1=blue 2=extinguish 3=purple
    int target_id;  // 工程编号体系 1..8
    bool large;
};
// clang-format off
constexpr ArmorProp kYolo11Props[] = {
    // sentry(small): blue / red / extinguish
    {1, 7, false}, {0, 7, false}, {2, 7, false},
    // one(small)
    {1, 1, false}, {0, 1, false}, {2, 1, false},
    // two(small)
    {1, 2, false}, {0, 2, false}, {2, 2, false},
    // three(small)
    {1, 3, false}, {0, 3, false}, {2, 3, false},
    // four(small)
    {1, 4, false}, {0, 4, false}, {2, 4, false},
    // five(small)
    {1, 5, false}, {0, 5, false}, {2, 5, false},
    // outpost(small)
    {1, 6, false}, {0, 6, false}, {2, 6, false},
    // base(big): blue / red / extinguish / purple
    {1, 8, true}, {0, 8, true}, {2, 8, true}, {3, 8, true},
    // base(small)
    {1, 8, false}, {0, 8, false}, {2, 8, false}, {3, 8, false},
    // three(big)
    {1, 3, true}, {0, 3, true}, {2, 3, true},
    // four(big)
    {1, 4, true}, {0, 4, true}, {2, 4, true},
    // five(big)
    {1, 5, true}, {0, 5, true}, {2, 5, true},
};
// clang-format on
constexpr int kYolo11ClassNum = 38;

// yolov5 编号 one-hot 顺序（sp_vision_25 Armor(color_id, num_id, ...) 构造函数）：
// 0=sentry 1=one 2=two 3=three 4=four 5=five 6=outpost 7=base 8=not_armor。
// 映射到工程 target_id：sentry->7, one..five->1..5, outpost->6, base->8, 8->0(丢弃)。
constexpr int kYolov5NumToTargetId[9] = {7, 1, 2, 3, 4, 5, 6, 8, 0};

double sigmoid(double x)
{
    // 与 sp_vision_25 相同的数值稳定写法。
    return x > 0 ? 1.0 / (1.0 + std::exp(-x)) : std::exp(x) / (1.0 + std::exp(x));
}

// 由四角点外接框构造 NMS 用矩形。
cv::Rect boundingRectOf(const std::array<cv::Point2f, 4> &corners)
{
    float min_x = corners[0].x, max_x = corners[0].x;
    float min_y = corners[0].y, max_y = corners[0].y;
    for (int i = 1; i < 4; ++i)
    {
        min_x = std::min(min_x, corners[i].x);
        max_x = std::max(max_x, corners[i].x);
        min_y = std::min(min_y, corners[i].y);
        max_y = std::max(max_y, corners[i].y);
    }
    return cv::Rect(static_cast<int>(min_x), static_cast<int>(min_y),
                    static_cast<int>(max_x - min_x), static_cast<int>(max_y - min_y));
}

// ===== 大小板几何估计（yolov5 输出不含大小类）=====
//
// ratio = 外缘宽（上下边均值）/ 条高（左右边均值）
// 理论值：小板 135/57 = 2.368，大板 230/57 = 4.035。
//
// 【关键性质】板绕竖轴转过 yaw 角 θ 后，横向被透视压缩 cosθ、竖向几乎不受影响：
//       ratio_obs ≈ ratio_true · cosθ      (cosθ ≤ 1)
// 即 ratio_obs 只会「偏小」，永远不可能超过真值——这是一个单向不等式。
//
// 旧实现在 2.37 / 4.04 之间取「偏差小者」，等价于在 (2.368+4.035)/2 = 3.20 处
// 硬切，于是大板只要 cosθ < 3.20/4.035 = 0.793（θ > 37.5°）就被判成小板，
// 板宽按 135mm 而不是 230mm 进 PnP，深度直接错 230/135 = 1.70 倍，位置跟着
// 跳 0.3~0.8 m；θ 在 37.5° 附近抖动时尺寸逐帧翻，半径先验（0.20/0.25 m）、
// half_width（0.0675/0.115 m）同时翻——这就是"观测源跳变"的机械来源。
//
// 因此这里只做单向判定：ratio 达标 => 必然大板；不达标 => 无法证明，先按小板
// 占位，最终尺寸交给上层（编号先验 + 时域上包络），见 StudentTasks.cpp。
constexpr double kLargePlateRatio = 230.0 / 57.0; // 4.035（大板正视图理论值）
constexpr double kSmallPlateRatio = 135.0 / 57.0; // 2.368（小板正视图理论值）
// 大板证书阈值：介于小板最大值 2.368 与旧硬切点 3.20 之间，取 3.00 ——
// 往 2.368 一侧留 27% 余量吸收角点噪声，往 4.035 一侧仍能容忍
// cosθ ≥ 3.00/4.035 = 0.743，即 |θ| ≤ 42°（旧阈值只有 37.5°）。
constexpr double kLargeCertificateRatio = 3.00;
// 证书阈值必须落在 (小板理论值, 大板理论值) 之间，否则单向性被破坏：
// 低于 2.368 会把正对小板的帧判成大板，高于 4.035 则任何角度都不出证书。
static_assert(kLargeCertificateRatio > kSmallPlateRatio &&
                  kLargeCertificateRatio < kLargePlateRatio,
              "大板证书阈值必须严格落在小板/大板理论比例之间");

// 测量板面比例。返回 false 表示角点退化（贴图边、共线、NaN），调用方按
// "无法判断" 处理，不要退回任何默认值。
bool measurePlateRatio(const std::array<cv::Point2f, 4> &c, double &ratio)
{
    const double outer_w = (cv::norm(c[0] - c[1]) + cv::norm(c[3] - c[2])) / 2.0;
    const double h_avg = (cv::norm(c[0] - c[3]) + cv::norm(c[1] - c[2])) / 2.0;
    // 条高 <4px 时角点量化误差就能把 ratio 翻倍，测了也没意义。
    if (h_avg < 4.0 || !std::isfinite(outer_w) || !std::isfinite(h_avg))
    {
        return false;
    }
    ratio = outer_w / h_avg;
    return std::isfinite(ratio) && ratio > 0.0;
}

} // namespace

bool ArmorDetectorNet::load(const std::string &xml_path, ModelKind kind,
                            int input_width, int input_h, const std::string &device,
                            std::string &error)
{
    try
    {
        input_width_ = input_width;
        input_height_ = input_h > 0 ? input_h : input_width;
        // OpenVINO 直接读 IR（bin 同名换后缀自动查找），LATENCY 模式，
        // 与 sp_vision_25 standard3 配置一致。
        std::shared_ptr<ov::Model> model = core_.read_model(xml_path);
        // 预处理融合进模型图：u8 BGR NHWC 直投，模型内转 RGB + /255，
        // 比 blobFromImage 少两次全图拷贝。
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input().tensor()
            .set_element_type(ov::element::u8)
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR)
            .set_shape({1, static_cast<std::int64_t>(input_height_),
                        static_cast<std::int64_t>(input_width_), 3});
        ppp.input().model().set_layout("NCHW");
        ppp.input().preprocess()
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .convert_element_type(ov::element::f32)
            .scale(255.0F);
        model = ppp.build();
        // 非 CPU 设备编译失败（无 /dev/dri 等）时自动回退 CPU。
        std::string target = device.empty() ? "CPU" : device;
        try
        {
            compiled_ = core_.compile_model(
                model, target,
                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
        }
        catch (const std::exception &e)
        {
            if (target == "CPU")
            {
                throw;
            }
            std::cerr << "设备 " << target << " 编译失败（" << e.what()
                      << "），回退 CPU\n";
            target = "CPU";
            compiled_ = core_.compile_model(
                model, target,
                ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
        }
        device_ = target;
        infer_request_ = compiled_.create_infer_request();
        kind_ = kind;
        ready_ = true;
        return true;
    }
    catch (const std::exception &e)
    {
        error = e.what();
        ready_ = false;
        return false;
    }
}

std::vector<ArmorDetectorNet::Detection> ArmorDetectorNet::detect(const cv::Mat &bgr)
{
    std::vector<Detection> results;
    if (!ready_ || bgr.empty() || bgr.type() != CV_8UC3)
    {
        return results;
    }
    // 黑帧早退：main 连续取流失败时会补黑帧走完整链路，没必要浪费一次推理。
    // 实拍曝光下正常帧三通道均值远大于此阈值。
    const cv::Scalar mean = cv::mean(bgr);
    if (mean[0] + mean[1] + mean[2] < 3.0)
    {
        return results;
    }

    // 预处理：等比缩放 + 左上对齐放入黑画布（letterbox，与 sp_vision_25 一致，
    // padding 在右下，还原坐标只需 /scale）。画布复用，每帧清零后写入。
    const double scale =
        std::min(static_cast<double>(input_height_) / bgr.rows,
                 static_cast<double>(input_width_) / bgr.cols);
    const int w = static_cast<int>(bgr.cols * scale);
    const int h = static_cast<int>(bgr.rows * scale);
    if (canvas_.empty())
    {
        canvas_ = cv::Mat(input_height_, input_width_, CV_8UC3);
    }
    canvas_.setTo(cv::Scalar(0, 0, 0));
    cv::resize(bgr, canvas_(cv::Rect(0, 0, w, h)), cv::Size(w, h));

    // u8 BGR 直投输入 tensor，BGR->RGB 与 /255 由模型内预处理完成。
    ov::Tensor input_tensor = infer_request_.get_input_tensor();
    std::memcpy(input_tensor.data(), canvas_.data,
                canvas_.total() * canvas_.elemSize());
    infer_request_.infer();
    const ov::Tensor output = infer_request_.get_output_tensor();

    return kind_ == ModelKind::YOLOv5 ? decodeYolov5(output, scale)
                                      : decodeYolo11(output, scale);
}

std::vector<ArmorDetectorNet::Detection>
ArmorDetectorNet::decodeYolov5(const ov::Tensor &output, double scale) const
{
    // 输出 1x25200x22，逐行解析（只读，const_cast 仅为借用内存）。
    const ov::Shape shape = output.get_shape();
    cv::Mat rows(static_cast<int>(shape[1]), static_cast<int>(shape[2]), CV_32F,
                 const_cast<float *>(output.data<float>()));

    std::vector<Detection> candidates;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    for (int r = 0; r < rows.rows; ++r)
    {
        const float *row = rows.ptr<float>(r);
        const float score = static_cast<float>(sigmoid(row[8]));
        if (score < score_threshold_)
        {
            continue;
        }

        // 颜色：0=blue 1=red 其他=extinguish（丢弃）。
        int color_id = 0;
        float best = row[9];
        for (int i = 1; i < 4; ++i)
        {
            if (row[9 + i] > best)
            {
                best = row[9 + i];
                color_id = i;
            }
        }
        if (color_id > 1)
        {
            continue;
        }

        // 编号 argmax（cols 13..22）。
        int num_id = 0;
        best = row[13];
        for (int i = 1; i < 9; ++i)
        {
            if (row[13 + i] > best)
            {
                best = row[13 + i];
                num_id = i;
            }
        }
        const int target_id = kYolov5NumToTargetId[num_id];
        if (target_id == 0)
        {
            continue; // not_armor
        }

        Detection det;
        // 列布局关键点顺序 TL,BL,BR,TR -> 工程约定 TL,TR,BR,BL。
        det.corners[0] = {row[0] / static_cast<float>(scale),
                          row[1] / static_cast<float>(scale)};
        det.corners[1] = {row[6] / static_cast<float>(scale),
                          row[7] / static_cast<float>(scale)};
        det.corners[2] = {row[4] / static_cast<float>(scale),
                          row[5] / static_cast<float>(scale)};
        det.corners[3] = {row[2] / static_cast<float>(scale),
                          row[3] / static_cast<float>(scale)};
        det.box = boundingRectOf(det.corners);
        det.color = color_id == 0 ? TeamColor::Blue : TeamColor::Red;
        det.target_id = target_id;
        // 尺寸不在本层定死：只输出测量值与单向证书，见文件头推导。
        double ratio = 0.0;
        det.ratio_valid = measurePlateRatio(det.corners, ratio);
        det.plate_ratio = static_cast<float>(ratio);
        det.size = (det.ratio_valid && ratio >= kLargeCertificateRatio) ? ArmorSize::Large
                                                                       : ArmorSize::Small;
        det.confidence = score;

        candidates.push_back(det);
        boxes.push_back(det.box);
        confidences.push_back(score);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

    std::vector<Detection> results;
    results.reserve(indices.size());
    for (const int i : indices)
    {
        results.push_back(candidates[i]);
    }
    return results;
}

std::vector<ArmorDetectorNet::Detection>
ArmorDetectorNet::decodeYolo11(const ov::Tensor &output, double scale) const
{
    // 输出 1x50x8400 -> transpose 成 8400x50 逐行解析（只读，const_cast 仅为借用内存）。
    const ov::Shape shape = output.get_shape();
    cv::Mat transposed;
    {
        cv::Mat raw(static_cast<int>(shape[1]), static_cast<int>(shape[2]), CV_32F,
                    const_cast<float *>(output.data<float>()));
        cv::transpose(raw, transposed);
    }

    std::vector<Detection> candidates;
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;

    for (int r = 0; r < transposed.rows; ++r)
    {
        const float *row = transposed.ptr<float>(r);

        // 38 类分数 argmax（cols 4..42）。
        int class_id = 0;
        float score = row[4];
        for (int i = 1; i < kYolo11ClassNum; ++i)
        {
            if (row[4 + i] > score)
            {
                score = row[4 + i];
                class_id = i;
            }
        }
        if (score < score_threshold_)
        {
            continue;
        }

        const ArmorProp &prop = kYolo11Props[class_id];
        if (prop.color_id > 1)
        {
            continue; // extinguish / purple
        }

        Detection det;
        const float inv = 1.0F / static_cast<float>(scale);
        std::array<cv::Point2f, 4> kps{};
        for (int i = 0; i < 4; ++i)
        {
            kps[i] = {row[42 + i * 2] * inv, row[43 + i * 2] * inv};
        }
        // 排序为 TL,TR,BR,BL（sp_vision_25 sort_keypoints：先按 y 分组，组内按 x）。
        std::sort(kps.begin(), kps.end(),
                  [](const cv::Point2f &a, const cv::Point2f &b) { return a.y < b.y; });
        cv::Point2f top[2] = {kps[0], kps[1]};
        cv::Point2f bottom[2] = {kps[2], kps[3]};
        if (top[1].x < top[0].x)
        {
            std::swap(top[0], top[1]);
        }
        if (bottom[1].x < bottom[0].x)
        {
            std::swap(bottom[0], bottom[1]);
        }
        det.corners = {top[0], top[1], bottom[1], bottom[0]};
        det.box = boundingRectOf(det.corners);
        det.color = prop.color_id == 0 ? TeamColor::Red : TeamColor::Blue;
        det.target_id = prop.target_id;
        // 尺寸由 38 类类别直接给出（颜色×编号×大小），这是唯一可靠的来路。
        det.size = prop.large ? ArmorSize::Large : ArmorSize::Small;
        // ratio 仍测一份，但 ratio_valid=false 明确告诉上层"别用 ratio 反推"，
        // 免得把模型给出的可靠尺寸又退回到几何猜测。
        {
            double ratio = 0.0;
            measurePlateRatio(det.corners, ratio);
            det.plate_ratio = static_cast<float>(ratio);
            det.ratio_valid = false;
        }
        det.confidence = score;

        candidates.push_back(det);
        boxes.push_back(det.box);
        confidences.push_back(score);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

    std::vector<Detection> results;
    results.reserve(indices.size());
    for (const int i : indices)
    {
        results.push_back(candidates[i]);
    }
    return results;
}
