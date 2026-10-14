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

// 大小板几何判定（yolov5 输出不含大小类）：
// 与传统路线 classifyArmorSize 同标准——外缘宽（上下边均值）/ 条高（左右边均值）
// ≈2.37 小板 / ≈4.04 大板，取偏差小者。
ArmorSize classifySizeByGeometry(const std::array<cv::Point2f, 4> &c)
{
    const double outer_w = (cv::norm(c[0] - c[1]) + cv::norm(c[3] - c[2])) / 2.0;
    const double h_avg = (cv::norm(c[0] - c[3]) + cv::norm(c[1] - c[2])) / 2.0;
    if (h_avg < 1.0)
    {
        return ArmorSize::Small;
    }
    const double ratio = outer_w / h_avg;
    const double d_small = std::fabs(ratio - 2.37);
    const double d_large = std::fabs(ratio - 4.04);
    return d_large < d_small ? ArmorSize::Large : ArmorSize::Small;
}

} // namespace

bool ArmorDetectorNet::load(const std::string &xml_path, ModelKind kind, std::string &error)
{
    try
    {
        // OpenVINO 直接读 IR（bin 同名换后缀自动查找），CPU + LATENCY 模式，
        // 与 sp_vision_25 standard3 配置一致。
        std::shared_ptr<ov::Model> model = core_.read_model(xml_path);
        compiled_ = core_.compile_model(
            model, "CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
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

    // 预处理：等比缩放 + 左上对齐放入 640x640 黑画布（letterbox），
    // 与 sp_vision_25 一致——padding 在右下，还原坐标只需 /scale。
    const double scale =
        std::min(static_cast<double>(kInputSize) / bgr.rows,
                 static_cast<double>(kInputSize) / bgr.cols);
    const int w = static_cast<int>(bgr.cols * scale);
    const int h = static_cast<int>(bgr.rows * scale);
    cv::Mat canvas(kInputSize, kInputSize, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::resize(bgr, canvas(cv::Rect(0, 0, w, h)), cv::Size(w, h));

    // BGR -> RGB，/255，NCHW FP32，直接拷贝进输入 tensor。
    const cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0 / 255.0,
                                                cv::Size(kInputSize, kInputSize),
                                                cv::Scalar(), true, false);
    ov::Tensor input_tensor = infer_request_.get_input_tensor();
    std::memcpy(input_tensor.data<float>(), blob.ptr<float>(),
                blob.total() * sizeof(float));
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
        det.size = classifySizeByGeometry(det.corners);
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
        det.size = prop.large ? ArmorSize::Large : ArmorSize::Small;
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
