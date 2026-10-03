#pragma once

#include <string>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

// 装甲板数字分类器封装（任务 2 配套，已由 mentor 验证）。
//
// 负责：加载 models/armor_cls_64.onnx、预处理、前向推理、取 top1。
// 输入约定：crop 为 BGR 图像，内容应"紧贴数字"（见下方警告）。
//
// 预处理与训练一致（验证记录见 models/README.md）：
//   灰度 -> Otsu 二值化 -> 短边缩放到 64 -> 中心裁剪 64x64 -> /255 -> NCHW(RGB)
//   ONNX 输出为 softmax 概率，直接取 argmax。
//
// 警告：训练裁片里数字几乎充满整幅图。若传入的区域数字占比过小
// （实测 < ~10% 画面），会被误判为 9neg。调用前请先裁剪到数字外接框附近。
//
// 类别下标 -> target_id 映射：index 0..7 -> 1..8（含 6outpost/7guard/8base），
// index 8 (9neg) -> target_id 0，表示"不是数字"，调用方必须丢弃。
class ArmorClassifier {
public:
    struct Result {
        int class_index{-1};   // -1 表示输入无效或推理失败
        int target_id{0};      // 0 = 非数字（9neg）；1..8 为有效编号
        float confidence{0.0F};
    };

    // 加载 ONNX 模型。失败时 error 写入原因，返回 false。
    bool load(const std::string &onnx_path, std::string &error);

    bool ready() const { return ready_; }

    // 对一幅"紧贴数字"的 BGR 裁剪推理。输入为空时返回 class_index=-1。
    // 注意：本函数修改内部 Net 状态，只应在单一（操作）线程调用。
    Result infer(const cv::Mat &crop_bgr);

    static constexpr int kNegativeClass = 8;  // '9neg'
    static constexpr int kNumClasses = 9;

private:
    cv::dnn::Net net_;
    bool ready_{false};
};
