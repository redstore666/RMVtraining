#include "ArmorClassifier.hpp"

#include <algorithm>

#include <opencv2/imgproc.hpp>

namespace {

constexpr int kInputSize = 64;  // 与训练/导出时的 imgsz 一致

// 复刻训练侧 classify_transforms(64) 的推理预处理，并加一步 Otsu 二值化
// （训练图本身是白字黑底的二值图，实拍裁剪先二值化可贴近训练分布）。
cv::Mat preprocess(const cv::Mat &crop_bgr) {
    cv::Mat gray;
    if (crop_bgr.channels() == 3) {
        cv::cvtColor(crop_bgr, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = crop_bgr;
    }

    cv::Mat binary;
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Mat rgb;
    cv::cvtColor(binary, rgb, cv::COLOR_GRAY2RGB);

    // 短边缩放到 64（截断取整，与 torchvision Resize(int) 一致）
    const int w = rgb.cols;
    const int h = rgb.rows;
    int new_w = 0;
    int new_h = 0;
    if (w <= h) {
        new_w = kInputSize;
        new_h = static_cast<int>(static_cast<double>(kInputSize) * h / w);
    } else {
        new_h = kInputSize;
        new_w = static_cast<int>(static_cast<double>(kInputSize) * w / h);
    }
    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 中心裁剪 64x64（整除取整与 torchvision CenterCrop 一致）
    const cv::Rect roi((new_w - kInputSize) / 2, (new_h - kInputSize) / 2,
                       kInputSize, kInputSize);
    return resized(roi);
}

}  // namespace

bool ArmorClassifier::load(const std::string &onnx_path, std::string &error) {
    try {
        net_ = cv::dnn::readNetFromONNX(onnx_path);
    } catch (const cv::Exception &e) {
        error = "加载 ONNX 失败: " + std::string(e.what());
        ready_ = false;
        return false;
    }
    ready_ = !net_.empty();
    if (!ready_) {
        error = "ONNX 为空或路径错误: " + onnx_path;
    }
    return ready_;
}

ArmorClassifier::Result ArmorClassifier::infer(const cv::Mat &crop_bgr) {
    Result result;
    if (!ready_ || crop_bgr.empty() || crop_bgr.cols < 8 || crop_bgr.rows < 8) {
        return result;
    }

    const cv::Mat face = preprocess(crop_bgr);
    // 已经是 64x64 三通道 RGB，swapRB=false，无均值，仅 /255。
    const cv::Mat blob =
        cv::dnn::blobFromImage(face, 1.0 / 255.0, cv::Size(kInputSize, kInputSize),
                               cv::Scalar(), false, false);
    net_.setInput(blob);
    const cv::Mat out = net_.forward().reshape(1, 1);

    int best = 0;
    float best_value = out.at<float>(0, 0);
    for (int i = 1; i < std::min(out.cols, kNumClasses); ++i) {
        const float value = out.at<float>(0, i);
        if (value > best_value) {
            best_value = value;
            best = i;
        }
    }

    result.class_index = best;
    result.confidence = best_value;
    result.target_id = (best == kNegativeClass) ? 0 : best + 1;
    return result;
}
