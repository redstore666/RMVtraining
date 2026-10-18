#pragma once

#include "AutoAimTypes.hpp"

#include <array>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

// 神经网络装甲板四点检测器封装（任务 2 神经网络路线，当前默认路线）。
//
// 推理引擎：OpenVINO（与模型来源 sp_vision_25 一致，IR 模型原生加载，
// 避免图转换引入的关键点语义偏差）。模型文件为 OpenVINO IR：
//   * YOLOv5（上场默认，models/yolov5.xml|bin）：输出 1x25200x22，
//     行布局 [4关键点(TL,BL,BR,TR) | objectness | 颜色4类 | 编号9类]；
//   * YOLO11（备选，models/yolo11.xml|bin）：输出 1x50x8400，
//     行布局 [xywh | 38类(颜色x编号x大小) | 4关键点]。
// 解码布局与 sp_vision_25 tasks/auto_aim/yolos/{yolov5,yolo11}.cpp 严格一致；
// 完整 spec 见 models/README.md。
//
// 语义映射到本工程约定：
//   * 颜色：仅保留 red/blue，extinguish/purple 直接丢弃（不可击打状态）；
//   * 编号：sentry->7(7guard)、one..five->1..5、outpost->6、base->8，
//     not_armor 丢弃；
//   * 大小板：yolo11 查 38 类属性表；yolov5 输出不含大小，
//     用四角点几何 ratio（外缘宽/条高 ≈2.37 小 / 4.04 大）判定。
//
// 注意：detect() 复用内部 InferRequest，只应在单一（操作）线程调用。
class ArmorDetectorNet
{
public:
    enum class ModelKind
    {
        YOLOv5,
        YOLO11,
    };

    struct Detection
    {
        // 图像角点按顺时针排列：左上、右上、右下、左下（与 ArmorDetection 一致）。
        std::array<cv::Point2f, 4> corners{};
        cv::Rect box{};                  // 关键点外接框（原图坐标）
        TeamColor color{TeamColor::Red}; // 仅 Red/Blue，其余已在解码时丢弃
        int target_id{0};                // 1..8（工程编号体系）
        ArmorSize size{ArmorSize::Small};
        float confidence{0.0F}; // objectness（yolov5）/ 38类分数（yolo11）
    };

    // 加载 OpenVINO IR 模型（xml 路径，bin 同名换后缀自动查找）。
    // input_width/input_height 为推理输入尺寸（须与模型实际尺寸一致；非 640x640
    // 的模型需先用 tools/make_small_ir.py 生成，例如
    // `make_small_ir.py models/yolov5.xml models/yolov5_640x480 640x480`）。
    // 相机是 4:3，640x480 输入与 640x640 相比图像内容与缩放比例完全相同，只是
    // 不再为 160 行黑边算卷积，计算量少 25%。input_height 传 0 表示正方形。
    // device 为推理设备（"CPU"/"GPU"/"AUTO"；非 CPU 设备编译失败时自动回退 CPU）。
    // 预处理（BGR->RGB、/255）由 OpenVINO PrePostProcessor 融合进模型，
    // detect() 直接投 u8 letterbox 图像，省去两次全图拷贝。
    bool load(const std::string &xml_path, ModelKind kind, int input_width,
              int input_height, const std::string &device, std::string &error);

    bool ready() const { return ready_; }
    ModelKind kind() const { return kind_; }
    int input_size() const { return input_width_; } // 兼容旧名字：返回宽度
    int input_width() const { return input_width_; }
    int input_height() const { return input_height_; }
    const std::string &device() const { return device_; }
    // 检测置信度阈值：默认 0.7（sp_vision_25 standard3 配置）。复检/跟踪阶段可以
    // 降到 0.3~0.5 换取召回（自转目标实拍只有约一半的帧能过 0.7）。
    void setScoreThreshold(float threshold) { score_threshold_ = threshold; }
    float scoreThreshold() const { return score_threshold_; }

    // 对一帧 BGR 图像推理。输入为空或模型未就绪时返回空。
    std::vector<Detection> detect(const cv::Mat &bgr);

    static constexpr int kInputSize = 640;

private:
    std::vector<Detection> decodeYolov5(const ov::Tensor &output, double scale) const;
    std::vector<Detection> decodeYolo11(const ov::Tensor &output, double scale) const;

    ov::Core core_;
    ov::CompiledModel compiled_;
    ov::InferRequest infer_request_;
    ModelKind kind_{ModelKind::YOLOv5};
    int input_width_{kInputSize};
    int input_height_{kInputSize};
    std::string device_{"CPU"};
    cv::Mat canvas_; // 复用的 letterbox 画布（单线程使用）
    bool ready_{false};
    float score_threshold_{0.7F}; // 与 sp_vision_25 standard3 配置一致
    float nms_threshold_{0.3F};
};
