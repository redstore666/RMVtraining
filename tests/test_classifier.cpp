// 数字分类器独立测试工具。
//
// 用法:
//   ./build/test_classifier <图片> [--onnx 路径] [--expect <类别名>]
//
// 示例:
//   ./build/test_classifier tests/data/sticker_1.png --expect 1
//   ./build/test_classifier tests/data/sticker_9neg.jpg --expect 9neg
//
// 校验分类器（ONNX + 预处理）是否工作正常；--expect 不一致时返回 1，
// 便于脚本或 ctest 检查。图片应"紧贴数字"，如数据集裁片。
#include "ArmorClassifier.hpp"

#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

namespace {

const char *const kClassNames[] = {"1", "2",   "3",      "4",      "5",
                                   "6outpost", "7guard", "8base", "9neg"};

bool loadWithFallback(ArmorClassifier &classifier, const std::string &path,
                      std::string &error) {
    if (classifier.load(path, error)) {
        return true;
    }
    // 从 build/ 目录运行时，尝试相对仓库根目录的路径。
    return classifier.load("../" + path, error);
}

}  // namespace

int main(int argc, char **argv) {
    std::string image_path;
    std::string onnx_path = "models/armor_cls_64.onnx";
    std::string expect;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--onnx" && i + 1 < argc) {
            onnx_path = argv[++i];
        } else if (arg == "--expect" && i + 1 < argc) {
            expect = argv[++i];
        } else if (!arg.empty() && arg[0] != '-') {
            image_path = arg;
        } else {
            std::cerr << "未知参数: " << arg << "\n";
            return 2;
        }
    }
    if (image_path.empty()) {
        std::cerr << "用法: " << argv[0]
                  << " <图片> [--onnx 路径] [--expect <类别名>]\n";
        return 2;
    }

    ArmorClassifier classifier;
    std::string error;
    if (!loadWithFallback(classifier, onnx_path, error)) {
        std::cerr << error << "\n";
        return 2;
    }

    const cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "图片读取失败: " << image_path << "\n";
        return 2;
    }

    const ArmorClassifier::Result result = classifier.infer(image);
    if (result.class_index < 0) {
        std::cerr << "推理失败（输入无效？）\n";
        return 2;
    }

    std::cout << "图片: " << image_path << " (" << image.cols << "x" << image.rows << ")\n"
              << "类别: " << kClassNames[result.class_index]
              << "  target_id: " << result.target_id
              << "  置信度: " << result.confidence << "\n";

    if (!expect.empty()) {
        if (kClassNames[result.class_index] == expect) {
            std::cout << "[PASS] 与期望一致\n";
            return 0;
        }
        std::cout << "[FAIL] 期望 " << expect << "，实际 " << kClassNames[result.class_index]
                  << "\n";
        return 1;
    }
    return 0;
}
