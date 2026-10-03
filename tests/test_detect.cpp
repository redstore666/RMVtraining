// 装甲板识别（armor_detect）离线测试工具。
//
// 用法:
//   ./build/test_detect <图片> [--red|--blue] [--save 输出.png]
//
// 示例:
//   ./build/test_detect tests/data/synth_small_1.png --red
//   ./build/test_detect tests/data/synth_dual_3_7_blue.png --blue --save /tmp/out.png
//
// 直接调用 student/src/StudentTasks.cpp 中的 armor_detect，打印检测结果；
// --save 输出角点叠加图。合成测试图由 tools/gen_synth_armor.py 生成：
//   synth_small_1.png / synth_large_5.png  单目标（红）
//   synth_dual_3_7_blue.png                双目标（蓝）
//   synth_tilted_2.png                     倾斜 12°（红）
//   synth_fake_number.png                  配对成立但中间是噪声数字（应被分类器拒绝）
//   synth_negative.png                     空场景（应无检测）
#include "StudentTasks.hpp"

#include <chrono>
#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

int main(int argc, char **argv) {
    std::string image_path;
    std::string save_path;
    TeamColor enemy = TeamColor::Red;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--red") {
            enemy = TeamColor::Red;
        } else if (arg == "--blue") {
            enemy = TeamColor::Blue;
        } else if (arg == "--save" && i + 1 < argc) {
            save_path = argv[++i];
        } else if (!arg.empty() && arg[0] != '-') {
            image_path = arg;
        } else {
            std::cerr << "未知参数: " << arg << "\n";
            return 2;
        }
    }
    if (image_path.empty()) {
        std::cerr << "用法: " << argv[0] << " <图片> [--red|--blue] [--save 输出.png]\n";
        return 2;
    }

    const cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "图片读取失败: " << image_path << "\n";
        return 2;
    }

    const auto start = std::chrono::steady_clock::now();
    const std::vector<ArmorDetection> detections = armor_detect(image, enemy);
    const double ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - start)
                          .count();

    std::cout << "图片: " << image_path << "  敌方颜色: "
              << (enemy == TeamColor::Red ? "红" : "蓝") << "\n"
              << "检测到 " << detections.size() << " 块装甲板，耗时 " << ms << " ms\n";

    cv::Mat canvas = image.clone();
    for (std::size_t i = 0; i < detections.size(); ++i) {
        const ArmorDetection &d = detections[i];
        std::cout << "  [" << i << "] id=" << d.target_id
                  << " size=" << (d.size == ArmorSize::Large ? "大" : "小")
                  << " conf=" << d.confidence << " corners=";
        for (const cv::Point2f &p : d.corners) {
            std::cout << "(" << p.x << "," << p.y << ") ";
        }
        std::cout << "\n";

        std::vector<cv::Point> poly;
        for (const cv::Point2f &p : d.corners) {
            poly.emplace_back(cv::Point(cvRound(p.x), cvRound(p.y)));
        }
        cv::polylines(canvas, poly, true, cv::Scalar(0, 255, 0), 2);
        cv::circle(canvas, poly[0], 5, cv::Scalar(0, 0, 255), -1);  // 左上角标记
        const cv::Point2f center = (d.corners[0] + d.corners[1] + d.corners[2] + d.corners[3]) * 0.25F;
        cv::putText(canvas, std::to_string(d.target_id) + " " + cv::format("%.2f", d.confidence),
                    cv::Point(cvRound(center.x) - 30, cvRound(center.y)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 255), 2);
    }

    if (!save_path.empty()) {
        cv::imwrite(save_path, canvas);
        std::cout << "叠加图已保存: " << save_path << "\n";
    }
    return 0;
}
