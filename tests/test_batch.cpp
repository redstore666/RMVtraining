// 实拍批量评估工具：对目录内所有图片跑 armor_detect，输出逐图结果与汇总检出率。
//
// 用法:
//   ./build/test_batch <目录> --red|--blue [--save 输出目录]
// 示例:
//   ./build/test_batch tmp/red-test --red
//   ./build/test_batch tmp/blue-test --blue --save /tmp/batch_out
//
// 用途：实拍调参的量化工具（before/after 检出率对照、回归检查）。
// 从仓库根目录运行（分类器模型按相对路径 models/armor_cls_64.onnx 加载）。
#include "StudentTasks.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

const char *const kClassNames[] = {"1", "2",       "3",      "4",     "5",
                                   "6outpost", "7guard", "8base", "9neg"};

std::vector<std::string> listImages(const std::filesystem::path &dir) {
    std::vector<std::string> files;
    if (!std::filesystem::exists(dir)) {
        return files;
    }
    for (const auto &entry : std::filesystem::directory_iterator(dir)) {
        const auto ext = entry.path().extension().string();
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

cv::Mat drawDetections(const cv::Mat &image, const std::vector<ArmorDetection> &dets) {
    cv::Mat canvas = image.clone();
    for (const ArmorDetection &d : dets) {
        std::vector<cv::Point> poly;
        for (const cv::Point2f &p : d.corners) {
            poly.emplace_back(cvRound(p.x), cvRound(p.y));
        }
        cv::polylines(canvas, poly, true, cv::Scalar(0, 255, 0), 2);
        cv::circle(canvas, poly[0], 5, cv::Scalar(0, 0, 255), -1);
        const cv::Point2f center =
            (d.corners[0] + d.corners[1] + d.corners[2] + d.corners[3]) * 0.25F;
        cv::putText(canvas,
                    std::to_string(d.target_id) + " " + cv::format("%.2f", d.confidence),
                    cv::Point(cvRound(center.x) - 30, cvRound(center.y)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 255), 2);
    }
    return canvas;
}

}  // namespace

int main(int argc, char **argv) {
    std::string dir;
    std::string save_dir;
    TeamColor enemy = TeamColor::Red;
    bool enemy_set = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--red") {
            enemy = TeamColor::Red;
            enemy_set = true;
        } else if (arg == "--blue") {
            enemy = TeamColor::Blue;
            enemy_set = true;
        } else if (arg == "--save" && i + 1 < argc) {
            save_dir = argv[++i];
        } else if (!arg.empty() && arg[0] != '-') {
            dir = arg;
        } else {
            std::cerr << "未知参数: " << arg << "\n";
            return 2;
        }
    }
    if (dir.empty() || !enemy_set) {
        std::cerr << "用法: " << argv[0] << " <目录> --red|--blue [--save 输出目录]\n";
        return 2;
    }

    const std::vector<std::string> files = listImages(dir);
    if (files.empty()) {
        std::cerr << "目录内没有图片: " << dir << "\n";
        return 2;
    }
    if (!save_dir.empty()) {
        std::filesystem::create_directories(save_dir);
    }

    int hit = 0;
    double total_ms = 0.0;
    for (const std::string &file : files) {
        const cv::Mat image = cv::imread(file);
        if (image.empty()) {
            std::cout << "[跳过] 读图失败: " << file << "\n";
            continue;
        }
        const auto start = std::chrono::steady_clock::now();
        const std::vector<ArmorDetection> dets = armor_detect(image, enemy);
        const double ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - start)
                              .count();
        total_ms += ms;

        const std::string name = std::filesystem::path(file).filename().string();
        std::cout << (dets.empty() ? "[ x ] " : "[ √ ] ") << name << "  det=" << dets.size();
        for (const ArmorDetection &d : dets) {
            const char *cls = (d.target_id >= 1 && d.target_id <= 8)
                                  ? kClassNames[d.target_id - 1]
                                  : "?";
            std::cout << "  id=" << d.target_id << "(" << cls << ")"
                      << (d.size == ArmorSize::Large ? " 大" : " 小") << " conf="
                      << cv::format("%.3f", d.confidence);
        }
        std::cout << cv::format("  %.1fms", ms) << "\n";
        if (!dets.empty()) {
            ++hit;
        }
        if (!save_dir.empty()) {
            cv::imwrite((std::filesystem::path(save_dir) / name).string(),
                        drawDetections(image, dets));
        }
    }

    std::cout << "----------\n"
              << "检出率: " << hit << "/" << files.size() << " ("
              << cv::format("%.1f%%", 100.0 * hit / files.size()) << ")  平均耗时 "
              << cv::format("%.1fms", total_ms / files.size()) << "\n";
    return 0;
}
