// 标定图像采集工具：实时预览 + 角点覆盖实时指示，
// 自动每 N 秒保存一帧（默认 15s），空格立即补拍（重置计时），p 暂停/继续，q 退出。
// 保存目录 calib_data/（自动创建），文件编号自动递增，不覆盖已有图像。
//
// 用法: ./build/calib_capture [间隔秒数]      # 默认 15
//
// 画面指示:
//   saved: N                 本次会话已保存张数
//   corner coverage r_max=.. 棋盘格角点相对画面中心的径向覆盖（画面四角 = 1.0）。
//                            目标 >= 0.85（变绿）——表示板已推到画面角落附近，
//                            此时畸变的边缘信息才被采到。
#include "StudentTasks.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>

#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "MvCameraControl.h"

namespace {

constexpr int kPatternCols = 9;  // 内角点列数
constexpr int kPatternRows = 6;  // 内角点行数
constexpr double kCoverageTarget = 0.85;  // r_max 目标（与自查工具一致）

struct Coverage {
    bool found = false;  // 是否检出棋盘格
    double r_max = 0.0;  // 角点到画面中心的径向覆盖（画面四角 = 1.0）
};

// 打印相机序列号（标定材料中需要注明"相机编号"）。
void printCameraSerial() {
    MV_CC_DEVICE_INFO_LIST device_list;
    memset(&device_list, 0, sizeof(device_list));
    if (MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, &device_list) != MV_OK ||
        device_list.nDeviceNum == 0) {
        std::cout << "未枚举到相机设备（序列号未打印）\n";
        return;
    }
    for (unsigned int i = 0; i < device_list.nDeviceNum; ++i) {
        const MV_CC_DEVICE_INFO* info = device_list.pDeviceInfo[i];
        if (info->nTLayerType == MV_USB_DEVICE) {
            std::cout << "相机 " << i << " 序列号: "
                      << reinterpret_cast<const char*>(
                             info->SpecialInfo.stUsb3VInfo.chSerialNumber)
                      << "\n";
        } else if (info->nTLayerType == MV_GIGE_DEVICE) {
            std::cout << "相机 " << i << " 序列号: "
                      << reinterpret_cast<const char*>(
                             info->SpecialInfo.stGigEInfo.chSerialNumber)
                      << "\n";
        }
    }
}

// 扫描目录，返回下一个可用的文件编号（避免覆盖之前的采集结果）。
int nextFrameIndex(const std::filesystem::path& dir) {
    int max_index = -1;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        const std::string stem = entry.path().stem().string();  // 形如 frame_07
        const auto pos = stem.find_last_of('_');
        if (pos == std::string::npos) {
            continue;
        }
        try {
            max_index = std::max(max_index, std::stoi(stem.substr(pos + 1)));
        } catch (...) {
        }
    }
    return max_index + 1;
}

bool saveFrame(const cv::Mat& pic, const std::filesystem::path& dir,
               int& index, int& saved) {
    char name[64];
    std::snprintf(name, sizeof(name), "frame_%02d.png", index);
    const auto path = dir / name;
    if (!cv::imwrite(path.string(), pic)) {
        std::cout << "保存失败: " << path.string() << "\n";
        return false;
    }
    std::cout << "已保存 " << path.string() << "（本次会话第 " << ++saved << " 张）\n";
    ++index;
    return true;
}

// 半分辨率快速检测（仅用于实时指示，不做亚像素精化）。
Coverage measureCoverage(const cv::Mat& bgr) {
    cv::Mat small;
    cv::resize(bgr, small, cv::Size(), 0.5, 0.5, cv::INTER_AREA);
    cv::Mat gray;
    cv::cvtColor(small, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point2f> corners;
    const bool ok = cv::findChessboardCorners(
        gray, cv::Size(kPatternCols, kPatternRows), corners,
        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    Coverage cov;
    if (!ok) {
        return cov;
    }
    cov.found = true;
    const cv::Point2f center(small.cols / 2.0F, small.rows / 2.0F);
    const double half_diag = std::hypot(small.cols / 2.0, small.rows / 2.0);
    for (const auto& p : corners) {
        cov.r_max = std::max(cov.r_max, cv::norm(p - center) / half_diag);
    }
    return cov;
}

// 在预览画面上绘制倒计时进度条、覆盖指示与状态文字（ASCII，避免中文字体缺失）。
void drawOverlay(cv::Mat& view, double remain_s, double interval_s, int saved,
                 bool paused, bool flashing, const Coverage& cov) {
    const int bar_w = view.cols / 2;
    const int bar_x = (view.cols - bar_w) / 2;
    const int bar_y = 40;
    const int bar_h = 18;
    cv::rectangle(view, {bar_x, bar_y}, {bar_x + bar_w, bar_y + bar_h},
                  {255, 255, 255}, 2);
    if (!paused) {
        const double frac = std::min(1.0, std::max(0.0, 1.0 - remain_s / interval_s));
        cv::rectangle(view, {bar_x, bar_y},
                      {bar_x + static_cast<int>(bar_w * frac), bar_y + bar_h},
                      {0, 255, 0}, cv::FILLED);
    }

    char buf[128];
    if (paused) {
        cv::putText(view, "PAUSED - press p to resume", {bar_x, bar_y + 90},
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, {0, 165, 255}, 2);
    } else {
        std::snprintf(buf, sizeof(buf), "%.0f s", std::max(0.0, remain_s));
        cv::putText(view, buf, {view.cols / 2 - 55, bar_y + 105},
                    cv::FONT_HERSHEY_SIMPLEX, 2.0, {0, 255, 255}, 4);
    }
    std::snprintf(buf, sizeof(buf), "saved: %d", saved);
    cv::putText(view, buf, {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 1.0,
                {0, 255, 0}, 2);

    // 角点覆盖实时指示（引导把板推到画面四角）
    if (!cov.found) {
        cv::putText(view, "board: NOT FOUND - keep the whole board in view",
                    {20, 84}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 0, 255}, 2);
    } else {
        const bool ok = cov.r_max >= kCoverageTarget;
        const cv::Scalar color =
            ok ? cv::Scalar(0, 255, 0)
               : (cov.r_max >= 0.60 ? cv::Scalar(0, 215, 255) : cv::Scalar(0, 0, 255));
        std::snprintf(buf, sizeof(buf), "corner coverage r_max=%.2f %s", cov.r_max,
                      ok ? "(OK - hold still)" : "(push toward image corners)");
        cv::putText(view, buf, {20, 84}, cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
    }

    if (flashing) {
        cv::putText(view, "SAVED", {view.cols / 2 - 90, view.rows - 60},
                    cv::FONT_HERSHEY_SIMPLEX, 1.6, {0, 255, 0}, 3);
    }
}

}  // namespace

int main(int argc, char** argv) {
    double interval_s = 15.0;
    if (argc > 1) {
        try {
            interval_s = std::max(1.0, std::stod(argv[1]));
        } catch (...) {
        }
    }

    const std::filesystem::path out_dir{"calib_data"};
    std::filesystem::create_directories(out_dir);

    printCameraSerial();
    std::cout << "操作: 每 " << interval_s << " 秒自动拍摄；空格=立即补拍(重置计时)；"
              << "p=暂停/继续；q=退出\n"
              << "覆盖指示: 左上角实时显示 r_max —— 板推到画面角落附近时 >= 0.85 变绿，"
              << "那时保持不动等自动拍摄。\n"
              << "提示: 板要全程完整入画（四角留白）。\n"
              << "保存目录: " << std::filesystem::absolute(out_dir) << "\n";

    int saved = 0;
    int index = nextFrameIndex(out_dir);
    cv::namedWindow("calib_capture", cv::WINDOW_NORMAL);
    cv::Mat pic;
    bool paused = false;
    auto next_capture =
        std::chrono::steady_clock::now() + std::chrono::duration<double>(interval_s);
    auto flash_until = std::chrono::steady_clock::time_point{};

    while (true) {
        if (!get_pic(pic) || pic.empty()) {
            std::cout << "取流失败，重试...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        const auto now = std::chrono::steady_clock::now();
        const double remain_s =
            std::chrono::duration<double>(next_capture - now).count();
        const bool flashing = now < flash_until;
        const Coverage cov = measureCoverage(pic);

        cv::Mat view = pic.clone();
        drawOverlay(view, remain_s, interval_s, saved, paused, flashing, cov);
        cv::imshow("calib_capture", view);

        const int key = cv::waitKey(30) & 0xFF;
        if (key == 'q') {
            break;
        }
        if (key == 'p') {
            paused = !paused;
            if (!paused) {
                next_capture = now + std::chrono::duration<double>(interval_s);
            }
            std::cout << (paused ? "已暂停自动拍摄" : "继续自动拍摄") << "\n";
            continue;
        }
        if (key == ' ') {
            if (saveFrame(pic, out_dir, index, saved)) {
                flash_until = now + std::chrono::milliseconds(600);
            }
            next_capture = now + std::chrono::duration<double>(interval_s);
            continue;
        }
        if (!paused && now >= next_capture) {
            if (saveFrame(pic, out_dir, index, saved)) {
                flash_until = now + std::chrono::milliseconds(600);
            }
            next_capture = now + std::chrono::duration<double>(interval_s);
        }
    }
    std::cout << "退出，本次会话共保存 " << saved << " 张\n";
    return 0;
}
