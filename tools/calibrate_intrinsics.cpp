// 相机内参标定脚本（骨架）
//
// 功能：读取 calib_data/ 下的棋盘格图像，检测内角点，解算相机内参矩阵、
//       畸变系数与重投影误差，并把结果保存为 YAML（字段名与 config/camera.yaml
//       一致，方便后续填写）。
//
// 用法：
//   ./calibrate_intrinsics [图像目录] [输出yaml路径]
//   默认： ./calibrate_intrinsics calib_data tools/intrinsics_result.yaml
//
// 棋盘格约定（与 tools/checkerboard_a4.pdf 一致）：
//   9x6 内角点，方格边长 20 mm。如果你打印后实测边长不是 20mm，改下面的
//   kSquareSizeMm 常量（打印缩放会导致标称值失真，务必实测）。
//
// 编译后直接运行；若检测不到角点或标定结果异常，脚本会给出提示。

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace
{

    constexpr int kPatternCols = 7; // 内角点列数（8×5 方格 → 7×4 内角点）
    constexpr int kPatternRows = 4; // 内角点行数（纵向格子数 - 1）
    // 方格实测边长（mm）。新标定板：8×5 方格，边长 28.5mm。
    // 注意：边长误差会 1:1 传递到位移尺度（深度），提交前用直尺量 4 格总宽复核
    // （应 ≈114mm）。
    constexpr double kSquareSizeMm = 28.5;

    // 收集目录下所有图像文件（按文件名排序）。
    std::vector<std::string> listImages(const std::filesystem::path &dir)
    {
        std::vector<std::string> files;
        if (!std::filesystem::exists(dir))
        {
            return files;
        }
        for (const auto &entry : std::filesystem::directory_iterator(dir))
        {
            const auto ext = entry.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp")
            {
                files.push_back(entry.path().string());
            }
        }
        std::sort(files.begin(), files.end());
        return files;
    }

} // namespace

int main(int argc, char **argv)
{
    const std::string image_dir = argc > 1 ? argv[1] : "calib_data";
    const std::string out_yaml = argc > 2 ? argv[2] : "tools/intrinsics_result.yaml";

    const std::vector<std::string> files = listImages(image_dir);
    if (files.empty())
    {
        std::cerr << "目录中没有图像: " << image_dir
                  << "\n先用 ./build/calib_capture 采集标定图像。\n";
        return 1;
    }
    std::cout << "共找到 " << files.size() << " 张图像\n";

    // 棋盘格内角点在"棋盘坐标系"下的三维坐标：z=0 平面，按方格边长排列。
    // 顺序与 findChessboardCorners 的输出一致（先逐行，从左到右）。
    std::vector<cv::Point3f> object_points;
    for (int row = 0; row < kPatternRows; ++row)
    {
        for (int col = 0; col < kPatternCols; ++col)
        {
            object_points.emplace_back(static_cast<float>(col * kSquareSizeMm),
                                       static_cast<float>(row * kSquareSizeMm),
                                       0.0F);
        }
    }

    std::vector<std::vector<cv::Point3f>> all_object_points;
    std::vector<std::vector<cv::Point2f>> all_image_points;
    std::vector<std::string> valid_files; // 与上两个数组一一对应，用于按文件名输出逐视图误差
    cv::Size image_size{};
    const cv::Size pattern_size(kPatternCols, kPatternRows);

    std::filesystem::create_directories("calib_debug");

    for (const std::string &file : files)
    {
        cv::Mat img = cv::imread(file);
        if (img.empty())
        {
            std::cerr << "读取失败，跳过: " << file << "\n";
            continue;
        }
        image_size = img.size();
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;

        // ==================== TODO 1 ====================
        // 目标：在 gray 中检测棋盘格内角点，写入 corners。
        bool found = cv::findChessboardCorners(gray, pattern_size, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        // ==========================================================

        if (!found)
        {
            std::cout << "[跳过] 未检测到角点: " << file << "\n";
            continue;
        }

        // ==================== TODO 2 ====================
        // 目标：把 corners 从像素级精度精化到亚像素级（显著影响标定质量）。
        // 在这里调用 cornerSubPix（对 corners 原地精化）：
        cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
        all_object_points.push_back(object_points);
        all_image_points.push_back(corners);
        valid_files.push_back(file);
        // 保存带角点标记的调试图，便于人工核查检测质量。
        cv::Mat debug = img.clone();
        cv::drawChessboardCorners(debug, pattern_size, corners, found);
        const std::filesystem::path debug_path =
            std::filesystem::path("calib_debug") /
            (std::filesystem::path(file).stem().string() + "_corners.png");
        cv::imwrite(debug_path.string(), debug);
        std::cout << "[成功] " << file << " -> " << debug_path.string() << "\n";
    }

    if (all_image_points.empty())
    {
        std::cerr << "\n没有任何图像检测到棋盘格角点。\n"
                  << "排查建议：\n"
                  << "  1) TODO 1 是否已完成（findChessboardCorners 调用）？\n"
                  << "  2) 棋盘格是否完整出现在画面中（四周留白）？\n"
                  << "  3) 图像是否太暗/过曝/模糊？\n"
                  << "  4) 内角点数量约定 (9x6) 是否与实际棋盘一致？\n";
        return 1;
    }

    // ==================== TODO 3 ====================
    // 目标：用所有视图的 3D-2D 对应点解算相机内参和畸变系数。
    // 声明输出变量并调用 calibrateCamera，把返回值赋给 rms：
    cv::Mat camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat dist_coeffs = cv::Mat::zeros(1, 5, CV_64F);
    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    // 畸变模型选择：本批数据棋盘格始终位于画面中部（四角未覆盖），k2/k3 与切向畸变
    // 不可靠（自由估计会出现 k3≈-3.8 的过拟合值）。固定 k2=k3=0、切向为 0，
    // 仅估计一阶径向畸变 k1 —— 稳健且物理合理（对标 6mm 镜头）。
    constexpr int kDistortionFlags =
        cv::CALIB_FIX_K2 | cv::CALIB_FIX_K3 | cv::CALIB_ZERO_TANGENT_DIST;
    double rms = cv::calibrateCamera(all_object_points, all_image_points, image_size,
                                     camera_matrix, dist_coeffs, rvecs, tvecs,
                                     kDistortionFlags);
    // ==========================================================

    if (rms < 0.0)
    {
        std::cerr << "\nTODO 3 未完成：请调用 cv::calibrateCamera 并给 rms 赋值。\n";
        return 1;
    }

    // ---------------- 以下由框架提供：结果打印与保存 ----------------
    std::cout << "\n===== 标定结果 =====\n"
              << "图像尺寸: " << image_size.width << "x" << image_size.height << "\n"
              << "有效视图: " << all_image_points.size() << " / " << files.size() << "\n"
              << std::fixed << std::setprecision(6)
              << "内参矩阵 K:\n"
              << camera_matrix << "\n"
              << "畸变系数 [k1 k2 p1 p2 k3]:\n"
              << dist_coeffs << "\n"
              << std::setprecision(4) << "RMS 重投影误差: " << rms << " px\n";

    if (rms > 1.0)
    {
        std::cout << "\n[提醒] RMS 偏大（> 1px）。常见原因：角点检测不准、\n"
                  << "运动模糊、棋盘格不平整、方格边长填错。建议重拍。\n";
    }
    else if (rms <= 0.5)
    {
        std::cout << "\n[评价] RMS < 0.5px，标定质量良好。\n";
    }

    // 逐视图重投影误差：定位是哪几张图在拖后腿（个别视图误差明显偏大时应重拍该位姿）。
    std::cout << "\n逐视图重投影误差:\n";
    for (std::size_t i = 0; i < all_image_points.size(); ++i)
    {
        std::vector<cv::Point2f> projected;
        cv::projectPoints(all_object_points[i], rvecs[i], tvecs[i],
                          camera_matrix, dist_coeffs, projected);
        double err_sum = 0.0;
        double err_max = 0.0;
        for (std::size_t j = 0; j < projected.size(); ++j)
        {
            const double err = cv::norm(projected[j] - all_image_points[i][j]);
            err_sum += err;
            if (err > err_max)
            {
                err_max = err;
            }
        }
        std::cout << "  " << valid_files[i] << ": 平均 " << std::setprecision(3)
                  << err_sum / static_cast<double>(projected.size())
                  << " px, 最大 " << err_max << " px\n";
    }

    cv::FileStorage fs(out_yaml, cv::FileStorage::WRITE);
    if (!fs.isOpened())
    {
        std::cerr << "无法写入: " << out_yaml << "\n";
        return 1;
    }
    fs << "image_width" << image_size.width;
    fs << "image_height" << image_size.height;
    fs << "camera_matrix" << camera_matrix;
    fs << "distortion_coefficients" << dist_coeffs;
    fs << "mean_reprojection_error_px" << rms;
    fs.release();
    std::cout << "已保存: " << out_yaml << "\n"
              << "下一步：把结果填入 config/camera.yaml（外参标定后再填旋转/平移）。\n";
    return 0;
}
