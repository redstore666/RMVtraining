// 检测耗时基准：把 detect() 的十几毫秒拆成 预处理 / 推理 / 解码 三段，
// 并横向对比模型尺寸、OpenVINO 线程数与性能提示的差异。
//
// 用法:
//   ./build/bench_detect [图片] [重复次数]
// 默认图片 tests/data/synth_small_1.png（1440x1080，与实拍同尺寸），
// 默认重复 50 次（先热身 10 次，不计入统计）。
//
// 目的：优化 detect 时间时必须先知道时间花在哪；否则容易去抠 1ms 的
// resize，而真正的 12ms 在推理里。
#include "ArmorDetectorNet.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

namespace
{
using Clock = std::chrono::steady_clock;

double toMs(Clock::duration duration)
{
    return std::chrono::duration<double, std::milli>(duration).count();
}

double percentile(std::vector<double> values, double ratio)
{
    if (values.empty())
    {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(ratio * (values.size() - 1));
    return values[index];
}

// 与 ArmorDetectorNet::load 完全一致地把预处理融合进模型图：
// u8 BGR NHWC 直投，模型内转 RGB + /255。
std::shared_ptr<ov::Model> buildPreprocessedModel(const std::string &xml_path, int width,
                                                 int height)
{
    std::shared_ptr<ov::Model> model = ov::Core{}.read_model(xml_path);
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor()
        .set_element_type(ov::element::u8)
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR)
        .set_shape({1, static_cast<std::int64_t>(height), static_cast<std::int64_t>(width), 3});
    ppp.input().model().set_layout("NCHW");
    ppp.input().preprocess()
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .convert_element_type(ov::element::f32)
        .scale(255.0F);
    return ppp.build();
}

struct InferRow
{
    std::string label;
    double min_ms{0.0};
    double median_ms{0.0};
    double p90_ms{0.0};
};

double minOf(const InferRow &row)
{
    return row.min_ms;
}

// 只测推理：同一个 letterbox 画布反复 infer，排除 resize/解码的干扰。
InferRow benchmarkInfer(ov::Core &core, const std::string &xml_path, int width, int height,
                        ov::hint::PerformanceMode hint, int threads, int repeats)
{
    InferRow row;
    row.label = (hint == ov::hint::PerformanceMode::LATENCY ? "LATENCY" : "THROUGHPUT");
    row.label += "/threads=" + (threads > 0 ? std::to_string(threads) : std::string("auto"));

    ov::AnyMap properties{ov::hint::performance_mode(hint)};
    if (threads > 0)
    {
        properties.emplace(ov::inference_num_threads(threads));
    }
    ov::CompiledModel compiled =
        core.compile_model(buildPreprocessedModel(xml_path, width, height), "CPU", properties);
    ov::InferRequest request = compiled.create_infer_request();

    cv::Mat canvas(height, width, CV_8UC3, cv::Scalar(30, 30, 30));
    ov::Tensor input = request.get_input_tensor();
    std::memcpy(input.data(), canvas.data, canvas.total() * canvas.elemSize());

    for (int i = 0; i < 5; ++i) // 热身
    {
        request.infer();
    }

    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i)
    {
        const auto start = Clock::now();
        request.infer();
        samples.push_back(toMs(Clock::now() - start));
    }
    row.min_ms = percentile(samples, 0.0);
    row.median_ms = percentile(samples, 0.5);
    row.p90_ms = percentile(samples, 0.9);
    return row;
}

struct ModelCase
{
    std::string label;
    std::string xml;
    int width;
    int height;
    ArmorDetectorNet::ModelKind kind;
};

} // namespace

int main(int argc, char **argv)
{
    const std::string image_path = argc > 1 ? argv[1] : "tests/data/synth_small_1.png";
    const int repeats = argc > 2 ? std::atoi(argv[2]) : 50;
    if (repeats < 5)
    {
        std::fprintf(stderr, "重复次数至少 5\n");
        return 2;
    }

    // OpenCV 的并行后端（cv::mean / cv::resize）会自己开满线程，和 OpenVINO 的
    // 线程池抢核。基准里统一关掉，避免把"两个线程池打架"算进单段耗时。
    cv::setNumThreads(1);

    const cv::Mat frame = cv::imread(image_path);
    if (frame.empty())
    {
        std::fprintf(stderr, "读不到图片: %s\n", image_path.c_str());
        return 2;
    }
    std::printf("图片: %s  %dx%d  重复 %d 次（热身 10 次）\n\n", image_path.c_str(), frame.cols,
                frame.rows, repeats);

    // 640x480 是生产默认（相机 4:3，640x640 的 160 行黑边是白算的）
    const std::vector<ModelCase> cases{
        {"yolov5@640x480", "models/yolov5_640x480.xml", 640, 480,
         ArmorDetectorNet::ModelKind::YOLOv5},
        {"yolov5@640x640", "models/yolov5.xml", 640, 640, ArmorDetectorNet::ModelKind::YOLOv5},
        {"yolov5@512x512", "models/yolov5_512.xml", 512, 512,
         ArmorDetectorNet::ModelKind::YOLOv5},
        {"yolo11@640x640", "models/yolo11.xml", 640, 640, ArmorDetectorNet::ModelKind::YOLO11},
    };
    const std::string sweep_xml = "models/yolov5_640x480.xml";
    const int sweep_width = 640;
    const int sweep_height = 480;

    // ---- 一、生产路径的端到端耗时（detect() 全流程）----
    std::printf("== 一、detect() 端到端（生产路径）==\n");
    std::printf("%-24s %8s %8s %10s %8s %8s\n", "模型", "最小ms", "中位ms", "p90ms", "检出数",
                "推理ms");
    ov::Core core;
    for (const ModelCase &model_case : cases)
    {
        ArmorDetectorNet detector;
        std::string error;
        if (!detector.load(model_case.xml, model_case.kind, model_case.width,
                           model_case.height, "CPU", error))
        {
            std::printf("%-24s 加载失败: %s\n", model_case.xml.c_str(), error.substr(0, 60).c_str());
            continue;
        }
        for (int i = 0; i < 10; ++i)
        {
            detector.detect(frame);
        }
        std::vector<double> samples;
        std::size_t detections = 0;
        for (int i = 0; i < repeats; ++i)
        {
            const auto start = Clock::now();
            const auto result = detector.detect(frame);
            samples.push_back(toMs(Clock::now() - start));
            detections = result.size();
        }
        // 同一模型只测推理一段，用于把端到端拆成三段
        const InferRow infer = benchmarkInfer(core, model_case.xml, model_case.width,
                                              model_case.height,
                                              ov::hint::PerformanceMode::LATENCY, 0, repeats);
        std::printf("%-24s %8.2f %8.2f %10.2f %8zu %8.2f\n", model_case.label.c_str(),
                    percentile(samples, 0.0), percentile(samples, 0.5),
                    percentile(samples, 0.9), detections, infer.median_ms);
    }

    // ---- 二、预处理各段（1440x1080 原图，640 画布）----
    std::printf("\n== 二、预处理各段（640 输入）==\n");
    {
        const int size = 640;
        const double scale = std::min(static_cast<double>(size) / frame.rows,
                                      static_cast<double>(size) / frame.cols);
        cv::Mat canvas(size, size, CV_8UC3);
        std::vector<double> mean_ms, clear_ms, resize_ms, copy_ms;
        for (int i = 0; i < repeats; ++i)
        {
            auto start = Clock::now();
            const cv::Scalar mean = cv::mean(frame);
            mean_ms.push_back(toMs(Clock::now() - start));
            (void)mean;

            start = Clock::now();
            canvas.setTo(cv::Scalar(0, 0, 0));
            clear_ms.push_back(toMs(Clock::now() - start));

            const int w = static_cast<int>(frame.cols * scale);
            const int h = static_cast<int>(frame.rows * scale);
            start = Clock::now();
            cv::resize(frame, canvas(cv::Rect(0, 0, w, h)), cv::Size(w, h));
            resize_ms.push_back(toMs(Clock::now() - start));

            start = Clock::now();
            std::memcpy(canvas.data, canvas.data, canvas.total() * canvas.elemSize());
            copy_ms.push_back(toMs(Clock::now() - start));
        }
        std::printf("%-24s %10.3f\n", "黑帧检查 cv::mean", percentile(mean_ms, 0.5));
        std::printf("%-24s %10.3f\n", "画布清零 setTo", percentile(clear_ms, 0.5));
        std::printf("%-24s %10.3f\n", "cv::resize 1440->640", percentile(resize_ms, 0.5));
        std::printf("%-24s %10.3f\n", "memcpy 640x640x3", percentile(copy_ms, 0.5));
    }

    // ---- 三、推理线程数与性能提示 ----
    std::printf("\n== 三、推理配置对比（yolov5@640x480）==\n");
    std::printf("%-24s %8s %8s %8s\n", "配置", "最小ms", "中位ms", "p90ms");
    for (const auto &config :
         std::vector<std::pair<ov::hint::PerformanceMode, int>>{
             {ov::hint::PerformanceMode::LATENCY, 0}, {ov::hint::PerformanceMode::LATENCY, 2},
             {ov::hint::PerformanceMode::LATENCY, 4}, {ov::hint::PerformanceMode::LATENCY, 8},
             {ov::hint::PerformanceMode::THROUGHPUT, 0}})
    {
        const InferRow row =
            benchmarkInfer(core, sweep_xml, sweep_width, sweep_height, config.first,
                           config.second, repeats);
        std::printf("%-24s %8.2f %8.2f %8.2f\n", row.label.c_str(), minOf(row), row.median_ms,
                    row.p90_ms);
    }

    // ---- 四、AB 交替对比（机器有负载时唯一可信的比法）----
    // 两个模型在同一进程里轮流推理，负载漂移对两者影响相同，比值可用。
    std::printf("\n== 四、AB 交替（同进程轮流推理，抵消负载漂移）==\n");
    {
        struct AbCase
        {
            const char *label;
            const char *xml;
            int w;
            int h;
        };
        const AbCase ab[3] = {{"yolov5@640x480", "models/yolov5_640x480.xml", 640, 480},
                              {"yolov5@512x384", "models/yolov5_512x384.xml", 512, 384},
                              {"yolov5@640x640", "models/yolov5.xml", 640, 640}};
        const int ab_count = 3;
        ov::InferRequest request[ab_count];
        bool ok[ab_count] = {true, true, true};
        for (int i = 0; i < ab_count; ++i)
        {
            try
            {
                ov::CompiledModel compiled = core.compile_model(
                    buildPreprocessedModel(ab[i].xml, ab[i].w, ab[i].h), "CPU",
                    ov::AnyMap{ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)});
                request[i] = compiled.create_infer_request();
                cv::Mat canvas(ab[i].h, ab[i].w, CV_8UC3, cv::Scalar(30, 30, 30));
                std::memcpy(request[i].get_input_tensor().data(), canvas.data,
                            canvas.total() * canvas.elemSize());
                request[i].infer();
            }
            catch (const std::exception &error)
            {
                std::printf("%s 加载失败: %s\n", ab[i].label, error.what());
                ok[i] = false;
            }
        }
        bool all_ok = true;
        for (int i = 0; i < ab_count; ++i)
        {
            all_ok = all_ok && ok[i];
        }
        if (all_ok)
        {
            std::vector<double> samples[ab_count];
            for (int r = 0; r < repeats; ++r)
            {
                for (int i = 0; i < ab_count; ++i)
                {
                    const auto begin = std::chrono::steady_clock::now();
                    request[i].infer();
                    const auto end = std::chrono::steady_clock::now();
                    samples[i].push_back(
                        std::chrono::duration<double, std::milli>(end - begin).count());
                }
            }
            std::printf("%-18s %8s %8s %8s %8s\n", "配置", "最小ms", "p10ms", "中位ms",
                        "p90ms");
            for (int i = 0; i < ab_count; ++i)
            {
                std::printf("%-18s %8.2f %8.2f %8.2f %8.2f\n", ab[i].label,
                            percentile(samples[i], 0.0), percentile(samples[i], 0.1),
                            percentile(samples[i], 0.5), percentile(samples[i], 0.9));
            }
            const double base = percentile(samples[ab_count - 1], 0.0);
            for (int i = 0; i < ab_count - 1; ++i)
            {
                std::printf("%s / 640x640 最小耗时比 = %.3f（像素面积比 %.3f）\n",
                            ab[i].label,
                            percentile(samples[i], 0.0) / base,
                            static_cast<double>(ab[i].w) * ab[i].h / (640.0 * 640.0));
            }
        }
    }

    return 0;
}
