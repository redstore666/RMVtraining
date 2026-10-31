// 单一相位拟合器（RotatingArmorSpinFit）的单元回归。
//
// 这个测试守的是"换掉四路证据融合"这件事的数学底线：
//   1. 匀速自转：ω 解对（误差 < 0.05 rad/s）；
//   2. 静止：ω≈0，不能凭空造出转速；
//   3. **换板**：每帧随机露不同板（相位差 90° 的整数倍），折叠必须内建消化掉，
//      不许出现"续接到别的板 -> 假高速"；
//   4. 变速：线性相位模型有自平滑偏差，二次精化要把它修回来（且只在曲率显著时才启用，
//      匀速目标上不能因此变差）；
//   5. 数据自相矛盾（随机相位）时浓度门限必须拒绝。
//
// 合成方式：观测到的相位 = 车体相位 φ0+ω·t + k·(π/2) + 噪声，k 为当帧可见板号。
// 因为四块板的相位本来就相差 π/2 的整数倍，这与真实观测量在数学上等价。

#include "RotatingArmorSpinFit.hpp"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace
{
constexpr double kPi = 3.141592653589793238462643383279502884;
int failures = 0;

void expect(bool ok, const char *what)
{
    if (!ok)
    {
        std::printf("  [FAIL] %s\n", what);
        ++failures;
    }
}

double wrapPi(double value)
{
    return std::remainder(value, 2.0 * kPi);
}

struct Scenario
{
    double omega0{4.0};      // rad/s（t=0 时刻）
    double accel{0.0};       // rad/s²（线性变速）
    double noise_rad{0.02};
    bool random_plate{true}; // 每帧随机换板
    int samples{40};
    double span_s{0.35};
    int seed{7};
};

std::vector<rotating_armor::SpinSample> makeSamples(const Scenario &scenario)
{
    std::mt19937 rng(static_cast<unsigned>(scenario.seed));
    std::normal_distribution<double> noise(0.0, scenario.noise_rad);
    std::uniform_int_distribution<int> plate(0, 3);
    std::vector<rotating_armor::SpinSample> samples;
    samples.reserve(static_cast<std::size_t>(scenario.samples));
    const double phase0 = 0.7;
    for (int i = 0; i < scenario.samples; ++i)
    {
        const double t = scenario.span_s * static_cast<double>(i) /
                         static_cast<double>(scenario.samples - 1);
        const double body = phase0 + scenario.omega0 * t + 0.5 * scenario.accel * t * t;
        const int slot = scenario.random_plate ? plate(rng) : 0;
        rotating_armor::SpinSample sample;
        sample.timestamp_s = t;
        sample.phase_rad = wrapPi(body + 0.5 * kPi * static_cast<double>(slot) + noise(rng));
        sample.weight = 1.0;
        samples.push_back(sample);
    }
    return samples;
}

rotating_armor::SpinFitConfig baseConfig()
{
    rotating_armor::SpinFitConfig config;
    config.plate_count = 4;
    config.use_prior = false; // 单元测试只看数据本身，不引入先验
    config.sigma_scale = 1.0;
    config.sigma_floor_rad_s = 0.01;
    return config;
}
} // namespace

int main()
{
    using namespace rotating_armor;

    // 1) 匀速 + 换板：ω 必须解对，且不出现假高速。
    {
        Scenario scenario;
        scenario.omega0 = 4.0;
        const auto samples = makeSamples(scenario);
        const SpinFitReport report = fitSpinPhase(samples, baseConfig());
        std::printf("[1] 匀速 4 rad/s 换板: ok=%d w=%+.3f sigma=%.3f C=%.2f res=%.3f\n",
                    static_cast<int>(report.fit.ok), report.fit.omega_rad_s,
                    report.fit.sigma_rad_s, report.fit.concentration,
                    report.fit.residual_rms_rad);
        expect(report.fit.ok, "匀速场景应判有效");
        expect(std::fabs(report.fit.omega_rad_s - 4.0) < 0.05, "匀速 ω 误差应 < 0.05");
        expect(report.fit.concentration > 0.9, "匀速换板的相位集中度应接近 1");
    }

    // 2) 静止：不能造转速。
    {
        Scenario scenario;
        scenario.omega0 = 0.0;
        const auto samples = makeSamples(scenario);
        const SpinFitReport report = fitSpinPhase(samples, baseConfig());
        std::printf("[2] 静止: ok=%d w=%+.3f sigma=%.3f C=%.2f\n", static_cast<int>(report.fit.ok),
                    report.fit.omega_rad_s, report.fit.sigma_rad_s, report.fit.concentration);
        expect(report.fit.ok, "静止场景应判有效");
        expect(std::fabs(report.fit.omega_rad_s) < 0.06, "静止 ω 应接近 0");
    }

    // 3) 高速 8 rad/s + 换板 + 大噪声：仍要咬住。
    {
        Scenario scenario;
        scenario.omega0 = 8.0;
        scenario.noise_rad = 0.06;
        scenario.seed = 11;
        const auto samples = makeSamples(scenario);
        const SpinFitReport report = fitSpinPhase(samples, baseConfig());
        std::printf("[3] 8 rad/s 换板(噪声 3.4°): ok=%d w=%+.3f sigma=%.3f C=%.2f\n",
                    static_cast<int>(report.fit.ok), report.fit.omega_rad_s, report.fit.sigma_rad_s,
                    report.fit.concentration);
        expect(report.fit.ok, "高速场景应判有效");
        expect(std::fabs(report.fit.omega_rad_s - 8.0) < 0.15, "高速 ω 误差应 < 0.15");
    }

    // 4) 正弦变速（sinspeed 素材的形状）：窗口中心落在 ω 的**谷**上时，
    //    线性相位模型给出的是窗口平均 —— 谷被削平。二次精化要么修回来，
    //    要么至少不能更差。真值取窗口中心时刻的瞬时 ω。
    {
        const double amplitude = 3.0;
        const double mean = 5.0;
        const double period = 2.1; // 与 sinspeed 实测主周期一致
        const double omega_sine = 2.0 * kPi / period;
        const double span = 0.45;
        const double t_center = 0.5 * span;      // 窗口中心
        const double truth_center = mean - amplitude; // 谷底

        std::mt19937 rng(5);
        std::normal_distribution<double> noise(0.0, 0.02);
        std::uniform_int_distribution<int> plate(0, 3);
        std::vector<SpinSample> samples;
        for (int i = 0; i < 40; ++i)
        {
            const double t = span * static_cast<double>(i) / 39.0;
            // 让 ω 在窗口中心取到谷底：ω(t) = mean - A·cos(Ω·(t-t_center))
            const double u = t - t_center;
            const double phase = mean * u - (amplitude / omega_sine) * std::sin(omega_sine * u);
            const int slot = plate(rng);
            samples.push_back({t, wrapPi(phase + 0.5 * kPi * static_cast<double>(slot) + noise(rng)),
                               1.0});
        }

        SpinFitConfig linear = baseConfig();
        linear.quadratic_refinement = false;
        const SpinFitReport linear_report = fitSpinPhase(samples, linear);

        SpinFitConfig quadratic = baseConfig();
        quadratic.quadratic_refinement = true;
        quadratic.curvature_significance = 2.0;
        const SpinFitReport quadratic_report = fitSpinPhase(samples, quadratic);

        const double linear_error = std::fabs(linear_report.fit.omega_rad_s - truth_center);
        const double quadratic_error = std::fabs(quadratic_report.fit.omega_rad_s - truth_center);
        std::printf("[4] 正弦谷底: 真值 %+.3f  线性 %+.3f (误差 %.3f)  二次 %+.3f (误差 %.3f)\n",
                    truth_center, linear_report.fit.omega_rad_s, linear_error,
                    quadratic_report.fit.omega_rad_s, quadratic_error);
        expect(linear_error < 0.5, "正弦谷底：线性解的偏差应在 0.5 rad/s 以内");
        expect(quadratic_error <= linear_error + 1.0e-6, "二次精化不应让正弦谷底更差");
    }

    // 5) 匀速目标上二次项必须自动关闭：不能因为"允许二次"就把方差搬进 ω。
    {
        Scenario scenario;
        scenario.omega0 = 5.0;
        scenario.noise_rad = 0.03;
        scenario.seed = 21;
        const auto samples = makeSamples(scenario);
        SpinFitConfig quadratic = baseConfig();
        quadratic.quadratic_refinement = true;
        quadratic.curvature_significance = 2.0;
        const SpinFitReport report = fitSpinPhase(samples, quadratic);
        std::printf("[5] 匀速 5 rad/s（允许二次）: w=%+.3f 误差=%.3f\n", report.fit.omega_rad_s,
                    std::fabs(report.fit.omega_rad_s - 5.0));
        expect(std::fabs(report.fit.omega_rad_s - 5.0) < 0.06, "曲率不显著时应退回线性解");
    }

    // 6) 自相矛盾的数据必须被拒绝。
    {
        std::mt19937 rng(99);
        std::uniform_real_distribution<double> random(-kPi, kPi);
        std::vector<SpinSample> samples;
        for (int i = 0; i < 40; ++i)
        {
            samples.push_back({0.35 * static_cast<double>(i) / 39.0, random(rng), 1.0});
        }
        const SpinFitReport report = fitSpinPhase(samples, baseConfig());
        std::printf("[6] 随机相位: ok=%d status=%d C=%.2f\n", static_cast<int>(report.fit.ok),
                    static_cast<int>(report.status), report.fit.concentration);
        expect(!report.fit.ok, "随机相位应判无效");
        expect(report.status == SpinFitStatus::LowConcentration, "应以浓度不足为由拒绝");
    }

    // 7) 样本不足 / 跨度不足要明确拒绝，而不是给出一个看似合理的数。
    {
        std::vector<SpinSample> few{{0.0, 0.1, 1.0}, {0.1, 0.2, 1.0}};
        const SpinFitReport report = fitSpinPhase(few, baseConfig());
        std::printf("[7] 只有 2 条样本: ok=%d status=%d\n", static_cast<int>(report.fit.ok),
                    static_cast<int>(report.status));
        expect(!report.fit.ok, "样本不足应判无效");
    }

    std::printf("\n%s（%d 项失败）\n", failures == 0 ? "全部通过" : "有失败", failures);
    return failures == 0 ? 0 : 1;
}
