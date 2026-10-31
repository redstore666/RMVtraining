#include "RotatingArmorSpinFit.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>

namespace rotating_armor
{
namespace
{
constexpr double kPi = 3.141592653589793238462643383279502884;

// 折叠到 (-period/2, +period/2]。板数 N 的刚体上，相位相差 2π/N 的两条
// 观测**描述的是同一块物理板**，所以残差必须按 2π/N 折叠后再算。
double fold(double value, double period)
{
    return std::remainder(value, period);
}

// 相位集中度 C(ω) = |Σ w·e^{jN(θ-ω(t-t̄))}| / Σw ∈ [0,1]，
// 顺带把该 ω 下的 φ0 = arg(z)/N 写进 phase_out。
//
// 为什么要减 t̄：θ 与 ω·t 都是大数相减，直接算 exp(j·N·(θ-ωt)) 时相位可达
// 数百弧度，双精度下仍然可用，但减掉中心时刻能显著改善精化段的数值条件。
double concentration(const std::vector<SpinSample> &samples, double omega, double tbar,
                     int plates, double *phase_out)
{
    double re = 0.0;
    double im = 0.0;
    double weight_sum = 0.0;
    for (const SpinSample &sample : samples)
    {
        const double angle = static_cast<double>(plates) *
                             (sample.phase_rad - omega * (sample.timestamp_s - tbar));
        re += sample.weight * std::cos(angle);
        im += sample.weight * std::sin(angle);
        weight_sum += sample.weight;
    }
    if (!(weight_sum > 0.0))
    {
        return 0.0;
    }
    if (phase_out != nullptr)
    {
        *phase_out = std::atan2(im, re) / static_cast<double>(plates);
    }
    return std::hypot(re, im) / weight_sum;
}

// 打分函数：纯相位集中度。**先验不参与打分**，见头文件里"只做平局裁决"那段：
// 把先验塞进目标函数会让估计器自锁（偏了就再也回不来）。
double score(const std::vector<SpinSample> &samples, double omega, double tbar, int plates)
{
    return concentration(samples, omega, tbar, plates, nullptr);
}
} // namespace

SpinFitReport fitSpinPhase(const std::vector<SpinSample> &samples, const SpinFitConfig &config)
{
    SpinFitReport report;
    report.status = SpinFitStatus::Ok;
    if (config.plate_count <= 0 || !(config.max_abs_rad_s > 0.0) ||
        !(config.grid_step_rad_s > 0.0))
    {
        report.status = SpinFitStatus::BadConfig;
        return report;
    }
    const int plates = config.plate_count;
    const double fold_period = 2.0 * kPi / static_cast<double>(plates);

    // 1) 样本数与时间跨度：两条都要够，否则 ω 与 φ0 根本不可分。
    if (static_cast<int>(samples.size()) < config.min_samples)
    {
        report.status = SpinFitStatus::TooFewSamples;
        return report;
    }
    double weight_sum = 0.0;
    double weight_time_sum = 0.0;
    double t_min = std::numeric_limits<double>::infinity();
    double t_max = -std::numeric_limits<double>::infinity();
    for (const SpinSample &sample : samples)
    {
        if (!std::isfinite(sample.timestamp_s) || !std::isfinite(sample.phase_rad) ||
            !std::isfinite(sample.weight) || !(sample.weight > 0.0))
        {
            continue;
        }
        weight_sum += sample.weight;
        weight_time_sum += sample.weight * sample.timestamp_s;
        t_min = std::min(t_min, sample.timestamp_s);
        t_max = std::max(t_max, sample.timestamp_s);
    }
    if (!(weight_sum > 0.0) || !std::isfinite(t_min) || !std::isfinite(t_max))
    {
        report.status = SpinFitStatus::TooFewSamples;
        return report;
    }
    if (t_max - t_min < config.min_span_s)
    {
        report.status = SpinFitStatus::SpanTooShort;
        return report;
    }
    const double tbar = weight_time_sum / weight_sum;

    // 2) 粗网格枚举 C(ω)：步长 0.25 rad/s，保证不会漏掉任何一座峰，也不会
    //    把最优化停在两峰之间的鞍点上。
    const double max_abs = config.max_abs_rad_s;
    const int steps = std::max(1, static_cast<int>(std::ceil(2.0 * max_abs / config.grid_step_rad_s)));
    std::vector<double> grid_omega(static_cast<std::size_t>(steps) + 1, 0.0);
    std::vector<double> grid_score(static_cast<std::size_t>(steps) + 1, 0.0);
    int best_index = 0;
    for (int i = 0; i <= steps; ++i)
    {
        const double omega = -max_abs + 2.0 * max_abs * static_cast<double>(i) /
                                              static_cast<double>(steps);
        grid_omega[static_cast<std::size_t>(i)] = omega;
        grid_score[static_cast<std::size_t>(i)] = score(samples, omega, tbar, plates);
        if (grid_score[static_cast<std::size_t>(i)] >
            grid_score[static_cast<std::size_t>(best_index)])
        {
            best_index = i;
        }
    }

    // 3) 选峰：在 C 上叠一层**线性**先验罚（见头文件：为什么不能是二次）。
    //    罚项只参与"选哪座峰"，峰内落位仍用纯 C 精化，所以不会把估计值往先验上拽。
    int chosen_index = best_index;
    if (config.use_prior && std::isfinite(config.prior_rad_s) &&
        config.prior_slope_penalty > 0.0)
    {
        double best_adjusted = -std::numeric_limits<double>::infinity();
        for (int i = 0; i <= steps; ++i)
        {
            const double here = grid_score[static_cast<std::size_t>(i)];
            const bool is_peak = (i == 0 || here >= grid_score[static_cast<std::size_t>(i - 1)]) &&
                                 (i == steps || here > grid_score[static_cast<std::size_t>(i + 1)]);
            if (!is_peak)
            {
                continue;
            }
            const double adjusted =
                here - config.prior_slope_penalty *
                           std::fabs(grid_omega[static_cast<std::size_t>(i)] - config.prior_rad_s);
            if (adjusted > best_adjusted)
            {
                best_adjusted = adjusted;
                chosen_index = i;
            }
        }
    }
    const double best_omega = grid_omega[static_cast<std::size_t>(chosen_index)];

    // 黄金分割（最大化）。区间取 ±1.5 个网格步长，覆盖粗网格的量化误差。
    const double inv_phi = 0.6180339887498949;
    double lo = best_omega - 1.5 * config.grid_step_rad_s;
    double hi = best_omega + 1.5 * config.grid_step_rad_s;
    double c1 = hi - inv_phi * (hi - lo);
    double c2 = lo + inv_phi * (hi - lo);
    double f1 = score(samples, c1, tbar, plates);
    double f2 = score(samples, c2, tbar, plates);
    for (int i = 0; i < config.refine_iterations; ++i)
    {
        if (f1 < f2)
        {
            lo = c1;
            c1 = c2;
            f1 = f2;
            c2 = lo + inv_phi * (hi - lo);
            f2 = score(samples, c2, tbar, plates);
        }
        else
        {
            hi = c2;
            c2 = c1;
            f2 = f1;
            c1 = hi - inv_phi * (hi - lo);
            f1 = score(samples, c1, tbar, plates);
        }
    }
    const double omega = 0.5 * (lo + hi);

    // 3) 在最优点上取 φ0、集中度与残差。
    double phase = 0.0;
    const double c_star = concentration(samples, omega, tbar, plates, &phase);
    double residual_sq = 0.0;
    double time_sq = 0.0;
    double w_used = 0.0;
    for (const SpinSample &sample : samples)
    {
        if (!std::isfinite(sample.timestamp_s) || !std::isfinite(sample.phase_rad) ||
            !(sample.weight > 0.0))
        {
            continue;
        }
        const double dt = sample.timestamp_s - tbar;
        const double residual = fold(sample.phase_rad - phase - omega * dt, fold_period);
        residual_sq += sample.weight * residual * residual;
        time_sq += sample.weight * dt * dt;
        w_used += sample.weight;
    }
    if (!(w_used > 0.0) || !(time_sq > 1.0e-12))
    {
        report.status = SpinFitStatus::DegenerateTiming;
        return report;
    }

    // 最小二乘斜率的经典方差：σ_ω² = σ_res² / Σw·(t-t̄)²。
    // σ_res 用折叠后残差的加权 RMS；折叠本身把残差限制在 ±π/N 内，
    // 所以这里天然有下限，不会因为"残差恰巧很小"就报出 0。
    double residual_var = residual_sq / w_used;
    double omega_out = omega;
    double phase_out = phase;
    double sigma = std::sqrt(residual_var / time_sq) * config.sigma_scale;

    // 3b) 二次精化：把窗口内的相位曲率也解出来，ω 因此变成"窗口中心时刻的
    //     瞬时转速"，长窗口不再把变速目标的峰谷平均掉。
    if (config.quadratic_refinement &&
        static_cast<int>(samples.size()) >= config.min_samples_for_quadratic)
    {
        // 正规方程 (XᵀWX)·[a b c]ᵀ = XᵀWr，基函数 [1, x, x²]，x 以 t̄ 为原点。
        double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
        double t0 = 0.0, t1 = 0.0, t2 = 0.0;
        double count = 0.0;
        for (const SpinSample &sample : samples)
        {
            if (!std::isfinite(sample.timestamp_s) || !std::isfinite(sample.phase_rad) ||
                !(sample.weight > 0.0))
            {
                continue;
            }
            const double x = sample.timestamp_s - tbar;
            const double w = sample.weight;
            const double r = fold(sample.phase_rad - phase - omega * x, fold_period);
            const double x2 = x * x;
            s0 += w;
            s1 += w * x;
            s2 += w * x2;
            s3 += w * x2 * x;
            s4 += w * x2 * x2;
            t0 += w * r;
            t1 += w * x * r;
            t2 += w * x2 * r;
            count += 1.0;
        }
        Eigen::Matrix3d normal;
        normal << s0, s1, s2, s1, s2, s3, s2, s3, s4;
        Eigen::Vector3d rhs(t0, t1, t2);
        const double det = normal.determinant();
        if (count >= 3.0 && std::fabs(det) > 1.0e-12)
        {
            const Eigen::Vector3d coeff = normal.inverse() * rhs;
            const double a = coeff(0);
            const double b = coeff(1);
            const double c = coeff(2);
            // 残差方差（自由度 n-3）。
            double quad_sq = 0.0;
            for (const SpinSample &sample : samples)
            {
                if (!std::isfinite(sample.timestamp_s) || !std::isfinite(sample.phase_rad) ||
                    !(sample.weight > 0.0))
                {
                    continue;
                }
                const double x = sample.timestamp_s - tbar;
                const double r = fold(sample.phase_rad - phase - omega * x, fold_period);
                const double e = r - (a + b * x + c * x * x);
                quad_sq += sample.weight * e * e;
            }
            const double dof = std::max(1.0, count - 3.0);
            const double var = quad_sq / dof / std::max(s0, 1.0e-9);
            const Eigen::Matrix3d cov = normal.inverse();
            const double var_b = var * cov(1, 1) * s0;
            const double var_c = var * cov(2, 2) * s0;
            // 曲率显著性门限：**只有数据真的显示出相位曲率时才用二次项**。
            // 匀速目标（redrotate）上 c 与 0 不可区分，硬留二次项只会把方差从
            // 线性解搬进 ω（实测 std 0.19 -> 0.28）。所以 |c| 不显著就退回线性解。
            const double curvature_significance =
                (var_c > 0.0) ? std::fabs(c) / std::sqrt(var_c) : 0.0;
            const double candidate = omega + b;
            if (curvature_significance >= config.curvature_significance &&
                std::isfinite(candidate) && std::isfinite(var_b) && var_b > 0.0 &&
                std::fabs(candidate) <= config.max_abs_rad_s)
            {
                omega_out = candidate;
                phase_out = phase + a;
                residual_var = quad_sq / std::max(1.0, count - 3.0) / std::max(s0, 1.0e-9);
                sigma = std::sqrt(var_b) * config.sigma_scale;
            }
        }
    }

    if (!std::isfinite(sigma) || sigma < config.sigma_floor_rad_s)
    {
        sigma = config.sigma_floor_rad_s;
    }

    report.fit.ok = true;
    report.fit.omega_rad_s = omega_out;
    report.fit.phase_rad = phase_out;
    report.fit.sigma_rad_s = sigma;
    report.fit.concentration = c_star;
    report.fit.residual_rms_rad = std::sqrt(residual_var);
    report.fit.samples = static_cast<int>(samples.size());

    if (c_star < config.min_concentration)
    {
        report.fit.ok = false;
        report.status = SpinFitStatus::LowConcentration;
        return report;
    }
    if (!std::isfinite(omega_out) || std::fabs(omega_out) > config.max_abs_rad_s + 1.0e-9)
    {
        report.fit.ok = false;
        report.status = SpinFitStatus::BadConfig;
        return report;
    }
    return report;
}

} // namespace rotating_armor
