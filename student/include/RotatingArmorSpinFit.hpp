#pragma once

// 旋转车体的**单一**转速估计器（相位拟合）。
//
// 为什么要有这个文件（2026-10-31）：
//   原来 updateYawRateFromWindow() 里并排跑了四路证据——板面法向 Theil-Sen、
//   板相对车心的方位角 Theil-Sen、同帧双板弦向 Theil-Sen、全部原始法向全配对
//   Theil-Sen——再按"两路一致就逆方差加权 / 分歧就取另一路 / 异号只取幅度"
//   这套规则融合。每一路都有自己的折叠周期（2π、π、π/2）和自己的失效模式，
//   融合规则只能靠特例堆出来；更糟的是滑窗每移动一帧，参与两两配对的点集就
//   变一次，中位数斜率是**阶梯函数**——ω 曲线因此天然带锯齿，这就是旋转场景
//   里"omega 抖"的机制性来源，不是参数没调好。
//
// 本模块把四路换成一路：直接在相位域上拟合刚体旋转模型
//     θ(t) = φ0 + ω·(t - t̄)，残差按 2π/N 折叠（N = 板数）
// 其中
//   * 换板造成的 ±2π/N 跳变由折叠**内建**消除，不需要最近邻续接链，
//     也就不存在"续接到别的板 → 假高速"这一类伪值；
//   * 给定 ω 时 φ0 有闭式解：z = Σ w·exp(j·N·(θ - ω·t))，φ0 = arg(z)/N；
//     于是 ω 退化成**一维**搜索，粗网格 + 黄金分割即可，全局最优、无初值敏感；
//   * 不确定性由残差与时间散布直接给出：σ_ω = sqrt(Σw·r²/Σw / Σw·(t-t̄)²)。
//
// 数学上与 tools/gt_omega.py 的"4θ 周期图"是同一个量（都是 N 重折叠下的
// 相位集中度），所以真值工具与生产估计器口径一致，方便对拍。
//
// 依赖只有 Eigen 和标准库：不给 EKF、相机、OpenVINO 添任何耦合。

#include <vector>

namespace rotating_armor
{

// 一条相位样本。phase 是**板相位**，与 RotatingArmorEkf 的 state(6) 同定义
// （不是板面法向 yaw；法向要先用 worldYawToPlatePhase() 转过来）。
struct SpinSample
{
    double timestamp_s{0.0};
    double phase_rad{0.0};
    // 这一条的可信度。位置相位（板心绕车心的方位）精度高、给 1.0；
    // 法向相位受板面倾斜影响（实拍 ±20°），给低权重。
    double weight{1.0};
};

struct SpinFitConfig
{
    // 板数 N：折叠周期是 2π/N。四板车 N=4。
    int plate_count{4};
    // ω 的物理上限（rad/s）。搜索区间就是 [-max_abs, +max_abs]。
    double max_abs_rad_s{12.0};
    // 窗口至少要覆盖的时长；太短时 ω 与 φ0 不可分，直接判无效。
    double min_span_s{0.18};
    // 至少这么多条样本（加权前）。
    int min_samples{6};
    // 相位集中度门限 C ∈ [0,1]：C = |Σ w·e^{jN(θ-ωt)}| / Σw。
    // 数据说得清"这是一个刚体在匀速转"时 C 接近 1；散成一团时接近 0。
    // 低于门限说明这一窗的相位序列自相矛盾（换错板、混进别的车、法向野值），
    // 宁可这一帧不给 ω，让衰减项接管。
    double min_concentration{0.55};

    // === 先验：只用来"选哪座峰"，不决定峰内的落位 ===
    // 折叠后的目标函数在 ω 上有多座峰（相位每转 2π/N 就重合一次，主瓣两侧
    // 还有 -13 dB 的旁瓣，间距约 2π/(N·T)）。两座峰高度接近时靠"上一帧的 ω"
    // 挑更近的那座——真实转速是连续的，跳变几乎只可能是选错了峰。
    //
    // 关键：先验的代价必须**有界**。早期版本把它写成二次罚（C - ½w((ω-μ)/σ)²），
    // 实测在 8 rad/s 场景把 ω 锁死在 1 rad/s 再也不出来：一旦估计偏了，正确峰
    // 被罚 (8-1)²/2σ² ≈ 11，而 C 最多只有 1，罚项永远赢——估计器自锁。
    // 现在用线性罚，代价有界（5 rad/s 只罚 0.15），真峰（C 差 0.5~0.9）永远能翻盘。
    bool use_prior{true};
    double prior_rad_s{0.0};    // 上一帧 EKF 的 state(7)
    // 先验的**线性**（拉普拉斯）罚系数，单位 C/(rad/s)：score = C - penalty·|ω-prior|。
    //
    // 为什么是线性而不是二次：二次罚随偏离平方增长，一旦估计偏了，正确峰被罚到
    // 永远赢不回来（实测 8 rad/s 场景锁死在 1 rad/s）。线性罚的代价有界——帧间
    // 跳 5 rad/s 只罚 0.15——而"真峰 vs 别名峰"的集中度差通常有 0.5~0.9，所以
    // 真峰永远能翻盘，同时又足以压住"噪声把两座峰拉平、逐帧轮流赢"造成的方波
    // 突刺（实测 sinspeed 上 6 处单帧跳 5 rad/s 就是这么来的）。
    double prior_slope_penalty{0.03};

    // === 搜索 ===
    double grid_step_rad_s{0.25};    // 粗网格步长
    int refine_iterations{60};       // 黄金分割迭代次数（精度 ~1e-3 rad/s）

    // 二次精化：线性相位模型 θ=φ0+ω·t 对**变速**目标是错的——ω 在窗口内变化时，
    // 拟合出来的斜率是窗口平均，正弦素材的峰谷会被削平（实测 2.3~8.9 削成 3.6~8.1）。
    // 解决办法不是缩短窗口（那会放大噪声），而是把窗口内的相位曲率也当成参数：
    //     θ(t) = φ0 + ω·(t-t̄) + ½α·(t-t̄)²，
    // 此时 ω 恰好是**窗口中心时刻**的瞬时转速，峰谷不再被平均掉。
    // 做法：在全局搜索给出 ω0/φ0 后，把残差 r_i 展开成 3 参数加权最小二乘
    // (a, b, c)，取 ω = ω0 + b；σ_ω 由 (XᵀWX)⁻¹ 的对角项给出。
    bool quadratic_refinement{true};
    double min_samples_for_quadratic{10};
    // 曲率显著性门限：|c|/σ_c 超过它才采纳二次项。匀速目标上 c 与 0 不可区分，
    // 硬留二次项只会把方差搬进 ω（实测 std 0.19 -> 0.28），所以必须门控。
    double curvature_significance{2.0};

    // σ 的下限与放大系数。窗口逐帧滑动 => 相邻两帧的估计高度相关，
    // 直接把拟合协方差当独立量测喂进 EKF 会让滤波器过度自信（ω 追着噪声走）。
    double sigma_floor_rad_s{0.10};
    double sigma_scale{1.5};
};

struct SpinFitResult
{
    bool ok{false};
    double omega_rad_s{0.0};
    // t̄ 时刻的板相位（诊断用；不写进 EKF）。
    double phase_rad{0.0};
    double sigma_rad_s{0.0};
    // 相位集中度（0~1），越大说明数据越"像刚体匀速转"。
    double concentration{0.0};
    // 折叠后残差的 RMS（弧度），反映这一窗的相位抖动。
    double residual_rms_rad{0.0};
    int samples{0};
};

// 失败原因（诊断/日志用；ok=false 时才有意义）。
enum class SpinFitStatus : unsigned char
{
    Ok = 0,
    TooFewSamples,
    SpanTooShort,
    BadConfig,
    LowConcentration,
    DegenerateTiming,
};

struct SpinFitReport
{
    SpinFitResult fit{};
    SpinFitStatus status{SpinFitStatus::Ok};
};

SpinFitReport fitSpinPhase(const std::vector<SpinSample>& samples, const SpinFitConfig& config);

} // namespace rotating_armor
