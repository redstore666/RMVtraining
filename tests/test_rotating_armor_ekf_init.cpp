// 首帧锁定（初始化）契约测试（任务 4 第 2 步验收 b/c）。
//
// 覆盖三件事：
//   c) 生产 EKF 的 initialize / reset 可独立调用，状态与协方差可观测，
//      非法输入被拒绝且不污染滤波器状态；
//   b) 四条初始化路径的收敛对比（打印 cx/cy/R/yaw 误差）：
//        Spinning + 板面法向（推荐）/ Spinning + 无相位先验（降级）/
//        Static 声明 + 静止真值（逃生舱）/ Static 声明 + 旋转真值（应当"响亮地失败"）
//   d) 推荐路径的多噪声种子鲁棒性统计（初始化契约存在的直接理由）
//
// 边界：slot 视为已知（关联不在本步范围内）。

#include "RotatingArmorEkf.hpp"
#include "ekf_test_support.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace ekf_test;
using namespace rotating_armor;

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

EkfInitPrior makePrior(SpinPriorKind spin, double yaw_rate_prior_rad_s) {
    EkfInitPrior prior;
    prior.radius_m = 0.28; // 真值 0.25：故意留 12% 偏差
    prior.plate_slot = kObservedSlot;
    prior.spin = spin;
    prior.yaw_rate_prior_rad_s = yaw_rate_prior_rad_s;
    prior.position_sigma_m = 0.5;
    prior.velocity_sigma_m_s = 1.0;
    prior.yaw_sigma_rad = 0.2;
    prior.yaw_rate_sigma_rad_s = 3.0;
    prior.radius_sigma_m = 0.05;
    return prior;
}

// 抹掉板面法向通道：用来比较"同一条位置序列"下有无相位先验的差别。
TruthRun withoutWorldYaw(const TruthRun& run) {
    TruthRun out = run;
    for (ArmorObservation& observation : out.observations) {
        observation.has_world_yaw = false;
        observation.world_yaw_rad = 0.0;
    }
    return out;
}

void printHeader(const char* title) {
    std::cout << "\n--- " << title << " ---\n";
    std::cout << std::left << std::setw(30) << "init path" << std::right << std::setw(10) << "dcx"
              << std::setw(10) << "dcy" << std::setw(10) << "dR" << std::setw(10) << "dyaw"
              << std::setw(12) << "domega" << std::setw(12) << "plate_err" << std::setw(12)
              << "plate_max" << "\n";
}

void printRow(const std::string& name, const Error& e) {
    std::cout << std::left << std::setw(30) << name << std::right << std::setw(10) << e.cx
              << std::setw(10) << e.cy << std::setw(10) << e.radius << std::setw(10) << e.yaw
              << std::setw(12) << e.yaw_rate << std::setw(12) << e.plate << std::setw(12)
              << e.plate_max << "\n";
}

bool tightly_converged(const Error& e) {
    return std::abs(e.cx) <= 0.03 && std::abs(e.cy) <= 0.03 && std::abs(e.radius) <= 0.02 &&
           std::abs(e.yaw) <= 0.10 && e.plate <= 0.02 && e.plate_max <= 0.05;
}

} // namespace

int main() {
    constexpr double kSigmaMeasurement = 0.01;
    constexpr double kSigmaYaw = 0.05;

    const TruthRun truth = simulateTruth(truthState(), kSigmaMeasurement, kSigmaYaw, 20240607u, true);
    const TruthRun degraded_truth = withoutWorldYaw(truth);
    const TruthRun static_truth =
        simulateTruth(staticTruthState(), kSigmaMeasurement, kSigmaYaw, 20240607u, true);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== 首帧锁定契约测试 ===\n";
    std::cout << "真值(旋转): omega=" << truth.states.front()(7) << " R=" << truth.states.front()(8)
              << "  |  真值(静止): omega=" << static_truth.states.front()(7)
              << " R=" << static_truth.states.front()(8) << "\n";
    if (truth.yaw_swept < kPi) {
        std::cerr << "FAIL: 旋转真值累计转过不足 pi，收敛结论不成立\n";
        return 1;
    }

    // ============================================================== (c) 生命周期
    std::cout << "\n--- (c) 生命周期：未初始化 / 非法输入 / reset ---\n";
    {
        RotatingArmorEkf ekf;
        expect(!ekf.initialized(), "新建的 EKF 必须是未初始化");
        expect(!ekf.phase_resolved(), "新建的 EKF 不该声称相位已解");
        expect(ekf.state().isZero() && ekf.covariance().isZero(),
               "未初始化时状态与协方差必须为零");
        expect(!ekf.predict(kDt), "未初始化时 predict 必须返回 false");
        expect(!ekf.update(truth.observations.front(), kObservedSlot),
               "未初始化时 update 必须返回 false");

        const ArmorObservation observation = truth.observations.front();

        // 非法半径：0 / 负 / NaN
        for (const double bad_radius : {0.0, -0.3, std::numeric_limits<double>::quiet_NaN()}) {
            EkfInitPrior prior = makePrior(SpinPriorKind::Spinning, 2.0);
            prior.radius_m = bad_radius;
            const EkfInitStatus status = ekf.initialize(observation, prior);
            expect(status == EkfInitStatus::InvalidRadiusPrior,
                   "非法半径必须返回 InvalidRadiusPrior，实际 " + std::string(toString(status)));
        }

        // 非法 slot
        {
            EkfInitPrior prior = makePrior(SpinPriorKind::Spinning, 2.0);
            prior.plate_slot = kPlateCount;
            expect(ekf.initialize(observation, prior) == EkfInitStatus::InvalidPlateSlot,
                   "slot 越界必须返回 InvalidPlateSlot");
        }

        // 非法观测：NaN 位置 / sigma <= 0
        {
            const EkfInitPrior prior = makePrior(SpinPriorKind::Spinning, 2.0);
            ArmorObservation bad = observation;
            bad.position(1) = std::numeric_limits<double>::quiet_NaN();
            expect(ekf.initialize(bad, prior) == EkfInitStatus::InvalidObservation,
                   "NaN 观测必须返回 InvalidObservation");
            bad = observation;
            bad.sigma_position_m = 0.0;
            expect(ekf.initialize(bad, prior) == EkfInitStatus::InvalidObservation,
                   "sigma <= 0 必须返回 InvalidObservation");
        }

        // 角速度先验的声明：默认（Unspecified）必须被拒绝 -> 禁止 omega=0 静默冷启动
        {
            EkfInitPrior prior = makePrior(SpinPriorKind::Unspecified, 0.0);
            const EkfInitStatus status = ekf.initialize(observation, prior);
            expect(status == EkfInitStatus::SpinPriorUnspecified,
                   "未声明 spin 时必须返回 SpinPriorUnspecified，实际 " + std::string(toString(status)));
        }
        // Spinning 却给 0 / NaN
        {
            EkfInitPrior prior = makePrior(SpinPriorKind::Spinning, 0.0);
            expect(ekf.initialize(observation, prior) == EkfInitStatus::InvalidSpinPrior,
                   "Spinning + omega=0 必须返回 InvalidSpinPrior");
            prior.yaw_rate_prior_rad_s = std::numeric_limits<double>::quiet_NaN();
            expect(ekf.initialize(observation, prior) == EkfInitStatus::InvalidSpinPrior,
                   "Spinning + NaN 角速度必须返回 InvalidSpinPrior");
        }
        // Static 却给非零角速度（自相矛盾的声明）
        {
            EkfInitPrior prior = makePrior(SpinPriorKind::Static, 2.0);
            expect(ekf.initialize(observation, prior) == EkfInitStatus::InvalidSpinPrior,
                   "Static + 非零角速度必须返回 InvalidSpinPrior");
        }
        expect(!ekf.initialized(), "所有失败的 initialize 都不得改变滤波器状态");

        // 成功路径：Spinning + 板面法向
        const EkfInitPrior prior = makePrior(SpinPriorKind::Spinning, 2.0);
        expect(ekf.initialize(observation, prior) == EkfInitStatus::Ok,
               "推荐路径的 initialize 必须成功");
        expect(ekf.initialized() && ekf.phase_resolved(),
               "有板面法向时 initialized / phase_resolved 必须为 true");

        const State x0 = ekf.state();
        const State& x_true0 = truth.states.front();
        const double center_error = std::hypot(x0(0) - x_true0(0), x0(2) - x_true0(2));
        std::cout << "  初始化状态: yaw=" << x0(6) << " (真值 " << x_true0(6) << ")  R=" << x0(8)
                  << " (先验 " << prior.radius_m << ")  cx=" << x0(0) << " (真值 " << x_true0(0)
                  << ")  cy=" << x0(2) << " (真值 " << x_true0(2) << ")  omega=" << x0(7)
                  << "  中心误差=" << center_error << " m\n";
        expect(std::abs(wrapAngle(x0(6) - x_true0(6))) <= 0.20,
               "初相必须落在窗口附近（由板面法向确定）");
        expect(std::abs(x0(8) - prior.radius_m) < 1e-12, "半径初值必须等于先验半径");
        expect(std::abs(x0(4) - observation.position(2)) < 1e-12, "z 初值必须等于观测 z");
        expect(std::abs(x0(7) - prior.yaw_rate_prior_rad_s) < 1e-12,
               "角速度初值必须等于声明的先验值");
        expect(center_error <= 0.10,
               "车体中心初值必须落在真值附近（实际偏 " + std::to_string(center_error) + " m）");
        expect((measurement(x0, kObservedSlot, kPlateCount) - observation.position).norm() < 1e-12,
               "初始化必须与首帧观测一致（残差为 0，无需再补一次 update）");

        // 可观测性：predict 让不确定度增长，update 让它下降
        const Covariance p0 = ekf.covariance();
        expect(ekf.predict(kDt), "已初始化后 predict 必须成功");
        const double pos_var_after_predict = ekf.covariance()(0, 0);
        expect(pos_var_after_predict > p0(0, 0), "predict 必须让位置方差增大（Q > 0）");
        expect(ekf.update(truth.observations[1], kObservedSlot), "合法观测的 update 必须成功");
        expect(ekf.covariance()(0, 0) < pos_var_after_predict, "update 必须让位置方差下降");
        expect(ekf.covariance().allFinite() && ekf.state().allFinite(),
               "状态与协方差必须保持有限");

        // 非法 update 不得改变状态
        {
            const State before = ekf.state();
            expect(!ekf.update(truth.observations[2], kPlateCount), "越界 slot 的 update 必须失败");
            ArmorObservation bad = truth.observations[2];
            bad.position(0) = std::numeric_limits<double>::quiet_NaN();
            expect(!ekf.update(bad, kObservedSlot), "NaN 观测的 update 必须失败");
            bad = truth.observations[2];
            bad.sigma_position_m = -1.0;
            expect(!ekf.update(bad, kObservedSlot), "非法 sigma 的 update 必须失败");
            expect((ekf.state() - before).norm() < 1e-15, "失败的 update 不得改变状态");
        }

        // dt 边界：dt == 0 是合法空推进；负值/NaN 被拒绝且状态不变
        {
            const State before = ekf.state();
            expect(ekf.predict(0.0), "dt == 0 应视为合法空推进");
            expect((ekf.state() - before).norm() < 1e-15, "dt == 0 不得改变状态");
            expect(!ekf.predict(-0.01), "负 dt 必须被拒绝");
            expect(!ekf.predict(std::numeric_limits<double>::quiet_NaN()), "NaN dt 必须被拒绝");
            expect((ekf.state() - before).norm() < 1e-15, "被拒绝的 predict 不得改变状态");
        }

        // reset 可独立调用，且之后能重新锁定
        ekf.reset();
        expect(!ekf.initialized() && !ekf.phase_resolved(), "reset 后必须回到未初始化");
        expect(ekf.state().isZero() && ekf.covariance().isZero(), "reset 后状态与协方差必须清零");
        expect(!ekf.predict(kDt), "reset 后 predict 必须回到 false");
        expect(!ekf.update(truth.observations[0], kObservedSlot), "reset 后 update 必须回到 false");
        ekf.reset(); // 重复调用必须安全
        expect(ekf.initialize(truth.observations.front(), prior) == EkfInitStatus::Ok,
               "reset 之后必须能重新 initialize");
        expect(ekf.initialized(), "重新 initialize 后必须重新进入已初始化状态");
    }

    // ============================================================== (b) 路径对比
    const auto init_and_run = [](RotatingArmorEkf& ekf, const TruthRun& run,
                                 const EkfInitPrior& prior) -> Error {
        const EkfInitStatus status = ekf.initialize(run.observations.front(), prior);
        expect(status == EkfInitStatus::Ok,
               "initialize 应成功，实际 " + std::string(toString(status)));
        return runFilter(ekf, run, kFrames);
    };

    printHeader("初始化路径收敛对比（300 帧 / 5 秒，末帧误差）");

    // 1) 推荐路径：Spinning + 板面法向
    Error e_spin_yaw;
    {
        RotatingArmorEkf ekf;
        e_spin_yaw = init_and_run(ekf, truth, makePrior(SpinPriorKind::Spinning, 2.0));
        printRow("Spinning+板面法向（断言）", e_spin_yaw);
        expect(tightly_converged(e_spin_yaw), "推荐路径必须紧收敛");
    }

    // 2) 降级路径：Spinning，但没有板面法向 -> 相位只能靠多帧收敛。
    //    实测（8 个种子）只有 4/8 能回到真值附近，所以这里只断言"有界、不发散"，
    //    并要求 phase_resolved() == false 让调用方知道"还没锁定"。
    Error e_spin_noyaw;
    {
        RotatingArmorEkf ekf;
        e_spin_noyaw = init_and_run(ekf, degraded_truth, makePrior(SpinPriorKind::Spinning, 2.0));
        printRow("Spinning+无相位先验（有界）", e_spin_noyaw);
        expect(!ekf.phase_resolved(), "降级路径不得声称相位已解");
        expect(std::abs(e_spin_noyaw.cx) <= 0.60 && std::abs(e_spin_noyaw.cy) <= 0.60 &&
                   e_spin_noyaw.plate <= 0.05 && std::isfinite(e_spin_noyaw.plate),
               "降级路径必须保持有界（可以不准，但不能跑飞）");
    }

    // 3) 逃生舱：显式声明 Static + 静止真值
    Error e_static_true;
    {
        RotatingArmorEkf ekf;
        e_static_true = init_and_run(ekf, static_truth, makePrior(SpinPriorKind::Static, 0.0));
        printRow("Static+静止真值（断言）", e_static_true);
        expect(tightly_converged(e_static_true), "Static 声明在静止真值上必须紧收敛");
        expect(ekf.phase_resolved(), "有板面法向时 phase_resolved 应为 true");
    }

    // 4) 声明的可证伪性：Static 声明用在旋转真值上必须"响亮地失败"。
    //    判别式**不是**板残差：有观测时 yaw 状态会把相位吸收掉，即使 omega 被
    //    钉成 0，每帧观测也能把 yaw 推着跟上旋转（实测这条真值 5 秒转过 12.5rad
    //    ≈ 4*pi，本来还有末帧 2*pi 混叠；再看全程 plate_max 也只有 0.029m，与
    //    正确声明的 0.031m 根本分不开）。omega 错的唯一响亮后果是**冻结观测后
    //    预测立刻跑飞**：omega=0 的模型板位置不动，真值以 R*omega=0.625 m/s 转走。
    {
        constexpr int kObserved = 200; // 先带观测跟 200 帧
        constexpr int kHoldout = 18;   // 再纯预测 0.3s
        const auto extrapolate = [&](const TruthRun& run, const EkfInitPrior& prior) {
            RotatingArmorEkf ekf;
            const EkfInitStatus status = ekf.initialize(run.observations.front(), prior);
            expect(status == EkfInitStatus::Ok,
                   "外推对比的 initialize 应成功，实际 " + std::string(toString(status)));
            return extrapolationPlateError(ekf, run, kObserved, kHoldout);
        };
        const double ext_spin = extrapolate(truth, makePrior(SpinPriorKind::Spinning, 2.0));
        const double ext_static_on_spin = extrapolate(truth, makePrior(SpinPriorKind::Static, 0.0));
        const double ext_static = extrapolate(static_truth, makePrior(SpinPriorKind::Static, 0.0));
        const double ext_spin_on_static =
            extrapolate(static_truth, makePrior(SpinPriorKind::Spinning, 2.0));
        std::cout << "  冻结观测纯外推 " << kHoldout << " 帧（0.3s）后的板位置误差:  旋转真值+Spinning="
                  << ext_spin << " m   旋转真值+Static=" << ext_static_on_spin
                  << " m   静止真值+Static=" << ext_static
                  << " m   静止真值+Spinning(声明错)=" << ext_spin_on_static << " m\n";
        expect(ext_spin < 0.05, "正确声明下 0.3s 纯外推必须仍然准（实际 " + std::to_string(ext_spin) +
                                    " m）");
        expect(ext_static < 0.05, "静止真值 + Static 声明下 0.3s 纯外推必须仍然准（实际 " +
                                      std::to_string(ext_static) + " m）");
        expect(ext_static_on_spin > 0.10,
               "Static 声明用在旋转真值上，0.3s 纯外推必须跑飞（实际 " +
                   std::to_string(ext_static_on_spin) + " m），否则该声明不可证伪");
    }

    // 4b) 诊断行：Static 声明 + 旋转真值（带观测跑满 300 帧）。用来记录"yaw 吸收
    //     相位"这个现象：板残差全程都在 3cm 内，但 omega 恒为 0（domega=-2.5）。
    {
        RotatingArmorEkf ekf;
        const Error e = init_and_run(ekf, truth, makePrior(SpinPriorKind::Static, 0.0));
        printRow("Static+旋转真值（诊断）", e);
    }

    // 5) 声明的容错：Spinning 声明用在静止真值上（声明错了但同族），应能收敛回 omega≈0。
    //    诊断行，只打印。
    {
        RotatingArmorEkf ekf;
        const Error e = init_and_run(ekf, static_truth, makePrior(SpinPriorKind::Spinning, 2.0));
        printRow("Spinning+静止真值（诊断）", e);
    }

    // ============================================================== (d) 多种子鲁棒性
    std::cout << "\n--- (d) 多噪声种子鲁棒性（每个种子独立生成观测）---\n";
    {
        const std::vector<std::uint32_t> seeds{1u, 2u, 3u, 7u, 42u, 99u, 12345u, 20240607u};
        int ok_spin_yaw = 0;
        int ok_spin_noyaw = 0;
        double worst_spin_yaw_cx = 0.0;
        for (const std::uint32_t seed : seeds) {
            const TruthRun run = simulateTruth(truthState(), kSigmaMeasurement, kSigmaYaw, seed, true);
            const TruthRun run_noyaw = withoutWorldYaw(run);
            {
                RotatingArmorEkf ekf;
                if (ekf.initialize(run.observations.front(), makePrior(SpinPriorKind::Spinning, 2.0)) ==
                    EkfInitStatus::Ok) {
                    const Error e = runFilter(ekf, run, kFrames);
                    if (tightly_converged(e)) {
                        ++ok_spin_yaw;
                    }
                    worst_spin_yaw_cx = std::max(worst_spin_yaw_cx, std::abs(e.cx));
                }
            }
            {
                RotatingArmorEkf ekf;
                if (ekf.initialize(run_noyaw.observations.front(),
                                   makePrior(SpinPriorKind::Spinning, 2.0)) == EkfInitStatus::Ok) {
                    const Error e = runFilter(ekf, run_noyaw, kFrames);
                    if (tightly_converged(e)) {
                        ++ok_spin_noyaw;
                    }
                }
            }
        }
        const int total = static_cast<int>(seeds.size());
        std::cout << "  Spinning+板面法向 紧收敛 " << ok_spin_yaw << "/" << total
                  << "（最差 |dcx|=" << worst_spin_yaw_cx << " m）\n";
        std::cout << "  Spinning+无相位先验 紧收敛 " << ok_spin_noyaw << "/" << total
                  << "（说明该路径必须靠 phase_resolved() == false 暴露「还没锁定」）\n";
        expect(ok_spin_yaw == total,
               "推荐路径必须在全部种子上紧收敛（实际 " + std::to_string(ok_spin_yaw) + "/" +
                   std::to_string(total) + "）");
    }

    if (failures != 0) {
        std::cout << "\n首帧锁定契约测试 FAILED（" << failures << " 处）\n";
        return 1;
    }
    std::cout << "\n首帧锁定契约测试通过\n";
    return 0;
}
