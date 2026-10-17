// 合成真值收敛测试（任务 4；第 2 步起改用生产 EKF）。
//
// 目的：证明 rotating_armor 的 F/H/Q 组合成 EKF 后，能把「匀速平移 + 匀角速度
//       旋转」的车体状态估收敛到真值——即模型数学不仅自洽（那是
//       test_rotating_armor_model.cpp 干的活），而且真的可观测、可收敛。
//
// 本文件同时承担「默认过程噪声」的防回退职责：
//   1) 断言 ProcessNoiseConfig 默认值仍处在"不鼓励状态乱跳"的量级；
//   2) 用默认值跑完整收敛，断言末帧误差在容差内。
//   历史默认值 accel=1.0 / angular=4.0 的那一行作为诊断打印保留，
//   它说明护栏在防什么，但不参与断言（避免以后调参时被它绊住）。
//
// 边界：slot 已知（等价于关联完全正确），因此本测试只验证滤波收敛，
//       不验证关联。

#include "RotatingArmorEkf.hpp"
#include "ekf_test_support.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>

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

void printHeader(const char* title) {
    std::cout << "\n--- " << title << " ---\n";
    std::cout << std::left << std::setw(26) << "scenario" << std::right << std::setw(10) << "dcx"
              << std::setw(10) << "dcy" << std::setw(10) << "dz" << std::setw(10) << "dR"
              << std::setw(10) << "dyaw" << std::setw(10) << "domega" << std::setw(12)
              << "plate_err" << std::setw(12) << "plate_max" << "\n";
}

void printRow(const std::string& name, const Error& e) {
    std::cout << std::left << std::setw(26) << name << std::right << std::setw(10) << e.cx
              << std::setw(10) << e.cy << std::setw(10) << e.z << std::setw(10) << e.radius
              << std::setw(10) << e.yaw << std::setw(10) << e.yaw_rate << std::setw(12) << e.plate
              << std::setw(12) << e.plate_max << "\n";
}

// 初始化先验：半径带 12% 偏差（0.28 vs 真值 0.25），角速度先验 2.0 rad/s
// （真值 2.5，即量级与符号对、数值差 20%）。
//
// 这里必须显式声明 Spinning：把 omega 先验留成 0（即使已经有板面法向给出的
// 相位先验）在默认 Q 下并不可靠——8 个噪声种子里有 3 个会锁进错误极小值
// （末帧中心差 0.30~0.43m、yaw 差 1.7~2.5rad，而板残差只有 1.5~2.1cm）。
// 证据见 test_rotating_armor_ekf_init.cpp 的多seed鲁棒性统计。
EkfInitPrior makePrior() {
    EkfInitPrior prior;
    prior.radius_m = 0.28;
    prior.plate_slot = kObservedSlot;
    prior.spin = SpinPriorKind::Spinning;
    prior.yaw_rate_prior_rad_s = 2.0;
    prior.position_sigma_m = 0.5;
    prior.velocity_sigma_m_s = 1.0;
    prior.yaw_sigma_rad = 0.2;
    prior.yaw_rate_sigma_rad_s = 3.0;
    prior.radius_sigma_m = 0.05;
    return prior;
}

} // namespace

int main() {
    constexpr double kSigmaMeasurement = 0.01; // 1cm，实拍距离帧间抖动量级
    constexpr double kSigmaYaw = 0.05;         // 板面法向的观测噪声（rad）

    const State x_truth = truthState();
    const TruthRun truth = simulateTruth(x_truth, kSigmaMeasurement, kSigmaYaw, 20240607u, true);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "=== 合成真值收敛测试（生产 EKF）===\n";
    std::cout << "真值: cx=" << x_truth(0) << " vx=" << x_truth(1) << " cy=" << x_truth(2)
              << " vy=" << x_truth(3) << " z=" << x_truth(4) << " yaw0=" << x_truth(6)
              << " omega=" << x_truth(7) << " R=" << x_truth(8) << "\n";
    std::cout << "参数: dt=" << kDt << "s  frames=" << kFrames
              << "  obs_sigma=" << kSigmaMeasurement << "m  yaw_sigma=" << kSigmaYaw
              << "rad  plate_count=" << kPlateCount << "  observed_slot=" << kObservedSlot << "\n";
    std::cout << "真值 yaw 累计转过 " << truth.yaw_swept << " rad (" << truth.yaw_swept / kPi
              << " pi)\n";

    if (kFrames < 200) {
        std::cerr << "FAIL: 帧数少于 200，转过角度不足以支撑收敛结论\n";
        return 1;
    }
    if (truth.yaw_swept < kPi) {
        std::cerr << "FAIL: 真值 yaw 累计转过不足 pi，观测不足以激励模型\n";
        return 1;
    }

    // --------------------------------------------------------- 默认过程噪声护栏
    const ProcessNoiseConfig defaults{};
    const double sigma_velocity_per_step = std::sqrt(defaults.accel_x * kDt);
    const double sigma_omega_per_step = std::sqrt(defaults.angular_accel * kDt);

    std::cout << "\n--- 默认过程噪声护栏 (dt=" << kDt << "s) ---\n";
    std::cout << "accel_x=" << defaults.accel_x << " angular_accel=" << defaults.angular_accel
              << " radius_random_walk=" << defaults.radius_random_walk << "\n";
    std::cout << "=> 每帧 sigma_v=" << sigma_velocity_per_step
              << " m/s  sigma_omega=" << sigma_omega_per_step << " rad/s\n";

    // 这四条就是"防回退"：任何一条被改回历史默认值的量级都会在这里失败。
    expect(defaults.accel_x <= 0.05, "default accel_x 必须 <= 0.05（历史默认值是 1.0）");
    expect(defaults.angular_accel <= 0.05, "default angular_accel 必须 <= 0.05（历史默认值是 4.0）");
    expect(defaults.radius_random_walk <= 1e-5,
           "default radius_random_walk 必须 <= 1e-5（历史默认值是 1e-4）");
    expect(sigma_velocity_per_step <= 0.02 && sigma_omega_per_step <= 0.02,
           "默认值换算到每帧的不确定度必须 <= 0.02（m/s 与 rad/s）");

    // --------------------------------------------------------- 默认值下的收敛断言
    const EkfInitPrior prior = makePrior();

    printHeader("末帧误差（300 帧 / 5 秒）");
    Error error_default;
    {
        RotatingArmorEkf ekf; // 默认配置 = 默认过程噪声
        const EkfInitStatus status = ekf.initialize(truth.observations.front(), prior);
        expect(status == EkfInitStatus::Ok,
               std::string("默认配置下 initialize 应成功，实际 ") + toString(status));
        expect(ekf.phase_resolved(), "有板面法向时 phase_resolved 应为 true");
        error_default = runFilter(ekf, truth, kFrames);
        printRow("default Q（断言）", error_default);
    }

    Error error_historical;
    {
        // 诊断行：历史默认值 accel=1.0 / angular=4.0 / radius_rw=1e-4。
        // 在"先验正确"的前提下它也能收敛，只是精度明显更差（见下面的相对断言）——
        // 真正的灾难要"大 Q + 错误初始化"叠加才会出现（配合本步的初始化契约后，
        // 那条路已经被 SpinPriorKind 挡住）。
        EkfConfig historical;
        historical.process = ProcessNoiseConfig{1.0, 1.0, 1.0, 4.0, 1e-4};
        RotatingArmorEkf ekf(historical);
        const EkfInitStatus status = ekf.initialize(truth.observations.front(), prior);
        expect(status == EkfInitStatus::Ok, "历史默认 Q 下的 initialize 也应成功");
        error_historical = runFilter(ekf, truth, kFrames);
        printRow("历史默认 Q（诊断）", error_historical);
    }

    const bool converged = std::abs(error_default.cx) <= 0.03 && std::abs(error_default.cy) <= 0.03 &&
                           std::abs(error_default.z) <= 0.03 &&
                           std::abs(error_default.radius) <= 0.02 &&
                           std::abs(error_default.yaw) <= 0.10 && error_default.plate <= 0.02;
    std::cout << "  [断言] 收敛判据 |dcx|,|dcy|,|dz|<=0.03m |dR|<=0.02m |dyaw|<=0.10rad "
                 "plate_err<=0.02m -> "
              << (converged ? "PASS" : "FAIL") << "\n";
    if (!converged) {
        std::cerr << "FAIL[default Q]: dcx=" << error_default.cx << " dcy=" << error_default.cy
                  << " dz=" << error_default.z << " dR=" << error_default.radius
                  << " dyaw=" << error_default.yaw << " plate_err=" << error_default.plate << "\n";
        ++failures;
    }

    // 相对护栏：默认值在同一个场景下不得差于历史默认值。
    expect(std::abs(error_default.cx) <= std::abs(error_historical.cx) + 1e-9,
           "默认 Q 的末帧中心误差不得差于历史默认值（default=" +
               std::to_string(error_default.cx) + " historical=" +
               std::to_string(error_historical.cx) + "）");

    // --------------------------------------------------------- 逐段收敛轨迹
    // 确认是"逐步收敛"而不是"末帧碰巧对上"。
    {
        RotatingArmorEkf ekf;
        ekf.initialize(truth.observations.front(), prior);

        std::cout << "\n--- 默认配置收敛轨迹 ---\n";
        std::cout << std::setw(6) << "frame" << std::setw(10) << "dcx" << std::setw(10) << "dcy"
                  << std::setw(10) << "dz" << std::setw(10) << "dR" << std::setw(10) << "dyaw"
                  << std::setw(12) << "plate_err" << "\n";
        for (int k = 1; k < kFrames; ++k) {
            ekf.predict(kDt);
            const bool updated =
                ekf.update(truth.observations[static_cast<std::size_t>(k)], kObservedSlot);
            expect(updated, "update 在合法输入下不应失败（帧 " + std::to_string(k) + "）");
            if (k % 50 == 0 || k == kFrames - 1) {
                const Error e = errorAgainst(ekf.state(), truth.states[static_cast<std::size_t>(k)]);
                std::cout << std::setw(6) << k << std::setw(10) << e.cx << std::setw(10) << e.cy
                          << std::setw(10) << e.z << std::setw(10) << e.radius << std::setw(10)
                          << e.yaw << std::setw(12) << e.plate << "\n";
            }
        }
        const State& estimate = ekf.state();
        const State& truth_final = truth.states.back();
        std::cout << "  末帧估计: cx=" << estimate(0) << " vx=" << estimate(1)
                  << " cy=" << estimate(2) << " vy=" << estimate(3) << " z=" << estimate(4)
                  << " yaw=" << estimate(6) << " omega=" << estimate(7) << " R=" << estimate(8)
                  << "\n";
        std::cout << "  末帧真值: cx=" << truth_final(0) << " vx=" << truth_final(1)
                  << " cy=" << truth_final(2) << " vy=" << truth_final(3) << " z=" << truth_final(4)
                  << " yaw=" << truth_final(6) << " omega=" << truth_final(7)
                  << " R=" << truth_final(8) << "\n";
        expect(estimate.allFinite(), "末帧状态必须有限");
    }

    if (failures != 0) {
        std::cout << "\n合成真值收敛测试 FAILED（" << failures << " 处）\n";
        return 1;
    }
    std::cout << "\n合成真值收敛测试通过\n";
    return 0;
}
