#pragma once

// 合成真值测试夹具（任务 4 第 2 步）：被收敛测试与初始化契约测试共用，
// 避免两份真值生成/误差统计代码漂移。
//
// 约定与生产 EKF 一致：观测是装甲板中心在跟踪系（前 x / 左 y / 上 z）的 3D
// 位置，观测噪声 sigma 取实拍距离帧间抖动的量级（~1cm）。slot 视为已知
// （等价于关联完全正确）——关联做错时这些测试不会报警，这是刻意的边界。

#include "RotatingArmorEkf.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace ekf_test {

using namespace rotating_armor;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kPlateCount = 4;
constexpr int kObservedSlot = 2; // 真值生成与滤波更新都用同一块板
constexpr double kDt = 1.0 / 60.0;
constexpr int kFrames = 300; // >= 200 帧，且 yaw 累计转过 >= pi

// 真值：[cx, vx, cy, vy, z, vz, yaw, yaw_rate, radius]
// 注：按 measurement() 的相位约定（板中心 = 中心 - R[cos, sin]），
//     x(0)=cx 是车体中心初值，而不是任何时刻的板位置。
inline State truthState() {
    // 状态已从 9 维扩到 11 维（尾部补 pair_radius_diff / pair_height_diff），
    // 这里必须填满 kStateDim 个数，否则 Eigen 的逗号初始化会断言 abort。
    return (State() << 0.40, -0.25, 1.80, 0.10, 0.35, 0.0, 0.20, 2.50, 0.25, 0.0, 0.0).finished();
}

// 同一条平移轨迹，但目标不转：用来验证 Static 声明的路径。
inline State staticTruthState() {
    State x = truthState();
    x(7) = 0.0;
    return x;
}

// 这块板的面法向在世界系里的 yaw（真值）。见 RotatingArmorModel.hpp 的约定。
inline double truthWorldYaw(const State& x, int plate_slot) {
    const double phase =
        x(6) + 2.0 * kPi * static_cast<double>(plate_slot) / static_cast<double>(kPlateCount);
    return platePhaseToWorldYaw(phase);
}

struct TruthRun {
    std::vector<State> states;                 // 每个采样时刻的真值
    std::vector<ArmorObservation> observations; // 对应的带噪观测
    double yaw_swept{0.0};                     // 真值 yaw 累计转过的角度（取绝对值累加）
};

// 生成一段"匀速平移 + 匀角速度旋转"的真值轨迹与观测序列（x_truth 的第 7 项为 0
// 时就是不转的目标）。with_world_yaw 为 true 时，每帧观测还带上板面法向
// （加 sigma_yaw_rad 噪声）。
inline TruthRun simulateTruth(const State& x_truth, double sigma_measurement, double sigma_yaw_rad,
                              std::uint32_t seed, bool with_world_yaw) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> gaussian(0.0, 1.0);

    TruthRun run;
    run.states.reserve(kFrames);
    run.observations.reserve(kFrames);

    State x = x_truth;
    for (int k = 0; k < kFrames; ++k) {
        run.states.push_back(x);

        ArmorObservation observation;
        observation.position = measurement(x, kObservedSlot, kPlateCount);
        for (int i = 0; i < kMeasurementDim; ++i) {
            observation.position(i) += sigma_measurement * gaussian(rng);
        }
        observation.sigma_position_m = sigma_measurement;
        observation.has_world_yaw = with_world_yaw;
        if (with_world_yaw) {
            observation.world_yaw_rad =
                wrapAngle(truthWorldYaw(x, kObservedSlot) + sigma_yaw_rad * gaussian(rng));
        }
        run.observations.push_back(observation);

        const State next = transition(x, kDt);
        run.yaw_swept += std::abs(wrapAngle(next(6) - x(6)));
        x = next;
    }
    return run;
}

struct Error {
    double cx{0.0};
    double cy{0.0};
    double z{0.0};
    double radius{0.0};
    double yaw{0.0};
    double yaw_rate{0.0};
    double plate{0.0};     // 末帧：用估计状态预测装甲板位置，与真值装甲板位置的差
    double plate_max{0.0}; // 全程最大（末帧存在 2*pi 混叠，见 runFilter 的注释）
};

inline Error errorAgainst(const State& estimate, const State& truth) {
    Error e;
    e.cx = estimate(0) - truth(0);
    e.cy = estimate(2) - truth(2);
    e.z = estimate(4) - truth(4);
    e.radius = estimate(8) - truth(8);
    e.yaw = wrapAngle(estimate(6) - truth(6));
    e.yaw_rate = estimate(7) - truth(7);
    e.plate = (measurement(estimate, kObservedSlot, kPlateCount) -
               measurement(truth, kObservedSlot, kPlateCount))
                  .norm();
    e.plate_max = e.plate;
    return e;
}

// 跑 frames 帧（第 0 帧的观测已经被 initialize() 吃进状态），返回末帧误差。
//
// 末帧的 plate 存在 2*pi 混叠：真值 yaw 累计转过接近 2*pi 整数倍时，"相位冻结"
// 的错误状态在末帧会与真值重合——实测 omega*5s = 12.5rad ≈ 4*pi，末帧板残差
// 只有 2cm，看起来像收敛了。所以这里额外累计全程最大值 plate_max，它才是
// "声明是否可证伪 / 相位是否真的跟上了"的判据。
inline Error runFilter(RotatingArmorEkf& ekf, const TruthRun& run, int frames) {
    double plate_max = 0.0;
    for (int k = 1; k < frames; ++k) {
        ekf.predict(kDt);
        ekf.update(run.observations[static_cast<std::size_t>(k)], kObservedSlot);
        plate_max = std::max(
            plate_max,
            errorAgainst(ekf.state(), run.states[static_cast<std::size_t>(k)]).plate);
    }
    Error last = errorAgainst(ekf.state(), run.states[static_cast<std::size_t>(frames - 1)]);
    last.plate_max = plate_max;
    return last;
}

// 先跟 observed_frames 帧（含 update），再冻结观测纯预测 steps 帧，返回末帧板位置误差。
//
// 为什么需要它：有观测时 yaw 状态会把相位吸收掉——即使 omega 被钉成 0，每帧的
// 观测也能把 yaw"推着"跟上旋转。实测 Static 声明用在旋转真值上 300 帧，板残差
// 全程最大只有 0.029m（与正确声明的 0.031m 无法区分），但 omega 错得离谱。
// 所以"转不转"这个声明只能靠**外推**证伪：冻结观测后，omega=0 的模型板位置
// 不动，而真值以 R*omega = 0.625 m/s 转走。
inline double extrapolationPlateError(RotatingArmorEkf ekf, const TruthRun& run,
                                      int observed_frames, int steps) {
    for (int k = 1; k < observed_frames; ++k) {
        ekf.predict(kDt);
        ekf.update(run.observations[static_cast<std::size_t>(k)], kObservedSlot);
    }
    const int start = observed_frames - 1;
    for (int k = 0; k < steps; ++k) {
        ekf.predict(kDt);
    }
    return (measurement(ekf.state(), kObservedSlot, kPlateCount) -
            measurement(run.states[static_cast<std::size_t>(start + steps)], kObservedSlot,
                        kPlateCount))
        .norm();
}

} // namespace ekf_test
