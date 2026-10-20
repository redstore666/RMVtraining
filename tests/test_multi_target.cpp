// 多目标粘合层（MultiTargetTracker）的验收测试：
//   A 同场两辆车：都能各自建出一辆、都收敛到真值附近，且不会为同一辆反复新建
//   B 只有一辆车时：不会凭空多出一辆（零误报）
#include "MultiTargetTracker.hpp"
#include "tracker_test_support.hpp"

#include <cmath>
#include <cstdio>

namespace
{
using rotating_armor::MultiTargetConfig;
using rotating_armor::MultiTargetTrack;
using rotating_armor::MultiTargetTracker;
using rotating_armor::TrackerConfig;
using rotating_armor::State;

int g_failures = 0;

void check(bool condition, const char *what)
{
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
    {
        ++g_failures;
    }
}

struct Body
{
    State truth{State::Zero()};
    int target_id{0};
};

// 一辆车 = 4 块板；两辆车的板拼在一起，各自带自己的 target_id。
std::vector<ArmorPose> observeBoth(const std::vector<Body> &bodies, const GimbalState &gimbal,
                                   std::mt19937 &rng)
{
    std::vector<ArmorPose> all;
    for (const Body &body : bodies)
    {
        const auto plates = sim::platesAt(body.truth, 4);
        const auto poses = sim::observe(plates, gimbal, body.target_id, rng);
        all.insert(all.end(), poses.begin(), poses.end());
    }
    return all;
}

// 找"中心离 truth 最近"的那辆车，返回它的中心误差；找不到返回 inf。
double nearestCenterError(const MultiTargetTracker &multi, const State &truth)
{
    double best = std::numeric_limits<double>::infinity();
    for (const MultiTargetTrack &track : multi.tracks())
    {
        if (!track.tracker || !track.tracker->initialized())
            continue;
        const State &state = track.tracker->ekf().state();
        best = std::min(best, std::hypot(state(0) - truth(0), state(2) - truth(2)));
    }
    return best;
}

double nearestRadius(const MultiTargetTracker &multi, const State &truth)
{
    double best = std::numeric_limits<double>::infinity();
    for (const MultiTargetTrack &track : multi.tracks())
    {
        if (!track.tracker || !track.tracker->initialized())
            continue;
        const State &state = track.tracker->ekf().state();
        const double center_error = std::hypot(state(0) - truth(0), state(2) - truth(2));
        if (center_error < 0.30)
            best = std::min(best, std::fabs(state(8) - truth(8)));
    }
    return best;
}

// A：两辆车（一辆自转、一辆静止），相距 1.2 m，跑 3 秒。
void scenarioTwoBodies()
{
    Body spinning;
    spinning.truth = State::Zero();
    spinning.truth(0) = 3.0;
    spinning.truth(7) = 2.0;  // omega
    spinning.truth(8) = 0.25; // R
    spinning.target_id = 3;

    Body static_body;
    static_body.truth = State::Zero();
    static_body.truth(0) = 3.2;
    static_body.truth(2) = 1.2;
    static_body.truth(8) = 0.20;
    static_body.target_id = 1;

    std::mt19937 rng(20241020);
    MultiTargetTracker multi{MultiTargetConfig{}, TrackerConfig{}};
    GimbalState gimbal{};
    double time_s = 0.0;
    const double dt = 1.0 / 60.0;
    for (int frame = 0; frame < 180; ++frame)
    {
        time_s += dt;
        spinning.truth = rotating_armor::transition(spinning.truth, dt);
        static_body.truth = rotating_armor::transition(static_body.truth, dt);
        // 云台盯住明显更近的那辆（自转车），另一辆靠 FOV 边缘也能看到。
        gimbal.yaw = 0.0;
        gimbal.pitch = 0.0;
        multi.step(observeBoth({spinning, static_body}, gimbal, rng), gimbal, time_s, 0.0);
    }
    check(multi.createdTracks() >= 2, "A 两辆车各自建出一辆");
    check(multi.createdTracks() <= 3, "A 没有为同一辆反复新建");
    check(nearestCenterError(multi, spinning.truth) < 0.20, "A 自转车车心误差 < 20 cm");
    check(nearestCenterError(multi, static_body.truth) < 0.20, "A 静止车车心误差 < 20 cm");
    check(nearestRadius(multi, spinning.truth) < 0.10, "A 自转车半径误差 < 10 cm");
    check(nearestRadius(multi, static_body.truth) < 0.10, "A 静止车半径误差 < 10 cm");
}

// B：只有一辆车，跑 2 秒，必须只有一辆。
void scenarioSingleBody()
{
    Body body;
    body.truth = State::Zero();
    body.truth(0) = 3.0;
    body.truth(8) = 0.25;
    body.target_id = 7;

    std::mt19937 rng(7);
    MultiTargetTracker multi{MultiTargetConfig{}, TrackerConfig{}};
    GimbalState gimbal{};
    double time_s = 0.0;
    const double dt = 1.0 / 60.0;
    for (int frame = 0; frame < 120; ++frame)
    {
        time_s += dt;
        body.truth = rotating_armor::transition(body.truth, dt);
        multi.step(observeBoth({body}, gimbal, rng), gimbal, time_s, 0.0);
    }
    check(multi.createdTracks() == 1, "B 单车只建一辆");
    check(multi.tracks().size() == 1, "B 单车末态仍在跟踪一辆");
    check(nearestCenterError(multi, body.truth) < 0.20, "B 车心误差 < 20 cm");
}

} // namespace

int main()
{
    scenarioTwoBodies();
    scenarioSingleBody();
    std::printf("%s 失败 %d 项\n", g_failures == 0 ? "全部通过," : "有", g_failures);
    return g_failures == 0 ? 0 : 1;
}
