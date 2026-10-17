// 任务 4 的管线测试：观测 -> ekf_predict -> applyPrediction -> 通信结构体。
// 验收重点是"安全输出"不变量（任务书 §4/§7、验收与评分"安全约束"）：
//   * 无目标 / 未稳定 / 结果非法时，armor_flag、distance、coo_x、coo_y、
//     yaw_vel、pitch_vel、fire_allowance 必须全零，crc=0；
//   * 只有 Stable 才可能出现 fire_allowance=1；
//   * 任何 float 字段都不得出现 NaN/Inf；
//   * center_x/y 以米给出、通信协议里是毫米。
#include "AppSupport.hpp"
#include "StudentTasks.hpp"
#include "tracker_test_support.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

namespace
{
int g_failures = 0;

void check(bool condition, const char *what)
{
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what);
    if (!condition)
    {
        ++g_failures;
    }
}

bool frameFieldsFinite(const MessData_AutoAim &message)
{
    return std::isfinite(message.yaw) && std::isfinite(message.pitch) &&
           std::isfinite(message.roll) && std::isfinite(message.bias) &&
           std::isfinite(message.distance) && std::isfinite(message.coo_x) &&
           std::isfinite(message.coo_y) && std::isfinite(message.target_yaw) &&
           std::isfinite(message.target_pitch) && std::isfinite(message.yaw_vel) &&
           std::isfinite(message.pitch_vel);
}

bool isSafeFrame(const MessData_AutoAim &message)
{
    return message.head == kFrameHead && message.tail == kFrameTail && message.crc == 0U &&
           message.armor_flag == 0U && message.fire_allowance == 0U &&
           message.distance == 0.0F && message.coo_x == 0.0F && message.coo_y == 0.0F &&
           message.yaw_vel == 0.0F && message.pitch_vel == 0.0F;
}

} // namespace

int main()
{
    using rotating_armor::State;
    using TrackingState = ::TrackingState;
    using PredictionResult = ::PredictionResult;

    Translator translator{};
    translator.message.head = kFrameHead;
    translator.message.tail = kFrameTail;
    translator.message.status = 0; // 我方红：敌方蓝
    translator.message.bias = 0.0F;

    State truth;
    truth << 3.00, 0.05, 0.00, 0.02, 0.35, 0.0, 0.30, 2.50, 0.25;

    sim::GimbalAim gimbal;
    gimbal.yaw = rotating_armor::vectorYaw(Eigen::Vector3d(3.0, 0.0, 0.0));
    gimbal.pitch = rotating_armor::vectorPitch(Eigen::Vector3d(3.0, 0.0, 0.35));
    gimbal.rate_rad_s = 0.0;

    std::mt19937 rng(20240607U);
    const double dt = 1.0 / 60.0;
    double time_s = 0.0;

    int crc_counts[3] = {0, 0, 0};
    int stable_frames = 0;
    int fire_frames = 0;
    int first_stable = -1;
    bool finite_always = true;
    bool fire_only_when_stable = true;
    bool safe_when_invalid = true;
    bool coo_matches_prediction = true;
    bool armor_flag_matches = true;

    for (int frame = 0; frame < 330; ++frame)
    {
        time_s += dt;
        truth = rotating_armor::transition(truth, dt);
        const std::vector<sim::Plate> plates = sim::platesAt(truth, 4);

        // main.cpp 的顺序：先把手里的云台数据放进消息，再取云台状态。
        const GimbalState gimbal_state = gimbal.state(0.0);
        translator.message.yaw = gimbal_state.yaw;
        translator.message.pitch = gimbal_state.pitch;
        translator.message.roll = gimbal_state.roll;
        translator.message.bias = gimbal_state.prediction_bias_s;

        // 200~240 帧模拟短时遮挡：观测为空，但 ekf_predict 仍必须被调用。
        const bool occluded = (frame >= 200 && frame < 240);
        std::vector<ArmorPose> observations;
        if (!occluded)
        {
            observations = sim::observe(plates, gimbal_state, 7, rng);
        }

        const PredictionResult prediction =
            ekf_predict(observations, gimbalStateFromMessage(translator.message), time_s);
        const bool applied = applyPrediction(translator, prediction);
        const MessData_AutoAim &message = translator.message;

        finite_always = finite_always && frameFieldsFinite(message);
        fire_only_when_stable = fire_only_when_stable && !(message.fire_allowance != 0U && message.crc != 1U);
        if (!applied)
        {
            safe_when_invalid = safe_when_invalid && isSafeFrame(message);
        }
        else
        {
            coo_matches_prediction =
                coo_matches_prediction &&
                std::fabs(message.coo_x - prediction.center_x_m * 1000.0F) <= 1.0F &&
                std::fabs(message.coo_y - prediction.center_y_m * 1000.0F) <= 1.0F;
            armor_flag_matches = armor_flag_matches && message.armor_flag == prediction.target_id;
        }
        if (message.crc <= 2U)
        {
            ++crc_counts[message.crc];
        }
        if (message.crc == 1U)
        {
            ++stable_frames;
            if (first_stable < 0)
            {
                first_stable = frame;
            }
        }
        if (message.fire_allowance != 0U)
        {
            ++fire_frames;
        }
        if (frame == 0 || frame == 5 || frame == 60 || frame == 199 || frame == 239 ||
            frame == 300)
        {
            std::printf("帧%3d crc=%u fire=%u armor_flag=%u coo=(%.3f, %.3f) 距离=%.3f "
                        "target_yaw=%.3f\n",
                        frame, static_cast<unsigned>(message.crc),
                        static_cast<unsigned>(message.fire_allowance),
                        static_cast<unsigned>(message.armor_flag),
                        static_cast<double>(message.coo_x), static_cast<double>(message.coo_y),
                        static_cast<double>(message.distance), static_cast<double>(message.target_yaw));
        }
    }

    std::printf("\ncrc 计数: 0=%d 1=%d 2=%d 稳定帧=%d 开火帧=%d 首次稳定帧=%d\n", crc_counts[0],
                crc_counts[1], crc_counts[2], stable_frames, fire_frames, first_stable);

    check(finite_always, "所有帧的 float 字段有限（无 NaN/Inf 进入协议）");
    check(safe_when_invalid, "预测非法时的帧是完整的安全输出");
    check(fire_only_when_stable, "fire_allowance=1 只出现在 crc=1（Stable）");
    check(coo_matches_prediction, "center_x/y 米 -> 毫米 的换算一致");
    check(armor_flag_matches, "armor_flag 等于预测的目标编号");
    check(first_stable > 0, "链路最终进入 crc=1");
    check(stable_frames >= 150, "稳定（crc=1）帧数 >= 150");
    check(crc_counts[0] >= 5, "早期未锁定时输出 crc=0");

    if (g_failures != 0)
    {
        std::printf("\n%d 项失败\n", g_failures);
        return 1;
    }
    std::printf("\n全部通过\n");
    return 0;
}
