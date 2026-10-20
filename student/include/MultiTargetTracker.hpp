#pragma once

// 多目标跟踪：把一帧里的全部装甲板分成若干"车"，每辆车各跑一个
// RotatingArmorTracker。
//
// 为什么需要它：原来的 RotatingArmorTracker 只维护一个 target_id，遇到
// 两台车同时入画时只能在两者之间选一个（v4 回放里"只锁住右边静止的那台，
// 左边在动的迟迟进不来"就是这么来的）。而 target_id 本身是**板号**（1..4），
// 不是车的身份——同一辆车会先后报 1/2/3/4，两台车也可能同时报同一个号，
// 所以不能按 id 分车，只能按几何分组：
//
//   * 同一辆车的四块板都落在半径 R 的圆上，彼此距离 <= 2R（相邻 R*sqrt2），
//     所以世界系里"相距 0.5 m 之内的一簇板"就是一辆车；
//   * 每辆车用自己的 EKF 预测下一帧的板位，观测按"离哪辆车的预测板最近"分配；
//     预测暂时不可用（还在 Collecting）时退回"离这辆车最近一次观测到的位置"。
//
// 这个类只做分组与生命周期（新建/注销/合并），滤波、锁定、状态机全部复用
// RotatingArmorTracker，因此单目标与多目标的行为完全一致。

#include "RotatingArmorTracker.hpp"

#include <memory>
#include <vector>

namespace rotating_armor
{

struct MultiTargetConfig
{
    // 最多同时跟踪几辆车（比赛里同场通常 <=2，留 4 做余量）。
    int max_targets{4};
    // 观测分配给某辆车的位置门：观测到该车任一预测板的距离小于它即认领。
    // 四块板相邻间距 R*sqrt2 ≈ 0.4 m，门限不宜大于半个间距，否则两辆车
    // 靠得近时会把对方的板吸过来。
    double association_gate_m{0.45};
    // 新建一辆车之前，先看它离已有车辆的"最近观测位置"多远；小于它就当成
    // 那辆车（预测暂时不准时不要重复开一辆新车）。
    double spawn_gate_m{0.60};
    // 多久没有观测就注销这辆车。
    double prune_after_s{0.80};
    // 两个跟踪中心的距离小于它就合并（保留更新次数多的那个）。
    double merge_gate_m{0.60};
    // 合并/注销只在两车都进入 Tracking 之后做，避免 Collecting 期间中心
    // 还没收敛时误合并。
    bool merge_when_tracking_only{false};
};

// 一辆车的全部对外状态。tracker 用 unique_ptr 是为了让 vector 可增删，
// 同时不给 RotatingArmorTracker 增加拷贝语义。
struct MultiTargetTrack
{
    int track_id{0};              // 本类分配的序号（1 起），只用于显示
    int target_id{0};             // 最近一帧观测到的板号（1..4，0 = 未知）
    std::unique_ptr<RotatingArmorTracker> tracker;
    PredictionResult result;      // 本帧输出（与 tracker->step 一致）
    TrackerDiagnostics diagnostics;
    double last_seen_s{-1.0e9};   // 最近一次拿到观测的时间
    double created_s{0.0};
    int observations{0};          // 累计认领的观测数
    Eigen::Vector3d last_position{0.0, 0.0, 0.0}; // 最近一次认领的观测位置
};

class MultiTargetTracker
{
public:
    explicit MultiTargetTracker(MultiTargetConfig config = {},
                                TrackerConfig tracker_config = TrackerConfig{});

    // 与 RotatingArmorTracker::step 同签名：一帧里所有车的全部装甲板位姿。
    // 返回的 vector 长度可变（新建/注销），顺序稳定（按 track_id）。
    std::vector<MultiTargetTrack>& step(const std::vector<ArmorPose>& observations,
                                        const GimbalState& gimbal,
                                        double timestamp_seconds,
                                        double measured_latency_s = 0.0);

    const std::vector<MultiTargetTrack>& tracks() const { return tracks_; }

    // 累计新建过多少辆（含已注销的）。和 tracks().size() 一起看能区分
    // "同时在跟踪两辆"和"同一辆被反复重开"。
    int createdTracks() const { return created_tracks_; }
    std::vector<MultiTargetTrack>& tracks() { return tracks_; }
    const MultiTargetConfig& config() const { return config_; }
    void reset();

private:
    MultiTargetConfig config_;
    TrackerConfig tracker_config_;
    // 新建车要把"同一簇"的观测一起给它，所以新车的种子点先压两帧：
    // 单帧野点（误检、另一块板的跳变）不会凭空造出一辆幽灵车。
    struct PendingSpawn
    {
        Eigen::Vector3d position{Eigen::Vector3d::Zero()};
        int target_id{0};
        int hits{0};
        double last_update_s{-1.0e9};
    };

    std::vector<MultiTargetTrack> tracks_;
    std::vector<PendingSpawn> pending_spawns_;
    int created_tracks_{0};
    int next_track_id_{1};
    bool has_time_{false};
    double last_time_s_{0.0};
};

} // namespace rotating_armor
