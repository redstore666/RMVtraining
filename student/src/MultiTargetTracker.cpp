#include "MultiTargetTracker.hpp"

#include "AimGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rotating_armor
{

namespace
{

constexpr double kInfinity = std::numeric_limits<double>::infinity();

// 观测与一辆车的"匹配代价"。
//
// 已锁定的车：板永远在半径 R 上，所以用 | |p-c| - R |（径向一致性）当代价——
// 它对目标的转角不敏感，不需要准确的 omega/相位，四块板里哪一块看见都行，
// 这是多目标关联里最稳的一个量。
// 还没锁定的车：只有"上次观测到的那块板"可用，就用直线距离。
double associationCost(const MultiTargetTrack& track, const Eigen::Vector3d& position)
{
    if (!track.tracker || !track.tracker->initialized())
    {
        return (position - track.last_position).norm();
    }
    const State& state = track.tracker->ekf().state();
    const double radius = std::max(0.05, state(8));
    const double radial = std::hypot(position.x() - state(0), position.y() - state(2));
    return std::fabs(radial - radius);
}

} // namespace

MultiTargetTracker::MultiTargetTracker(MultiTargetConfig config, TrackerConfig tracker_config)
    : config_(config), tracker_config_(tracker_config)
{
}

void MultiTargetTracker::reset()
{
    tracks_.clear();
    pending_spawns_.clear();
    created_tracks_ = 0;
    next_track_id_ = 1;
    has_time_ = false;
    last_time_s_ = 0.0;
}

std::vector<MultiTargetTrack>& MultiTargetTracker::step(
    const std::vector<ArmorPose>& observations, const GimbalState& gimbal,
    double timestamp_seconds, double measured_latency_s)
{
    if (!std::isfinite(timestamp_seconds))
    {
        return tracks_;
    }
    const double dt = has_time_ ? timestamp_seconds - last_time_s_ : 0.0;
    if (has_time_ && (!(dt >= 0.0) || dt > 1.0))
    {
        // 时间倒流或长时间断帧：上一帧的"最近观测位置"已经不可信，当作重来。
        for (MultiTargetTrack& track : tracks_)
        {
            track.last_seen_s = -1.0e9;
            track.last_position.setZero();
        }
    }
    has_time_ = true;
    last_time_s_ = timestamp_seconds;

    const Eigen::Matrix3d world_from_gimbal = gimbalToWorld(gimbal);
    struct WorldObs
    {
        Eigen::Vector3d position{0.0, 0.0, 0.0};
        int source_index{0};
        int target_id{0};
    };
    std::vector<WorldObs> world;
    world.reserve(observations.size());
    for (std::size_t i = 0; i < observations.size(); ++i)
    {
        const ArmorPose& pose = observations[i];
        if (!std::isfinite(pose.position_gimbal_m[0]) ||
            !std::isfinite(pose.position_gimbal_m[1]) ||
            !std::isfinite(pose.position_gimbal_m[2]))
        {
            continue;
        }
        WorldObs item;
        item.position = world_from_gimbal * Eigen::Vector3d(pose.position_gimbal_m[0],
                                                            pose.position_gimbal_m[1],
                                                            pose.position_gimbal_m[2]);
        item.source_index = static_cast<int>(i);
        item.target_id = pose.detection.target_id;
        world.push_back(item);
    }

    // 1) 全局贪心分配：把所有 (观测, 车, 距离) 按距离升序排，先来先占。
    //    一台车一帧最多认领 3 块板（四块板在侧面最多同时看到 2-3 块）。
    struct Candidate
    {
        double distance;
        int obs_index;
        int track_index;
    };
    std::vector<Candidate> candidates;
    candidates.reserve(world.size() * std::max<std::size_t>(1, tracks_.size()));
    for (std::size_t o = 0; o < world.size(); ++o)
    {
        for (std::size_t t = 0; t < tracks_.size(); ++t)
        {
            const double distance = associationCost(tracks_[t], world[o].position);
            if (distance < config_.association_gate_m)
            {
                candidates.push_back({distance, static_cast<int>(o), static_cast<int>(t)});
            }
        }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b) { return a.distance < b.distance; });

    std::vector<int> claimed_by(world.size(), -1);
    std::vector<int> claims(tracks_.size(), 0);
    for (const Candidate& candidate : candidates)
    {
        if (claimed_by[static_cast<std::size_t>(candidate.obs_index)] >= 0)
        {
            continue;
        }
        if (claims[static_cast<std::size_t>(candidate.track_index)] >= 3)
        {
            continue;
        }
        claimed_by[static_cast<std::size_t>(candidate.obs_index)] = candidate.track_index;
        ++claims[static_cast<std::size_t>(candidate.track_index)];
    }

    // 2) 没被认领的观测：先补给"预测暂时不准"的已有车（离它的板不到 spawn 门），
    //    否则进"待建车"缓冲；同一个位置连续两帧都解释不了，才真的新建一辆。
    std::vector<int> extra_claims(tracks_.size(), 0);
    for (std::size_t o = 0; o < world.size(); ++o)
    {
        if (claimed_by[o] >= 0)
        {
            continue;
        }
        int best_track = -1;
        double best_distance = config_.spawn_gate_m;
        for (std::size_t t = 0; t < tracks_.size(); ++t)
        {
            const double distance = associationCost(tracks_[t], world[o].position);
            if (distance < best_distance)
            {
                best_distance = distance;
                best_track = static_cast<int>(t);
            }
        }
        if (best_track >= 0 && claims[static_cast<std::size_t>(best_track)] < 3)
        {
            claimed_by[o] = best_track;
            ++claims[static_cast<std::size_t>(best_track)];
            ++extra_claims[static_cast<std::size_t>(best_track)];
            continue;
        }

        // 待建车缓冲：同一处连续两帧都"谁都解释不了"，才当它是辆新车。
        PendingSpawn* pending = nullptr;
        for (PendingSpawn& candidate : pending_spawns_)
        {
            if ((candidate.position - world[o].position).norm() < 0.30)
            {
                pending = &candidate;
                break;
            }
        }
        if (pending == nullptr)
        {
            PendingSpawn candidate;
            candidate.position = world[o].position;
            candidate.target_id = world[o].target_id;
            candidate.hits = 1;
            candidate.last_update_s = timestamp_seconds;
            pending_spawns_.push_back(candidate);
            continue;
        }
        ++pending->hits;
        pending->position = world[o].position;
        pending->target_id = world[o].target_id;
        pending->last_update_s = timestamp_seconds;
        if (pending->hits < 2 || static_cast<int>(tracks_.size()) >= config_.max_targets)
        {
            continue;
        }
        MultiTargetTrack track;
        track.track_id = next_track_id_++;
        ++created_tracks_;
        track.created_s = timestamp_seconds;
        track.tracker = std::make_unique<RotatingArmorTracker>(tracker_config_);
        track.last_position = world[o].position;
        track.target_id = world[o].target_id;
        track.last_seen_s = timestamp_seconds;
        tracks_.push_back(std::move(track));
        const int new_index = static_cast<int>(tracks_.size()) - 1;
        claims.resize(tracks_.size(), 0);
        claims[static_cast<std::size_t>(new_index)] = 1;
        claimed_by[o] = new_index;
        // 同一簇的其它未认领观测一起交给新车，避免第二块板又造一辆。
        for (std::size_t other = 0; other < world.size(); ++other)
        {
            if (claimed_by[other] >= 0 || other == o)
            {
                continue;
            }
            if ((world[other].position - world[o].position).norm() < config_.spawn_gate_m &&
                claims[static_cast<std::size_t>(new_index)] < 4)
            {
                claimed_by[other] = new_index;
                ++claims[static_cast<std::size_t>(new_index)];
            }
        }
        pending_spawns_.erase(pending_spawns_.begin() +
                              (pending - pending_spawns_.data()));
    }
    // 太久没复现的待建点直接丢掉。
    pending_spawns_.erase(
        std::remove_if(pending_spawns_.begin(), pending_spawns_.end(),
                       [&](const PendingSpawn& candidate) {
                           return timestamp_seconds - candidate.last_update_s > 0.20;
                       }),
        pending_spawns_.end());
    (void)extra_claims;

    // 3) 每辆车只吃自己的观测，走原来的单目标链路。
    for (std::size_t t = 0; t < tracks_.size(); ++t)
    {
        MultiTargetTrack& track = tracks_[t];
        std::vector<ArmorPose> subset;
        for (std::size_t o = 0; o < world.size(); ++o)
        {
            if (claimed_by[o] == static_cast<int>(t))
            {
                subset.push_back(observations[static_cast<std::size_t>(world[o].source_index)]);
                track.last_position = world[o].position;
                track.target_id = world[o].target_id;
                track.last_seen_s = timestamp_seconds;
                ++track.observations;
            }
        }
        track.result = track.tracker->step(subset, gimbal, timestamp_seconds, measured_latency_s);
        track.diagnostics = track.tracker->diagnostics();
    }

    // 4) 注销：太久没观测的车直接删掉（把槽位留给新车）。
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](const MultiTargetTrack& track) {
                                     if (!(timestamp_seconds - track.last_seen_s >
                                           config_.prune_after_s))
                                     {
                                         return false;
                                     }
                                     // 从未观测过的车（刚建出来）至少留一帧。
                                     return track.observations > 0;
                                 }),
                  tracks_.end());

    // 5) 合并：两辆车的跟踪中心相距 merge_gate 之内，说明是同一辆车被分成了两簇。
    //    一次扫描合并所有对，保留观测更多的那辆（回到循环头重来，直到没有可合并的）。
    bool merged_any = true;
    while (merged_any)
    {
        merged_any = false;
        for (std::size_t i = 0; i < tracks_.size() && !merged_any; ++i)
        {
            for (std::size_t j = i + 1; j < tracks_.size(); ++j)
            {
                const bool both_tracking =
                    tracks_[i].diagnostics.phase == LockPhase::Tracking &&
                    tracks_[j].diagnostics.phase == LockPhase::Tracking;
                if (config_.merge_when_tracking_only && !both_tracking)
                {
                    continue;
                }
                if (!tracks_[i].tracker->initialized() || !tracks_[j].tracker->initialized())
                {
                    continue;
                }
                const State& a = tracks_[i].tracker->ekf().state();
                const State& b = tracks_[j].tracker->ekf().state();
                const double distance = std::hypot(a(0) - b(0), a(2) - b(2));
                if (distance < config_.merge_gate_m)
                {
                    const std::size_t drop =
                        tracks_[i].observations >= tracks_[j].observations ? j : i;
                    tracks_.erase(tracks_.begin() + static_cast<std::ptrdiff_t>(drop));
                    merged_any = true;
                    break;
                }
            }
        }
    }
    return tracks_;
}

} // namespace rotating_armor
