#pragma once

// 旋转车体目标跟踪器（任务 4 的粘合层 + 关联 + 状态机 + 输出映射）。
//
// 分工：
//   RotatingArmorModel   —— 唯一一份运动模型 / 观测模型公式（自由函数）
//   RotatingArmorEkf     —— 单次 predict / update、初始化契约、NIS
//   RotatingArmorTracker —— 本文件：观测坐标系变换、装甲板槽位关联、
//                            离群点拒绝、锁定与重新锁定、超时注销、
//                            稳定/不稳定状态机、PredictionResult 输出
//
// 为什么锁定要攒多帧：单帧的 3D 板心 + 板面法向只给出"相位 mod pi"，
// "车体在板的哪一侧"是另一个自由度（等价于半径的符号）。所以锁定阶段用
// 最近若干帧做一次最小二乘：
//     p_k = c + v * (t_k - t_last) - R * [cos(phi_k), sin(phi_k)]
// 未知量 (cx, cy, vx, vy, R)，phi_k 由板面法向给出（mod pi 的两个候选取
// 拟合半径 > 0 的那个）。相位扫过角太小（目标不转）时 R 不可辨识，此时退回
// "车体在板背后"的几何先验（相机只能看到板正面），半径用配置先验。

#include "AimGeometry.hpp"
#include "AutoAimTypes.hpp"
#include "RotatingArmorEkf.hpp"

#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <string>
#include <vector>

namespace rotating_armor
{

// 跟踪系（世界系）里某块装甲板的中心位置。
// 薄封装 measurement()：让上层不必关心观测模型的公式，也保证只有一份真源。
Eigen::Vector3d platePosition(const State& state, int plate_slot, int plate_count);

struct TrackerConfig
{
    EkfConfig ekf{};

    // === 观测 ===
    // 位置噪声（米）。单帧 PnP 的板心抖动实测约 1cm，这里取 3cm 留出
    // "倾斜角/四角点抖动"的余量；重投影误差大的观测再按比例放大。
    double position_sigma_m{0.03};
    // 车体平动速度的先验：单板可见时"板在动"和"车在动"不可分，把速度放开
    // 会让滤波器把旋转解释成平移、中心以 1 m/s 飞出去（实拍 v4 旋转目标
    // 中心跑 1.4 m 后位置门全拒）。所以速度先验压到很小、基本钉在 0。
    double velocity_sigma_m_s{0.05};
    // 过程噪声缩放：默认 1.0 = ProcessNoiseConfig 的原值。自转目标的板心每帧能走
    // 30-50 mm，滤波器 P 收得太紧时会"拒跟"观测（实测状态离刚更新的观测 87 mm）。
    double process_accel_scale{1.0};
    // 板面法向 yaw 的噪声（弧度）。IPPE 孪生解会让 yaw 差 10~30°，取 0.35。
    double normal_sigma_rad{0.35};

    // === 转速窗口估计（"到底转不转"由数据决定，而不是由调用方声明）===
    //
    // 实拍 v1/v4 的核心教训：166 fps 下 omega 与观测噪声是混叠的。10 rad/s × R=0.3
    // 的切向位移只有 2 cm/帧，和 PnP 噪声同量级，所以"逐帧位置更新的自由积分"
    // 可以把静止目标推成 10 rad/s（NIS 依然只有 2，滤波器认为完全自洽）。
    // 反过来，板面法向在 0.3 s 窗口上会累积 60° 的真实转角，远大于法向噪声。
    // 因此这里维护一个窗口，用最小二乘斜率作为 omega 的伪量测送进 EKF：
    //   * 窗口说"在转" -> omega 跟着数据走；
    //   * 窗口说"没转"   -> omega 被拉回 0，不需要 Static 声明。
    bool rate_update_enabled{true};
    double rate_window_s{0.35};        // 拟合窗口（秒）
    double rate_min_span_s{0.18};      // 窗口至少要覆盖这么久
    int rate_min_samples{12};          // 至少要这么多帧
    double rate_max_gap_s{0.05};       // 相邻样本间隔超过它就清空窗口（掉帧/换目标）
    double rate_max_residual_rad{0.60};// 拟合残差超过它就认为法向序列不可信
    double rate_gate_m{0.60};
    // 转速窗口的关联门限：比位置门严得多。同一台车/相邻车的同编号板可能只差
    // 0.2~0.4 m（实拍 v4 的斜置副板），混进窗口会把相位序列扯成假高速旋转。
    double rate_associate_gate_m{0.15};          // 观测离预测板超过它就不算"自己这台车"
    double rate_sigma_floor_rad_s{0.15};
    // 反证据：窗口超过这么久没给出有效斜率，就喂一个"没在转"的弱伪量测
    // （omega -> 0，sigma 见下）。没有这一条时 omega 会被位置更新经由协方差
    // 耦合慢慢推到 -6 rad/s（回放 v4 实测），模型整块板跟着自转——正是用户
    // 看到的"圆心偏了"。数据支持不了旋转时，停在 0 才是安全解。
    double rate_decay_after_s{0.60};
    // 衰减项是一次"omega=0"的伪量测，sigma 取大一点：它只负责在长时间没有
    // 任何斜率证据时把 omega 慢慢拉回 0，压不过一次真实量测（sigma 0.15 量级）。
    double rate_decay_sigma_rad_s{2.00};
    // 斜率标准误上限：超过这个值说明这一窗数据根本说明不了转速（v4 静态目标
    // 上法向乱翻，MAD 给出 ±2-3 rad/s），宁可不用这一帧，让衰减项把 omega
    // 留在 0，也不要把噪声当证据。
    // 2026-10-23 接上这条闸门：先用 2.0，v4 左车 178/201 个窗口被拒（bearing
    // 序列 MAD 常在 3~5 rad/s），omega 直接被衰减项压成 0，等于"车不转了"；
    // 4.0 仍拒掉 174 窗口；6.0 只拦真正乱翻的窗口，剩下的交给 EKF 按 sigma 加权。
    double rate_max_sigma_rad_s{6.00};
    // 转速的物理闸门：四板车实测最快 8 rad/s（G 组），取 1.5 倍余量。
    // 超过它的斜率只可能来自"续接跳到别的板 / 极点选错 / 折叠算术出错"，
    // 以前这些伪转速（v4 实测 +19~+34 rad/s）会被原样喂进 EKF，把 omega
    // 打成 ±10 的方波。0 表示关闭闸门。
    double rate_max_abs_rad_s{12.00};
    // 两路证据（法向斜率 vs 板相对车心的方位角斜率）允许的差；超过就当
    // "这一窗说法向不可信"，直接不更新。
    double rate_agreement_rad_s{1.50};
    // 弦向转速证据：同帧两块同号板的"弦方向"随车体严格同转，不依赖极点/车心、
    // 不受平移影响（v4 实拍 7.85 rad/s 自转车：209 个双板帧、弦长稳定 0.40m、
    // 0.35s 窗内斜率干净）。它是打破自举死锁的钥匙：omega 卡在 0 时
    // spinning_now=false，方位角路只能用逐帧抖动的伪极点，斜率是垃圾且被
    // 物理闸门拒掉，omega 永远起不来。板序互换造成的 180° 翻转按 π 折叠消除。
    bool rate_from_chord_enabled{true};
    double rate_chord_min_m{0.15};
    double rate_chord_max_m{0.65};
    // 原始法向路线（3f）：全部同号板法向、跨帧全配对、逐对折叠（fold ±π）
    // Theil-Sen。mod-90° 折叠把板间推进抹成 0、链式解缠被噪声随机游走——
    // 逐对折叠没有这两类失效。前向区检测（位置/弦向/方位角全部读 ≈0）时
    // 唯一稳定的转速来源。窗口比转速窗口长：法向噪声大，靠配对数量压方差。
    bool rate_from_raw_normal_enabled{true};
    double rate_raw_window_s{0.70};
    // 转速自检：板心到极点的距离与当前半径 R 差超过它的样本过半，或 R 本身
    // 掉出 [radius_plausible_min_m, radius_plausible_max_m] 时，这一窗的方位角
    // 斜率必然是伪值（实测 R 崩到 0.152 时 bearing 斜率冲到 +8~+10，全靠物理
    // 闸门兜底）——整窗放弃，宁可让衰减项慢慢接管，也不要拿坏几何估转速。
    double rate_pole_residual_m{0.10};
    // 非自转（|omega|<0.5）时方位角用"每帧当时车心"当极点：纯平移下真方位角
    // 是常数，斜率只可能来自车心估计误差的漂移，而误差漂移与车速同源
    // （H3 平移 0.5 m/s 实测 bear 斜率稳定 +0.4~+0.7，omega 被推到 +0.51）。
    // 车心速度超过它就不信方位角路，退回法向斜率；自转时极点是同一个点
    // （当前 EKF 车心），平移是共模量，不受此限。
    double rate_bearing_max_centre_speed_m_s{0.30};

    // === 锁定 ===
    int lock_min_samples{8};         // 至少攒到的帧数（只收带法向的观测）
    double lock_min_span_s{0.18};     // 至少覆盖的时间跨度（短了圆拟合与转速都不可辨）
    double lock_max_age_s{0.50};      // 更早的样本丢掉（目标可能已经换了）
    double radius_prior_m{0.25};      // 无板型信息时的兜底先验：实车按底盘实测
    // 按板型给的半径先验（车体中心到装甲板面的距离）。实测 v4 的 3 号小装甲车
    // 是 0.197 m，用 0.20 比 0.25 的残差中位从 114 mm 降到 87 mm。
    double radius_prior_small_m{0.20};
    double radius_prior_large_m{0.25};
    // 是否按板型选半径先验。关掉（或用 --radius 显式给值）时一律用 radius_prior_m。
    bool radius_prior_from_size{true};
    double radius_sigma_m{0.10};
    // 锁定后是否把半径钉在锁定值：单板可见时 (中心, 半径) 退化，位置更新会
    // 把 R 一点点拉走（实测 0.25 -> 0.147，中心随之跑 1.4 m）。实时跟踪按
    // "半径由板型先验给出"处理，默认钉死；合成/离线场景想测半径收敛可以关掉。
    bool freeze_radius{true};
    double radius_min_m{0.05};
    double radius_max_m{0.60};
    // 半径写入限速：刚性车的 R 物理恒定，弦长证据里混着"相邻/对置构型误判"
    // 的双峰噪声（两簇差 √2 倍，rotate.avi 实测 R 在 0.20~0.385 摆、帧间最大
    // 跳 51mm），低通压不住，俯视图的板圆周会"呼吸"。applySlowRadius 每次写入
    // 相对当前 R 最多走这么多（米/调用），显示端即稳；0 = 关闭。0.5mm/帧
    // @165fps ≈ 8cm/s，先验收径到实测值约 1s 收敛。
    double radius_max_step_m{0.0005};
    double phase_spread_for_radius_rad{0.30}; // 相位扫过小于它 -> 半径不可辨识
    // 位置圆拟合锁定：高速自转时单帧板面 yaw 的抖动（实拍 ±0.3 rad）比单帧
    // 真实转角还大，yaw 序列拟合不出转速；板心位置却准（重投影 0.5 px），
    // 四块板共圆 => 圆拟合给 (c, R)，相邻板心的切向位移/半径给 omega。
    double lock_circle_residual_m{0.05};              // 圆拟合 RMS 残差超过它不可信
    double lock_circle_max_step_ratio{0.25};          // 单步位移 > 0.25R 视为换板，跳过
    double lock_circle_trusted_ratio{0.35};
    double lock_circle_min_span_rad{1.2};           // 有效步数占比低于它不采信位置速率
    double static_rate_rad_s{1.2};            // |omega| 低于它判为"不转"（Static）
    // 运动类型诊断（TrackerDiagnostics::motion）的平移门限：车心速度超过它
    // 且转速不够大时报 Translating。只影响诊断显示，不参与任何滤波。
    double motion_translating_speed_m_s{0.30};

    // === 关联 / 门限 ===
    double position_gate_m{0.60}; // 单板位置门限（含中心先验不确定的余量）
    // 新息门限。量测已改成 4 维（yaw/pitch/distance + 板法向角），
    // chi2(4) 的 99.5% 分位是 14.86；没有板法向时第 4 维方差为 1e9、
    // 贡献为 0，等价于 chi2(3)，所以取 14.86 是"最多 4 维"的保守上限。
    double nis_gate{14.86};
    double handoff_sigma_scale{2.5}; // 已废弃（仅保留字段）：换板不再放大 sigma
    // 观测标准差的地板：单板预测的误差里有一块是"模型误差"（半径/相位/中心
    // 的系统性偏差），不体现在 P 里。地板太小 → NIS 门把有信息的帧全拒了
    // （实拍 v4 旋转目标 466/466 帧被拒）；地板太大 → 更新变钝。
    double association_sigma_floor_m{0.06};
    // 观测噪声自适应：用"连续三帧观测的二阶差分"估当前目标的检测抖动。
    // 固定地板（0.06 m）对静止车大了一个量级（实测帧间抖动 4 mm），滤波器
    // 于是几乎不信观测（NIS 中位 0.03，理论应 ~2.4）；对高速自转车 0.06 m
    // 又略小。改成按实测抖动给 sigma，静止车更贴合、自转车不被夸大。
    bool adaptive_observation_noise{true};
    int adaptive_noise_window{24};
    double adaptive_noise_min_sigma_m{0.004};
    double adaptive_noise_max_sigma_m{0.090};
    // 同帧两块板的中点直接给"车体中心"量测：对置板中点恰是圆心（误差 ~sigma/sqrt2），
    // 相邻板中点沿弦法向偏 R*cos45。半径已知时这块观测比单板位置强得多。
    bool pair_center_update_enabled{true};
    double pair_center_sigma_m{0.05};
    double pair_center_gate_m{0.45};
    int pair_center_min_static_frames{10};

    // 观测平滑：同一槽位连续命中时，对世界系位置做指数平均后再喂滤波器。
    // 单帧检测抖动（实测 v4 旋转目标相邻帧位移中位 52 mm、静止 3.8 mm）是
    // 白噪声，平均 N 帧能压到 1/sqrt(N)；换板/换 id/超时立即重置，避免把
    // 不同板的点混在一起（那会引入 0.4 m 假跳变）。
    bool observation_smoothing_enabled{false};
    int smoothing_window{4};            // 参与平均的最大帧数
    double smoothing_max_spread_m{0.12}; // 新观测离均值超过它 -> 重置而不是并入
    double smoothing_max_gap_s{0.10};    // 与上次命中的间隔超过它 -> 重置

    // 同帧两块板同时可见时，弦长能直接量出半径：R = chord / |n_i - n_j|
    // （相邻两板 |Δn| = √2）。静止目标只露一块板时 R 不可辨识，这条观测是
    // 唯一能让 R 脱离先验的来源。
    // 默认打开：静止目标逐帧观测抖动 ~5 cm（166 fps 下相当于 8 m/s），不平均
    // 时状态只能追噪声；平均 4 帧把白噪声压到 1/2，代价是 ~2 帧滞后。
    // 历史实测（σ=0.02 的合成数据）对静止目标无改善（本来就 7.4 mm），
    // 对旋转目标反而变差（99 -> 125~140 mm），v1 只从 412 降到 380 mm。
    // 原因是锁定时已经用双板弦长定过 R（reinitializeFromPair），运行期再连续
    // 拉 R 会跟位置更新抢"中心该往哪挪"，旋转目标的 yaw 噪声把 R 拖偏。
    // 用同帧两块板（弦长 + 法向差）持续修正半径。默认打开：实拍 v4 的 3 号
    // 道具半径 0.28，按板型先验 0.20 钉死时模型四块板整体小一圈，第二块观测
    // 到模型板的中位误差 12.4 cm；打开后降到 6.8 cm。自由半径更新（freeze_radius
    // = false，位置残差驱动）仍然默认关闭——那个才会塌。
    // 双板整车求解：同帧看到两块相邻板时，板心位置 + 板面法向给出一个精确的
    // (车心 c, 半径 R, 车体相位 yaw) 闭式解，不依赖半径先验、也不依赖当前状态。
    //
    // 推导（四板，相邻相差 period = 90 度，p_i = c - R u(theta_i)）：
    //   d = p_b - p_a = -R (u(theta_b) - u(theta_a))
    //   相邻 +90 度：u(theta+90) - u(theta) = sqrt(2) u(theta+135)
    //                => theta_a = atan2(-d) - 135 度
    //   相邻 -90 度：u(theta-90) - u(theta) = -sqrt(2) u(theta+45)
    //                => theta_a = atan2(d) - 45 度
    //   半径 R = |d| / sqrt(2)；车心 c = p_a + R u(theta_a)。
    // 这就是"整车建模"的直接体现：一台车只有一个 c、一个 R、一个 yaw，
    // 四块板绕它整体转动，速度和方向天然一致，不存在单板各自漂移。
    // 同帧两块板的联合绑定：两块板是同一台车上相邻的两个面，槽位差由几何决定，
    // 只有 4 种整体旋转可选。逐块独立"找最近的预测板"会绑到互相矛盾的槽位上，
    // 两台更新互相拉扯——这正是"转一部分卡回去"的机制之一。
    // 槽位选择时把板面法向算进代价（单位 m/rad）：四块板同模、相隔 90 度，
    // 位置门对相邻两块板同时放行，光看位置分不出这一帧看到的是哪一面，
    // 分错就等于画面上模型板停在相邻那一面上（相位差 90 度）。
    // 关联按"角度"打分（借鉴 sp_vision_25 target.cpp:140-169）：
    // 四块板彼此相差 90 度、位置门又宽，只比 3-D 距离在原理上就分不清"看到的是
    // 哪一面"。sp_vision 的做法是先按距离取最近 3 块，再用
    //     |板法向角差| + |方位角差|
    // 选出真正的那一块。这里把这套打分接进候选评分：
    //     score = 距离 + association_angle_weight_m_per_rad * (法向差 + 方位差)
    // 加距离项是刻意的偏离：sp_vision 是"距离筛完再纯按角度挑"，我们没有两趟
    // 筛选，所以用距离项兜住"角度碰巧对上但隔了半米"的错配。
    // 权重单位是 m/rad：0.30 表示 1 rad 的角度差等价于 0.30 m 的位置差。
    // 板面法向进量测（借鉴 sp_vision_25）。默认关：PnP 法向噪声实测 ±35°，
    // 直接当量测喂进去在 v4 上反而变差（84.3 -> 86.7~88.8 px，见 docs）。
    bool plate_normal_measurement_enabled{false};
    double plate_normal_sign{1.0};
    bool association_by_angle_enabled{true};
    double association_angle_weight_m_per_rad{0.30};
    double normal_tiebreak_weight{0.0};
    bool joint_pair_binding_enabled{true};
    double joint_pair_binding_max_cost_m{0.80};
    // ===== S2：板身份精确后验（单射配对枚举）=====
    // 【2026-10-25 参考对齐】默认改成**开**。原默认关是因为角度通道 σ=2.5 时
    // 身份判错也拽不动状态（量测基本被忽略），后验只多花 CPU。σ 收到参考刻度
    // 0.31 之后，槽位判错就是一次 ±90° 的硬拽，必须靠整帧最优配对兜住。
    // 实测（q=4、σ=0.31）：身份关 v4 78.52/v1 36.38，身份开 86.85/51.28。
    // 详细机制：同帧 ≥2 块同编号板时不再"每个观测各自贪心 argmin 判
    // 槽位"，而是穷举全部单射配对（N=4、k≤4 时最多 24 种），取 Σ新息NIS 最小
    // 的一组作为**整帧**配对；同时用 softmax 记录最优/次优的后验与 margin。
    // 单板帧（身份无歧义）硬旁路，走原路径。
    // 详见 docs/架构重构-板身份多假设方案_2026-10-25.md 与
    // docs/身份后验-从环序假设到精确枚举_2026-10-25.md。
    bool identity_posterior_enabled{true};
    bool pair_body_solve_enabled{false};
    // 只在慢转/平移时用双板闭式解：高速自转（>=3 rad/s）时同一帧两块板的
    // 曝光不是同一瞬间，弦长和法向差都会被拖尾污染，闭式解反而会把相位推歪。
    double pair_body_max_rate_rad_s{3.0};
    double pair_body_max_phase_error_rad{0.35};
    double pair_body_center_gain{0.35};
    double pair_body_yaw_gain{0.5};
    double pair_body_radius_gain{0.0};
    bool pair_radius_update_enabled{true};
    double radius_update_sigma_m{0.06};
    double radius_update_alpha{0.25}; // freeze_radius 时的低通系数（每帧最多靠拢 1/4）
    double radius_update_min_chord_m{0.08}; // 太近的两块板（法向几乎平行）弦长除噪会爆
    // 半径更新只认两种构型：法向差 ~90°（相邻，弦 = √2·R）或 ~180°（对置，弦 = 2R）。
    // 阈值取在两者中间偏实测噪声：< 50° 当同一块板重复检出丢掉。
    double radius_update_adjacent_yaw_rad{0.87};  // 50°
    double radius_update_opposite_yaw_rad{2.27};  // 130°

    // 板面法向当"相位观测"塞进滤波器（折叠掉 2pi/N）。实拍小目标法向抖 0.05~0.15
    // rad，取 0.12；关掉它就退回"yaw 只靠位置间接约束"。
    // 默认关闭：合成 A 场景实测"打开后 yaw 抖动、半径被压到 0.18（真值 0.25）"，
    // 因为位置更新与相位更新在"哪个槽位"这件事上互相拉扯。能力保留、单测覆盖
    // RotatingArmorEkf::updatePhase 本身；等有"按槽位一致"的写法再默认打开。
    bool phase_update_enabled{false};
    // 静止目标（|omega| < static_rate_rad_s）且同一块板连续可见时，用板面法向
    // 约束车体 yaw。静止时法向没有旋转模糊，是干净量测；单板+钉死半径时
    // (center, yaw) 是规范自由度，不用法向相位就会自由漂移（实拍 v4 静止车
    // 1400 帧时相位偏 1.08 rad，另外三块模型板飞到车外）。
    bool phase_update_when_static{true};
    // 旋转目标上的"按槽位一致"相位更新：位置更新已经把这块观测分给了
    // best_slot，相位更新用同一个槽位（updatePhaseForSlot，不做 2pi/N 折叠），
    // 两者不再打架。折叠版只能修 45 度以内、永远改不了槽位分配，而位置更新
    // 能推过 45 度边界——实测 tmp/v4.avi 上 yaw 因此单帧跳 140~215 度，
    // 画面上就是"蓝板转一部分然后卡回去"。
    // 默认关闭：合成 A/G/H 全部回归（见 git log），只在"按槽位一致 + 双板
    // 共同约束"完整实现后才能打开。
        bool slot_from_normal_enabled{false};
    // 由板法向反推车心：c = p − R·n（true = 用，默认开）。
    bool center_from_normal_enabled{false};
    // 相位用板法向直接量测（default 关，--normal-phase 1 打开）
    // 本轮（2026-10-21）：深度次序硬约束 —— 由板法向直接定"车心在板的后面"。
    // 模型 plate = centre − R·u(phase)、外法向 = u(phase+π)，所以观测给出
    //   phase_slot = atan2(n) + π  =>  yaw = phase_slot − slot·(2π/N)
    //   centre     = p − R·n
    // 两者是同一个约束的两种写法，同时落地才能保证"蓝板就画在被看见的那块板上、
    // 且车心比它更远离相机"（用户反馈：蓝板/aim 一直落在远离 gimbal 的一端）。
    bool normal_anchor_enabled{false};
    double normal_anchor_gate_m{0.60};
    double normal_anchor_center_gain{1.0};
    bool circle_phase_enabled{false};
    double circle_phase_sigma_rad{0.12};
    // 多秒窗口联合批处理（default 关，--batch-fit 1 打开）
    bool batch_fit_enabled{false};
    double batch_fit_window_s{1.5};
    int batch_fit_every{2};
    int batch_fit_min_samples{12};
    double batch_fit_max_circle_residual_m{0.05};
    double batch_fit_max_phase_residual_rad{0.50};
    double batch_fit_min_phase_residual_rad{0.15};
    // 双板弦方向的相位量测（不吃 ω，见 docs）
    bool pair_phase_update_enabled{false};
    double pair_phase_sigma_rad{0.15};
    double batch_fit_min_coverage_rad{1.20};
    // 下面两个开关在理论上更正确，但在 v4 上实测更差（见 docs/任务进度备份.md）：
    // 打开后合成 A/G/H 的 ω 才能估准，所以保留给合成调试用。
    bool batch_fit_fold_phase{false};
    bool batch_fit_alias_gate{false};
    // 只施加"不足一块板"的相位差，保持槽位标号不跳（合成 G 需要）。
    bool batch_fit_slot_phase{false};
    // 用相邻两帧观测的切向位移直接估 ω：自转目标每帧沿切向走 |Δp|，
    // ω = (Δp · v̂)/R，v̂ 只依赖当前观测相对车心的角度，不依赖槽位。
    // EKF 自己的 ω 在单板自转时会衰减到 ~0（位置更新不断把相位拉回），
    // 导致模型落后真实旋转，这是 v4 相位残差的主要来源。
    // 同板轨迹段联合解 (c, ω)：相邻帧位移小于阈值的连续片段必属同一块板，
    // 给定 R 后只剩 3 个未知数，且角度沿片段连续、可无歧义展开 —— ω 因此可观测。
    // 这是 v4 上唯一有望修掉"每帧相位落后 0.16 rad"的结构性手段。
    // 观测连续性门（数据关联）：v4 里除机器人上的板之外还有一两块独立摆放的
    // "3"号板道具，位置只差 0.4-0.7 m，位置门（0.6 m）放不住它们，检测器会在
    // 它们之间来回跳（实测 61% 的相邻帧框中心跳 >25 px）。30 fps 下同一块板
    // 一秒内不可能跳 0.15 m，所以用"与上一次被接受的观测的距离"做硬门，
    // 把另一块道具永久排除，而不是靠预测门（预测本身就偏，收紧预测门反而更差）。
    // 每帧用最近窗口的板心圆 (c, R) 直接覆盖状态，相位/ω 不动。
    // 依据：v4 上把模型整体转任意 δ 都救不回来（残差恒 9cm），说明错的是圆不是相位；
    // 而同一段数据的窗口圆拟合残差只有 0.011-0.014 m。
    bool body_circle_enabled{false};
    double body_circle_window_s{1.2};
    int body_circle_min_samples{24};
    double body_circle_max_residual_m{0.05};
    bool obs_continuity_gate_enabled{false};
    double obs_continuity_gate_m{0.15};
    bool tracklet_rate_enabled{false};
    double tracklet_max_step_m{0.15};
    int tracklet_min_samples{12};
    double tracklet_min_duration_s{0.25};
    double tracklet_min_arc_rad{1.00};
    double tracklet_max_residual_m{0.03};
    bool tangent_rate_enabled{false};
    double tangent_rate_alpha{0.25};
    double tangent_rate_max_step_m{0.15};
    bool normal_phase_enabled{false};
    double normal_phase_sigma_rad{0.35};
    double normal_phase_min_facing{0.30};
    // 整车"原地自转"假设：车心当准静态量，更新后低通拉回去（默认开）。
    // 整车重拟合：用滑动窗口里的板心圆拟合，定期校准车心与半径。
    bool body_refit_enabled{false};
    int body_refit_every{25};
    int body_refit_min_samples{60};
    double body_refit_max_residual{0.05};
    double body_refit_window_s{2.0};
    double center_from_normal_sigma_m{0.05};
    double center_from_normal_gate_m{0.15};
    double center_from_normal_max_rate_rad_s{2.5};
    bool slot_phase_update_enabled{false};
    // ===== 参考实现移植（sp_vision_25 target.cpp:187-222，2026-10-25）=====
    // 朝向角直接量测进默认路径：每次位置更新用同一个 best_slot 做角度更新
    // （updateFaceYawForSlot，残差折叠 ±pi）。σ 按斜视角对数放大
    // （sqrt(log(|Δ|+1)+1)，Δ=板外法向与"板→相机"方向夹角，正对 ~1.0 rad）——
    // 角度量测天生弱，与位置更新不打架（round-17 的回归用的是 σ=0.12 强锁）；
    // w 由 Q 的 (a,w) 交叉协方差被连续拉拽（angular_accel 已对齐 400）。
    // 环境变量 EKF_NO_ANGLE_UPDATE 可整体关闭。
    bool reference_angle_update_enabled{true};
    // σ 取 2.5 的依据（G 组扫参 2026-10-25）：1.0/1.55 在 ω=6/8 rad/s 上把相位
    // 抖出 3cm 门（命中率 33~74%），2.5/4.0 全过（96~98%）。参考实现能扛 1.55
    // 是因为它的法向经 optimize_yaw 统一到高质量、且 4 维量测一次性融合；
    // 我们是顺序更新 + 换板瞬态，需要更弱的角度通道。
    //
    // 【2026-10-25 参考对齐】上面这段"2.5 才稳"的结论是在 Q 错 241 倍的坐标系
    // 下扫出来的——Q 过刚时 ω 不会变，角度残差只能全压到相位上，于是必须把
    // σ 放得极大才不抖。Q 对齐到 q=4 之后实测 σ=0.31 反而更好（v4 73.15→86.85、
    // v1 47.02→51.28）。所以默认改成参考刻度 0.31 rad（17°）。
    // 注意 σ 与身份后验是耦合的：小 σ 一旦槽位判错就硬拽 ±90°，所以
    // identity_posterior_enabled 必须同时打开。
    //
    // angle_state_obliquity_sigma=true 时用参考实现的原公式
    //   σ = sqrt(log(|Δ|+1)/200 + 9e-2)
    // （Δ=板外法向与"板→相机"方向的夹角）。该式在 Δ∈[0,π] 上几乎是常数
    // ≈0.30 rad——它本来就是"角度量测天生弱"的弱化通道，不是真的想把 σ 拉大。
    bool angle_state_obliquity_sigma{true};
    double angle_state_sigma_rad{0.31};
    // 槽位相位更新的触发下界。用 static_rate_rad_s(1.2) 会让静止车上的 omega
    // 抖动也偶尔触发（实测 id=1 的 PIX 样本数 516 -> 279）。但取 3.0 会让转子
    // 低速段失去槽位锁定（id=3 82.2 -> 91.2），所以仍用 1.2。
    double slot_phase_min_rate_rad_s{1.2};
    // 单板重锚：更新后状态里的板离观测还超过这个距离（相位漂了一整块板 / NIS
    // 门锁死）时，直接用这块板把相位+中心对齐。0 表示关闭。
    //
    // 默认 0.25 而不是关闭：回放 tmp/v1.avi（近处目标真转 2.4 rad/s + 手持
    // 抖动）时，状态一旦漂出位置门就再也回不来——1415 帧里只有 62 次更新被
    // 接收、预测命中中位 33 cm、p90 83 cm。打开重锚后同一段视频命中中位
    // 8.6 cm、p90 18 cm（≤5 cm 占比 26% → 54%）。v4（静止目标）不受影响：
    // 那里的板离预测一直只有毫米级，重锚根本不会触发。
    double single_plate_realign_m{0.25};
    // 重锚只在"已经连续若干帧完全更新不进去"时才允许（发散恢复），不在单帧
    // 上乱开：回归测试 test_rotating_armor_tracker 的 D 组要求单个离群观测
    // 被拒绝、绝不能把状态拉走，无条件重锚会直接违反这条。
    int realign_after_misses{3};
    // 重锚候选的最大距离：状态漂到位置门外时，只要还在这个范围内就当成"重新
    // 捕获同一块板"（跳过 NIS 门，先重锚再更新）。超过它说明观测根本不是这台车。
    double single_plate_realign_gate_m{1.5};
    double phase_update_sigma_rad{0.12};
    // 相位更新后是否"反锚"（把中心平移回去、让板留在原地）。单板可见时
    // (中心, 相位) 是规范自由度；但中心已经由位置更新约束住时，反锚会把
    // 相位修正变成中心漂移。两种做法都保留，A/B 用。
    bool phase_anti_anchor{true};
    int bootstrap_updates{10};   // 锁定期头几次更新先放宽 NIS 门（初值靠单帧法向，偏差可比预想大）
    // 声明反驳检查：法向历史里混入相邻板样本时斜率会失真，实拍里会误杀，
    // 默认关闭；需要时打开（合成夹具与 v4 的静止车都不需要它）。
    bool contradiction_check{false};
    // 同帧每块匹配上的板都做一次完整 EKF 更新（sp_vision_25 tracker.cpp:286-293
    // 就是 for-each-armor 依次 update）。四板车体同帧最多可能看到 2~3 块，
    // 上限设 4 等于不设限，仍保留一个上限防止病态输入。
    int max_updates_per_frame{4}; // 同一帧最多用几块板更新（防止病态约束）
    // 每个槽位是否各自学"半径/角度/高度"偏移（只影响显示用的模型板位置）。
    // 打开时四块模型板各有圆心，单块板能贴得很紧，代价是它们不再是一个刚体：
    // 槽位分配在两块同号板之间跳时偏移量跟着跳，画出来就是"转一半又卡回"。
    // 关掉后四块板严格共享 (cx, cy, R, phase)，同速度同方向同圆心。
    bool slot_offsets_enabled{false};
    // 允许"锁定号消失、画面里出现别的号"时换目标（单目标模式用）。
    // 多目标模式必须关：一辆车的 EKF 不许自己改板号。
    bool target_switch_enabled{true};
    // 半径的"慢"估计：>0 时用一阶滤波把每次锁定解出的半径平均进去（每次锁定的
    // alpha），重锁不清零。半径要稳定（偏平均），速度要跟手（偏瞬时）。
    double radius_track_smooth_alpha{0.06};
    // 弦长证据的滑动窗口（秒）。窗口越长半径越稳，但适应真实尺寸变化越慢。
    double radius_chord_window_s{8.0};
    // 弦长证据取"低分位"而不是中位/平均：错误板法向会把圆心解歪、把半径解大，
    // 这些大值不该参与。实测 v4 相邻弦长半径 p05-p90 = 0.262-0.301（中位 0.283），
    // v1 = 0.247-0.266（中位 0.259），过大的 0.36+ 全是法向错的那几帧。
    double radius_chord_quantile{0.20};
    // 物理合理区间：RM 车体中心到板面的半径不会超过这个范围（实测 0.25-0.30，
    // 小装甲 0.20，大装甲 0.25，底盘再大也就 0.32）。区间外的样本直接丢弃。
    double radius_plausible_min_m{0.18};
    double radius_plausible_max_m{0.34};
    // 无弦长证据时，拟合半径每步最多只能把估计拉动这么多（法向错时拟合值不可信）。
    double radius_fit_nudge_m{0.02};
    // 观测可信度：sigma = base * clamp(0.5 + reproj/gain, floor, 3.0)
    double position_sigma_quality_gain{2.0};
    double position_sigma_quality_floor{0.35};
    // 局部预测：连续 N 帧无观测后，每帧把角速度乘以 decay（避免无依据地继续转）
    int coast_decay_after_frames{3};
    double coast_omega_decay{0.90};
    // 角速度修正（PLL）：相位残差的斜率就是角速度误差，直接补到 omega 上。
    // PLL 版（相位残差斜率）：残差里混着槽位归属误差，实测会飙到 ±12 饱和，
    // 默认关，只作诊断开关保留。
    bool rate_from_residual_enabled{false};
    double rate_residual_window_s{0.6};
    int rate_residual_min_samples{5};
    double rate_residual_gain{0.8};
    // 同板角速度量测：连续两帧确认是"同一块板"时，板心绕车心的转角差就是 ω。
    // 直接写 omega 会让合成 F/G/H 回归（预测延迟、高速自转、转速自判）失准：
    // 噪声量测把角速度的协方差信息绕过去了。默认关，用 --rate-tracklet 1 打开。
    // 实测打开后 v4 id3 PIX 91.5 -> 83.0（转速确实跟得更紧）。
    bool rate_from_tracklet_enabled{true};
    double rate_tracklet_max_step_m{0.08};
    double rate_tracklet_alpha{0.35};
    // 车本来不转时不要往 omega 里灌噪声：量到的转角和当前估计都接近 0 就不修正。
    int rate_tracklet_min_streak{20};
    double rate_tracklet_max_centre_speed_m_s{0.60};
    double rate_tracklet_consistency_rad_s{3.0};
    double rate_tracklet_min_rate_rad_s{3.0};
    // 先攒够更新再动 omega；每次最多改这么多，避免一两帧噪声把转速带飞。
    int rate_tracklet_min_updates{12};
    // 绿色窗口圆（仅显示）：窗口长度与最小样本数。
    // 窗口转速估计：切段跳变阈值、最小转速、拟合不确定度下限。
    // （窗口长度复用上面的 rate_window_s，拟合器用同一个窗口。）
    // 窗口转速量测：实测能给出 |ω|≈3.9 但拟合残差 0.19 rad/0.22 s（σ≈0.85），
    // 滤波器正确地不太相信它；默认关，留作诊断。
    bool rate_window_enabled{false};
    double rate_segment_jump_m{0.12};
    double rate_window_min_rate_rad_s{0.5};
    double rate_window_sigma_floor_rad_s{0.25};
    double detected_circle_window_s{1.0};
    int detected_circle_min_samples{8};
    // 角速度量测的噪声：板心角度差的 1-sigma，以及下限（rad/s）。
    double rate_tracklet_sigma_theta_rad{0.06};
    double rate_tracklet_sigma_floor_rad_s{0.35};
    // 用量测而不是硬写之后，这个开关可以默认打开（合成 F/G/H 仍然要绿）。
    bool rate_measurement_enabled{true};
    // 角加速度过程噪声（连续白噪声谱密度 q，Q_ω = q · dt）。
    //
    // 【2026-10-25 参考对齐】原来的"按 tracklet |ω| EMA 分档的线性软门"
    // （floor 1.5 / full 3.0 rad/s，开满到 400）+ 两处回写已整段删除。两个原因：
    //   ① 门控变量 measured_rate_ema_ 只在 tracklet |ω|≥3 rad/s 且连续同向 20 步
    //      时才更新 → |ω|<3 时门恒关，angular_accel 恒 0.01，等于禁止 ω 变化；
    //   ② 0.01 与 400 分属两套离散化口径，本身不可比（见下），所以历史上
    //      "抬到 400 就对齐参考了"这个判断是错的。
    //
    // 量纲对照（这是本次对齐的核心）：
    //   sp_vision_25: Q_ω = v2 · dt²，v2 = 400  → dt=1/166 时 0.0145 /帧
    //   rmcs        : Q_ω = noise_rotation_speed 每帧常量 = 1.0
    //   本项目      : Q_ω = q · dt            → q = 0.01 时 6.0e-5 /帧（= 参考的 1/241）
    //   换算关系    : q ≈ v2 · dt，故 166 fps 上 v2=400 ⇔ q≈2.4。
    // 取 4.0（对应 v2≈660）而不是 2.4：本项目单帧位置噪声实测 13.8 mm，
    // 高于参考场景；两片段实测在 q∈[2,5] 内取 4 最优，8 会把 v1 打崩。
    // 环境变量 SPIN_Q 仍是实验旋钮（angular_accel = 0.01·SPIN_Q），勿在成品里设。
    double process_angular_accel{4.0};
    // 仅 tracklet 路径使用的有符号 EMA 系数（真自转同向、抖动正负抵消）。
    double adaptive_spin_alpha{0.25};
    double rate_tracklet_max_step_rad_s{0.4};

    // ===== 唯一量测通道：窗口相位斜率（2026-10-25）=====
    //
    // 依据 docs/转速架构-唯一量测通道方案_2026-10-25.md：单帧转速信噪比只有
    // **0.08**（1.1 rad/s、R=0.17 m、dt=6 ms 下每帧旋转位移 1.13 mm，而单帧
    // 位置噪声 σ=13.8 mm）。所以任何"逐帧差分"（角度差/切向投影/PLL 残差）
    // 在数学上都不可能工作——这不是调参问题。窗口最小二乘能把不确定性压到
    // σ_ω = σ_α/√Sxx：σ_α=0.081 rad 时，60 帧（0.36 s）给出 σ_ω≈0.10 rad/s，
    // 对 1.1 rad/s 是 9% 精度，够用。
    //
    // 三段职责，缺一不可：
    //   ① 观测量 = 窗口内"同一块板"板心绕**固定**车心的方位角斜率。
    //      车心取 EKF 当前 state(0),state(2)，半径取 state(8) 且**不参与拟合**。
    //      R 自由时短弧上 (R,ω) 病态——batch_fit 实测解出 R=0.008~0.019 m
    //      （真值 0.17），ω 一路撞 ±12 饱和，一开 v4 ≤5cm 73.2→48.7。
    //   ② 融合 = ekf_.updateRate(ω, σ_ω)。**不许** setRate/rotatePhase/
    //      shiftCenter 硬写：硬写抹掉协方差，量测对了滤波器照样崩。
    //   ③ 门控 = 显著性检验 |ω| > 3σ_ω，不是 |ω| > 0.5 或 3.0 这种绝对阈值。
    //      静止时斜率≈0 而 σ 覆盖 0，自动不更新——H 组契约不需要靠
    //      "1.5/3.0 rad/s"这类场景相关门限来保护。
    //
    // Q 保持常量（见 process_angular_accel）：φ 由 ω 传播（F(6,7)=dt），
    // 不再靠角度残差硬拽，所以"Q 大→φ 漂、Q 小→ω 冻"这个矛盾从根上消失。
    bool rate_phase_window_enabled{false};
    int rate_phase_window_n{60};        // 窗口样本数（166 fps ≈ 0.36 s）
    int rate_phase_min_samples{40};     // 少于它不估（σ_ω 还压不下来）
    int rate_phase_stride_frames{8};    // 每隔几帧估一次：相邻窗共享样本，太密等于重复用同一份证据
    double rate_phase_sigma_floor_rad_s{0.05}; // σ_ω 下界
    double rate_phase_significance{5.0};       // |ω| > k·σ_ω 才更新
    // 切段阈值（米）。**不**按"换板"切段：四块板相隔 90°，fold90 展开时换板的
    // 90° 跳变正好被折掉，弧长本来就该连续。按 0.10 m 切会把每次换板都开一段，
    // 合成夹具里 chosen 频繁换板 → 窗口永远攒不满 40 个样本 → 通道完全不触发
    // （G 组 ω 估计掉到 0）。所以阈值放到 0.45 m（R=0.25 时相邻板相距 0.35 m），
    // 只切"跳到别的车/别的目标"这种真正的不连续；斜置副板那种非圆轨迹由
    // rate_phase_max_radius_rms_m 兜住。
    double rate_phase_jump_m{0.45};
    double rate_phase_max_abs_rad_s{12.0};     // 物理上限（四板车实测最快 8）
    // 半径一致性门（RMS，米）：窗口内样本到车心的距离必须恒定（=R）。
    // 这条是 2026-10-25 实测补出来的：v4 有一段窗口切向斜率读出 -3.5 rad/s
    // （真值 ~1.1），离线查出来是样本半径在 0.19~0.33 m 之间散开（RMS 4.0 cm）。
    // 车心在动、或续到别的板上时，"绕车心的方位角"已经不再是转角，斜率就没意义。
    // 好窗口的 RMS ≈1.4~2 cm（就是 PnP 位置噪声），所以 2.8 cm 能把两者分开。
    double rate_phase_max_radius_rms_m{0.028};
    // ===== 两路交叉校验（2026-10-25 第二轮）=====
    //
    // 单路的实验数据（§四十八）说明"必须收敛成一条"这个判断只对了一半：
    // 单条通道各有自己的致命盲区，而它们的**误差来源正交**——
    //   · 相位斜率（板心绕车心的方位角）：不需要板面法向，对 PnP 法向噪声免疫；
    //     但车心漂 4 cm 就能把斜率污染成 −3.46 rad/s（真值 ~1.1）。
    //   · 法向斜率（板面 yaw）：**完全不需要车心**，对中心漂移免疫；
    //     但 PnP 法向噪声实测 ±35°，单帧不可用。
    // 所以两条都算、只在互相印证时才用，且不新增任何场景相关门限：
    // 判据是"两个独立估计在各自的不确定度内一致"，不是"速度超过某个阈值"。
    //
    // 法向的 90°（换板）与 180°（法向翻转）跳变都会被 fold ±45° 折掉，
    // 因为两者都是 90° 的整数倍——这一点和相位斜率完全同源。
    bool rate_phase_require_normal{true};
    // 两路一致判据：|ω_p − ω_n| ≤ k·√(σ_p²+σ_n²)。k 取 2.5 是"两个估计都
    // 落在 2.5σ 内"，对单个野值不敏感，又不至于把真实的一致判成不一致。
    double rate_phase_agreement_sigma{2.5};
    // 段末端允许落后多久（秒）：太旧的段不能用来描述"现在"转多快。
    double rate_phase_max_tail_gap_s{0.15};
    // 同一段至少要覆盖的时间跨度（秒）。0.12 s 是 60 帧窗口在 8 帧节流下的
    // 最坏情况；跨度再短时 Sxx 太小，σ_ω 会自己炸大，门会自然拒掉。
    double rate_phase_min_span_s{0.12};

    // === 状态机 ===
    int stable_updates{12};
    double stable_min_lock_s{0.60}; // 上锁后至少这么久才允许报 Stable          // 连续接受多少帧才允许报 Stable
    int unstable_after_missed{2};    // 仅供诊断：Stable 的降级判据已改为按秒（见 stale_after_s）
    double stale_after_s{0.10};      // 距上次被接收更新超过此秒数即判 Unstable（上限）
    // 自适应过期窗口：窗口 = min(stale_after_s, max(stale_min_s, stale_min_frames*dt))。
    // 166fps 下 0.10s 等于 17 帧，旋转目标早转过 90°；30fps 下又刚好 3 帧。
    // 用中位帧间隔把"秒"翻译成"帧"，让低帧率端不会过紧、高帧率端不会过松。
    double stale_min_frames{5.0};
    double stale_min_s{0.05};
    double timeout_s{0.60};          // 超过多久没有任何有效观测 -> 注销
    // 掉线快照续接：注销时把状态留一份，同一编号在 relock_reuse_s 内又出现，
    // 直接把旧状态装回去，省掉 10 个样本的重新收集。
    // 默认关闭：实测（v4 --only-id 3 / v1）续接能多救回 10% 的帧、把 Collecting
    // 从 319 帧压到 233 帧、p99 从 0.99 m 降到 0.58 m，但**中位残差变差**
    // （99 -> 115 mm），因为救回来的那批帧本身就是"拿 0.25 s 前的状态外推"，
    // 旋转目标在空窗里已经转过几十度。空窗越短越糟（0.10 s 时中位 1.35 m），
    // 说明这类帧不是"续一下就好"，而是必须重新锁定。
    bool relock_reuse_enabled{false};
    double relock_reuse_s{0.25};

    // === 声明校验（Static/Spinning 被数据反驳时重锁）===
    int relock_frames{6};
    double relock_omega_diff_rad_s{1.0};
    double relock_min_span_s{0.18};

    // === 输出 ===
    double max_bias_s{0.80};
    // 本帧视觉侧处理延迟（取到帧 -> 算出结果）的允许上限。
    double max_latency_s{0.30};        // prediction_bias_s 的限幅
    // 弹道补偿：预测时刻再往后推"子弹飞行时间 = 距离 / v0"。
    // 不补这一项时，转得快的板在飞行时间里就移出板宽了（见 docs/实验报告.md 命中率基线）。
    bool flight_compensation_enabled{true};
    double bullet_speed_m_s{30.0}; // 弹速 m/s（与 armor/include/BulletModel.hpp 的 v0 一致）
    double max_flight_s{0.50};     // 飞行时间限幅
    double fire_angle_error_rad{0.05}; // 云台已对准的阈值（Planner 的 fire_thresh）
    double fire_max_age_s{0.20};    // 最近一次有效观测距今多久内允许开火
    double min_distance_m{0.40};
    double max_distance_m{12.0};
    double switch_margin_rad{0.06}; // 换板迟滞：别的板要明显更好才换
    // 瞄准点只允许落在"最近真的被看见过"的板上：四块板里总有一两块朝向车体
    // 背面（几何上真实存在、但检测不到），按角度挑会挑到它们身上（实拍 v4 静止
    // 1384 帧全aim=3、看见的是 slot 0，瞄准点画到板外 0.35 m）。这个时间窗内
    // 见过的板才算候选，窗内的板里再挑离云台最近的。
    double aim_visible_hold_s{0.30};
    // 诊断用：强制锁定角速度（rad/s，NaN=不强制）。用来扫"真实角速度大概是多少"：
    // 在不同 forced omega 下看 30 ms 预测误差哪个最小，比从含跳变的序列里拟合更可靠。
    double forced_omega_rad_s{std::numeric_limits<double>::quiet_NaN()};
    // 瞄准点直接用"最近一次观测到的板心"的时间窗（秒）。0 = 关闭，退回 EKF。
    double aim_observation_hold_s{0.012};
    // 允许"新鲜观测外推"在任意预测时域下使用（默认只在零时域时用 EKF 四板模型）。
    // 实拍 v4 末段状态落后 0.19 m、而观测是 0~6 ms 前的真值，外推量只有 omega*30 ms。
    bool aim_observation_any_horizon{false};
    // 观测外推时是否按 omega 转动：omega 不可靠时（旋转模糊/换板）外推会把
    // 瞄准点推得比"原地不动"更远，关掉更稳。
    bool aim_observation_extrapolate{true};
    // 目标已经这么久没被观测到时，不再用状态外推（速度项会把瞄准点推着满场走），
    // 直接冻结在最后一次看到的那块板上。实拍 v4 有 100+ 帧的整段丢失，外推
    // 会让瞄准圈飘到另一台车上。
    double aim_freeze_after_s{0.30};
    // 瞄准点的一阶平滑系数（0 = 不平滑）。观测有 ~35 mm 的帧间抖动，直接送
    // 云台会抖；按 omega 做运动补偿后再做一阶平滑，不引入相位滞后。
    double aim_smoothing_alpha{0.0};
    // 瞄准点局部直线拟合的样本数与时间窗（0 = 关闭，直接用当帧观测）。
    int aim_fit_samples{0};
    double aim_fit_window_s{0.15};
    // 目标选择：自瞄应瞄"正压在准星上"的那台车。两台车同时在场时，另一台车的板
    // 连续 relock_frames 帧都比当前目标更靠近准星（多出 target_switch_margin_rad
    // 的迟滞），就丢目标重锁到它。
    //
    // 迟滞取 0.30 rad（约 17°）而不是 0.10：回放 tmp/v4.avi 里两台静止车分别在
    // 准星左右约 15°，0.10 的迟滞会让目标每 ~270 帧在两台车之间翻一次，
    // 每次翻转都要重新 Collecting（~30 帧），EKF 从没稳定过——实测 1415 帧里
    // 只接收 299 次更新、预测命中中位 5.9 cm、p90 22.9 cm。改成 0.30 后同一段
    // 视频目标切换 0 次、接收 1385 次、命中中位 6.8 mm、p90 1.5 cm。
    // 目标真的消失时走另一条路（锁定编号连续两帧检测不到）照样会切换，
    // 所以更大的迟滞不会"锁死在看不见的车上"。
    bool prefer_centered_target{true};
    double target_switch_margin_rad{0.30};
};

enum class LockPhase : std::uint8_t
{
    Idle = 0,     // 没有目标
    Collecting,   // 正在攒锁定样本
    Tracking,     // EKF 已初始化
};

// 由状态量自己判出来的运动类型（不是调用方声明的）。
// 判据只用 EKF 估计：|omega| 与 |v| 各自的门限，和"有没有证据"无关——
// 证据够不够是 rate_obs_* 的事，这里回答"车现在在干什么"。
enum class MotionMode : std::uint8_t
{
    Unknown = 0, // 还没有状态
    Static,      // |v| < static_speed_m_s 且 |omega| < static_rate_rad_s
    Translating, // 平移为主（|v| 够大、|omega| 不够）
    Spinning,    // 自转为主
};

inline const char *motionModeName(MotionMode mode)
{
    switch (mode)
    {
    case MotionMode::Static:
        return "static";
    case MotionMode::Translating:
        return "trans";
    case MotionMode::Spinning:
        return "spin";
    case MotionMode::Unknown:
    default:
        return "unknown";
    }
}

// 每帧诊断量。只给测试与 Web Debug 用，不进通信协议。
struct TrackerDiagnostics
{
    LockPhase phase{LockPhase::Idle};
    std::uint64_t batch_fits{0};
    std::uint64_t tracklet_rates{0};
    std::uint64_t continuity_rejects{0}; // 观测连续性门拒掉的候选数
    std::uint64_t body_circles{0};       // 每帧板心圆覆盖次数
    // 绿色（实测反推车体）显示用的窗口圆：窗口内板心的最小二乘圆。
    // 逐帧单板解会被 PnP 噪声带得抖 2-6 cm；窗口圆残差只有 ~1 cm。
    // 只用于显示，不写回状态。
    bool detected_circle_valid{false};
    Eigen::Vector2d detected_circle_center{0.0, 0.0};
    double detected_circle_radius{0.0};
    double detected_circle_residual{0.0};
    int detected_circle_samples{0};
    MotionMode motion{MotionMode::Unknown};
    double speed_m_s{0.0};
    int lock_samples{0};
    int last_lock_samples{0};
    double locked_radius_m{0.0};
    double locked_spin_rad_s{0.0};
    // 锁定时刻的速率估计细节（诊断"为什么判成不转/转多快"用）：
    double lock_rate_error_rad_s{0.0};
    bool lock_spin_from_positions{false}; // true = 速率来自板心位置圆拟合
    double lock_circle_radius_m{0.0};
    double lock_circle_residual_m{0.0};
    double lock_step_ratio{0.0};          // 有效（同板）步数占比
    bool radius_from_data{false};
    bool phase_flipped{false};
    std::uint64_t accepted_updates{0};
    // 双板整车求解（两块相邻板精确解出 c/R/yaw）生效的帧数。
    std::uint64_t pair_body_solves{0};
    // 单板重锚（rotatePhase+shiftCenter 硬对齐）触发次数。每次触发都会让
    // 状态瞬间跳到观测上，观感就是"转一部分又卡回去"；这个计数用来区分
    // "滤波器在跟踪"和"滤波器被硬覆盖"。
    std::uint64_t single_plate_realigns{0};
    std::uint64_t rate_measurements{0};
    std::uint64_t normal_anchors{0};
    std::uint64_t pair_phase_updates{0};
    std::uint64_t center_from_normal_updates{0};
    std::uint64_t body_refits{0};
    int consecutive_updates{0};
    int missed_frames{0};
    double observation_noise_m{0.0}; // 本帧使用的观测噪声 sigma（m）
    int updates_this_frame{0};
    double measured_latency_s{0.0};
    int rejected_this_frame{0};
    // 转速窗口估计（最近一次喂进 EKF 的 omega 伪量测）。
    bool rate_obs_used{false};
    double rate_obs_rad_s{0.0};
    double rate_obs_sigma_rad_s{0.0};
    int rate_obs_samples{0};
    int rate_window_samples{0};
    // 第二路证据（板相对车心的方位角斜率），用来和法向斜率互相校验。
    double rate_bearing_rad_s{0.0};
    double last_nis{0.0};
    double best_gate_distance_m{0.0}; // 本帧所有(观测,槽位)对里最近的位置残差（诊断用）
    double best_gate_nis{0.0};       // 本帧最小 NIS（诊断用）
    int aim_slot{-1};
    int target_id{0};
    int relock_count{0};
    std::string last_reject;
    std::string stability; // 本帧 Stable/Unstable 的判定理由
    // 本帧预测出的瞄准点（世界系），离线回放评估用：与同帧检测到的板心比距离。
    Eigen::Vector3d predicted_aim_world{0.0, 0.0, 0.0};

    // ===== 关联层诊断（S0）=====
    // 本帧被选中的那个观测，在 tracker 内部实际使用的法向（世界系，已过符号消歧）。
    // 与 dump 里的 oyaw（StudentTasks 的原始 ArmorPose）不是同一个量，两个都留着
    // 才能判断"法向到底在链路哪一段被改坏"。
    bool chosen_obs_has_yaw{false};
    double chosen_obs_yaw_rad{0.0};
    int chosen_slot{-1};
    // 环序配对的 4 个假设代价（S1，RING_DEBUG 时填）。单位 m²/σ²，越小越好。
    // 只有单板时可全为 nan（4 个 h 同构，没有意义）。
    double ring_cost[4]{0.0, 0.0, 0.0, 0.0};
    int ring_obs_count{0};
    // ===== S2：身份后验运行态（只有启用且本帧走多板路径时有意义）=====
    bool identity_active{false};
    int identity_assignment_count{0};   // 本帧枚举了多少种单射配对
    double identity_best_posterior{0.0};
    double identity_second_posterior{0.0};
    // 相对 margin = 次优代价/最优代价 - 1。越接近 0 说明"身份打平"，
    // 此时强制的整帧配对没有信息量，应该让上层知道。
    double identity_margin_rel{0.0};
};

class RotatingArmorTracker
{
public:
    explicit RotatingArmorTracker(TrackerConfig config = TrackerConfig{});

    void reset();
    void applyProcessNoise();

    // 独立相位量测（LED 灯条给出的板面法向）。内部按 phase_anti_anchor 决定
    // 是否把中心反锚、让瞄准板保持原位。
    bool applyPhaseMeasurement(double world_yaw_rad, double sigma_rad);

    // 图像域转速量测注入（系统视图层用）：亮斑图案频率通道等"不经过
    // WorldObservation"的转速证据走这里。与窗口证据同一套记账（成功后
    // 盖章，衰减块让路）。检测只出正对板框时（v4 实拍：位置/法向/弦向
    // 全部读 ≈0，真实 8 rad/s 自转不可观测），这是唯一能立起 omega 的路。
    bool injectRateObservation(double omega_rad_s, double sigma_rad_s, double timestamp_s);


    // 与 ekf_predict 同签名：喂进本帧的全部装甲板位姿，返回本帧应该发出去的结果。
    // 没有目标、时间戳非法或跟踪失效时返回默认的 NoTarget 结果。
    // measured_latency_s：本帧从取到图到调用本函数已经花掉的时间（秒），
    // 会叠加到 prediction_bias_s 上一起作为预测时域。
    PredictionResult step(const std::vector<ArmorPose>& observations,
                          const GimbalState& gimbal, double timestamp_seconds,
                          double measured_latency_s = 0.0);

    bool initialized() const { return ekf_.initialized(); }

    // 自适应过期窗口：min(stale_after_s, max(stale_min_s, stale_min_frames * dt_median))。
    double staleWindowSeconds() const;
    const TrackerDiagnostics& diagnostics() const { return diagnostics_; }
    const TrackerConfig& config() const { return config_; }
    const RotatingArmorEkf& ekf() const { return ekf_; }
    // 每个槽位相对车体中心的高度偏移（仅用于显示/投影，不参与滤波状态）。
    // 实拍的装甲板不是一个平面：同四个槽位的 PnP 高度差能到 ±4 cm，单 z 的
    // 刚体模型画出来就会上下错开。这里用观测跑 EMA 学出来，画模型板时补上。
    double slotZOffset(int slot) const
    {
        return (slot >= 0 && slot < 4) ? slot_z_offset_[static_cast<std::size_t>(slot)] : 0.0;
    }
    // 四个槽位高度偏移的均值。车体中心 state(4) 会被位置更新拉到"最近更新的那块
    // 板的 z"上，于是它随哪块板可见而上下跳——这就是回放里白叉(中心)比蓝板圈
    // 偏上/偏下的原因。旋转轴是竖直的，中心应该落在这四条板高偏移的均值上，
    // 四块板才会对称地绕它转。显示时把中心抬到 state(4) + 这个均值。
    double slotZOffsetMean() const
    {
        double sum = 0.0;
        for (double v : slot_z_offset_)
        {
            sum += v;
        }
        return 0.25 * sum;
    }
    // 每个槽位相对车体中心的半径偏移（同样只用于显示）。实拍那台自制道具并不是
    // 正方形：同一帧两块板到车心的距离能差 3-4 cm，单一 R 只能折中，画出来必然
    // 一块压住、另一块偏出去。这里按槽位学半径偏移，让四块模型板各自贴自己的板。
    double slotRadiusOffset(int slot) const
    {
        return (slot >= 0 && slot < 4) ? slot_radius_offset_[static_cast<std::size_t>(slot)] : 0.0;
    }
    // 每个槽位的角度偏移（弧度，仅用于显示）。道具的板位不是严格的 90°，两块同
    // 帧可同时看到的板实测能偏 20°；单靠整体 yaw 只能折中，因此按槽位学偏差。
    double slotAngleOffset(int slot) const
    {
        return (slot >= 0 && slot < 4) ? slot_angle_offset_[static_cast<std::size_t>(slot)] : 0.0;
    }

private:
    // 一帧观测转到跟踪系之后的形态。
    struct WorldObservation
    {
        Eigen::Vector3d position{0.0, 0.0, 0.0};
        bool has_yaw{false};
        double yaw{0.0};
        int target_id{0};
        // 板型（大/小装甲）。车体半径先验按它选：小装甲车 0.20 m、大装甲车
        // 0.25 m；没有板型信息时才退回 radius_prior_m。默认按小板处理（比赛里
        // 绝大多数是 3 号/4 号这种小装甲车）。
        ArmorSize size{ArmorSize::Small};
        double sigma_position_m{0.03};
    };

    struct RateResidual
    {
        double timestamp_s{0.0};
        double residual_rad{0.0};
    };

    struct LockSample
    {
        double timestamp_s{0.0};
        Eigen::Vector3d position{0.0, 0.0, 0.0};
        double yaw{0.0}; // 板面法向 yaw（世界系，mod pi 待定）
        int target_id{0};
        ArmorSize size{ArmorSize::Small}; // 板型（选半径先验用）
        bool handoff{false}; // 与上一个样本之间换了板（相差 pi/2 的整数倍）
    };

    // 位置圆拟合（车体中心 + 半径），Gauss-Newton。
    struct CircleFit
    {
        double cx{0.0};
        double cy{0.0};
        double radius{0.0};
        double residual{0.0};
        bool ok{false};
    };
    static CircleFit fitArmorCircle(const std::deque<LockSample>& samples);

    // 沿一段"同一块板"的轨迹：中心固定后，各样本相对中心的角度连续可展开，
    // 对 (t, theta) 做最小二乘斜率即得 ω；arc 为展开后的总转角，
    // resid 为样本到圆心的径向残差 rms。
    static bool estimateRateAlongTracklet(const std::deque<LockSample>& samples, const CircleFit& centre,
                                          double* rate_rad_s, double* arc_rad, double* residual_m);

    // 多秒窗口联合批处理：一次性解 (cx, cy, R, ω, φ)。
    // 动机：单帧单板只有 2 个方程，相位不可观测；而 1 s 窗口里板心本来就
    // 落在同一个圆上（实测同窗口圆拟合残差 1.1 cm），数据是自洽的，
    // 是逐帧滤波器在浪费它。离线验证：窗口拟合预测下一帧的误差 0.086 m
    //（≈37 px），而逐帧 EKF 的 PIX 中位数是 90 px。
    struct BodyBatchFit
    {
        bool ok{false};
        Eigen::Vector2d center{Eigen::Vector2d::Zero()};
        double radius{0.0};
        double rate_rad_s{0.0};
        double phase_rad{0.0};        // 窗口起点时刻的相位
        double circle_residual_m{0.0};
        double phase_residual_rad{0.0};
        double coverage_rad{0.0};
        int used{0};
    };
    static BodyBatchFit fitBodyBatch(const std::deque<LockSample>& samples, int plate_count,
                                    double omega_hint, double config_coverage_min,
                                    bool fold_phase, bool alias_gate);
    // 半径的"慢"通道：所有想改半径的地方都走这里，先过一阶滤波再写进 EKF。
    // 直接 setRadius(fit.radius) 会让半径在重拟合时整段跳变（v4 实测 0.28->0.15）。
    void applySlowRadius(double candidate_radius);
    // 同帧两块相邻板（法向差 ~90°）的弦长直接给出半径：R = 弦/(2 sin(45°))。
    // 这是最稳的一路半径证据——只依赖两个板心位置，不依赖板法向精度。
    void noteChordRadius(const std::vector<WorldObservation>& world, double timestamp_s);
    std::deque<LockSample> chord_radius_samples_; // 只要 timestamp_s / position(0) 用作 R 缓存

    static CircleFit fitCenterFixedRadius(const std::deque<LockSample>& samples, double radius,
                                          const CircleFit& initial);

    // 观测 -> 跟踪系（含剔除明显非法的观测）。
    void collectWorldObservations(const std::vector<ArmorPose>& observations,
                                  const GimbalState& gimbal,
                                  std::vector<WorldObservation>& out) const;

    // 攒样本；够了就尝试锁定。成功返回 true。
    // 双板可见时用两块相邻板直接重定几何（p_i + R*n_i = p_j + R*n_j），
    // 用于"锁定后更新持续被拒"的场合：单板定出的中心偏 ~R，靠更新拉不回来。
    bool reinitializeFromPair(const std::vector<WorldObservation>& world);
    bool tryLock(const std::vector<WorldObservation>& world, double timestamp_s,
                 const GimbalState& gimbal);

    // 关联 + 更新。返回本帧接受的更新数。
    int associateAndUpdate(const std::vector<WorldObservation>& world, double timestamp_s);

    // ===== S2：板身份精确后验 =====
    // 不留任何跨帧状态：身份在单帧内已由穷举给出精确最优，时间维的不确定性
    // 由滤波器自己的协方差承担。跨帧累积权重反而会被"可见板数变化"打断。
    void resetIdentityPosterior();
    // 穷举全部单射配对，成功时 slot_of[world 下标] = 槽位（-1 表示不参与）。
    // 返回 false 表示本帧不该走这条路径（单板帧 / 未锁定 / 数值异常）。
    bool identityPosteriorAssignment(const std::vector<WorldObservation>& world, double timestamp_s,
                                     std::vector<int>& slot_of);
    // 观测平滑（见 TrackerConfig::observation_smoothing_enabled）：返回 true 时
    // out 是平滑后的观测，否则调用方用原始观测。
    bool smoothObservation(const WorldObservation& observation, int slot,
                           double timestamp_s, ArmorObservation& out);
    struct ObservationSmoother
    {
        Eigen::Vector3d mean{Eigen::Vector3d::Zero()};
        int slot{-1};
        int target_id{-1};
        int count{0};
        double timestamp_s{0.0};
    };
    ObservationSmoother smoother_;
    // 观测噪声估计用的历史（只放被接受的观测，按时间升序）。
    struct NoiseSample { Eigen::Vector3d position; double timestamp_s; };
    std::deque<NoiseSample> noise_samples_;
    double observation_noise_m_{0.06};
    void noteObservationForNoise(const Eigen::Vector3d &position, double timestamp_s);
    double observationSigma(double detection_sigma_m) const;
    std::array<double, 8> last_seen_s_{{-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0}};
    // 每个槽位最近一次被接受到的世界系板心。零延迟时它就是"此刻板在哪"
    // 的最好答案；EKF 只负责给没有新鲜观测的槽位做外推。
    std::array<Eigen::Vector3d, 8> last_seen_position_{};
    // 每个槽位最近几次观测的(时间, 世界系板心)。瞄准点用"局部直线拟合"而不是
    // EKF 的 omega：实拍里 omega 常被判 0，靠它就等于纯滞后；直线拟合不需要
    // omega，30 ms 内圆周的曲率可以忽略。
    static constexpr int kAimHistory = 5;
    std::array<std::array<Eigen::Vector3d, kAimHistory>, 8> aim_history_{};
    std::array<std::array<double, kAimHistory>, 8> aim_history_s_{};
    std::array<int, 8> aim_history_count_{};
    Eigen::Vector3d aim_smoothed_world_{};
    double aim_smoothed_s_{-1.0};
    bool aim_smoothed_valid_{false};
    // 同帧双板弦长 -> 半径观测（见 .cpp 里的推导）。
    void updateRadiusFromPairs(const std::vector<WorldObservation> &world);
    // 两块相邻板闭式解 (车心, 半径, 相位) 并软更新到状态上。返回是否生效。
    bool updateBodyFromPair(const std::vector<WorldObservation> &world);
    // 掉线快照续接（见 config_.relock_reuse_enabled）。
    bool tryRestoreSnapshot(const std::vector<WorldObservation> &world, double timestamp_seconds);

    // 声明是否被观测反驳（Static/Spinning 判错、或目标转速突变）。
    bool declarationContradicted(const std::vector<WorldObservation>& world,
                                 double timestamp_s);

    // 填写输出（含瞄准板选择、开火许可、数值合法性检查）。
    PredictionResult buildResult(const GimbalState& gimbal, double timestamp_s);

    // 丢掉当前目标并转入重新收集（保留会话级计数与诊断）。
    void dropTarget(const char* reason);

    // 转速窗口估计：见 TrackerConfig 里 rate_* 的说明。
    // 上一帧真正被接受的更新数；重锚要求"连续更新不进去"，单帧离群不行。
    int last_updates_{0};
    double last_rate_update_s_{-1.0e9};
    // 图像域注入的有效期：注入是强证据（σ 小），窗口路在这种场景下读出的
    // ≈0 是"看不见"不是"没转"（v4 实拍 8 rad/s 不可观测），两路不能对打——
    // 注入后这短时间内窗口证据让路（样本照常收集）。
    double injected_rate_until_s_{-1.0e9};
    struct RateSample
    {
        double timestamp_s{0.0};
        double unwrapped_yaw{0.0};
        Eigen::Vector3d position{Eigen::Vector3d::Zero()};
        // 采样时刻的车心估计：用"板相对车心的方位角"做第二路转速证据时，
        // 每个样本必须配它自己的车心，否则目标平移会被当成旋转。
        Eigen::Vector2d center{Eigen::Vector2d::Zero()};
        // true = 这一帧的观测是从上一帧那块板**位置连续**续接来的（最近邻 <
        // rate_associate_gate_m）。换板续接（handoff）的样本位置会跳半块板，
        // 方位角差近 90°，混进方位角序列会解缠出几十 rad/s 的假转速，所以
        // 方位角证据只用连续样本；法向证据仍可用全部样本（折叠后连续）。
        bool continuous{true};
    };
    std::deque<RateSample> rate_samples_{};
    void updateYawRateFromWindow(const std::vector<WorldObservation>& world,
                                 double timestamp_s);
    // 唯一量测通道（见 TrackerConfig::rate_phase_window_enabled 的注释）。
    void updateRateFromPhaseWindow(double timestamp_s);
    void resetPhaseWindow()
    {
        phase_window_samples_.clear();
        phase_window_stride_ = 0;
    }
    void resetRateWindow()
    {
        rate_samples_.clear();
        chord_samples_.clear();
        raw_normal_samples_.clear();
    }
    // 同帧双板弦方向样本：dir 已按 π 周期解缠（板序无关）。
    struct ChordSample
    {
        double timestamp_s{0.0};
        double dir{0.0};
        double chord{0.0};
    };
    std::deque<ChordSample> chord_samples_;
    // 全部同号板的原始法向样本（每帧每板一条，不挑选）：四块板的法向都以 ω
    // 推进，跨帧全配对后同板对给出 ≈ω 的斜率、换板对是 ±90°/dt 的离群对，
    // Theil-Sen 中位数恰好落在信号上。只用 chosen 的话，chosen 链频繁换板，
    // 同板对太少（v4 实测 median ≈0）。
    struct RawNormalSample
    {
        double timestamp_s{0.0};
        double yaw{0.0};
    };
    std::deque<RawNormalSample> raw_normal_samples_;

    // 把某个 slot 的法向 yaw 记入历史（用于检查"转/不转"的声明是否被数据反驳）。
    void pushNormalHistory(int plate_slot, double timestamp_s, double world_yaw_rad);

    TrackerConfig config_{};
    RotatingArmorEkf ekf_{config_.ekf};
    TrackerDiagnostics diagnostics_{};

    bool has_timestamp_{false};
    double last_timestamp_s_{0.0};
    double state_timestamp_s_{0.0}; // EKF 状态对应的时刻

    bool initialized_{false};
    int target_id_{0};
    int aim_slot_{-1};
    double last_accept_s_{0.0};
    double measured_latency_s_{0.0};
    bool has_accept_{false};
    // 上锁时刻：自转目标需要转过一定角度才能把 ω/相位/半径分开估出来，
    // 太早报 Stable 会让瞄准点在收敛过程中乱跳（合成 G：ω=4 时前 ~60 帧
    // ω 还停在 1.2，瞄准误差 6-8 cm）。
    double lock_start_s_{-1.0};
    // 掉线快照（注销时留一份，短间隔内同编号再现就直接续上）。
    bool has_snapshot_{false};
    State snapshot_state_{State::Zero()};
    Covariance snapshot_covariance_{Covariance::Zero()};
    double snapshot_s_{0.0};
    int snapshot_id_{0};
    bool snapshot_phase_resolved_{false};
    bool snapshot_spin_pinned_{false};
    int consecutive_updates_{0};
    int missed_frames_{0};
    std::deque<RateResidual> rate_residual_samples_;
    Eigen::Vector2d rate_last_position_{0.0, 0.0};
    double rate_last_time_s_{-1.0};
    double rate_signed_ema_{0.0};
    int spin_streak_{0};
    bool rate_has_last_{false};
    int other_id_streak_{0};
    int centered_streak_{0};
    // 上一次被接受更新的槽位：用来判断"同一块板连续可见"（换板时不能用板面法向
    // 约束车体 yaw）。
    int last_updated_slot_{-1};
    // 半径的慢估计（见 radius_track_smooth_alpha）。跨锁定保留，故意不在 reset() 里清。
    double radius_track_m_{-1.0};
    Eigen::Vector3d last_accepted_position_{0.0, 0.0, 0.0};
    bool has_last_accepted_position_{false};
    // 相邻两帧观测的位移 → 切向速度 → ω（不依赖相位/槽位，见 docs）
    Eigen::Vector2d tangent_last_position_{0.0, 0.0};
    double tangent_last_time_s_{-1.0};
    bool tangent_has_last_{false};
    int radius_updates_{0};   // 本次锁定后已用弦长修正过几次半径
    std::array<double, 4> slot_z_offset_{};
    std::array<double, 4> slot_radius_offset_{};
    std::array<double, 4> slot_angle_offset_{};
    int static_streak_{0};    // 连续判为 static 的帧数（半径学习只信它）

    std::deque<LockSample> lock_samples_{};
    // 每个 slot 一份：(timestamp, 解缠后的法向 yaw)，用于估计观测到的角速度。
    // 必须按 slot 分开：同一辆车不同装甲板的法向相差 2*pi*k/N，
    // 混在一起会让斜率完全失真。
    std::vector<std::deque<std::pair<double, double>>> normal_history_per_slot_{};
    // 最近若干帧的间隔，只用来做"帧 <-> 秒"的换算（自适应过期窗口）。
    std::deque<double> recent_dt_{};
    int contradiction_streak_{0};
    int relock_count_{0};

    // 最近一次成功锁定用到的量（诊断/测试用）
    double locked_radius_m_{0.0};
    double locked_spin_rad_s_{0.0};
    int body_refit_countdown_{0};
    int batch_fit_countdown_{0};
    Eigen::Vector2d refit_center_{Eigen::Vector2d::Zero()};
    bool refit_center_valid_{false};
    std::deque<LockSample> body_samples_;
    std::deque<LockSample> tracklet_samples_;
    // 窗口相位斜率通道的样本缓冲（与 body_samples_ 分开）：
    // body_samples_ 故意不在 reset() 里清（body_refit 要跨锁保留轨迹），
    // 但转速窗口绝不能跨锁——上一台车的板心轨迹算出来的斜率是垃圾。
    std::deque<LockSample> phase_window_samples_;
    int phase_window_stride_{0};
    bool has_pair_relock_{false};
    double last_pair_relock_s_{0.0};
    double timestamp_s_{0.0};
    bool lock_radius_from_data_{false};
    bool lock_phase_flipped_{false};
};

} // namespace rotating_armor
