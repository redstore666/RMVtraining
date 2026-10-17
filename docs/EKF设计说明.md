# EKF 跟踪与预测设计说明（任务 4）

本文说明自瞄链路第 4 段——`ekf_predict`——的实现方法、坐标系约定、参数与
验证方式。对应任务书 `docs/大作业任务书.md` §3.6 与 §4 的要求。

## 1. 文件与职责

| 文件 | 职责 |
| --- | --- |
| `student/include/RotatingArmorModel.hpp`、`student/src/RotatingArmorModel.cpp` | 旋转车体运动模型：状态、状态转移、观测方程、雅可比、过程噪声。**唯一的公式真源**。 |
| `student/include/RotatingArmorEkf.hpp`、`student/src/RotatingArmorEkf.cpp` | 9 维 EKF 本体：初始化契约、预测、观测更新（Joseph 形式）、NIS 计算。 |
| `student/include/AimGeometry.hpp`、`student/src/AimGeometry.cpp` | 云台系/世界系换算、极角定义、角度归一化。 |
| `student/include/RotatingArmorTracker.hpp`、`student/src/RotatingArmorTracker.cpp` | 粘合层：观测转换、首帧锁定、数据关联、门限、状态机、瞄准板选择、`PredictionResult` 填充。 |
| `student/src/StudentTasks.cpp:743` | `ekf_predict()`，按任务书固定签名，内部只调跟踪器。 |

调用关系：

```text
armor_solve (StudentTasks.cpp:606)
   └─ ArmorPose{position_gimbal_m, normal_gimbal, reprojection_error, target_id}
        └─ ekf_predict (StudentTasks.cpp:743)
             └─ RotatingArmorTracker::step (RotatingArmorTracker.cpp:655)
                  ├─ collectWorldObservations  ← AimGeometry::gimbalToWorld
                  ├─ tryLock                   ← 首帧锁定（相位扫频 + 几何先验）
                  ├─ predict / update          ← RotatingArmorEkf
                  ├─ associateAndUpdate        ← 距离门限 + NIS
                  ├─ declarationContradicted   ← 静态/旋转声明自检
                  └─ buildResult               ← PredictionResult / 开火许可
```

`ekf_predict` 不访问相机、不改原图、不写通信结构体；它只吃
`(std::vector<ArmorPose>, GimbalState, double)`，返回 `PredictionResult`。

## 2. 坐标系与状态

### 2.1 约定

* 世界系（跟踪系）：`前 x、左 y、上 z`，与 `armor/include/BulletModel.hpp` 的
  预测端约定一致；yaw 绕 z 逆时针为正，pitch 抬头为正。
* 云台系：同样轴序，随机头 yaw/pitch/roll 转动。
* 世界系原点取"程序启动后第一次处理帧时的云台位置"，只用于本章程内的
  相对几何；不做里程计，因此装甲板遮挡过久导致的重锁会在新原点重新起算。
* 变换：`p_world = Rz(yaw)·Ry(-pitch)·Rx(roll)·p_gimbal`，见
  `student/src/AimGeometry.cpp:11`。`pitch>0` 时云台系 `(1,0,0)` 映射到世界系
  `z>0`，由 `tests/test_aim_geometry.cpp` 钉住。
* 输出 `PredictionResult::target_yaw/target_pitch` 采用**绝对角**（同一个世界系
  下的极角），与 `setSafeOutput()` 把 `target_yaw` 置成当前 `yaw` 的语义一致。

### 2.2 状态向量（9 维）

```text
x = [ cx, vx, cy, vy, z, vz, yaw, yaw_rate, radius ]^T
```

| 分量 | 含义 | 单位 |
| --- | --- | --- |
| `cx, cy` | 车体旋转中心在跟踪系的位置 | m |
| `vx, vy` | 中心平移速度 | m/s |
| `z, vz` | 装甲板高度（本模型假设所有板等高）与高度变化率 | m, m/s |
| `yaw` | 车体偏航 | rad |
| `yaw_rate` | 车体自转角速度（小陀螺） | rad/s |
| `radius` | 装甲板到旋转中心的半径 | m |

观测（3 维）：第 `slot` 块板的板心位置

```text
phase = yaw + 2*pi*slot/plate_count
h(x)  = [ cx - radius*cos(phase), cy - radius*sin(phase), z ]^T
```

即板心 = 中心沿相位方向后退一个半径。板面**外法向**的 yaw 为
`phase + pi`，与 `platePhaseToWorldYaw/worldYawToPlatePhase`
（`student/src/RotatingArmorModel.cpp:85`、`:89`）互逆。

### 2.3 关于车体 yaw 的对称性

N 块装甲板几何上完全相同，把 `yaw` 整体平移 `2*pi/N` 后板位集合与原来完全
一致，任何观测都区分不出来。因此**车体 yaw 只能确定到 `2*pi/N`**（4 块板时
90°）。这不影响瞄准（板位集合相同）、也不影响 `yaw_rate`，但报告与答辩时应
说明：`PredictionResult::body_yaw` 是"某一等价分支"，测试里按模 `2*pi/N`
断言（`tests/test_rotating_armor_tracker.cpp`）。

## 3. 模型与噪声

* 状态转移 `transition`（`RotatingArmorModel.cpp:52`）：中心/高度按常速度，
  `yaw += yaw_rate*dt` 后归一化，`radius` 视为缓变常数。
* 雅可比 `transitionJacobian`（`:63`）：单位阵 + `F(0,1)=F(2,3)=F(4,5)=F(6,7)=dt`。
* 观测雅可比 `measurementJacobian`（`:93`）：

```text
H(0,0)=1,  H(0,6)=+R*sin(phase),  H(0,8)=-cos(phase)
H(1,2)=1,  H(1,6)=-R*cos(phase),  H(1,8)=-sin(phase)
H(2,4)=1
```

* 过程噪声 `processNoise`（`:113`）：三轴位置/速度与 yaw/yaw_rate 用连续白
  加速度模型 `q*[[dt^3/3, dt^2/2],[dt^2/2, dt]]`，`radius` 用随机游走 `q*dt`。

默认参数（`student/include/RotatingArmorModel.hpp:33`）：

```text
accel_x = accel_y = accel_z = 0.01   ((m/s^2)^2/Hz)
angular_accel = 0.01                 ((rad/s^2)^2/Hz)
radius_random_walk = 1e-6            (m^2/s)
```

量级：60 fps（dt=1/60 s）下每帧速度噪声 `sigma_v = sqrt(q*dt) ≈ 0.013 m/s`，
角速度噪声 `≈ 0.013 rad/s`。原来的默认值（`1.0 / 4.0 / 1e-4`）对应每帧
`0.13 m/s`、`0.26 rad/s`，等价于"允许状态每帧乱跳"，合成真值下会把角速度
估计带偏、并让"声明静止"失效。默认值现由
`tests/test_rotating_armor_convergence.cpp` 的护栏断言与"历史默认值"诊断行
共同防回退。

## 4. 观测从哪来、怎么变换

`armor_solve`（`student/src/StudentTasks.cpp:606`）产出 `ArmorPose`：

* `position_gimbal_m`：板心在云台系的位置（米），
  `p_gimbal = R_cam_to_gimbal * p_cam + t_cam_to_gimbal`。**外参仍是占位单位阵时
  （`config/camera.yaml` 现在是 `I` 且 `t=0`），代码会先补上"相机光学系
  （x 右 / y 下 / z 前）→ 云台系（x 前 / y 左 / z 上）"的固定旋转
  `kOpticalToGimbal`**（`StudentTasks.cpp` 的 `rotationCameraToGimbal()`，
  `t` 同时按 0 处理，首次调用向 stderr 打印一条警告）。单位阵在物理上不可能成立：
  相机光轴必然与云台前向对齐，两者至少差这 90° 的轴重排。直接套单位阵会把
  OpenCV 的"前"当成云台的"上"，EKF 的世界模型整体错位（2026-10-17 实跑 CSV 里
  `target_pitch≈1.15 rad`、半径塌到 `0.09 m`、全程锁不住，根因即此）。
  填好真外参后该分支自动失效，走 `R*p + t`。回归测试在 `ekf_pipeline`
  里（"占位外参按光学系->云台系重排坐标轴"）。
* `normal_gimbal`、`armor_yaw`：板面法向（云台系）：先用 `Rodrigues(rvec)`
  取物体 `(0,0,1)` 轴，再乘 `R_cam_to_gimbal`，归一化后
  `armor_yaw = atan2(n_y, n_x)`。**只保留法向；逐帧观测仍只用 3 维位置**，
  法向只在首帧锁定与"转/不转"自检时使用。
* `reprojection_error`：用于把观测噪声从 `position_sigma_m` 放大最多 3 倍
  （`1 + err/6`，`RotatingArmorTracker.cpp:155`）。

跟踪器每帧用一次 `gimbalToWorld(gimbal)` 把整帧观测转到世界系。所以
`ekf_predict` 天然满足任务书"无观测时把空观测传进来做纯预测"的要求：空
`std::vector` 只跳过观测更新，预测照常推进。

## 5. 初始化契约（两段式锁定）

单槽 EKF 不能冷启动：3 维观测 + 9 维状态，若角速度先验为 0，滤波器会把
"相位在转"解释成"中心+半径偏了一点"，300 帧后中心仍可能差 0.3~0.43 m 而板
残差只有 1.5~2 cm（"板对、车体全错"）。合成真值多噪声种子扫描：`omega=0`
冷启动只有 5/8 个种子能锁进正确极小值；给到 `omega≈2` 的先验则 8/8。所以
"转不转、转多快"必须由锁定阶段从数据里先声明出来。

### 5.1 攒样本（`tryLock`，`RotatingArmorTracker.cpp:197`）

只在同时满足下述条件时把观测放进锁定缓冲区：

1. 有目标编号（`target_id != 0`）；
2. 有板面法向（`normal_gimbal` 有效）；
3. 与上一帧样本的板面法向差 `<= 0.45 rad`（连续性）。

第 3 条必须优先于"哪块更准"：同一辆车不同装甲板的法向相差 `2*pi/N`，混进
一块不连续的样本会让解缠后的角速度完全失真。没有连续候选时本帧不加样本，
等缓冲区自然过期（0.5 s）后重开一条链。

### 5.2 解算并初始化

至少 6 个样本、时间跨度 `>= 0.18 s` 后：

1. 对法向 yaw 序列解缠，最小二乘斜率 = 角速度 `omega`（同时得到斜率的标准误）；
2. 相位扫过 `>= 0.30 rad` 且样本 `>= 12` 时，解 5 未知量最小二乘
   `p_k = c + v*(t_k - t_last) - R*[cos(phi_k), sin(phi_k)]`，用 `R` 的符号
   直接判断"车体在板的哪一侧"；
3. 否则用几何先验：相机只能看到板正面，板面外法向应背离视线方向，
   即模型相位 = `yaw_normal + pi`（`flipped`）；半径取配置先验 `0.25 m`；
4. 拟合出的 `R` 符号必须与几何先验一致，否则整段退回先验半径——相位判断
   一旦翻错，中心会落到相机这一侧，偏差 2R（自测里出现过 0.49 m 的偏差）；
5. `|omega| <= max(1.2, 2.5*slope_error)` 判为静止 → `SpinPriorKind::Static`
   （角速度硬钉在 0，过程噪声里角速度整块清零），否则 `Spinning` 并给出
   `omega` 与 `yaw_rate_sigma`。

`RotatingArmorEkf::initialize`（`student/src/RotatingArmorEkf.cpp:53`）用
"板心 + R*[cos, sin]" 反推中心，使首帧残差为 0；`yaw = phase - 2*pi*slot/N`；
`P0` 对角取各分量的 `sigma^2`。`EkfInitPrior` 的非有限/非正 sigma 会被
`positiveOr` 换成默认值，避免脏 `P0`（`P0` 一旦是脏矩阵再也修不回来）。

静态声明是**硬约束**：目标真的开始转时滤波器不会自己跟上，由 5.3 的自检
负责重锁。

### 5.3 "转/不转"声明被观测反驳时重锁

`declarationContradicted`（`:500`）：取样本最多的槽位的法向历史（0.4 s 窗口），
拟合斜率与 EKF 的 `yaw_rate` 相差 `> 1.0 rad/s` 连续 6 帧 → 丢弃目标重新锁定。
这条覆盖"把静止判成旋转、把旋转判成静止、转速突变"。

## 6. 关联与门限

`associateAndUpdate`（`:416`）按"先外推、再关联"的顺序施工：

1. `ekf_.predict(dt)`，把状态推进到本帧时刻；
2. 对每个未用观测 × 每个槽位，算预测板位与观测的距离，要求
   `距离 <= position_gate_m (0.60 m)` 且 NIS `<= 11.34`（3 维 chi-square，
   99.5%）。NIS 由 `RotatingArmorEkf::innovationNis`（`:196`）给出：
   `S = H P H' + R`，`nis = r' S^-1 r`；
3. 每轮取距离最小的 (观测, 槽位) 对做一次更新，最多 `max_updates_per_frame`
   (2) 次；命中槽位的法向 yaw 记入历史供 5.3 使用。

只做"最近邻 + 门限"而不做全局最优分配，是因为同一辆车同时可见的板通常
≤ 2 块，且槽位由运动模型唯一决定；换成匈牙利算法在这个规模上没有收益。

## 7. 状态机、超时与重锁

`step`（`:655`）的顺序与任务书 §4 一致：

1. 时间戳/云台角非有限 → 清空并返回无目标；时间戳回退 → 清空；
2. 观测转世界系（顺带剔除非法观测）；
3. 用本帧出现最多的非零编号做"换车"判断：连续 2 帧是别的编号才丢；
4. 未初始化 → `tryLock`，成功则直接输出，失败返回无目标；
5. 已初始化 → 校验 `dt ∈ (0, timeout_s]`，`predict(dt)`，关联更新；
6. 更新成功则 `consecutive_updates_++`、`missed_frames_=0`；否则
   `missed_frames_++`（仍然输出预测，即"短时丢失外推"）；
7. 距最后一次成功更新超过 `timeout_s (0.6 s)` → 清空，返回无目标；
8. 5.3 的声明自检 → 需要时重锁；
9. `buildResult` 填 `PredictionResult`。

状态映射：`consecutive_updates >= 12` 且相位已锁定、连续丢帧 `< 2`、未超时
→ `Stable`（协议 `crc=1`）；否则 `Unstable`（`crc=2`）；清空后 → `NoTarget`
（`crc=0`）。`Unstable` 允许继续跟踪但禁止开火。

## 8. 输出映射与开火许可

`buildResult`（`:548`）：

1. 复制一份 EKF，按 `prediction_bias_s + 实测处理延迟` 预测到"子弹该对准的时刻"：
   * `prediction_bias_s` 由电控下发（限幅 `[0, 0.8] s`），它知道弹道飞行时间与
     自身回路延迟；
   * 实测处理延迟由 `ekf_predict` 用 `steady_clock::now() - timestamp_seconds`
     得到（帧时间戳即取流时刻），限幅 `[0, 0.3] s`，只在测到合理值时采用，
     避免时钟口径不一致时误补偿。这部分是"取到这一帧到算出结果"已经花掉的
     时间，属于视觉侧必须自己补的那一段；
2. 在预测时刻对每个槽位算板位，选 `max(|yaw 误差|, |pitch 误差|)` 最小的
   那块作为瞄准板；当前瞄准板享受 `switch_margin_rad (0.06)` 的迟滞，避免
   在两块板等距时来回跳；
3. `distance_m = |板位|`，`target_yaw/target_pitch` 用绝对极角；
4. 角速度用"再预测 1 ms 后做前向差分"，比手推 `d(atan2)/dt` 更不容易错；
5. 开火许可：
   `Stable && |target_yaw - 当前云台 yaw| <= 0.05 rad && |pitch 误差| <= 0.05
   && 本帧有更新 && 距上次更新 <= 0.2 s && 距离在 [0.4, 12] m`；
6. 任一字段非有限 → 整帧返回无目标（框架 `applyPrediction` 还会兜底一次）。

`radius_1_m` 与 `radius_2_m` 目前都填同一个半径：本模型是单半径，
`radius_2_m` 留给"交替半径"目标（如前哨站三块板）的未来扩展。

## 9. 参数表（`TrackerConfig`，`student/include/RotatingArmorTracker.hpp`）

| 参数 | 默认值 | 作用 |
| --- | ---: | --- |
| `position_sigma_m` | 0.03 | 观测位置噪声（再按重投影误差放大 1~3 倍） |
| `normal_sigma_rad` | 0.35 | 首帧相位先验的噪声 |
| `lock_min_samples` / `lock_min_span_s` / `lock_max_age_s` | 6 / 0.18 s / 0.5 s | 首帧锁定样本量、跨度、过期 |
| `phase_spread_for_radius_rad` | 0.30 | 允许用数据定半径的最小相位扫幅 |
| `radius_prior_m` / `radius_sigma_m` | 0.25 / 0.10 | 半径先验与不确定度 |
| `radius_min_m` / `radius_max_m` | 0.05 / 0.60 | 半径合法区间 |
| `static_rate_rad_s` | 1.2 | 判定"静止"的角速度门限（另加 2.5*斜率标准误） |
| `position_gate_m` / `nis_gate` | 0.60 / 11.34 | 关联门限（NIS 取 3 维 99.5%） |
| `max_updates_per_frame` | 2 | 每帧最多接受几块板的更新 |
| `stable_updates` / `unstable_after_missed` | 12 / 2 | 稳定判据 |
| `timeout_s` | 0.60 | 超时清空 |
| `relock_frames` / `relock_omega_diff_rad_s` / `relock_min_span_s` | 6 / 1.0 / 0.18 | 声明被反驳时重锁 |
| `max_bias_s` | 0.80 | 预测时间偏置限幅 |
| `max_latency_s` | 0.30 | 实测处理延迟限幅 |
| `fire_angle_error_rad` / `fire_max_age_s` | 0.05 / 0.20 | 开火角度门限与数据新鲜度 |
| `min_distance_m` / `max_distance_m` | 0.40 / 12.0 | 开火距离窗口 |
| `switch_margin_rad` | 0.06 | 换板迟滞 |

## 10. 已知限制

1. **车体 yaw 只有模 `2*pi/N` 的分辨率**（见 2.3）：物理上不可观，不是估计误差。
2. **单半径、等高**：本模型假设各装甲板到中心等距、等高。前哨站那种"三块板
   交替高度/半径"目标需要把 `radius`/`z` 扩成按槽位的量，`PredictionResult`
   已经预留 `radius_2_m`。
3. **首帧锁定需要相位扫过一段**：目标完全静止且视场内只有一块板时，半径只能
   用先验 `0.25 m`；若实际机器人半径偏离较多，中心估计会按先验偏差偏移。
4. **只做最近邻关联**：同一编号下同时出现多辆同色车时无法区分（依赖
   `target_id`）。
5. **外参未标定时的绝对精度**：占位外参由
   `rotationCameraToGimbal()` 补成"光学系 → 云台系"的固定旋转（见 §4），
   所以相对几何（中心/半径/角速度）可用；但相机相对云台回转中心的平移 `t`
   与安装倾斜仍未知，绝对瞄准角有系统偏差，需按 `docs/云台外参.md` 标定后复测。
6. **预测偏置限幅 0.8 s**：更长的偏置会被截断（防止异常值把目标预测到几米外）。
7. **不做弹道补偿**：`target_pitch` 输出几何仰角，重力/飞行时间补偿在电控侧。
   若要求视觉侧补，`armor/include/BulletModel.*` 提供了带空气阻力的迭代模型。

## 11. 验证与证据

| ctest 名 | 覆盖内容 |
| --- | --- |
| `ekf_model_unit` | 模型数值、雅可比与解析式一致、噪声对称半正定 |
| `ekf_model_convergence` | 合成真值收敛（单槽、1 cm 观测噪声、300 帧）、默认噪声防回退 |
| `ekf_init_contract` | 生命周期/非法输入/reset/dt 边界、有先验 vs 无先验收敛对比、多噪声种子、外推判别 |
| `ekf_aim_geometry` | 坐标系约定、正交性、往返一致、极角定义 |
| `ekf_tracker` | 锁定/收敛/换板、静止目标开火、遮挡外推、超时清空、离群拒绝、换目标重锁、预测时域含实测延迟 |
| `ekf_pipeline` | 观测 → `ekf_predict` → `applyPrediction` → 64 字节帧，安全输出不变量 |

关键实测（`ctest -E camera_grab`，全部通过）：

* 合成真值 300 帧（1 cm 观测噪声）：末帧 `dcx=0.007 m`、`dR=-0.006 m`、
  `dyaw=0.010 rad`、`domega=0.007 rad/s`；每帧板残差 4.6 mm，外推 0.3 s 后
  板位置误差 0.0198 m。
* 跟踪器场景：11 帧完成锁定、22 帧进入 `Stable`；末帧中心误差 0.0015 m、
  半径 0.0010 m、角速度 0.009 rad/s；遮挡 0.25 s 期间持续外推且禁止开火，
  0.6 s 后自动清空；1.5 m 的离群观测被门限拒绝（中心仅动 0.5 mm）。
* 延迟补偿：0.5 m/s 平移目标加 0.1 s 实测延迟，预测距离正好前推 0.050 m
  （与 `v*latency` 一致）。
* 端到端：`crc` 计数 `0=26 / 1=247 / 2=57`，`fire_allowance=1` 的帧全部
  落在 `crc=1`，`center_x/y` 米→毫米换算与预测值一致。

## 12. 真数据回放验证（`tmp/v1.avi`、`tmp/v4.avi`）

工具：`tests/test_camera_playback`（`tests/test_camera_playback.cpp`），离线读视频、
跑 `armor_detect` → `armor_solve` → 可选 EKF，把检测框与预测板心画回画面，并导出
CSV。回放没有云台角，工具按"云台角恒为 0 + 固定旋转把光学系摆正到水平系"处理。

| 视频 | 内容 | 结果（本轮修完后的最终数） |
| --- | --- | --- |
| `v4.avi` | 静止大板（id=1，1.98~2.08 m）+ 旋转小板（id=3，1.0~1.9 m），相机固定 | 帧 30 锁定 id=1，`Stable` 1374/1415（97%），重锁 0、目标切换 0；位置抖动 3.8 mm |
| `v4.avi --only-id 3` | 只保留旋转目标 | 帧 152 锁定，`Stable` 716/1415，重锁 1 次（目标消失超 0.6 s 的 `timeout`），目标切换 0；被拒观测从 466 降到 37 |
| `v1.avi` | 手持晃动，id=1 大板与 id=3 小板先后可见 | 帧 22 锁定，`Stable` 324/608，重锁 2（目标离场）、目标切换 2（确实换成了另一辆车） |

`Stable` 的判定从"丢帧数 < 2"改成"距上次被接收的更新 ≤ `stale_after_s`（默认
0.10 s）"：166 fps 下 2 帧只有 12 ms，按帧数门限会让正常跟踪也长期停在 `Unstable`。
改完后 `v4 --only-id 3` 的 `Stable` 从 426 涨到 716，且剩下的 `Unstable` 几乎全部对应
"目标这 0.1 s 内确实没有可用的观测"（诊断字段 `stability`：`stale` 332 帧、
`few_updates` 44 帧，而 Tracking 期间因门限被拒的帧只剩 37）。

本轮在真数据上定位并修掉的四个问题（都不是调参）：

1. **锁定后更新全拒**：锁定后的前十几个更新是在"初值协方差还很大"时进来的，
   `χ²(3,99.5%)` 门限把合法观测判成离群。改成前 `bootstrap_updates` 次更新不做
   NIS 否决（只做位置门），`v4 --only-id 3` 的接收数从 0 变成正常跟踪。
2. **换板断链**：同编号两块板逐帧交替，严格的 `|Δyaw| ≤ 0.45 rad` 连续性门把换板
   判成断链，链永远攒不够样本。改成"同板优先、换板也接受"（换板那一跳是 `π/2`
   整数倍，可用 `round(Δ/π·2)·π/2` 从相位里扣掉，得到车体连续相位再测速率），
   同时把选点优先级修正为同板优先（原实现写反了，优先换板）。
3. **双板直解几何**：同一帧露出两块相邻板时用 `p_i + R*n_i = p_j + R*n_j` 直接解
   `R` 与中心，3 个样本、0.05 s 就能锁；锁定后连续 5 帧更新失败会自动用双板重定几何
   （`reinitializeFromPair()`）。
4. **`declaration_contradicted` 误杀**：法向历史里混入相邻板的样本，斜率和状态角速度
   差出 1 rad/s 就重锁，实拍里每 30~50 帧误杀一次。默认改为关闭
   （`contradiction_check=false`），需要时可在配置里打开。

仍然成立的限制：高速自转目标（每秒换板 2 次以上）在"每次只露一块板"的帧上只能
靠外推，`Stable` 比例受可见率限制；这是观测不足，不是滤波器发散。若验收要求
"高速自转下也连续 `Stable`"，下一步应把"车身相位（模 `π/2`）"提升为主观测、
位置作辅助，并把 `stale_after_s` 与该场景的检测间隔挂钩。

### 12.1 预测精度：模型解释度指标

`Stable` 帧数只说明"滤波器在跑"，不说明"状态对不对"。为此在回放工具里加了一个
客观指标（`--ekf` 时自动统计）：

**模型解释度** = 每帧把该车实测的板心，与"此刻 EKF 状态推出的 4 个板位"取最近
距离。单块板不用比（4 板对称，槽位标签本来只能确定到 `2π/N`），但"每一块观测都
能被模型解释"是状态正确的硬指标：状态错 `2R`（换板）时残差是 0.3 m 量级，
状态正确时是几厘米。

| 视频 | 观测帧 | 中位 | p90 | p99 | ≤2 cm | ≤5 cm |
| --- | --- | --- | --- | --- | --- | --- |
| `v4.avi`（静止目标） | 1384 | 7.4 mm | 16.8 mm | 28.0 mm | 94.5% | 99.6% |
| `v4.avi --only-id 3`（旋转目标） | 923 | 99.0 mm | 181.5 mm | 995 mm | 7.9% | 17.7% |
| `v1.avi`（两车轮换） | 513 | 412 mm | 2575 mm | 3005 mm | 3.5% | 12.7% |

**"救回重锁帧"也试过、也不划算**：注销时留一份状态快照，同编号在 0.25 s 内再现
就直接续上（`relock_reuse_enabled`）。效果是 Collecting 从 319 帧压到 233 帧、
多评估 86 帧、p99 从 0.99 m 降到 0.58 m，但**中位残差 99 -> 115 mm 变差**：
救回来的帧本来就是"拿 0.25 s 前的状态外推"，旋转目标在空窗里已经转过几十度。
窗口缩到 0.10 s 更糟（中位 1.35 m）。结论：这些帧不是"续一下就好"，而是必须
重新锁定——它们本来就不该被算作"能预测"的帧。默认关闭。

旋转目标残差的分布（v4 `--only-id 3`，923 帧）：p25 60 mm、p50 99 mm、
p75 139 mm、p90 181 mm——单峰宽分布，不是"少数帧炸了"。同时 1415 帧里有 323 帧
本帧一次更新都没有（`ok=0`）、只有 4 次 timeout。**更可能的解释是"状态陈旧"**：
目标在转，而这一帧没吃观测，模型拿 0.1~0.3 s 前的状态外推，2·R·sin(ωΔt/2) 在
R=0.28、ω≈1.5 rad/s、Δt=0.2 s 时正好是 8 cm 量级，和 99 mm 对得上。

**半径假设已被实测否掉**：连续用同帧双板弦长更新 R（`updateRadiusFromPairs`，
σ = 0.02/0.06/50 都试过），静止目标纹丝不动（本来就 7.4 mm），旋转目标反而从
99 mm 变差到 125~140 mm，v1 只从 412 → 380 mm。锁定时（`reinitializeFromPair`）
已经用双板定过 R，运行期再连续拉 R 会和位置更新抢"中心往哪挪"。

旋转目标那 10 cm 的大头不在"转"，在**半径**：同帧两块板都可见时，模型推出的
"另一块板"位置由 `R` 决定，先验 `R=0.25` 与真实半径差 7 cm 就会让第二块板偏 14 cm。
试过把换板更新的 R 放大到 50 倍（等于只吃瞄准槽位的观测）——残差反而从 99 mm 涨到
152 mm，说明"只用一块板"更糟；正确的做法是用同帧双板的弦长连续估计半径
（`|p_i − p_j| / |n_i − n_j|`，相邻两板 `|Δn| = √2`）。后来做了，见 §12.4：不是半径。

结论：静止目标的状态精度是 **7 mm 量级**（和 PnP 自身抖动同量级，说明 EKF 没引入
额外偏差）；旋转目标目前只有 **10 cm 量级**——这是"旋转目标不可用"的定量表述，
比"Stable 帧数还行"诚实得多。`v1` 的 0.4 m 来自目标切换：换目标的那几帧里，
"当前跟踪的 id"和"这一帧的观测"不是同一辆车，指标按上一帧的 id 过滤不掉。

### 12.2 试过但默认关闭：板面法向当相位观测

`RotatingArmorEkf::updatePhase(measured_world_yaw, sigma_rad)` 把板面法向塞进滤波器，
残差按 `2π/N` 折叠——换板时差一个整周期，折叠后为 0，所以它**不依赖"这一帧看到
的是哪块板"**，这正是位置关联最脆弱的地方。

合成 A 场景实测：打开后 yaw 在相邻帧抖动 ±0.4 rad、半径被压到 0.18 m（真值 0.25），
中心误差也从 1.6 mm 退化到 20 mm。原因是位置更新和相位更新对"哪个槽位"的归属
互相拉扯：位置更新按槽位算，相位更新按折叠算，两者给 `θ` 的信息不一致。

因此 `TrackerConfig::phase_update_enabled` 默认 `false`，能力保留（`tests` 里
`ekf_init_contract` 覆盖同板/换板/偏差三种情形）。要默认打开，得先改成"按槽位
一致"的写法：用位置更新认定的槽位 `s`，残差取 `wrap(θ + π + s·2π/N − ψ)`，
但那和位置更新本身高度相关，增益要重新推导。

### 12.3 自适应过期窗口

`stale_after_s`（默认 0.10 s）是"距上次被接收更新超过多久判 `Unstable`"。固定秒数
在两端都不合适：166 fps 下 0.10 s 等于 17 帧（旋转目标早转过 90°），30 fps 下又
刚好 3 帧。现在取

```
window = min(stale_after_s, max(stale_min_s, stale_min_frames × dt_median))
```

`dt_median` 是最近 9 帧间隔的中位数（`stale_min_frames` 5、`stale_min_s` 0.05 s）：
166 fps 下窗口是 0.05 s，30 fps 下是 0.10 s。代价是 `Stable` 帧数略降
（`v4 --only-id 3` 716 → 684、`v1` 331 → 315），换来的是"高帧率下不会拿着 17 帧
前的目标当新鲜"。

### 12.4 10 cm 的机理：相位残差是均匀分布

用 `--dump` 把"每帧实测板心 vs 模型 4 块板的最近距离"导出，再算实测板心相对
状态中心 `c` 的相位残差（对 `c_yaw + k·π/2` 取模）：

| 场景 | R 中位 | 相位残差 p10/p90 | 最近板距中位 |
| --- | --- | --- | --- |
| `v4` 静止（1384 帧） | 0.247 m | −1.1° / +0.9° | 7 mm |
| `v4 --only-id 3`（923 帧） | **0.197 m** | **−34° / +35°** | **99 mm** |

静止时相位几乎钉死（±1°），旋转时是**接近均匀的 ±35°**（直方图 −40°…+40° 每 10°
一档 84~162 行，没有峰）。相位残差 × `R` 正好是最近板距：`0.197 × 2·sin(17.5°) ≈ 0.12 m`。

这说明 99 mm 不是半径误差、不是重锁延迟（帧间隔 1 的中位仍 93 mm），而是
**单板可见时 yaw 在 ±35° 内游走**——一块板的位置只约束 2 个自由度，
`(c, θ)` 沿"绕板心转"这条零空间方向几乎不可观；相位更新（§12.2）想补这一维，
但它是绕着 `c` 转的，等于把板挪走、再被位置更新拉回来，所以打架。

试验"反锚"版本（改完 yaw 把 `c` 平移回去、让板留在原地）：

| 配置 | 中位 | p90 | p99 |
| --- | --- | --- | --- |
| 关闭（基线） | 99 mm | 181 mm | 995 mm |
| 相位 0.12 rad | 87 mm | 235 mm | 425 mm |
| 相位 0.30 rad | 91 mm | 213 mm | 372 mm |
| 相位 1.0 rad | 107 mm | 728 mm | 2220 mm |

中位降了 12%，尾巴却变差——板面法向本身的噪声（6 cm 板、3 m 外约 2~4°）
足以把中心抖起来。因此仍默认关闭，但"反锚"这个写法留在 `updatePhase` 调用处：
它至少把"相位更新与位置更新抢槽位"这个结构性问题解决了，剩下的是观测噪声。

## 13. 可视化验收

模型板直接投回图像——这是最省事、也最不会自欺的验收方式：

```bash
# 窗口实时看（需要显示器；esc/q 退出）
./build/test_camera_playback tmp/v4.avi --red --only-id 3 --ekf

# 无头落盘：每 N 帧存一张标注图到 build/replay_marks/
./build/test_camera_playback tmp/v4.avi --red --only-id 3 --ekf --headless --dump-every 60
```

图中元素：

| 记号 | 含义 |
| --- | --- |
| 绿框 + `ID=n S/B conf` | YOLO 检测 |
| 橙/红圆圈 `M0…M3` | **EKF 状态推出的 4 块模型板**（红色 = 当前瞄准槽位 `aim_slot`） |
| 品红十字 | 模型车体中心 `(cx, cy, z)` |

判读标准：**模型板圆圈压住真实装甲板中心 = 状态对**。3 m 外 1° ≈ 25 px（1440 宽、
约 60° 视场），所以：

- 圆圈与板心差 **< 10 px** ≈ 残差 < 4 mm，等同静止目标水平；
- 差 **40~50 px** ≈ 10 cm，就是旋转目标当前的中位残差；
- 圆圈突然跳到另一块板上 = 换板/重锁帧，此时看 `EKF:` 行里的 `phase` 是否
  `Tracking`（`Collecting` 表示还没锁定，画的仍是上一段状态）。

配套的数字口径：跑完打印的 `预测命中评估: …` 一行（中位/p90/p99 就是上表的
来源），需要逐帧排查时加 `--dump build/dump.csv` 导出 CSV。

## 14. 旋转目标"状态飞掉"的定位与修复（提交 6ac0b73）

### 14.1 现象与误判

`--dump-every` 的标注帧里模型板点飞到画面外（品红十字离车 2 m 以上）。先核实
"点是不是画错了"：dump 的 `cx,cy` 与十字像素位置一致，**飞的是 EKF 状态本身**，
不是投影 bug。

### 14.2 真因：按 id 关联，而 id 覆盖 4 块板

v4 里旋转车 4 块板共用 `id=3`，`--only-id 3` 把 4 块板全留下。同一段数据里
相邻帧观测位移中位 0.052 m、p90 0.358 m，而同帧两条观测间距 0.405 m —— 6 ms
内"板动了"的幅度经常等于半块板的间距，即**每帧拿到的是不同的板**。滤波器把它
当同一块板做位置更新，状态被来回拽，外推出去的另外 3 块模型板就飞到空中。

对照：静止目标同帧位移只有 3.8 mm，跟踪残差 7 mm —— 差两个数量级，说明不是
估计器调参问题，是关联粒度问题。

### 14.3 修复（4 条，均可用配置回退）

| 改动 | 原因 | 实测效果 |
| --- | --- | --- |
| 去掉"换板时放大 sigma"（`handoff_sigma_scale` 废弃） | 单板可见时每帧都可能换板，放大 sigma 等于每帧弱更新，ω 永远学不上去 | 换板帧从"几乎没信息"变成正常更新 |
| 新增 `association_sigma_floor_m{0.06}` | 模型误差（半径/相位/中心系统偏差）不在 P 里，σ 取 1 cm 会让 NIS 门把有信息的帧全拒（466/466） | 接受帧回升，p90 0.36→0.17 |
| 速度先验 1.0 → 0.05 m/s | 单板可见时"板在动"与"车在动"不可分，速度放开后滤波器把旋转解释成平移，中心以 1 m/s 飞出 1.4 m | 中心不再跑飞 |
| 新增 `freeze_radius{true}` | 单板下 (中心, 半径) 退化，R 从 0.25 被拉到 0.147，中心随之偏移 | R 钉在板型先验，稳定帧 +10% |

另外：更新后把 `aim_slot_` 重绑到刚观测到的板（4 块板物理等价，绑在看得见的
那块误差最小），对外瞄准点不再指向模型外推 90° 的另一块板。

### 14.4 结果（命令同 §13）

| 数据 | 稳定帧 | 中位残差 | p90 |
| --- | --- | --- | --- |
| v4 静止 | 1374/1415 | 7.3 mm | 16.7 mm |
| v4 旋转（R=0.25 先验） | 744/1415 | 114 mm | 240 mm |
| v4 旋转（`--radius 0.20`） | 744/1415 | 87 mm | 147 mm |
| v1 遮挡/双目标 | 399/608 | 296 mm | 703 mm |

### 14.5 剩下的硬限制

- 旋转目标单帧观测噪声大：相邻帧位移中位 52 mm 就是观测本身的抖动，滤波器
  无法比观测更准；要再降只能靠多帧平滑或把检测的角点抖动压下去。
- 角速度仍常在 0 附近：`--only-id 3` 下可见的板不连续（遮挡），相位序列没有
  连续斜率可拟合；一旦连续锁定≥12 帧会自动切到 Spinning。
- id3 仍有 ~25% 帧被 NIS 门拒（观测噪声 6 cm vs 预测 1 cm，属正常取舍）。
