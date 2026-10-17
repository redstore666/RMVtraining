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

| 视频 | 内容 | 结果 |
| --- | --- | --- |
| `v4.avi` | 静止大板（id=1，1.98~2.08 m）+ 旋转小板（id=3，1.0~1.9 m），相机固定 | 帧 50 锁定 id=1，`Stable` 1354/1415，重锁 0、目标切换 0；锁定为 `Static`、`ω=0`、半径用先验 0.25；位置抖动 3.8 mm |
| `v4.avi --only-id 3` | 只保留旋转目标 | 帧 172 锁定，但后续更新全部被拒（`Stable` 117/1415），不可用 |
| `v1.avi` | 手持晃动，id=1 大板与 id=3 小板先后可见 | 不稳定：2 次目标切换、1 次重锁，末端半径状态漂到 0.68 m |

旋转目标不可用的根因（不是调参能修的）：

1. 车体每转 90° 换一块板露出，检测在同一编号的两块板之间逐帧交替；`yaw` 序列
   因此逐帧跳 `π/2`，按相邻帧解缠会得到 ~8 rad/s 的伪角速度，按圆拟合又会因为
   弧段太短（0.18~0.30 s）而病态。
2. 换板瞬间的"初值"用的是单帧板面法向，实测小目标法向抖动可达 10°以上，中心
   初值偏差接近 `R`；随后的真实观测落到 `χ²(3,99.5%)` 门限外侧，于是出现
   "锁定成功、更新全拒"。
3. 已知可走的方向：锁定阶段按"同一块板"的连续聚类（位置 + 法向同时连续）而不是
   按编号聚类；把两块可见板同时喂给滤波器（观测模型本来就支持多槽位）；或者
   用车身相位（模 `π/2`）作为主观测、位置作为辅助。这三条都超出当前实现范围，
   留作后续工作。

对验收的影响：静止/低速平移目标（`ekf_tracker` 场景 A/B/F 与 `v4` 的静止车）表现
正常；高速自转目标目前只能给出"锁定 + 短时外推"，不满足"旋转车体稳定跟踪"的
验收要求，需要在上述方向上继续做。
