# 借鉴 sp_vision_25 的整车 EKF（本次改动说明）

参考仓库：`tmp/clones/sp_vision_25`（同济 TongjiSuperPower/sp_vision_25）。
本次只做**结构移植**，没有跑 v4 的指标（按你的要求先不测试）。

## 一、对照表：改了什么、抄的是哪一行

| # | 改动 | 我们原来 | 现在 | sp_vision_25 出处 |
|---|---|---|---|---|
| 1 | 状态维度 | 9 维：`[cx,vx,cy,vy,z,vz,yaw,ω,r]` | **11 维**，新增 `radius_diff`(l)、`height_diff`(h) | `tasks/auto_aim/target.cpp:34-39` |
| 2 | 板半径/高度 | 四块板同一个 `r`、同一个 `z` | 偶数槽位 `r`，奇数槽位 `r+l`；高度 `z` / `z+h` | `target.cpp:267-278` |
| 3 | 量测方程 | 板心 xyz 直接比 | **球坐标** `(yaw, pitch, distance)` + **板面法向角** | `target.cpp:187-222`、`tools/math_tools.cpp:103` |
| 4 | 量测噪声 R | 各向同性常数 | 角度维由位置噪声按距离折算；测距方差随板侧向程度增长；法向方差随距离增长 | `target.cpp:192-196` |
| 5 | 数据关联 | 只比 3-D 距离（另有一个默认关的法向 tie-break） | 距离 **+** 0.30·(法向角差 + 方位角差) | `target.cpp:140-169` |
| 6 | 同帧多板 | `max_updates_per_frame = 2` | **4**（每块匹配上的板都做一次完整更新） | `tracker.cpp:286-293` |
| 7 | NIS 门限 | chi2(3) 99.5% = 11.34 | chi2(4) 99.5% = **14.86**，且新息与 update 用同一方程 | `target.cpp:192-231` |

## 二、为什么这四项是配套的

我们之前在 v4 上做过的失败实验，恰好都能被这套结构解释：

1. **相位/槽位对不上**（模型板停在相邻那一面）→ 第 5 项：四块板位置相隔 0.4 m，
   位置门（0.6 m）对两块板同时放行，光比距离在原理上分不清"看到的是哪一面"。
   sp_vision 用板法向角 + 方位角做关联，这正是我们 `slot_from_normal` 想做但
   做砸了的那件事——差别在于**它们是拿法向去"选观测"，再让滤波器吃进去**，
   而我们那次是拿法向直接**改槽位分配并把不一致的观测全部拒掉**。

2. **绿板转心偏高、蓝板转半圈卡回去** → 第 1、2 项：真实车体的前后两对板
   并不共圆。用一个半径画四块板，必然有一对系统性偏内/偏外，表现出来就是
   "圆心看起来偏了"。我们之前用 per-slot 自由偏移去拟合，结果两块同 id 板
   交替时偏移来回震荡（已回退）；改成两个**刚体参数** l、h 就不会震荡。

3. **改了半径反而变差**（`--radius 0.28` 只有 91.8 px）→ 第 4 项：半径错的时候
   深度维误差被当成角度误差吃进去了。球坐标把深度单独一维、并给它一个
   随侧向程度增长的大方差，半径的先验偏差就不会再污染 yaw。

## 三、代码落点

- `student/include/RotatingArmorModel.hpp`
  - `kStateDim = 11`、`kSphericalMeasurementDim = 4`
  - 新增 `SphericalMeasurement` / `SphericalJacobian` 类型
  - 新增 `sphericalMeasurement()` / `sphericalJacobian()` 声明
  - `ProcessNoiseConfig` 新增 `pair_radius_random_walk`、`pair_height_random_walk`
- `student/src/RotatingArmorModel.cpp`
  - 新增内部 `plateRadius()` / `plateHeight()` / `platePhase()`，**奇偶槽位走不同半径/高度**
  - `measurement()` / `measurementJacobian()` 增加第 9、10 列的雅可比
  - 新增球坐标量测与其解析雅可比（链式：`d(yaw,pitch,dist)/dp · dp/dx`）
- `student/src/RotatingArmorEkf.cpp`
  - `initialize()`：`x(9)=x(10)=0`，协方差补两项 `pair_sigma=0.10 m`
  - `update()`：整条换成球坐标量测 + 自适应 R + 角度残差折叠 + `l/h` 夹紧
  - `innovationNis()`：与 `update()` 共用同一套量测方程与 R
  - 所有写死的 `9` 维度改成 `kStateDim`
- `student/include/RotatingArmorTracker.hpp`
  - `association_by_angle_enabled{true}`、`association_angle_weight_m_per_rad{0.30}`
  - `max_updates_per_frame` 2 → 4、`nis_gate` 11.34 → 14.86
- `student/src/RotatingArmorTracker.cpp`
  - 候选评分里加入角度项（法向差 + 方位差）
- `tests/*.cpp`：6 处 9 元逗号初始化补成 11 元；`Eigen::Matrix<double,9,1>` 改成 `State`

## 四、刻意没抄的三点（以及原因）

1. **纯角度关联**：sp_vision 是"距离筛最近 3 块 → 纯按角度挑"。我们仍然保留
   距离项（权重 0.30 m/rad），因为我们的候选循环是单趟的，没有预筛；
   纯角度会让"角度碰巧对上但隔了半米"的板赢。
2. **`radius` 初值给 1.0 方差**：它们的 P0 里 `r`、`l`、`h` 方差都是 1.0（完全放开）。
   我们用 0.05 / 0.10 m，因为我们的观测噪声比它们大得多，完全放开会在单帧内发散。
3. **前哨站/基地的特判与 2、3 板构型**：`l/h` 只在 `plate_count == 4` 时启用
   （它们也是 `armor_num_ == 4` 才用），2 板平衡车、3 板前哨站退回共圆假设。

## 五、待验证（下一步，未跑）

- 合成回归：`test_rotating_armor_model` 的雅可比数值校验、A/G/H 三组
- v4：`--only-id 3` 下的 PIX 中位（基线 90.0，body_circle 78.1）
- 重点看两件事：`l` 是否收敛到一个稳定的非零值；关联还换不换板
