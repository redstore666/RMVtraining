# 真数据回放调参脚本备份

这些脚本原先散落在 `build/`（被 .gitignore 忽略、`rm -rf build` 就没了），
现在备份到这里。全部以 **仓库根目录** 为工作目录运行，依赖已构建的
`build/test_camera_playback` 和 `tmp/v4.avi`、`tmp/v1.avi`（视频也被 .gitignore
忽略，在 `tmp/` 下，机器重启不会丢，但换机器要重新拷）。

用法：`bash tools/replay_scripts/<脚本名>.sh`

## 一、三组基线

| 脚本 | 作用 |
| --- | --- |
| `run_all.sh` | 三组数据（v4 静止 / v4 `--only-id 3` / v1）一行一组，打印预测命中评估与瞄准板残差 |
| `g6.sh` | 同上，只取中位数，最常用的一眼对比 |
| `s1.sh` | 打印瞄准板残差的完整分状态（Idle/Collecting/Tracking）与分 ω 段 |

## 二、参数扫描

| 脚本 | 扫的参数 | 结论（已写入 docs/EKF设计说明.md） |
| --- | --- | --- |
| `sweep.sh` | `--radius` 0.18/0.20/0.22/0.24 | 半径先验只解释 ~10 mm，不是主因（§13） |
| `floor.sh` | `--sigma-floor` 0.02/0.04/0.06/0.10 | 0.06 最优（§19） |
| `phase_ab.sh` / `phase2.sh` / `pa.sh` | `--phase-yaw --phase-sigma`（含 `--no-phase-anchor`） | 相位 yaw 融合一律更差（§19/§22） |
| `ab.sh` / `v1sweep.sh` | `--v-sigma` | v1 是镜头运动，0.5 明显好；但 v4 会变差，默认仍 0.05 |
| `ao.sh` / `aw.sh` | `--aim-obs`（新鲜观测直用的 hold 窗口） | 从 0 到 12 ms 都可用；≤0.002 s 时合成场景 G 会退化 |
| `bias.sh` | `--bias` 0 / 0.03 | bias=0 瞄准残差 0 mm；0.03 → 86 mm（误差全在外推那一步，§22） |
| `omegasweep.sh` | `--omega` 0/10/20/30/40/50 | 强制 ω 反而更差（§22） |
| `ab2.sh` | 无参数，三组一起打印 Stable 帧数 | 提交前的快速回归 |

## 三、dump 分析（Python）

`--dump build/x.csv` 导出的列是：
`frame,t,det,target_id,aim_slot,obs_id,x,y,z,best_slot,best_dist,second_dist,cx,cy,cz,cyaw,R,oyaw`

| 脚本 | 作用 |
| --- | --- |
| `pred.py` | 用第 f 帧的瞄准点对第 f+5 帧观测做"到期对账"，替代被 30 ms bias 污染的旧指标 |
| `pred2.py` / `pred3.py` | 上一行的迭代版本（分测量状态 / 分 ω 段） |
| `sum3.py` | 三组 dump 的汇总统计 |

## 四、复现一次完整回归

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest -E camera_grab --test-dir build          # 期望 8/8
bash tools/replay_scripts/g6.sh                # 三组中位数
bash tools/replay_scripts/bias.sh              # bias 0 / 0.03 对照
```

## 五、当前基线数字（本轮实测，2025-10-18）

默认配置 `--bias 0.0`（注意：bias=0 时瞄准点直接取本帧观测，残差恒为 0，属自证，
不能当精度指标；有意义的对照是 `--bias 0.03`）。

`--bias 0.03` 口径（`bias.sh` 输出）：

| 数据 | 帧数 | 瞄准残差中位 | p90 | ≤5 cm | 其中"本帧有观测" |
| --- | --- | --- | --- | --- | --- |
| v4 静止 | 1197 | 23.3 mm | 868 mm | 53.1% | 1190 帧，24.1 mm |
| v4 `--only-id 3` | 745 | **86.4 mm** | 143 mm | 22.4% | 742 帧，86.8 mm |
| v1 | 492 | 158 mm | 834 mm | 19.9% | 313 帧，97.9 mm；无观测 179 帧 606 mm |

`--bias 0.0` 口径的"预测命中评估"（瞄准点对下一帧观测，不是自证）：

| 数据 | 预测命中评估中位 |
| --- | --- |
| v4 静止 | 61.2 mm |
| v4 `--only-id 3` | 83.8 mm |
| v1 | 163.8 mm |

判读：v4 自转车在 **有观测的帧** 上也已经是 86.8 mm，说明误差不是"没观测只能外推"
造成的，而是状态相位本身偏 ~20°（§19/§22 有分解）。v1 的大头则在无观测段（606 mm）。
