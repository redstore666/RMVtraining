# ω 诊断工具（原 tmp/handoff，2026-10-25 入库）

交接包（2026-10-24）带来的转速诊断脚本，此前只存在于 tmp/（被 gitignore），
现收编入库以便长期使用。注意：脚本按"工作目录 = 原 tmp/handoff"的习惯写
（输出默认进 out/，POSE_DEBUG 日志路径自行指定），从仓库根目录跑时注意路径。

| 工具 | 用途 |
|---|---|
| `gt_omega.py` | 从 `POSE_DEBUG=1` 的日志提取 ω_ref"真值参考"（法向 4θ 折叠周期图，不依赖车心/不受平移影响），对比 EKF 输出。静止车应 ≈0。用法见文件头 docstring |
| `summarize_ev.py` | 汇总 `TRACK_DEBUG` 的 `[ev]` 证据行（斜率/σ/采用率/拒绝类计数；含 nan 解析修复） |
| `analyze_sse.py` | 抓包/解析网页 `/data` SSE 流，批量统计各键分布 |
| `parse_poses.py` | 解析 `[pose]`/`[state]` 调试行（ts 时间轴供离线分析） |
| `run_tool.sh` | 带超时跑 test_camera_system 并落日志的封装 |
| `dump_mini.sh` | 抓迷你俯视图帧序列 |
| `HANDOFF.md` | 2026-10-24 交接包原文（§7 修复排序与背景，历史材料） |

注：锁频通道已删（任务进度备份§35，commit bce1e87）；重建时的对照信号
obs_rate（白十字观测净转速）仍在 test_camera_system 主循环里。
