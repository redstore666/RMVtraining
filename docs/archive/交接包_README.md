<!--
归档说明（2026-10-31）：本文原在 tmp/handoff/，用于交接当时的 omega 问题。
tmp/ 已整体移出版本库，脚本迁到 tools/（analyze_sse.py、dump_mini.sh、gt_omega.py、
parse_poses.py、run_tool.sh、summarize_ev.py），本文移到这里存档。
文中出现的 tmp/handoff/ 路径、提交号、ctest 计数均为当时的快照，**不代表现状**；
现状以 README.md 与 docs/omega相位拟合说明.md 为准。
-->

# tmp/handoff —— 交接包

先看 **[HANDOFF.md](HANDOFF.md)**，那里是全部背景、已确证的结论、已证伪的方案、下一步建议和环境坑。

```
HANDOFF.md              交接正文（务必先读 §6 已证伪的方向，别重做）
tools/run_tool.sh       跑一次工具 + 抓 SSE（唯一可靠的跑法：随机端口、同调用内起停）
tools/analyze_sse.py    SSE -> 对比表（dYaw/净转角/omega/lag/R/pair_solves/零更新率）
tools/summarize_ev.py   TRACK_DEBUG 日志 -> 转速窗口接受/拒绝统计 + used 斜率分布
tools/parse_poses.py    POSE_DEBUG 日志 -> normal-bear 偏置、R/w 分布
tools/dump_mini.sh      抓迷你俯视图 PNG（每 N 帧一张）
out/                    历史 A/B 数据（base / pll / pair_gate095 / snap_unwrap / 报告原始 dump）
```

全部工具都能从这里直接跑，不用先 `cd`（脚本自己会回到仓库根）：

```bash
cd /home/oliver/coding/learn/RMVtrainingHomework/RMVtraining
tmp/handoff/tools/run_tool.sh base 25
python3 tmp/handoff/tools/analyze_sse.py out/base.sse
```

注意：`tmp/` 在 `.gitignore` 里，**这个包只存在于本机工作区**。要带到别的机器/新 clone，把 `tmp/handoff/` 整个拷走，或者把 `HANDOFF.md` 提交进 `docs/`。
