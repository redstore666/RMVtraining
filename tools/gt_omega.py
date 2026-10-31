#!/usr/bin/env python3
"""从 POSE_DEBUG 日志提取被选车 omega 的"真值参考"（ω_ref），对比 EKF 的 w。

跑：
    POSE_DEBUG=1 tools/run_tool.sh gt3 25 -- --id 3   # 写 out/gt3.log
    tools/gt_omega.py out/gt3.log --id 3
    tools/gt_omega.py out/gt1.log --id 1              # 静止车应 ≈ 0

原理（方向统计周期图，2026-10-24 三次试错后定稿）：
  位置路线全部失败：EKF 车心逐帧抖 ±0.15m（绕它算方位角斜率是 ±25 rad/s
  垃圾）；观测板心"同板段"也切不开——斜视角 PnP 深度误差让相邻帧板心跳
  0.10~0.15m，而 8 rad/s 真板每帧只动 1.3cm，位置关联不可靠。
  法向没有这些问题：它不依赖车心、不受平移影响、与深度误差解耦。四块板
  90° 对称 → 用 4θ 折叠（e^{i·4θ}），换板跳变和 ±45° 折叠翻转在 4θ 域
  完全消失（θ=+44° 与 -46° 的 4θ 相同 mod 360°）。对候选 ω 找相干峰：
      C(ω) = |Σ_j e^{i·4·(θ_j − ω·t_j)}| / N
  峰值处的 ω 即该窗转速（法向噪声 ~20° 只压低峰值、不偏移峰位）。

判读：
  * coh 高（>0.3）= 该窗法向确实在匀速旋转，ω_ref 可信；
  * 静止车：法向恒定 → 峰在 ω≈0；
  * coh 低 = 该窗没有一致的旋转信号（乱转/断帧），ω_ref 仅供参考。
  * --video-loop 的时间会回卷，按 t 回卷自动切段。
"""
import argparse
import math
import re
import statistics
import sys

import numpy as np

# 格式串里 %+7.1f 会在 '=' 后打出空格（如 "bear= +162.7"），所以统一允许 \s*。
STATE = re.compile(
    r"\[state\] ts=\s*([+-]?[\d.]+) c=\(\s*([+-]?[\d.]+),\s*([+-]?[\d.]+)\) z=\s*([+-]?[\d.]+) "
    r"R=\s*([\d.]+) yaw=\s*([+-]?[\d.]+) w=\s*([+-]?[\d.]+)")
POSE = re.compile(
    r"\[pose\] id=\s*(\d+) x=\s*([+-]?[\d.]+) y=\s*([+-]?[\d.]+) bear=\s*([+-]?[\d.]+) "
    r"normal=\s*([+-]?[\d.]+) normal-bear=\s*([+-]?[\d.]+)")


def parse(path):
    """返回段列表；每段是 [(t, w_ekf, [(id, normal_rad), ...]), ...]。

    一个 [state] 后跟 0~n 条 [pose]（同车多板各一条，同 id）；normal 全收——
    4θ 折叠让多板/换板天然等价，不需要挑。
    """
    segments = []
    cur = []
    last_t = None
    cur_row = None
    for line in open(path, errors="ignore"):
        m = STATE.search(line)
        if m:
            t = float(m.group(1))
            if last_t is not None and t < last_t - 1.0:  # --video-loop 回卷
                if cur:
                    segments.append(cur)
                cur = []
            last_t = t
            cur_row = [t, float(m.group(7)), []]
            cur.append(cur_row)
            continue
        m = POSE.search(line)
        if m and cur_row is not None:
            cur_row[2].append((int(m.group(1)), math.radians(float(m.group(5)))))
    if cur:
        segments.append(cur)
    return segments


def periodogram_omega(ts, thetas, w_max=12.0, w_step=0.02):
    """4θ 折叠相干峰找转速。返回 (omega_ref, coherence)。"""
    if len(ts) < 20:
        return float("nan"), 0.0
    t0 = ts[0]
    t = np.asarray(ts) - t0
    th4 = 4.0 * np.asarray(thetas)
    grid = np.arange(-w_max, w_max + w_step, w_step)
    # C(w) = |mean exp(i*4*(th - w*t))| —— 对每帧（含同帧多板）求均值
    phase = th4[:, None] - 4.0 * grid[None, :] * t[:, None]
    coh = np.abs(np.exp(1j * phase).mean(axis=0))
    k = int(np.argmax(coh))
    return float(grid[k]), float(coh[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--id", type=int, default=0, help="被选车 id（默认取出现最多的）")
    ap.add_argument("--window", type=float, default=2.0, help="滑窗长（秒）")
    ap.add_argument("--step", type=float, default=0.5, help="滑窗步进（秒）")
    ap.add_argument("--min-pts", type=int, default=30, help="窗内最少法向样本数")
    ap.add_argument("--w-max", type=float, default=12.0, help="转速搜索范围（rad/s）")
    args = ap.parse_args()

    segments = parse(args.log)
    if not segments:
        sys.exit("没有解析到 [state]/[pose] 行——日志是 POSE_DEBUG=1 跑出来的吗？")
    if args.id == 0:
        from collections import Counter
        cnt = Counter(i for seg in segments for _t, _w, ps in seg for i, _n in ps)
        if not cnt:
            sys.exit("日志里没有 [pose] 行")
        args.id = cnt.most_common(1)[0][0]
        print(f"未指定 --id，取出现最多的 id={args.id}")

    for si, seg in enumerate(segments):
        n_obs = sum(len([1 for i, _n in ps if i == args.id]) for _t, _w, ps in seg)
        if n_obs < args.min_pts:
            print(f"\n段 {si + 1}: 样本不足（{n_obs}），跳过")
            continue
        t0, t1 = seg[0][0], seg[-1][0]
        print(f"\n段 {si + 1}: t={t0:.2f}~{t1:.2f}s  被选车法向样本 {n_obs}")
        print(f"  {'t窗中点':>8}  {'w_ref(rad/s)':>12}  {'coh':>6}  "
              f"{'EKF w 中位':>10}  {'差':>8}")
        w_refs, cohs = [], []
        w_start = t0
        while w_start + args.window <= t1:
            w_end = w_start + args.window
            ts, ths, ekf_ws = [], [], []
            for t, w, ps in seg:
                if not (w_start <= t < w_end):
                    continue
                for i, n in ps:
                    if i == args.id:
                        ts.append(t)
                        ths.append(n)
                ekf_ws.append(w)
            if len(ts) >= args.min_pts:
                w_ref, coh = periodogram_omega(ts, ths, args.w_max)
                ekf_med = statistics.median(ekf_ws) if ekf_ws else float("nan")
                w_refs.append(w_ref)
                cohs.append(coh)
                mark = "" if coh >= 0.3 else "  (coh 低，仅供参考)"
                print(f"  {0.5 * (w_start + w_end):>8.2f}  {w_ref:>12.3f}  "
                      f"{coh:>6.2f}  {ekf_med:>10.2f}  {ekf_med - w_ref:>8.2f}{mark}")
            w_start += args.step
        if w_refs:
            strong = [w for w, c in zip(w_refs, cohs) if c >= 0.3]
            pool = strong if strong else w_refs
            med = statistics.median(pool)
            pos = sum(1 for v in pool if v > 0.2)
            neg = sum(1 for v in pool if v < -0.2)
            print(f"  汇总（coh≥0.3 的窗 {len(strong)}/{len(w_refs)}）: "
                  f"|w_ref| 中位 {abs(med):.2f} rad/s，"
                  f"窗符号 正{pos}/负{neg}/近零{len(pool) - pos - neg}")


if __name__ == "__main__":
    main()
