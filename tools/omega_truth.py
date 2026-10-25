#!/usr/bin/env python3
"""独立转速估计（滑窗位置角最小二乘）vs EKF 转速。

为什么不用法向帧差：PnP 法向噪声实测 ±35°，除以 6 ms 帧间隔等于上百 rad/s，
出来的全是噪声（实测符号一致率 47%，就是抛硬币）。位置噪声只有 1~2 cm，
在 R≈0.17 m 上折合角度 ~7°，所以改用**位置角 + 滑窗最小二乘**：
    1) 车心取 dump 里的模型 cx/cy（外推量，只当参考原点，不参与斜率）；
    2) 每帧在"上一帧选中的那块板"附近挑观测（帧间位移 ~7 mm，不会挑错）；
    3) 逐帧用 fold90 展开角度（换板的 90° 整数倍被折掉）；
    4) 窗口内对 (t, θ) 做最小二乘斜率 -> ω。

用法: python3 tools/omega_truth.py <dump.csv> [窗口帧数，默认 61]
"""
import math
import sys


def load(path):
    with open(path) as handle:
        header = handle.readline().strip().split(',')
        rows = []
        for line in handle:
            parts = line.rstrip('\n').split(',')
            if len(parts) != len(header):
                continue
            row = {}
            for key, value in zip(header, parts):
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = float('nan')
            rows.append(row)
    return rows


def wrap(x):
    while x > math.pi:
        x -= 2.0 * math.pi
    while x <= -math.pi:
        x += 2.0 * math.pi
    return x


def fold90(x):
    q = math.pi * 0.5
    return x - q * round(x / q)


def pct(values, q):
    if not values:
        return float('nan')
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]


def main():
    path = sys.argv[1]
    window = int(sys.argv[2]) if len(sys.argv) > 2 else 61
    rows = load(path)

    # 按帧聚合：一帧可能有多行（同帧多块板）
    frames = {}
    for row in rows:
        frames.setdefault(int(row['frame']), []).append(row)
    keys = sorted(frames)

    # 每帧的候选观测 (x, y)
    obs = {}
    for key in keys:
        points = []
        for row in frames[key]:
            x, y = row['x'], row['y']
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
        if points:
            obs[key] = points
    usable = [k for k in keys if k in obs]
    if len(usable) < window:
        print('可用帧 %d 少于窗口 %d' % (len(usable), window))
        return

    half = window // 2
    est = []  # (t, omega_meas, ekf_omega)
    for centre in range(half, len(usable) - half):
        span = usable[centre - half: centre + half + 1]
        ts, angles = [], []
        prev = None
        theta = 0.0
        prev_angle = None
        ok = True
        for key in span:
            points = obs[key]
            if prev is None:
                chosen = points[0]
            else:
                chosen = min(points, key=lambda p: (p[0] - prev[0]) ** 2 + (p[1] - prev[1]) ** 2)
                # 帧间位移超过半块板间距(0.12m)说明跟丢了
                if math.hypot(chosen[0] - prev[0], chosen[1] - prev[1]) > 0.12:
                    ok = False
                    break
            row = frames[key][0]
            cx, cy = row['cx'], row['cy']
            if not (math.isfinite(cx) and math.isfinite(cy)):
                ok = False
                break
            angle = math.atan2(chosen[1] - cy, chosen[0] - cx)
            if prev_angle is None:
                theta = angle
            else:
                theta += fold90(wrap(angle - prev_angle))
            prev_angle = angle
            prev = chosen
            ts.append(row['t'])
            angles.append(theta)
        if not ok or len(ts) < window:
            continue
        # 最小二乘斜率
        n = len(ts)
        mean_t = sum(ts) / n
        mean_a = sum(angles) / n
        num = sum((t - mean_t) * (a - mean_a) for t, a in zip(ts, angles))
        den = sum((t - mean_t) ** 2 for t in ts)
        if den <= 0.0:
            continue
        omega = num / den
        if not math.isfinite(omega) or abs(omega) > 25.0:
            continue
        ekf_omega = frames[usable[centre]][0]['ekfw']
        est.append((frames[usable[centre]][0]['t'], omega, ekf_omega))

    if not est:
        print('没有可用窗')
        return

    meas = [abs(e[1]) for e in est]
    ekf = [abs(e[2]) for e in est if math.isfinite(e[2])]
    print('窗口 %d 帧（≈%.2f s），有效窗 %d 个' %
          (window, window / 166.0, len(est)))
    print('独立估计 |ω|  p10 %.2f  p50 %.2f  p90 %.2f  max %.2f' %
          (pct(meas, .1), pct(meas, .5), pct(meas, .9), max(meas)))
    print('EKF      |ω|  p10 %.2f  p50 %.2f  p90 %.2f  max %.2f' %
          (pct(ekf, .1), pct(ekf, .5), pct(ekf, .9), max(ekf)))

    fast = [(t, m, e) for t, m, e in est if abs(m) >= 1.0]
    print('\n|实测| ≥ 1 rad/s 的窗: %d / %d' % (len(fast), len(est)))
    if fast:
        mm = [abs(f[1]) for f in fast]
        ee = [abs(f[2]) for f in fast]
        print('  独立估计 |ω|  p50 %.2f  p90 %.2f' % (pct(mm, .5), pct(mm, .9)))
        print('  EKF      |ω|  p50 %.2f  p90 %.2f' % (pct(ee, .5), pct(ee, .9)))
        ratio = sorted(a / b for a, b in zip(mm, ee) if b > 0.05)
        print('  比值 实测/EKF  p10 %.1f  p50 %.1f  p90 %.1f' %
              (pct(ratio, .1), pct(ratio, .5), pct(ratio, .9)))
        for thr in (0.3, 0.5, 0.8):
            hit = sum(1 for f in fast if abs(f[2]) >= thr * abs(f[1]))
            print('  EKF ≥ %.0f%%×实测: %d/%d = %.1f%%' %
                  (thr * 100, hit, len(fast), 100.0 * hit / len(fast)))
        same = sum(1 for f in fast if f[1] * f[2] > 0)
        print('  符号一致: %d/%d = %.1f%%' % (same, len(fast), 100.0 * same / len(fast)))


if __name__ == '__main__':
    main()
