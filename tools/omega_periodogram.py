#!/usr/bin/env python3
"""用 90° 点阵周期图估计"视频里的真实转速"。

为什么不用差分/展开：v4 是 166 fps，ω≈1.5 rad/s 时每帧相位只前进 0.5°，
而 PnP 位置噪声折合角度 ~7°，逐帧差分与展开全部被噪声吞掉
（实测逐帧差分给出 28~37 rad/s，纯噪声）。

正确做法是**相干**：四块板在方位角上间隔 90°，所以
    (θ_ij − ω·t_i) mod 90°  ≡  θ0 mod 90°      对所有 i,j 成立
于是把 ρ(ω) = |mean(exp(j·4·(θ − ω·t)))| 在 ω 上做网格搜索，峰值就是 ω。
这个量对所有样本做相干叠加，不依赖逐帧展开，噪声被平均掉。
峰值高度（concentration）直接就是估计的可信度：1.0 = 完美相干。

用法: python3 tools/omega_periodogram.py <dump.csv> [窗长秒, 默认 1.0]
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


def concentration(samples, omega):
    """samples: [(t, theta)]，theta 为相对车心的方位角。返回 |mean(exp(j4(θ-ωt)))|。"""
    if not samples:
        return 0.0
    re = 0.0
    im = 0.0
    for t, theta in samples:
        phase = 4.0 * (theta - omega * t)
        re += math.cos(phase)
        im += math.sin(phase)
    return math.hypot(re, im) / len(samples)


def main():
    path = sys.argv[1]
    window_s = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    rows = load(path)

    frames = {}
    for row in rows:
        frames.setdefault(int(row['frame']), []).append(row)
    keys = sorted(frames)

    # 每帧的 (t, [θ...])，θ 用模型车心做原点
    per_frame = []
    for key in keys:
        rows_k = frames[key]
        cx, cy, t = rows_k[0]['cx'], rows_k[0]['cy'], rows_k[0]['t']
        if not all(map(math.isfinite, (cx, cy, t))):
            continue
        thetas = []
        for row in rows_k:
            x, y = row['x'], row['y']
            if math.isfinite(x) and math.isfinite(y):
                thetas.append(math.atan2(y - cy, x - cx))
        if thetas:
            per_frame.append((t, thetas, rows_k[0]['ekfw']))
    if not per_frame:
        print('没有可用帧')
        return

    grid = [(-6.0 + 0.005 * i) for i in range(int(12.0 / 0.005) + 1)]
    t0 = per_frame[0][0]
    windows = []
    start = 0
    while start < len(per_frame):
        t_start = per_frame[start][0]
        end = start
        while end < len(per_frame) and per_frame[end][0] - t_start < window_s:
            end += 1
        if end - start < 30:
            break
        samples = []
        ekf_sum = 0.0
        ekf_n = 0
        for t, thetas, ekf in per_frame[start:end]:
            for theta in thetas:
                samples.append((t - t0, theta))
            if math.isfinite(ekf):
                ekf_sum += ekf
                ekf_n += 1
        best_w, best_c = 0.0, -1.0
        for omega in grid:
            c = concentration(samples, omega)
            if c > best_c:
                best_c, best_w = c, omega
        windows.append((per_frame[start][0], best_w, best_c,
                        ekf_sum / ekf_n if ekf_n else float('nan'), end - start))
        start = end

    if not windows:
        print('窗口不足')
        return

    print('窗长 %.2f s，共 %d 个窗' % (window_s, len(windows)))
    print('  t(s)   真实ω(周期图)  相干度   EKFω    帧数')
    for t, w, c, e, n in windows:
        print('  %6.2f   %+7.2f      %.3f   %+7.2f   %4d' % (t, w, c, e, n))

    good = [(t, w, c, e) for t, w, c, e, n in windows if c >= 0.5]
    if good:
        meas = sorted(abs(g[1]) for g in good)
        ekf = sorted(abs(g[3]) for g in good if math.isfinite(g[3]))

        def pct(v, q):
            return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]

        print('\n相干度 ≥ 0.5 的窗: %d / %d（这些才是可信估计）' % (len(good), len(windows)))
        print('  真实 |ω|  p10 %.2f  p50 %.2f  p90 %.2f' %
              (pct(meas, .1), pct(meas, .5), pct(meas, .9)))
        print('  EKF  |ω|  p10 %.2f  p50 %.2f  p90 %.2f' %
              (pct(ekf, .1), pct(ekf, .5), pct(ekf, .9)))
        ratio = sorted(b / a for a, b in zip(meas, ekf) if a > 0.3)
        print('  比值 EKF/真实  p10 %.2f  p50 %.2f  p90 %.2f' %
              (pct(ratio, .1), pct(ratio, .5), pct(ratio, .9)))
        same = sum(1 for g in good if g[1] * g[3] > 0)
        print('  符号一致: %d/%d = %.1f%%' % (same, len(good), 100.0 * same / len(good)))


if __name__ == '__main__':
    main()
