#!/usr/bin/env python3
"""S2 A/B 分析：多假设（环序 + 高斯和）vs 原逐块 argmin。

只读两份 dump csv，不依赖 numpy。指标：
  1. 本帧是否走多假设路径（rhact）、主导假设分布、权重尖锐度
  2. 多板帧槽位翻转率（验收判据：41.6% -> <5%）
  3. EKF 转速 ekfw 的稳定性（相邻帧跳变量、|w| 分布）
"""
import sys
import math


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
    return header, rows


def by_frame(rows):
    """每个 frame 只留一行（frame 级量在所有行上重复）。"""
    frames = {}
    for row in rows:
        key = int(row['frame'])
        if key not in frames:
            frames[key] = row
    return [frames[k] for k in sorted(frames)]


def pct(values, q):
    if not values:
        return float('nan')
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def flip_rate(frames, require_multi=True):
    """相邻帧 chosen_slot 变化率。只在两帧都是多板帧时统计。"""
    changes = 0
    total = 0
    prev = None
    for row in frames:
        multi = row['ringk'] >= 2
        if require_multi and not multi:
            prev = None
            continue
        slot = int(row['tslot'])
        if slot < 0:
            prev = None
            continue
        if prev is not None:
            total += 1
            if slot != prev:
                changes += 1
        prev = slot
    return changes, total


def main():
    base_path, hyp_path = sys.argv[1], sys.argv[2]
    _, base_rows = load(base_path)
    header, hyp_rows = load(hyp_path)
    base = by_frame(base_rows)
    hyp = by_frame(hyp_rows)

    print('=== 帧数 ===')
    print('baseline 帧 %d / hypothesis 帧 %d' % (len(base), len(hyp)))

    print('\n=== 身份后验启用情况（hyp）===')
    active = [r for r in hyp if r.get('idact', 0) >= 1]
    multi = [r for r in hyp if r['ringk'] >= 2]
    print('多板帧 %d，其中走后验配对 %d（%.1f%%）' %
          (len(multi), len(active), 100.0 * len(active) / max(1, len(multi))))
    enum = [r['idn'] for r in active]
    print('每帧枚举配对数 p50 %.0f（k=2 应 12、k=3 应 24）' % pct(enum, 0.5))
    wbest = [r['idwbest'] for r in active]
    margin = [r['idmargin'] for r in active]
    print('最优后验 p10/p50/p90: %.4f / %.4f / %.4f' %
          (pct(wbest, 0.1), pct(wbest, 0.5), pct(wbest, 0.9)))
    print('相对 margin p10/p50/p90: %.3f / %.3f / %.3f' %
          (pct(margin, 0.1), pct(margin, 0.5), pct(margin, 0.9)))
    tie = sum(1 for w in wbest if w < 0.9)
    print('最优后验 <0.9 的帧（真打平）: %d / %d' % (tie, len(active)))

    print('\n=== 多板帧槽位翻转率（**不是有效指标**，仅留存对照）===')
    print('    tslot 是\'本帧被选中那块板的槽位\'，两块板都可见时随\'哪块更近\'\n'
          '    自然交替，与身份判对判错无关。验收请看工具汇总里的瞄准板残差 p90 / ≤5cm。')
    for name, frames in (('baseline', base), ('hypothesis', hyp)):
        changes, total = flip_rate(frames, require_multi=True)
        print('%-11s 相邻多板帧 %4d，翻转 %4d = %.1f%%' %
              (name, total, changes, 100.0 * changes / max(1, total)))
    for name, frames in (('baseline', base), ('hypothesis', hyp)):
        changes, total = flip_rate(frames, require_multi=False)
        print('%-11s 全部相邻帧   %4d，翻转 %4d = %.1f%%' %
              (name, total, changes, 100.0 * changes / max(1, total)))

    print('\n=== EKF 转速 ekfw ===')
    for name, frames in (('baseline', base), ('hypothesis', hyp)):
        rate = [r['ekfw'] for r in frames if 'ekfw' in r and math.isfinite(r['ekfw'])]
        if not rate:
            print('%-11s 无 ekfw 列' % name)
            continue
        absrate = [abs(v) for v in rate]
        jumps = [abs(rate[i] - rate[i - 1]) for i in range(1, len(rate))]
        big = sum(1 for j in jumps if j > 2.0)
        print('%-11s |w| p50 %.2f p90 %.2f p99 %.2f max %.2f；'
              '相邻帧跳变>2rad/s %d/%d (%.1f%%)，p90 跳变 %.2f' %
              (name, pct(absrate, 0.5), pct(absrate, 0.9), pct(absrate, 0.99),
               max(absrate), big, len(jumps), 100.0 * big / max(1, len(jumps)),
               pct(jumps, 0.9)))

    print('\n=== 车心 z（cz）与半径 R ===')
    for name, frames in (('baseline', base), ('hypothesis', hyp)):
        cz = [r['cz'] for r in frames if math.isfinite(r['cz'])]
        rad = [r['R'] for r in frames if math.isfinite(r['R']) and r['R'] > 0]
        print('%-11s cz p50 %.4f p10 %.4f p90 %.4f；R p50 %.4f p10 %.4f p90 %.4f' %
              (name, pct(cz, 0.5), pct(cz, 0.1), pct(cz, 0.9),
               pct(rad, 0.5), pct(rad, 0.1), pct(rad, 0.9)))


if __name__ == '__main__':
    main()
