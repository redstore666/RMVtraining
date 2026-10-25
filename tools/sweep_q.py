#!/usr/bin/env python3
"""扫描 process 角加速度常量（SPIN_Q）对瞄准/转速的影响。

为什么单独扫这一维：`applyProcessNoise` 里有一段按 |ω| 分档的"线性软门"
（floor=1.5 / full=3.0 rad/s），低于 floor 就把 angular_accel 压回 0.01
——也就是"转得慢的时候不许转速变化"。SPIN_Q 环境变量会整段绕开这段门，
等于直接给一个常量 angular_accel = 0.01 * SPIN_Q。

用法: python3 tools/sweep_q.py v4 v1 --q 1 100 1000 4000 [--sigma 2.5]
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, 'build', 'test_camera_playback')


def pct(values, q):
    if not values:
        return float('nan')
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))]


def read_dump(path):
    header = None
    rows = []
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip('\n').split(',')
            if header is None:
                header = parts
                continue
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


def main():
    clips = []
    qs = []
    sigma = None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--q':
            i += 1
            while i < len(args) and not args[i].startswith('--'):
                qs.append(args[i])
                i += 1
            continue
        if args[i] == '--sigma':
            sigma = args[i + 1]
            i += 2
            continue
        clips.append(args[i])
        i += 1
    if not clips or not qs:
        print(__doc__)
        return 1

    out_dir = os.path.join(ROOT, 'tmp', 's2')
    os.makedirs(out_dir, exist_ok=True)
    print('%-6s %-7s %-9s %-11s %-9s %s' %
          ('clip', 'SPIN_Q', 'ang_accel', 'aim p50/p90', '<=5cm', '|w| p50/p90'))
    for clip in clips:
        video = os.path.join(ROOT, 'tmp', clip + '.avi')
        for q in qs:
            tag = '%s_q%s' % (clip, q)
            dump = os.path.join(out_dir, 'c_' + tag + '.csv')
            env = dict(os.environ)
            env['SPIN_Q'] = q
            if sigma:
                env['RMV_ANGLE_SIGMA'] = sigma
            proc = subprocess.run([BIN, video, '--red', '--only-id', '3', '--ekf', '--headless',
                                   '--dump', dump], cwd=ROOT, env=env,
                                  capture_output=True, text=True)
            text = proc.stdout + proc.stderr
            match = re.search(r'瞄准板残差:.*?中位 ([0-9.]+) m，p90 ([0-9.]+) m.*?≤5cm ([0-9.]+)%',
                              text)
            aim = '%s/%s' % (match.group(1)[:6], match.group(2)[:6]) if match else 'n/a'
            hit = match.group(3) if match else 'n/a'
            rows = read_dump(dump)
            w = sorted(abs(r['ekfw']) for r in rows if r['ekfw'] == r['ekfw'])
            wstat = '%.2f/%.2f' % (pct(w, .5), pct(w, .9)) if w else 'n/a'
            print('%-6s %-7s %-9s %-11s %-9s %s' %
                  (clip, q, '%.2f' % (0.01 * float(q)) if float(q) > 0 else 'gate',
                   aim, hit, wstat))


if __name__ == '__main__':
    sys.exit(main() or 0)
