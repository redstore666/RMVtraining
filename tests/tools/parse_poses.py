#!/usr/bin/env python3
"""解析 POSE_DEBUG=1 打出的观测几何，量"法向 vs 位置相位"的系统偏置。

跑：
    POSE_DEBUG=1 tools/run_tool.sh pose20 20      # -> out/pose20.log
    tools/parse_poses.py out/pose20.log

日志（student/src/RotatingArmorTracker.cpp 里 POSE_DEBUG 分支）：
    [state] c=(+1.338,+0.115) z=+0.106 R=0.152 yaw=-98.6 w=+8.11
    [pose] id=3 x=+1.206 y=+0.156 bear=+162.7 normal=-163.9 normal-bear=-326.6

为什么关心这个：
  * 静止车（v4 的 id=1）上 `phase_lag` 恒为 -45.0°±2.5，`normal-bear` 恒在 +33~45°：
    观测法向与"板相对车心的方位"有固定偏置，不是噪声。
  * 双板整车解的相位差门限 `pair_body_max_phase_error_rad=0.35rad(20°)` 就是被这个
    偏置挡死的（1600 帧只成功 11 次）。
  * 任何"用绝对法向做相位校正"的改动都要先减掉这个偏置。
"""
import re
import sys

STATE = re.compile(r"\[state\] c=\(([+-][\d.]+),([+-][\d.]+)\) z=([+-][\d.]+) R=([\d.]+) "
                   r"yaw=\s*([+-][\d.]+) w=\s*([+-][\d.]+)")
POSE = re.compile(r"\[pose\] id=(\d+) x=([+-][\d.]+) y=([+-][\d.]+) bear=\s*([+-][\d.]+) "
                  r"normal=\s*([+-][\d.]+) normal-bear=\s*([+-][\d.]+)")


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def wrap(deg):
    return (deg + 180.0) % 360.0 - 180.0


def main(path):
    radii, omegas, diffs, bears = [], [], [], []
    per_id = {}
    for line in open(path, errors="ignore"):
        m = STATE.search(line)
        if m:
            radii.append(float(m.group(4)))
            omegas.append(float(m.group(6)))
        m = POSE.search(line)
        if m:
            ident = m.group(1)
            bear = float(m.group(4))
            normal = float(m.group(5))
            d = wrap(normal - bear)
            diffs.append(d)
            bears.append(bear)
            per_id.setdefault(ident, []).append(d)
    print(path)
    if not radii:
        print("  （没有 [state] 行：确认 POSE_DEBUG=1 且 tracker 已初始化）")
        return
    print("  state: n=%d  R med=%.3f p10=%.3f min=%.3f | w med=%+.2f p10=%+.2f p90=%+.2f"
          % (len(radii), q(radii, 0.5), q(radii, 0.1), min(radii), q(omegas, 0.5),
             q(omegas, 0.1), q(omegas, 0.9)))
    if not diffs:
        print("  （没有 [pose] 行：确认 POSE_DEBUG=1 且 tracker 已初始化）")
        return
    print("  normal-bear: n=%d med=%+.1f° p10=%+.1f° p90=%+.1f° （期望 0；45° 量级 = 折叠/约定偏置）"
          % (len(diffs), q(diffs, 0.5), q(diffs, 0.1), q(diffs, 0.9)))
    for ident, ds in sorted(per_id.items()):
        print("    id=%s n=%-5d med=%+.1f° p10=%+.1f° p90=%+.1f°"
              % (ident, len(ds), q(ds, 0.5), q(ds, 0.1), q(ds, 0.9)))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        main(p)
