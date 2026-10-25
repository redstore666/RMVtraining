#!/usr/bin/env python3
"""统计 TRACK_DEBUG=1 日志里转速估计窗口的接受/拒绝情况。

跑：
    TRACK_DEBUG=1 tools/run_tool.sh ev25 25       # 会写 out/ev25.log
    tools/summarize_ev.py out/ev25.log

日志行（student/src/RotatingArmorTracker.cpp 的 updateYawRateFromWindow）：
    [ev]     n=35 yaw=+1.20(0.30) bear=-1.60(0.95) used=-1.55(0.90) src=bear
    [evskip] n=35 used=+0.20(9.00) src=bear (sigma 过大)
    [evskip] n=35 yaw=+21.00(0.40) bear=+19.30(3.10) (超物理闸门)
    [evskip] n=35 R=0.152 bad=20/35 (R/极点不可信)
    [cleargap]/[cleargap2]/[nopick]  窗口没续上

判读要点：
  * 修 ee46ae7 之前，"sigma 过大"占 85%（相位折叠没解开，MAD 巨大）。
  * 加了 12 rad/s 物理闸门之后，"超物理闸门"成了主因 → 说明**斜率本身就是伪值**，
    来源是极点（车心）误差 / 续接到别的板 / R 崩了，而不是噪声大。
  * used 的分布才是喂给 EKF 的东西：|used| 常年 ~0 或正负对半 → ω 立不起来。
"""
import math
import re
import statistics
import sys

# yaw/bear 一侧可能是 +nan（法向野值被标 NaN 的帧），数字部分照常解析。
NUM = r"([+-]?(?:[\d.]+|nan))"
MAD = r"([\d.]+)"
EV = re.compile(r"\[ev\] n=(\d+) yaw=" + NUM + r"\(" + MAD + r"\) bear=" + NUM + r"\(" + MAD +
                r"\) used=" + NUM + r"\(" + MAD + r"\) src=(\w+)")
SKIP_SIGMA = re.compile(r"\[evskip\] n=(\d+) used=" + NUM + r"\(" + MAD + r"\) src=(\w+) \(sigma 过大\)")
SKIP_GATE = re.compile(r"\[evskip\] n=(\d+) yaw=" + NUM + r"\(" + MAD + r"\) bear=" + NUM +
                       r"\(" + MAD + r"\) \(超物理闸门\)")
SKIP_GEOM = re.compile(r"\[evskip\] n=(\d+) R=([+-]?[\d.]+) bad=(\d+)/(\d+) \(R/极点不可信\)")


def q(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def summarize(xs, label):
    xs = [v for v in xs if math.isfinite(v)]
    if not xs:
        print("  %-10s 无样本" % label)
        return
    print("  %-10s n=%-5d med=%+8.2f p10=%+8.2f p90=%+8.2f" %
          (label, len(xs), statistics.median(xs), q(xs, 0.10), q(xs, 0.90)))


def main(path):
    used, bear, yaw, sigmas = [], [], [], []
    gate_bear, cleaned, nopick = [], 0, 0
    n_ev = n_sigma = n_gate = n_geom = 0
    geom_r = []
    for line in open(path, errors="ignore"):
        if "[cleargap" in line:
            cleaned += 1
        if "[nopick]" in line:
            nopick += 1
        m = EV.search(line)
        if m:
            n_ev += 1
            used.append(float(m.group(6)))
            sigmas.append(float(m.group(7)))
            bear.append(float(m.group(4)))
            yaw.append(float(m.group(2)))
            continue
        m = SKIP_SIGMA.search(line)
        if m:
            n_sigma += 1
            continue
        m = SKIP_GATE.search(line)
        if m:
            n_gate += 1
            gate_bear.append(float(m.group(4)))
            continue
        m = SKIP_GEOM.search(line)
        if m:
            n_geom += 1
            geom_r.append(float(m.group(2)))
    total = n_ev + n_sigma + n_gate + n_geom
    print("%s" % path)
    print("  窗口评估 %d 次：接受 %d（%.1f%%）、sigma 过大 %d（%.1f%%）、超物理闸门 %d（%.1f%%）、"
          "R/极点不可信 %d（%.1f%%）"
          % (total, n_ev, 100.0 * n_ev / max(total, 1), n_sigma,
             100.0 * n_sigma / max(total, 1), n_gate, 100.0 * n_gate / max(total, 1),
             n_geom, 100.0 * n_geom / max(total, 1)))
    if geom_r:
        summarize(geom_r, "被拒时R")
    print("  清窗 %d 次、nopick %d 次" % (cleaned, nopick))
    summarize(used, "used")
    summarize(sigmas, "used_sigma")
    summarize(bear, "bear(接受)")
    summarize(yaw, "yaw(接受)")
    summarize(gate_bear, "bear(被拒)")
    if used:
        pos = sum(1 for v in used if v > 0)
        print("  used 符号：正 %d / 负 %d（正负对半 = 估计量是噪声，不是在跟转速）"
              % (pos, len(used) - pos))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        main(p)
