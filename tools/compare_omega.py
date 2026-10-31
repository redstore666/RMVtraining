#!/usr/bin/env python3
"""把 test_camera_system 抓的 SSE 曲线变成"omega 抖不抖"的对照表 + 前后对照图。

用法：
    # 1) 抓数据（随机端口 + 同一次调用内起停，见 tests/tools/run_tool.sh 的说明）
    ./tools/run_tool.sh before 20 -- --video tests/resources/redrotate.avi --red
    ./tools/run_tool.sh after  20 -- --video tests/resources/redrotate.avi --red

    # 2) 对照
    tools/compare_omega.py before=tmp/handoff/out/before.sse after=tmp/handoff/out/after.sse \
        --png tmp/handoff/out/redrotate_before_after.png --title redrotate

指标口径（只看稳态段，锁定瞬态会污染 std）：
  w_std      稳态段 omega 的标准差（rad/s）—— 用户看到的"抖"
  dw_std     帧间差分标准差 —— 高频抖动（锯齿）
  jump%      |dw|>1 rad/s 的帧占比 —— 突跳
  flips      omega 符号翻转次数 —— "方波"指标
  zero%      |omega|<0.1 的帧占比 —— 掉零突刺
  smooth_rng 0.44 s 滑动平均后的极差 —— 慢漂移/正弦峰谷是否被削平
稳态段定义：第一次 |omega|>0.5 之后再跳过 --warm 秒。
"""
import argparse
import json
import sys

import numpy as np


def load_csv(path):
    """读 test_camera_playback --dump 的 CSV（列 t / ekfw 就是时间与 EKF 的 omega）。"""
    import csv

    keys = {"omega": []}
    with open(path, newline="", errors="ignore") as handle:
        for row in csv.DictReader(handle):
            try:
                stamp = float(row["t"])
                value = float(row["ekfw"])
            except (KeyError, TypeError, ValueError):
                continue
            keys["omega"].append((stamp, value))
    return keys


def load_sse(path):
    keys = {}
    with open(path, errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("data:"):
                continue
            try:
                batch = json.loads(line[5:].strip())
            except Exception:
                continue
            for item in batch:
                key = item.get("k")
                value = item.get("v")
                if key is None or value is None:
                    continue
                keys.setdefault(key, []).append((item.get("t"), value))
    return keys


def steady_series(path, warm_s=1.0, fps=57.0):
    keys = load_csv(path) if path.lower().endswith(".csv") else load_sse(path)
    if "omega" not in keys:
        return None, None
    stamps = np.array([t for t, _ in keys["omega"]], dtype=float)
    values = np.array([v for _, v in keys["omega"]], dtype=float)
    moving = np.where(np.abs(values) > 0.5)[0]
    if not len(moving):
        return stamps, values
    start = moving[0] + int(round(warm_s * fps))
    return stamps[start:], values[start:]


def metrics(path, warm_s, smooth_win=25):
    stamps, values = steady_series(path, warm_s)
    if values is None or len(values) < 10:
        return None
    diff = np.diff(values)
    smooth = np.convolve(values, np.ones(smooth_win) / smooth_win, mode="valid")
    return {
        "frames": len(values),
        "median": float(np.median(values)),
        "w_std": float(np.std(values)),
        "dw_std": float(np.std(diff)) if len(diff) else 0.0,
        "jump_pct": float(100.0 * np.mean(np.abs(diff) > 1.0)) if len(diff) else 0.0,
        "flips": int(np.sum(np.diff(np.sign(values)) != 0)),
        "zero_pct": float(100.0 * np.mean(np.abs(values) < 0.1)),
        "smooth_range": float(np.ptp(smooth)) if len(smooth) else 0.0,
        "smooth_min": float(np.min(smooth)) if len(smooth) else 0.0,
        "smooth_max": float(np.max(smooth)) if len(smooth) else 0.0,
    }


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("series", nargs="+", help="标签=路径（.sse 或 playback 的 --dump .csv）")
    parser.add_argument("--warm", type=float, default=1.0, help="跳过锁定瞬态的秒数（默认 1.0）")
    parser.add_argument("--png", default="", help="对照图输出路径（可选）")
    parser.add_argument("--title", default="omega", help="图标题")
    args = parser.parse_args(argv)

    table = []
    for spec in args.series:
        if "=" not in spec:
            parser.error(f"参数要写成 标签=路径：{spec}")
        label, path = spec.split("=", 1)
        data = metrics(path, args.warm)
        if data is None:
            print(f"{label}: 没有 omega 数据（{path}）")
            continue
        table.append((label, path, data))

    if not table:
        return 1

    header = f"{'tag':<14}{'frames':>7}{'median':>9}{'w_std':>8}{'dw_std':>8}{'jump%':>7}{'flips':>7}{'zero%':>7}{'smooth_rng':>11}"
    print(header)
    print("-" * len(header))
    for label, _, data in table:
        print(f"{label:<14}{data['frames']:>7}{data['median']:>+9.2f}{data['w_std']:>8.2f}"
              f"{data['dw_std']:>8.2f}{data['jump_pct']:>7.1f}{data['flips']:>7d}"
              f"{data['zero_pct']:>7.1f}{data['smooth_range']:>11.2f}")

    if args.png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("没有 matplotlib，跳过画图", file=sys.stderr)
            return 0
        figure, axes = plt.subplots(len(table), 1, figsize=(11, 2.6 * len(table)), sharex=False)
        if len(table) == 1:
            axes = [axes]
        for axis, (label, path, data) in zip(axes, table):
            stamps, values = steady_series(path, args.warm)
            axis.plot(stamps, values, linewidth=0.9, color="#2a6fdb", label="EKF omega")
            window = 25
            if len(values) > window:
                smooth = np.convolve(values, np.ones(window) / window, mode="valid")
                axis.plot(stamps[window // 2: len(stamps) - (window - 1) + window // 2], smooth,
                          linewidth=1.8, color="#e05a00", label="0.44s moving average")
            axis.set_title(f"{label}: std={data['w_std']:.2f}  dw_std={data['dw_std']:.2f}  "
                           f"flips={data['flips']}", fontsize=10)
            axis.set_ylabel("omega (rad/s)")
            axis.grid(alpha=0.3)
            axis.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("time (s)")
        figure.suptitle(args.title)
        figure.tight_layout()
        figure.savefig(args.png, dpi=130)
        print(f"\n图已写入 {args.png}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
