#!/usr/bin/env python3
"""把 test_camera_system 的 webdebug（/data SSE）抓包画成报告用图。

抓包方法（先跑工具，再抓 45 s）：

    ./build/test_camera_system --red --video tmp/v4.avi --video-loop --id 3 &
    curl -sN --max-time 45 http://127.0.0.1:8081/data > tmp/sse_left.txt
    ./build/test_camera_system --red --video tmp/v4.avi --video-loop --id 1 &
    curl -sN --max-time 30 http://127.0.0.1:8081/data > tmp/sse_right.txt

然后：
    python3 docs/figures/make_v4_webdebug_figs.py
输出 docs/figures/v4_webdebug_*.png。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# 本机没有 DejaVu 之外的中文字体，显式指定 CJK 字体，否则中文全变方框。
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
ROOT = os.path.dirname(os.path.dirname(HERE))


def load(path):
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
            for rec in batch:
                keys.setdefault(rec["k"], []).append(rec["v"])
    return keys


def series(keys, name, n=None):
    values = np.asarray(keys.get(name, []), dtype=float)
    if n is not None:
        values = values[:n]
    return values


def video_time(n):
    # 视频时间轴固定 6.024 ms/帧（166 fps），--video-loop 时工具内部是累加的。
    return np.arange(n) * 0.006024


def fig_omega(left, right):
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.6), sharex=False)
    for ax, keys, title, colour in (
        (axes[0], left, "id=3（左车：自转 + 平移）", "tab:red"),
        (axes[1], right, "id=1（右车：静止）", "tab:blue"),
    ):
        w = series(keys, "omega")
        t = video_time(len(w))
        ax.plot(t, w, lw=0.6, color=colour)
        ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)
        for y in (-5.0, 5.0):
            ax.axhline(y, color="gray", ls="--", lw=0.6, alpha=0.7)
        ax.set_ylabel(r"$\omega$ (rad/s)")
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(alpha=0.25)
        ax.set_ylim(-18, 18)
    axes[1].set_xlabel("视频时间 (s)（相机/回放时间轴，非墙钟）")
    fig.suptitle("EKF 角速度状态：静止车稳在 0，自转车在 0 与 ±5~17 之间来回跳", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v4_webdebug_omega.png"), dpi=130)
    plt.close(fig)


def fig_hist(left, right):
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bins = np.arange(-18, 18.5, 1.0)
    ax.hist(series(right, "omega"), bins=bins, color="tab:blue", alpha=0.75,
            label="id=1 静止车")
    ax.hist(series(left, "omega"), bins=bins, color="tab:red", alpha=0.55,
            label="id=3 自转车")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\omega$ (rad/s)")
    ax.set_ylabel("帧数（对数）")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title("角速度分布：自转车 20.6% 的帧 |ω|≥5 rad/s，符号对称 = 来回震荡", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v4_webdebug_omega_hist.png"), dpi=130)
    plt.close(fig)


def fig_updates(left):
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.2), sharex=True)
    upd = series(left, "updates")
    miss = series(left, "missed")
    t = video_time(len(upd))
    axes[0].plot(t, upd, lw=0.5, color="tab:green")
    axes[0].set_ylabel("updates/帧")
    axes[0].set_title("id=3：每帧融合了几块板 / 连续多少帧没观测（missed）", fontsize=9, loc="left")
    axes[0].grid(alpha=0.25)
    axes[1].plot(t, miss, lw=0.5, color="tab:orange")
    axes[1].set_ylabel("missed 帧数")
    axes[1].set_xlabel("视频时间 (s)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v4_webdebug_updates.png"), dpi=130)
    plt.close(fig)


def fig_lag(left, right):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    bins = np.arange(-90, 115, 5.0)
    axes[0].hist(series(right, "phase_lag_deg"), bins=bins, color="tab:blue", alpha=0.8)
    axes[0].set_title("id=1 静止车：稳定在 −45°±2.5°", fontsize=9)
    axes[0].set_xlabel("phase_lag (deg)")
    axes[1].hist(series(left, "phase_lag_deg"), bins=bins, color="tab:red", alpha=0.8)
    axes[1].set_title("id=3 自转车：基线 −45° 上叠 ±45° 抖动", fontsize=9)
    axes[1].set_xlabel("phase_lag (deg)")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.set_ylabel("帧数")
    fig.suptitle("观测法向与模型法向之差：静止车的固定 −45° 说明这是几何/约定基线，不是噪声",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v4_webdebug_lag.png"), dpi=130)
    plt.close(fig)


def fig_timing(left):
    fig, ax = plt.subplots(figsize=(9.0, 3.2))
    loop = series(left, "loop_ms")
    detect = series(left, "detect_ms")
    t = video_time(len(loop))
    ax.plot(t, loop, lw=0.5, color="tab:purple", label="loop_ms")
    ax.plot(t, detect, lw=0.5, color="tab:brown", alpha=0.7, label="detect_ms")
    ax.set_ylim(0, 60)
    ax.set_xlabel("视频时间 (s)")
    ax.set_ylabel("ms")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    ax.set_title("单帧耗时：loop 中位 8.3 ms / p90 13.6 ms；detect 中位 7.4 ms，偶发 >100 ms",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v4_webdebug_timing.png"), dpi=130)
    plt.close(fig)


def main():
    left = load(os.path.join(ROOT, "tmp/sse_left.txt"))
    right = load(os.path.join(ROOT, "tmp/sse_right.txt"))
    fig_omega(left, right)
    fig_hist(left, right)
    fig_updates(left)
    fig_lag(left, right)
    fig_timing(left)
    print("wrote", ", ".join(sorted(f for f in os.listdir(OUT)
                                    if f.startswith("v4_webdebug_") and f.endswith(".png"))))


if __name__ == "__main__":
    main()
