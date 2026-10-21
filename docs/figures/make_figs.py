#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate figures for the course paper (docs/figures)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

plt.rcParams["font.family"] = ["Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)


def box(ax, x, y, w, h, text, fc, ec, fs=10):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=1.2, facecolor=fc, edgecolor=ec, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3)


def arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 linewidth=lw, color=color,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


# ---------------------------------------------------------------- fig1 pipeline
def fig_pipeline():
    # designed at final print size: 15 cm wide -> figsize 5.9 in, so fonts stay readable
    fig, ax = plt.subplots(figsize=(5.9, 2.55))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    c_trad = ("#FDEBD0", "#CA6F1E")
    c_nn = ("#D6EAF8", "#2471A3")

    trad = ["相机取流\n1440×1080", "灰度阈值 t=150\n轮廓 + 外接矩形", "灯条几何筛选\n角度/长度/长宽比",
            "同色配对\n板宽比/垂直度", "ROI 分类\n编号/颜色/大小", "PnP 位姿\nIPPE 双解消歧"]
    nn = ["相机取流\n1440×1080", "letterbox\n归一化 640×640", "YOLOv5\nCSP + PAN",
          "检测头\n25 200×22\n解码 + NMS", "关键点\nTL/TR/BR/BL", "PnP 位姿\nIPPE 双解消歧"]

    w, gap = 2.18, 0.24
    x0 = (15 - (len(trad) * w + (len(trad) - 1) * gap)) / 2
    y1, y2, h = 3.72, 1.52, 1.25

    for i, t in enumerate(trad):
        box(ax, x0 + i * (w + gap), y1, w, h, t, *c_trad, fs=7.4)
    for i in range(len(trad) - 1):
        xe = x0 + i * (w + gap) + w
        arrow(ax, xe + 0.02, y1 + h / 2, xe + gap - 0.02, y1 + h / 2)

    for i, t in enumerate(nn):
        box(ax, x0 + i * (w + gap), y2, w, h, t, *c_nn, fs=7.4)
    for i in range(len(nn) - 1):
        xe = x0 + i * (w + gap) + w
        arrow(ax, xe + 0.02, y2 + h / 2, xe + gap - 0.02, y2 + h / 2)

    ax.text(x0, 5.42, "传统视觉：人工设计特征 + 人工设定决策边界", fontsize=8, color="#A04000")
    ax.text(x0, 3.22, "深度学习：端到端学习特征与角点位置", fontsize=8, color="#1A5276")

    ax.text(7.5, 0.45, "两条路线共享同一后端（PnP → 时序平滑 → 输出），区别在于特征与决策边界由谁确定",
            ha="center", fontsize=6.8, color="#333333")
    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(OUT, "fig1_pipeline.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- fig2 latency & far recall
def fig_latency():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.3), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.3})

    sizes = ["640×480\n(默认)", "640×640\n(原始)", "512×384"]
    mn = [5.06, 6.23, 3.52]
    md = [7.32, 9.40, 5.19]
    idx = np.arange(len(sizes))
    w = 0.36
    b1 = ax1.bar(idx - w / 2, mn, w, label="最小延迟", color="#5DADE2", edgecolor="#1B4F72", linewidth=0.8)
    b2 = ax1.bar(idx + w / 2, md, w, label="中位延迟", color="#F5B041", edgecolor="#7E5109", linewidth=0.8)
    for bars in (b1, b2):
        for r in bars:
            ax1.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.15, f"{r.get_height():.2f}",
                     ha="center", fontsize=8.5)
    ax1.set_xticks(idx)
    ax1.set_xticklabels(sizes, fontsize=9)
    ax1.set_ylabel("单帧推理耗时 / ms")
    ax1.set_ylim(0, 11.5)
    ax1.set_title("(a) 同进程 AB 交替推理（150 次/尺寸）", fontsize=10.5)
    ax1.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)
    ax1.annotate("中位 −22%", xy=(0.18, 7.32), xytext=(0.55, 10.4), fontsize=9, color="#1B4F72",
                 arrowprops=dict(arrowstyle="->", color="#1B4F72", lw=0.9))
    ax1.annotate("中位 −45%", xy=(1.82, 5.19), xytext=(1.55, 9.0), fontsize=9, color="#7E5109",
                 arrowprops=dict(arrowstyle="->", color="#7E5109", lw=0.9))

    bins = ["6.0 m", "6.8 m", "7.5 m", "8.2 m", "9.0 m"]
    full = [1, 3, 1, 1, 1]
    small = [0, 1, 1, 0, 0]
    x = np.arange(len(bins))
    ax2.bar(x - 0.2, full, 0.4, label="640×480", color="#48C9B0", edgecolor="#0E6655", linewidth=0.8)
    ax2.bar(x + 0.2, small, 0.4, label="512×384", color="#CD6155", edgecolor="#641E16", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bins, fontsize=9)
    ax2.set_ylabel("检出数 / 个", labelpad=6)
    ax2.set_ylim(0, 3.6)
    ax2.set_title("(b) 远距实拍（6~9 m）检出", fontsize=10.5)
    ax2.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.text(0.03, 0.05, "缩到 512×384 后远距目标大量漏检", transform=ax2.transAxes,
             fontsize=8.5, color="#641E16")

    fig.savefig(os.path.join(OUT, "fig2_latency_recall.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- fig3 traditional funnel
def fig_funnel():
    # print-size design: 11 cm wide -> 4.33 in
    fig, ax = plt.subplots(figsize=(4.33, 2.05))
    labels = ["灯条提取/几何筛选", "灯条配对", "编号分类"]
    vals = [15, 1, 2]
    colors = ["#E59866", "#F4D03F", "#7FB3D5"]
    y = np.arange(len(labels))[::-1]
    bars = ax.barh(y, vals, color=colors, edgecolor="#555555", linewidth=0.6, height=0.55)
    for r, v in zip(bars, vals):
        ax.text(v + 0.3, r.get_y() + r.get_height() / 2, f"{v}/18（{v/18*100:.0f}%）",
                va="center", fontsize=7.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.6)
    ax.set_xlim(0, 20)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.tick_params(axis="x", labelsize=7)
    ax.set_xlabel("失败案例数（共 18 例）", fontsize=7.6)
    ax.set_title("18 例失败中 83% 卡在灯条提取", fontsize=8.2)
    ax.grid(axis="x", linestyle=":", alpha=0.5, linewidth=0.6)
    ax.text(19.6, 2.45, "修复后 18/18", ha="right", va="top", fontsize=7.2, color="#1E8449")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "fig3_traditional_failures.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- fig4 EKF convergence
def fig_ekf():
    scen = ["推荐路径\n(Spinning+法向前验)", "无相位先验", "静止声明\n+静止真值",
            "静止声明\n+旋转真值", "旋转声明\n+静止真值"]
    short = ["推荐路径", "无相位先验", "静止→静止", "静止→旋转", "旋转→静止"]
    pos = np.array([[0.0070, 0.0051, 0.0064],
                    [0.0460, 0.0830, 0.0141],
                    [0.0161, 0.0193, 0.0195],
                    [0.0193, 0.1024, 0.0232],
                    [0.4350, 0.1797, 0.0189]])
    ang = np.array([[0.0099, 0.0070],
                    [0.3271, 3.1669],
                    [0.0473, 0.0],
                    [0.3493, 2.5000],
                    [2.7299, 0.5692]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.28})
    x = np.arange(len(scen))
    w = 0.26
    names = ["|Δcx|", "|Δcy|", "|ΔR|"]
    cols = ["#2E86C1", "#28B463", "#CA6F1E"]
    for j in range(3):
        ax1.bar(x + (j - 1) * w, np.maximum(pos[:, j], 1e-4), w, label=names[j],
                color=cols[j], edgecolor="white", linewidth=0.6)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-3, 1.0)
    ax1.set_ylabel("收敛误差 / m（对数坐标）")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, fontsize=8.6)
    ax1.set_xlabel("(标签：声明状态 → 真值状态)")
    ax1.set_title("(a) 位置类状态收敛误差", fontsize=10.5)
    ax1.legend(fontsize=8.5, frameon=False, ncol=3, loc="upper left")
    ax1.grid(axis="y", linestyle=":", alpha=0.5)

    w2 = 0.34
    ax2.bar(x - w2 / 2, np.maximum(ang[:, 0], 1e-4), w2, label="|Δyaw| / rad",
            color="#8E44AD", edgecolor="white", linewidth=0.6)
    ax2.bar(x + w2 / 2, np.maximum(ang[:, 1], 1e-4), w2, label="|Δω| / (rad/s)",
            color="#D35400", edgecolor="white", linewidth=0.6)
    ax2.set_yscale("log")
    ax2.set_ylim(5e-4, 10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(short, fontsize=8.6)
    ax2.set_xlabel("(标签：声明状态 → 真值状态)")
    ax2.set_title("(b) 角度类状态收敛误差", fontsize=10.5)
    ax2.legend(fontsize=8.5, frameon=False, loc="upper left")
    ax2.grid(axis="y", linestyle=":", alpha=0.5)
    ax2.text(2 + w2 / 2, 6.5e-4, "ω=0", ha="center", fontsize=8, color="#555555")

    fig.savefig(os.path.join(OUT, "fig4_ekf_convergence.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- fig5 playback
def fig_playback():
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    videos = ["v4 静止大板\n(1384 观测帧)", "v4 仅 id=3\n(923 观测帧)", "v1 两车轮换\n(513 观测帧)"]
    med = [7.4, 99.0, 412.0]
    p90 = [16.8, 181.5, 2575.0]
    x = np.arange(len(videos))
    w = 0.34
    b1 = ax.bar(x - w / 2, med, w, label="中位误差", color="#5499C7", edgecolor="white")
    b2 = ax.bar(x + w / 2, p90, w, label="p90 误差", color="#E59866", edgecolor="white")
    for r, v in zip(list(b1) + list(b2), med + p90):
        ax.text(r.get_x() + r.get_width() / 2, v * 1.12, f"{v:.1f}" if v < 1000 else f"{v:.0f}",
                ha="center", fontsize=8.5)
    ax.set_yscale("log")
    ax.set_ylim(4, 15000)
    ax.set_xticks(x)
    ax.set_xticklabels(videos, fontsize=9)
    ax.set_ylabel("板心预测误差 / mm（对数坐标）")
    ax.legend(fontsize=8.8, frameon=False, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    for i, frac in enumerate([94.5, 7.9, 3.5]):
        ax.text(i, p90[i] * 2.4, f"≤2 cm 占 {frac}%", ha="center", fontsize=8.5, color="#1E8449")
    ax.set_title("真数据回放：预测板心与实测板心的距离", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_playback_error.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- fig6 observation model
def fig_symmetry():
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.set_xlim(-2.2, 3.2)
    ax.set_ylim(-1.25, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")

    cx, cy, R = 1.15, 0.15, 0.62
    ax.add_patch(Circle((cx, cy), R, fill=False, linestyle="--", linewidth=1.0, color="#7F8C8D"))
    ax.plot(cx, cy, "o", color="#2C3E50", ms=6, zorder=5)
    ax.text(cx - 0.02, cy - 0.30, "车体中心 (cx, cy)", ha="center", fontsize=9.5)

    yaw = np.deg2rad(25)
    for k in range(4):
        ph = yaw + k * np.pi / 2
        px, py = cx + R * np.cos(ph), cy + R * np.sin(ph)
        ax.add_patch(Rectangle((px - 0.16, py - 0.045), 0.32, 0.09,
                               angle=np.rad2deg(ph) + 90, facecolor="#F5B041",
                               edgecolor="#7E5109", linewidth=0.9, zorder=4))

    ax.annotate("", xy=(cx + R * np.cos(yaw), cy + R * np.sin(yaw)), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.4))
    ax.text(cx + 0.24, cy + 0.30, "R", fontsize=12, color="#C0392B")
    ax.annotate("", xy=(cx + 0.52, cy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="->", color="#2E86C1", lw=1.2))
    ax.text(cx + 0.24, cy + 0.04, "yaw", fontsize=9.5, color="#2E86C1")

    p0 = (cx + R * np.cos(yaw), cy + R * np.sin(yaw))
    ax.annotate("P$_k$ = 板心\nφ$_k$ = yaw + 2πk/N", xy=p0, xytext=(cx + 1.0, cy + 0.62),
                fontsize=9, color="#7B241C",
                arrowprops=dict(arrowstyle="->", color="#7B241C", lw=0.8),
                ha="left", va="center")

    cam = (-1.9, -0.5)
    ax.plot(cam[0], cam[1], "s", color="#1B4F72", ms=9)
    ax.text(cam[0], cam[1] - 0.2, "相机", ha="center", fontsize=9.5, color="#1B4F72")
    ax.plot([cam[0], cx - R * np.cos(yaw)], [cam[1], cy - R * np.sin(yaw)],
            color="#1B4F72", lw=1.0, linestyle=(0, (3, 2)))
    ax.annotate("观测 z：板心三维位置（PnP 输出）", xy=(-0.4, -0.16),
                xytext=(-2.15, 0.55), fontsize=9, color="#1B4F72",
                arrowprops=dict(arrowstyle="->", color="#1B4F72", lw=0.8))

    ax.text(-2.1, 1.85, "P$_k$ = (cx − R·cos φ$_k$, cy − R·sin φ$_k$, z)", fontsize=10)
    ax.text(-2.1, 1.55, "φ$_k$ = yaw + 2πk/N,  k = 0…N−1", fontsize=10)
    ax.text(-2.1, 1.2, "N 块板几何全同 ⇒ yaw 只能确定到 2π/N", fontsize=9.2, color="#7B241C")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_observation_model.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_latency()
    fig_funnel()
    fig_ekf()
    fig_playback()
    fig_symmetry()
    print("figures written to", OUT)
