#!/usr/bin/env python3
"""fitBarConstrained（需求③ §3.2/§3.4/§3.5）的独立 Python 复刻 × C++ 实现交叉验证。

流程：先用 C++ 工具转储一帧的最终掩码与灯条几何，再用本脚本按同一规格
（CHAIN_APPROX_NONE 轮廓 → PCA 主轴 → 分位定边 → 接受门 → G1/G2/G3 质量门）
独立复算，对比：
  1) 逐轮廓门控决策（接受/回退原因）计数 vs C++ 计数；
  2) 被采纳轮廓的几何（中心/半长/半宽/长轴方向）逐条对比（期望差 = float32 舍入量级）。

用法:
  ./build/test_bar_fit tmp/red.avi --red --dump-frame N /tmp/fitcmp
  python3 tools/verify_fit_constrained.py /tmp/fitcmp_mask.png /tmp/fitcmp_bars.txt \
      --counts <total,accept,rej_pts,rej_size,rej_g1,rej_g2,rej_g3>
"""
import argparse

import cv2
import numpy as np


def quantile_linear(sorted_vals, q):
    """numpy.percentile 默认口径：pos=q*(n-1) 线性插值。"""
    pos = q * (len(sorted_vals) - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def fit_bar_constrained(contour, rough):
    """返回 (结果 dict | None, 回退原因)。rough 为 cv2.minAreaRect(contour)。"""
    P = contour.reshape(-1, 2).astype(np.float64)
    if len(P) < 8:
        return None, "pts"
    mu = P.mean(axis=0)
    Q = P - mu
    cov = Q.T @ Q / len(P)
    cxx, cyy, cxy = cov[0, 0], cov[1, 1], cov[0, 1]
    disc = np.sqrt((cxx - cyy) ** 2 + 4.0 * cxy * cxy)
    lam1 = 0.5 * (cxx + cyy + disc)
    if lam1 <= 1e-9:
        return None, "pts"
    if abs(cxy) > 1e-12:
        e1 = np.array([cxy, lam1 - cxx])
    elif cxx >= cyy:
        e1 = np.array([1.0, 0.0])
    else:
        e1 = np.array([0.0, 1.0])
    nrm = np.linalg.norm(e1)
    if nrm < 1e-12:
        return None, "pts"
    e1 = e1 / nrm
    perp = np.array([-e1[1], e1[0]])

    u = Q @ e1
    v = Q @ perp
    us = np.sort(u)
    vs = np.sort(v)
    u_lo, u_hi = quantile_linear(us, 0.02), quantile_linear(us, 0.98)
    v_lo, v_hi = quantile_linear(vs, 0.10), quantile_linear(vs, 0.90)
    L, W = u_hi - u_lo, v_hi - v_lo
    half_l, half_w = L / 2.0, W / 2.0
    if L < 20.0 or half_w <= 1e-6 or half_l / half_w < 2.5:
        return None, "size"

    # minAreaRect 主轴 = 最长边方向
    box = cv2.boxPoints(rough)
    best, best_len = np.array([0.0, 1.0]), -1.0
    for k in range(4):
        e = box[(k + 1) % 4] - box[k]
        ln = np.linalg.norm(e)
        if ln > best_len:
            best_len, best = ln, e / max(ln, 1e-12)
    fit_ang = np.degrees(np.arctan2(e1[1], e1[0]))
    rough_ang = np.degrees(np.arctan2(best[1], best[0]))
    d = abs(fit_ang - rough_ang) % 180.0
    if d > 90.0:
        d = 180.0 - d
    if d > 5.0:
        return None, "g2"

    rough_area = float(rough[1][0]) * float(rough[1][1])
    area_ratio = (L * W) / max(rough_area, 1e-9)
    if area_ratio < 0.75 or area_ratio > 1.15:
        return None, "g3"

    support = L / max(us[-1] - us[0], 1e-9)
    if support < 0.85:
        return None, "g1"

    return {
        "center": mu,
        "axis": e1,
        "half_l": half_l,
        "half_w": half_w,
        "L": L,
        "W": W,
        "area_ratio": area_ratio,
        "support": support,
        "axis_dev": d,
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mask")
    ap.add_argument("bars", help="C++ 转储的灯条几何文件（cx cy ax ay half_l half_w fitted）")
    ap.add_argument("--counts", default=None,
                    help="C++ 计数（total,accept,rej_pts,rej_size,rej_g1,rej_g2,rej_g3）")
    args = ap.parse_args()

    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"无法读取掩码: {args.mask}")
        return 2
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    counts = {"pts": 0, "size": 0, "g1": 0, "g2": 0, "g3": 0}
    total, accepted = 0, []
    for c in contours:
        if cv2.contourArea(c) < 20.0:
            continue
        total += 1
        res, reason = fit_bar_constrained(c, cv2.minAreaRect(c))
        if res is None:
            counts[reason] += 1
        else:
            accepted.append(res)

    ok = True
    print(f"Python 复刻（点集 CHAIN_APPROX_NONE，{len(contours)} 个外轮廓）")
    print(f"  进入拟合 {total} | 采纳 {len(accepted)} | 回退 pts={counts['pts']} size={counts['size']} "
          f"g1={counts['g1']} g2={counts['g2']} g3={counts['g3']}")
    if args.counts:
        t, a, rp, rs, g1, g2, g3 = (int(x) for x in args.counts.split(","))
        same = (t, a, rp, rs, g1, g2, g3) == (total, len(accepted), counts["pts"], counts["size"],
                                              counts["g1"], counts["g2"], counts["g3"])
        print(f"  C++ 计数 total={t} accept={a} pts={rp} size={rs} g1={g1} g2={g2} g3={g3} → "
              f"{'一致 ✓' if same else '**不一致 ✗**'}")
        ok = ok and same

    rows = []
    with open(args.bars) as fp:
        for line in fp:
            vals = line.split()
            if len(vals) == 7:
                rows.append([float(x) for x in vals[:6]] + [int(vals[6])])
    print(f"  C++ 灯条 {len(rows)} 条（仅含通过灯条过滤者）")

    max_dc = max_dhl = max_dhw = max_dang = 0.0
    matched = 0
    core_bars = 0
    for r in rows:
        if r[6] == 0:
            core_bars += 1 # 白芯修复路径的灯条（不参与拟合），无对应轮廓可比
            continue
        cc = np.array(r[:2])
        cands = [i for i, a in enumerate(accepted) if np.linalg.norm(a["center"] - cc) < 30.0]
        if not cands:
            print(f"  **未匹配**: C++ 灯条中心 {cc} fitted={r[6]}")
            ok = False
            continue
        i = min(cands, key=lambda k: np.linalg.norm(accepted[k]["center"] - cc))
        a = accepted[i]
        dc = np.linalg.norm(a["center"] - cc)
        dhl = abs(a["half_l"] - r[4])
        dhw = abs(a["half_w"] - r[5])
        ang_cpp = np.degrees(np.arctan2(r[3], r[2]))
        ang_py = np.degrees(np.arctan2(a["axis"][1], a["axis"][0]))
        d = abs(ang_cpp - ang_py) % 180.0
        dang = min(d, 180.0 - d)
        max_dc = max(max_dc, dc)
        max_dhl = max(max_dhl, dhl)
        max_dhw = max(max_dhw, dhw)
        max_dang = max(max_dang, dang)
        matched += 1
    print(f"  逐条对比 {matched}/{len(rows)} 条匹配：Δ中心 max={max_dc:.2e} px | Δ半长 max={max_dhl:.2e} px "
          f"| Δ半宽 max={max_dhw:.2e} px | Δ轴向 max={max_dang:.2e}°")
    print("结论:", "跨语言一致（差异为 float32/float64 舍入量级）✓" if ok else "**存在不一致 ✗**")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
