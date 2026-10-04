#!/usr/bin/env python3
"""畸变模型对比：自由 5 参数 vs k1 受限（与 tools/calibrate_intrinsics.cpp 同参数）。

方法与上一轮（打印板 19 张）一致，便于报告里两轮对比：
  1) 全量拟合两种模型，比较 RMS 与畸变系数合理性；
  2) bootstrap（重采样视图 N 次）：看 fx 的不确定度、k2/k3 是否乱跳——
     "自由模型的 k2/k3 符号都不稳定" 就是受限模型的实证理由；
  3) 5 折交叉验证：比较泛化 RMS（在没参与训练的视图上的重投影误差）。

用法: python3 tools/compare_calib_models.py [图像目录]   # 默认 calib_data/
"""
import sys
from pathlib import Path

import cv2
import numpy as np

COLS, ROWS = 7, 4
SQUARE_MM = 28.5
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
RESTRICTED_FLAGS = cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
N_BOOTSTRAP = 60
N_FOLDS = 5
SEED = 42


def collect_views(image_dir):
    objp = np.zeros((COLS * ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2) * SQUARE_MM
    views = []
    for f in sorted(Path(image_dir).glob("*.png")):
        img = cv2.imread(str(f))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(
            gray, (COLS, ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if not ok:
            print(f"[跳过] 检出失败: {f.name}")
            continue
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)
        views.append((objp, corners, f.name))
    return views


def fit(views, flags):
    objp = [v[0] for v in views]
    imgp = [v[1] for v in views]
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objp, imgp, (1440, 1080), None, None, flags=flags)
    return rms, K, dist.ravel(), rvecs, tvecs


def holdout_rms(train, test, flags):
    """在 train 上标定，把 test 视图做 PnP 后算其在测试视图上的重投影 RMS。"""
    rms, K, dist, _, _ = fit(train, flags)
    sq_sum, n = 0.0, 0
    for objp, corners, _ in test:
        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return np.nan
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        err2 = np.sum((proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2)
        sq_sum += err2
        n += len(objp)
    return float(np.sqrt(sq_sum / n))


def main():
    image_dir = sys.argv[1] if len(sys.argv) > 1 else "calib_data"
    views = collect_views(image_dir)
    print(f"有效视图: {len(views)}\n")

    print("== 1) 全量拟合 ==")
    rows = []
    for name, flags in [("自由5参数", 0), ("k1受限", RESTRICTED_FLAGS)]:
        rms, K, dist, _, _ = fit(views, flags)
        rows.append((name, rms, K, dist))
        print(f"{name}: RMS={rms:.4f}px  fx={K[0,0]:.1f} fy={K[1,1]:.1f} "
              f"cx={K[0,2]:.1f} cy={K[1,2]:.1f}")
        print(f"   畸变 [k1 k2 p1 p2 k3] = "
              f"{np.array2string(dist, precision=4, suppress_small=True)}")

    print(f"\n== 2) bootstrap x{N_BOOTSTRAP}（重采样视图）==")
    rng = np.random.default_rng(SEED)
    samples = {"自由5参数": [], "k1受限": []}
    k2s, k3s = [], []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(views), len(views))
        sub = [views[i] for i in idx]
        for name, flags in [("自由5参数", 0), ("k1受限", RESTRICTED_FLAGS)]:
            try:
                _, K, dist, _, _ = fit(sub, flags)
                samples[name].append(K[0, 0])
                if name == "自由5参数":
                    k2s.append(dist[1])
                    k3s.append(dist[4])
            except cv2.error:
                pass
    for name in ("自由5参数", "k1受限"):
        arr = np.array(samples[name])
        print(f"{name}: fx = {arr.mean():.1f} ± {arr.std():.1f} px")
    print(f"自由模型 k2 抽样: {np.array2string(np.array(k2s), precision=3)}")
    print(f"自由模型 k3 抽样: {np.array2string(np.array(k3s), precision=3)}")

    print(f"\n== 3) {N_FOLDS} 折交叉验证（留出视图上的 RMS）==")
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(views))
    folds = np.array_split(order, N_FOLDS)
    for name, flags in [("自由5参数", 0), ("k1受限", RESTRICTED_FLAGS)]:
        scores = []
        for k in range(N_FOLDS):
            train = [views[i] for j, f in enumerate(folds) if j != k for i in f]
            test = [views[i] for i in folds[k]]
            scores.append(holdout_rms(train, test, flags))
        arr = np.array(scores, dtype=float)
        print(f"{name}: 各折 {np.array2string(arr, precision=3)}  "
              f"均值 {np.nanmean(arr):.3f}px")


if __name__ == "__main__":
    main()
