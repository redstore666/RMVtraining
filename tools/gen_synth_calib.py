"""生成几何正确的合成棋盘格标定图（9x6 内角点，20mm 方格）。

用途：在没有相机的环境下自测 calibrate_intrinsics 的 TODO 实现是否正确。
用法：
    python3 tools/gen_synth_calib.py [输出目录]     # 默认 calib_synth/

画法：完整的棋盘有 10x7 个方格。将所有方格角点在板坐标系下投影，
只填充 (i+j) 为偶数的方格为黑色。内角点（相邻 4 格的交点）恰好是
9x6 个，与 objp 的 mgrid 约定一致。
"""
import os
import sys

import cv2
import numpy as np

COLS, ROWS, SQ = 9, 6, 20.0          # 内角点 9x6，方格 20mm
SQ_X, SQ_Y = COLS + 1, ROWS + 1      # 方格数 10x7

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "calib_synth"

# objp 约定：内角点 (c*sq, r*sq), c=0..8, r=0..5，板原点在第一个内角点
# 因此方格角点坐标范围为 [-0.5*sq, 9.5*sq] x [-0.5*sq, 6.5*sq]
grid_pts = []
for j in range(SQ_Y + 1):            # 8 行点
    for i in range(SQ_X + 1):        # 11 列点
        grid_pts.append([(i - 0.5) * SQ, (j - 0.5) * SQ, 0.0])
grid_pts = np.array(grid_pts, np.float32)  # (88, 3)

K = np.array([[1200.0, 0, 720.0], [0, 1195.0, 540.0], [0, 0, 1]])
dist = np.array([0.08, -0.12, 0.0005, -0.0007, 0.02])

os.makedirs(OUT_DIR, exist_ok=True)
# 清理旧图
for f in os.listdir(OUT_DIR):
    os.remove(os.path.join(OUT_DIR, f))

rng = np.random.default_rng(42)
saved = 0
for k in range(15):
    while True:
        rvec = rng.uniform(-0.45, 0.45, 3)
        tvec = np.array([rng.uniform(-60, 60), rng.uniform(-45, 45),
                         rng.uniform(260, 480)])
        proj, _ = cv2.projectPoints(grid_pts, rvec, tvec, K, dist)
        proj = proj.reshape(SQ_Y + 1, SQ_X + 1, 2)
        # 完整图案必须落在图内，且至少留 40px 白边（静区）
        if proj[:, :, 0].min() < 40 or proj[:, :, 0].max() > 1440 - 40:
            continue
        if proj[:, :, 1].min() < 40 or proj[:, :, 1].max() > 1080 - 40:
            continue
        break

    img = np.full((1080, 1440), 255, np.uint8)
    for j in range(SQ_Y):
        for i in range(SQ_X):
            if (i + j) % 2 != 0:
                continue
            quad = np.array([proj[j, i], proj[j, i + 1],
                             proj[j + 1, i + 1], proj[j + 1, i]], np.int32)
            cv2.fillPoly(img, [quad], 0)

    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    noise = rng.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT_DIR, f'synth_{k:02d}.png'), img)
    saved += 1

print('saved', saved, 'synthetic images to', OUT_DIR)
