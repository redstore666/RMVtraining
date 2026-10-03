#!/usr/bin/env python3
"""标定图像自查：拍摄完成后、正式标定前，快速评估这一批图像是否"能标出好结果"。

用法:
    python3 tools/check_calib_images.py [图像目录]     # 默认 calib_data/

逐张检查（与标定程序使用同一套角点检测参数）:
  - 能否检出 9x6 棋盘格
  - 亮度: 白块亮度用 p99 近似。记录用信息——实测把图像调亮到 p99≈200 后，
          角点位置平均只移动 0.005px、标定结果逐位相同；亮度不影响标定精度。
  - 覆盖: 角点到画面中心的径向距离 r_max（画面四角 = 1.0）。覆盖不足时高阶畸变
          (k2/k3) 不可估计——标定工具已采用 k1 受限模型；追求更高精度可补拍四角。
  - 清晰度: Laplacian 方差（<25 可能模糊）。

判定规则: 通过 = 可检出视图 >= 15 且未检出 <= 2。
          亮度/覆盖/清晰度为提示信息，不阻塞使用。
"""
import glob
import os
import sys

import cv2
import numpy as np

COLS, ROWS = 9, 6
FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE


def analyze(path):
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    lapvar = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    p99 = float(np.percentile(gray, 99))
    sat = float((gray >= 250).mean() * 100.0)
    ok, corners = cv2.findChessboardCorners(gray, (COLS, ROWS), FLAGS)
    r = {"ok": bool(ok), "p99": p99, "sat": sat, "lapvar": lapvar}
    if ok:
        pts = corners.reshape(-1, 2)
        center = np.array([w / 2.0, h / 2.0])
        half_diag = float(np.hypot(w / 2.0, h / 2.0))
        dist = np.sqrt(((pts - center) ** 2).sum(axis=1))
        r["r_max"] = float(dist.max() / half_diag)
        r["width"] = float((pts[:, 0].max() - pts[:, 0].min()) / w)
    return r


def main():
    image_dir = sys.argv[1] if len(sys.argv) > 1 else "calib_data"
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    files = sorted(f for f in glob.glob(os.path.join(image_dir, "*"))
                   if os.path.splitext(f)[1].lower() in exts)
    if not files:
        print(f"目录中没有图像: {image_dir}")
        return 1

    print(f"检查 {len(files)} 张图像（目录: {image_dir}）")
    header = (f"{'文件':<22}{'检出':<5}{'亮度p99':<9}{'过曝%':<8}"
              f"{'覆盖r_max':<10}{'板宽占比':<9}{'清晰度':<8}备注")
    print(header)
    print("-" * 88)

    n_ok = n_edge = 0
    r_max_all, width_all, p99_all = [], [], []
    for f in files:
        name = os.path.basename(f)
        r = analyze(f)
        if r is None:
            print(f"{name:<22}读取失败")
            continue
        p99_all.append(r["p99"])
        notes = []
        if not r["ok"]:
            notes.append("未检出")
        if r["p99"] < 100:
            notes.append("偏暗")
        if r["sat"] > 0.5:
            notes.append("可能过曝")
        if r["lapvar"] < 25:
            notes.append("可能模糊")
        if r["ok"]:
            n_ok += 1
            r_max_all.append(r["r_max"])
            width_all.append(r["width"])
            if r["r_max"] >= 0.85:
                n_edge += 1
            if r["r_max"] < 0.55:
                notes.append("位置偏中")
            r_max_s = f"{r['r_max']:.2f}"
            width_s = f"{r['width'] * 100:.0f}%"
        else:
            r_max_s = width_s = "-"
        det = "Y" if r["ok"] else "N"
        print(f"{name:<22}{det:<5}{r['p99']:<9.0f}{r['sat']:<8.2f}"
              f"{r_max_s:<10}{width_s:<9}{r['lapvar']:<8.0f}{'，'.join(notes)}")

    print()
    print(f"检出: {n_ok}/{len(files)}")
    if r_max_all:
        print(f"覆盖 r_max: {min(r_max_all):.2f} ~ {max(r_max_all):.2f}"
              f"（画面四角=1.0；达到 0.85 以上: {n_edge} 张）")
        print(f"距离多样性（板宽占画面）: {min(width_all) * 100:.0f}% ~ {max(width_all) * 100:.0f}%")
    if p99_all:
        print(f"亮度 p99 范围: {min(p99_all):.0f} ~ {max(p99_all):.0f}")

    problems = []
    if n_ok < 15:
        problems.append(f"可检出视图不足（{n_ok} 张；建议 18~25 张，至少 15 张）")
    if len(files) - n_ok > 2:
        problems.append(f"{len(files) - n_ok} 张未检出（多为对比度严重不足/板不完整/剧烈模糊）")

    suggestions = []
    if n_edge < 5:
        suggestions.append(f"四角覆盖不足（{n_edge} 张 >=0.85）：高阶畸变 k2/k3 不可估计，"
                           f"标定已用 k1 受限模型；如需更高精度可补拍四角照片")
    if n_ok and sum(1 for p in p99_all if p < 150) > 0:
        suggestions.append("部分图像偏暗：不影响标定精度（实测调亮前后角点仅差 0.005px），仅观感提示")
    if any(r.get("sat", 0) > 0.5 for r in [analyze(f) or {} for f in files] if r):
        suggestions.append("个别图像有过曝风险：请核对对应图像白块是否死白")

    print()
    if problems:
        print("结论: 未通过，建议修复后再标定：")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("结论: 通过，可用于标定。（提示项不阻塞）")
    for s in suggestions:
        print(f"  - 提示: {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
