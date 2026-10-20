#!/usr/bin/env python3
"""用回放 dump 的 CSV 评估"模型四块板 vs 实测板"的贴合度。

dump 列（tests/test_camera_playback.cpp 的 --dump）：
  frame,t,det,target_id,aim_slot,obs_id,x,y,z(世界系观测板心),
  best_slot,best_dist,second_dist,cx,cy,cz,cyaw,R(车心/朝向/半径),
  oyaw(观测法向),aimx,aimy,aimz(发布出去的瞄准点),u,v,stab,upd,missed

指标：一帧里第一块观测通常刚被 update 吃掉（误差≈0，会掩盖问题），所以对
"同帧第二块观测"算它到最近模型板的距离——半径/相位错了这里立刻能看出来。

用法: python3 tools/eval_model_fit.py build/v4.csv
"""
import csv
import math
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({k: float(v) for k, v in r.items() if k != "stab"})
            except ValueError:
                continue
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "build/v4.csv"
    rows = load(path)
    frames = defaultdict(list)
    for r in rows:
        frames[int(r["frame"])].append(r)
    errs = []
    radii = defaultdict(list)
    for _, obs in frames.items():
        for o in obs:
            radii[int(o["target_id"])].append(o["R"])
        if len(obs) < 2:
            continue
        a = obs[0]
        cx, cy, yaw, radius = a["cx"], a["cy"], a["cyaw"], a["R"]
        for o in obs[1:]:
            best = min(
                math.hypot(o["x"] - (cx - radius * math.cos(yaw + k * math.pi / 2)),
                           o["y"] - (cy - radius * math.sin(yaw + k * math.pi / 2)))
                for k in range(4))
            errs.append(best)
    if not errs:
        print("没有同帧多观测的样本")
        return
    errs.sort()
    n = len(errs)
    print(f"{path}: 同帧多余观测 n={n}")
    print(f"  到最近模型板: 中位 {errs[n//2]:.3f} m  p75 {errs[int(.75*n)]:.3f}  "
          f"p90 {errs[int(.9*n)]:.3f}  <=5cm {100*sum(e<=.05 for e in errs)/n:.0f}%  "
          f"<=10cm {100*sum(e<=.10 for e in errs)/n:.0f}%")
    for tid, rs in sorted(radii.items()):
        rs = sorted(rs)
        print(f"  target {tid}: 半径中位 {rs[len(rs)//2]:.3f} m ({min(rs):.3f}..{max(rs):.3f})")


if __name__ == "__main__":
    main()
