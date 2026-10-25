#!/usr/bin/env python3
"""从 test_camera_playback --dump 的 CSV 统计"瞄准点 vs 未来实测板心"的误差。

用法：python3 tools/pred_error.py build/d.csv [帧数偏移，默认 5 = 30 ms @166fps]
逐条观测有一行，这里按帧聚合成"每帧最近观测"，避免同帧双板把中位数抬高。
"""
import csv,math,sys
rows=[r for r in csv.DictReader(open(sys.argv[1]))]
obs={}; aim={}
for r in rows:
    f=int(r['frame'])
    if r['obs_id']==r['target_id']:
        obs.setdefault(f,[]).append((float(r['x']),float(r['y']),float(r['z'])))
    a=(float(r['aimx']),float(r['aimy']),float(r['aimz']))
    if abs(a[0])+abs(a[1])+abs(a[2])>0: aim[f]=a
def med(xs):
    xs=sorted(xs); return xs[len(xs)//2] if xs else float('nan')
same=[min(math.dist(a,o) for o in obs[f]) for f,a in aim.items() if f in obs]
fut=[min(math.dist(a,o) for o in obs[f+5]) for f,a in aim.items() if f+5 in obs]
print(f'|aim(t)-obs(t)| 中位={med(same):.3f}   |aim(t)-obs(t+30ms)| 中位={med(fut):.3f}   n={len(fut)}')
