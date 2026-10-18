import csv,math,statistics,sys
rows=[r for r in csv.DictReader(open(sys.argv[1]))]
obs={}; aim={}
for r in rows:
    f=int(r['frame'])
    o=(float(r['x']),float(r['y']),float(r['z']))
    if r['obs_id']==r['target_id']:
        if f not in obs or float(r['best_dist'])<obs[f][1]:
            obs[f]=(o,float(r['best_dist']))
    a=(float(r['aimx']),float(r['aimy']),float(r['aimz']))
    if abs(a[0])+abs(a[1])+abs(a[2])>0: aim[f]=a
errs=[]
for f,a in aim.items():
    # 30 ms 后 = 5 帧
    for k in (5,):
        g=f+k
        if g in obs: errs.append(math.dist(a,obs[g][0]))
errs.sort()
n=len(errs)
print(f'n={n} 中位={errs[n//2]:.3f} p90={errs[int(0.9*n)]:.3f} ≤5cm={100*sum(1 for e in errs if e<=0.05)/n:.1f}%')
