import csv,math,sys
rows=[r for r in csv.DictReader(open(sys.argv[1]))]
obs={}; aim={}
for r in rows:
    f=int(r['frame'])
    if r['obs_id']==r['target_id']:
        obs.setdefault(f,[]).append((float(r['x']),float(r['y']),float(r['z'])))
    a=(float(r['aimx']),float(r['aimy']),float(r['aimz']))
    if abs(a[0])+abs(a[1])+abs(a[2])>0: aim[f]=a
errs=[]
for f,a in aim.items():
    g=f+5
    if g in obs: errs.append(min(math.dist(a,o) for o in obs[g]))
errs.sort(); n=len(errs)
print(f'n={n} 中位={errs[n//2]:.3f} p90={errs[int(0.9*n)]:.3f} ≤5cm={100*sum(1 for e in errs if e<=0.05)/n:.1f}%')
