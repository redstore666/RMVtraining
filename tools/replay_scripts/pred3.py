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
