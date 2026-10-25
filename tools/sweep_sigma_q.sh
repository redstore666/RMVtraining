#!/bin/sh
# σ_yaw × SPIN_Q 二维扫（v4，身份后验开启）。结果逐行打到 stdout。
cd "$(dirname "$0")/.." || exit 1
printf "%-8s %-7s %-30s %s\n" "SPIN_Q" "sigma" "aim_p50/p90/<=5cm" "omega_p10/p50/p90_jump90"
for combo in "100 1.0" "100 0.6" "100 0.4" "1000 0.6" "40000 0.4" "40000 0.3"; do
    q=$(echo "$combo" | cut -d' ' -f1)
    s=$(echo "$combo" | cut -d' ' -f2)
    SPIN_Q=$q RMV_ANGLE_SIGMA=$s ./build/test_camera_playback tmp/v4.avi --red --only-id 3 \
        --ekf --headless --id-posterior 1 --dump tmp/s2/sweep.csv > tmp/s2/sweep.log 2>&1
    aim=$(grep -E "^  瞄准板残差:" tmp/s2/sweep.log | sed 's/.*中位 \([0-9.]*\) m，p90 \([0-9.]*\).*≤5cm \([0-9.]*\)%.*/\1\/\2\/\3/')
    om=$(/home/oliver/.workbuddy/binaries/python/versions/3.13.12/bin/python3 - <<'PY'
hdr = None
v = []
for ln in open('tmp/s2/sweep.csv'):
    p = ln.rstrip().split(',')
    if hdr is None:
        hdr = p
        continue
    if len(p) != len(hdr):
        continue
    d = dict(zip(hdr, p))
    try:
        v.append(float(d['ekfw']))
    except ValueError:
        pass
a = sorted(abs(x) for x in v)
j = sorted(abs(b - c) for c, b in zip(v, v[1:]))
print('%.2f/%.2f/%.2f/%.2f' % (a[len(a)//10], a[len(a)//2], a[len(a)*9//10], j[len(j)*9//10]))
PY
)
    printf "%-8s %-7s %-30s %s\n" "$q" "$s" "$aim" "$om"
done
