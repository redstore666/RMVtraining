cd /home/oliver/coding/learn/RMVtrainingHomework/RMVtraining
m() { timeout 250 ./build/test_camera_playback "$@" 2>&1 | grep -E "瞄准板残差" | grep -oE "中位 [0-9.]+" ; }
for s in 0.05 0.10 0.15 0.25; do
  printf "phase=%-5s noanchor " $s; m tmp/v4.avi --red --only-id 3 --ekf --headless --phase-yaw --phase-sigma $s --no-phase-anchor
done
