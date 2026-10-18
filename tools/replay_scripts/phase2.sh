cd /home/oliver/coding/learn/RMVtrainingHomework/RMVtraining
m() { timeout 250 ./build/test_camera_playback "$@" 2>&1 | grep -E "瞄准板残差" | grep -oE "中位 [0-9.]+" ; }
for s in off 0.05 0.10 0.20; do
  printf "phase=%-5s " $s
  if [ "$s" = off ]; then m tmp/v4.avi --red --only-id 3 --ekf --headless;
  else m tmp/v4.avi --red --only-id 3 --ekf --headless --phase-yaw --phase-sigma $s; fi
done
