cd /home/oliver/coding/learn/RMVtrainingHomework/RMVtraining
m() { timeout 250 ./build/test_camera_playback "$@" 2>&1 | grep -E "预测命中评估|瞄准板残差" | grep -oE "中位 [0-9.]+" | tr '\n' ' '; echo; }
for s in off 0.05 0.10 0.20 0.40; do
  printf "phase=%-5s " $s
  if [ "$s" = off ]; then m tmp/v4.avi --red --only-id 3 --ekf --headless;
  else m tmp/v4.avi --red --only-id 3 --ekf --headless --phase-yaw --phase-sigma $s; fi
done
