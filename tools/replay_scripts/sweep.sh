cd /home/oliver/coding/learn/RMVtrainingHomework/RMVtraining
for R in 0.18 0.20 0.22 0.24; do
  printf "R=%.2f  " $R
  timeout 250 ./build/test_camera_playback tmp/v4.avi --red --only-id 3 --ekf --headless --radius $R 2>&1 | grep -E "预测命中评估|瞄准板残差" | grep -oE "中位 [0-9.e-]+" | tr '\n' ' '
  echo
done
