cd /home/oliver/coding/learn/RMVtrainingHomework/RMVtraining
for s in 0.05 0.2 0.5 2.0; do
  printf "v-sigma=%-5s " $s
  timeout 250 ./build/test_camera_playback tmp/v1.avi --red --ekf --headless --v-sigma $s 2>&1 | grep -E "预测命中评估|瞄准板残差" | grep -oE "中位 [0-9.e-]+" | tr '\n' ' '
  echo
done
