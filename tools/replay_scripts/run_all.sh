cd /home/oliver/coding/learn/RMVtrainingHomework/RMVtraining
for args in "tmp/v4.avi --red" "tmp/v4.avi --red --only-id 3" "tmp/v1.avi --red"; do
  printf "%-30s " "$args"
  timeout 250 ./build/test_camera_playback $args --ekf --headless 2>&1 | grep -E "预测命中评估|瞄准板残差|稳定" | head -3 | tr '\n' ' '
  echo
done
