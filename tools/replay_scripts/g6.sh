cd /home/oliver/coding/learn/RMVtrainingHomework/RMVtraining
for ds in "v4.avi" "v4.avi --only-id 3" "v1.avi"; do
  printf "[%s] " "$ds"
  timeout 250 ./build/test_camera_playback tmp/${ds} --red --ekf --headless 2>&1 | grep -E "预测命中评估|瞄准板残差" | grep -oE "中位 [0-9.e-]+" | tr '\n' ' '; echo
done
