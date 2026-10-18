cd /home/oliver/coding/learn/RMVtrainingHomework/RMVtraining
run() { timeout 250 ./build/test_camera_playback "$@" 2>&1 | grep -E "预测命中评估|瞄准板残差" | grep -oE "中位 [0-9.e-]+" | tr '\n' ' '; echo; }
for v in 0.05 0.5; do
  echo "--- v-sigma=$v"
  printf "v4静态      "; run tmp/v4.avi --red --ekf --headless --v-sigma $v
  printf "v4-id3      "; run tmp/v4.avi --red --only-id 3 --ekf --headless --v-sigma $v
  printf "v1          "; run tmp/v1.avi --red --ekf --headless --v-sigma $v
done
