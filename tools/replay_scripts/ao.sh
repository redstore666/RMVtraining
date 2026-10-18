cd /home/oliver/coding/learn/RMVtrainingHomework/RMVtraining
m() { timeout 250 ./build/test_camera_playback "$@" 2>&1 | grep -E "预测命中评估|瞄准板残差" | grep -oE "中位 [0-9.]+" | tr '\n' ' '; echo; }
for a in 0 0.05; do
  echo "--- aim-obs=$a"
  printf "v4静态   "; m tmp/v4.avi --red --ekf --headless --aim-obs $a --only-id 1
  printf "v4-id3   "; m tmp/v4.avi --red --only-id 3 --ekf --headless --aim-obs $a
  printf "v1       "; m tmp/v1.avi --red --ekf --headless --aim-obs $a
done
