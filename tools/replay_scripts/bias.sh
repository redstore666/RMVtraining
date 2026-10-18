cd /home/oliver/coding/learn/RMVtrainingHomework/RMVtraining
run() { timeout 250 ./build/test_camera_playback "$@" 2>/dev/null | grep -E "瞄准板残差: |本帧有观测|本帧无观测" | tr '\n' '|' | sed 's/  */ /g'; echo; }
for b in 0.0 0.03; do
  echo "=== bias=$b"
  printf "v4静态 "; run tmp/v4.avi --red --ekf --headless --bias $b
  printf "v4-id3 "; run tmp/v4.avi --red --only-id 3 --ekf --headless --bias $b
  printf "v1     "; run tmp/v1.avi --red --ekf --headless --bias $b
done
