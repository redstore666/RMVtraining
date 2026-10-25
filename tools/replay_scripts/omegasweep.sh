cd /home/oliver/coding/learn/RMVtrainingHomework/RMVtraining
for w in 0 10 20 30 40 50; do
  printf "omega=%-3s " $w
  timeout 250 ./build/test_camera_playback tmp/v4.avi --red --only-id 3 --ekf --headless --bias 0.03 --omega $w 2>/dev/null | grep -E "瞄准板残差: " | head -1
done
