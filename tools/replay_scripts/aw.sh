cd /home/oliver/coding/learn/RMVtrainingHomework/RMVtraining
for a in 0.02 0.05 0.10; do
  printf "hold=%s id3 " $a
  timeout 250 ./build/test_camera_playback tmp/v4.avi --red --only-id 3 --ekf --headless --aim-obs $a 2>&1 | grep "瞄准板残差" | grep -oE "中位 [0-9.]+ m，p90 [0-9.]+|≤2cm [0-9.]+%" | tr '\n' ' '; echo
done
