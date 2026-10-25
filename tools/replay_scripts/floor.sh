cd /home/oliver/coding/learn/RMVtrainingHomework/RMVtraining
m() { timeout 250 ./build/test_camera_playback "$@" 2>&1 | grep -E "瞄准板残差" | grep -oE "中位 [0-9.]+" ; }
for f in 0.02 0.04 0.06 0.10; do
  printf "floor=%-5s id3 " $f; m tmp/v4.avi --red --only-id 3 --ekf --headless --sigma-floor $f
done
