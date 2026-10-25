cd /home/oliver/coding/learn/RMVtrainingHomework/RMVtraining
m() { timeout 250 ./build/test_camera_playback "$@" 2>&1 | grep -E "预测命中评估|瞄准板残差|Stable" | tr '\n' ' ' | sed 's/  */ /g'; echo; }
printf "v4静态   "; m tmp/v4.avi --red --ekf --headless
printf "v4-id3   "; m tmp/v4.avi --red --only-id 3 --ekf --headless
printf "v1       "; m tmp/v1.avi --red --ekf --headless
