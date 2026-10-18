cd /home/oliver/coding/learn/RMVtrainingHomework/RMVtraining
run() { timeout 250 ./build/test_camera_playback "$@" 2>&1 | grep -E "瞄准板残差|帧数 |样本 |Stable 帧|评估帧"; }
echo "=== v4 静止 ==="; run tmp/v4.avi --red --ekf --headless
echo "=== v4 --only-id 3 ==="; run tmp/v4.avi --red --only-id 3 --ekf --headless
echo "=== v1 ==="; run tmp/v1.avi --red --ekf --headless
