#!/usr/bin/env bash
# 跑一次 test_camera_system，抓 SSE 到 out/<name>.sse，日志到 out/<name>.log。
#
# 【为什么必须这样写】沙箱里每次 bash 调用是独立的 PID 命名空间：
#   * 上一次调用里 `&` 起的进程会跨调用存活，但 ps 看不见、kill 不掉，
#     于是 8081 端口被旧二进制占着，curl 抓到的是**旧数据**（曾经因此误判
#     "新加的日志键没生效"）。
#   * `pkill -f "test_camera_system"` 会把当前这条 bash 命令自己也杀掉（exit 143）。
# 所以：随机端口 + 本进程 $! + 同一次调用内起停。
#
# 用法：
#   tools/run_tool.sh <name> [秒数默认 25] [-- 额外参数...]
# 例：
#   tools/run_tool.sh base
#   tools/run_tool.sh pair --pair-body --pair-phase-err 0.95
#   tools/run_tool.sh pll --rate-pll
set -u
cd "$(dirname "$0")/../../.." || exit 1   # -> 仓库根目录
NAME=${1:?用法: run_tool.sh <name> [秒数] [额外参数...]}
shift
DURATION=${1:-25}
shift || true
[ "${1:-}" = "--" ] && shift

PORT=$((8100 + RANDOM % 400))
OUT=tmp/handoff/out
mkdir -p "$OUT"

./build/test_camera_system --red --video tmp/v4.avi --video-loop --id 3 \
  "$@" --web-port "$PORT" > "$OUT/$NAME.log" 2>&1 &
PID=$!
# 等端口起来（最多 3 s）
for _ in $(seq 1 30); do
  curl -sf "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
  sleep 0.1
done
curl -sN --max-time "$DURATION" "http://127.0.0.1:$PORT/data" > "$OUT/$NAME.sse"
kill "$PID" 2>/dev/null
wait "$PID" 2>/dev/null
echo "-> $OUT/$NAME.sse ($(grep -c '^data:' "$OUT/$NAME.sse" 2>/dev/null) 批)  log: $OUT/$NAME.log"
