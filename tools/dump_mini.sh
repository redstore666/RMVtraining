#!/usr/bin/env bash
# 抓迷你俯视图的 PNG dump（每 N 帧一张），放到 out/mini_<name>/。
#
# 用法： tools/dump_mini.sh <name> [秒数=12] [间隔帧数=300] [-- 额外参数...]
# 例：   tools/dump_mini.sh left 12 300 -- --id 3
#        tools/dump_mini.sh cam  10 300 -- --id 3 --show-camera
#
# 坑：test_camera_system 写到 tmp/system_mini_%06d.png，**不会清旧文件**。
# 上一次跑留下的同号文件会混进来，曾经让我把 v4 的帧当成 v1 的帧看。
# 所以这里先删干净再把本次的挪走。
set -u
cd "$(dirname "$0")/../../.." || exit 1
NAME=${1:?用法: dump_mini.sh <name> [秒数] [间隔] [额外参数...]}
shift
DURATION=${1:-12}; shift || true
EVERY=${1:-300};   shift || true
[ "${1:-}" = "--" ] && shift

OUT=tmp/handoff/out/mini_$NAME
PORT=$((8100 + RANDOM % 400))
rm -f tmp/system_mini_*.png
rm -rf "$OUT"
mkdir -p "$OUT"

./build/test_camera_system --red --video tmp/v4.avi --video-loop --id 3 \
  --dump-every "$EVERY" "$@" --web-port "$PORT" > "$OUT/run.log" 2>&1 &
PID=$!
sleep "$DURATION"
kill "$PID" 2>/dev/null
wait "$PID" 2>/dev/null
mv tmp/system_mini_*.png "$OUT"/ 2>/dev/null
echo "-> $OUT ($(ls "$OUT" | grep -c png) 张 png)"
