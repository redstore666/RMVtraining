#!/usr/bin/env bash
# 旋转目标 EKF 的三组回归（真数据回放），输出可直接对比的数字表。
#
#   ./tools/run_ekf_regression.sh              # 用默认配置
#   ./tools/run_ekf_regression.sh --radius 0.25  # 额外参数透传给 playback
#
# 数据约定：tmp/v4.avi（静止 + 旋转两种读法）、tmp/v1.avi（遮挡/双目标）。
# 缺数据时跳过对应组，不算失败。
#
# 判读口径：
#   静止  v4        —— 中位应 < 10 mm，p90 < 20 mm
#   旋转  v4 id3    —— 中位 ~90 mm，p90 ~150 mm（观测单帧抖动 50 mm 是下限）
#   遮挡  v1        —— 中位 ~300 mm（重锁段的刚性外推误差）
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXTRA_ARGS=("$@")
BIN=build/test_camera_playback

if [[ ! -x $BIN ]]; then
    echo "未找到 $BIN，先构建…"
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null || exit 1
    cmake --build build -j >/dev/null || exit 1
fi

run() {
    local name=$1 data=$2
    shift 2
    if [[ ! -f $data ]]; then
        echo "== $name: 缺少 $data，跳过"
        return
    fi
    echo "== $name ($data ${*:-默认})"
    # shellcheck disable=SC2068
    timeout 300 "$BIN" "$data" --red --ekf --headless "$@" "${EXTRA_ARGS[@]}" 2>&1 |
        grep -E "首次 Tracking 帧|Stable 帧数|预测命中评估" || echo "   （没有输出，检查数据/模型路径）"
}

run "静止 v4" tmp/v4.avi
run "旋转 v4 (id3)" tmp/v4.avi --only-id 3
run "遮挡 v1" tmp/v1.avi
