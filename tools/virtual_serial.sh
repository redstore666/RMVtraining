#!/usr/bin/env bash
# 虚拟串口工具：在 PC 上没有车上电控设备时，用 socat 建一对虚拟串口，
# 并可用内置"模拟电控"按 64 字节协议帧喂给程序，让上车程序在 PC 上完整跑起来。
#
# 用法:
#   tools/virtual_serial.sh start                 # 创建虚拟串口对
#   tools/virtual_serial.sh send [选项]           # 启动模拟电控（喂帧）
#       --status 0|5    自瞄状态字节（决定敌方颜色：0=打蓝，5=打红），默认 0
#       --hz N          发帧频率，默认 100
#       --yaw-sweep     云台 yaw 缓慢摆动（模拟真实转动，默认关闭）
#   tools/virtual_serial.sh stop                  # 停掉模拟电控和虚拟串口
#
# 然后跑程序:
#   ./build/rmv_training /tmp/ttyVMS 115200 config/camera.yaml 8080
#
# 程序应打印 "收到首帧，status = ..."。

set -u

MS_LINK="/tmp/ttyVMS"      # 程序端打开这个
CTRL_LINK="/tmp/ttyVCtrl"  # 模拟电控写这个
SOCAT_PID="/tmp/ttyVMS.socat.pid"
SEND_PID="/tmp/ttyVMS.sender.pid"

do_start() {
    if [ ! -e /dev/ptmx ]; then
        echo "错误: 系统没有 /dev/ptmx，无法创建虚拟串口（容器/精简内核可能如此）"
        exit 1
    fi
    if ! command -v socat > /dev/null; then
        echo "错误: 未安装 socat，请先: sudo apt install socat"
        exit 1
    fi
    if [ -e "$MS_LINK" ] || [ -e "$CTRL_LINK" ]; then
        echo "已存在 $MS_LINK / $CTRL_LINK，先执行 stop 再 start"
        exit 1
    fi
    socat -d -d \
        pty,raw,echo=0,link="$MS_LINK" \
        pty,raw,echo=0,link="$CTRL_LINK" &
    echo $! > "$SOCAT_PID"
    sleep 0.3
    if [ -e "$MS_LINK" ] && [ -e "$CTRL_LINK" ]; then
        echo "虚拟串口已创建:"
        echo "  程序端:     $MS_LINK"
        echo "  模拟电控端: $CTRL_LINK"
        echo "下一步: tools/virtual_serial.sh send"
    else
        echo "创建失败，查看 socat 输出"
        exit 1
    fi
}

do_send() {
    if [ ! -e "$CTRL_LINK" ]; then
        echo "虚拟串口不存在，先执行 start"
        exit 1
    fi
    python3 - "$CTRL_LINK" "$@" << 'PYEOF'
import math
import os
import struct
import sys
import time

link = sys.argv[1]
args = sys.argv[2:]

status = 0
hz = 100
yaw_sweep = False
i = 0
while i < len(args):
    if args[i] == "--status":
        status = int(args[i + 1]); i += 2
    elif args[i] == "--hz":
        hz = int(args[i + 1]); i += 2
    elif args[i] == "--yaw-sweep":
        yaw_sweep = True; i += 1
    else:
        i += 1

# MessData_AutoAim，1 字节对齐，共 64 字节：
# head, yaw, pitch, roll, status, is_far, armor_flag, latency, bias, distance,
# pitch_offset, coo_x, coo_y, wheel_w, fire_allowance, target_yaw, target_pitch,
# yaw_vel, pitch_vel, crc(=EKF状态), tail
fmt = "<BfffBBBffff fff Bffff H B".replace(" ", "")
assert struct.calcsize(fmt) == 64, struct.calcsize(fmt)

fd = os.open(link, os.O_WRONLY | os.O_NOCTTY)
period = 1.0 / hz
t0 = time.monotonic()
n = 0
print(f"模拟电控开始发帧 -> {link}  status={status}  {hz}Hz  yaw_sweep={yaw_sweep}")
print("Ctrl+C 停止")
try:
    while True:
        t = time.monotonic() - t0
        yaw = 0.5 * math.sin(0.6 * t) if yaw_sweep else 0.0
        pitch = 0.0
        roll = 0.0
        frame = struct.pack(
            fmt,
            0x71,          # head
            yaw, pitch, roll,
            status,        # 0=自瞄+己方蓝(打红方? 见 enemyColorFromStatus), 5=另一色
            0,             # is_far
            0,             # armor_flag
            0.0,           # latency（视觉回写方向，占位）
            0.0,           # bias
            0.0,           # distance
            0.0,           # pitch_offset
            0.0, 0.0,      # coo_x, coo_y
            0.0,           # wheel_w
            0,             # fire_allowance
            0.0, 0.0, 0.0, 0.0,  # target_yaw/pitch/yaw_vel/pitch_vel
            0,             # crc（EKF 状态）
            0x4C,          # tail
        )
        os.write(fd, frame)
        n += 1
        time.sleep(period)
except KeyboardInterrupt:
    print(f"已停止，共发送 {n} 帧")
finally:
    os.close(fd)
PYEOF
}

do_stop() {
    for pidfile in "$SEND_PID" "$SOCAT_PID"; do
        if [ -f "$pidfile" ]; then
            kill "$(cat "$pidfile")" 2>/dev/null
            rm -f "$pidfile"
        fi
    done
    pkill -f "virtual_serial_sender" 2>/dev/null
    sleep 0.2
    rm -f "$MS_LINK" "$CTRL_LINK"
    echo "已停止并清理 $MS_LINK / $CTRL_LINK"
}

case "${1:-}" in
    start) do_start ;;
    send) shift; do_send "$@" ;;
    stop) do_stop ;;
    *)
        sed -n '2,20p' "$0"
        exit 1
        ;;
esac
