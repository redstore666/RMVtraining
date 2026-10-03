"""HSV 采样器：为灯条颜色掩码确定 H/S/V 阈值。

界面模式（桌面环境）:
    python3 tools/hsv_picker.py tests/data/synth_small_1.png
    - 左键点击像素：终端打印该点与 5x5 邻域中位数的 BGR/HSV，并画圈标记
    - 快捷键 h/s/v 查看对应单通道灰度图（灯条在哪个 H 段一目了然），c 回原图，ESC 退出
    - 无桌面环境时用下面的脚本模式

脚本模式（无界面，可批量）:
    python3 tools/hsv_picker.py <图片> --point X1 Y1 --point X2 Y2
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def resolve_image(raw: str) -> Path:
    """优先按当前目录解析图片路径，其次按仓库根解析，方便从任意目录启动。"""
    p = Path(raw)
    if p.exists():
        return p
    alt = Path(__file__).resolve().parent.parent / raw
    if alt.exists():
        return alt
    print(f"读图失败: {raw}\n  当前目录: {Path.cwd()}\n  也尝试过: {alt}",
          file=sys.stderr)
    sys.exit(2)


def sample(bgr: np.ndarray, hsv: np.ndarray, x: int, y: int, k: int = 2) -> None:
    h, w = bgr.shape[:2]
    x0, x1 = max(0, x - k), min(w, x + k + 1)
    y0, y1 = max(0, y - k), min(h, y + k + 1)
    med_bgr = np.median(bgr[y0:y1, x0:x1].reshape(-1, 3), axis=0).astype(int)
    med_hsv = np.median(hsv[y0:y1, x0:x1].reshape(-1, 3), axis=0).astype(int)
    print(f"({x:4d},{y:4d})  点值 BGR={tuple(int(v) for v in bgr[y, x])} "
          f"HSV={tuple(int(v) for v in hsv[y, x])}  |  "
          f"5x5中位数 BGR={tuple(int(v) for v in med_bgr)} "
          f"HSV={tuple(int(v) for v in med_hsv)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="HSV 采样器：确定灯条掩码阈值用")
    parser.add_argument("image")
    parser.add_argument("--point", nargs=2, type=int, action="append",
                        metavar=("X", "Y"), help="无界面模式：打印指定坐标的采样值，可多次")
    args = parser.parse_args()

    image_path = resolve_image(args.image)
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        print(f"读图失败: {image_path}", file=sys.stderr)
        return 2
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    if args.point:
        for x, y in args.point:
            sample(bgr, hsv, x, y)
        return 0

    window = "hsv_picker"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 720)
    state = {"channel": None}

    def render():
        if state["channel"] is None:
            cv2.imshow(window, bgr)
        else:
            cv2.imshow(window, hsv[:, :, state["channel"]])

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            sample(bgr, hsv, x, y)
            cv2.circle(bgr, (x, y), 5, (0, 255, 255), 1)
            render()

    cv2.setMouseCallback(window, on_mouse)
    render()
    while True:
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC 退出
            break
        if key in (ord('h'), ord('s'), ord('v')):
            state["channel"] = {ord('h'): 0, ord('s'): 1, ord('v'): 2}[key]
            render()
        elif key == ord('c'):
            state["channel"] = None
            render()
        try:
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break  # 窗口被手动关闭
        except cv2.error:
            break  # Qt 后端在窗口销毁后查询属性会直接抛异常
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
