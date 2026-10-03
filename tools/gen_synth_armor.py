"""生成合成装甲板测试图（tests/data/*.png），用于在没有相机/实车时调试 armor_detect。

原理：按 装甲板尺寸.txt 的比例（小 135x57mm，大 230x57mm）画两根发光灯条，
并把数据集里的数字贴纸裁片贴到中间。场景加梯度背景和噪声，模拟真实画面。

用法:  python3 tools/gen_synth_armor.py
说明:  数字裁片取自 RMVtrainingHomework/train_cls/dataset/val/（只读）；
       生成物写入 RMVtraining/tests/data/，随作业离线提交。
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DIGIT_SRC = Path("/home/oliver/coding/learn/RMVtrainingHomework/train_cls/dataset/val")
OUT_DIR = REPO / "tests" / "data"

W, H = 1440, 1080
RED = (0, 0, 255)   # BGR
BLUE = (255, 0, 0)

rng = np.random.default_rng(7)


def make_background() -> np.ndarray:
    """深色渐变背景 + 高斯噪声。"""
    base = np.linspace(25, 45, W, dtype=np.float32)[None, :, None]
    img = np.repeat(base, H, axis=0)
    img = np.repeat(img, 3, axis=2)
    img += rng.normal(0.0, 4.0, img.shape).astype(np.float32)
    return img


def draw_bar(img: np.ndarray, cx: int, cy: int, w: int, h: int, color) -> None:
    """画一根灯条：亮芯 + 外发光。"""
    x0, y0 = cx - w // 2, cy - h // 2
    core = np.zeros_like(img)
    core[y0:y0 + h, x0:x0 + w] = color
    halo = cv2.GaussianBlur(core, (0, 0), sigmaX=6, sigmaY=6)
    img += halo * 0.7
    img += core


def paste_digit(img: np.ndarray, center_x: int, center_y: int, target_h: int, cls: str) -> None:
    """把数据集中的数字裁片按高度 target_h 贴到指定中心。"""
    files = sorted((DIGIT_SRC / cls).glob("*.png"))
    crop = cv2.imread(str(files[len(files) // 2]), cv2.IMREAD_GRAYSCALE)
    ch, cw = crop.shape
    scale = target_h / ch
    resized = cv2.resize(crop, (max(1, int(cw * scale)), target_h), interpolation=cv2.INTER_LINEAR)
    h, w = resized.shape
    x0, y0 = center_x - w // 2, center_y - h // 2
    bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR).astype(np.float32)
    img[y0:y0 + h, x0:x0 + w] += bgr


def draw_armor(img: np.ndarray, center_x: int, center_y: int, outer_w: int, plate_h: int,
               bar_w: int, color, digit_cls: str | None) -> None:
    """画一块装甲板：两灯条 + 中间数字。outer_w 为两灯条外缘间距（≈ 装甲板宽度）。

    比例来自 装甲板尺寸.txt：小 135x57（W/H=2.37），大 230x57（W/H=4.04）。
    """
    half = (outer_w - bar_w) // 2
    draw_bar(img, center_x - half, center_y, bar_w, plate_h, color)
    draw_bar(img, center_x + half, center_y, bar_w, plate_h, color)
    if digit_cls is not None:
        inner_w = outer_w - 2 * bar_w
        paste_digit(img, center_x, center_y, int(plate_h * 0.68), digit_cls)
    return


def finish(img: np.ndarray, path: Path) -> None:
    out = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(str(path), out)
    print("saved", path.relative_to(REPO))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def plate_h_for(outer_w: int) -> int:
        # 采样：小装甲 300px 宽 → 高 300/2.37≈127；大装甲 560px 宽 → 高 560/4.04≈139（取整后同高）
        return int(round(outer_w / (2.37 if outer_w < 400 else 4.04)))

    # 1. 小装甲，数字 1，攻击红方
    img = make_background()
    w = 300
    draw_armor(img, 720, 540, w, plate_h_for(w), 16, RED, "1")
    finish(img, OUT_DIR / "synth_small_1.png")

    # 2. 大装甲，数字 5
    img = make_background()
    w = 560
    draw_armor(img, 720, 540, w, plate_h_for(w), 16, RED, "5")
    finish(img, OUT_DIR / "synth_large_5.png")

    # 3. 双目标：小装甲（3）+ 大装甲（7），蓝色灯条
    img = make_background()
    w1, w2 = 260, 480
    draw_armor(img, 380, 500, w1, plate_h_for(w1), 16, BLUE, "3")
    draw_armor(img, 1020, 560, w2, plate_h_for(w2), 16, BLUE, "7guard")
    finish(img, OUT_DIR / "synth_dual_3_7_blue.png")

    # 4. 倾斜 12 度（整个画面旋转，模拟相机滚转）
    img = make_background()
    w = 300
    draw_armor(img, 720, 540, w, plate_h_for(w), 16, RED, "2")
    upright = np.clip(img, 0, 255).astype(np.uint8)
    m = cv2.getRotationMatrix2D((W / 2, H / 2), 12.0, 1.0)
    rotated = cv2.warpAffine(upright, m, (W, H), borderValue=(35, 35, 35))
    img = rotated.astype(np.float32) + rng.normal(0.0, 3.0, rotated.shape)
    finish(img, OUT_DIR / "synth_tilted_2.png")

    # 5. 假数字：配对成立，但中间贴的是 9neg 噪声 → 应被分类器拒绝
    img = make_background()
    w = 300
    draw_armor(img, 720, 540, w, plate_h_for(w), 16, RED, "9neg")
    finish(img, OUT_DIR / "synth_fake_number.png")

    # 6. 空场景：无灯条（只有几个杂乱亮点）→ 应无检测
    img = make_background()
    for _ in range(12):
        x, y = rng.integers(60, W - 60), rng.integers(60, H - 60)
        s = rng.integers(3, 10)
        img[y - s:y + s, x - s:x + s] += 180
    finish(img, OUT_DIR / "synth_negative.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())
